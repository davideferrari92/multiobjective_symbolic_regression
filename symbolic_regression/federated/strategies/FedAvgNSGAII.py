import copy
import logging
import numpy as np

import pandas as pd
from symbolic_regression.Program import Program
from symbolic_regression.callbacks.CallbackSave import MOSRCallbackSaveCheckpoint

from symbolic_regression.federated.strategies.FedNSGAII import FedNSGAII
from symbolic_regression.Population import Population
from symbolic_regression.multiobjective.fitness.Base import BaseFitness


class FedAvgNSGAII(FedNSGAII):
    """ 
    This class implements the Symbolic Merger strategy. It is a federated strategy
    that aggregates the best programs of clients and merges them into a single
    population. The best programs are selected based on their fitness values.

    Args:
        - name: str
            The name of the strategy
        - mode: str
            The mode of the strategy. It can be either 'server', 'client', or 'orchestrator'
        - configuration: Dict
            The configuration of the strategy

    Attributes:
        - name: str
            The name of the strategy
        - mode: str
            The mode of the strategy. It can be either 'server', 'client', or 'orchestrator'
        - configuration: Dict
            The configuration of the strategy
        - federated_rounds_executed: int
            The number of federated rounds executed
        - regressor: SymbolicRegressor
            The regressor of the strategy after the aggregation
        - regressors: Dict[str, SymbolicRegressor]
            The regressors of clients who submitted for the aggregation strategy
        - symbolic_regressor_configuration: Dict
            The configuration of the symbolic regressor

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(
            name=kwargs['name'], mode=kwargs['mode'], configuration=kwargs['configuration'])

        self.stage: int = 1
        self.federated_round_is_terminated: bool = False

    def aggregation(self, data: pd.DataFrame = None, val_data: pd.DataFrame = None, **kwargs):
        '''
        This method implements the aggregation function of the strategy for both
        server and client. Depending on the mode, it executes different branches 
        of the algorithm as stated below.

        When the mode is 'server':
            - Aggregate the regressors from the clients

        When the mode is 'client':
            - Train the local regressor for a number of generations passed as generations_to_train

        In this algorithm there are two phases for both the cliend and the server:
            - Phase 1 Client: train a MOSR population

            - Phase 1 Server: collect the 1PF and 2PF from all clients' poulations

            - Phase 2 Client: receive the combined PFs from the server and optimize the constants on the local dataset

            - Phase 2 Server: calculate the constant-wise average of each program and generate a new population in 
                which each program has the average of the constants of the programs in the same position of the
                previous populations

        '''

        if self.mode == 'server':

            if self.stage == 1:

                logging.debug(f'Server {self.name} is executing stage 1')

                for client_name, client_regressor in self.regressors.items():
                    logging.info(
                        f'Incorporating {client_name} regressor population')

                    if self.configuration['federated'].get('hypervolume_selection', False):
                        num_selected_programs = int(
                            self.symbolic_regressor_configuration['population_size'] / len(self.regressors))

                        client_regressor.population.sort(
                            key=lambda program: program.program_hypervolume, reverse=True)
                        self.regressor.population.extend(
                            client_regressor.population[:num_selected_programs])

                    else:
                        for r in range(1, self.federated_configuration.get('max_rank_aggregation', 2) + 1):
                            self.regressor.population.extend(
                                client_regressor.extract_pareto_front(population=client_regressor.population, rank=r))

                logging.debug(
                    f'Incorporated population size: {len(self.regressor.population)}')

                self.stage = 2
                self.federated_round_is_terminated = False

            elif self.stage == 2:

                logging.debug(f'Server {self.name} is executing stage 2')

                ineligible = []

                for k, ith_programs in enumerate(zip(*[regressor.population for regressor in self.regressors.values()])):
                    """ Apply constants_confidence_intervals_overlap() to each pair of programs in ith_programs.
                    If any one return False, then all of them are set is_valid=False 
                    """
                    is_ineligible = False
                    for i in range(len(ith_programs)):
                        for j in range(i + 1, len(ith_programs)):
                            if not ith_programs[i].is_valid or not ith_programs[j].is_valid or not ith_programs[i].constants_confidence_intervals_overlap(ith_programs[j]):
                                is_ineligible = True

                    if is_ineligible:
                        ineligible.append(k)

                """ Copy the population of the first client in the aggregated regressor. I just need the blueprint of
                the models because I will replace the constants with the average of the constants of the programs in
                ith_programs. We need to copy it here because we want to bring forward the invalidity of the models
                that was calculated just before.
                """

                self.regressor.population: Population = copy.deepcopy(
                    self.regressors[list(self.regressors.keys())[0]].population)

                aggregated_population_size = len(self.regressor.population)

                # Iterate in iverse order the ineligible programs and remove them from the population
                for i in sorted(ineligible, reverse=True):
                    del self.regressor.population[i]
                    for reg in self.regressors.values():
                        del reg.population[i]

                print(
                    f'######### Ineligible: {len([p for p in self.regressor.population if not p.is_valid])}/{aggregated_population_size}')

                """ This next for loop implements a weighted average of the constants of the programs in ith_programs
                and sets the constants of the new program to the average. The weights are the relative size of the
                dataset of the training set of each client.
                """

                regressors_weights = [r.data_shape[0]
                                      for k, r in self.regressors.items()]

                logging.info(f'Regressor weights: {regressors_weights}')
                regressors_weights = np.array(
                    regressors_weights) / np.sum(regressors_weights)

                for index, programs in enumerate(zip(*[regressor.population for regressor in self.regressors.values()])):

                    if len(programs[0].get_constants()) == 0:
                        continue

                    new_constants = []
                    for constant_index in range(0, len(programs[0].get_constants())):
                        ith_constants = np.array(
                            [program.get_constants()[constant_index] for program in programs])

                        # Element-wise multiplication of the constants with the weights of the regressors
                        ith_constant_final = np.sum(
                            ith_constants * regressors_weights)

                        # Sum the weighted constants
                        new_constants.append(ith_constant_final)

                    self.regressor.population[index].set_constants(
                        new_constants)

                self.stage = 1
                self.federated_round_is_terminated = True

                if self.training_configuration.get('verbose', 0) > 0:
                    logging.info(
                        f"Ineligible: {len(ineligible)}/{aggregated_population_size}")

                if self.federated_configuration.get('track_performance'):
                    ################################################################################
                    ############################# PERFORMANCE LOGGING ##############################
                    ineligible_path = f"./{self.name}.ineligible_df.csv"
                    if self.federated_rounds_executed > 0:
                        for cb in self.symbolic_regressor_configuration.get('callbacks', list()):
                            if isinstance(cb, MOSRCallbackSaveCheckpoint):
                                ineligible_path = cb.checkpoint_file + \
                                    f'.{self.name}.ineligible_df.csv'
                        try:
                            ineligible_df = pd.read_csv(ineligible_path)
                        except:
                            ineligible_df = pd.DataFrame()

                        complexities = [
                            p.complexity for p in self.regressor.population]

                        if len(complexities) == 0:
                            complexities = [-1]

                        ineligible_df = pd.concat(
                            [ineligible_df, pd.DataFrame([{'Client': self.name,
                                                           'Federated Round': self.federated_rounds_executed,
                                                           'Ineligible': len(ineligible),
                                                           'Population size': len(self.regressor.population),
                                                           'Ineligible / Population size ratio': len(ineligible) / aggregated_population_size * 100,
                                                           'Complexity Min': np.min(complexities),
                                                           'Complexity 25th percentile': np.percentile(complexities, 25),
                                                           'Complexity Median': np.median(complexities),
                                                           'Complexity 75th percentile': np.percentile(complexities, 75),
                                                           'Complexity Max': np.max(complexities),
                                                           'Complexity Mean': np.mean(complexities),
                                                           'Complexity Std': np.std(complexities),
                                                           }])], ignore_index=True)

                        ineligible_df.to_csv(ineligible_path, index=False)

            else:
                raise ValueError(f'Invalid stage {self.stage}')

        elif self.mode == 'client':

            if self.stage == 1:

                logging.debug(f'Client {self.name} is executing stage 1')

                self.regressor.compute_fitness_population(
                    data=data, validation=False, validation_federated=True, simplify=False)
                self.regressor.compute_fitness_population(
                    data=val_data, validation=True, validation_federated=True, simplify=False)

                if self.federated_configuration.get('track_performance'):
                    ################################################################################
                    ############################# PERFORMANCE LOGGING ##############################
                    performance_path = f"./{self.name}.performance.csv"
                    if self.federated_rounds_executed > 0:

                        for cb in self.symbolic_regressor_configuration.get('callbacks', list()):
                            if isinstance(cb, MOSRCallbackSaveCheckpoint):
                                performance_path = cb.checkpoint_file + \
                                    f'.{self.name}.performance.csv'

                        try:
                            performance = pd.read_csv(performance_path)
                        except:
                            performance = pd.DataFrame()

                        extracted = []
                        to_append = {'client': self.name,
                                     'federated_round': self.federated_rounds_executed}

                        get_best_k = 5

                        for fitness in self.regressor.fitness_functions:
                            fitness: BaseFitness
                            if fitness.smaller_is_better:
                                best_p = sorted([p for p in self.regressor.first_pareto_front],
                                                key=lambda obj: obj.fitness.get(fitness.label, +float('inf')))[:get_best_k]

                                best_p_validation = sorted([p for p in self.regressor.first_pareto_front],
                                                           key=lambda obj: obj.fitness_validation.get(fitness.label, +float('inf')))[:get_best_k]

                            else:
                                best_p = sorted([p for p in self.regressor.first_pareto_front],
                                                key=lambda obj: obj.fitness.get(fitness.label, -float('inf')), reverse=True)[:get_best_k]

                                best_p_validation = sorted([p for p in self.regressor.first_pareto_front],
                                                           key=lambda obj: obj.fitness_validation.get(fitness.label, -float('inf')), reverse=True)[:get_best_k]

                            for i, p in enumerate(best_p):

                                to_append = {
                                    'client': self.name, 'federated_round': self.federated_rounds_executed}
                                to_append['index'] = i + 1
                                to_append[f'Complexity'] = p.complexity
                                for fitness_inner in self.regressor.fitness_functions:
                                    to_append[f'{fitness_inner.label} on training'] = p.fitness.get(
                                        fitness_inner.label, np.nan)
                                    to_append[f'{fitness_inner.label} on validation'] = p.fitness_validation.get(
                                        fitness_inner.label, np.nan)
                                    to_append[f'best_of'] = f'Best {get_best_k} by {fitness.label} on training'
                                extracted.append(to_append.copy())

                            for i, p in enumerate(best_p_validation):

                                to_append = {
                                    'client': self.name, 'federated_round': self.federated_rounds_executed}
                                to_append['index'] = i + 1
                                to_append[f'Complexity'] = p.complexity
                                for fitness_inner in self.regressor.fitness_functions:
                                    to_append[f'{fitness_inner.label} on training'] = p.fitness.get(
                                        fitness_inner.label, np.nan)
                                    to_append[f'{fitness_inner.label} on validation'] = p.fitness_validation.get(
                                        fitness_inner.label, np.nan)
                                    to_append[f'best_of'] = f'Best {get_best_k} by {fitness.label} on validation'
                                extracted.append(to_append.copy())

                        to_append = {'client': self.name,
                                     'federated_round': self.federated_rounds_executed}

                        least_complexity = sorted([p for p in self.regressor.first_pareto_front],
                                                  key=lambda obj: obj.complexity)[:get_best_k]

                        to_append = {'client': self.name,
                                     'federated_round': self.federated_rounds_executed}
                        for i, p in enumerate(least_complexity):
                            to_append['index'] = i
                            to_append[f'Complexity'] = p.complexity
                            to_append[f'best_of'] = f'Best {get_best_k} by complexity'

                            for fitness in self.regressor.fitness_functions:
                                to_append[f'{fitness.label} on training'] = p.fitness.get(
                                    fitness.label, np.nan)
                                to_append[f'{fitness.label} on validation'] = p.fitness_validation.get(
                                    fitness.label, np.nan)
                            extracted.append(to_append.copy())

                        performance = pd.concat(
                            [performance, pd.DataFrame(extracted)], ignore_index=True)

                        performance.to_csv(performance_path, index=False)

                    ############################# PERFORMANCE LOGGING ##############################
                    ################################################################################

                self.regressor.fit(
                    data=data,
                    val_data=val_data,
                    features=self.training_configuration['features'],
                    operations=self.training_configuration['operations'],
                    fitness_functions=self.training_configuration['fitness_functions'],
                    generations_to_train=self.training_configuration['generations_to_train'],
                    n_jobs=self.training_configuration['n_jobs'],
                    stop_at_convergence=self.training_configuration['stop_at_convergence'],
                    verbose=self.training_configuration['verbose'],
                )

                # Print how many programs have is_valid=False
                print(
                    f'######### Invalid: {len([p for p in self.regressor.population if not p.is_valid])}/{len(self.regressor.population)}')

                # This is needed to not increment the federated round at the end of the MOSR in the client
                self.federated_round_is_terminated = False

            elif self.stage == 2:

                logging.debug(f'Client {self.name} is executing stage 2')

                self.regressor.data_shape = data.shape

                if self.configuration['federated'].get('compatibility_check', False):
                    for program in self.regressor.population:
                        program: Program
                        program.bootstrap(
                            data=data,
                            target=self.training_configuration['target'],
                            weights=self.training_configuration['weights'],
                            constants_optimization=self.training_configuration['constants_optimization'],
                            constants_optimization_conf=self.training_configuration[
                                'constants_optimization_conf'],
                            k=self.training_configuration.get(
                                'bootstrap_k', 100),
                            frac=self.training_configuration.get(
                                'bootstrap_frac', 0.6),
                            inplace=True
                        )

                else:
                    self.regressor.compute_fitness_population(
                        data=data, validation=False, validation_federated=False, simplify=False)
                    self.regressor.compute_fitness_population(
                        data=val_data, validation=True, validation_federated=False, simplify=False)

                self.federated_round_is_terminated = False

            else:
                raise ValueError(f'Invalid stage {self.stage}')
