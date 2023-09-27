import copy
import logging
import random
from typing import List
import numpy as np

import pandas as pd

from symbolic_regression.federated.strategies.FedNSGAII import FedNSGAII
from symbolic_regression.Population import Population
from symbolic_regression.Program import Program


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

                    for r in range(1, self.federated_configuration.get('max_rank_aggregation', 2) + 1):
                        self.regressor.population.extend(
                            client_regressor.extract_pareto_front(rank=r))

                logging.debug(
                    f'Incorporated population size: {len(self.regressor.population)}')

                self.stage = 2
                self.federated_round_is_terminated = False

            elif self.stage == 2:

                logging.debug(f'Server {self.name} is executing stage 2')

                self.regressor.population: Population = copy.deepcopy(
                    self.regressors[list(self.regressors.keys())[0]].population)

                ineligible = 0

                for ith_programs in zip(*[regressor.population for regressor in self.regressors.values()]):
                    """ Apply constants_confidence_intervals_overlap() to each pair of programs in ith_programs.
                    If any one return False, then all of them are set is_valid=False 
                    """
                    stop = False
                    for i in range(len(ith_programs)):
                        for j in range(i + 1, len(ith_programs)):
                            if not ith_programs[i].is_valid or not ith_programs[j].is_valid or not ith_programs[i].constants_confidence_intervals_overlap(ith_programs[j]):
                                stop = True
                                break

                    if stop:
                        ineligible += 1
                        for program in ith_programs:
                            program._override_is_valid = False

                for index, programs in enumerate(zip(*[regressor.population for regressor in self.regressors.values()])):

                    if len(programs[0].get_constants()) == 0:
                        continue

                    self.regressor.population[index].set_constants(
                        [sum([program.get_constants()[i] for program in programs]) / len(programs) for i in range(len(programs[0].get_constants()))])

                self.stage = 1
                self.federated_round_is_terminated = True

                if self.training_configuration.get('verbose', 0) > 0:
                    logging.info(f"Ineligible: {ineligible}/{len(self.regressor.population)}")
            else:
                raise ValueError(f'Invalid stage {self.stage}')

        elif self.mode == 'client':

            if self.stage == 1:

                logging.debug(f'Client {self.name} is executing stage 1')

                self.regressor.compute_fitness_population(
                    data=data, validation=False, validation_federated=True, simplify=False)
                self.regressor.compute_fitness_population(
                    data=val_data, validation=True, validation_federated=True, simplify=False)

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

                # This is needed to not increment the federated round at the end of the MOSR in the client
                self.federated_round_is_terminated = False

            elif self.stage == 2:

                logging.debug(f'Client {self.name} is executing stage 2')

                for program in self.regressor.population:
                    program.bootstrap(
                        data=data,
                        target=self.training_configuration['target'],
                        weights=self.training_configuration['weights'],
                        constants_optimization=self.training_configuration['constants_optimization'],
                        constants_optimization_conf=self.training_configuration[
                            'constants_optimization_conf'],
                        k=self.training_configuration.get('bootstrap_k', 100),
                        frac=self.training_configuration.get(
                            'bootstrap_frac', 0.6),
                        inplace=True
                    )

                self.federated_round_is_terminated = False

            else:
                raise ValueError(f'Invalid stage {self.stage}')
