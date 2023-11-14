import logging
from typing import Dict, List

import pandas as pd

from symbolic_regression.federated.strategies.BaseStrategy import BaseStrategy
from symbolic_regression.Population import Population
from symbolic_regression.Program import Program
from symbolic_regression.SymbolicRegressor import SymbolicRegressor


class FedNSGAII(BaseStrategy):
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

    def on_start(self, data: pd.DataFrame = None, regressors: Dict[str, SymbolicRegressor] = None, **kwargs):
        if self.mode == 'server':
            self.regressors = regressors
            for client_name, client_regressor in self.regressors.items():
                client_regressor.population: Population = client_regressor.population.as_program()

            self.regressor: SymbolicRegressor = SymbolicRegressor(
                client_name=self.name,
                const_range=self.symbolic_regressor_configuration['const_range'],
                genetic_operators_frequency=self.symbolic_regressor_configuration[
                    'genetic_operators_frequency'],
                parsimony=self.symbolic_regressor_configuration['parsimony'],
                parsimony_decay=self.symbolic_regressor_configuration['parsimony_decay'],
                population_size=self.symbolic_regressor_configuration['population_size'],
                tournament_size=self.symbolic_regressor_configuration['tournament_size'],
                callbacks=self.symbolic_regressor_configuration.get('callbacks', list())
            )

            # Overrides this regressor's configuration with the configuration of the strategy
            for training_variable, value in self.training_configuration.items():
                setattr(self.regressor, training_variable, value)

            self.regressor.population: Population = Population()

        elif self.mode == 'client':

            if not self.regressor:
                logging.info(
                    f'Initializing the symbolic regressor for {self.name} with size {self.symbolic_regressor_configuration["population_size"]}')
                self.regressor: SymbolicRegressor = SymbolicRegressor(
                    client_name=self.name,
                    const_range=self.symbolic_regressor_configuration['const_range'],
                    genetic_operators_frequency=self.symbolic_regressor_configuration[
                        'genetic_operators_frequency'],
                    parsimony=self.symbolic_regressor_configuration['parsimony'],
                    parsimony_decay=self.symbolic_regressor_configuration['parsimony_decay'],
                    population_size=self.symbolic_regressor_configuration['population_size'],
                    tournament_size=self.symbolic_regressor_configuration['tournament_size'],
                    callbacks=self.symbolic_regressor_configuration.get('callbacks', list())
                )

            self.regressor.population: Population = self.regressor.population.as_program()

            self.regressor.times: pd.DataFrame = pd.DataFrame()

    def aggregation(self, data: pd.DataFrame = None, val_data: pd.DataFrame = None, **kwargs):
        '''
        This method implements the aggregation function of the strategy for both
        server and client. Depending on the mode, it executes different branches 
        of the algorithm as stated below.

        When the mode is 'server':
            - Aggregate the regressors from the clients

        When the mode is 'client':
            - Train the local regressor for a number of generations passed as generations_to_train
        '''

        if self.mode == 'server':
            '''
            This aggregator simply joins the populations of the clients and
            selects the best programs based on their fitness values.
            '''

            for client_name, client_regressor in self.regressors.items():
                logging.info(
                    f'Incorporating {client_name} regressor population')

                self.regressor.population.extend(client_regressor.population)

            self.regressor._create_pareto_front()
            self.regressor._crowding_distance()
            self.regressor.population.sort(
                key=lambda p: p.crowding_distance, reverse=True)
            self.regressor.population.sort(key=lambda p: p.rank, reverse=False)
            for p in self.regressor.population:
                p.programs_dominated_by: List[Program] = list()
                p.programs_dominates: List[Program] = list()
            self.regressor.population = self.regressor.population[:self.regressor.population_size]

        elif self.mode == 'client':

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

    def on_termination(self, **kwargs):

        if self.mode == 'server':
            self.regressor.population: Population = Population(
                self.regressor.population).as_binary()

        elif self.mode == 'client':
            self.regressor.population: Population = Population(
                self.regressor.population).as_binary()

    def on_validation(self, data: pd.DataFrame = None, val_data: pd.DataFrame = None, **kwargs):
        if self.mode == 'server':
            pass

        elif self.mode == 'client':
            self.regressor.population: Population = self.regressor.population.as_program()
            self.regressor.compute_fitness_population(data=data, validation=False, validation_federated=True, simplify=False)
            self.regressor.compute_fitness_population(data=val_data, validation=True, validation_federated=True, simplify=False)
            self.regressor.population: Population = self.regressor.population.as_binary()