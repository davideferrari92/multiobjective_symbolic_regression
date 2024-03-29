import logging
from abc import abstractmethod
from typing import Dict

from symbolic_regression.SymbolicRegressor import SymbolicRegressor


class BaseStrategy:
    def __init__(self, name: str, mode: str, configuration: Dict = {}) -> None:
        """ 
        This class is the base class for all strategies. It implements the basic
        methods and properties that are common to all strategies.

        Args:
            - name: str
                The name of the strategy
            - mode: str
                The mode of the strategy. It can be either 'server', 'client', or 'orchestrator'
            - configuration: Dict   (default: {})
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
        """
        self.name: str = name
        self.mode: str = mode

        self.configuration: Dict = configuration

        self.federated_rounds_executed: int = 0
        self.federated_round_is_terminated: bool = True

        self.regressor: SymbolicRegressor = None
        self.regressors: Dict[str, SymbolicRegressor] = {}

    @property
    def federated_configuration(self) -> Dict:
        return self.configuration['federated']

    @property
    def federated_rounds(self) -> int:
        return self.configuration['federated']['federated_rounds']

    @property
    def is_completed(self) -> bool:
        return self.federated_rounds_executed >= self.federated_configuration['federated_rounds']

    @property
    def min_clients(self) -> int:
        return self.configuration['federated']['min_clients']

    @property
    def symbolic_regressor_configuration(self) -> Dict:
        return self.configuration['symbolic_regressor']

    @property
    def training_configuration(self) -> Dict:
        return self.configuration['training']

    @abstractmethod
    def execute(self, **kwargs):
        """
        This method is the entry point of the strategy.
        All strategies should be executed using this method that automatically
        calls the the on_start, aggregation and on_termination methods.
        New strategies should implement these three methods according to their
        algorithm.
        """

        if hasattr(self, 'stage'):
            stage_str = f' (stage {self.stage})'
        else:
            stage_str = ''

        logging.info(
            f'Executing {self.mode} strategy iteration {self.federated_rounds_executed + 1}/{self.federated_rounds}{stage_str}: on_start')
        self.on_start(**kwargs)

        if self.mode == 'server' and len(self.regressors) < self.min_clients:
            logging.warning(
                f'Not enough clients to execute the strategy {self.name}')
            return

        logging.info(
            f'Executing {self.mode} strategy iteration {self.federated_rounds_executed + 1}/{self.federated_rounds}{stage_str}: aggregation')
        self.aggregation(**kwargs)

        logging.info(
            f'Executing {self.mode} strategy iteration {self.federated_rounds_executed + 1}/{self.federated_rounds}{stage_str}: on_termination')
        self.on_termination(**kwargs)

        if self.federated_round_is_terminated:
            self.federated_rounds_executed += 1

    @abstractmethod
    def aggregation(self, **kwargs):
        '''
        This method implements the aggregation function of the strategy for both
        server and client. Depending on the mode, it executes different branches 
        of the algorithm as stated below.

        You can use this blueprint to implement your strategy:
            if self.mode == 'server':
                # Here the server aggregates the regressors from the clients

            elif self.mode == 'client':
                # Here the client trains the local regressor for a number of generations

        When the mode is 'server':
            - Aggregate the regressors from the clients
            - Assign the aggregated regressor to self.aggregated_regressor

        When the mode is 'client':
            - Train the local regressor for a number of generations passed as generations_per_federated_round
                NB: the local regressor is the one stored in self.federated_regressors['clients'][self.name]
                    it is already accessible through the property self.local_regressor
            - Assign the trained regressor to self.federated_regressors['clients'][self.name]
        '''
        if self.mode == 'server':
            raise NotImplementedError

        elif self.mode == 'client':
            # Initialize the regressor or receive the population from the orchestrator
            raise NotImplementedError

    @abstractmethod
    def on_start(self, **kwargs):
        '''
        This method is called at the beginning of the execution of the strategy.
        It is used to initialize the strategy environment.
        '''
        if self.mode == 'server':
            raise NotImplementedError

        elif self.mode == 'client':
            raise NotImplementedError

    @abstractmethod
    def on_termination(self, **kwargs):
        '''
        This method is called at the end of the execution of the strategy.
        It is used to finalize the strategy environment and to return the aggregated regressor.
        '''
        if self.mode == 'server':
            raise NotImplementedError

        elif self.mode == 'client':
            raise NotImplementedError

    @abstractmethod
    def on_validation(self, **kwargs):
        '''
        This method is called at the end of the execution of the strategy.
        It is used to validat the performance of all the regressors on the local dataset.
        '''
        if self.mode == 'server':
            raise NotImplementedError

        elif self.mode == 'client':
            raise NotImplementedError