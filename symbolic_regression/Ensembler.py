import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


class SymbolicEnsembler:

    def __init__(self, programs_selection: List[Program]) -> None:
        self.programs_selection = programs_selection

    def predict(self, data, threshold: float = None, logistic: bool = False, voting_threshold: float = None):
        """
        Predicts the output for the given input data using the ensemble of programs.

        Parameters:
        - data: Union[pd.DataFrame, pd.Series, Dict]
            The input data for which predictions need to be made.
        - threshold: float  (default=None)
            The threshold value to be used for binary classification. If None, regression predictions are made.
        - logistic: bool (default=False)
            If True, the logistic function is applied to the predictions.
        - voting_threshold: float (default=None)
            The threshold value to be used for voting. If None, regression predictions are made.

        Returns:
        - predictions: pd.Series
            The predicted output for the given input data.

        """

        predictions = pd.DataFrame()
        for index, program in enumerate(self.programs_selection):
            if logistic:
                predictions[index] = program.to_logistic(
                    inplace=False).predict(data)
            else:
                predictions[index] = program.predict(data)

        if isinstance(threshold, float) and 0 <= threshold <= 1:
            predictions = (predictions > threshold).astype(int)

            predictions = predictions.mean(axis=1)
            if isinstance(voting_threshold, float) and 0 <= voting_threshold <= 1:
                predictions = predictions.apply(lambda x: 1 if x >= voting_threshold else (0 if x <= (1 - voting_threshold) else np.nan))
            
        else:
            predictions = predictions.mean(axis=1)

            if isinstance(voting_threshold, float) and 0 <= voting_threshold <= 1:
                predictions = predictions.apply(lambda x: 1 if x >= voting_threshold else (0 if x <= (1 - voting_threshold) else np.nan))
            
        return predictions


    def evaluate(self, data, threshold: float = None, logistic: bool = False, voting_threshold: float = None):
        return self.predict(data, threshold, logistic, voting_threshold)

    def compute_fitness(self, data: Union[pd.DataFrame, pd.Series, Dict], fitness_functions: List[BaseFitness], validation: bool = False):
        """
        Computes the fitness of the ensembler model on the given data using the specified fitness functions.

        Args:
            data: Union[pd.DataFrame, pd.Series, Dict]
                The input data for which fitness needs to be computed.
            fitness_functions: List[BaseFitness]
                A list of fitness functions to be used for computing the fitness.

        Returns:
            fitness: Dict
                A dictionary containing the fitness values for each fitness function.
        """

        fitness = {}

        for fitness_function in fitness_functions:
            predictions = pd.Series(
                self.predict(data,
                             threshold=fitness_function.threshold if hasattr(
                                 fitness_function, 'threshold') else None,
                             logistic=fitness_function.logistic if hasattr(fitness_function, 'logistic') else False)
            )
            try:
                fitness_ret = fitness_function.evaluate(
                    program=None, data=data, validation=validation, pred=predictions)

                if isinstance(fitness_ret, tuple):
                    fitness[fitness_function.label] = fitness_ret[0]
                else:
                    fitness[fitness_function.label] = fitness_ret
            except Exception as e:
                logging.warning(
                    f'Unable to compute fitness {fitness_function.label}')

        return fitness
