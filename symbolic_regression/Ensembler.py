import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


class SymbolicEnsembler:

    def __init__(self, programs_selection: List[Program]) -> None:
        self.programs_selection = programs_selection

    def predict(self,
                data,
                logistic: bool = False,
                raw: bool = False,
                threshold: float = None,
                voting_threshold: float = None,
                min_proba: float = None,
                ):
        """
        Predicts the output for the given input data using the ensemble of programs.

        Parameters:
        - data: Union[pd.DataFrame, pd.Series, Dict]
            The input data for which predictions need to be made.
        - logistic: bool (default=False)
            If True, the logistic function is applied to the predictions.
        - raw: bool (default=False)
            If True, the raw predictions of each program are returned.
        - threshold: float  (default=None)
            The threshold value to be used for binary classification. If None, regression predictions are made.
        - voting_threshold: float (default=None)
            The threshold value to be used for voting. If None, regression predictions are made.
        - min_proba: float (default=None)
            The minimum odd value to be used for voting. If None, regression predictions are made

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

        if raw:
            return predictions
        if threshold is None:
            predictions = predictions.mean(axis=1)
            if min_proba is not None:
                predictions[(predictions >= (1 - min_proba)) &
                            (predictions <= min_proba)] = np.nan
            return predictions

        if not isinstance(threshold, float):
            raise TypeError(
                'Invalid threshold. Threshold should be a float.')

        if not 0 <= threshold <= 1:
            raise ValueError(
                'Invalid threshold. Threshold should be between 0 and 1.')

        if voting_threshold is None:
            predictions = (predictions.mean(axis=1) > threshold).astype(int)
            return predictions
        else:
            predictions = (predictions > threshold).astype(int)

        if not isinstance(voting_threshold, float):
            raise TypeError(
                'Invalid voting threshold. Voting threshold should be a float.')

        def cutoff(x, threshold):
            if x >= threshold:
                return 1
            elif x <= (1 - threshold):
                return 0
            else:
                return np.nan

        if 0.5 <= voting_threshold <= 1:
            # return predictions.mean(axis=1)
            predictions = predictions.mean(axis=1).apply(
                cutoff, threshold=voting_threshold)
            return predictions

        else:
            raise ValueError(
                'Invalid voting threshold. Voting threshold should be between 0.5 and 1.')

    def predict_proba(self, data):
        """
        Predicts the probability of the output for the given input data using the ensemble of programs.

        Parameters:
        - data: Union[pd.DataFrame, pd.Series, Dict]
            The input data for which predictions need to be made.

        Returns:
        - predictions: pd.Series
            The predicted probability of the output for the given input data.

        """

        probas = [pd.DataFrame()]
        for index, program in enumerate(self.programs_selection):
            # we iterate over program and append dataframe of probas
            # TODO: reset index is necessary, maybe check if returned obj is a dataframe
            proba = program.predict_proba(data=data).reset_index()
            probas.append(proba)
        
        proba = pd.concat(probas)  # we concat every proba
        # here using the row number of instance proba we aggregate over program probas
        proba = proba.groupby(
            level=list(range(len(proba.index.names))),
            observed=True) \
            .aggregate('mean')

        return proba[['proba_0', 'proba_1']]

    def evaluate(self,
                 data,
                 logistic: bool = False,
                 raw: bool = False,
                 threshold: float = None,
                 voting_threshold: float = None,
                 min_proba: float = None,
                 ):
        return self.predict(data=data, logistic=logistic, raw=raw, threshold=threshold, voting_threshold=voting_threshold, min_proba=min_proba)

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
