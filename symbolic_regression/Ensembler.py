import logging
import numpy as np
import pandas as pd


class SymbolicEnsembler:

    def __init__(self, programs_selection) -> None:
        self.programs_selection = programs_selection

    def predict(self, data, threshold: float = None):
        """
        Predicts the output for the given input data using the ensemble of programs.

        Parameters:
        - data: Union[pd.DataFrame, pd.Series, Dict]
            The input data for which predictions need to be made.
        - threshold: float  (default=None)
            The threshold value to be used for binary classification. If None, regression predictions are made.

        Returns:
        - predictions: pd.Series
            The predicted output for the given input data.

        """
        predictions = pd.DataFrame()
        for index, program in enumerate(self.programs_selection):
            predictions[index] = program.predict(data)

        if threshold is not None:
            predictions = (predictions > threshold).astype(int)
            return predictions.mean(axis=1).astype(int)
        else:
            return predictions.mean(axis=1)

    def compute_fitness(self, data, fitness_functions):
        """
        Computes the fitness of the ensembler model on the given data using the specified fitness functions.

        Args:
            data: Union[pd.DataFrame, pd.Series, Dict]
                The input data for which fitness needs to be computed.
            fitness_functions: List[FitnessFunction]
                A list of fitness functions to be used for computing the fitness.

        Returns:
            fitness: Dict
                A dictionary containing the fitness values for each fitness function.
        """

        fitness = {}

        for fitness_function in fitness_functions:
            predictions = pd.Series(
                self.predict(data,
                             threshold=fitness_function.threshold if hasattr(fitness_function, 'threshold') else None))
            try:
                fitness[fitness_function.label] = fitness_function.evaluate(
                    program=None, data=data, validation=False, pred=predictions)
            except Exception as e:
                logging.warning(
                    f'Unable to compute fitness {fitness_function.label}')

        return fitness
