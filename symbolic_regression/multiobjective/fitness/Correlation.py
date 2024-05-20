import logging
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


class BaseCorrelation(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """
        This fitness requires the following arguments:
        - target: str
        - one_minus: bool
        """
        super().__init__(**kwargs)
        self.correlation_function: callable = None

        if not kwargs.get('one_minus', False) and kwargs.get('smaller_is_better', False):
            logging.warning(
                'Correlations one_minus=False (default) should have smaller_is_better=False (default).')

    def evaluate(self, program: Program, data: pd.DataFrame, pred=None, **kwargs) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program
            pred = np.array(program_to_evaluate.evaluate(data=data))

        ground_truth = data[self.target].astype(
            'int') if self.logistic else data[self.target]

        try:
            cs, _ = self.correlation_function(ground_truth, pred)

            if self.one_minus:
                return 1 - cs
            return cs

        except TypeError:
            return np.inf
        except ValueError:
            return np.inf


class PearsonCorrelation(BaseCorrelation):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.correlation_function = pearsonr


class SpearmanCorrelation(BaseCorrelation):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.correlation_function = spearmanr


class KendallTauCorrelation(BaseCorrelation):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.correlation_function = kendalltau
