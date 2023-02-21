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

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        if not program.is_valid:
            return np.nan

        if self.logistic:
            program_to_evaluate = program.to_logistic(inplace=False)
            ground_truth = data[self.target].astype('int')
        else:
            program_to_evaluate = program
            ground_truth = data[self.target]

        try:
            pred = np.array(program_to_evaluate.evaluate(data=data))

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
