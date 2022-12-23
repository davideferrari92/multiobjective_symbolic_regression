from abc import abstractmethod
from astropy import stats
import pandas as pd
import numpy as np


class BaseFitness:
    """
    Base class for fitness functions.
    """

    def __init__(self,
                 label: str,
                 logistic: bool = False,
                 one_minus: bool = False,
                 minimize: bool = True,
                 convergence_threshold: float = 0.01,
                 constants_optimization: str = None,
                 constants_optimization_conf: dict = None,
                 **kwargs
                 ) -> None:

        self.label: str = label
        self.logistic: bool = logistic
        self.one_minus: bool = one_minus
        self.minimize: bool = minimize
        self.convergence_threshold: float = convergence_threshold
        self.constants_optimization: str = constants_optimization
        self.constants_optimization_conf: dict = constants_optimization_conf

        self.data: pd.DataFrame = None
        self.target: str = None
        self.weights: str = None
        self.bins: int = None
        self.epsilon: float = None
        self.hypervolume_reference: float = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def evaluate(self, program) -> pd.DataFrame:
        raise NotImplementedError

    def optimize(self, program, data: pd.DataFrame) -> None:
        if not self.constants_optimization:
            return

        program.optimize(
            data=data,
            target=self.target,
            weights=self.weights,
            constants_optimization=self.constants_optimization,
            constants_optimization_conf=self.constants_optimization_conf,
            inplace=True
        )

    def _get_cumulant_hist(self, data: pd.DataFrame, target: str, bins: int = None) -> np.array:

        y_true = np.array(data[target])

        # rescale
        rescaled_y_true = (y_true-np.min(y_true)) / \
            (np.max(y_true)-np.min(y_true))

        # compute optimal density function histogram
        if not bins:
            pd_y_true_grid, y_grid = stats.histogram(
                rescaled_y_true, bins='knuth', density=True)
        else:
            pd_y_true_grid, y_grid = stats.histogram(
                rescaled_y_true, bins=bins, density=True)

        # compute grid steps
        dy = y_grid[1]-y_grid[0]

        # compute optimal cumulative histogram
        F_y = np.sum(dy*pd_y_true_grid *
                     np.tril(np.ones(pd_y_true_grid.size), 0), 1)

        return F_y

    def _create_regression_weights(self, data: pd.DataFrame, target: str, bins: int = None):

        y = np.array(data[target])

        if not bins:
            count, division = stats.histogram(y, bins='knuth', density=True)
        else:
            count, division = stats.histogram(y, bins=bins, density=True)

        effective_bins = np.sum((count != 0).astype(int))
        aw = (np.sum(count)/effective_bins)
        weights = np.where(count != 0., aw/count, 0.)
        w_column = np.zeros((y.size,))  # create the weight column

        for i in range(len(count)):
            w_column += (y >= division[i])*(y <= division[i+1])*weights[i]

        return w_column