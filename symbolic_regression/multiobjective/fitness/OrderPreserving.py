from symbolic_regression.Program import Program
from symbolic_regression.multiobjective.fitness.Base import BaseFitness
import pandas as pd
import numpy as np
from astropy import stats


class Wasserstein(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This method requires the following arguments:
        - data: pd.DataFrame
        - target: str
        - weights: str  # will be calculated, do not provide!
        - bins: int

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)
        
        F_y = self._get_cumulant_hist(
            data=data, target=self.target, bins=self.bins)
        
        data[self.weights] = self._create_regression_weights(
            data=data, target=self.target, bins=self.bins)

        features = program.features

        try:
            y_pred = np.array(program.evaluate(data[features]))
        except KeyError:
            return np.inf

        # we add -1 so that wasserstein distance belongs to [0,1]
        dy = 1./(F_y.shape[0]-1)

        # rescale between [0,1]
        try:
            rescaled_y_pred = (y_pred-np.min(y_pred)) / \
                (np.max(y_pred)-np.min(y_pred))
            # compute density function histogram based on target optimal one
            pd_y_pred_grid, _ = stats.histogram(
                rescaled_y_pred, bins=F_y.shape[0], density=True)
            # compute optimal cumulative histogram
            F_y_pred = np.sum(dy*pd_y_pred_grid *
                              np.tril(np.ones(pd_y_pred_grid.size), 0), 1)
        except:
            F_y_pred = np.ones_like(F_y)

        return dy*np.sum(np.abs(F_y_pred-F_y))
