from abc import abstractmethod

import numpy as np
import pandas as pd
from astropy import stats


class BaseFitness:
    """
    Base class for fitness functions.
    """

    def __init__(self,
                 label: str,
                 logistic: bool = False,
                 one_minus: bool = False,
                 minimize: bool = True,
                 convergence_threshold: float = None,
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

