from abc import abstractmethod

import pandas as pd


class BaseFitness:
    """
    Base class for fitness functions.
    """

    def __init__(self,
                 label: str,
                 logistic: bool = False,
                 one_minus: bool = False,
                 smaller_is_better: bool = True,
                 minimize: bool = True,
                 convergence_threshold: float = None,
                 constants_optimization: str = None,
                 constants_optimization_conf: dict = None,
                 **kwargs
                 ) -> None:

        self.label: str = label
        self.logistic: bool = logistic
        self.one_minus: bool = one_minus
        self.smaller_is_better: bool = smaller_is_better

        self.minimize: bool = minimize
        self.convergence_threshold: float = convergence_threshold
        self.constants_optimization: str = constants_optimization
        self.constants_optimization_conf: dict = constants_optimization_conf
        self.export: bool = False

        self.target: str = None
        self.weights: str = None
        self.bins: int = None
        self.epsilon: float = None
        self.hypervolume_reference: float = None
        self.max_error: float = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        # Return the name of the class with all the arguments
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"

    @abstractmethod
    def evaluate(self, program) -> pd.DataFrame:
        raise NotImplementedError

    def optimize(self, program, data: pd.DataFrame) -> None:
        if self.constants_optimization:
            program.optimize(
                data=data,
                target=self.target,
                weights=self.weights,
                constants_optimization=self.constants_optimization,
                constants_optimization_conf=self.constants_optimization_conf,
                inplace=True
            )

        return program
