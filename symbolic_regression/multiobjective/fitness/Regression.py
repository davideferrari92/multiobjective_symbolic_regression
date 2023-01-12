import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


class WeightedMeanSquaredError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)

        program_to_evaluate = program.to_logistic(
            inplace=False) if self.logistic else program

        pred = program_to_evaluate.evaluate(data=data)

        if self.weights not in data.columns:
            data[self.weights] = self._create_regression_weights(
                data=data, target=self.target, bins=self.bins)

        try:
            wmse = (((pred - data[self.target])**2) * data[self.weights]
                    ).mean() if self.weights else ((pred - data[self.target])**2).mean()

            return wmse
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf


class WeightedMeanAbsoluteError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)

        program_to_evaluate = program.to_logistic(
            inplace=False) if self.logistic else program

        pred = program_to_evaluate.evaluate(data=data)

        if self.weights not in data.columns:
            data[self.weights] = self._create_regression_weights(
                data=data, target=self.target, bins=self.bins)

        try:
            wmae = (np.abs(pred - data[self.target]) * data[self.weights]
                    ).mean() if self.weights else np.abs(pred - data[self.target]).mean()

            return wmae
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf


class WeightedRelativeRootMeanSquaredError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)

        program_to_evaluate = program.to_logistic(
            inplace=False) if self.logistic else program

        pred = program_to_evaluate.evaluate(data=data)

        if self.weights not in data.columns:
            data[self.weights] = self._create_regression_weights(
                data=data, target=self.target, bins=self.bins)

        try:
            if self.weights:
                y_av = 1e-20+(data[self.target] *
                              data[self.weights]).mean()
                wmse = np.sqrt(
                    (((pred - data[self.target])**2) * data[self.weights]).mean())*100./y_av
            else:
                y_av = 1e-20+(data[self.target]).mean()
                wmse = np.sqrt(
                    (((pred - data[self.target])**2)).mean())*100./y_av
            return wmse
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf


class MeanAveragePercentageError(BaseFitness):
    """ Mean Average Percentage Error (MAPE) """

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)

        if not program.is_valid:
            return np.inf

        program_to_evaluate = program.to_logistic(
            inplace=False) if self.logistic else program

        pred = pd.Series(program_to_evaluate.evaluate(data=data))

        if len(pred.dropna()) < len(pred):
            return np.inf

        try:
            """
            We need to normalize between 0 and 1 both the prediction and the target
            because the MAPE is not scale invariant.
            """
            if isinstance(pred, float):
                pred = np.full(shape=len(data[self.target]), fill_value=pred)

            scaler = MinMaxScaler()
            target = np.array(data[self.target]).reshape(-1, 1)
            scaler.fit(target)

            pred = scaler.transform(np.array(pred).reshape(-1, 1))

            mape = np.mean(np.abs((pred - target) / target))
            return mape
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf


class NotConstant(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - epsilon: float

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)

        pred = program.evaluate(data=data)

        try:
            std_dev = np.std(pred)
            return np.max([0, self.epsilon - std_dev])
        except AttributeError:
            return np.nan
        except TypeError:
            return self.epsilon


class ValueRange(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:
        - lower_bound: float
        - upper_bound: float

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)

        pred = program.evaluate(data=data)

        upper_bound_constraint = np.mean(
            np.where(
                np.array(pred) - self.upper_bound >= 0,
                np.array(pred) - self.upper_bound, 0))
        lower_bound_constraint = np.mean(
            np.where(self.lower_bound - np.array(pred) >= 0,
                     self.lower_bound - np.array(pred), 0))

        return upper_bound_constraint + lower_bound_constraint
