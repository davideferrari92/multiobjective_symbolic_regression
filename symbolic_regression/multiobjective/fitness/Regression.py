import numpy as np
import pandas as pd
from astropy import stats
from sklearn.preprocessing import MinMaxScaler

from symbolic_regression.multiobjective.fitness.Base import BaseFitness


class WeightedMeanSquaredError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            if not validation:
                program = self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = program_to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf

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

    def evaluate(self, program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
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

    def evaluate(self, program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
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

    def evaluate(self, program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
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


class MaxAbsoluteError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program, data: pd.DataFrame, pred=None, **kwargs) -> float:

        if pred is None:
            if not program.is_valid:
                return np.inf

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = pd.Series(program_to_evaluate.evaluate(data=data))

        if len(pred.dropna()) < len(pred):
            return np.inf

        try:
            ''' Compute the difference between the prediction and the target and extract the maximum value '''
            max_error = np.max(np.abs(pred - data[self.target]))
            return max_error
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf


class WMSEAkaike(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            if not validation:
                program = self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = program_to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf

        try:
            k = len(program_to_evaluate.get_constants())

            WSSE = (((pred - data[self.target])**2) * data[self.weights]
                    ).sum() if self.weights else ((pred - data[self.target])**2).sum()

            AIC = 2*k+WSSE
            return AIC

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

    def evaluate(self, program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
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

    def evaluate(self, program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
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


class Complexity(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program, **kwargs) -> float:

        if program is None:
            return np.nan

        if not program.is_valid:
            return np.nan

        return program.complexity


def create_regression_weights(data: pd.DataFrame, target: str, bins: int = None):

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
