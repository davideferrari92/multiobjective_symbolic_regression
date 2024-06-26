from typing import Dict
import numpy as np
import pandas as pd
from astropy import stats
from sklearn.preprocessing import MinMaxScaler
from sympy import lambdify
import sympy as sym

from symbolic_regression.multiobjective.fitness.Base import BaseFitness


def DiracDeltaV(x):
    return np.where(np.abs(x) < 1e-7, 1e7, 0)


class WeightedMeanSquaredError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program=None, data: pd.DataFrame = None, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        pred = inject.get('pred', pred)

        if pred is None:
            if not program.is_valid:
                return np.nan if not self.export else (np.nan, {'pred': np.nan})

            if not validation:
                program = self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = program_to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf if not self.export else (np.inf, {'pred': pred})

        try:

            wmse = (((pred - data[self.target])**2) * data[self.weights]
                    ).mean() if self.weights and not validation else ((pred - data[self.target])**2).mean()

            if isinstance(self.max_error, float) and wmse > self.max_error:
                return np.inf if not self.export else (np.inf, {'pred': pred})

            return wmse if not self.export else (wmse, {'pred': pred})
        except TypeError:
            return np.inf if not self.export else (np.inf, {'pred': pred})
        except ValueError:
            return np.inf if not self.export else (np.inf, {'pred': pred})


class WeightedMeanAbsoluteError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program=None, data: pd.DataFrame = None, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        pred = inject.get('pred', pred)

        if pred is None:
            if not validation:
                self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = program_to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf if not self.export else (np.inf, {'pred': pred})

        try:
            wmae = (np.abs(pred - data[self.target]) * data[self.weights]
                    ).mean() if self.weights and not validation else np.abs(pred - data[self.target]).mean()

            if isinstance(self.max_error, float) and wmae > self.max_error:
                return np.inf if not self.export else (np.inf, {'pred': pred})

            return wmae if not self.export else (wmae, {'pred': pred})
        except TypeError:
            return np.inf if not self.export else (np.inf, {'pred': pred})
        except ValueError:
            return np.inf if not self.export else (np.inf, {'pred': pred})


class WeightedRelativeRootMeanSquaredError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program=None, data: pd.DataFrame = None, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        if pred is None:
            if not validation:
                self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = program_to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf

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
            return wmse if not self.export else (wmse, {'pred': pred})
        except TypeError:
            return np.inf if not self.export else (np.inf, {'pred': pred})
        except ValueError:
            return np.inf


class MeanAveragePercentageError(BaseFitness):
    """ Mean Average Percentage Error (MAPE) """

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program=None, data: pd.DataFrame = None, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        if pred is None:
            if not validation:
                self.optimize(program=program, data=data)

            if not program.is_valid:
                return np.inf

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = pd.Series(program_to_evaluate.evaluate(data=data))

        if np.isnan(pred).any():
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
            return mape if not self.export else (mape, {'pred': pred})
        except TypeError:
            return np.inf if not self.export else (np.inf, {'pred': pred})
        except ValueError:
            return np.inf if not self.export else (np.inf, {'pred': pred})


class MaxAbsoluteError(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program=None, data: pd.DataFrame = None, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        if pred is None:
            if not program.is_valid:
                return np.inf if not self.export else (np.inf, {'pred': np.nan})

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = pd.Series(program_to_evaluate.evaluate(data=data))

        if np.isnan(pred).any():
            return np.inf

        try:
            ''' Compute the difference between the prediction and the target and extract the maximum value '''
            max_error = np.max(np.abs(pred - data[self.target]))
            return max_error if not self.export else (max_error, {'pred': pred})
        except TypeError:
            return np.inf if not self.export else (np.inf, {'pred': pred})
        except ValueError:
            return np.inf if not self.export else (np.inf, {'pred': pred})


class WMSEAkaike(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program=None, data: pd.DataFrame = None, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan if not self.export else (np.nan, {'pred': np.nan})

            if not validation:
                program = self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = program_to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf

        try:
            k = len(program_to_evaluate.get_constants())

            if self.weights is not None:
                WMSE = (((pred - data[self.target])**2)
                        * data[self.weights]).mean()
            else:
                WMSE = ((pred - data[self.target])**2).mean()

            NLL = len(data[self.target]) / 2 * (1 + np.log(WMSE))

            AIC = (2 * k) + (2 * NLL)
            return AIC

        except TypeError:
            return np.inf
        except ValueError:
            return np.inf
        except NameError:
            return np.inf


class RegressionMinimumDescriptionLength(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

    def evaluate(self, program=None, data: pd.DataFrame = None, validation: bool = False, pred: pd.DataFrame = None, inject: Dict = dict()) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan if not self.export else (np.nan, {'pred': np.nan})

            if not validation:
                program = self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = program_to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf

        try:
            if self.weights is not None:
                WMSE = (((pred - data[self.target])**2)
                        * data[self.weights]).mean()
            else:
                WMSE = ((pred - data[self.target])**2).mean()

            NLL = len(data[self.target]) / 2 * (1 + np.log(WMSE))

            n_features = len(program.features)
            constants = np.array(
                [item.feature for item in program.get_constants(return_objects=True)])
            n_constants = constants.size

            node_states = len(program.operations)+len(program.features)+1
            tree_complexity = program.complexity*np.log(node_states)

            if n_constants == 0:  # No constants in program
                MDL = NLL + tree_complexity
                return MDL

            # Initialize symbols for variables and constants
            x_sym = ''
            for f in program.features:
                x_sym += f'{f},'
            x_sym = sym.symbols(x_sym)
            c_sym = sym.symbols('c0:{}'.format(n_constants))
            p_sym = program.program.render(format_diff=True)

            split_c = np.split(
                constants*np.ones_like(data[[self.target]]), n_constants, 1)
            split_X = np.split(
                data[program.features].to_numpy(), n_features, 1)

            grad = []
            diag_hess = []
            for i in range(n_constants):
                grad.append(sym.diff(p_sym, f'c{i}'))
                diag_hess.append(sym.diff(sym.diff(p_sym, f'c{i}'), f'c{i}'))

            pyf_grad = lambdify([x_sym, c_sym], grad, modules=[
                                'numpy', {'DiracDelta': DiracDeltaV, 'Sqrt': np.sqrt}])
            pyf_diag_hess = lambdify([x_sym, c_sym], diag_hess, modules=[
                                     'numpy', {'DiracDelta': DiracDeltaV}])
            num_grad = pyf_grad(tuple(split_X), tuple(split_c))
            num_diag_hess = pyf_diag_hess(tuple(split_X), tuple(split_c))

            residual = data[self.target] - pred
            residual = np.expand_dims(residual, -1)

            if self.weights is not None:
                w = data[[self.weights]].to_numpy()
                FIM_diag = [np.sum(w * gr**2 - w * residual*hess) /
                            WMSE for (gr, hess) in zip(num_grad, num_diag_hess)]
            else:
                FIM_diag = [np.sum(gr**2 - residual*hess) /
                            WMSE for (gr, hess) in zip(num_grad, num_diag_hess)]

            Delta = [min(np.sqrt(12/fi), np.abs(c))
                     for fi, c in zip(FIM_diag, constants)]

            constant_complexities = [np.log(np.abs(
                c)/d) + np.log(2) if np.abs(c) != d else 0 for c, d in zip(constants, Delta)]
            constant_complexity = np.sum(constant_complexities)

            MDL = NLL + tree_complexity + constant_complexity
            return MDL

        except TypeError:
            return np.inf
        except ValueError:
            return np.inf
        except NameError:
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
