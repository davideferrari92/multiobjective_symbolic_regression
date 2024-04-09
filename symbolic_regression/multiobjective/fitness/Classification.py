import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, log_loss,
                             recall_score, roc_auc_score,
                             roc_curve)
from sympy import lambdify
import sympy as sym

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


def DiracDeltaV(x):
    return np.where(np.abs(x) < 1e-7, 1e7, 0)


class BaseClassification(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str
        - threshold: float 

        """
        super().__init__(**kwargs)
        self.classification_metric = None

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            if not validation:
                self.optimize(program=program, data=data)

            if not self.classification_metric:
                raise AttributeError('Classification metric not defined')

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            try:
                pred = (np.array(program_to_evaluate.evaluate(data=data))
                        > self.threshold).astype('int')
            except TypeError:
                return np.nan

        ground_truth = data[self.target].astype('int')

        try:
            metric = self.classification_metric(ground_truth, pred)
        except ValueError:
            metric = np.nan
        except TypeError:  # Singleton array 0 cannot be considered a valid collection.
            metric = np.nan

        if self.one_minus:
            return 1 - metric
        return metric


class BinaryCrossentropy(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str
        - logistic: bool

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
                self.optimize(program=program, data=data)

            if self.logistic:
                program_to_evaluate = program.to_logistic(inplace=False)

                pred = np.array(program_to_evaluate.evaluate(data=data))

        ground_truth = data[self.target]

        try:
            return log_loss(y_true=ground_truth,
                            y_pred=pred,
                            sample_weight=data[self.weights] if (self.weights and not validation) else None)
        except ValueError:
            return np.inf
        except TypeError:
            return np.inf


class Accuracy(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = accuracy_score


class Precision(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            if not validation:
                self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            try:
                pred = (np.array(program_to_evaluate.evaluate(data=data))
                        > self.threshold).astype('int')
            except TypeError:
                return np.nan

        ground_truth = data[self.target].astype('int')

        try:
            cm = confusion_matrix(ground_truth, pred)

            TP_train = cm[0][0]
            TN_train = cm[1][1]
            FP_train = cm[0][1]
            FN_train = cm[1][0]

            metric = TP_train / (TP_train + FP_train)

        except ValueError:
            metric = np.nan
        except TypeError:  # Singleton array 0 cannot be considered a valid collection.
            metric = np.nan

        if self.one_minus:
            return 1 - metric
        return metric


class Recall(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = recall_score


class Sensitivity(Recall):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class Specificity(BaseFitness):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            if not validation:
                self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            try:
                pred = (np.array(program_to_evaluate.evaluate(data=data))
                        > self.threshold).astype('int')
            except TypeError:
                return np.nan

        ground_truth = data[self.target].astype('int')

        try:
            cm = confusion_matrix(ground_truth, pred)

            TP_train = cm[0][0]
            TN_train = cm[1][1]
            FP_train = cm[0][1]
            FN_train = cm[1][0]

            metric = TN_train / (TN_train + FP_train)

        except ValueError:
            metric = np.nan
        except TypeError:  # Singleton array 0 cannot be considered a valid collection.
            metric = np.nan

        if self.one_minus:
            return 1 - metric
        return metric


class PPV(BaseClassification):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            if not validation:
                self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            try:
                pred = (np.array(program_to_evaluate.evaluate(data=data))
                        > self.threshold).astype('int')
            except TypeError:
                return np.nan

        ground_truth = data[self.target].astype('int')

        try:
            cm = confusion_matrix(ground_truth, pred)

            TP_train = cm[0][0]
            TN_train = cm[1][1]
            FP_train = cm[0][1]
            FN_train = cm[1][0]

            metric = TP_train / (TP_train + FP_train)

        except ValueError:
            metric = np.nan
        except TypeError:  # Singleton array 0 cannot be considered a valid collection.
            metric = np.nan

        if self.one_minus:
            return 1 - metric
        return metric


class NPV(BaseClassification):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not program.is_valid:
                return np.nan

            if not validation:
                self.optimize(program=program, data=data)

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            try:
                pred = (np.array(program_to_evaluate.evaluate(data=data))
                        > self.threshold).astype('int')
            except TypeError:
                return np.nan

        ground_truth = data[self.target].astype('int')

        try:
            cm = confusion_matrix(ground_truth, pred)

            TP_train = cm[0][0]
            TN_train = cm[1][1]
            FP_train = cm[0][1]
            FN_train = cm[1][0]

            metric = TN_train / (TN_train + FP_train)

        except ValueError:
            metric = np.nan
        except TypeError:  # Singleton array 0 cannot be considered a valid collection.
            metric = np.nan

        if self.one_minus:
            return 1 - metric
        return metric


class F1Score(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = f1_score


class AUC(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = roc_auc_score

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
                self.optimize(program=program, data=data)

            if not self.classification_metric:
                raise AttributeError('Classification metric not defined')

            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            try:
                pred = np.array(program_to_evaluate.evaluate(data=data))
            except TypeError:
                return np.nan

        ground_truth = data[self.target].astype('int')

        if pred.size == 1:
            pred = np.repeat(pred, ground_truth.shape[0])

        try:
            metric = self.classification_metric(ground_truth, pred)
        except ValueError:
            metric = np.nan

        if self.one_minus:
            return 1 - metric
        return metric


class AveragePrecision(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = average_precision_score


class GMeans(BaseFitness):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, pred=None, **kwargs) -> pd.DataFrame:

        if pred is None:
            program_to_evaluate = program.to_logistic(
                inplace=False) if self.logistic else program

            pred = np.array(program_to_evaluate.evaluate(data=data))

        ground_truth = data[self.target]

        try:
            fpr, tpr, _ = roc_curve(y_true=ground_truth,
                                    y_score=pred)
        except ValueError:
            return np.inf
        except TypeError:
            return np.inf

        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)

        return gmeans[ix]


class BCEAkaike(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str
        - logistic: bool

        """
        super().__init__(**kwargs)

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if pred is None:
            if not validation:
                self.optimize(program=program, data=data)

            if self.logistic:
                program_to_evaluate = program.to_logistic(inplace=False)

            nconstants = len(program.get_constants())

            pred = np.array(program_to_evaluate.evaluate(data=data))

        try:
            BCE = log_loss(y_true=data[self.target],
                           y_pred=pred,
                           sample_weight=data[self.weights] if (self.weights and not validation) else None)

            AIC = 2 * (nconstants / len(data) + BCE)

            return AIC

        except ValueError:
            return np.inf
        except TypeError:
            return np.inf
        except UnboundLocalError:
            return np.inf


class AkaikeInformationCriteriaBCE(BCEAkaike):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str
        - logistic: bool

        """
        super().__init__(**kwargs)


class ClassificationMinimumDescriptionLength(BaseFitness):

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

            to_evaluate = program.to_logistic(
                inplace=False)

            pred = to_evaluate.evaluate(data=data)

        if np.isnan(pred).any():
            return np.inf

        try:
            if self.weights is not None and not validation:
                BCE = log_loss(y_true=data[self.target],
                               y_pred=pred,
                               sample_weight=data[self.weights])
            else:
                BCE = log_loss(y_true=data[self.target],
                               y_pred=pred,
                               sample_weight=None)

            NLL = len(data) * BCE

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

            pred = np.reshape(
                pred, (data[self.target].shape[0], 1))

            if self.weights is not None:
                w = data[[self.weights]].to_numpy()
                FIM_diag = [np.sum(w*((pred-data[[self.target]].to_numpy())*hess +
                                      (1-pred)*pred*gr**2)) for (gr, hess) in zip(num_grad, num_diag_hess)]
            else:
                FIM_diag = [np.sum((pred-data[[self.target]].to_numpy())*hess +
                                   (1-pred)*pred*gr**2) for (gr, hess) in zip(num_grad, num_diag_hess)]

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
