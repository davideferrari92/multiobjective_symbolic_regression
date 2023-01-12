import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


class BaseClassification(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str
        - threshold: float 

        """
        super().__init__(**kwargs)
        self.classification_metric = None

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

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

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

        self.optimize(program=program, data=data)

        if self.logistic:
            program_to_evaluate = program.to_logistic(inplace=False)

        pred = np.array(program_to_evaluate.evaluate(data=data))
        ground_truth = data[self.target]

        sample_weights = data[self.weights] if self.weights else None

        try:
            return log_loss(y_true=ground_truth,
                            y_pred=pred,
                            sample_weight=sample_weights)
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
        self.classification_metric = precision_score


class Recall(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = recall_score


class F1Score(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = f1_score


class AUC(BaseClassification):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.classification_metric = roc_auc_score

    def evaluate(self, program: Program, data: pd.DataFrame) -> float:

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

    def evaluate(self, program: Program, data: pd.DataFrame) -> pd.DataFrame:

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
