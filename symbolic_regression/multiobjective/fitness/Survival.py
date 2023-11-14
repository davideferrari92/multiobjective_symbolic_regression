
import numpy as np
import pandas as pd

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.Program import Program


class CoxEfron(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

        assert hasattr(
            self, 'status'), "Status must be specified. In a Cox model, it is whether the event happened or not."

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if not program.is_valid:
            return np.nan

        if not validation:
            program = self.optimize(program=program, data=data)

        status = self.status
        unique_target = np.sort(
            data[self.target].loc[data[status] == True].unique())

        # how many events at given time
        powers = [len(np.where((data[self.target] == unique_target[el])
                      * (data[status] == True))[0]) for el in range(len(unique_target))]

        # people at risk at time tj
        RJs_indices = [np.where(data[self.target] >= unique_target[el])[
            0] for el in range(len(unique_target))]

        # events at time tj
        DJs_indices = [np.where((data[self.target] == unique_target[el])
                                * (data[status] == True))[0] for el in range(len(unique_target))]

        pred = program.evaluate(data=data)

        if hasattr(pred, 'to_numpy'):
            pred = pred.to_numpy()
        if hasattr(pred, 'shape') and pred.size != 1:
            pred = np.reshape(pred, (len(data), 1))
        elif isinstance(pred, float) or (hasattr(pred, 'shape') and pred.size == 1):
            pred = pred*np.ones((len(data), 1))

        assert pred.shape == (len(data), 1), "wrong shape of prediction array"

        try:
            DFs = [np.sum(pred[els]) for els in DJs_indices]
            MEs = [np.mean(np.exp(pred)[els]) for els in DJs_indices]
            REs = [np.sum(np.exp(pred)[els]) for els in RJs_indices]
            F_TIDES = [np.sum(np.log((REs[el]
                                      - np.expand_dims(np.arange(powers[el]), 1)*MEs[el]))) for el in range(len(powers))]
            LogLikelihood = np.sum(
                np.array([DFs[el]-F_TIDES[el] for el in range(len(powers))]))

            nll = -LogLikelihood
            return nll
        except TypeError:
            return np.inf
        except ValueError:
            return np.inf


class CoxAkaike(BaseFitness):

    def __init__(self, **kwargs) -> None:
        """ This fitness requires the following arguments:

        - target: str
        - weights: str

        """
        super().__init__(**kwargs)

        assert hasattr(
            self, 'status'), "Status must be specified. In a Cox model, it is whether the event happened or not."

    def evaluate(self, program: Program, data: pd.DataFrame, validation: bool = False, pred=None) -> float:

        if not program.is_valid:
            return np.nan

        if not validation:
            program = self.optimize(program=program, data=data)

        status = self.status
        unique_target = np.sort(
            data[self.target].loc[data[status] == True].unique())

        # how many events at given time
        powers = [len(np.where((data[self.target] == unique_target[el])
                      * (data[status] == True))[0]) for el in range(len(unique_target))]

        # people at risk at time tj
        RJs_indices = [np.where(data[self.target] >= unique_target[el])[
            0] for el in range(len(unique_target))]

        # events at time tj
        DJs_indices = [np.where((data[self.target] == unique_target[el])
                                * (data[status] == True))[0] for el in range(len(unique_target))]

        pred = program.evaluate(data=data)

        if hasattr(pred, 'to_numpy'):
            pred = pred.to_numpy()
        if hasattr(pred, 'shape') and pred.size != 1:
            pred = np.reshape(pred, (len(data), 1))
        elif isinstance(pred, float) or (hasattr(pred, 'shape') and pred.size == 1):
            pred = pred*np.ones((len(data), 1))

        assert pred.shape == (len(data), 1), "wrong shape of prediction array"

        try:
            DFs = [np.sum(pred[els]) for els in DJs_indices]
            MEs = [np.mean(np.exp(pred)[els]) for els in DJs_indices]
            REs = [np.sum(np.exp(pred)[els]) for els in RJs_indices]
            F_TIDES = [np.sum(np.log((REs[el]
                                      - np.expand_dims(np.arange(powers[el]), 1)*MEs[el]))) for el in range(len(powers))]
            LogLikelihood = np.sum(
                np.array([DFs[el]-F_TIDES[el] for el in range(len(powers))]))

            nconstants = len(program.get_constants())
            nll = -LogLikelihood
            AIC = 2*(nconstants+nll)
            return AIC

        except TypeError:
            return np.inf
        except ValueError:
            return np.inf
