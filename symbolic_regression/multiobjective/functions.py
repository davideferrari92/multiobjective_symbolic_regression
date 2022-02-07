import math
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score, roc_curve
from symbolic_regression.multiobjective.optimization import optimize
from symbolic_regression.multiobjective.utils import to_logistic
from symbolic_regression.Program import Program


def binary_cross_entropy(program: Program,
                         data: Union[pd.DataFrame, pd.Series],
                         target: str,
                         logistic: bool = True,
                         constants_optimization: bool = False,
                         constants_optimization_conf: dict = {}):

    if logistic:
        prog = to_logistic(program=program)
    else:
        prog = program

    if constants_optimization:
        optimized = optimize(program=prog,
                             data=data,
                             target=target,
                             weights=None,
                             constants_optimization_conf=constants_optimization_conf,
                             task='binary:logistic')
        prog = optimized
        program.program = prog.program.operands[0]
        program.program.father = None

    pred = np.array(prog.evaluate(data=data))
    ground_truth = data[target]

    try:
        bce = log_loss(ground_truth, pred)
        return bce
    except ValueError:
        return np.inf


def accuracy_bce(program: Program,
                 data: Union[pd.DataFrame, pd.Series],
                 target: str,
                 logistic: bool = True,
                 threshold: float = .5
                 ):
    if logistic:
        prog = to_logistic(program=program)
    else:
        prog = program

    try:
        pred = np.array(prog.evaluate(data=data))
        pred = (pred > threshold).astype('int')
    except TypeError:
        return np.nan

    ground_truth = data[target].astype('int')

    return accuracy_score(ground_truth, pred)


def precision_bce(program: Program,
                  data: Union[pd.DataFrame, pd.Series],
                  target: str,
                  logistic: bool = True,
                  threshold: float = .5
                  ):
    if logistic:
        prog = to_logistic(program=program)
    else:
        prog = program

    try:
        pred = np.array(prog.evaluate(data=data))
        pred = (pred > threshold).astype('int')
    except TypeError:
        return np.nan

    ground_truth = data[target].astype('int')

    return precision_score(ground_truth, pred)


def recall_bce(program: Program,
               data: Union[pd.DataFrame, pd.Series],
               target: str,
               logistic: bool = True,
               threshold: float = .5
               ):
    if logistic:
        prog = to_logistic(program=program)
    else:
        prog = program

    try:
        pred = np.array(prog.evaluate(data=data))
        pred = (pred > threshold).astype('int')
    except TypeError:
        return np.nan

    ground_truth = data[target].astype('int')

    return recall_score(ground_truth, pred, normalize=True)


def f1_bce(program: Program,
           data: Union[pd.DataFrame, pd.Series],
           target: str,
           logistic: bool = True,
           threshold: float = .5
           ):
    if logistic:
        prog = to_logistic(program=program)
    else:
        prog = program

    try:
        pred = np.array(prog.evaluate(data=data))
        pred = (pred > threshold).astype('int')
    except TypeError:
        return np.nan

    ground_truth = data[target].astype('int')

    return f1_score(ground_truth, pred)


def auroc_bce(program: Program,
              data: Union[pd.DataFrame, pd.Series],
              target: str,
              logistic: bool = True):
    """
    We compute ROC curve at different threshold values.
    The function returns the AUC and the performance at the optimal
    threshold expressed by means of G-mean value.
    """
    if logistic:
        prog = to_logistic(program=program)
    else:
        prog = program

    try:
        pred = np.array(prog.evaluate(data=data))
        ground_truth = data[target].astype('int')
        # 1- is because the Pareto optimality minimizes the fitness function
        # instead the AUC should be maximized
        return 1-roc_auc_score(ground_truth, pred)

    except TypeError:
        return np.inf
    except ValueError:
        return np.inf


def gmeans(program: Program,
           data: Union[pd.DataFrame, pd.Series],
           target: str,
           logistic: bool = True):
    """
    Best performance at the threshold variation.
    Interpret this as the accuracy with the best threshold
    """
    if logistic:
        prog = to_logistic(program=program)
    else:
        prog = program

    try:
        pred = np.array(prog.evaluate(data=data))
        if len(pred) == 0:
            return np.inf
    except TypeError:
        return np.inf

    ground_truth = data[target]
    fpr, tpr, thresholds = roc_curve(ground_truth, pred)
    gmeans = np.sqrt(tpr * (1-fpr))
    best_gmean = gmeans[np.argmax(gmeans)]

    # 1- is because the Pareto optimality minimizes the fitness function
    # instead the G-mean should be maximized
    return 1 - best_gmean


def complexity(program: Program):
    return program.complexity


def wmse(program: Program,
         data: Union[pd.DataFrame, pd.Series],
         target: str,
         weights: str = None,
         constants_optimization: bool = False,
         constants_optimization_conf: dict = {}
         ) -> float:
    """ Evaluates the weighted mean squared error
    """

    if constants_optimization:
        optimized = optimize(program=program,
                             data=data,
                             target=target,
                             weights=weights,
                             constants_optimization_conf=constants_optimization_conf,
                             task='regression:wmse'
                             )
        program.program = optimized.program

    pred = optimized.evaluate(data=data)

    if weights:
        wmse = (((pred - data[target]) ** 2) * data[weights]).mean()
    else:
        wmse = (((pred - data[target]) ** 2)).mean()
    return wmse


def not_constant(program: Program,
                 data: Union[dict, pd.DataFrame],
                 epsilon: float = .01) -> float:
    """ This function measures how much a program differs from a constant.

    We use the standard deviation as a measure of the variability
    of the output of the program when evaluated on the given dataset

    Args:
        program: The program to evaluate
        data: The data on which to evaluate the program
        epsilon: A minimum accepted variation for the given program
    """

    result = program.evaluate(data=data)

    try:
        std_dev = np.std(result)
    except AttributeError:
        return np.nan

    return np.max([0, epsilon - std_dev])


def value_range(program: Program,
                data: Union[dict, pd.DataFrame],
                lower_bound: float,
                upper_bound: float) -> float:
    """
    f(x) - upper_bound <= 0
    lower_bound - f(x) <= 0
    """

    result = program.evaluate(data=data)

    upper_bound_constraint = np.mean(np.where(
        np.array(result) - upper_bound >= 0, np.array(result) - upper_bound, 0))
    lower_bound_constraint = np.mean(np.where(
        lower_bound - np.array(result) >= 0, lower_bound - np.array(result), 0))

    return upper_bound_constraint + lower_bound_constraint


def ordering(program: Program,
             data: pd.DataFrame,
             target: str,
             method: str = 'error'
             ) -> float:

    if method not in ['inversions', 'error']:
        print(f'Only support inversions or error. Default is error')
        method = 'error'

    data_ord = data.copy(deep=True)
    data_ord['pred'] = program.evaluate(data=data_ord)

    data_ord.sort_values(by='pred', ascending=False, inplace=True)

    error = 0
    inversions = 0
    for index, row in data_ord.iterrows():
        mask = (row['pred'] - data_ord.loc[(data_ord.index > index)
                & (data_ord[target] < row[target]), 'pred'])
        inversions += len(mask)
        error += (mask).sum()

    if method == 'error':
        return error

    if method == 'inversions':
        return inversions


def ordering_preserving(program: Program,
                        data: pd.DataFrame,
                        target: str,
                        method: str = 'abs_val'
                        ) -> float:
    """
    1) sort original index
    2) get position of each sample
    3) evaluate program
    4) sort program result
    5) difference positions original vs program result
    6) error = error + difference ????
    """

    if method not in ['abs_val', 'inversions', 'inversions_and_error', 'error']:
        print(
            f'Only support abs_val, inversions, inversions_and_error, or error. Default is inversions')

        method == 'inversion'

    data_ord = data.copy(deep=True)

    ''' Index ordered by baseline target '''
    data_ord.sort_values(by=target, ascending=False, inplace=True)
    data_ord['ordering_target'] = data_ord.reset_index(inplace=False).index

    ''' Index ordered by program prediction '''
    data_ord['pred'] = program.evaluate(data=data_ord)

    # The number of inversions to match target ordering
    argsort_pred = len(data_ord) - 1 - np.argsort(data_ord['pred'].to_numpy())

    if method in ['inversions', 'inversions_and_error', 'error']:
        from symbolic_regression.multiobjective.utils import \
            merge_sort_inversions
        inversions, err = merge_sort_inversions(
            arr=argsort_pred, data_ord_pred=data_ord['pred'])

        # Maximum number of inversions
        inv = len(argsort_pred) * (len(argsort_pred) - 1) * 0.5

        if method == 'inversions':

            return inversions / inv

        if method == 'inversions_and_error':

            return inversions / inv, err / inversions

        if method == 'error':

            return err / inversions

    elif method == 'abs_val':
        data_ord.sort_values(by='pred', ascending=False, inplace=True)
        data_ord['ordering_program'] = data_ord.reset_index(
            inplace=False).index

        data_ord['error'] = data_ord['pred'] - data_ord[target]
        data_ord['error_ordering'] = data_ord['ordering_program'] - \
            data_ord['ordering_target']

        error = 0.0

        for _, row_i in data_ord.iterrows():
            for _, row_j in data_ord.iterrows():

                if row_j[target] < row_i[target]:
                    error += row_i['pred'] - row_j['pred']
        return error
