from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from symbolic_regression.Node import OperationNode
from symbolic_regression.Program import Program
from symbolic_regression.multiobjective.optimization import optimize


def binary_cross_entropy(program: Program,
                         data: Union[pd.DataFrame, pd.Series],
                         target: str,
                         logistic: bool = True,
                         constants_optimization: bool = False,
                         constants_optimization_conf: dict = {}):

    if logistic:
        from symbolic_regression.operators import OPERATOR_SIGMOID
        logistic_node = OperationNode(
            operation=OPERATOR_SIGMOID['func'],
            arity=OPERATOR_SIGMOID['arity'],
            format_str=OPERATOR_SIGMOID['format_str'],
            format_tf=OPERATOR_SIGMOID.get('format_tf'),
            father=None
        )
        # So the upward pointer of the father is not permanent
        copied = deepcopy(program)
        logistic_node.operands.append(copied.program)
        copied.program.father = logistic_node
        copied.program = logistic_node
        prog = copied
    else:
        prog = program

    if constants_optimization:
        program.program = optimize(program=prog,
                                   data=data,
                                   target=target,
                                   weights=None,
                                   constants_optimization_conf=constants_optimization_conf)
    
    # B C E here
    pass


def complexity(program: Program):
    return program.complexity


def wmse(program: Program,
         data: Union[pd.DataFrame, pd.Series],
         target: str,
         weights: str = None,
         logistic: bool = False,
         constants_optimization: bool = False,
         constants_optimization_conf: dict = {}
         ) -> float:
    """ Evaluates the weighted mean squared error
    """

    if logistic:
        from symbolic_regression.operators import OPERATOR_SIGMOID
        logistic_node = OperationNode(
            operation=OPERATOR_SIGMOID['func'],
            arity=OPERATOR_SIGMOID['arity'],
            format_str=OPERATOR_SIGMOID['format_str'],
            format_tf=OPERATOR_SIGMOID.get('format_tf'),
            father=None
        )
        # So the upward pointer of the father is not permanent
        copied = deepcopy(program)
        logistic_node.operands.append(copied.program)
        copied.program.father = logistic_node
        copied.program = logistic_node
        prog = copied
    else:
        prog = program

    if constants_optimization:
        program.program = optimize(program=program,
                                   data=data,
                                   target=target,
                                   weights=weights,
                                   constants_optimization_conf=constants_optimization_conf
                                   )
            
    pred = prog.evaluate(data=data)

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

    std_dev = np.std(result)

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
