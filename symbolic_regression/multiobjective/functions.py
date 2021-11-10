from typing import Union

import numpy as np
import pandas as pd
from symbolic_regression.Program import Program


def not_constant(program: Program,
                 data: Union[dict, pd.DataFrame],
                 epsilon: float = .01) -> float:
    """

    """
    result = program.evaluate(data=data)

    # I consider only the maximum difference because if it is < epsilon
    # then every other difference is < epsilon as well
    diff = np.max(np.abs(np.diff(result)))

    return np.max([0, epsilon - diff])


def value_range(program: Program,
                data: Union[dict, pd.DataFrame],
                lower_bound: float,
                upper_bound: float) -> float:
    """
    f(x) - upper_bound <= 0
    lower_bound - f(x) <= 0
    """

    result = program.evaluate(data=data)

    upper_bound_constraint = np.sum(np.where(
        np.array(result) - upper_bound >= 0, np.array(result) - upper_bound, 0))
    lower_bound_constraint = np.sum(np.where(
        lower_bound - np.array(result) >= 0, lower_bound - np.array(result), 0))

    return upper_bound_constraint + lower_bound_constraint


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

    if method not in ['abs_val', 'inversions']:
        raise AttributeError(f'Only support abs_val or inversions. Default is abs_val')

    data_ord = data.copy(deep=True)

    ''' Index ordered by baseline target '''
    data_ord.sort_values(by=target, ascending=False, inplace=True)
    data_ord['ordering_target'] = data_ord.reset_index(inplace=False).index

    ''' Index ordered by program prediction '''
    data_ord['pred'] = program.evaluate(data=data_ord)
    
    # The number of inversions to match target ordering
    argsort_pred = len(data_ord) - 1 - np.argsort(data_ord['pred'].to_numpy())

    if method == 'inversions':
        from symbolic_regression.multiobjective.utils import merge_sort_inversions
        inversions = merge_sort_inversions(argsort_pred)

        return inversions

    elif method == 'abs_val':
        data_ord.sort_values(by='pred', ascending=False, inplace=True)
        data_ord['ordering_program'] = data_ord.reset_index(inplace=False).index

        data_ord['error'] = data_ord['pred'] - data_ord[target]
        data_ord['error_ordering'] = data_ord['ordering_program'] - data_ord['ordering_target']

        error = 0.0

        for _, row_i in data_ord.iterrows():
            for _, row_j in data_ord.iterrows():
                
                if row_j[target] < row_i[target]:
                    error += row_i['pred'] - row_j['pred']
        return error