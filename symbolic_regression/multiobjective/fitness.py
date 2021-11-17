from typing import Union

import pandas as pd
from symbolic_regression.Program import Program


def wmse(program: Program,
         data: Union[pd.DataFrame, pd.Series],
         target: str,
         weights: str = None) -> float:
    """ Evaluates the weighted mean squared error
    """

    pred = program.evaluate(data=data)

    if weights:
         wmse = (((pred - data[target]) ** 2) * data[weights]).mean()
    else:
         wmse = (((pred - data[target]) ** 2)).mean()
    return wmse
