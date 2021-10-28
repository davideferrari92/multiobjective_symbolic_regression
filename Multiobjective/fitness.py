from typing import Union
from Models.Program import Program

import pandas as pd


def wmse(program: Program,
         data: Union[pd.DataFrame],
         target: str,
         weights: str) -> float:
    """ Evaluates the weighted mean squared error
    """

    pred = program.evaluate(data=data)

    wmse = (((pred - data[target]) ** 2) * data[weights]).mean()

    return wmse
