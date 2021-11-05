import math
import operator

import numpy as np


def _protected_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return 0.


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _protected_pow(x1, x2):
    """Closure of pow for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            return math.pow(abs(x1), x2)
        except OverflowError:
            return 0.


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


OPERATOR_ADD = {"func": operator.add,
                "arity": 2, "format_str": "({} + {})"}
OPERATOR_SUB = {"func": operator.sub,
                "arity": 2, "format_str": "({} - {})"}

OPERATOR_MUL = {"func": operator.mul,
                "arity": 2, "format_str": "({} * {})"}
OPERATOR_DIV = {"func": _protected_division,
                "arity": 2, "format_str": "({} / {})"}

OPERATOR_INV = {"func": _protected_inverse,
                "arity": 1, "format_str": "(1 / {})"}

OPERATOR_NEG = {"func": operator.neg, "arity": 1, "format_str": "-({})"}

OPERATOR_ABS = {"func": operator.abs, "arity": 1, "format_str": "abs({})"}

OPERATOR_MOD = {"func": operator.mod, "arity": 2, "format_str": "{} mod {}"}

OPERATOR_LOG = {"func": _protected_log,
                "arity": 1, "format_str": "log(abs({}))"}

OPERATOR_EXP = {"func": _protected_exp, "arity": 1, "format_str": "exp({})"}

OPERATOR_POW = {"func": _protected_pow, "arity": 2, "format_str": "({} ** {})"}

OPERATOR_SQRT = {"func": _protected_sqrt, "arity": 1, "format_str": "sqrt({})"}
