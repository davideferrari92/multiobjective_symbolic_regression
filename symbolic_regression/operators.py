import math
import operator

import numpy as np


def _protected_exp(x):
    try:
        return np.exp(x)
    except OverflowError:
        return 0.


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 1e-4, np.divide(x1, x2), np.sign(x1)*np.sign(x2)*10**3)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x1 > 1e-4, np.log(x1), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 1e-4, 1. / x1, np.sign(x1)*10**3)

def _protected_pow(x1, x2):
    """Closure of pow for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            return np.power(x1, x2)
        except OverflowError:
            return 0.
        except ValueError:  # The math domain error
            return 0.
        except RuntimeWarning:
            return 0.


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1. / (1. + np.exp(-x1))


OPERATOR_ADD = {
    "func": operator.add,
    "format_tf": 'tf.add({}, {})',
    "arity": 2,
    "format_str": "({} + {})"
}

OPERATOR_SUB = {
    "func": operator.sub,
    "format_tf": 'tf.subtract({}, {})',
    "arity": 2,
    "format_str": "({} - {})"
}

OPERATOR_MUL = {
    "func": operator.mul,
    "format_tf": 'tf.multiply({}, {})',
    "arity": 2,
    "format_str": "({} * {})"
}

OPERATOR_DIV = {
    "func": _protected_division,
    "format_tf": 'tf.divide({}, {})',
    "arity": 2,
    "format_str": "({} / {})"
}

OPERATOR_INV = {
    "func": _protected_inverse,
    "format_tf": 'tf.pow({}, -1)',
    "arity": 1,
    "format_str": "(1 / {})"
}

OPERATOR_NEG = {
    "func": operator.neg,
    "format_tf": 'tf.negative({})',
    "arity": 1,
    "format_str": "-({})"
}

OPERATOR_ABS = {
    "func": np.abs,
    "format_tf": 'tf.abs({})',
    "arity": 1,
    "format_str": "abs({})"
}

OPERATOR_LOG = {
    "func": _protected_log,
    "format_tf": 'tf.math.log({})',
    "arity": 1,
    "format_str": "log({})"
}

OPERATOR_EXP = {
    "func": _protected_exp,
    "format_tf": 'tf.exp({})',
    "arity": 1,
    "format_str": "exp({})"
}

OPERATOR_POW = {
    "func": _protected_pow,
    "format_tf": 'tf.pow({}, {})',
    "arity": 2,
    "format_str": "({} ** {})"
}

OPERATOR_SQRT = {
    "func": _protected_sqrt,
    "format_tf": 'tf.sqrt({})',
    "arity": 1,
    "format_str": "sqrt({})"}

OPERATOR_MAX = {
    "func": np.maximum,
    "format_tf": 'tf.maximum({}, {})',
    "arity": 2,
    "format_str": "max({}, {})"
}

OPERATOR_MIN = {
    "func": np.minimum,
    "format_tf": 'tf.minimum({}, {})',
    "arity": 2,
    "format_str": "min({}, {})"
}

OPERATOR_SIGMOID = {
    "func": _sigmoid,
    "format_tf": 'tf.sigmoid({})',
    "arity": 1,
    "format_str": "sigmoid({})"
}

############################# LOGIC OPERATORS

def _logic_and(x1, x2):
    return x1 and x2

OPERATOR_LOGIC_AND = {
    "func": _logic_and,
    "format_tf": 'tf.math.logical_and({}, {})',
    "arity": 2,
    "format_str": "and({}, {})"
}

def _logic_or(x1, x2):
    return x1 or x2

OPERATOR_LOGIC_OR = {
    "func": _logic_or,
    "format_tf": 'tf.math.logical_or({}, {})',
    "arity": 2,
    "format_str": "or({}, {})"
}

def _logic_xor(x1):
    return not x1

OPERATOR_LOGIC_XOR = {
    "func": _logic_xor,
    "format_tf": 'tf.math.logical_xor({}, {})',
    "arity": 2,
    "format_str": "xor({}, {})"
}

def _logic_not(x1):
    return not x1

OPERATOR_LOGIC_NOT = {
    "func": _logic_not,
    "format_tf": 'tf.math.logical_not({})',
    "arity": 1,
    "format_str": "not({})"
}

def _equal(x1, x2):
    return x1 == x2

OPERATOR_GREATER_EQUAL_THAN = {
    "func": _equal,
    "format_tf": 'tf.Equal({}, {})',
    "arity": 2,
    "format_str": "equal({}, {})"
}

def _less_than(x1, x2):
    return x1 < x2

OPERATOR_LESS_THAN = {
    "func": _less_than,
    "format_tf": 'tf.Less({}, {})',
    "arity": 2,
    "format_str": "less({}, {})"
}

def _less_equal_than(x1, x2):
    return x1 <= x2

OPERATOR_LESS_EQUAL_THAN = {
    "func": _less_equal_than,
    "format_tf": 'tf.LessEqual({}, {})',
    "arity": 2,
    "format_str": "lessEqual({}, {})"
}

def _greater_than(x1, x2):
    return x1 <= x2

OPERATOR_GREATER_THAN = {
    "func": _greater_than,
    "format_tf": 'tf.Greater({}, {})',
    "arity": 2,
    "format_str": "less({}, {})"
}

def _greater_equal_than(x1, x2):
    return x1 <= x2

OPERATOR_GREATER_EQUAL_THAN = {
    "func": _greater_equal_than,
    "format_tf": 'tf.GreaterEqual({}, {})',
    "arity": 2,
    "format_str": "lessEqual({}, {})"
}
