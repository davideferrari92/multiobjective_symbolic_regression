import sympy

from symbolic_regression.Node import FeatureNode, InvalidNode, OperationNode
from symbolic_regression.operators import *


def extract_operation(element, father=None):
    """ Extract a single operation from the sympy.simplify output
    It is meant to be used recursively.
    """

    current_operation = None

    if element.is_Pow:
        current_operation = OPERATOR_POW

    elif element.is_Add:
        current_operation = OPERATOR_ADD

    elif element.is_Mul:
        current_operation = OPERATOR_MUL

    elif str(element.func) == 'exp':
        current_operation = OPERATOR_EXP

    elif str(element.func) == 'log':
        current_operation = OPERATOR_LOG

    elif str(element.func) == 'abs':
        current_operation = OPERATOR_ABS

    elif str(element.func) == 'min':
        current_operation = OPERATOR_MIN

    elif str(element.func) == 'max':
        current_operation = OPERATOR_MAX
    
    if current_operation:

        new_operation = OperationNode(
            operation=current_operation['func'],
            arity=current_operation['arity'],
            format_str=current_operation['format_str'],
            format_tf=current_operation['format_tf'],
            father=father
        )

        args = list(element.args)
        n_args = len(args)

        if n_args > current_operation['arity']:

            ''' Case in which commutative operation are presented with more than arity operators

            Generate a subtree of the same operation so to have an equivalent
            binary tree.
            '''

            # Left child will be one of the arity+n operands
            n_op = extract_operation(element=args.pop(), father=new_operation)
            new_operation.add_operand(n_op)

            # args now has one element removed and need to be overwritten to converge the recursion.
            element._args = tuple(args)

            # Right child will be again the same element (same operation and one less of the args)
            # until n_args == arity.
            n_op = extract_operation(element=element, father=new_operation)
            new_operation.add_operand(n_op)
        else:
            # When n_args == arity, just loop on the remaining args and add as terminal children
            for op in args:
                n_op = extract_operation(element=op, father=new_operation)
                new_operation.add_operand(n_op)

        return new_operation

    else:  # Feature
        new_feature = None

        allowed_numeric_types = [
            sympy.core.numbers.Float,
            sympy.core.numbers.Integer,
            sympy.core.numbers.Rational,
            sympy.core.numbers.NegativeOne
        ]
        for a in allowed_numeric_types:
            if isinstance(element, a):
                new_feature = FeatureNode(
                    feature=float(element),
                    is_constant=True,
                    father=father
                )
                break

        if isinstance(element, sympy.core.symbol.Symbol):
            new_feature = FeatureNode(
                feature=str(element),
                is_constant=False,
                father=father
            )

        elif element == sympy.simplify('E'):
            new_feature = FeatureNode(
                feature=np.exp(1.),
                is_constant=True,
                father=father
            )

        if new_feature:
            return new_feature

        return InvalidNode()
