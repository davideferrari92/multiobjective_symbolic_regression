import sympy

from symbolic_regression.Node import FeatureNode, InvalidNode, Node, OperationNode
from symbolic_regression.operators import *


def extract_operation(element_to_extract, father: Node = None) -> Node:
    """ Extract the operation from the sympy expression and return a tree of OperationNode and FeatureNode.

    Args:
        - element_to_extract: 
            The sympy expression to extract the operation from.
        - father:
            The father of the node to be created.

    Returns:
        - Node:
            The root of the tree of OperationNode and FeatureNode.
    """

    element = element_to_extract

    current_operation = None

    """ Here we allow the use of the following operators:
        - power: ** or ^ (x**2 or x^2)
        - addition: + (x + y)
        - multiplication: * (x * y)
        - division: / (x / y)
        - exponent: exp(x)
        - logarithm: log(x)
        - absolute value: abs(x)
        - minimum: Min(x, y)
        - maximum: Max(x, y)
    """
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

    elif str(element.func) == 'Min':
        current_operation = OPERATOR_MIN

    elif str(element.func) == 'Max':
        current_operation = OPERATOR_MAX

    if current_operation:
        """ Case in which the element is an operation.
        If the element is an operation, we extract the arguments and we create a new OperationNode.
        Otherwise we create a FeatureNode.
        """
        args = list(element._args)

        # 1/x is treated as pow(x, -1) which is more unstable.
        # We convert it to an actual 1/x
        if element.is_Pow and len(args) == 2 and isinstance(args[1], sympy.core.numbers.NegativeOne):
            current_operation = OPERATOR_DIV
            args[0], args[1] = sympy.parse_expr('1'), args[0]

        # sqrt(x) is treated as pow(x, .5) which is more unstable.
        # We convert it to an actual sqrt(x)
        if element.is_Pow and len(args) == 2 and args[1] == sympy.parse_expr('1/2'):            
            current_operation = OPERATOR_SQRT
            args = [args[0]]

        new_operation = OperationNode(
            operation=current_operation['func'],
            arity=current_operation['arity'],
            format_str=current_operation['format_str'],
            format_tf=current_operation['format_tf'],
            symbol=current_operation['symbol'],
            format_diff=current_operation.get('format_diff', current_operation['format_str']),
            father=father
        )
        
        n_args = len(args)
        if n_args > current_operation['arity']:

            ''' Case in which commutative operation are presented with more than arity operators.
            We receive an operation with n_args > arity and we need to generate a subtree of the same operation so to have an equivalent
            binary tree.
            For example, if we have an addition with 3 operands, we need to generate a tree like this:
                +
                / \
                +   z
                / \
                x   y
            
            We do this by popping the first element of the args and adding it as a left child of the new_operation.
            Then we overwrite the args of the element to extract with the remaining args and we call the function again.
            This will generate the right child of the new_operation.
            '''

            # Left child will be one of the arity+n operands
            left_child = args.pop(0)
            n_op = extract_operation(element_to_extract=left_child, father=new_operation)
            new_operation.add_operand(n_op)

            # args now has one element removed and need to be overwritten to converge the recursion.
            element._args = tuple(args)

            # Right child will be again the same element (same operation and one less of the args)
            # until n_args == arity.
            n_op = extract_operation(element_to_extract=element, father=new_operation)
            new_operation.add_operand(n_op)
        else:
            # When n_args == arity, just loop on the remaining args and add as terminal children
            for op in args:
                n_op = extract_operation(element_to_extract=op, father=new_operation)
                new_operation.add_operand(n_op)

        return new_operation

    else:
        """ Case in which the element is a feature.

        If the element is a feature, we create a new FeatureNode.
        Otherwise we create an InvalidNode.
        """
        new_feature = None

        if isinstance(element, sympy.core.symbol.Symbol):
            new_feature = FeatureNode(feature=str(element), is_constant=False, father=father)
        
        if isinstance(element, sympy.core.numbers.Float) or isinstance(element, sympy.core.numbers.Integer) or isinstance(element, sympy.core.numbers.Rational) or isinstance(element, sympy.core.numbers.NegativeOne):
            new_feature = FeatureNode(feature=float(element), is_constant=True, father=father)

        if element == sympy.simplify('E'):
            new_feature = FeatureNode(feature=np.exp(1.), is_constant=True, father=father)

        return new_feature if new_feature else InvalidNode()