import logging

import sympy

from Models.Program import Program
from Models.Node import FeatureNode, OperationNode
from Models.operators import *


def extract_operation(element, depth: int = 0, father = None):
    """ Extract a single operation from the sympy.simplify output
    It is meant to be used recursively.
    """
    
    ''' Two terminals: constants and features
    '''
    if element.is_Float or element.is_Integer or element.is_Rational:
        new_feature = FeatureNode(
            feature=float(list(element.expr_free_symbols)[0]),
            depth=depth,
            is_constant=True
        )
        return new_feature

    if element.is_Symbol:
        new_feature = FeatureNode(
            feature=str(list(element.expr_free_symbols)[0]),
            depth=depth,
            is_constant=False
        )
        return new_feature

    ''' Non terminal cases
    '''
    if str(element.func) == 'exp':
        current_operation = OPERATOR_EXP
    
    elif str(element.func) == 'log':
        current_operation = OPERATOR_LOG
    
    elif str(element.func) == 'abs':
        current_operation = OPERATOR_ABS

    elif str(element.func) == 'mod':
        current_operation = OPERATOR_MOD

    elif element.is_Pow:
        current_operation = OPERATOR_POW

    elif element.is_Add:
        current_operation = OPERATOR_ADD

    elif element.is_Mul:
        current_operation = OPERATOR_MUL

    else:
        print(element)

    new_operation = OperationNode(
            operation=current_operation['func'],
            arity=current_operation['arity'],
            format_str=current_operation['format_str'],
            depth=depth,
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
        new_operation.add_operand(
            extract_operation(element=args.pop(), depth=depth+1, father=element)
        )

        # args now has one element removed and need to be overwritten to converge the recursion.
        element._args = tuple(args)

        # Right child will be again the same element (same operation and one less of the args)
        # until n_args == arity.
        new_operation.add_operand(
            extract_operation(element=element, depth=depth+1, father=element)
        )
    else:
        # When n_args == arity, just loop on the remaining args and add as terminal children
        for op in args:
            new_operation.add_operand(
                extract_operation(element=op, depth=depth+1, father=new_operation)
            )
    
    return new_operation

def simplify_program(program: Program) -> Program:
    """

    """
    
    logging.debug(f'Simplifying program {program}')
    simplified = sympy.simplify(program.program)

    logging.debug(f'Extracting the program tree from the simplified')
    extracted_program = extract_operation(element=simplified, depth=0, father=None)
    
    new_program = Program(
        operations=program.operations,
        features=program.features,
        const_range=program.const_range,
        max_depth=program.max_depth,
        program=extracted_program
    )
    logging.debug(f'Simplified program generator')

    return new_program
