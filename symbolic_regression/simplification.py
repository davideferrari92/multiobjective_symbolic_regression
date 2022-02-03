import logging
import warnings
from typing import Union

import pandas as pd
import sympy
from joblib import Parallel, delayed

from symbolic_regression.Node import FeatureNode, InvalidNode, OperationNode
from symbolic_regression.operators import *



def extract_operation(element, father=None):
    """ Extract a single operation from the sympy.simplify output
    It is meant to be used recursively.
    """

    ''' Two terminals: constants and features
    '''
    new_feature = InvalidNode()

    if element.is_Float or element.is_Integer or element.is_Rational:
        new_feature = FeatureNode(
            feature=round(float(list(element.expr_free_symbols)[0]), 2),
            is_constant=True
        )

    elif element.is_Symbol:
        new_feature = FeatureNode(
            feature=str(list(element.expr_free_symbols)[0]),
            is_constant=False
        )

    elif element == sympy.simplify('E'):
        new_feature = FeatureNode(
            feature=np.exp(1.),
            is_constant=True
        )

    if isinstance(new_feature, FeatureNode):
        return new_feature

    if str(element.func) == 'exp':
        current_operation = OPERATOR_EXP

    elif str(element.func) == 'log':
        current_operation = OPERATOR_LOG

    elif str(element.func) == 'abs':
        current_operation = OPERATOR_ABS

    elif element.is_Pow:
        current_operation = OPERATOR_POW

    elif element.is_Add:
        current_operation = OPERATOR_ADD

    elif element.is_Mul:
        current_operation = OPERATOR_MUL

    else:
        #logging.warning(f'Non valid object during simplification {element}')
        return InvalidNode()


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
        n_op = extract_operation(element=args.pop(), father=element)
        new_operation.add_operand(n_op)

        # args now has one element removed and need to be overwritten to converge the recursion.
        element._args = tuple(args)

        # Right child will be again the same element (same operation and one less of the args)
        # until n_args == arity.
        n_op = extract_operation(element=element, father=element)
        new_operation.add_operand(n_op)
    else:
        # When n_args == arity, just loop on the remaining args and add as terminal children
        for op in args:
            n_op = extract_operation(element=op, father=new_operation)
            new_operation.add_operand(n_op)

    return new_operation 

'''
from symbolic_regression.Program import Program
def simplify_program(program: Program) -> Program:
    """

    """
    try:
        logging.debug(f'Simplifying program {program}')
        simplified = sympy.simplify(program.program, rational=True, inverse=True)

        logging.debug(f'Extracting the program tree from the simplified')
        
        extracted_program = extract_operation(
            element=simplified, father=None)

        new_program = Program(
            operations=program.operations,
            features=program.features,
            const_range=program.const_range,
            program=extracted_program,
            constants_optimization=program.constants_optimization,
            constants_optimization_conf=program.constants_optimization_conf,
            parsimony=program.parsimony, parsimony_decay=program.parsimony_decay
        )

        new_program.fitness = program.fitness

        logging.debug(f'Simplified program {new_program}')

        return new_program

    except UnboundLocalError:
        return program
    except sympy.core.sympify.SympifyError:
        return program

def simplify_population(population: list,
                        fitness: list,
                        data: Union[dict, pd.Series, pd.DataFrame],
                        target: str,
                        weights: str = None,
                        n_jobs: int = -1) -> list:
    """ This function implement a parallelization of the program simplification

    Args:
        population: The population of programs to simplify
        fitness: The fitness function to evaluate after the semplification
        data: The data on which to evaluate the fitness after the simplification
        target: The target variable for the fitness function
        weights: The weights to use for weighted fitness functions
        n_jobs: The number of threads to allocate
    """

    logging.info(f'Simplifying population with {n_jobs} threads')

    def simplify_single_p(p, fitness, data, target, weights):
        warnings.filterwarnings("ignore")

        try:
            simp = simplify_program(p)
        except Exception:
            return p

        simp.evaluate_fitness(
            fitness=fitness,
            data=data,
            target=target,
            weights=weights,
        )

        return simp

    import time
    start_time = time.perf_counter()

    population = Parallel(n_jobs=n_jobs)(
        delayed(simplify_single_p)(
            program, fitness, data, target, weights
        ) for program in population
    )

    end_time = time.perf_counter()

    logging.info(f'Simplified in {round(end_time-start_time, 2)} seconds')

    return population
'''