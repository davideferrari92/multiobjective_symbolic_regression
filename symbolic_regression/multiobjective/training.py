import logging
import random
from typing import Union

import pandas as pd
from symbolic_regression.Program import Program


def generate_population(
    data: Union[dict, pd.Series, pd.DataFrame],
    features: list,
    operations: list,
    parsimony: float,
    parsimony_decay: float,
    fitness: dict,
    const_range: tuple,
):
    """ This method generate a new program and evaluate its fitness

    The program generation is an iterative process that can be parallelized.
    This function can therefore be called iteratively or parallely easily
    as there are no shared resources. Moreover, the evaluation of the fitness
    in this stage is convenient as it can be embedded in the parallel execution.

    Args:
        features: The features of the training dataset for the generation of FeatureNodes
        operations: The allowed operation for the generation of the OperationNodes
        parsimony: The parsimony that modulate the depth of the program
        parsimony_decay: The decay ration to which the parsimony is decreased as the program depth increases
        const_range: The numeric range between it is accepted to generate the constants in the program
        fitness: The dict with the fitness functions
        data: The data on to which evaluate the fitness
    """
    p = Program(features=features,
                operations=operations,
                const_range=const_range,
                parsimony=parsimony,
                parsimony_decay=parsimony_decay)

    p.init_program()

    p.evaluate_fitness(fitness=fitness, data=data)

    return p


def dominance(program1: Program, program2: Program) -> bool:
    """
    Return True if program1 dominate over program2.
    It dominates if all the fitnesses are equal or better and at least one fitness is
    better
    """

    # How many element in the p1.fitness are less than p2.fitness
    at_least_one_less_than_zero = False
    all_less_or_eq_than_zero = True

    if program1.program and program2.program:
        for this_fitness in program1.fitness.keys():
            # Ignore the fitness which are not to be optimized
            if program1.is_fitness_to_minimize[this_fitness] == False:
                continue

            d = abs(program1.fitness[this_fitness]) - \
                abs(program2.fitness[this_fitness])

            if d < 0:
                at_least_one_less_than_zero = True
            if d > 0:
                all_less_or_eq_than_zero = False

        return at_least_one_less_than_zero and all_less_or_eq_than_zero

    return False


def create_pareto_front(population: list):
    """
    """

    pareto_front = []

    # Loop over the entire matrix, can be optimised to do only the triangular matrix
    for p1 in population:
        p1.rank = float('inf')

        if not p1.is_valid:
            continue
        p1.programs_dominates = []
        p1.programs_dominated_by = []

        for p2 in population:
            if p1 == p2 or not p2.is_valid:
                continue

            if dominance(p1, p2):
                p1.programs_dominates.append(p2)
            elif dominance(p2, p1):
                p1.programs_dominated_by.append(p2)

        if len(p1.programs_dominated_by) == 0:
            p1.rank = 1
            pareto_front.append(p1)

    i = 1

    # Set the belonging pareto front to every element of the population

    while pareto_front:
        next_pareto_front = []

        for p1 in pareto_front:
            if not p1.is_valid:
                continue
            for p2 in p1.programs_dominates:
                if not p2.is_valid:
                    continue
                p2.programs_dominated_by.remove(p1)

                if len(p2.programs_dominated_by) == 0:
                    p2.rank = i + 1
                    next_pareto_front.append(p2)

        i += 1

        pareto_front = next_pareto_front


def extract_pareto_front(population: list, rank: int):
    pareto_front = []
    for p in population:
        if p and p.rank == rank:
            pareto_front.append(p)

    return pareto_front


def crowding_distance(population: list):

    objectives = population[0].fitness.keys()

    rank_iter = 1
    pareto_front = extract_pareto_front(population=population, rank=rank_iter)

    while pareto_front:  # Exits when extract_pareto_front return an empty list
        for obj in objectives:
            # This exclude the fitness functions which are set not to be minimized
            if population[0].is_fitness_to_minimize[obj] == False:
                continue

            # Highest fitness first for each objective
            pareto_front.sort(key=lambda p: p.fitness[obj], reverse=True)

            norm = pareto_front[0].fitness[obj] - \
                pareto_front[-1].fitness[obj] + 1e-20

            for index, program in enumerate(pareto_front):
                if index == 0 or index == len(pareto_front) - 1:
                    program.crowding_distance = float('inf')
                else:
                    delta = pareto_front[index - 1].fitness[obj] - \
                        pareto_front[index + 1].fitness[obj]

                    program.crowding_distance = delta / norm

        rank_iter += 1
        pareto_front = extract_pareto_front(population=population,
                                            rank=rank_iter)


def tournament_selection(population: list, tournament_size: int,
                         generation: int):
    """ The tournament selection is used to choose the programs to which apply genetic operations

    Firstly a random selection of k elements from the population is selected and 
    the best program among them is chosen.

    Args:
        population: the population from which to select the program
        tournament_size: the number of programs from which to choose the selcted one
        generation: keeps track of the training generation
    """

    tournament_members = random.choices(population, k=tournament_size)

    best_member = tournament_members[0]

    for member in tournament_members:
        if member is None or not member.is_valid:
            continue

        if generation == 0:
            try:
                if best_member > member:
                    best_member = member
            except IndexError:
                pass  # TODO fix

        else:

            # In the other generations use the pareto front rank and the crowding distance
            if best_member == None or \
                    member.rank < best_member.rank or \
                    (member.rank == best_member.rank and
                        member.crowding_distance > best_member.crowding_distance):
                best_member = member

    return best_member


def get_offspring(population: list, data: pd.DataFrame, fitness: list,
                  generations: int, tournament_size: int,
                  genetic_operations_frequency: dict):
    """ This function generate an offspring of a program from the current population

    The offspring is a program to which a genetic alteration has been applied.
    The possible operations are as follow:
        - crossover: a random subtree from another program replaces
            a random subtree of the current program
        - mutation: a random subtree of the current program is replaced by a newly
            generated subtree
        - randomization: it is a crossover with a portion of the same program instead of a portion
            of another program
        - deletion: a random subtree is deleted from the current program
        - insertion: a newly generated subtree is inserted in a random spot of the current program
        - operator mutation: a random operation is replaced by another with the same arity
        - leaf mutation: a terminal node (feature or constant) is replaced by a different one
        - simplify: uses a sympy backend to simplify the program ad reduce its complexity
        - do nothing: in this case no mutation is applied

    The frequency of which those operation are applied is determined by the dictionary
    genetic_operations_frequency in which the relative frequency of each of the desired operations
    is expressed with integers. The higher the number the likelier the operation will be chosen.

    The program to which apply the operation is chosen using the tournament_selection, a method
    that identify the best program among a random selection of k programs from the population.

    Args:
        population: The population of programs from which to extract the program for the mutation
        data: The data on which to evaluate the fitness of the mutated program
        fitness: The list of fitness functions for this task
        tournament_size: The size of the pool of random programs from which to choose in for the mutations
        genetic_operations_frequency: The relative frequency with which to coose the genetic operation to apply
        generations: The number of training generations (used to appropriately behave in the first one)
    """

    # This allow to randomly chose a genetic operation to
    ops = list()
    for op, freq in genetic_operations_frequency.items():
        ops += [op] * freq
    gen_op = random.choice(ops)

    program1 = tournament_selection(population=population,
                                    tournament_size=tournament_size,
                                    generation=generations)

    if program1 is None or not program1.is_valid:
        return program1

    if gen_op == 'crossover':
        program2 = tournament_selection(population=population,
                                        tournament_size=tournament_size,
                                        generation=generations)
        if program2 is None or not program2.is_valid:
            return program1
        p_ret = program1.cross_over(other=program2, inplace=False)

    elif gen_op == 'randomize':
        # Will generate a new tree as other
        p_ret = program1.cross_over(other=None, inplace=False)

    elif gen_op == 'mutation':
        p_ret = program1.mutate(inplace=False)

    elif gen_op == 'delete_node':
        p_ret = program1.delete_node(inplace=False)

    elif gen_op == 'insert_node':
        p_ret = program1.insert_node(inplace=False)

    elif gen_op == 'mutate_operator':
        p_ret = program1.mutate_operator(inplace=False)

    elif gen_op == 'mutate_leaf':
        p_ret = program1.mutate_leaf(inplace=False)

    elif gen_op == 'simplification':
        p_ret = program1.simplify(inplace=False)

    elif gen_op == 'recalibrate':
        p_ret = program1.recalibrate(inplace=False)

    elif gen_op == 'do_nothing':
        p_ret = program1
    else:
        logging.warning(
            f'Supported genetic operations: crossover, delete_node, do_nothing, insert_node, mutate_leaf, mutate_operator, simplification, mutation and randomize'
        )
        return program1

    # Add the fitness to the object after the cross_over or mutation
    p_ret.evaluate_fitness(fitness=fitness, data=data)

    return p_ret
