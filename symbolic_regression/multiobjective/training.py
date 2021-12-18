import logging
import random
from typing import Union

import pandas as pd
from symbolic_regression.Program import Program


def generate_population(
    data: Union[dict, pd.Series, pd.DataFrame],
    features: list,
    target: str,
    weights: str,
    operations: list,
    parsimony: float,
    parsimony_decay: float,
    fitness: list,
    const_range: tuple,
    constants_optimization: bool = False,
    constants_optimization_conf: dict = {}
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
        fitness: The list of the fitness functions
        data: The data on to which evaluate the fitness
        target: The label of the target column for supervised tasks
        weights: The label of the weights columns of a weighted WMSE in case of unbalanced datasets
    """
    p = Program(
        features=features,
        operations=operations,
        const_range=const_range,
        constants_optimization=constants_optimization,
        constants_optimization_conf=constants_optimization_conf
    )

    p.init_program(parsimony=parsimony, parsimony_decay=parsimony_decay)

    p.evaluate_fitness(fitness=fitness,
                       data=data, target=target, weights=weights)

    return p


def dominance(program1: Program, program2: Program) -> bool:
    """
    Return True if program1 dominate over program2.
    It dominates if all the fitnesses are equal or better and at least one fitness is
    better
    """

    # How many element in the p1.fitness are less than p2.fitness
    differences = [p1f - p2f for p1f,
                   p2f in zip(program1.fitness, program2.fitness)]

    # If all the elements of p1fitness are less than p2f
    at_least_one_less_than_zero = False
    all_less_or_eq_than_zero = True

    for d in differences:
        if d < 0:
            at_least_one_less_than_zero = True
        if d > 0:
            all_less_or_eq_than_zero = False

    return at_least_one_less_than_zero and all_less_or_eq_than_zero


def create_pareto_front(population: list):
    """
    """

    pareto_front = []

    # Loop over the entire matrix, can be optimised to do only the triangular matrix
    for p1 in population:
        p1.programs_dominates = []
        p1.programs_dominated_by = []

        for p2 in population:
            if p1 == p2:
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
            for p2 in p1.programs_dominates:
                p2.programs_dominated_by.remove(p1)

                if len(p2.programs_dominated_by) == 0:
                    p2.rank = i + 1
                    next_pareto_front.append(p2)

        i += 1
        logging.debug(f'Pareto Front: entering rank {i}')
        pareto_front = next_pareto_front


def extract_pareto_front(population: list, rank: int):
    pareto_front = []
    for p in population:
        if p.rank == rank:
            pareto_front.append(p)

    return pareto_front


def crowding_distance(population: list):

    objectives = len(population[0].fitness)

    rank_iter = 1
    pareto_front = extract_pareto_front(population=population, rank=rank_iter)

    while pareto_front:  # Exits when extract_pareto_front return an empty list
        for i in range(objectives):
            # Highest fitness first for each objective
            pareto_front.sort(key=lambda p: p.fitness[i], reverse=True)

            norm = pareto_front[0].fitness[i] - \
                pareto_front[-1].fitness[i] + 1e-20

            for index, program in enumerate(pareto_front):
                if index == 0 or index == len(pareto_front) - 1:
                    program.crowding_distance = float('inf')
                else:
                    delta = pareto_front[index - 1].fitness[i] - \
                        pareto_front[index + 1].fitness[i]

                    program.crowding_distance = delta / norm

        rank_iter += 1
        pareto_front = extract_pareto_front(
            population=population, rank=rank_iter)


def tournament_selection(population: list,
                         tournament_size: int,
                         generation: int):
    """
    """

    torunament_members = random.choices(population, k=tournament_size)

    best_member = None

    for member in torunament_members:
        if generation == 0:
            # The first generation compare only the fitness

            if best_member is None or best_member.fitness[0] > member.fitness[0]:
                best_member = member
        else:

            # In the other generations use the pareto front rank and the crowding distance
            if best_member == None or \
                    member.rank < best_member.rank or \
                    (member.rank == best_member.rank and
                        member.crowding_distance > best_member.crowding_distance):
                best_member = member

    return best_member


def get_offspring(population: list,
                  data: pd.DataFrame,
                  target: str,
                  weights: str,
                  fitness: list,
                  generations: int,
                  tournament_size: int,
                  cross_over_perc: float = .5):
    """ This function generate an offspring of a program from the current population

    The offspring is a mutation of a program from the current population by means of
    cross-over or mutation. The choice of the two is random according to the genetic
    nature of this algorithm. The prevalence of one over the other can be modulated
    using the parameter `cross_over_perc` (the higher the likely the cross-over will
    be chosen).
    In case the cross-over is chosen, a second program is extracted from the population
    and the operation is performed selecting a sub-tree from the program 2 and appending
    in place of a random sub-tree of program 1.
    In case the mutation is chosen, a random sub-tree of the program 1 is replaces by a 
    newly generated subtree.
    The generated mutated program will then be returned as a new object for the population.

    Args:
        population: The population of programs from which to extract the program for the mutation
        data: The data on which to evaluate the fitness of the mutated program
        target: The label of the target variable in the training dataset for supervised tasks
        weights: The label of the weights columns of a weighted WMSE in case of unbalanced datasets
        fitness: The list of fitness functions for this task
        tournament_size: The size of the pool of random programs from which to choose in for the mutations
        cross_over_perc: The value that modulates the prevalence of cross-over over simple mutations
        generations: The number of training generations (used to appropriately behave in the first one)
    """
    program1 = tournament_selection(
        population=population, tournament_size=tournament_size, generation=generations
    )

    if random.random() < cross_over_perc:
        program2 = tournament_selection(
            population=population, tournament_size=tournament_size, generation=generations
        )
        p_ret = program1.cross_over(other=program2, inplace=False)

    else:
        p_ret = program1.mutate(inplace=False)

    # Add the fitness to the object after the cross_over or mutation
    p_ret.evaluate_fitness(
        fitness=fitness, data=data, target=target, weights=weights)

    return p_ret
