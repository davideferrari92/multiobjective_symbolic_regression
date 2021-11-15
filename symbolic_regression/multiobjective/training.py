import logging
import random

import pandas as pd
from symbolic_regression.Program import Program


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

    ''' Set the belonging pareto front to every element of the population
    '''
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
                         iterations: int):
    """
    """

    torunament_members = random.choices(population, k=tournament_size)

    best_member = None

    for member in torunament_members:
        if iterations == 0:
            # The first round compare only wmse, then also the multiobj funcs

            if not best_member or best_member.fitness[0] > member.fitness[0]:
                best_member = member
        else:

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
                  fitness,
                  tournament_size: int,
                  cross_over_perc: float = .7,
                  iterations: int = 0):

    program1 = tournament_selection(
        population=population, tournament_size=tournament_size, iterations=iterations
    )

    if random.random() > cross_over_perc:
        program2 = tournament_selection(
            population=population, tournament_size=tournament_size, iterations=iterations
        )
        p_ret = program1.cross_over(other=program2, inplace=False)

    else:
        p_ret = program1.mutate(inplace=False)

    # Add the fitness to the object after the cross_over or mutation
    p_ret.fitness = fitness(program=p_ret, data=data,
                            target=target, weights=weights)
    return p_ret
