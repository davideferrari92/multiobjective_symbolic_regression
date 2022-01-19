import copy
import logging
import time
from typing import Union

import pandas as pd
import numpy as np
from joblib.parallel import Parallel, delayed

from symbolic_regression.multiobjective.training import (
    create_pareto_front,
    crowding_distance,
    generate_population,
    get_offspring,
)
from symbolic_regression.simplification import simplify_population


class SymbolicRegressor:
    def __init__(
        self,
        const_range: tuple,
        constants_optimization: bool,
        constants_optimization_conf: dict,
        objective_functions: callable,
        parsimony=0.9,
        parsimony_decay=0.9,
        population_size: int = 100,
        simplification_frequency: int = 0,
        tournament_size: int = 10,
    ) -> None:

        self.population_size = population_size
        self.population = None
        self.best_program = None
        self.best_programs_history = []
        self.fitness_history = {}
        self.converged_generation = None
        self.generation = None
        self.training_duration = None
        self.status = "Uninitialized"

        self.tournament_size = tournament_size
        self.simplification_frequency = simplification_frequency
        self.objective_functions = objective_functions
        self.constants_optimization = constants_optimization
        self.constants_optimization_conf = constants_optimization_conf
        self.const_range = const_range
        self.parsimony = parsimony
        self.parsimony_decay = parsimony_decay

        self.average_complexity = None

    def drop_duplicates(self, inplace: bool = False) -> list:

        for index, p in enumerate(self.population):
            if p.is_valid and not p._is_duplicated:
                for p_confront in self.population[index + 1 :]:
                    if p.is_duplicate(p_confront):
                        p_confront._is_duplicated = True  # Makes p.is_valid = False

        if inplace:
            self.population = list(
                filter(lambda p: p._is_duplicated == False, self.population)
            )
            return self.population

        return list(filter(lambda p: p._is_duplicated == False, self.population))

    def drop_invalids(self, inplace: bool = False) -> list:
        if inplace:
            self.population = list(
                filter(lambda p: p.is_valid == True, self.population)
            )
            return self.population

        return list(filter(lambda p: p.is_valid == True, self.population))

    def fit(
        self,
        data: Union[dict, pd.Series, pd.DataFrame],
        features: list,
        target: str,
        weights: str,
        generations: int,
        genetic_operators_frequency: dict,
        operations: list,
        n_jobs: int = -1,
        stop_at_convergence: bool = True,
        verbose: int = 0
    ):
        """This method support a KeyboardInterruption of the fit process

        This allow to interrupt the training at any point without losing
        the progress made.
        """
        start = time.perf_counter()
        try:
            self._fit(
                data=data,
                features=features,
                target=target,
                weights=weights,
                generations=generations,
                genetic_operators_frequency=genetic_operators_frequency,
                operations=operations,
                n_jobs=n_jobs,
                stop_at_convergence=stop_at_convergence,
                verbose=verbose
            )
        except KeyboardInterrupt:
            stop = time.perf_counter()
            self.training_duration = stop - start
            self.status = "Interrupted by KeyboardInterrupt"
            logging.warning(f"Training terminated by a KeyboardInterrupt")
            return
        stop = time.perf_counter()
        self.training_duration = stop - start

    def _fit(
        self,
        data: Union[dict, pd.Series, pd.DataFrame],
        features: list,
        target: str,
        weights: str,
        generations: int,
        genetic_operators_frequency: dict,
        operations: list,
        n_jobs: int = -1,
        stop_at_convergence: bool = True,
        verbose: int = 0
    ) -> list:

        if not self.population:
            logging.info(f"Initializing population")
            self.status = "Generating population"
            self.population = Parallel(n_jobs=n_jobs)(
                delayed(generate_population)(
                    data=data,
                    features=features,
                    target=target,
                    weights=weights,
                    const_range=self.const_range,
                    operations=operations,
                    fitness=self.objective_functions,
                    constants_optimization=self.constants_optimization,
                    constants_optimization_conf=self.constants_optimization_conf,
                    parsimony=self.parsimony,
                    parsimony_decay=self.parsimony_decay,
                )
                for _ in range(self.population_size)
            )
        else:
            logging.info("Fitting with existing population")

        if not self.generation:
            self.generation = 0

        while True:
            self.generation += 1

            start_time_generation = time.perf_counter()

            print("##############################################")
            print("##############################################")
            logging.info(f"Generation {self.generation}/{generations}")

            logging.debug(f"Generating offspring")
            self.status = "Generating offspring"
            offsprings = Parallel(n_jobs=n_jobs, prefer="processes")(
                delayed(get_offspring)(
                    self.population,
                    data,
                    target,
                    weights,
                    self.objective_functions,
                    self.generation,
                    self.tournament_size,
                    genetic_operators_frequency,
                )
                for _ in range(self.population_size)
            )

            self.population += offsprings

            if (
                self.simplification_frequency > 0
                and self.generation % self.simplification_frequency == 0
            ):
                logging.info(f"Simplifying population")
                self.status = "Simplifying population"
                self.population = simplify_population(
                    population=self.population,
                    fitness=self.objective_functions,
                    data=data,
                    target=target,
                    weights=weights,
                    n_jobs=n_jobs,
                )

            # Removes all non valid programs in the population
            logging.debug(f"Removing duplicates")
            before_cleaning = len(self.population)

            self.drop_duplicates(inplace=True)

            after_drop_duplicates = len(self.population)
            logging.debug(
                f"{before_cleaning-after_drop_duplicates}/{before_cleaning} duplicates programs removed"
            )

            self.drop_invalids(inplace=True)

            after_cleaning = len(self.population)
            if before_cleaning != after_cleaning:
                logging.debug(
                    f"{after_drop_duplicates-after_cleaning}/{after_drop_duplicates} invalid programs removed"
                )

            # Integrate population in case of too many invalid programs
            if len(self.population) < self.population_size:
                self.status = "Refilling population"
                missing_elements = self.population_size - len(self.population)

                logging.warning(
                    f"Population of {len(self.population)} elements is less than population_size:{self.population_size}. Integrating with {missing_elements} new elements"
                )

                self.population += Parallel(
                    n_jobs=-1, prefer="processes", batch_size=28
                )(
                    delayed(generate_population)(
                        data=data,
                        features=features,
                        target=target,
                        weights=weights,
                        const_range=self.const_range,
                        operations=operations,
                        fitness=self.objective_functions,
                        constants_optimization=self.constants_optimization,
                        constants_optimization_conf=self.constants_optimization_conf,
                        parsimony=self.parsimony,
                        parsimony_decay=self.parsimony_decay,
                    )
                    for _ in range(missing_elements)
                )

            logging.debug(f"Creating pareto front")
            self.status = "Creating pareto front"
            create_pareto_front(self.population)

            logging.debug(f"Creating crowding distance")
            self.status = "Creating crowding distance"
            crowding_distance(self.population)

            self.population.sort(reverse=False)
            self.population = self.population[: self.population_size]

            self.best_program = self.population[0]
            self.best_programs_history.append(self.best_program)

            self.average_complexity = np.mean([p.complexity for p in self.population])

            if verbose > 0:
                print()
                print(
                    f"Population of {len(self.population)} elements and average complexity of {self.average_complexity}"
                )
                print(
                    f"\tBest individual (complexity {self.population[0].complexity})\n\t{self.best_program.program}"
                )
                print()
                print(f"\twith fitness\n1)\t{self.population[0].fitness}")
                print()
            if verbose > 1:
                print(f"Following best fitness")
                print(f"2)\t{self.population[1].fitness}")
                print(f"3)\t{self.population[2].fitness}")
                print(f"4)\t{self.population[3].fitness}")
                print(f"5)\t{self.population[4].fitness}")
                print()

            end_time_generation = time.perf_counter()
            logging.debug(
                f"Generation {self.generation} completed in {round(end_time_generation-start_time_generation, 1)} seconds"
            )

            for fitness, value in self.best_program.fitness.items():
                if not self.fitness_history.get(fitness):
                    self.fitness_history[fitness] = list()
                self.fitness_history[fitness].append(value)

            if self.best_program.converged:
                if not self.converged_generation:
                    self.converged_generation = self.generation
                logging.info(
                    f"Training converged after {self.converged_generation} generations."
                )
                if stop_at_convergence:
                    self.status = "Terminated: converged"
                    return

            # Use generations = -1 to rely only on convergence (risk of infinite loop)
            if generations > 0 and self.generation == generations:
                logging.info(f"Training terminated after {self.generation} generations")
                self.status = "Terminated: generations completed"
                return

    def save_model(self, file: str):
        import pickle

        with open(file, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, file: str):
        import pickle

        with open(file, "rb") as f:
            return pickle.load(f)
