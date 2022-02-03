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

backend_parallel = 'multiprocessing'

class SymbolicRegressor:
    def __init__(
        self,
        const_range: tuple=None,
        parsimony=0.9,
        parsimony_decay=0.9,
        population_size: int = 100,
        simplification_frequency: int = 0,
        tournament_size: int = 10,
    ) -> None:

        """ This class implements the basic features for training a Symbolic Regression algorithm

        Args:
            - const_range: this is the range of values from which to generate constants in the program
            - fitness_functions: the functions to use for evaluating programs' performance
            - parsimony: the ratio to which a new operation is chosen instead of a terminal node in program generations
            - parsimony_decay: a modulation parameter to decrease the parsimony and limit program generation depth
            - simplification_frequency: how often in the training are the program simplified
            - tournament_size: this modulate the tournament selection and set the dimension of the selection
        """

        # Model characteristics
        self.best_fitness_history = []
        self.best_program = None
        self.best_programs_history = []
        self.converged_generation = None
        self.fitness_history = {}
        self.generation = None
        self.population = None
        self.population_size = population_size
        self.status = "Uninitialized"
        self.training_duration = None

        # Training configurations
        self.const_range = const_range
        self.parsimony = parsimony
        self.parsimony_decay = parsimony_decay
        self.simplification_frequency = simplification_frequency
        self.tournament_size = tournament_size

        # Population characteristics
        self.average_complexity = None

    def drop_duplicates(self, inplace: bool = False) -> list:
        """ This method removes duplicated programs

        Programs are considered duplicated if they have the same performance

        Args:
            - inplace: allow to overwrite the current population or duplicate the object
        """

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
        """ This program removes invalid programs from the population

        A program can be invalid when mathematical operation are not possible
        or if the siplification generated operation which are not supported.

        Args:
            - inplace: allow to overwrite the current population or duplicate the object
        """
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
        fitness_functions: dict,
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
                fitness_functions=fitness_functions,
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
        fitness_functions: dict,
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
            self.population = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
                delayed(generate_population)(
                    data=data,
                    features=features,
                    const_range=self.const_range,
                    operations=operations,
                    fitness=fitness_functions,
                    parsimony=self.parsimony,
                    parsimony_decay=self.parsimony_decay,
                )
                for _ in range(self.population_size)
            )
        else:
            logging.info("Fitting with existing population")

        if not self.generation:
            self.generation = 0

        start_time = time.perf_counter()
        while True:
            self.generation += 1

            start_time_generation = time.perf_counter()
            converged_time = None

            print("#################################################################")
            print("#################################################################")
            seconds = round(time.perf_counter()-start_time)
            
            if self.generation > 1:
                seconds_iter = round(seconds/(self.generation-1), 1)
                print(f"Generation {self.generation}/{generations} ({seconds} sec, {seconds_iter} sec/generation)")
            else:
                print(f"Generation {self.generation}/{generations} ({seconds} sec)")

            logging.debug(f"Generating offspring")
            self.status = "Generating offspring"
            offsprings = Parallel(n_jobs=n_jobs, backend=backend_parallel)(
                delayed(get_offspring)(
                    self.population,
                    data,
                    fitness_functions,
                    self.generation,
                    self.tournament_size,
                    genetic_operators_frequency,
                )
                for _ in range(self.population_size)
            )

            self.population += offsprings
            
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
            if len(self.population) < self.population_size*2:
                self.status = "Refilling population"
                missing_elements = 2*self.population_size - len(self.population)

                logging.warning(
                    f"Population of {len(self.population)} elements is less than 2*population_size:{self.population_size*2}. Integrating with {missing_elements} new elements"
                )

                self.population += Parallel(
                    n_jobs=-1, batch_size=28, backend=backend_parallel
                )(
                    delayed(generate_population)(
                        data=data,
                        features=features,
                        const_range=self.const_range,
                        operations=operations,
                        fitness=fitness_functions,
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
            self.best_fitness_history.append(self.best_program.fitness)

            self.average_complexity = np.mean([p.complexity for p in self.population])

            if verbose > 0:
                print()
                print(
                    f"Population of {len(self.population)} elements and average complexity of {self.average_complexity}\n"
                )
                print(
                    f"\tBest individual (complexity {self.population[0].complexity})\n\t{self.best_program.program}"
                )
                print()
                print(f"\twith fitness\n1)\t{self.population[0].fitness}")
                print()
            if verbose > 1:
                try:
                    print(f"Following best fitness")
                    print(f"2)\t{self.population[1].fitness}")
                    print(f"3)\t{self.population[2].fitness}")
                    print(f"4)\t{self.population[3].fitness}")
                    print(f"5)\t{self.population[4].fitness}")
                    print('...\t...\n')
                    
                except IndexError:
                    pass  # Stops printing in very small populations

            end_time_generation = time.perf_counter()
            logging.debug(
                f"Generation {self.generation} completed in {round(end_time_generation-start_time_generation, 1)} seconds"
            )

            if self.best_program.converged:
                converged_time = time.perf_counter()
                if not self.converged_generation:
                    self.converged_generation = self.generation
                logging.info(
                    f"Training converged after {self.converged_generation} generations. ({round(converged_time-start_time)} seconds)"
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

    @property
    def summary(self):
        istances = []

        for index, p in enumerate(self.population):
            row = {}
            row['rank'] = index + 1
            row['program'] = p.program

            for f_k, f_v in p.fitness:
                row[f_k] = f_v

            istances.append(row)

        return pd.DataFrame(istances)