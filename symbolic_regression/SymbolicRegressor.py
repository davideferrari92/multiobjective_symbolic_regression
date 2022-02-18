import logging
import os
import time
from typing import Union

import numpy as np
import pandas as pd
from joblib.parallel import Parallel, delayed

from symbolic_regression.multiobjective.training import (create_pareto_front,
                                                         crowding_distance,
                                                         generate_population,
                                                         get_offspring)

backend_parallel = 'loky'


class SymbolicRegressor:
    def __init__(
        self,
        checkpoint_file: str = None,
        checkpoint_frequency: int = -1,
        const_range: tuple = None,
        parsimony=0.9,
        parsimony_decay=0.9,
        population_size: int = 100,
        tournament_size: int = 10,
    ) -> None:
        """ This class implements the basic features for training a Symbolic Regression algorithm

        Args:
            - const_range: this is the range of values from which to generate constants in the program
            - fitness_functions: the functions to use for evaluating programs' performance
            - parsimony: the ratio to which a new operation is chosen instead of a terminal node in program generations
            - parsimony_decay: a modulation parameter to decrease the parsimony and limit program generation depth
            - tournament_size: this modulate the tournament selection and set the dimension of the selection
        """

        # Model characteristics
        self.best_program = None
        self.best_programs_history = []
        self.converged_generation = None
        self.generation = None
        self.population = None
        self.population_size = population_size
        self.status = "Uninitialized"
        self.training_duration = None

        # Training configurations
        self.checkpoint_file = checkpoint_file
        self.checkpoint_frequency = checkpoint_frequency
        self.const_range = const_range
        self.parsimony = parsimony
        self.parsimony_decay = parsimony_decay
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
                for p_confront in self.population[index + 1:]:
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
        timeout_offspring: float = None,
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
                timeout_offspring=timeout_offspring,
                verbose=verbose
            )
        except KeyboardInterrupt:
            self.generation -= 1  # The increment is applied even if the generation is interrupted
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
        timeout_offspring: float = None,
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

        elapsed_time = 0
        while True:
            self.generation += 1

            start_time_generation = time.perf_counter()
            converged_time = None

            print("#################################################################")
            print("#################################################################")

            print(f"Generation {self.generation}/{generations}")

            logging.debug(f"Generating offspring")
            self.status = "Generating offspring"

            offsprings = []

            if n_jobs > 0:
                m_workers = n_jobs
            else:
                m_workers = os.cpu_count()

            import concurrent.futures
            with concurrent.futures.ProcessPoolExecutor(max_workers=m_workers) as executor:
                for _ in range(self.population_size):
                    offsprings.append(executor)

                offsprings = [
                    o.submit(
                        get_offspring,
                        self.population,
                        data,
                        fitness_functions,
                        self.generation,
                        self.tournament_size,
                        genetic_operators_frequency
                    ) for o in offsprings
                ]
                                
                for index, o in enumerate(offsprings):
                    if isinstance(o, concurrent.futures._base.Future):
                        offsprings[index] = o.result()
                
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
                missing_elements = 2*self.population_size - \
                    len(self.population)

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

            self.average_complexity = np.mean(
                [p.complexity for p in self.population])

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

            if self.best_program.converged:
                converged_time = time.perf_counter()
                if not self.converged_generation:
                    self.converged_generation = self.generation
                logging.info(
                    f"Training converged after {self.converged_generation} generations."
                )
                if stop_at_convergence:
                    self.status = "Terminated: converged"
                    return

            if self.checkpoint_file and self.checkpoint_frequency > 0 and self.checkpoint_frequency == self.generation:
                try:
                    self.save_model(file=self.checkpoint_file)
                except FileNotFoundError:
                    logging.warning(
                        f'FileNotFoundError raised in checkpoint saving')

            # Use generations = -1 to rely only on convergence (risk of infinite loop)
            if generations > 0 and self.generation == generations:
                logging.info(
                    f"Training terminated after {self.generation} generations")
                self.status = "Terminated: generations completed"
                return

            elapsed_time += end_time_generation-start_time_generation

            if self.generation > 1:
                seconds_iter = round(elapsed_time/(self.generation), 1)
                print(f"{elapsed_time} sec, {seconds_iter} sec/generation")
            else:
                print(f"{elapsed_time} sec")

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
            row['index'] = index + 1
            row['program'] = p.program
            row['complexity'] = p.complexity

            for f_k, f_v in p.fitness.items():
                row[f_k] = f_v

            istances.append(row)

        return pd.DataFrame(istances)

    @property
    def best_history(self):
        istances = []

        for index, p in enumerate(self.best_programs_history):
            row = {}
            row['generation'] = index + 1
            row['program'] = p.program
            row['complexity'] = p.complexity

            for f_k, f_v in p.fitness.items():
                row[f_k] = f_v

            istances.append(row)
        istances.reverse()
        return pd.DataFrame(istances)
