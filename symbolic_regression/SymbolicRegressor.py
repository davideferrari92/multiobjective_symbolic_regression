import copy
import datetime
import json
import logging
import multiprocessing
import os
import pickle
import random
import time
import zlib
from itertools import repeat
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from loky import get_reusable_executor
from scipy.stats import spearmanr

from symbolic_regression.callbacks.CallbackBase import MOSRCallbackBase
from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.multiobjective.hypervolume import _HyperVolume
from symbolic_regression.Population import Population
from symbolic_regression.Program import Program

backend_parallel = 'loky'


class SymbolicRegressor:

    def __init__(self, client_name: str, const_range: tuple = (0, 1), parsimony=0.8, parsimony_decay=0.85, population_size: int = 300, tournament_size: int = 3, genetic_operators_frequency: dict = {'crossover': 1, 'mutation': 1}, statistics_computation_frequency: int = 10, callbacks: List[MOSRCallbackBase] = list()) -> None:
        """ This class implements the basic features for training a Symbolic Regression algorithm

        Args:
            - client_name: str
                the name of the client

            - const_range: tuple (default: (0, 1))
                this is the range of values from which to generate constants in the program

            - parsimony: float (default: 0.8)
                the ratio to which a new operation is chosen instead of a terminal node in program generations

            - parsimony_decay: float (default: 0.85)
                a modulation parameter to decrease the parsimony and limit program generation depth

            - population_size: int (default: 300)
                the size of the population

            - tournament_size: int (default: 3)
                this modulate the tournament selection and set the dimension of the selection

            - genetic_operators_frequency: dict (default: {'crossover': 1, 'mutation': 1})
                this is a dictionary that set the relative frequency of each genetic operator

            - statistics_computation_frequency: int (default: 10)
                this is the frequency of computing the statistics of the population
                - if set to -1, the statistics are computed only at the end of the training
                - if set to 1, the statistics are computed at each generation
                - if set to 10, the statistics are computed every 10 generations
                Statistics are:
                    - Hypervolume
                    - Tree Diversity
                    - Spearman Diversity

            - callbacks: List[MOSRCallbackBase] (default: list())
                this is the list of callbacks to be executed during the training

        Returns:
            - None
        """

        # Regressor Configuration
        self.client_name: str = client_name
        self.features: List = None
        self.operations: List = None
        self.population_size: int = population_size

        # Population Configuration
        self.population: Population = Population()
        self._average_complexity: float = None
        self.const_range: tuple = const_range
        self.parsimony: float = parsimony
        self.parsimony_decay: float = parsimony_decay
        self.tournament_size: int = tournament_size

        # Training Configuration
        self.data_shape: tuple = None
        self.converged_generation: int = None
        self.fitness_functions: List[BaseFitness] = None
        self.generations_to_train: int = None
        self.generation: int = 0
        self.genetic_operators_frequency: dict = genetic_operators_frequency

        # Statistics
        self.best_history: Dict = dict()
        self.first_pareto_front_history: Dict = dict()
        self.fpf_hypervolume: float = None
        self.fpf_hypervolume_reference: float = None
        self.fpf_tree_diversity: float = None
        self.fpf_spearman_diversity: float = None
        self.fpf_stats: pd.DataFrame = pd.DataFrame()
        self.training_duration: int = 0

        self._callbacks: List[MOSRCallbackBase] = list()
        self.callbacks: List[MOSRCallbackBase] = callbacks

        self.times: pd.DataFrame = pd.DataFrame(
            columns=['count_average_complexity',
                     'count_duplicated_elements',
                     'count_invalid_elements',
                     'ratio_duplicated_elements',
                     'ratio_invalid_elements',
                     'time_crowding_distance_computation',
                     'time_duplicated_drop',
                     'time_generation_total'
                     'time_hypervolume_computation',
                     'time_initialization',
                     'time_invalids_drop',
                     'time_offsprings_generation',
                     'time_pareto_front_computation',
                     'time_refill_invalid',
                     'time_spearman_diversity_computation',
                     'time_tree_diversity_computation',]
        )

    def save_model(self, file: str, checkpoint_overwrite: bool = None):
        # Gets the number of the generation padded with zeros to the number
        # of digits of the total number of generations
        generation_str = str(self.generation).zfill(
            len(str(self.generations_to_train)))

        if not checkpoint_overwrite:
            file = file + f".gen{generation_str}.sr"
        else:
            file = file + ".sr"

        # Dump this object in a pickle file
        for p in self.population:
            p.programs_dominated_by: List[Program] = list()
            p.programs_dominates: List[Program] = list()

        with open(file, "wb") as f:
            pickle.dump(compress(self), f)

        # Dump the self.metadata in a json file
        with open(file + ".metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def load_model(self, file: str):
        with open(file, "rb") as f:
            sr: SymbolicRegressor = pickle.load(f)

        try:
            sr = decompress(sr)
            logging.debug(f"Loaded compressed model from {file}")
        except TypeError:
            pass

        return sr

    @property
    def average_complexity(self):
        return self._average_complexity

    @average_complexity.getter
    def average_complexity(self):
        return np.mean([p.complexity for p in self.population]) if len(self.population) > 0 else 0

    @property
    def callbacks(self):
        if hasattr(self, '_callbacks'):
            return self._callbacks
        else:
            return list()

    @callbacks.setter
    def callbacks(self, callbacks: List[MOSRCallbackBase]):
        if not isinstance(callbacks, List):
            raise TypeError(
                f"Expected a list of MOSRCallbackBase, got {type(callbacks)}")
        self._callbacks = callbacks
        for c in self._callbacks:
            c.sr: 'SymbolicRegressor' = self
            try:
                c.on_callback_set_init()
            except:
                logging.warning(
                    f"Callback {c.__class__.__name__} raised an exception on callback set init")

    def compute_hypervolume(self, exclusive: bool = False):
        """
        This method computes the hypervolume of the current population

        The hypervolume is computed using the hypervolume reference point
        defined in the fitness functions.

        If the reference point is not defined the hypervolume is computed 
        using the maximum value of each fitness function as reference point.

        If the reference point is defined but the fitness function is not
        a minimization problem, the hypervolume is computed using the maximum
        value of each fitness function as reference point.

        If the reference point is defined but is lower than the maximum value
        of the fitness function, the hypervolume is computed using the maximum
        value of each fitness function as reference point and the reference
        point is updated to the new value.

        Args:
            - exclusive: bool  (default: False)
                If True, the hypervolume is computed on the whole pareto front and
                saved in the statistics. If False, the hypervolume is computed only
                in an exclusive way and not saved. This is for final checks on the
                model. This last option may be inefficient and slow.

        Returns:
            - hypervolume: float
                The hypervolume of the current population
        """

        fitness_to_hypervolume = list()
        for fitness in self.fitness_functions:
            if fitness.hypervolume_reference and fitness.minimize:
                fitness_to_hypervolume.append(fitness)

        points = np.array([np.array([p.fitness[ftn.label] for ftn in fitness_to_hypervolume])
                           for p in self.first_pareto_front])

        references = np.array(
            [ftn.hypervolume_reference for ftn in fitness_to_hypervolume])

        if not exclusive:
            try:
                points = points[np.sum((points - references)
                                       <= 0, axis=1) == points.shape[1]]
                self.fpf_hypervolume = _HyperVolume(references).compute(points)

            except ValueError:
                self.fpf_hypervolume = np.nan

            self.fpf_stats.loc[self.generation, 'n_individuals'] = len(
                self.first_pareto_front)
            self.fpf_stats.loc[self.generation,
                               'fpf_hypervolume'] = self.fpf_hypervolume
            self.fpf_stats.loc[self.generation,
                               'fpf_hypervolume_reference'] = self.fpf_hypervolume_reference

            self.fpf_hypervolume_reference = np.product(references)

            return self.fpf_hypervolume

        else:
            try:
                points = points[np.sum((points - references)
                                       <= 0, axis=1) == points.shape[1]]
                hypervolume = _HyperVolume(references).exclusive(points)
            except ValueError:
                hypervolume = np.nan

            return hypervolume

    def compute_fitness_population(self, fitness_functions: List[BaseFitness] = None, data: Union[dict, pd.DataFrame, pd.Series] = None, validation: bool = False, validation_federated: bool = False, simplify: bool = True):
        """
        This method computes the performance of each program in the population

        Args:
            - fitness_functions: List[BaseFitness] (default: None)
                The fitness functions to use to compute the performance. Can override by the fitness_functions
                attribute of the class
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance is computed
            - validation: bool (default: False)
                If True the performance is computed on the validation data without optimization
            - validation_federated: bool (default: False)
                If True the performance is computed on the training and validation data without optimization (only for federated training)
            - simplify: bool (default: False)
                If True the program is simplified before computing the performance
        """
        if data is None:
            logging.warning(
                f'No {"training" if not validation else "validation"} data provided to compute the fitness of the population')
            return

        for p in self.population:
            p: Program
            p.compute_fitness(
                fitness_functions=self.fitness_functions if fitness_functions is None else fitness_functions,
                data=data,
                validation=validation,
                validation_federated=validation_federated,
                simplify=simplify)

    def _create_pareto_front(self):
        """
        This method creates the pareto front of the population
        The pareto front is a group of programs that are non-dominated by any other program in the population
        The first pareto front is the one with the lowest rank and therefore the best programs
        The following pareto fronts are the ones with the next lowest rank and so on

        We use the first pareto fron to identify the most optimal programs the crowding distance to identify the most
        diverse programs.
        """
        pareto_front = list()

        # Loop over the entire matrix, can be optimised to do only the triangular matrix
        for p1 in self.population:
            p1.rank = np.inf

            if not p1.is_valid:
                continue
            p1.programs_dominates = list()
            p1.programs_dominated_by = list()

            for p2 in self.population:
                if p1 == p2 or not p2.is_valid:
                    continue

                if self.dominance(p1, p2):
                    p1.programs_dominates.append(p2)
                elif self.dominance(p2, p1):
                    p1.programs_dominated_by.append(p2)

            if len(p1.programs_dominated_by) == 0:
                p1.rank = 1
                pareto_front.append(p1)

        i = 1

        # Set the belonging pareto front to every element of the population

        while pareto_front:
            next_pareto_front = list()

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

    def _crowding_distance(self):
        """
        This method calculates the crowding distance for each program in the population
        The crowding distance is used to identify the most diverse programs in the population
        It is calculated as the sum of the normalized distances between the programs in the pareto front
        The distance is calculated for each objective function.

        The higher the crowding distance the more diverse the program is with respect to the other programs in the same pareto front
        """

        rank_iter = 1
        pareto_front = self.extract_pareto_front(rank=rank_iter)

        while pareto_front:  # Exits when extract_pareto_front return an empty list
            for ftn in self.fitness_functions:

                fitness_label = ftn.label
                # This exclude the fitness functions which are set not to be minimized
                if not ftn.minimize:
                    continue

                # Highest fitness first for each objective
                pareto_front.sort(
                    key=lambda p: p.fitness[fitness_label], reverse=True)

                norm = pareto_front[0].fitness[fitness_label] - \
                    pareto_front[-1].fitness[fitness_label] + 1e-20

                for index, program in enumerate(pareto_front):
                    if index == 0 or index == len(pareto_front) - 1:
                        program.crowding_distance = float('inf')
                    else:
                        delta = pareto_front[index - 1].fitness[fitness_label] - \
                            pareto_front[index + 1].fitness[fitness_label]

                        program.crowding_distance = delta / norm

            rank_iter += 1
            pareto_front = self.extract_pareto_front(rank=rank_iter)

    @staticmethod
    def dominance(program1: Program, program2: Program) -> bool:
        """
        This method checks if program1 dominates program2
        A program p1 dominates a program p2 if all the fitness of p1 are 
        less or equal than p2 and at least one is less than p2

        We use this method to create the pareto fronts
        We allow fitness functions to be set not to be minimized, in this 
        case the fitness is calculated but not used to create the pareto fronts

        Args:
            = program1: Program
                The program that is being checked if it dominates program2
            = program2: Program
                The program that is being checked if it is dominated by program1

        Returns:
            = True if program1 dominates program2
            = False otherwise
        """

        # How many element in the p1.fitness are less than p2.fitness
        at_least_one_less_than_zero = False
        all_less_or_eq_than_zero = True

        if program1.program and program2.program:
            for this_fitness in program1.fitness.keys():
                # Ignore the fitness which are not to be optimized
                if program1.is_fitness_to_minimize[this_fitness] == False:
                    continue

                try:
                    p1_fitness = program1.fitness[this_fitness]
                    p2_fitness = program2.fitness[this_fitness]
                except KeyError:
                    break

                d = abs(p1_fitness) - \
                    abs(p2_fitness)

                if d < 0:
                    at_least_one_less_than_zero = True
                if d > 0:
                    all_less_or_eq_than_zero = False

            return at_least_one_less_than_zero and all_less_or_eq_than_zero

        return False

    def drop_duplicates(self, inplace: bool = False) -> list:
        """ 
        This method removes the duplicates from the population
        A program is considered a duplicate if it has the same fitness rounded to 
        the number of decimals specified in the implementation of the program

        Args:
            - inplace: bool (default False)
                If True the population is updated, if False a new list is returned
        """

        complexities_index = np.argsort(
            [p.complexity for p in self.population])

        for index, complexity_index in enumerate(complexities_index):
            p = self.population[complexity_index]

            if p.is_valid and not p._is_duplicated:

                other_indices = [complexities_index[l] for l in range(
                    index + 1, len(complexities_index))]

                for j in other_indices:
                    p_confront = self.population[j]
                    if p_confront.is_valid and not p_confront._is_duplicated:
                        p_confront._is_duplicated = p.is_duplicate(p_confront)

        if inplace:
            self.population: Population = Population(
                filter(lambda p: p._is_duplicated == False, self.population))
            return self.population

        return Population(
            filter(lambda p: p._is_duplicated == False, self.population))

    def drop_invalids(self, inplace: bool = False) -> list:
        """
        This method removes the invalid programs from the population
        A program is considered invalid if it has an InvalidNode in its tree or 
        if at least one of the operations is mathematically impossible, like 
        division by zero.

        Args:
            - inplace: bool (default False)
                If True the population is updated, if False a new list is returned
        """
        if inplace:
            self.population: Population = Population(
                filter(lambda p: p.is_valid == True, self.population))
            return self.population

        return list(filter(lambda p: p.is_valid == True, self.population))

    def extract_pareto_front(self, rank: int):
        """
        This method extracts the programs in the population that are in the pareto front
        of the specified rank

        Args:
            - rank: int
                The rank of the pareto front to be extracted

        Returns:
            - pareto_front: List
                The list of programs in the pareto front of the specified rank
        """
        pareto_front = list()
        for p in self.population:
            if p and p.rank == rank:
                pareto_front.append(p)

        return pareto_front

    @property
    def first_pareto_front(self):
        """
        First Pareto Front

        This method returns the first Pareto Front of the population.

        Args:
            - None

        Returns:
            - first_pareto_front: list
                The first Pareto Front of the population
        """
        return [p for p in self.population if p.rank == 1]

    def fit(self, data: Union[dict, pd.DataFrame, pd.Series], features: List[str], operations: List[dict], fitness_functions: List[BaseFitness], generations_to_train: int, n_jobs: int = -1, stop_at_convergence: bool = False, verbose: int = 0, val_data: Union[dict, pd.DataFrame, pd.Series] = None) -> None:
        """
        This method trains the population.

        This method implements store in this object the arguments passed at its execution;
        because these are required in other methods of the class.

        We implement here the call to the method that performs the training allowing to
        catch the exception KeyboardInterrupt that is raised when the user stops the training.
        This allow the user to stop the training at any time and not lose the progress made.
        The actual training is then implemented in the private method _fit.

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the training is performed
            - features: List
                The list of features to be used in the training
            - operations: List
                The list of operations to be used in the training
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the training
            - generations_to_train: int
                The number of generations to train
            - n_jobs: int (default -1)
                The number of jobs to be used in the training
            - stop_at_convergence: bool (default False)
                If True the training stops when the population converges
            - verbose: int (default 0)
                The verbosity level of the training
            - val_data: Union[dict, pd.DataFrame, pd.Series] (default None)
                The data on which the validation is performed

        Returns:
            - None
        """
        self.features = features
        self.operations = operations

        self.data_shape = data.shape

        self.fitness_functions = fitness_functions
        self.generations_to_train = generations_to_train
        self.n_jobs = n_jobs
        self.stop_at_convergence = stop_at_convergence
        self.verbose = verbose

        start = time.perf_counter()
        try:
            self._fit(data=data, val_data=val_data)
        except KeyboardInterrupt:
            self.generation -= 1  # The increment is applied even if the generation is interrupted
            self.status = "Interrupted by KeyboardInterrupt"
            logging.warning(f"Training terminated by a KeyboardInterrupt")

            if not isinstance(self.population, Population):
                self.population = Population(self.population)

        stop = time.perf_counter()
        self.training_duration += stop - start

    def _fit(self, data: Union[dict, pd.DataFrame, pd.Series], val_data: Union[dict, pd.DataFrame, pd.Series] = None) -> None:
        """
        This method is the main loop of the genetic programming algorithm.

        Firstly we initialize the population of N individuals if it is not already initialized.
        Then we loop over the generations and apply the genetic operators
        to the population.

        We then generate other N offspring using the genetic operators and
        we add them to the population. This generates a population of 2N individuals.

        To select the population of N individuals to pass to the next generation
        we use the NSGA-II algorithm (Non-dominant Sorting Genetic Algorithm) which 
        sort the 2N individuals in non-dominated fronts and then select the N individuals
        that have the best performance.

        We need to remove the individuals that are not valid because and also the individuals
        that are duplicated. If removing the invalid and duplicated individuals leaves us
        with less than N individuals, we generate new individuals to fill the population.

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the training is performed
            - val_data: Union[dict, pd.DataFrame, pd.Series] (default None)
                The data on which the validation is performed

        Returns:
            - None

        """
        total_generation_time = 0
        jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()

        executor = get_reusable_executor(max_workers=jobs, timeout=100)

        if not self.population:

            for c in self.callbacks:
                try:
                    c.on_initialization_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on initialization start")

            before = time.perf_counter()
            if self.verbose > 0:
                print(f"Initializing population")
            self.status = "Generating population"

            self.population = Population(executor.map(
                lambda p: self.generate_individual(*p),
                zip(
                    repeat(copy.deepcopy(data), self.population_size),
                    repeat(self.features, self.population_size),
                    repeat(self.operations, self.population_size),
                    repeat(copy.deepcopy(self.fitness_functions),
                           self.population_size),
                    repeat(self.const_range, self.population_size),
                    repeat(self.parsimony, self.population_size),
                    repeat(self.parsimony_decay, self.population_size),
                    repeat(copy.deepcopy(val_data), self.population_size),
                )
            ))

            self.times.loc[self.generation+1,
                           "time_initialization"] = time.perf_counter() - before

            for c in self.callbacks:
                try:
                    c.on_initialization_end()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on initialization end")

        else:
            self.population = [
                program for program in self.population if program.is_valid]

            logging.info(
                f"Fitting with existing population of {len(self.population)} valid elements")

            if len(self.population) < self.population_size:
                logging.info(
                    f"Population of {len(self.population)} elements is less than population_size:{self.population_size}. Integrating with new elements")
                new_individuals = self.population_size - len(self.population)
                refill_training_start = Population(executor.map(
                    lambda p: self.generate_individual(*p),
                    zip(
                        repeat(copy.deepcopy(data), new_individuals),
                        repeat(self.features, new_individuals),
                        repeat(self.operations, new_individuals),
                        repeat(copy.deepcopy(self.fitness_functions),
                               new_individuals),
                        repeat(self.const_range, new_individuals),
                        repeat(self.parsimony, new_individuals),
                        repeat(self.parsimony_decay, new_individuals),
                        repeat(copy.deepcopy(val_data), new_individuals),
                    )
                ))

                self.population.extend(refill_training_start)
                self.population = Population(self.population)

        while True:
            self.status = "Training"

            if self.generation > 0 and self.generations_to_train <= self.generation:
                logging.info(
                    f"The model already had trained for {self.generation} generations")
                self.status = "Terminated: generations completed"
                return

            for c in self.callbacks:
                try:
                    c.on_generation_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on generation start")

            start_time_generation = time.perf_counter()

            for p in self.population:
                p.programs_dominates: List[Program] = list()
                p.programs_dominated_by: List[Program] = list()

            if self.generation > 0:
                time_total = datetime.timedelta(seconds=int(
                    self.times['time_generation_total'].sum()))
                seconds_iter_avg = datetime.timedelta(seconds=int(
                    self.times['time_generation_total'].tail(5).median()))
                seconds_iter_std = datetime.timedelta(seconds=int(self.times['time_generation_total'].tail(
                    5).std())) if self.generation > 1 else datetime.timedelta(seconds=0)
                time_per_generation = f"{seconds_iter_avg} ± {seconds_iter_std}"
                expected_time = datetime.timedelta(seconds=int(self.times['time_generation_total'].tail(
                    10).median() * (self.generations_to_train - self.generation)))
            else:
                time_total = f"00:00:00"
                time_per_generation = f"00:00:00 ± 00:00:00"
                expected_time = 'Unknown'

            generation_time = datetime.timedelta(
                seconds=int(total_generation_time))
            timing_str = f"Generation {generation_time} - On average: {time_per_generation} - Total: {time_total} - To completion: {expected_time}"

            self.generation += 1

            if self.verbose > 1:
                print("#" * len(timing_str))
                print(timing_str)
                print("#" * len(timing_str))
            if self.verbose > 0:
                print(
                    f"{self.client_name}: starting generation {self.generation}/{self.generations_to_train}", end='\r' if self.verbose == 1 else '\n', flush=True)

            before = time.perf_counter()

            for c in self.callbacks:
                try:
                    c.on_offspring_generation_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on offspring generation start")

            self.status = "Generating Offsprings"

            offsprings: List[Program] = list()
            procs: List[Process] = list()

            queue = multiprocessing.Manager().Queue(maxsize=self.population_size)

            for _ in range(jobs):
                try:
                    population_to_pass = self.population.as_binary()
                except AttributeError:
                    population_to_pass = self.population
                proc = Process(
                    target=self._get_offspring_batch,
                    args=(
                        copy.deepcopy(data),
                        self.genetic_operators_frequency,
                        self.fitness_functions,
                        population_to_pass,
                        self.tournament_size,
                        self.generation,
                        int(max(os.cpu_count(), self.population_size/jobs)),
                        queue,
                        val_data
                    )
                )
                procs.append(proc)

            was_limited_str = ''
            for index, proc in enumerate(procs):
                proc.start()

            q_size = 0
            while q_size < self.population_size:
                # Make sure at least some processes are alive
                q_size = queue.qsize()
                if self.verbose > 1:
                    _elapsed = max(1, int(round(time.perf_counter() - before)))
                    print(
                        f'Offsprings generated: {q_size}/{self.population_size} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s) {was_limited_str}   ', end='\r', flush=True)
                time.sleep(.2)

            else:
                q_size = queue.qsize()
                if self.verbose > 1:
                    _elapsed = max(1, int(round(time.perf_counter() - before)))
                    print(
                        f'Offsprings generated: {q_size}/{self.population_size} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s). Completed!  {was_limited_str}   ', flush=True)
                for p in procs:
                    if p.is_alive():
                        p.join(timeout=.1)

            for _ in range(self.population_size):
                offsprings.append(queue.get())

            for proc in procs:
                try:
                    proc.kill()
                except:
                    pass

            self.times.loc[self.generation,
                           "time_offsprings_generation"] = time.perf_counter() - before

            self.population = Population(self.population + offsprings)

            for c in self.callbacks:
                try:
                    c.on_offspring_generation_end()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on offspring generation end")

            # Removes all duplicated programs in the population
            before_cleaning = len(self.population)

            before = time.perf_counter()
            self.drop_duplicates(inplace=True)
            self.times.loc[self.generation,
                           "time_duplicated_drop"] = time.perf_counter() - before

            after_drop_duplicates = len(self.population)
            logging.debug(
                f"{before_cleaning-after_drop_duplicates}/{before_cleaning} duplicates programs removed")

            self.times.loc[self.generation,
                           "count_duplicated_elements"] = before_cleaning-after_drop_duplicates
            self.times.loc[self.generation, "ratio_duplicated_elements"] = (
                before_cleaning-after_drop_duplicates)/before_cleaning

            # Removes all non valid programs in the population
            before = time.perf_counter()
            self.drop_invalids(inplace=True)
            self.times.loc[self.generation,
                           "time_invalids_drop"] = time.perf_counter() - before

            after_cleaning = len(self.population)
            if before_cleaning != after_cleaning:
                logging.debug(
                    f"{after_drop_duplicates-after_cleaning}/{after_drop_duplicates} invalid programs removed")

            # Integrate population in case of too many invalid programs
            if len(self.population) < self.population_size * 2:
                self.status = "Refilling Individuals"

                before = time.perf_counter()
                missing_elements = 2*self.population_size - \
                    len(self.population)

                logging.debug(
                    f"Population of {len(self.population)} elements is less than 2*population_size:{self.population_size*2}. Integrating with {missing_elements} new elements")

                for c in self.callbacks:
                    try:
                        c.on_refill_start()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on refill start")

                refill: List[Program] = list()
                procs: List[Process] = list()
                queue = multiprocessing.Manager().Queue(maxsize=missing_elements)

                for _ in range(jobs):
                    proc = Process(
                        target=self.generate_individual_batch,
                        args=(
                            data,
                            self.features,
                            self.operations,
                            self.fitness_functions,
                            self.const_range,
                            self.parsimony,
                            self.parsimony_decay,
                            int(max(os.cpu_count(), missing_elements/jobs)),
                            queue,
                            val_data
                        )
                    )
                    procs.append(proc)
                    proc.start()

                q_size = 0
                while q_size < missing_elements:
                    q_size = queue.qsize()
                    if self.verbose > 1:
                        _elapsed = max(
                            1, int(round(time.perf_counter() - before)))
                        print(
                            f'Duplicates/invalid refilled: {q_size}/{missing_elements} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s)', end='\r', flush=True)
                    time.sleep(.5)

                else:
                    if self.verbose > 1:
                        _elapsed = max(
                            1, int(round(time.perf_counter() - before)))
                        print(
                            f'Duplicates/invalid refilled: {q_size}/{missing_elements} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s). Completed!', flush=True)
                    for p in procs:
                        if p.is_alive():
                            p.join(timeout=.1)

                for _ in range(missing_elements):
                    refill.append(queue.get())

                for proc in procs:
                    try:
                        proc.kill()
                    except:
                        pass

                self.population: Population = Population(
                    self.population + refill)

                for c in self.callbacks:
                    try:
                        c.on_refill_end()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on refill end")

                # exludes every program in refill with an empty fitness
                self.population = [
                    p for p in self.population if not p._has_incomplete_fitness]

                self.times.loc[self.generation,
                               "time_refill_invalid"] = time.perf_counter() - before
                self.times.loc[self.generation,
                               "count_invalid_elements"] = missing_elements
                self.times.loc[self.generation,
                               "ratio_invalid_elements"] = missing_elements / len(self.population)

            for c in self.callbacks:
                try:
                    c.on_pareto_front_computation_start()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on pareto front computation start")

            self.status = "NSGA-II Pareto Front Computation"

            # Calculates the Pareto front
            before = time.perf_counter()
            self._create_pareto_front()
            self.times.loc[self.generation,
                           "time_pareto_front_computation"] = time.perf_counter() - before

            # Calculates the crowding distance
            before = time.perf_counter()
            self._crowding_distance()
            self.times.loc[self.generation,
                           "time_crowding_distance_computation"] = time.perf_counter() - before

            self.population.sort(
                key=lambda p: p.crowding_distance, reverse=True)
            self.population.sort(key=lambda p: p.rank, reverse=False)
            self.population = Population(
                self.population[:self.population_size])

            self.times.loc[self.generation,
                           "count_average_complexity"] = self.average_complexity

            if any(p.converged for p in self.population):
                if not self.converged_generation:
                    self.converged_generation = self.generation
                if self.verbose > 1:
                    print(
                        f"Training converged after {self.converged_generation} generations.")

            for c in self.callbacks:
                try:
                    c.on_pareto_front_computation_end(
                        data=data, val_data=val_data)
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on pareto front computation end")

            end_time_generation = time.perf_counter()
            self._print_first_pareto_front(verbose=self.verbose)

            total_generation_time = end_time_generation - start_time_generation
            self.times.loc[self.generation,
                           "time_generation_total"] = total_generation_time

            for c in self.callbacks:
                try:
                    c.on_generation_end()
                except:
                    logging.warning(
                        f"Callback {c.__class__.__name__} raised an exception on generation end")

            # Use generations = -1 to rely only on convergence (risk of infinite loop)
            if self.generations_to_train > 0 and self.generation == self.generations_to_train:
                for c in self.callbacks:
                    try:
                        c.on_training_completed()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on training completed")

                print(
                    f"Training completed {self.generation}/{self.generations_to_train} generations")
                return

            if self.converged_generation and self.stop_at_convergence:
                for c in self.callbacks:
                    try:
                        c.on_convergence()
                    except:
                        logging.warning(
                            f"Callback {c.__class__.__name__} raised an exception on convergence")

                print(
                    f"Training converged after {self.converged_generation} generations and requested to stop.")
                return

    def generate_individual(self, data: Union[dict, pd.DataFrame, pd.Series], features: List[str], operations: List[dict], fitness_functions: List[BaseFitness], const_range: tuple = (0, 1), parsimony: float = 0.8, parsimony_decay: float = 0.85, val_data: Union[dict, pd.DataFrame, pd.Series] = None) -> Program:
        """
        This method generates a new individual for the population

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance are evaluated. We could use compute_fitness
                later, but we need to evaluate the fitness here to compute it in the
                parallel initialization.
            - features: List[str]
                The list of features to be used in the generation
            - operations: List[dict]
                The list of operations to be used in the generation
            - const_range: tuple
                The range of the constants to be used in the generation. There will then be
                adapted during the optimization process.
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the generation
            - parsimony: float (default 0.8)
                The parsimony coefficient to be used in the generation. This modulates how
                likely the program is to be complex (deep) or simple (shallow).
            - parsimony_decay: float (default 0.85)
                The parsimony decay to be used in the generation. This modulates how the parsimony
                coefficient is updated at each generation. The parsimony coefficient is multiplied
                by this value at each generation. This allows to decrease the parsimony coefficient
                over time to prevent the program to be too complex (deep).
            - val_data: Union[dict, pd.DataFrame, pd.Series] (default None)
                The data on which the validation is performed

        Returns:
            - p: Program
                The generated program

        """
        new_p = Program(features=features, operations=operations, const_range=const_range,
                        parsimony=parsimony, parsimony_decay=parsimony_decay)
        new_p.init_program()
        new_p.compute_fitness(fitness_functions=fitness_functions,
                              data=data, validation=False, simplify=True)

        if val_data is not None:
            new_p.compute_fitness(
                fitness_functions=fitness_functions, data=val_data, validation=True)

        return new_p

    def generate_individual_batch(self, data: Union[dict, pd.DataFrame, pd.Series], features: List[str], operations: List[dict], fitness_functions: List[BaseFitness], const_range: tuple = (0, 1), parsimony: float = 0.8, parsimony_decay: float = 0.85, batch_size: int = 1000, queue: Queue = None, val_data: Union[dict, pd.DataFrame, pd.Series] = None) -> Program:
        """
        This method generates a new individual for the population

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance are evaluated. We could use compute_fitness
                later, but we need to evaluate the fitness here to compute it in the
                parallel initialization.
            - features: List[str]
                The list of features to be used in the generation
            - operations: List[dict]
                The list of operations to be used in the generation
            - const_range: tuple
                The range of the constants to be used in the generation. There will then be
                adapted during the optimization process.
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the generation
            - parsimony: float (default 0.8)
                The parsimony coefficient to be used in the generation. This modulates how
                likely the program is to be complex (deep) or simple (shallow).
            - parsimony_decay: float (default 0.85)
                The parsimony decay to be used in the generation. This modulates how the parsimony
                coefficient is updated at each generation. The parsimony coefficient is multiplied
                by this value at each generation. This allows to decrease the parsimony coefficient
                over time to prevent the program to be too complex (deep).
            - batch_size: int (default 1000)
                The number of individuals to generate
            - queue: Queue (default None)
                The queue in which to put the generated individuals
            - val_data: Union[dict, pd.DataFrame, pd.Series] (default None)
                The data on which the validation is performed

        Returns:
            - p: Program
                The generated program

        """

        new_ps: List[Program] = list()

        submitted = 0
        while submitted < batch_size * 3:

            if not self.status == "Refilling Individuals":
                return

            new_p = self.generate_individual(data=data, features=features, operations=operations,
                                             fitness_functions=fitness_functions, const_range=const_range,
                                             parsimony=parsimony, parsimony_decay=parsimony_decay, val_data=val_data)

            if new_p._has_incomplete_fitness:
                continue

            if queue is not None:
                queue.put(new_p)
            else:
                new_ps.append(new_p)

            submitted += 1

        return new_ps

    @staticmethod
    def _get_offspring(data: Union[dict, pd.DataFrame, pd.Series], genetic_operators_frequency: Dict[str, float], fitness_functions: List[BaseFitness], population: Population, tournament_size: int, generation: int, val_data: Union[Dict, pd.Series, pd.DataFrame] = None) -> Program:
        """
        This method generates an offspring of a program from the current population

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
            - do nothing: in this case no mutation is applied and the program is returned as is

        The frequency of which those operation are applied is determined by the dictionary
        genetic_operations_frequency in which the relative frequency of each of the desired operations
        is expressed with integers. The higher the number the likelier the operation will be chosen.
        Use integers > 1 to increase the frequency of an operation. Use 1 as minimum value.

        The program to which apply the operation is chosen using the tournament_selection, a method
        that identify the best program among a random selection of k programs from the population.

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance are evaluated. We could use compute_fitness
                later, but we need to evaluate the fitness here to compute it in the
                parallel initialization.
            - genetic_operators_frequency: Dict[str, float]
                The dictionary of the genetic operators and their frequency
            - fitness_functions: List[BaseFitness]
                The list of fitness functions to be used in the generation
            - population: List[Program]
                The current population from which to choose the program to which apply the genetic
                operator
            - tournament_size: int
                The size of the tournament selection
            - generation: int
                The current generation
            - val_data: Union[Dict, pd.Series, pd.DataFrame] (default None)
                The data on which the validation is performed

        Returns:
            - _offspring: Program
                The offspring of the current population

            - program1: Program     (only in case of errors or if the program is not valid)
                The program from which the offspring is generated
        """

        def _tournament_selection(population: Population, tournament_size: int, generation: int) -> Program:
            """
            Tournament selection

            This method selects the best program from a random sample of the population.
            It is used to apply the genetic operators during the generation of the
            offsprings.

            We firstly chose a random set of K programs from the population. Then we
            select the best program from this set. If the generation is 0, we use the
            program's complexity to select the best program. In the other generations
            we use the pareto front rank and the crowding distance.

            Args:
                - population: Population
                    The population of programs
                - tournament_size: int
                    The size of the tournament
                - generation: int
                    The current generation

            Returns:
                - best_member: Program
                    The best program selected by the tournament selection
            """

            tournament_members = random.choices(
                population, k=tournament_size)

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

        # Select the genetic operation to apply
        # The frequency of each operation is determined by the dictionary
        # genetic_operators_frequency. The higher the number the likelier the operation will be chosen.

        ops = list()
        for op, freq in genetic_operators_frequency.items():
            ops += [op] * freq
        gen_op = random.choice(ops)

        program1 = _tournament_selection(
            population=population, tournament_size=tournament_size, generation=generation)

        if program1 is None or not program1.is_valid:
            # If the program is not valid, we return a the same program without any alteration
            # because it would be unlikely to generate a valid offspring.
            # This program will be removed from the population later.
            program1.init_program()
            program1.compute_fitness(
                fitness_functions=fitness_functions, data=data, validation=False, simplify=True)
            if val_data is not None:
                program1.compute_fitness(
                    fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)
            return program1

        _offspring: Program = None
        if gen_op == 'crossover':
            program2 = _tournament_selection(
                population=population, tournament_size=tournament_size, generation=generation)
            if program2 is None or not program2.is_valid:
                return program1
            _offspring = program1.cross_over(
                other=program2, inplace=False)

        elif gen_op == 'randomize':
            # Will generate a new tree as other
            _offspring = program1.cross_over(other=None, inplace=False)

        elif gen_op == 'mutation':
            _offspring = program1.mutate(inplace=False)

        elif gen_op == 'delete_node':
            _offspring = program1.delete_node(inplace=False)

        elif gen_op == 'insert_node':
            _offspring = program1.insert_node(inplace=False)

        elif gen_op == 'mutate_operator':
            _offspring = program1.mutate_operator(inplace=False)

        elif gen_op == 'mutate_leaf':
            _offspring = program1.mutate_leaf(inplace=False)

        elif gen_op == 'simplification':
            _offspring = program1.simplify(inplace=False)

        elif gen_op == 'recalibrate':
            _offspring = program1.recalibrate(inplace=False)

        elif gen_op == 'do_nothing':
            _offspring = program1
        else:
            logging.warning(
                f'Supported genetic operations: crossover, delete_node, do_nothing, insert_node, mutate_leaf, mutate_operator, simplification, mutation and randomize'
            )

            return program1

        # Add the fitness to the object after the cross_over or mutation
        _offspring.compute_fitness(
            fitness_functions=fitness_functions, data=data, simplify=True)
        if val_data is not None:
            _offspring.compute_fitness(
                fitness_functions=fitness_functions, data=val_data, validation=True, simplify=False)

        # Reset the hash to force the re-computation
        _offspring._hash = None

        return _offspring

    def _get_offspring_batch(self, data: Union[dict, pd.DataFrame, pd.Series], genetic_operators_frequency: Dict[str, float], fitness_functions: List[BaseFitness], population: Population, tournament_size: int, generation: int,
                             batch_size: int, queue: Queue = None, val_data: Union[Dict, pd.Series, pd.DataFrame] = None) -> Program:

        population = population.as_program()
        offsprings: List[Program] = list()

        submitted = 0

        jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()

        while submitted < batch_size * jobs:

            if not self.status == "Generating Offsprings":
                return

            offspring = self._get_offspring(data=data, val_data=val_data,
                                            genetic_operators_frequency=genetic_operators_frequency,
                                            fitness_functions=fitness_functions, population=population,
                                            tournament_size=tournament_size, generation=generation)

            if offspring._has_incomplete_fitness:
                continue

            if queue is not None:
                queue.put(offspring)
            else:
                offsprings.append(offspring)

            submitted += 1

        return offsprings

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        This method returns the metadata of the current population

        Args:
            - None

        Returns:
            - metadata: Dict[str, Any]
                The metadata of the current population
        """
        metadata = {
            'average_complexity': self.average_complexity,
            'client_name': self.client_name,
            'const_range': self.const_range,
            'converged_generation': self.converged_generation,
            'fitness_functions': [ff.__repr__() for ff in self.fitness_functions],
            'fpf_hypervolume': self.fpf_hypervolume,
            'fpf_tree_diversity': self.fpf_tree_diversity,
            'generation': self.generation,
            'generations_to_train': self.generations_to_train,
            'genetic_operators_frequency': self.genetic_operators_frequency,
            'operations': [op['symbol'] for op in self.operations],
            'parsimony_decay': self.parsimony_decay,
            'parsimony': self.parsimony,
            'population_size': self.population_size,
            'total_time': self._total_time,
            'tournament_size': self.tournament_size,
        }

        return metadata

    def plot_generations_time(self):
        """ This method plots the time spent in each generation

        Args:
            - None

        Returns:
            - fig: go.Figure
                The plotly figure
        """

        import plotly.express as px

        def format_time(seconds):
            return datetime.datetime.utcfromtimestamp(seconds)

        self.times['time_generation_total_formatted'] = self.times['time_generation_total'].apply(
            format_time)

        fig = px.line(
            self.times, y='time_generation_total_formatted', title='Generation Time')
        fig.update_yaxes(title='Time', tickformat='%M:%S')
        return fig

    def plot_best_individual(self, fitness: str, on_val: bool = False):
        """ This method plots the best individual in a given fitness for each generation

        Args:
            - fitness: str
                The name of the fitness to plot
            - on_val: bool (default False)
                Whether to plot the performance on the validation data or on the training data

        Returns:
            - fig: go.Figure
                The plotly figure
        """

        if not fitness in [f.label for f in self.fitness_functions]:
            raise ValueError(
                f'Fitness {fitness} not found in the fitness functions. Please use one of the following: {", ".join([f.label for f in self.fitness_functions])}')

        import plotly.express as px

        branch = 'validation' if on_val else 'training'

        if on_val and not self.best_history.get('validation'):
            logging.warning(
                f'No validation data available. Plotting training data instead')
            branch = 'training'

        data = [
            {'Generation': generation,
             fitness: decompress(self.best_history[branch][generation][fitness]).fitness_validation[fitness] if on_val else decompress(
                 self.best_history[branch][generation][fitness]).fitness[fitness]
             } for generation in self.best_history[branch]
        ]

        # Create a scatter plot with the generation on the x-axis and the fitness on the y-axis
        fig = px.scatter(data, x='Generation', y=fitness,
                         title=f'Best individual on {fitness} for each generation on {branch} set')

        return fig

    def plot_compare_fitness(self, perf_x: str, perf_y: str, pf_rank: int = 1, generation: int = None, on_val: bool = False,
                             marker_dict: Dict[str, Any] = dict(size=5, color='grey'), highlight_best: bool = True,
                             xlim: Tuple[float, float] = None, ylim: Tuple[float, float] = None,
                             xlog: bool = False, ylog: bool = False,
                             figsize: Union[Dict[str,  Any], None] = None, title: str = None):
        """
        This method plots the distribution of two performance metrics for the programs in the first pareto front

        Args:
            - perf_x: str
                The name of the first performance metric to plot
            - perf_y: str
                The name of the second performance metric to plot
            - pf_rank: int (default 1)
                The rank of the pareto front to plot
            - generation: int (default None)
                The generation to plot. If -1, the last generation is plotted
            - on_val: bool (default False)
                Whether to plot the performance on the validation data or on the training data
            - marker_dict: Dict[str, Any] (default dict(size=5,color='grey'))
                The dictionary of the marker to use in the plot
            - highlight_best: bool (default True)
                Whether to highlight the best program in the pareto front
            - xlim: Tuple[float, float] (default None)
                The limits of the x axis
            - ylim: Tuple[float, float] (default None)
                The limits of the y axis
            - xlog: bool (default False)
                Whether to use a log scale for the x axis
            - ylog: bool (default False)
                Whether to use a log scale for the y axis
            - figsize: Union[Dict[str,  Any], None] (default None)
                The size of the figure

        Returns:
            - fig: go.Figure    
                The plotly figure
        """

        if generation and generation != self.generation and pf_rank > 1:
            raise ValueError(
                "Cannot plot pareto fronts beyond the first for historic generations")

        generation = max(self.first_pareto_front_history.keys()
                         ) if not generation else generation

        if not generation:
            iterate_over = self.extract_pareto_front(rank=pf_rank)
        else:
            if not self.first_pareto_front_history.get(generation):
                raise ValueError(
                    f"Generation {generation} not found in the history. Please use one of the following: {', '.join([str(g) for g in self.first_pareto_front_history.keys()])}")

            iterate_over = self.first_pareto_front_history[generation]

        try:
            iterate_over = decompress(iterate_over)
        except TypeError:
            logging.warning(
                f"Legacy version. PF was not compressed. Consider using compress() to save memory and disk space")

        perf_df = pd.DataFrame()
        for index, p in enumerate(iterate_over):
            p: 'Program'
            if on_val:
                for perf in p.fitness_validation:
                    perf_df.loc[index, perf] = p.fitness_validation[perf]
            else:
                for perf in p.fitness:
                    perf_df.loc[index, perf] = p.fitness[perf]

        fig = go.Figure()

        perf_df['color'] = marker_dict.get('color', 'grey')

        if highlight_best:
            perf_x_max = perf_df[perf_x].max()
            perf_x_min = perf_df[perf_x].min()
            perf_y_max = perf_df[perf_y].max()
            perf_y_min = perf_df[perf_y].min()
            perf_x_max_index = perf_df[perf_df[perf_x] == perf_x_max].index[0]
            perf_x_min_index = perf_df[perf_df[perf_x] == perf_x_min].index[0]
            perf_y_max_index = perf_df[perf_df[perf_y] == perf_y_max].index[0]
            perf_y_min_index = perf_df[perf_df[perf_y] == perf_y_min].index[0]
            perf_df.loc[perf_x_max_index, 'color'] = 'red'
            perf_df.loc[perf_y_max_index, 'color'] = 'red'
            perf_df.loc[perf_x_min_index, 'color'] = 'blue'
            perf_df.loc[perf_y_min_index, 'color'] = 'blue'

        # Add the points to the plot
        fig.add_trace(go.Scatter(
            x=perf_df[perf_x],
            y=perf_df[perf_y],
            mode='markers',
            marker=marker_dict,
            marker_color=perf_df['color'],
            # customdata=list(perf_df.columns)
        ))

        fig.update_traces(
            hovertemplate="<br>".join([
                f"{perf_x}: %{{x}}",
                f"{perf_y}: %{{y}}",
                # f"Program: %{{customdata[0]}}",
                # f"Complexity: %{{customdata[1]}}"
            ])
        )

        title = title if title is not None else f"Distribution of {perf_x} and {perf_y} for the {pf_rank} Pareto Front of generation {generation}/{self.generations_to_train} on {'validation' if on_val else 'training'} data"

        fig.update_layout(
            title=title,
            xaxis_title=perf_x,
            yaxis_title=perf_y
        )

        if xlim is not None:
            fig.update_xaxes(range=xlim)
        if ylim is not None:
            fig.update_yaxes(range=ylim)
        if xlog:
            fig.update_xaxes(type="log")
        if ylog:
            fig.update_yaxes(type="log")

        if isinstance(figsize, Dict):
            fig.update_layout(
                autosize=False,
                width=figsize['width'],
                height=figsize['height'],
            )

        # Set background color to white but with the grid lines
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(
                gridcolor="lightgrey",
                gridwidth=2,
            ),
            yaxis=dict(
                gridcolor="lightgrey",
                gridwidth=2,
            ),
        )

        return fig

    def _print_first_pareto_front(self, verbose: int = 3):
        """
        Print best programs

        This method prints the programs of the first pareto front of the current population.

        Args:
            - None
        Returns:
            - None
        """
        if verbose > 1:
            fpf_hypervolume_str = f'and Hypervolume {round(self.fpf_hypervolume, 2)}/{int(self.fpf_hypervolume_reference)}' if self.fpf_hypervolume is not None else ''

            print()
            print(
                f"Average complexity of {round(self.average_complexity,1)} and 1PF of length {len(self.first_pareto_front)} {fpf_hypervolume_str}\n")
        if verbose > 2:
            print(f"\tBest individual(s) in the first Pareto Front")
            for index, p in enumerate(self.first_pareto_front):
                print(f'{index})\t{p.program}')
                print()
                print(f'\tTrain fitness')
                print(f'\t{p.fitness}')
                if hasattr(p, 'fitness_validation') and len(p.fitness_validation) > 0:
                    print(f'\tValidation fitness')
                    print(f'\t{p.fitness_validation}')
                print()

    def spearman_diversity(self, data: Union[dict, pd.Series, pd.DataFrame]) -> float:
        """
        This method computes the diversity of the first pareto front using the spearman correlation

        Args:
            - None

        Returns:
            - diversity: float
                The spearman diversity of the first pareto front

        """

        if len(self.first_pareto_front) == 1:
            self.fpf_spearman_diversity = 0
        else:
            diversities = list()
            for index, program in enumerate(self.first_pareto_front):
                for other_program in self.first_pareto_front[index + 1:]:
                    diversities.append(spearmanr(program.evaluate(
                        data), other_program.evaluate(data)))

            self.fpf_spearman_diversity = np.mean(1 - np.array(diversities))

        self.fpf_stats.loc[self.generation, 'n_individuals'] = len(
            self.first_pareto_front)
        self.fpf_stats.loc[self.generation,
                           'fpf_spearman_diversity'] = self.fpf_spearman_diversity

        return self.fpf_spearman_diversity

    @property
    def summary(self):
        """ Summary of the evolutionary process

        This method returns a summary of the evolutionary process.

        Args:
            - None
        Returns:
            - summary: pd.DataFrame
        """
        istances = list()

        for index, p in enumerate(self.population):
            p: 'Program'
            row = dict()
            row['index'] = index + 1
            row['program'] = p.program
            row['complexity'] = p.complexity
            row['rank'] = p.rank

            for f_k, f_v in p.fitness.items():
                row[f_k] = f_v

            istances.append(row)

        return pd.DataFrame(istances)

    @property
    def _total_time(self) -> float:
        if not 'time_generation_total' in self.times.columns:
            return 0
        tot = self.times['time_generation_total'].sum()
        if pd.isna(tot):
            return 0
        return tot

    def tree_diversity(self) -> float:
        """
        This method computes the tree diversity of the current population

        The tree diversity is computed as the average diversity of
        programs in the population. The diversity is computed as the
        1 - the similarity between two programs. The similarity is
        computed as the number of common sub-trees between two programs
        divided by the total number of sub-trees in the two programs.

        Args:
            - None

        Returns:
            - tree_diversity: float
                The tree diversity of the current population
        """

        if len(self.first_pareto_front) == 1:
            self.fpf_tree_diversity = 0
        else:
            diversities = list()
            for index, program in enumerate(self.first_pareto_front):
                for other_program in self.first_pareto_front[index + 1:]:
                    diversities.append(program.similarity(other_program))

            self.fpf_tree_diversity = np.mean(1 - np.array(diversities))

        self.fpf_stats.loc[self.generation, 'n_individuals'] = len(
            self.first_pareto_front)
        self.fpf_stats.loc[self.generation,
                           'fpf_tree_diversity'] = self.fpf_tree_diversity

        return self.fpf_tree_diversity


def compress(object):
    serialized_data = pickle.dumps(object)
    compressed_data = zlib.compress(serialized_data)
    return compressed_data


def decompress(compressed_data):
    serialized_data = zlib.decompress(compressed_data)
    object = pickle.loads(serialized_data)
    return object
