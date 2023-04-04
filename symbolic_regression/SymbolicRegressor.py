import copy
import logging
import os
import random
import time
from itertools import repeat
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from loky import get_reusable_executor
from scipy.stats import spearmanr

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.multiobjective.hypervolume import _HyperVolume
from symbolic_regression.Population import Population
from symbolic_regression.Program import Program

backend_parallel = 'loky'


class SymbolicRegressor:

    def __init__(self, client_name: str, checkpoint_file: str = None, checkpoint_frequency: int = -1, const_range: tuple = (0, 1), parsimony=0.8, parsimony_decay=0.85, population_size: int = 300, tournament_size: int = 3, genetic_operators_frequency: dict = {'crossover': 1, 'mutation': 1}, statistics_computation_frequency: int = 10) -> None:
        """ This class implements the basic features for training a Symbolic Regression algorithm

        Args:
            - client_name: str
                the name of the client

            - checkpoint_file: str (default: None)
                the file to save the model

            - checkpoint_frequency: int (default: -1)
                the frequency of saving the model

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

        Returns:
            - None
        """

        # Regressor Configuration
        self.client_name: str = client_name
        self.checkpoint_file: str = checkpoint_file
        self.checkpoint_frequency: int = checkpoint_frequency
        self.features: List = None
        self.operations: List = None
        self.population_size: int = population_size

        # Population Configuration
        self._average_complexity: float = None
        self.const_range: tuple = const_range
        self.parsimony: float = parsimony
        self.parsimony_decay: float = parsimony_decay
        self.tournament_size: int = tournament_size

        # Training Configuration
        self.best_program = None
        self.best_programs_history: List = list()
        self.converged_generation: int = None
        self.fitness_functions: List[BaseFitness] = None
        self.first_pareto_front_history: List = list()
        self.fpf_hypervolume: float = None
        self.fpf_hypervolume_reference: float = None
        self.fpf_tree_diversity: float = None
        self.fpf_spearman_diversity: float = None
        self.fpf_stats: pd.DataFrame = pd.DataFrame()
        self.generations_to_train: int = None
        self.generation: int = 0
        self.genetic_operators_frequency: dict = genetic_operators_frequency
        self.statistics_computation_frequency: int = statistics_computation_frequency
        self.population: Population = Population()
        self.training_duration: int = 0

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

    def save_model(self, file: str):
        import json
        import pickle

        # Dump this object in a pickle file
        with open(file, "wb") as f:
            for p in self.population:
                p.programs_dominated_by: List[Program] = list()
                p.programs_dominates: List[Program] = list()
            pickle.dump(self, f)

        # Dump the self.metadata in a json file
        with open(file + ".metadata.json", "w") as f:
            json.dump(self.metadata, f)

    def load_model(self, file: str):
        import pickle

        with open(file, "rb") as f:
            sr: SymbolicRegressor = pickle.load(f)

        return sr

    @property
    def average_complexity(self):
        return self._average_complexity

    @average_complexity.getter
    def average_complexity(self):
        return np.mean([p.complexity for p in self.population]) if len(self.population) > 0 else 0

    @property
    def best_history(self):
        """
        Best history

        This method returns a summary of the best programs found during the evolutionary process.

        Args:
            - None
        Returns:
            - best_history: pd.DataFrame
        """
        istances = list()

        for index, p in enumerate(self.best_programs_history):
            row = dict()
            row['generation'] = index + 1
            row['program'] = p.program
            row['complexity'] = p.complexity
            row['rank'] = p.rank

            for f_k, f_v in p.fitness.items():
                row[f_k] = f_v

            istances.append(row)

        return pd.DataFrame(istances)

    def compute_hypervolume(self):
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
            - None

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

    def compute_performance(self, fitness_functions: List[BaseFitness] = None, data: Union[dict, pd.DataFrame, pd.Series] = None, validation: bool = False):
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
        """
        for p in self.population:
            p.compute_fitness(
                fitness_functions=self.fitness_functions if fitness_functions is None else fitness_functions, data=data, validation=validation)

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

        for index, p in enumerate(self.population):
            if p.is_valid and not p._is_duplicated:
                for p_confront in self.population[index + 1:]:
                    if p.is_duplicate(p_confront):
                        p_confront._is_duplicated = True  # Makes p.is_valid = False

        if inplace:
            self.population: Population = Population(
                filter(lambda p: p._is_duplicated == False, self.population))
            return self.population

        return list(
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
            before = time.perf_counter()
            logging.info(f"Initializing population")
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
                )
            ))

            self.times.loc[self.generation+1,
                           "time_initialization"] = time.perf_counter() - before
        else:
            logging.info("Fitting with existing population")

        while True:
            if self.generation > 0 and self.generations_to_train <= self.generation:
                logging.info(
                    f"The model already had trained for {self.generation} generations")
                self.status = "Terminated: generations completed"
                return

            start_time_generation = time.perf_counter()

            for p in self.population:
                p.programs_dominates: List[Program] = list()
                p.programs_dominated_by: List[Program] = list()

            if self.generation > 0:
                minutes_total = round(self._total_time/60)
                if minutes_total >= 60:
                    time_total = f"{round(minutes_total/60)}:{round(minutes_total%60):02d} hours"
                elif self._total_time > 60:
                    time_total = f"{minutes_total} mins"
                else:
                    time_total = f"{round(self._total_time)} secs"

                seconds_iter_avg = self.times['time_generation_total'].tail(
                    5).median()
                seconds_iter_std = self.times['time_generation_total'].tail(
                    5).std()
                if seconds_iter_avg >= 60 and not pd.isna(seconds_iter_std):
                    time_per_generation = f"{round(seconds_iter_avg//60)}:{round(seconds_iter_avg%60):02d} ± {round(seconds_iter_std//60)}:{round(seconds_iter_std%60):02d} mins"
                else:
                    time_per_generation = f"{round(seconds_iter_avg, 2)} ± {round(seconds_iter_std, 1)} secs"

                expected_time = self.times['time_generation_total'].tail(
                    5).median() * (self.generations_to_train - self.generation) / 60
                if pd.isna(expected_time):
                    expected_time = 'Unknown'
                elif expected_time >= 60:
                    expected_time = f"{round(expected_time//60)}:{round(expected_time%60):02d} hours"
                else:
                    expected_time = f"{round(expected_time)}:{round((expected_time%1)*60):02d} mins"
            else:
                time_total = f"0 secs"
                time_per_generation = f"0 secs ± 0 secs"
                expected_time = 'Unknown'

            if total_generation_time >= 60:
                generation_time = f"{round(total_generation_time//60)}:{round(total_generation_time%60):02d} mins"
            else:
                generation_time = f"{round(total_generation_time)} secs"

            timing_str = f"Generation {generation_time} - On average: {time_per_generation} - Total: {time_total} - To completion: {expected_time}"

            self.generation += 1

            if self.verbose > 0:
                print("#" * len(timing_str))
                print(timing_str)
                print("#" * len(timing_str))
            print(
                f"{self.client_name}: starting generation {self.generation}/{self.generations_to_train}")

            before = time.perf_counter()

            offsprings: List[Program] = list()
            procs: List[Process] = list()
            queue: Queue = Queue(maxsize=self.population_size)
            for _ in range(jobs):
                proc = Process(
                    target=self._get_offspring_batch,
                    args=(
                        data,
                        self.genetic_operators_frequency,
                        self.fitness_functions,
                        self.population.as_binary(),
                        self.tournament_size,
                        self.generation,
                        int(self.population_size/jobs),
                        queue
                    )
                )
                procs.append(proc)

            for proc in procs:
                proc.start()

            q_size = 0
            while q_size < self.population_size:
                q_size = queue.qsize()
                if self.verbose > 0:
                    _elapsed = max(1, int(round(time.perf_counter() - before)))
                    print(
                        f'Offsprings generated: {q_size}/{self.population_size} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s)', end='\r', flush=True)
                time.sleep(.2)

            else:
                q_size = queue.qsize()
                if self.verbose > 0:
                    _elapsed = max(1, int(round(time.perf_counter() - before)))
                    print(
                        f'Offsprings generated: {q_size}/{self.population_size} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s). Completed!', flush=True)
                for p in procs:
                    p.join(timeout=.5)

            for _ in range(self.population_size):
                offsprings.append(queue.get())

            for proc in procs:
                proc.kill()

            self.times.loc[self.generation,
                           "time_offsprings_generation"] = time.perf_counter() - before

            self.population = Population(self.population + offsprings)

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
                before = time.perf_counter()
                missing_elements = 2*self.population_size - \
                    len(self.population)

                logging.info(
                    f"Population of {len(self.population)} elements is less than 2*population_size:{self.population_size*2}. Integrating with {missing_elements} new elements")

                refill: List[Program] = list()
                procs: List[Process] = list()
                queue: Queue = Queue(maxsize=missing_elements)
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
                            int(missing_elements/jobs),
                            queue
                        )
                    )
                    procs.append(proc)
                    proc.start()

                q_size = 0
                while q_size < missing_elements:
                    q_size = queue.qsize()
                    if self.verbose > 0:
                        _elapsed = max(
                            1, int(round(time.perf_counter() - before)))
                        print(
                            f'Duplicates/invalid refilled: {q_size}/{missing_elements} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s)', end='\r', flush=True)
                    time.sleep(.5)

                else:
                    if self.verbose > 0:
                        _elapsed = max(
                            1, int(round(time.perf_counter() - before)))
                        print(
                            f'Duplicates/invalid refilled: {q_size}/{missing_elements} ({_elapsed} s, {round(q_size/_elapsed, 2)} /s). Completed!', flush=True)
                    for p in procs:
                        p.join(timeout=.2)

                for _ in range(missing_elements):
                    refill.append(queue.get())

                for proc in procs:
                    proc.kill()

                self.population: Population = Population(
                    self.population + refill)

                # exludes every program in refill with an empty fitness
                self.population = [
                    p for p in self.population if not p._has_incomplete_fitness]

                self.times.loc[self.generation,
                               "time_refill_invalid"] = time.perf_counter() - before
                self.times.loc[self.generation,
                               "count_invalid_elements"] = missing_elements
                self.times.loc[self.generation,
                               "ratio_invalid_elements"] = missing_elements / len(self.population)

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

            self.best_program = self.population[0]
            self.best_programs_history.append(self.best_program)
            self.first_pareto_front_history.append(
                list(self.first_pareto_front))

            self.times.loc[self.generation,
                           "count_average_complexity"] = self.average_complexity

            if any(p.converged for p in self.population):
                if not self.converged_generation:
                    self.converged_generation = self.generation
                if self.verbose > 0:
                    print(
                        f"Training converged after {self.converged_generation} generations.")

            if (self.generation == 1) or (self.statistics_computation_frequency == -1 and (self.generation == self.generations_to_train or self.converged_generation)) or (self.statistics_computation_frequency > 0 and self.generation % self.statistics_computation_frequency == 0):
                if self.verbose > 0:
                    print(
                        f'Computing statistics for generation {self.generation}')
                # Calculates the hypervolume
                before = time.perf_counter()
                self.compute_hypervolume()
                self.times.loc[self.generation,
                               "time_hypervolume_computation"] = time.perf_counter() - before

                before = time.perf_counter()
                self.tree_diversity()
                self.times.loc[self.generation,
                               "time_tree_diversity_computation"] = time.perf_counter() - before

                before = time.perf_counter()
                self.spearman_diversity(data=data)
                self.times.loc[self.generation,
                               "time_spearman_diversity_computation"] = time.perf_counter() - before

                if val_data is not None:
                    self.compute_performance(
                        fitness_functions=self.fitness_functions, data=val_data, validation=True)

            end_time_generation = time.perf_counter()
            self._print_first_pareto_front(verbose=self.verbose)

            total_generation_time = end_time_generation - start_time_generation
            self.times.loc[self.generation,
                           "time_generation_total"] = total_generation_time

            if self.checkpoint_file and self.checkpoint_frequency > 0 and self.generation % self.checkpoint_frequency == 0 or self.generation == self.generations_to_train or self.generations_to_train:
                try:
                    self.save_model(file=self.checkpoint_file)
                except FileNotFoundError:
                    logging.warning(
                        f'FileNotFoundError raised in checkpoint saving: {self.checkpoint_file}')

            # Use generations = -1 to rely only on convergence (risk of infinite loop)
            if self.generations_to_train > 0 and self.generation == self.generations_to_train:
                print(
                    f"Training completed {self.generation}/{self.generations_to_train} generations")
                return

            if self.converged_generation and self.stop_at_convergence:
                if val_data is not None:
                    self.compute_performance(
                        fitness_functions=self.fitness_functions, data=val_data, validation=True)
                print(
                    f"Training converged after {self.converged_generation} generations and requested to stop.")
                return

    def generate_individual(self, data: Union[dict, pd.DataFrame, pd.Series], features: List[str], operations: List[dict], fitness_functions: List[BaseFitness], const_range: tuple = (0, 1), parsimony: float = 0.8, parsimony_decay: float = 0.85) -> Program:
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

        Returns:
            - p: Program
                The generated program

        """
        new_p = Program(features=features, operations=operations, const_range=const_range,
                        parsimony=parsimony, parsimony_decay=parsimony_decay)
        new_p.init_program()
        new_p.compute_fitness(fitness_functions=fitness_functions, data=data)
        return new_p

    def generate_individual_batch(self, data: Union[dict, pd.DataFrame, pd.Series], features: List[str], operations: List[dict], fitness_functions: List[BaseFitness], const_range: tuple = (0, 1), parsimony: float = 0.8, parsimony_decay: float = 0.85, batch_size: int = 1000, queue: Queue = None) -> Program:
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

        Returns:
            - p: Program
                The generated program

        """

        new_ps: List[Program] = list()

        submitted = 0
        while submitted < batch_size * 3:

            new_p = self.generate_individual(data=data, features=features, operations=operations,
                                             fitness_functions=fitness_functions, const_range=const_range,
                                             parsimony=parsimony, parsimony_decay=parsimony_decay)

            if new_p._has_incomplete_fitness:
                continue

            if queue is not None:
                queue.put(new_p)
            else:
                new_ps.append(new_p)

            submitted += 1

        return new_ps

    @staticmethod
    def _get_offspring(data: Union[dict, pd.DataFrame, pd.Series], genetic_operators_frequency: Dict[str, float], fitness_functions: List[BaseFitness], population: Population, tournament_size: int, generation: int) -> Program:
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

        import signal
        from contextlib import contextmanager

        class TimeoutException(Exception):
            pass

        @contextmanager
        def time_limit(seconds):
            def signal_handler(signum, frame):
                raise TimeoutException("Timed out!")
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)

        # Select the genetic operation to apply
        # The frequency of each operation is determined by the dictionary
        # genetic_operators_frequency. The higher the number the likelier the operation will be chosen.

        timeout = 20
        try:
            with time_limit(timeout):
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
                        fitness_functions=fitness_functions, data=data)
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
                    fitness_functions=fitness_functions, data=data)

                # Reset the hash to force the re-computation
                _offspring._hash = None

                return _offspring

        except TimeoutException:
            logging.debug(
                f'Genetic operation {gen_op} timed out after {timeout} seconds')
            return program1

    def _get_offspring_batch(self, data: Union[dict, pd.DataFrame, pd.Series], genetic_operators_frequency: Dict[str, float], fitness_functions: List[BaseFitness], population: Population, tournament_size: int, generation: int,
                             batch_size: int, queue: Queue = None) -> Program:

        population = population.as_program()
        offsprings: List[Program] = list()

        submitted = 0
        while submitted < batch_size * 3:

            offspring = self._get_offspring(data=data, genetic_operators_frequency=genetic_operators_frequency,
                                            fitness_functions=fitness_functions, population=population, tournament_size=tournament_size, generation=generation)

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
            'checkpoint_file': self.checkpoint_file,
            'checkpoint_frequency': self.checkpoint_frequency,
            'client_name': self.client_name,
            'const_range': self.const_range,
            'converged_generation': self.converged_generation,
            'fpf_hypervolume': self.fpf_hypervolume,
            'fpf_tree_diversity': self.fpf_tree_diversity,
            'generation': self.generation,
            'generations_to_train': self.generations_to_train,
            'genetic_operators_frequency': self.genetic_operators_frequency,
            'parsimony_decay': self.parsimony_decay,
            'parsimony': self.parsimony,
            'population_size': self.population_size,
            'total_time': self._total_time,
            'tournament_size': self.tournament_size,
        }

        return metadata

    def _print_first_pareto_front(self, verbose: int = 2):
        """
        Print best programs

        This method prints the programs of the first pareto front of the current population.

        Args:
            - None
        Returns:
            - None
        """
        if verbose > 0:
            fpf_hypervolume_str = f'and Hypervolume {round(self.fpf_hypervolume, 2)}/{int(self.fpf_hypervolume_reference)}' if self.fpf_hypervolume is not None else ''

            print()
            print(
                f"Average complexity of {round(self.average_complexity,1)} and 1PF of length {len(self.first_pareto_front)} {fpf_hypervolume_str}\n")
        if verbose > 1:
            print(f"\tBest individual(s) in the first Pareto Front")
            for index, p in enumerate(self.first_pareto_front):
                print(f'{index})\t{p.program}')
                print()
                print(f'\t{p.fitness}')
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
