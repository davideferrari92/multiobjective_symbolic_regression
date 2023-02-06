import logging
import random
import time
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from joblib.parallel import Parallel, delayed

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.multiobjective.hypervolume import _HyperVolume
from symbolic_regression.Program import Program

backend_parallel = 'loky'


class SymbolicRegressor:

    def __init__(
        self,
        client_name: str,
        checkpoint_file: str = None,
        checkpoint_frequency: int = -1,
        const_range: tuple = (0, 1),
        parsimony=0.8,
        parsimony_decay=0.85,
        population_size: int = 300,
        tournament_size: int = 3,
        genetic_operators_frequency: dict = {'crossover': 1, 'mutation': 1},
    ) -> None:
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
        """

        # Regressor Configuration
        self.client_name: str = client_name
        self.checkpoint_file: str = checkpoint_file
        self.checkpoint_frequency: int = checkpoint_frequency
        self.elapsed_time: list = []
        self.features: List = None
        self.operations: List = None
        self.population_size: int = population_size

        # Population Configuration
        self.population: List[Program] = list()
        self.average_complexity: float = None
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
        self.fpf_hypervolume_history: List = list()
        self.fpf_tree_diversity: float = None
        self.fpf_tree_diversity_history: List = list()
        self.generations_to_train: int = None
        self.generation: int = 0
        self.genetic_operators_frequency: dict = genetic_operators_frequency
        self.population: List[Program] = list()
        self.training_duration: int = 0

        self.times = pd.DataFrame()

    def save_model(self, file: str):
        import pickle

        with open(file, "wb") as f:
            pickle.dump(self, f)

    def load_model(self, file: str):
        import pickle

        with open(file, "rb") as f:
            return pickle.load(f)

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

    def compute_performance(self, data: Union[dict, pd.DataFrame, pd.Series]):
        """
        This method computes the performance of each program in the population

        Args:
            - data: Union[dict, pd.DataFrame, pd.Series]
                The data on which the performance is computed
        """
        for p in self.population:
            p.compute_fitness(self.fitness_functions, data)

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
            p1.rank = float('inf')

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

                d = abs(program1.fitness[this_fitness]) - \
                    abs(program2.fitness[this_fitness])

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
            self.population: List[Program] = list(
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
            self.population: List[Program] = list(
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

    def fit(self, data: Union[dict, pd.DataFrame, pd.Series], features: List[str], operations: List[dict], fitness_functions: List[BaseFitness], generations_to_train: int, n_jobs: int = -1, stop_at_convergence: bool = False, verbose: int = 0) -> None:
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
            self._fit(data=data)
        except KeyboardInterrupt:
            self.generation -= 1  # The increment is applied even if the generation is interrupted
            self.status = "Interrupted by KeyboardInterrupt"
            logging.warning(f"Training terminated by a KeyboardInterrupt")

        stop = time.perf_counter()
        self.training_duration += stop - start

    def _fit(self, data: Union[dict, pd.DataFrame, pd.Series]) -> None:
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

        Returns:
            - None

        """
        total_generation_time = 0

        if not self.population:
            before = time.perf_counter()
            logging.info(f"Initializing population")
            self.status = "Generating population"
            self.population: List[Program] = Parallel(
                n_jobs=self.n_jobs,
                backend=backend_parallel)(delayed(self.generate_individual)(
                    data=data,
                    features=self.features,
                    operations=self.operations,
                    const_range=self.const_range,
                    fitness_functions=self.fitness_functions,
                    parsimony=self.parsimony,
                    parsimony_decay=self.parsimony_decay,
                ) for _ in range(self.population_size))

            self.times.loc[self.generation+1,
                           "initialization"] = time.perf_counter() - before
        else:
            logging.info("Fitting with existing population")

        while True:
            if self.generation > 0 and self.generations_to_train <= self.generation:
                logging.info(
                    f"The model already had trained for {self.generation} generations")
                self.status = "Terminated: generations completed"
                return

            for p in self.population:
                p.programs_dominates: List[Program] = list()
                p.programs_dominated_by: List[Program] = list()

            start_time_generation = time.perf_counter()

            if self.generation > 0:
                minutes_total = round(np.sum(self.elapsed_time)/60)
                if minutes_total >= 60:
                    time_total = f"{round(minutes_total/60)}:{round(minutes_total%60):02d} hours"
                elif np.sum(self.elapsed_time) > 60:
                    time_total = f"{minutes_total} mins"
                else:
                    time_total = f"{round(np.sum(self.elapsed_time))} secs"

                seconds_iter = np.mean(self.elapsed_time)
                if seconds_iter >= 60:
                    seconds_iter = f"{round(seconds_iter/60)}:{round(seconds_iter%60):02d} ± {round(np.std(self.elapsed_time)/60)}:{round(np.std(self.elapsed_time)%60):02d} mins"
                else:
                    seconds_iter = f"{round(seconds_iter, 2)} ± {round(np.std(self.elapsed_time), 1)} secs"

                expected_time = np.mean(self.elapsed_time) * (self.generations_to_train - self.generation)/60
                if expected_time >= 60:
                    expected_time = f"{round(expected_time/60)}:{round(expected_time%60):02d} hours"
                else:
                    expected_time = f"{round(expected_time)}:{round((expected_time%1)*60):02d} mins"
            else:
                time_total = f"0 secs"
                seconds_iter = f"0 secs ± 0 secs"
                expected_time = 'Unknown'

            if total_generation_time >= 60:
                generation_time = f"{round(total_generation_time/60)}:{round(total_generation_time%60):02d} mins"
            else:
                generation_time = f"{round(total_generation_time)} secs"

            timing_str = f"Generation: {generation_time} - Average time per generation: {seconds_iter} - Total: {time_total} - Time to completion: {expected_time}"

            self.generation += 1

            print("############################################################")
            print(timing_str)
            print("############################################################")
            print(
                f"Generation {self.generation}/{self.generations_to_train}")

            # Generates the offsprings
            before = time.perf_counter()
            offsprings: List[Program] = Parallel(
                n_jobs=self.n_jobs,
                backend=backend_parallel)(
                    delayed(self._get_offspring)(
                        data, self.genetic_operators_frequency, self.fitness_functions, self.population, self.tournament_size, self.generation
                    ) for _ in range(self.population_size)
            )
            self.times.loc[self.generation,
                           "offsprings_generation"] = time.perf_counter() - before

            logging.info(f"Offsprings generated: {len(offsprings)}")
            self.population += offsprings

            # Removes all duplicated programs in the population
            before_cleaning = len(self.population)

            before = time.perf_counter()
            self.drop_duplicates(inplace=True)
            self.times.loc[self.generation,
                           "duplicated_drop"] = time.perf_counter() - before

            after_drop_duplicates = len(self.population)
            logging.debug(
                f"{before_cleaning-after_drop_duplicates}/{before_cleaning} duplicates programs removed")

            self.times.loc[self.generation,
                           "duplicated_elements_count"] = before_cleaning-after_drop_duplicates
            self.times.loc[self.generation, "duplicated_elements_ratio"] = (
                before_cleaning-after_drop_duplicates)/before_cleaning

            # Removes all non valid programs in the population
            before = time.perf_counter()
            self.drop_invalids(inplace=True)
            self.times.loc[self.generation,
                           "invalids_drop"] = time.perf_counter() - before

            after_cleaning = len(self.population)
            if before_cleaning != after_cleaning:
                logging.debug(
                    f"{after_drop_duplicates-after_cleaning}/{after_drop_duplicates} invalid programs removed")

            # Integrate population in case of too many invalid programs
            self.times.loc[self.generation, "invalid_elements"] = 0
            if len(self.population) < self.population_size * 2:
                before = time.perf_counter()
                missing_elements = 2*self.population_size - \
                    len(self.population)

                logging.info(
                    f"Population of {len(self.population)} elements is less than 2*population_size:{self.population_size*2}. Integrating with {missing_elements} new elements")

                refill = Parallel(
                    n_jobs=self.n_jobs,
                    backend=backend_parallel)(delayed(self.generate_individual)(
                        data=data,
                        features=self.features,
                        operations=self.operations,
                        const_range=self.const_range,
                        fitness_functions=self.fitness_functions,
                        parsimony=self.parsimony,
                        parsimony_decay=self.parsimony_decay,
                    ) for _ in range(missing_elements))

                self.population += refill
                self.times.loc[self.generation,
                               "refill_invalid"] = time.perf_counter() - before

                self.times.loc[self.generation,
                               "invalid_elements_count"] = missing_elements
                self.times.loc[self.generation,
                               "invalid_elements_ratio"] = missing_elements / len(self.population)

            # Calculates the Pareto front
            before = time.perf_counter()
            self._create_pareto_front()
            self.times.loc[self.generation,
                           "pareto_front_computation"] = time.perf_counter() - before

            # Calculates the crowding distance
            before = time.perf_counter()
            self._crowding_distance()
            self.times.loc[self.generation,
                           "crowding_distance_computation"] = time.perf_counter() - before

            self.population.sort(
                key=lambda p: p.crowding_distance, reverse=True)
            self.population.sort(key=lambda p: p.rank, reverse=False)
            self.population = self.population[:self.population_size]

            self.best_program = self.population[0]
            self.best_programs_history.append(self.best_program)
            self.first_pareto_front_history.append(
                list(self.first_pareto_front))

            self.average_complexity = np.mean(
                [p.complexity for p in self.population])

            # Calculates the hypervolume
            before = time.perf_counter()
            self.compute_hypervolume()
            self.times.loc[self.generation,
                           "hypervolume_computation"] = time.perf_counter() - before

            end_time_generation = time.perf_counter()
            self._print_first_pareto_front()

            if any(p.converged for p in self.population):
                if not self.converged_generation:
                    self.converged_generation = self.generation
                logging.info(
                    f"Training converged after {self.converged_generation} generations.")
                if self.stop_at_convergence:
                    print(
                        f"Training converged after {self.converged_generation} generations.")
                    return

            if self.checkpoint_file and self.checkpoint_frequency > 0 and self.generation % self.checkpoint_frequency == 0:
                try:
                    self.save_model(file=self.checkpoint_file)
                except FileNotFoundError:
                    logging.warning(
                        f'FileNotFoundError raised in checkpoint saving')

            # Use generations = -1 to rely only on convergence (risk of infinite loop)
            if self.generations_to_train > 0 and self.generation == self.generations_to_train:
                logging.info(
                    f"Training terminated after {self.generation} generations")
                return

            total_generation_time = round(
                end_time_generation - start_time_generation, 1)
            self.elapsed_time.append(total_generation_time)

            self.times.loc[self.generation,
                           "generation_time"] = total_generation_time

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

    @staticmethod
    def _get_offspring(data: Union[dict, pd.DataFrame, pd.Series], genetic_operators_frequency: Dict[str, float], fitness_functions: List[BaseFitness], population: List[Program], tournament_size: int, generation: int) -> Program:
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

        def _tournament_selection(population: List[Program], tournament_size: int, generation: int) -> Program:
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
                - population: List[Program]
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
                fitness_functions=fitness_functions, data=data)
            return program1

        _offspring: Program = None

        if gen_op == 'crossover':
            program2 = _tournament_selection(
                population=population, tournament_size=tournament_size, generation=generation)
            if program2 is None or not program2.is_valid:
                return program1
            _offspring = program1.cross_over(other=program2, inplace=False)

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

        points = [np.array([p.fitness[ftn.label] for ftn in fitness_to_hypervolume])
                  for p in self.first_pareto_front]

        references = np.array(
            [ftn.hypervolume_reference for ftn in fitness_to_hypervolume])

        try:
            for p_list in points:
                for index, (p_i, r_i) in enumerate(zip(p_list, references)):
                    if p_i > r_i and not p_i == float('inf'):
                        references[index] = p_i + 1e-1

            self.fpf_hypervolume = _HyperVolume(references).compute(points)
            self.fpf_hypervolume_history.append(self.fpf_hypervolume)

        except ValueError:
            self.fpf_hypervolume = np.nan
            self.fpf_hypervolume_history.append(self.fpf_hypervolume)

    def _print_first_pareto_front(self):
        """
        Print best programs

        This method prints the programs of the first pareto front of the current population.

        Args:
            - None
        Returns:
            - None
        """
        if self.verbose > 0:
            print()
            print(
                f"Population of {len(self.population)} elements and average complexity of {round(self.average_complexity,1)} and 1PF hypervolume of {round(self.fpf_hypervolume, 3)}\n")
            print(f"\tBest individual(s) in the first Pareto Front")
            for index, p in enumerate(self.first_pareto_front):
                print(f'{index})\t{p.program}')
                print()
                print(f'\t{p.fitness}')
                print()

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
            self.fpf_tree_diversity_history.append(self.fpf_tree_diversity)
            return self.fpf_tree_diversity

        diversities = list()
        for index, program in enumerate(self.first_pareto_front):
            for other_program in self.first_pareto_front[index + 1:]:
                diversities.append(program.similarity(other_program))

        self.fpf_tree_diversity = 1 - np.mean(diversities)
        self.fpf_tree_diversity_history.append(self.fpf_tree_diversity)

        return self.fpf_tree_diversity
