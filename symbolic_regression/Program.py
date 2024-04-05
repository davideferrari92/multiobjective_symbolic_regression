import copy
import logging
import random
import signal
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import sympy
from joblib import Parallel, delayed
from pytexit import py2tex

from symbolic_regression.multiobjective.fitness.Base import BaseFitness
from symbolic_regression.multiobjective.hypervolume import _HyperVolume
from symbolic_regression.multiobjective.optimization import (ADAM, ADAM2FOLD, SCIPY,
                                                             SGD)
from symbolic_regression.Node import (FeatureNode, InvalidNode, Node,
                                      OperationNode)
from symbolic_regression.operators import (OPERATOR_ADD, OPERATOR_MUL,
                                           OPERATOR_POW)


class Program:
    """ A program is a tree that represent an arithmetic formula

    The nodes are the operations, each of which has an arbitrary number of operands
    (usually 1 or 2). An operand can be another operation, or a terminal node.
    A terminal node in a feature taken from the training dataset or a numerical
    constant.

    The generation of a tree follow a genetic approach in which, starting from a root
    node, the choice of which type of node will be the operand is modulated by a random
    distribution. Based on how likely the next node will be an operation or a terminal
    node, the tree will become deeper and so the formula more complex.
    The choice of the terminal node is determined by a random distribution as well to
    modulate how likely a random feature from the dataset will be chosen instead of a
    random numerical constant.

    The program can be evaluated and also printed.
    The evaluation execute recursively the defined operation using the values
    provided by the dataset. In the same way, printing the tree will recursively print
    each operation and terminal node based on its formatting pattern or numerical value.

    According to the genetic approach, the tree can perform a mutation or a cross-over.
    A mutation is a modification of a random inner node of the tree with a newly generated
    subtree. The cross-over is the replacement of a random subtree of this program with
    a random subtree of another given program.

    The program need to be provided with a set of possible operations, and a set of
    possible terminal node features from which to choose. Also, if numerical constant are
    desired, also the range from which to choose the constant must be provided.

    """

    def __init__(self, operations: List[Dict], features: List[str], const_range: Tuple = (0, 1), program: Node = None, parsimony: float = .8, parsimony_decay: float = .85) -> None:
        """


        Args:
            - operations: List[Dict]
                List of possible operations. Each operation is a dictionary with the following keys:
                    - func: callable
                        The function that will be executed by the operation
                    - arity: int
                        The number of operands of the operation
                    - format_str: str
                        The format pattern of the operation when printed in a string formula
                    - format_tf: str
                        The format pattern of the operation when printed in a tensorflow formula
                    - symbol: str
                        The symbol of the operation

            - features: List[str]
                List of possible features from which to choose the terminal nodes

            - const_range: Tuple  (default: (0, 1))
                Range from which to choose the numerical constants.

            - program: Node  (default: None)
                The root node of the tree. If None, a new tree will be generated.

            - parsimony: float  (default: .8)
                The parsimony coefficient. It is used to modulate the depth of the program.
                Use values between 0 and 1. The higher the value, the deeper the program.

            - parsimony_decay: float  (default: .85)
                The decay of the parsimony coefficient. It is used to modulate the depth of the program.
                Use values between 0 and 1. The higher the value, the deeper the program.

        """

        self.operations: List[Dict] = operations
        self.features: List[str] = features
        self.const_range: Tuple = const_range
        self._constants: List = list()
        self.converged: bool = False

        # Operational attributes
        self._override_is_valid: bool = True
        self._is_duplicated: bool = False
        self._program_depth: int = 0
        self._complexity: int = 0
        self._exclusive_hypervolume: float = np.nan

        # Pareto Front Attributes
        self.rank: int = np.inf
        self.programs_dominates: List[Program] = list()
        self.programs_dominated_by: List[Program] = list()
        self.crowding_distance: float = 0
        self.program_hypervolume: float = np.nan
        self._hash: List[int] = None

        self.parsimony: float = parsimony
        self._parsimony_bkp: float = parsimony
        self.parsimony_decay: float = parsimony_decay

        self.is_logistic: bool = False
        self.is_affine: bool = False

        if program:
            self.program: Node = program
        else:
            self.program: Node = InvalidNode()
            self.fitness: Dict[str, float] = dict()
            self.fitness_validation: Dict[str, float] = dict()
            self.fitness_functions: List[BaseFitness] = list()
            self.is_fitness_to_minimize: Dict = dict()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))

        result.programs_dominated_by = list()
        result.programs_dominates = list()

        result.program_hypervolume = np.nan
        result._override_is_valid = True
        return result

    def __lt__(self, other: 'Program') -> bool:
        """
        Overload of the less than operator. It is used to compare two programs.

        A program is less than another if it has a lower or equal rank and a higher or equal crowding distance.

        Args:
            - other: Program
                The other program to compare with

        Returns:
            - bool
                True if the program is less than the other program, False otherwise
        """

        rank_dominance = self.rank <= other.rank
        crowding_distance_dominance = self.crowding_distance >= other.crowding_distance

        return rank_dominance and crowding_distance_dominance

    def __len__(self) -> int:
        return self.complexity

    @property
    def complexity(self) -> int:
        """ The complexity of a program is the number of nodes (OperationNodes or FeatureNodes)
        """
        return self._complexity

    @complexity.getter
    def complexity(self, base_complexity=0):
        return self.program._get_complexity(base_complexity)

    @property
    def program_depth(self):
        """ The depth of a program is the length of the deepest branch
        """
        return self._program_depth

    @program_depth.getter
    def program_depth(self, base_depth=0):
        return self.program._get_depth(base_depth)

    @property
    def operations_used(self):
        """ This allow to get a list of all unique operations used in a program
        """
        return self._operations_used

    @operations_used.getter
    def operations_used(self):
        return self.program._get_operations(base_operations_used={})

    @property
    def all_operations(self):
        """ This allow to get a list of all the operations used in a program
        """
        return self._all_operations

    @all_operations.getter
    def all_operations(self):
        return self.program._get_all_operations(all_operations=[])

    @property
    def exclusive_hypervolume(self):
        """ This allow to get the exclusive hypervolume of the program
        """
        return self._exclusive_hypervolume

    @exclusive_hypervolume.setter
    def exclusive_hypervolume(self, value):
        self._exclusive_hypervolume = value

    @property
    def features_used(self):
        """ This allow to get all the unique features used in the tree
        """
        return self._features_used

    @features_used.getter
    def features_used(self):
        return self.program._get_features(features_list=[])

    def _internal_bootstrap(self, data: Union[dict, pd.Series, pd.DataFrame], target: str, weights: str, constants_optimization: str, constants_optimization_conf: dict, frac: float = .6) -> 'Program':
        bs_data = data.sample(frac=frac, replace=True)
        # !!! Weights should now be adapted to the bs_data target distribution

        # Increase the epochs to be used for the bootstrap
        constants_optimization_conf = copy.deepcopy(
            constants_optimization_conf)
        constants_optimization_conf['epochs'] = max(
            200, constants_optimization_conf['epochs'])

        recalibrated = copy.deepcopy(self)
        recalibrated.set_constants(
            new=list(np.random.uniform(
                low=self.const_range[0],
                high=self.const_range[1],
                size=len(self.get_constants())
            ))
        )
        return recalibrated.optimize(
            data=bs_data,
            target=target,
            weights=weights,
            constants_optimization=constants_optimization,
            constants_optimization_conf=constants_optimization_conf,
            inplace=False
        ).get_constants(return_objects=False)

    def bootstrap(self, data: Union[dict, pd.Series, pd.DataFrame], target: str, weights: str, constants_optimization: str, constants_optimization_conf: dict, inplace: bool = False, k: int = 1000, frac: float = .6) -> 'Program':
        """ This method allow to bootstrap a program.

        The bootstrapping is a statistical method to estimate the uncertainty of a model.
        We optimize the constants k times over a random fraction frac of the dataset.
        The result is a list of k constants for each constant of the program. These are
        then used to create a new program with the same structure of the original one
        but with the new constants. It leverages the confidence intervals capability of
        FeatureNode to receive a list of values for each constant.

        Args:
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data on which the program will be evaluated
            - target: str
                The target variable of the data
            - weights: str
                The weights of the data
            - constants_optimization: str
                The constants optimization method to use
            - constants_optimization_conf: dict
                The configuration of the constants optimization method
            - inplace: bool  (default=False)
                If True, the method will modify the program in place, otherwise it will return a new program
            - k: int  (default=1000)
                The number of bootstrap iterations
            - frac: float  (default=.6)
                The fraction of the data to use for each bootstrap iteration
        """

        n_constants = len(self.get_constants(return_objects=False))
        n_features_used = len(self.features_used)
        if not isinstance(self.program, FeatureNode) and n_constants > 0 and n_features_used > 0:
            if inplace:
                return self
            else:
                return copy.deepcopy(self)

        bootstrapped_constants: List[float] = Parallel(n_jobs=-1)(delayed(self._internal_bootstrap)(
            data=data,
            target=target,
            weights=weights,
            constants_optimization=constants_optimization,
            constants_optimization_conf=constants_optimization_conf,
            frac=frac
        ) for _ in range(k))

        if inplace:
            bootstrapped_program = self
        else:
            bootstrapped_program = copy.deepcopy(self)

        bootstrapped_program.set_constants(
            [list(new_constants) for new_constants in zip(*bootstrapped_constants)])

        return bootstrapped_program

    def constants_confidence_intervals_overlap(self, other: 'Program') -> bool:

        if not len(self.get_constants(return_objects=False)) == len(other.get_constants(return_objects=False)):
            raise ValueError(
                "The two programs have different number of constants")

        for self_constant, other_constant in zip(self.get_constants(return_objects=True), other.get_constants(return_objects=True)):
            if self_constant.feature_confidence_intervals == [np.nan, np.nan] or other_constant.feature_confidence_intervals == [np.nan, np.nan]:
                return False
            overlap = max(
                0,
                min(self_constant.feature_confidence_intervals[1], other_constant.feature_confidence_intervals[1]) -
                max(self_constant.feature_confidence_intervals[0],
                    other_constant.feature_confidence_intervals[0])
            )
            if overlap <= 1e-7:
                return False

        return True

    def compute_fitness(self, fitness_functions: List[BaseFitness], data: Union[dict, pd.Series, pd.DataFrame], validation: bool = False, validation_federated: bool = False, simplify: bool = True) -> None:
        """ This function evaluate the fitness of the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - fitness_functions: List[BaseFitness]
                The fitness functions to evaluate
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data on which the program will be evaluated
            - validation: bool  (default: False)
                If True, the fitness will be computed on the validation data without optimization
            - validation_federated: bool  (default: False)
                If True, the fitness will be computed on the training and validation data without optimization (only for federated training)
            - simplify: bool  (default: True)
                If True, the program will be simplified before computing the fitness

        Returns:
            - None
        """

        # We store the fitness functions in the program object
        # as we need them in other parts of the code (e.g. in the hypervolume computation)
        self.fitness_functions: List[BaseFitness] = fitness_functions
        for ftn in self.fitness_functions:
            for ftn2 in self.fitness_functions:
                if ftn.label == ftn2.label and ftn != ftn2:
                    raise ValueError(
                        f"Fitness function with label {ftn.label} already used")

        if validation or validation_federated:
            simplify = False
            self.fitness_validation: Dict[str, float] = dict()
        else:
            self.fitness: Dict[str, float] = dict()

        self.is_fitness_to_minimize: Dict[str, bool] = dict()

        for ftn in self.fitness_functions:
            self.is_fitness_to_minimize[ftn.label] = ftn.minimize

        if simplify:
            try:
                self.simplify(inplace=True)
            except ValueError:
                self._override_is_valid = False
                if validation:
                    self.fitness_validation = {
                        ftn.label: np.inf for ftn in self.fitness_functions}
                elif not validation:
                    self.fitness = {
                        ftn.label: np.inf for ftn in self.fitness_functions}
                return

        _converged: List[bool] = list()

        for ftn in self.fitness_functions:
            try:
                """ We don't want optimization of the constants in the validation stage, both local (validation)
                and federated (validation_federated)
                """
                fitness_value = round(ftn.evaluate(
                    program=self, data=data, validation=validation or validation_federated), 5)
            except KeyError:
                fitness_value = np.inf

            if pd.isna(fitness_value):
                fitness_value = np.inf

            if validation:
                self.fitness_validation[ftn.label] = fitness_value
            else:
                self.fitness[ftn.label] = fitness_value

                if ftn.minimize and isinstance(ftn.convergence_threshold, (int, float)):
                    if fitness_value <= ftn.convergence_threshold:
                        _converged.append(True)
                    else:
                        _converged.append(False)

                    # Only if all the fitness functions have converged, then the program has converged
                    self.converged = all(_converged) if len(
                        _converged) > 0 else False

        if not validation:
            self._compute_hypervolume()

    def _compute_hypervolume(self) -> float:
        """
        This method return the hypervolume of the program

        The hypervolume is the volume occupied by the fitness space by the program.
        It can be of any dimension. We allow to compute the hypervolume only if the
        fitness functions are set to be minimized, otherwise we assume that the fitness
        are computed only for comparison purposes and not for optimization.

        Args:
            - None

        Returns:
            - float
                The hypervolume of the program.
        """

        if not self.program.is_valid:
            return np.nan

        fitness_to_hypervolume: List[BaseFitness] = list()
        for fitness in self.fitness_functions:
            if fitness.hypervolume_reference and fitness.minimize:
                fitness_to_hypervolume.append(fitness)

        if not fitness_to_hypervolume:
            return np.nan

        points = [np.array([self.fitness[ftn.label]
                            for ftn in fitness_to_hypervolume])]
        references = np.array(
            [ftn.hypervolume_reference for ftn in fitness_to_hypervolume])

        self.program_hypervolume = _HyperVolume(references).compute(points)

    def predict(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This function predict the value of the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - data: dict, pd.Series, pd.DataFrame
                The data on which the program will be evaluated

        Returns:
            - int, float
                The result of the prediction
        """
        return self.evaluate(data=data)

    def predict_proba(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This function predict the value of the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - data: dict, pd.Series, pd.DataFrame
                The data on which the program will be evaluated

        Returns:
            - int, float
                The result of the prediction
        """
        return self.evaluate(data=data, logistic=True)

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame], logistic: bool = False, threshold: float = None) -> Union[int, float]:
        """ This function evaluate the program on the given data.
        The data can be a dictionary, a pandas Series or a pandas DataFrame.

        Args:
            - data: dict, pd.Series, pd.DataFrame
                The data on which the program will be evaluated
            - logistic: bool  (default: False)
                If True, the program will be evaluated using the logistic function

        Returns:
            - int, float
                The result of the evaluation
        """
        if not self.is_valid:
            return np.nan

        if logistic:
            if isinstance(threshold, float) and 0 <= threshold <= 1:
                return np.where(self.to_logistic(inplace=False).evaluate(data=data) > threshold, 1, 0)

            return self.to_logistic(inplace=False).evaluate(data=data)

        return self.program.evaluate(data=data)

    def _generate_tree(self, depth=-1, parsimony: float = .8, parsimony_decay: float = .85, father: Union[Node, None] = None, force_constant: bool = False) -> Node:
        """ This function generate a tree of a given depth.

        Args:
            - depth: int  (default=-1)
                The depth of the tree to generate. If -1, the depth will be randomly generated

            - parsimony: float  (default=.8)
                The parsimony coefficient. This modulates the depth of the generated tree.
                Use values between 0 and 1; the closer to 1, the deeper the tree will be.

            - parsimony_decay: float  (default=.85)
                The parsimony decay coefficient. This value is multiplied to the parsimony coefficient
                at each depth level. Use values between 0 and 1; the closer to 1, the quicker the
                parsimony coefficient will decay and therefore the shallower the tree will be.
                Use a lower value to prevent the tree from exploding and reaching a RecursionError.

            - father: Node  (default=None)
                The father of the node to generate. If None, the node will be the root of the tree.

            - force_constant: bool  (default=False)
                If True, the next node will be a constant node.

        Returns:
            - Node
                The generated tree. It's a recursive process and the returned node is the root of the tree.
        """
        def gen_operation(operation_conf: Dict, father: Union[Node, None] = None):
            return OperationNode(operation=operation_conf['func'],
                                 arity=operation_conf['arity'],
                                 format_str=operation_conf.get('format_str'),
                                 format_tf=operation_conf.get('format_tf'),
                                 symbol=operation_conf.get('symbol'),
                                 format_diff=operation_conf.get(
                                     'format_diff', operation_conf.get('format_str')),
                                 father=father)

        def gen_feature(feature: str, father: Union[Node, None], is_constant: bool = False):
            return FeatureNode(feature=feature, father=father, is_constant=is_constant)

        self._override_is_valid = True

        # We can either pass dedicated parsimony and parsimony_decay values or use the ones
        # defined in the class
        if not parsimony:
            parsimony = self.parsimony
        if not parsimony_decay:
            parsimony_decay = self.parsimony_decay

        selector_operation = random.random()
        if ((selector_operation < parsimony and depth == -1) or (depth > 0)) and not force_constant:
            # We generate a random operation

            operation = random.choice(self.operations)
            node = gen_operation(operation_conf=operation, father=father)

            new_depth = -1 if depth == -1 else depth - 1

            for i in range(node.arity):
                if operation == OPERATOR_POW and i == 1:
                    # In case of the power operator, the second operand must be a constant
                    # We do not want to generate a tree for the second operand otherwise it may
                    # generate an unrealistic mathematical model
                    force_constant = True

                node.add_operand(
                    self._generate_tree(depth=new_depth,
                                        father=node,
                                        parsimony=parsimony * parsimony_decay,
                                        parsimony_decay=parsimony_decay,
                                        force_constant=force_constant))
            force_constant = False

        else:
            # We generate a random feature

            # The probability to get a feature from the training data is
            # (n-1) / n where n is the number of features.
            # Otherwise a constant value will be generated.

            selector_feature = random.random()
            # To prevent constant features to be underrepresented, we set a threshold
            threshold = max(.1, (1 / len(self.features)))

            if selector_feature > threshold and not force_constant:
                # A Feature from the dataset

                feature = random.choice(self.features)

                node = gen_feature(
                    feature=feature, father=father, is_constant=False)

            else:
                # Generate a constant

                feature = random.uniform(
                    self.const_range[0], self.const_range[1])

                # Arbitrary rounding of the generated constant

                node = gen_feature(
                    feature=feature, father=father, is_constant=True)

        return node

    def get_constants(self, return_objects: bool = False):
        """
        This method allow to get all constants used in a tree.

        The constants are used for the neuronal-based constants optimizer; it requires
        all constants to be in a fixed order explored by a DFS descent of the tree.

        Args:
            - return_objects: bool  (default=False)
                If True, the method will return a list of FeatureNode objects instead of a list of
                feature names.

        Returns:
            - List[FeatureNode]
                A list of FeatureNode objects representing the constants used in the tree.
        """
        to_return = None
        if isinstance(self.program, OperationNode):
            to_return = self.program._get_constants(const_list=[])

        # Only one constant FeatureNode
        elif self.program.is_constant:
            to_return = [self.program]

        else:
            # Only one non-constant FeatureNode
            to_return = list()

        for index, constant in enumerate(to_return):
            self._set_constants_index(constant=constant, index=index)

        if not return_objects:
            to_return = [f.feature for f in to_return]

        return to_return

    def get_features(self, return_objects: bool = False):
        """
        This method recursively explore the tree and return a list of unique features used.

        Args:
            - return_objects: bool  (default=False)
                If True, the method will return a list of FeatureNode objects instead of a list of
                feature names.

        Returns:
            - List[str] or List[FeatureNode]
                A list of unique features used in the tree.
        """
        if isinstance(self.program, OperationNode):
            return self.program._get_features(features_list=[],
                                              return_objects=return_objects)

        # Only one non-constant FeatureNode
        elif not self.program.is_constant:
            return [self.program]

        # Case for programs of only one constant FeatureNode.
        # Use get_constants() to have a list of all constant FeatureNode objects
        return []

    @property
    def _has_incomplete_fitness(self):
        """
        This method return True if the program has an incomplete fitness.
        An incomplete fitness cannot be interpreted as invalid program because
        initially all programs have an incomplete fitness. We need a dedicated
        method to check if the fitness is incomplete.

        Args:
            - None

        Returns:
            - bool
                True if the program has an incomplete fitness.
        """
        return len(self.fitness) != len(self.fitness_functions)

    @property
    def hash(self):
        """ This method return the hash of the program

        The hash is a list of unique ideantifiers of the nodes of the tree.
        It is used to compare two programs.

        Args:
            - None

        Returns:
            - List[int]
                A list of unique identifiers of the nodes of the tree.
        """
        if not self._hash:
            if isinstance(self.program, OperationNode):
                self._hash = sorted(self.program.hash(hash_list=[]))
            else:
                self._hash = [self.program.hash(hash_list=[])]

        return self._hash

    @property
    def has_valid_fitness(self) -> bool:
        """
        This method return True if the program has a valid fitness.

        Args:
            - None

        Returns:
            - bool
                True if the program has a valid fitness.
        """
        for label, value in self.fitness.items():
            if np.isnan(value) or np.isinf(value):
                return False
        return True

    def hypervolume(self) -> float:
        if not self.program_hypervolume:
            self._compute_hypervolume()

        return self.program_hypervolume

    def init_program(self) -> None:
        """
        This method initialize a new program calling the recursive generation function.

        The generation of a program follows a genetic algorithm in which the choice on how to
        progress in the generation randomly choose whether to put anothe operation (deepening
        the program) or to put a terminal node (a feature from the dataset or a constant)

        Args:
            - None

        Returns:
            - None
        """

        logging.debug(
            f'Generating a tree with parsimony={self.parsimony} and parsimony_decay={self.parsimony_decay}')

        # Father=None is used to identify the root node of the program
        self.program = self._generate_tree(
            father=None,
            parsimony=self.parsimony,
            parsimony_decay=self.parsimony_decay)

        self.parsimony = self._parsimony_bkp  # Restore parsimony for future operations

        logging.debug(f'Generated a program of depth {self.program_depth}')
        logging.debug(self.program)

        # Reset the hash to force the re-computation
        self._hash = None

    def is_duplicate(self, other: 'Program') -> bool:
        """ Determines whether two programs are equivalent based on equal fitnesses

        If the fitness of two programs are identical, we assume they are equivalent to each other.
        We round to the 5th decimal to state whether the fitness are equal.

        Args:
            - other: Program
                The other program to compare to

        Returns:
            - bool
                True if the two programs are equivalent, False otherwise.
        """
        for (a_label, a_fit), (b_label, b_fit) in zip(self.fitness.items(),
                                                      other.fitness.items()):
            # One difference is enough for them not to be identical
            if round(a_fit, 3) != round(b_fit, 3):
                return False

        return True

    def is_constant(self):
        """ This method return True if the program is a constant, False otherwise.
        """
        return isinstance(self.program, FeatureNode) and self.program.is_constant

    @property
    def is_valid(self) -> bool:
        """ This method return True if the program is valid, False otherwise.

        A program is valid if:
            - It is a valid tree
            - It has a valid fitness

        Returns:
            - bool (default=True)
                True if the program is valid, False otherwise.    
        """

        return self.program.is_valid and self._override_is_valid and self.has_valid_fitness

    def optimize(self,
                 data: Union[dict, pd.Series, pd.DataFrame],
                 target: str,
                 weights: str,
                 constants_optimization: str,
                 constants_optimization_conf: dict,
                 bootstrap: bool = False,
                 inplace: bool = False) -> 'Program':
        """ This method allow to optimize the constants of a program.

        The optimization of constants consists of executing a gradient descent strategy on the constants
        based on the task of the training (classification or regression). The optimization is done using
        implementations of Stochastic Gradient Descent (SGD) and ADAM (both in 1D and 2D version).

        In the input dictionary constants_optimization_conf, the following parameters can be set:
            - learning_rate: float
                The learning rate
            - batch_size: int
                The batch size
            - epochs: int
                The number of epochs
            - gradient_clip: bool
                Whether to clip the gradients
            - beta_1: float
                The beta 1 parameter for ADAM
            - beta_2: float
                The beta 2 parameter for ADAM
            - epsilon: floatbins
                The l1 regularization parameter
            - l2_param: float
                The l2 regularization parameter

        Args:
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data on which the program will be evaluated
            - target: str
                The target variable of the data
            - weights: str
                The weights of the data
            - constants_optimization: str
                The constants optimization method to use
            - constants_optimization_conf: dict
                The configuration of the constants optimization method
            - bootstrap: bool  (default=False)
                If True, the constants will be optimized using bootstrapping
            - inplace: bool  (default=False)
                If True, the method will modify the program in place, otherwise it will return a new program

        Returns:
            - Program
                The optimized program
        """
        if not constants_optimization or not self.is_valid:
            return self
        
        task = constants_optimization_conf['task']
        n_constants = len(self.get_constants(return_objects=False))
        n_features_used = len(self.features_used)
        
        if n_constants > 100:
            logging.debug('Program has more than 100 constants. Optimizing using ADAM')
            constants_optimization = 'ADAM'
            constants_optimization_conf = {
                'task': task,
                'learning_rate': 1e-4,
                'batch_size': int(np.ceil(len(data)/10)),
                'epochs': 200,
                'verbose': 0,
                'gradient_clip': False,
                'beta_1': 0.9,
                'beta_2': 0.999,
                'epsilon': 1e-7,
                'l1_param': 0,
                'l2_param': 0,
            }

        if task not in ['regression:wmse', 'regression:wrrmse', 'regression:cox', 'binary:logistic']:
            raise AttributeError(
                f'Task supported are regression:wmse, regression:wrrmse, regression:cox or binary:logistic')

        if not isinstance(self.program, FeatureNode) and n_constants > 0 and n_features_used > 0:
            ''' Rationale for the conditions:

            not isinstance(program.program, FeatureNode)
                programs with only a FeatureNode are not acceptable anyway

            n_constants > 0
                as the optimization algorithm optimize only constants

            n_features_used > 0
                as it is a constant program anyway and the optimized won't work with this configuration
            '''
            if constants_optimization == 'SGD':
                f_opt = SGD
                self.to_affine(data=data, target=target, inplace=True)

            elif constants_optimization == 'ADAM':
                f_opt = ADAM
                self.to_affine(data=data, target=target, inplace=True)

            elif constants_optimization == 'ADAM2FOLD':
                # Here there can be more than one target so need the index
                f_opt = ADAM2FOLD
                self.to_affine(data=data, target=target[0], inplace=True)

            elif constants_optimization == 'scipy':
                f_opt = SCIPY
                self.to_affine(data=data, target=target, inplace=True)

            else:
                raise AttributeError(
                    f'Constants optimization method {constants_optimization} not supported')

            to_optimize = self if inplace else copy.deepcopy(self)

            def handler(signum, frame):
                raise TimeoutError("Operation timed out")
            
            class TimeoutError(Exception):
                def __str__(self):
                    return ""
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(60)

            try:
                final_parameters, _, _ = f_opt(
                    program=to_optimize,
                    data=data,
                    target=target,
                    weights=weights,
                    constants_optimization_conf=constants_optimization_conf,
                    task=task,
                    bootstrap=bootstrap,
                )
            except NameError:
                return to_optimize
            except TimeoutError:
                return to_optimize
            finally:
                signal.alarm(0)

            if len(final_parameters) > 0:
                to_optimize.set_constants(new=final_parameters)

            return to_optimize

        return self

    def _select_random_node(self, root_node: Union[OperationNode, FeatureNode, InvalidNode], depth: float = .8, only_operations: bool = False) -> Union[OperationNode, FeatureNode]:
        """ This method return a random node of a sub-tree starting from root_node.

        Args:
            - root_node: OperationNode, FeatureNode, InvalidNode
                The root node of the sub-tree.
            - depth: float (default=.8)
                The depth to which select the node. The value must be between 0 and 1.
            - only_operations: bool (default=False)
                If True, the method will return only an OperationNode, otherwise it can return a FeatureNode as well.

        Returns:
            - OperationNode, FeatureNode
                The selected node.
        """

        if isinstance(root_node, InvalidNode):
            return None

        if isinstance(root_node, FeatureNode):
            if only_operations:
                return root_node.father
            return root_node

        if isinstance(root_node, OperationNode):

            if random.random() < depth:
                # Select a random operand
                operand = random.choice(root_node.operands)

                return self._select_random_node(root_node=operand, depth=depth * .9, only_operations=only_operations)
            else:
                return root_node

    def set_constants(self, new: List[float]) -> None:
        """ This method allow to overwrite the value of constants after the neuron-based optimization

        Args:
            - new: list
                The new values of the constants

        Returns:
            - None
        """
        for constant, new_value in zip(self.get_constants(return_objects=True), new):
            constant.feature = new_value

    @staticmethod
    def _set_constants_index(constant, index) -> None:
        """ This method allow to overwrite the index of a constant

        Args:
            - constant: ConstantNode
                The constant to modify

            - index: int
                The new index of the constant

        Returns:
            - None
        """
        constant.index = index

    def similarity(self, other: 'Program') -> float:
        """ This method return the similarity between two programs

        The similarity is computed as the number of common elements between the two programs
        divided by the total number of elements in the two programs.

        Args:
            - other: Program
                The other program to compare to

        Returns:
            - float
                The similarity between the two programs
        """
        def common_elements(list1, list2):
            result = []
            for element in list1:
                if element in list2:
                    result.append(element)
            return result

        c_elements = common_elements(self.hash, other.hash)

        return 2 * len(c_elements) / (len(self.hash) + len(other.hash))

    def simplify(self, inplace: bool = False, inject: Union[str, None] = None) -> 'Program':
        """ This method allow to simplify the structure of a program using a SymPy backend

        Args:
            - inplace: bool (default=False)
                If True, the program is simplified in place. If False, a new Program object is returned.
            - inject: Union[str, None] (default=None)
                If not None, the program is simplified using the inject string as a root node.

        Returns:
            - Program
                The simplified program
        """
        from symbolic_regression.simplification import extract_operation

        if self._hash:
            return self

        def simplify_program(program: Union[Program, str]) -> Union[FeatureNode, OperationNode, InvalidNode]:
            """ This function simplify a program using a SymPy backend

            try: the root node of the program, not the Program object

            """
            try:
                if isinstance(program, Program) and isinstance(
                        program.program, FeatureNode):
                    return program.program

                logging.debug(f'Simplifying program {program}')

                try:
                    if isinstance(program, Program):
                        simplified = sympy.parse_expr(program.program.render())
                    else:
                        simplified = sympy.parse_expr(program)
                except:
                    program._override_is_valid = False
                    return program.program
                logging.debug(
                    f'Extracting the program tree from the simplified')

                new_program = extract_operation(element_to_extract=simplified,
                                                father=None)

                logging.debug(f'Simplified program {new_program}')

                return new_program

            except UnboundLocalError:
                return program.program

        if inplace:
            to_return = self
        else:
            to_return = copy.deepcopy(self)

        if inject:
            simp = simplify_program(inject)
        else:
            simp = simplify_program(to_return)

        to_return.program = simp
        if not simp:
            to_return._override_is_valid = False

        # Reset the hash to force the re-computation
        to_return._hash = None
        return to_return

    def to_affine(self, data: Union[dict, pd.Series, pd.DataFrame], target: str, inplace: bool = False) -> 'Program':
        """ This function create an affine version of the program between the target maximum and minimum

        The affine of a program is defined as:

        .. math::

            \\hat{y} = \\frac{y_{max} - y_{min}}{\\hat{y}_{max} - \\hat{y}_{min}} \\hat{y} + \\frac{y_{min} \\hat{y}_{max} - y_{max} \\hat{y}_{min}}{\\hat{y}_{max} - \\hat{y}_{min}}

        Where:

        - :math:`\\hat{y}` is the output of the program
        - :math:`y_{min}` and :math:`y_{max}` are the minimum and maximum of the target
        - :math:`\\hat{y}_{min}` and :math:`\\hat{y}_{max}` are the minimum and maximum of the output of the program

        Args:
            - data: Union[dict, pd.Series, pd.DataFrame]
                The data used to evaluate the program

            - target: str
                The target of the program

            - inplace: bool (default=False)
                If True, the program is simplified in place. If False, a new Program object is returned.

        Returns:
            - Program
                The affine program
        """

        if not self.is_valid:
            return self

        if inplace:
            prog = self
        else:
            prog = copy.deepcopy(self)

        y_pred = prog.evaluate(data=data)

        y_pred_min = min(y_pred)
        y_pred_max = max(y_pred)
        target_min = data[target].min()
        target_max = data[target].max()

        if y_pred_min == y_pred_max:
            return prog

        alpha = target_max + (target_max - target_min) * \
            y_pred_max / (y_pred_min - y_pred_max)
        beta = (target_max - target_min) / (y_pred_max - y_pred_min)

        if pd.isna(alpha) or pd.isna(beta):
            return prog

        add_node = OperationNode(
            operation=OPERATOR_ADD['func'],
            arity=OPERATOR_ADD['arity'],
            format_str=OPERATOR_ADD['format_str'],
            format_tf=OPERATOR_ADD.get('format_tf'),
            symbol=OPERATOR_ADD.get('symbol'),
            format_diff=OPERATOR_ADD.get(
                'format_diff', OPERATOR_ADD['format_str']),
            father=None
        )
        add_node.add_operand(FeatureNode(
            feature=alpha, father=add_node, is_constant=True))

        mul_node = OperationNode(
            operation=OPERATOR_MUL['func'],
            arity=OPERATOR_MUL['arity'],
            format_str=OPERATOR_MUL['format_str'],
            format_tf=OPERATOR_MUL.get('format_tf'),
            symbol=OPERATOR_MUL.get('symbol'),
            format_diff=OPERATOR_MUL.get(
                'format_diff', OPERATOR_MUL['format_str']),
            father=add_node
        )

        mul_node.add_operand(FeatureNode(
            feature=beta, father=mul_node, is_constant=True))

        prog.program.father = mul_node
        mul_node.add_operand(prog.program)
        add_node.add_operand(mul_node)

        prog.program = add_node
        prog.is_affine = True

        # Reset the hash to force the re-computation
        self._hash = None
        return prog

    def to_logistic(self, inplace: bool = False) -> 'Program':
        """ This function create a logistic version of the program

        The logistic of a program defined as the program between the sigmoid function.

        Args:
            - inplace: bool (default=False)
                If True, the program is simplified in place. If False, a new Program object is returned.

        Returns:
            - Program
                The logistic program
        """
        from symbolic_regression.operators import OPERATOR_SIGMOID
        logistic_node = OperationNode(
            operation=OPERATOR_SIGMOID['func'],
            arity=OPERATOR_SIGMOID['arity'],
            format_str=OPERATOR_SIGMOID['format_str'],
            format_tf=OPERATOR_SIGMOID.get('format_tf'),
            symbol=OPERATOR_SIGMOID.get('symbol'),
            format_diff=OPERATOR_SIGMOID.get(
                'format_diff', OPERATOR_SIGMOID['format_str']),
            father=None
        )
        # So the upward pointer of the father is not permanent
        if inplace:
            program_to_logistic = self
        else:
            program_to_logistic = copy.deepcopy(self)

        logistic_node.operands.append(program_to_logistic.program)
        program_to_logistic.program.father = logistic_node
        program_to_logistic.program = logistic_node
        program_to_logistic.is_logistic = True

        # Reset the hash to force the re-computation
        self._hash = None

        return program_to_logistic

    def to_latex(self) -> str:
        """ This allow to print the program in LaTeX format

        Returns:
            A string representing the program in LaTeX format
        """
        return py2tex(str(self.program))

    def to_mathematica(self) -> str:
        """ This allow to print the program in Mathematica format

        Returns:
            A string representing the program in Mathematica format
        """
        return sympy.printing.mathematica.mathematica_code(self.program)

    # GENETIC OPERATIONS

    def cross_over(self, other: 'Program' = None, inplace: bool = False) -> None:
        """ This module perform a cross-over between this program and another from the population

        A cross-over is the switch between sub-trees from two different programs.
        The cut point are chosen randomly from both programs and the sub-tree from the second
        program (other) will replace the sub-tree from the current program.

        This is a modification only on the current program, so the other one will not be
        affected by this switch.

        It can be performed inplace, overwriting the current program, or returning a new program
        equivalent to the current one after the cross-over is applied.

        Args:
            - other: Program (default=None)
                The other program to cross-over with. If None, a new program is created and used.
            - inplace: bool (default=False)
                If True, the cross-over is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the cross-over is applied
        """

        if not other:
            other = Program(operations=self.operations,
                            features=self.features,
                            const_range=self.const_range,
                            program=self.program,
                            parsimony=self.parsimony,
                            parsimony_decay=self.parsimony_decay)
            other.init_program()

        if not isinstance(other, Program):
            raise TypeError(
                f'Can cross-over only using another Program object: {type(other)} provided'
            )

        if self.features != other.features:
            raise AttributeError(
                f'The two programs must have the same features set')

        if self.operations != other.operations:
            raise AttributeError(
                f'The two programs must have the same operations')

        offspring = copy.deepcopy(self.program)

        cross_over_point1 = self._select_random_node(root_node=offspring)

        cross_over_point2 = copy.deepcopy(
            self._select_random_node(root_node=other.program))

        if not cross_over_point1 or not cross_over_point2:
            return self

        cross_over_point2.father = cross_over_point1.father

        if cross_over_point1.father:
            cross_over_point1.father.operands[
                cross_over_point1.father.operands.index(
                    cross_over_point1)] = cross_over_point2
        else:
            offspring = cross_over_point2

        if inplace:
            self.program = offspring
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def mutate(self, inplace: bool = False) -> 'Program':
        """ This method perform a mutation on a random node of the current program

        A mutation is a random generation of a new sub-tree replacing another random
        sub-tree from the current program.

        Args:
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """

        if self.program_depth == 0:
            # Case in which a one FeatureNode only program is passed.
            # A new tree is generated.
            new = Program(operations=self.operations,
                          features=self.features,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                return self

            return new

        offspring = copy.deepcopy(self.program)

        mutate_point = self._select_random_node(root_node=offspring)

        if (not mutate_point) or not (mutate_point.father):
            new = Program(operations=self.operations,
                          features=self.features,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                return self

            return new

        mutated = self._generate_tree(
            father=mutate_point.father, parsimony=self.parsimony, parsimony_decay=self.parsimony_decay)

        mutate_point.father.operands[
            mutate_point.father.operands.index(mutate_point)] = mutated

        if inplace:
            self.program = offspring
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def insert_node(self, inplace: bool = False) -> 'Program':
        """ This method allow to insert a OperationNode in a random spot in the program

        The insertion of a OperationNode must comply with the arity of the existing
        one and must link to the existing operands.

        Args:
            - inplace: bool (default=False)
                If True, the insertion is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the insertion is applied
        """
        offspring = copy.deepcopy(self.program)
        mutate_point = self._select_random_node(root_node=offspring)

        if not mutate_point:
            new = Program(operations=self.operations,
                          features=self.features,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                return self

            return new

        new_node = self._generate_tree(father=mutate_point.father, depth=1)

        if mutate_point.father:  # Can be None if it is the root
            # Is a new tree of only one OperationNode
            mutate_point.father.operands[mutate_point.father.operands.index(
                mutate_point)] = new_node

        # Choose a random children to attach the previous mutate_point
        # The new_node is already a tree with depth = 1 so, in case of arity=2
        # operations the other operator is already set.
        new_node.operands[random.randint(0, new_node.arity - 1)] = mutate_point

        # If the new_node is also the new root, offspring need to be updated.
        if not mutate_point.father:
            offspring = new_node

        mutate_point.father = new_node

        if inplace:
            self.program = offspring
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def delete_node(self, inplace: bool = False) -> 'Program':
        """ This method delete a random OperationNode from the program.

        It selects a random children of the deleted node to replace itself
        as child of its father.

        Args:
            - inplace: bool (default=False)
                If True, the deletion is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the deletion is applied
        """
        offspring = copy.deepcopy(self.program)
        mutate_point = self._select_random_node(
            root_node=offspring, only_operations=True)

        if mutate_point:
            mutate_father = mutate_point.father
        else:  # When the mutate point is None, can happen when program is only a FeatureNode
            return self

        mutate_child = random.choice(mutate_point.operands)
        mutate_child.father = mutate_father

        if mutate_father:  # Can be None if it is the root
            mutate_father.operands[mutate_father.operands.index(
                mutate_point)] = mutate_child
        else:
            offspring = mutate_child

        if inplace:
            self.program = offspring
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def mutate_leaf(self, inplace: bool = False) -> 'Program':
        """ This method selects a random FeatureNode and changes the associated feature

        The new FeatureNode will replace one random leaf among features and constants.

        Args:
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """
        offspring = copy.deepcopy(self)

        leaves = offspring.get_features(
            return_objects=True) + offspring.get_constants(return_objects=True)

        mutate_point = random.choice(leaves)
        mutate_father = mutate_point.father

        # depth=0 generate a tree of only one FeatureNode
        new_feature = offspring._generate_tree(depth=0, father=mutate_father)

        if mutate_father:
            mutate_father.operands[mutate_father.operands.index(
                mutate_point)] = new_feature
        else:
            offspring.program = new_feature

        if inplace:
            self.program = offspring.program
            return self

        return offspring

    def mutate_operator(self, inplace: bool = False) -> 'Program':
        """ This method selects a random OperationNode and changes the associated operation

        The new OperationNode will replace one random leaf among features and constants.

        Args:
            - inplace: bool (default=False)
                If True, the mutation is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the mutation is applied
        """
        offspring = copy.deepcopy(self.program)

        mutate_point = self._select_random_node(
            root_node=offspring, only_operations=True)

        if not mutate_point:  # Only a FeatureNode without any OperationNode
            new = Program(operations=self.operations,
                          features=self.features,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()

            if inplace:
                self.program = new
                return self

            return new

        new_operation = random.choice(self.operations)
        while new_operation['arity'] != mutate_point.arity:
            new_operation = random.choice(self.operations)

        mutate_point.operation = new_operation.get('func')
        mutate_point.format_str = new_operation.get('format_str')
        mutate_point.format_tf = new_operation.get('format_tf')
        mutate_point.symbol = new_operation.get('symbol')
        mutate_point.format_diff = new_operation.get(
            'format_diff', new_operation.get('format_str'))

        if inplace:
            self.program = offspring
            return self

        new = Program(program=offspring,
                      operations=self.operations,
                      features=self.features,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new

    def recalibrate(self, inplace: bool = False) -> 'Program':
        """ This method recalibrate the constants of the program

        The new constants will be sampled from a uniform distribution.

        Args:
            - inplace: bool (default=False)
                If True, the recalibration is performed in place. If False, a new Program object is returned.

        Returns:
            - Program
                The new program after the recalibration is applied
        """
        offspring: Program = copy.deepcopy(self)

        offspring.set_constants(
            new=list(np.random.uniform(
                low=self.const_range[0],
                high=self.const_range[1],
                size=len(self.get_constants())
            ))
        )

        if inplace:
            self.program = offspring.program
            return self

        new = Program(program=offspring.program,
                      operations=self.operations,
                      features=self.features,
                      const_range=self.const_range,
                      parsimony=self.parsimony,
                      parsimony_decay=self.parsimony_decay)

        return new
