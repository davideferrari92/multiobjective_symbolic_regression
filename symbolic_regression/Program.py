import logging
import random
from copy import deepcopy
from typing import Union
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from symbolic_regression.Node import FeatureNode, InvalidNode, Node, OperationNode


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

    def __init__(self,
                 operations: list,
                 features: list,
                 const_range: tuple = None,
                 program: Node = None,
                 constants_optimization: bool = False,
                 constants_optimization_conf: dict = {}
                 ) -> None:
        """

        Args:
            operations: The list of possible operations to use as operations
            features: The list of possible features to use as terminal nodes
            const_range: The range of possible values for constant terminals in the program
            program: An existing program from which to initialize this object
        """

        self.operations = operations
        self.features = features
        self.const_range = const_range
        self._constants = []
        self.converged = False

        self.constants_optimization = constants_optimization
        self.constants_optimization_conf = constants_optimization_conf
        self.constants_optimization_details = {}

        self._override_is_valid = True
        self._is_duplicated = False

        self._program_depth: int = 0
        self._complexity: int = 0

        # Pareto Front Attributes
        self.rank: int = np.inf
        self.programs_dominates: list = []
        self.programs_dominated_by: list = []
        self.crowding_distance: float = 0

        if program:
            self.program: Node = program

        else:
            self.program: Node = InvalidNode()
            self.fitness = float(np.inf)

    @property
    def complexity(self):
        """ The complexity of a program is the number of nodes (OperationNodes or FeatureNodes)
        """
        return self._complexity

    @complexity.getter
    def complexity(self, base_complexity=0):
        return self.program._get_complexity(base_complexity)

    @property
    def program_depth(self):
        return self._program_depth

    @program_depth.getter
    def program_depth(self, base_depth=0):
        return self.program._get_depth(base_depth)

    @property
    def operations_used(self):
        return self._operations_used

    @operations_used.getter
    def operations_used(self):
        return self.program._get_operations(base_operations_used={})

    @property
    def features_used(self):
        return self._features_used

    @features_used.getter
    def features_used(self):
        return self.program._get_features(base_features={})

    @property
    def is_valid(self):
        return self.program.is_valid() and self._override_is_valid

    def get_constants(self):
        to_return = None
        if isinstance(self.program, OperationNode):
            to_return = self.program._get_constants(const_list=[])

        # Only one constant FeatureNode
        elif self.program.is_constant:
            to_return = [self]

        else:
            # Only one non-constant FeatureNode
            to_return = []

        for index, constant in enumerate(to_return):
            self._set_constants_index(constant=constant, index=index)

        return to_return

    def get_features(self):

        if isinstance(self.program, OperationNode):
            return list(set(self.program._get_features(features_list=[])))

        # Only one non-constant FeatureNode
        elif not self.program.is_constant:
            return [self]

        # Only one constant FeatureNode
        return []

    def set_constants(self, new):

        for constant, new_value in zip(self.get_constants(), new):
            constant.feature = new_value

    @staticmethod
    def _set_constants_index(constant, index):
        constant.index = index

    def cross_over(self, other=None, inplace: bool = False) -> None:
        """ This module perform a cross-over between this program and another from the population

        A cross-over is the switch between sub-trees from two different programs.
        The cut point are chosen randomly from both programs and the sub-tree from the second
        program (other) will replace the sub-tree from the current program.

        This is a modification only on the current program, so the other one will not be
        affected by this switch.

        It can be performed inplace, overwriting the current program, or returning a new program
        equivalent to the current one after the cross-over is applied.

        Args:
            other: the program from which to extract a sub-tree
            inplace: whether to overwrite this object or return a new equivalent object
        """
        if self.program_depth == 0 or other.program_depth == 0:
            new = deepcopy(self)
            new.mutate(inplace=True)
            return new

        if not isinstance(other, Program):
            raise TypeError(
                f'Can cross-over only using another Program object: {type(other)} provided')

        if self.features != other.features:
            raise AttributeError(
                f'The two programs must have the same features set')

        if self.operations != other.operations:
            raise AttributeError(
                f'The two programs must have the same operations')

        offspring = deepcopy(self.program)
        cross_over_point1 = self._select_random_node(root_node=offspring)

        if not cross_over_point1:
            return self

        cross_over_point2 = deepcopy(
            self._select_random_node(root_node=other.program))

        if cross_over_point2:
            cross_over_point1 = cross_over_point2

        child_count_cop1 = len(cross_over_point1.operands)

        child_to_replace = random.randrange(child_count_cop1)
        logging.debug(
            f'Cross-over: {cross_over_point1.operands[child_to_replace]} replaced by {cross_over_point2}')

        cross_over_point1.operands[child_to_replace] = cross_over_point2

        ''' We need to reset the operations and feature usage counts and
        also re-evaluate the depth of all nodes because we mixed two programs
        and those information are not consistent anymore.
        '''
        if inplace:
            self.program = offspring
            return self

        new = Program(program=offspring, operations=self.operations,
                      features=self.features,
                      constants_optimization=self.constants_optimization,
                      constants_optimization_conf=self.constants_optimization_conf,
                      const_range=self.const_range)

        new.parsimony = self.parsimony
        new.parsimony_decay = self.parsimony_decay

        return new

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This call evaluate the program on some given data calling the function
        of the evaluation of the root node of the program.
        This will recursively call the evaluation on all the children returning the
        final result based on the given data

        Args:
            data: the data on which to evaluate this program
        """
        if not self.is_valid:
            return None

        return self.program.evaluate(data=data)

    def evaluate_fitness(self, data, fitness, target, weights):

        self.fitness = dict()

        if not self.is_valid:
            return

        n_features = len(self.get_features())
        n_constants = len(self.get_constants())

        if not isinstance(self.program, FeatureNode) and self.constants_optimization and n_constants > 0:
            if self.const_range:
                const_range_min = self.const_range[0]
                const_range_max = self.const_range[1]
            else:
                const_range_min = -1
                const_range_max = 1

            from symbolic_regression.multiobjective.optimization import \
                NNOptimizer
            constants_optimizer = NNOptimizer(
                units=1,
                n_features=len(self.features),
                n_constants=n_constants,
                const_range_min=const_range_min,
                const_range_max=const_range_max,
                exec_string=self.program.render(
                    data=data.iloc[0, :], format_tf=True)
            )

            data_tensor = tf.constant(
                data[self.features].to_numpy(), dtype=tf.float32)

            target_tensor = tf.constant(
                data[target].to_numpy(), dtype=tf.float32)
            if weights:
                weights_tensor = tf.constant(
                    data[weights].to_numpy(), dtype=tf.float32)
            else:
                weights_tensor = tf.ones_like(target_tensor)

            inputs = Input(shape=[len(self.features)], name="Input")

            model = Model(inputs=inputs, outputs=constants_optimizer(inputs))
            loss_mse = tf.keras.losses.MeanSquaredError()
            opt = tf.keras.optimizers.Adam(
                learning_rate=self.constants_optimization_conf['learning_rate'])
            model.compile(loss=loss_mse, optimizer=opt, run_eagerly=False)

            model.fit(
                data_tensor,
                target_tensor,
                sample_weight=weights_tensor,
                batch_size=self.constants_optimization_conf['batch_size'],
                epochs=self.constants_optimization_conf['epochs'],
                verbose=self.constants_optimization_conf['verbose']
            )

            final_parameters = list(model.get_weights()[0][0])

            self.set_constants(new=final_parameters)

            for old, new in zip(self.get_constants(), final_parameters):
                self.constants_optimization_details[old] = new

        evaluated = fitness(program=self, data=data,
                            target=target, weights=weights)

        _converged = []

        for ftn_label, ftn in evaluated.items():

            f = ftn['func']

            threshold = ftn.get('threshold')

            if isinstance(f, float) or isinstance(f, int) or isinstance(f, np.float):
                if pd.isna(f):
                    f = np.inf
                    self._override_is_valid = False

                self.fitness[ftn_label] = f

                if threshold and f <= threshold:
                    _converged.append(True)
                    logging.info(f'Converged {ftn_label}: {f} <= {threshold}')
            elif isinstance(f, tuple):
                for elem_index, elem in enumerate(f):
                    if pd.isna(f):
                        f = np.inf
                        self._override_is_valid = False

                    self.fitness[ftn_label + f'_{elem_index}'] = elem
                    if threshold and f <= threshold[elem_index]:
                        self._converged.append(True)
                    else:
                        self._converged.append(False)

        # Use any or all to have at least one or all fitness converged when the threshold is provided
        if len(_converged) > 0:
            self.converged = all(_converged)

    def init_program(self,
                     parsimony: float = 0.95,
                     parsimony_decay: float = 0.95,
                     ) -> None:
        """ This method initialize a new program calling the recursive generation function.

        The generation of a program follows a genetic algorithm in which the choice on how to
        progress in the generation randomly choose whether to put anothe operation (deepening
        the program) or to put a terminal node (a feature from the dataset or a constant)

        Args:
            parsimony: The ratio with which to choose operations among terminal nodes
            parsimony_decay: The ratio with which the parsimony decreases to prevent infinite programs
        """

        self.parsimony = parsimony
        self.parsimony_decay = parsimony_decay

        logging.debug(
            f'Generating a tree with parsimony={parsimony} and parsimony_decay={parsimony_decay}')

        # Father=None is used to identify the root node of the program
        self.program = self._generate_tree(
            parsimony=parsimony, parsimony_decay=parsimony_decay,
            father=None)

        logging.debug(f'Generated a program of depth {self.program_depth}')
        logging.debug(self.program)

    def __lt__(self, other):
        """ This ordering function allow to compare programs by their fitness value

        TODO fix docstring with new fitness dictionaries

        Args:
            other: The program to which compare the current one
        """

        if isinstance(self.fitness, dict):
            return self.rank <= other.rank and self.crowding_distance >= other.crowding_distance

        else:
            raise TypeError(
                f'program.fitness is not a dict: {type(self.fitness)}')

    def is_duplicate(self, other):
        """ Determines whether two programs are equivalent based on equal fitnesses

        If the fitness of two programs are identical, we assume they are equivalent to each other.
        """
        is_d = True
        for a_fit, b_fit in zip(self.fitness.values(), other.fitness.values()):
            if round(a_fit, 3) != round(b_fit, 3):  # One difference is enough for them not to be identical
                is_d = False
        
        return is_d

    def _generate_tree(self,
                       parsimony: float,
                       parsimony_decay: float,
                       father: Union[Node, None] = None):
        """ This method run the recursive generation of a subtree.

        If the node generated in a recursion loop is an OperationNode, then a deeper
        recursion is performed to populate its operands.
        If the node is a FeatureNode instead, the recursion terminate and a FeatureNode
        is added to the operation's operands list


        Args:
            parsimony: The ratio with which to choose operations among terminal nodes
            parsimony_decay: The ratio with which the parsimony decreases to prevent infinite programs
            father: The father to the next generated node (None for the root node)
        """

        if random.random() < parsimony:

            operation = random.choice(self.operations)

            node = OperationNode(
                operation=operation['func'],
                arity=operation['arity'],
                format_str=operation['format_str'],
                format_tf=operation.get('format_tf'),
                father=father
            )

            # Recursive call to populate the operands of the new OperationNode
            for _ in range(node.arity):
                node.add_operand(
                    self._generate_tree(
                        parsimony=parsimony * parsimony_decay,
                        parsimony_decay=parsimony_decay,
                        father=node
                    )
                )

        else:  # Generate a FeatureNode
            ''' The probability to get a feature from the training data is
            (n-1) / n where n is the number of features.
            Otherwise a constant value will be generated.
            '''

            if random.random() > (1 / len(self.features)):
                # A Feature from the dataset

                feature = random.choice(self.features)
                self.features_used[feature] += 1

                node = FeatureNode(
                    feature=feature,
                    father=father,
                    is_constant=False
                )
            else:
                # Generate a constant

                feature = random.uniform(
                    self.const_range[0], self.const_range[1])

                # Arbitrary rounding of the generated constant
                feature = round(feature, 2)

                node = FeatureNode(
                    feature=feature,
                    father=father,
                    is_constant=True
                )

        return node

    def mutate(self, inplace: bool = False):
        """ This method perform a mutation on a random node of the current program

        A mutation is a random generation of a new sub-tree replacing another random
        sub-tree from the current program.

        Args:
            inplace: Whether to overwrite the current program or to return a new mutated object
        """

        if self.program_depth == 0:
            # Case in which a one FeatureNode only program is passed.
            # A new tree is generated.
            new = Program(operations=self.operations,
                          features=self.features, const_range=self.const_range)

            new.init_program(
                parsimony=self.parsimony,
                parsimony_decay=self.parsimony_decay,
            )

            return new

        offspring = deepcopy(self.program)
        mutate_point = self._select_random_node(root_node=offspring)

        if not mutate_point:
            mutate_point = offspring

        try:
            child_to_mutate = random.randrange(mutate_point.arity)
            to_mutate = mutate_point.operands[child_to_mutate]
        except ValueError:  # Case in which the tree has depth 0 with a FeatureNode as root
            to_mutate = mutate_point

        logging.debug(f'Mutating {to_mutate}')

        ####################################################################
        # TODO manage mutation behavior using parsimony and parsimony_decay
        mutated = self._generate_tree(
            parsimony=self.parsimony,
            parsimony_decay=self.parsimony_decay,
            father=mutate_point
        )

        logging.debug(f'Mutated {to_mutate} in {mutated}')

        to_mutate = mutated

        if inplace:
            self.program = offspring
            logging.debug(f'Now the program has depth {self.program_depth}')
            return self

        new = Program(program=offspring, operations=self.operations,
                      constants_optimization=self.constants_optimization,
                      constants_optimization_conf=self.constants_optimization_conf,
                      features=self.features, const_range=self.const_range)

        new.parsimony = self.parsimony
        new.parsimony_decay = self.parsimony_decay

        return new

    def _select_random_node(self,
                            root_node: Union[OperationNode, FeatureNode],
                            deepness: float = 0.15
                            ) -> Union[OperationNode, FeatureNode]:
        """ This method return a random node of a sub-tree starting from root_node.

        To modulate the deepness to which the returned node will likely be, we can use 
        'deepness'. Should be between 0 and 1; the higher the value, the closest to the
        root_node the returned node will be. 

        Args:
            root_node: The node from which start the descent to select a random child node
            deepness: This modulates how deep the returned node will be
        """

        to_return = None

        if random.random() < deepness:
            to_return = root_node

            if isinstance(to_return, FeatureNode):
                to_return = root_node.father  # Can be OperationNode or None

        else:
            try:
                to_return = self._select_random_node(
                    root_node=random.choice(root_node.operands), deepness=deepness)
            except AttributeError as e:
                # TypeError happens when root_node is a FeatureNode
                to_return = root_node.father  # Can be OperationNode or None

        return to_return
