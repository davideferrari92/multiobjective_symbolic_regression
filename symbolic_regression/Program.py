import logging
from operator import ne
import random
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

from symbolic_regression.Node import FeatureNode, Node, OperationNode


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

    def __init__(self, operations: list, features: list, const_range: tuple, max_depth: int = np.inf,
                 program: Node = None) -> None:
        """

        Args:
            operations: The list of possible operations to use as operations
            features: The list of possible features to use as terminal nodes
            const_range: The range of possible values for constant terminals in the program
            max_depth: The max depth of a program (when reached the program is cut)
            program: An existing program from which to initialize this object
        """

        self.operations = operations
        self.features = features
        self.const_range = const_range

        self.program_depth: int = 0

        # Pareto Front Attributes
        self.rank: int = None
        self.programs_dominates: list = []
        self.programs_dominated_by: list = []
        self.crowding_distance: float = 0
        self._reset_operations_feature_usage()

        if program:
            self.program: Node = program
            self._reset_operations_feature_usage()
            self._reset_depths(
                node=self.program, current_depth=0)

        else:
            self.program: Node = None
            self.fitness = float(np.inf)

        self.max_depth: int = max_depth

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

            new._reset_operations_feature_usage()
            new._reset_depths(node=new.program, current_depth=0, father=None)
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

        logging.debug(
            f'Performing cross-over at depth {cross_over_point1.depth}')

        cross_over_point2 = deepcopy(
            self._select_random_node(root_node=other.program))

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
            self._reset_operations_feature_usage()
            self._reset_depths(node=self.program, current_depth=0, father=None)
            return self

        new = Program(program=offspring, operations=self.operations,
                      features=self.features, max_depth=self.max_depth,
                      const_range=self.const_range)

        new.parsimony = self.parsimony
        new.parsimony_decay = self.parsimony_decay

        new._reset_operations_feature_usage()
        new._reset_depths(node=new.program, current_depth=0, father=None)

        return new

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This call evaluate the program on some given data calling the function
        of the evaluation of the root node of the program.
        This will recursively call the evaluation on all the children returning the
        final result based on the given data

        Args:
            data: the data on which to evaluate this program
        """
        return self.program.evaluate(data=data)

    def init_program(self,
                     parsimony: float = 0.95,
                     parsimony_decay: float = 0.95,
                     max_depth: int = np.inf,
                     current_depth: int = 0,
                     ) -> None:
        """ This method initialize a new program calling the recursive generation function.

        The generation of a program follows a genetic algorithm in which the choice on how to
        progress in the generation randomly choose whether to put anothe operation (deepening
        the program) or to put a terminal node (a feature from the dataset or a constant)

        Args:
            parsimony: The ratio with which to choose operations among terminal nodes
            parsimony_decay: The ratio with which the parsimony decreases to prevent infinite programs
            max_depth: The program depth after which the generation will force a terminal node
            current_depth: Used by the recursive call to be aware to which depth the recursion is
        """

        self.parsimony = parsimony
        self.parsimony_decay = parsimony_decay

        self.program_depth = current_depth
        self.max_depth = max_depth

        logging.debug(
            f'Generating a tree with parsimony={parsimony} and parsimony_decay={parsimony_decay}')

        # Father=None is used to identify the root node of the program
        self.program = self._generate_tree(
            parsimony=parsimony, parsimony_decay=parsimony_decay,
            current_depth=0, father=None)

        logging.debug(f'Generated a program of depth {self.program_depth}')
        logging.debug(f'Operation Used: {self.operations_used}')
        logging.debug(f'Features Used: {self.features_used}')
        logging.debug(self.program)

    def __lt__(self, other):
        """ This ordering function allow to compare programs by their fitness value

        The fitnes can be a single or a list of values depending on the number of
        objective functions. If the objective is only one, the fitness will be a generic
        numeric comparison, wherheas in case of multiple objective functions, the 
        Pareto Front and the Crowding Distance methods are used in a multi-dimensional
        evaluation of the fitness ordering.

        Args:
            other: The program to which compare the current one
        """
        if isinstance(self.fitness, float):
            return self.fitness < other.fitness

        elif isinstance(self.fitness, list):
            return self.rank <= other.rank and self.crowding_distance >= other.crowding_distance

        else:
            raise TypeError(
                f'program.fitness is neither a float or a list: {type(self.fitness)}')

    def _generate_tree(self,
                       parsimony: float,
                       parsimony_decay: float,
                       current_depth: int = 0,
                       father: Union[Node, None] = None):
        """ This method run the recursive generation of a subtree.

        If the node generated in a recursion loop is an OperationNode, then a deeper
        recursion is performed to populate its operands.
        If the node is a FeatureNode instead, the recursion terminate and a FeatureNode
        is added to the operation's operands list

        If max_depth is set and the current recusion level reach that level,
        a FeatureNode node, which is terminal, is forced to be generated. 

        Args:
            parsimony: The ratio with which to choose operations among terminal nodes
            parsimony_decay: The ratio with which the parsimony decreases to prevent infinite programs
            current_depth: Used by the recursive call to be aware to which depth the recursion is
            father: The father to the next generated node (None for the root node)
        """

        current_depth += 1

        if self.max_depth and self.max_depth < self.program_depth:
            raise AssertionError(
                f'Max depth is reached and something went wrong')

        # If max_depth is reached, no more OperationNode will be generated and a FeatureNode is forced
        if self.program_depth == 0 or (random.random() < parsimony and current_depth < self.max_depth):

            operation = random.choice(self.operations)
            self.operations_used[operation['func']] += 1

            node = OperationNode(
                operation=operation['func'],
                arity=operation['arity'],
                format_str=operation['format_str'],
                depth=current_depth,
                father=father
            )

            if current_depth > self.program_depth:
                self.program_depth = current_depth

            # Recursive call to populate the operands of the new OperationNode
            for _ in range(node.arity):
                node.add_operand(
                    self._generate_tree(
                        parsimony=parsimony * parsimony_decay,
                        parsimony_decay=parsimony_decay,
                        current_depth=current_depth+1,
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
                    depth=current_depth,
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
                    depth=current_depth,
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
                          features=self.features, const_range=self.const_range,
                          max_depth=self.max_depth)

            new.init_program(
                parsimony=self.parsimony,
                parsimony_decay=self.parsimony_decay,
                max_depth=new.max_depth,
                current_depth=0,
            )

            return new

        offspring = deepcopy(self.program)
        mutate_point = self._select_random_node(root_node=offspring)

        logging.debug(
            f'Performing mutation at depth {mutate_point.depth}/{self.program_depth} on a {mutate_point.operation} node')

        child_to_mutate = random.randrange(mutate_point.arity)

        to_mutate = mutate_point.operands[child_to_mutate]
        logging.debug(f'Mutating {to_mutate}')

        ####################################################################
        # TODO manage mutation behavior using parsimony and parsimony_decay
        mutated = self._generate_tree(
            parsimony=self.parsimony,
            parsimony_decay=self.parsimony_decay,
            current_depth=to_mutate.depth,
            father=mutate_point
        )

        logging.debug(f'Mutated {to_mutate} in {mutated}')

        mutate_point.operands[child_to_mutate] = mutated

        if inplace:
            self.program = offspring
            self._reset_operations_feature_usage()
            self._reset_depths(node=self.program, current_depth=0, father=None)
            logging.debug(f'Now the program has depth {self.program_depth}')
            return self

        new = Program(program=offspring, operations=self.operations,
                      features=self.features, const_range=self.const_range,
                      max_depth=self.max_depth)

        new.parsimony = self.parsimony
        new.parsimony_decay = self.parsimony_decay

        return new

    def _reset_depths(self, node: Union[OperationNode, FeatureNode], current_depth: int = 0, father: Union[Node, None] = None) -> None:
        """ This method allow to re-evaluate all nodes depths when the tree is modified.

        This method should be called when a mutation or a cross-over are performed to ensure
        that all nodes' depths and parenthoods are set up correctly.

        Args:
            node: The node of which update the depth
            current_depth: Used for the recursive call to propagate the value to lower nodes
            father: The node to set as father to the children
        """
        node.depth = current_depth
        node.father = father

        if current_depth > self.program_depth:
            self.program_depth = current_depth

        if isinstance(node, OperationNode):
            if not self.operations_used.get(node.operation):
                self.operations_used[node.operation] = 0
            self.operations_used[node.operation] += 1

            for operand in node.operands:
                self._reset_depths(
                    node=operand, current_depth=current_depth+1, father=node)

        if isinstance(node, FeatureNode) and not node.is_constant:
            if not self.features_used.get(node.feature):
                self.features_used[node.feature] = 0
            self.features_used[node.feature] += 1

    def _reset_operations_feature_usage(self) -> None:
        """ This method re-evaluate the usage of features and operations of the program
        """
        self.operations_used = {}

        for op in self.operations:
            self.operations_used[op['func']] = 0

        self.features_used = {ft: 0 for ft in self.features}

    def _select_random_node(self,
                            root_node: Union[OperationNode, FeatureNode],
                            deepness: float = 0.75
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
                to_return = self._select_random_node(root_node=random.choice(root_node.operands), deepness=deepness)
            except AttributeError as e:
                # TypeError happens when root_node is a FeatureNode
                to_return = root_node.father  # Can be OperationNode or None

        return to_return