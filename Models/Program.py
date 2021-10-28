from copy import deepcopy
import logging
import random
from typing import Union

import pandas as pd
import numpy as np

from Models.Node import FeatureNode, Node, OperationNode


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
    The evaluation simply execute recursively the defined operation using the values
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
            const_range: 
            max_depth: 
            seed: 
            program: 
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

            # Do not set self.fitness because it should be already calculated 
        else:
            self.program: Node = None
            self.fitness = float(np.inf)
        
        self.max_depth: int = max_depth

    def cross_over(self, other=None, inplace: bool = False) -> None:
        """
        """
        if self.program_depth == 0:
            raise AssertionError(
                f'This program has depth 0 and cannot undergo cross-over')

        if other.program_depth == 0:
            raise AssertionError(
                f'The argument program has depth 0 and cannot be used for cross-over')

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
        cross_over_point2.father = cross_over_point1

        child_count_cop1 = len(cross_over_point1.operands)

        child_to_replace = random.randrange(child_count_cop1)
        logging.debug(
            f'Cross-over: {cross_over_point1.operands[child_to_replace]} replaced by {cross_over_point2}')

        cross_over_point1.operands[child_to_replace] = cross_over_point2

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
        return self.program.evaluate(data=data)

    def init_program(self,
                     parsimony: float = 0.95,
                     parsimony_decay: float = 0.95,
                     max_depth: int = np.inf,
                     current_depth: int = 0,
                     ) -> None:
        """ This method initialize a new program calling the recursive generaion function.
        """

        self.parsimony = parsimony
        self.parsimony_decay = parsimony_decay

        self.program_depth = current_depth
        self.max_depth = max_depth

        logging.debug(
            f'Generating a tree with parsimony={parsimony} and parsimony_decay={parsimony_decay}')

        self.program = self._generate_tree(
            parsimony=parsimony, parsimony_decay=parsimony_decay,
            current_depth=0, father=None)

        logging.debug(f'Generated a program of depth {self.program_depth}')
        logging.debug(f'Operation Used: {self.operations_used}')
        logging.debug(f'Features Used: {self.features_used}')
        logging.debug(self.program)

    def __lt__(self, other):
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

        If max_depth is set and the current recustion level reach that level,
        a FeatureNode is forced to be generated.
        """

        current_depth += 1

        if self.max_depth and self.max_depth < self.program_depth:
            raise AssertionError(
                f'Max depth is reached and something went wrong')

        # If max_depth is reached, no more OperationNode will be generated
        if self.program_depth == 0 or (random.random() < parsimony and current_depth < self.max_depth):
            # Generate an OperationNode
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

        else:
            # Generate a FeatureNode
            if random.random() > (1 / len(self.features)):
                # Use a feature from the dataset
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
                feature = random.uniform(self.const_range[0], self.const_range[1])
                
                feature = round(feature, 3)

                node = FeatureNode(
                    feature=feature,
                    depth=current_depth,
                    father=father,
                    is_constant=True
                )

        return node

    def mutate(self, inplace: bool = False):
        """
        """

        if self.program_depth == 0:
            raise AssertionError(
                f'This program has depth 0 and cannot undergo mutation')

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
            logging.DEBUG(f'Now the program has depth {self.depth}')
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
            self.features_used[node.feature] += 1

    def _reset_operations_feature_usage(self) -> None:
        """
        """
        self.operations_used = {}

        for op in self.operations:
            self.operations_used[op['func']] = 0

        self.features_used = {ft: 0 for ft in self.features} 

    def _select_random_node(self,
                            root_node: Union[OperationNode, FeatureNode],
                            deepness: float = 0.25
                            ) -> Union[OperationNode, FeatureNode]:
        """
        """

        to_return = None

        if random.random() < deepness:
            # Favor nodes higher in the tree (near the root)
            # In case it doesn't return here, it will eventually call recursively this function
            # so that the depth of the node will be such that the probability of returning
            # here increase at every call.
            #logging.debug(f'Node selected {subtree}')
            to_return = root_node

        if isinstance(root_node, FeatureNode):
            # Final recursion!
            # Case in which the node does not have children. It is a leaf node for sure
            # and therefore we need its parent (the operation node above)
            logging.debug(
                f'FeatureNode selected: returning its father {root_node.father}')
            to_return = root_node.father

        if to_return:
            logging.debug(
                f'Selected a {to_return.operation} node deep {to_return.depth}')
            return to_return

        return self._select_random_node(root_node=random.choice(root_node.operands))
