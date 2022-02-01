import logging
import random
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import sympy
from tensorboard import program

from symbolic_regression.Node import (FeatureNode, InvalidNode, Node,
                                      OperationNode)
from symbolic_regression.multiobjective.optimization import optimize


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
                 parsimony: float = .9,
                 parsimony_decay: float = .85
                 ) -> None:
        """

        Args:
            operations: The list of possible operations to use as operations
            features: The list of possible features to use as terminal nodes
            const_range: The range of possible values for constant terminals in the program
            program: An existing program from which to initialize this object
            parsimony: the modulator that determine how deep the tree generation will go
            parsimony_decay: the modulator that decays the parsimony to prevent infinite trees
        """

        self.operations = operations
        self.features = features
        self.const_range = const_range
        self._constants = []
        self.converged = False

        # Operational attributes
        self._override_is_valid = True
        self._is_duplicated = False
        self._program_depth: int = 0
        self._complexity: int = 0

        # Pareto Front Attributes
        self.rank: int = np.inf
        self.programs_dominates: list = []
        self.programs_dominated_by: list = []
        self.crowding_distance: float = 0

        self.parsimony = parsimony
        self._parsimony_bkp = parsimony
        self.parsimony_decay = parsimony_decay

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
    def features_used(self):
        """ This allow to get all the unique features used in the tree
        """
        return self._features_used

    @features_used.getter
    def features_used(self):
        return self.get_features(features_list=[])

    @property
    def is_valid(self):
        """ The validity of a program represent its executability and absence of errors or inconsistencies
        """
        return self.program.is_valid and self._override_is_valid

    def get_constants(self):
        """ This method allow to get all constants used in a tree.

        The constants are used for the neuronal-based constants optimizer; it requires
        all constants to be in a fixed order explored by a DFS descent of the tree.
        """
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

    def get_features(self, return_objects: bool = False):
        """ This method recursively explore the tree and return a list of unique features used.
        """
        if isinstance(self.program, OperationNode):
            return self.program._get_features(features_list=[], return_objects=return_objects)

        # Only one non-constant FeatureNode
        elif not self.program.is_constant:
            return [self]

        # Only one constant FeatureNode
        return []

    def set_constants(self, new):
        """ This method allow to overwrite the value of constants after the neuron-based optimization
        """
        for constant, new_value in zip(self.get_constants(), new):
            constant.feature = new_value

    @staticmethod
    def _set_constants_index(constant, index):
        constant.index = index

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

    def evaluate_fitness(self, data, fitness):
        """
        """
        self.fitness = dict()

        if not self.is_valid:
            return None

        evaluated = fitness(program=self, data=data)

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
                    logging.debug(f'Converged {ftn_label}: {f} <= {threshold}')
            elif isinstance(f, tuple):
                for elem_index, elem in enumerate(f):
                    if pd.isna(f):
                        f = np.inf
                        self._override_is_valid = False

                    self.fitness[ftn_label + f'_{elem_index}'] = elem
                    if threshold and f <= threshold[elem_index]:
                        _converged.append(True)
                    else:
                        _converged.append(False)

        # Use any or all to have at least one or all fitness converged when the threshold is provided
        if len(_converged) > 0:
            self.converged = all(_converged)

    def init_program(self) -> None:
        """ This method initialize a new program calling the recursive generation function.

        The generation of a program follows a genetic algorithm in which the choice on how to
        progress in the generation randomly choose whether to put anothe operation (deepening
        the program) or to put a terminal node (a feature from the dataset or a constant)

        Args:
            parsimony: The ratio with which to choose operations among terminal nodes
            parsimony_decay: The ratio with which the parsimony decreases to prevent infinite programs
        """

        logging.debug(
            f'Generating a tree with parsimony={self.parsimony} and parsimony_decay={self.parsimony_decay}')

        # Father=None is used to identify the root node of the program
        self.program = self._generate_tree(
            father=None,
            parsimony=self.parsimony,
            parsimony_decay=self.parsimony_decay
        )

        self.parsimony = self._parsimony_bkp  # Restore parsimony for future operations

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
        for a_fit, b_fit in zip(self.fitness.values(), other.fitness.values()):
            # One difference is enough for them not to be identical
            if round(a_fit, 3) != round(b_fit, 3):
                return False

        return True

    def _generate_tree(self,
                       depth=-1,
                       parsimony: float = .9,
                       parsimony_decay: float = .85,
                       father: Union[Node, None] = None):
        """ This method run the recursive generation of a subtree.

        If the node generated in a recursion loop is an OperationNode, then a deeper
        recursion is performed to populate its operands.
        If the node is a FeatureNode instead, the recursion terminate and a FeatureNode
        is added to the operation's operands list

        Args:
            depth: if a fixed depth is required, this arg is set to that value
            father: The father to the next generated node (None for the root node)
        """

        def gen_operation(operation_conf: dict, father):
            return OperationNode(
                operation=operation_conf['func'],
                arity=operation_conf['arity'],
                format_str=operation_conf['format_str'],
                format_tf=operation_conf.get('format_tf'),
                father=father
            )

        def gen_feature(feature, father, is_constant):
            return FeatureNode(
                feature=feature,
                father=father,
                is_constant=is_constant
            )

        if not parsimony:
            parsimony = self.parsimony
        if not parsimony_decay:
            parsimony_decay = self.parsimony_decay

        if (random.random() < parsimony and depth == -1) or (depth > 0):

            operation = random.choice(self.operations)
            node = gen_operation(operation_conf=operation, father=father)

            # Recursive call to populate the operands of the new OperationNode
            if depth == -1:
                new_depth = -1
            else:
                new_depth = depth-1

            for _ in range(node.arity):
                node.add_operand(
                    self._generate_tree(
                        depth=new_depth,
                        father=node,
                        parsimony=parsimony*parsimony_decay,
                        parsimony_decay=parsimony_decay
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

                node = gen_feature(
                    feature=feature, father=father, is_constant=False)

            else:
                # Generate a constant

                feature = random.uniform(
                    self.const_range[0], self.const_range[1])

                # Arbitrary rounding of the generated constant
                feature = round(feature, 2)

                node = gen_feature(
                    feature=feature, father=father, is_constant=True)

        return node

    def _select_random_node(self,
                            root_node: Union[OperationNode, FeatureNode, InvalidNode],
                            deepness: float = 0.15
                            ) -> Union[OperationNode, FeatureNode]:
        """ This method return a random node of a sub-tree starting from root_node.
        """

        try:
            return random.choice(self.all_operations)
        except IndexError:  # When the root is also a FeatureNode or an InvalidNode
            return None

        to_return = None

        if random.random() < deepness:
            to_return = root_node

            if not isinstance(to_return, OperationNode):
                to_return = root_node.father

        elif isinstance(root_node, OperationNode):
            to_return = self._select_random_node(
                root_node=random.choice(root_node.operands), deepness=deepness)
        else:  # The root_node is a FeatureNode or an InvalidNode. Returning its father.
            to_return = root_node.father

        return to_return

    def simplify(self, inplace: bool = False):
        """ This method allow to simplify the structure of a program using a SymPy backend

        Args:
            inplace: set to True to overwrite the current program with the simplified one
        """
        from symbolic_regression.simplification import extract_operation

        def simplify_program(program: Program) -> Program:
            """ This function simplify a program using a SymPy backend

            try: the root node of the program, not the Program object

            """
            try:
                logging.debug(f'Simplifying program {program}')
                simplified = sympy.simplify(
                    program.program, rational=True, inverse=True)

                logging.debug(
                    f'Extracting the program tree from the simplified')

                new_program = extract_operation(
                    element=simplified, father=None)

                logging.debug(f'Simplified program {new_program}')

                return new_program

            except UnboundLocalError:
                return program.program
            except sympy.core.sympify.SympifyError:
                return program.program

        if inplace:
            self.program = simplify_program(self)
            return self

        simp = deepcopy(self)
        simp.program = simplify_program(self)

        return simp

    # GENETIC OPERATIONS
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
            inplace: to replace the program in the current object or to return a new one
        """

        if not other:
            other = Program(
                operations=self.operations,
                features=self.features,
                const_range=self.const_range,
                program=self.program,
                parsimony=self.parsimony,
                parsimony_decay=self.parsimony_decay
            )
            other.init_program()

        if self.complexity == 1 or other.complexity == 1:
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

        cross_over_point2.father = cross_over_point1.father

        if cross_over_point2.father:
            cross_over_point2.father.operands[cross_over_point2.father.operands.index(
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

    def mutate(self, inplace: bool = False):
        """ This method perform a mutation on a random node of the current program

        A mutation is a random generation of a new sub-tree replacing another random
        sub-tree from the current program.

        Args:
            inplace: to replace the program in the current object or to return a new one
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
            return new

        offspring = deepcopy(self.program)

        mutate_point = self._select_random_node(
            root_node=offspring, deepness=.3)

        if not mutate_point:
            mutate_point = offspring

        try:
            child_to_mutate = random.randrange(mutate_point.arity)
            mutated = self._generate_tree(
                father=mutate_point,
                depth=int(self.program_depth/2)
            )
            #logging.debug(f'\n\nMutating this\n\n{mutate_point.operands[child_to_mutate]}\n\nin this\n\n{mutated}\n\n')

            mutate_point.operands[child_to_mutate] = mutated
        except ValueError:  # Case in which the tree has depth 0 with a FeatureNode as root
            pass
            #logging.debug(f'No child to mutate found in {mutate_point}')

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

    def insert_node(self, inplace: bool = False):
        """ This method allow to insert a FeatureNode in a random spot in the program

        The insertion of a OperationNode must comply with the arity of the existing
        one and must link to the existing operands.

        Args:
            inplace: to replace the program in the current object or to return a new one
        """
        offspring = deepcopy(self.program)
        mutate_point = self._select_random_node(root_node=offspring)

        if mutate_point:
            mutate_father = mutate_point.father
        else:  # When the mutate point is None, can happen when program is only a FeatureNode
            new = Program(operations=self.operations,
                          features=self.features,
                          const_range=self.const_range,
                          parsimony=self.parsimony,
                          parsimony_decay=self.parsimony_decay)

            new.init_program()
            return new

        new_node = self._generate_tree(father=mutate_father, depth=1)

        if mutate_father:  # Can be None if it is the root
            # Is a new tree of only one OperationNode
            mutate_father.operands[mutate_father.operands.index(
                mutate_point)] = new_node

        # Choose a random children to attach the previous mutate_point
        # The new_node is already a tree with depth = 1 so, in case of arity=2
        # operations the other operator is already set.
        new_node.operands[random.randint(0, new_node.arity-1)] = mutate_point
        mutate_point.father = new_node

        # If the new_node is also the new root, offspring need to be updated.
        if not mutate_father:
            offspring = new_node

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

    def delete_node(self, inplace: bool = False):
        """ This method delete a random OperationNode from the program.

        It selects a random children of the deleted node to replace itself
        as child of its father.

        Args:
            inplace: to replace the program in the current object or to return a new one
        """
        offspring = deepcopy(self.program)
        mutate_point = self._select_random_node(root_node=offspring)

        if mutate_point:
            mutate_father = mutate_point.father
        else:  # When the mutate point is None, can happen when program is only a FeatureNode
            return self

        # Case in which there is only a FeatureNode as root
        if not isinstance(mutate_point, OperationNode):
            return self

        mutate_child = random.choice(mutate_point.operands)

        mutate_child.father = mutate_father

        logging.debug(
            f'\\nMutate Father\n{mutate_father}\n\nMutate Child\n{mutate_child}')

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

    def mutate_leaf(self, inplace: bool = False):
        """ This method select a random FeatureNode and change the associated feature

        The new FeatureNode is randomly generated among all features and also constants.

        Args:
            inplace: to replace the program in the current object or to return a new one
        """
        offspring = deepcopy(self)

        feats = offspring.get_features(return_objects=True)

        mutate_father = None
        if feats:
            mutate_point = random.choice(feats)

            if isinstance(mutate_point, FeatureNode):
                mutate_father = mutate_point.father
            else:  # When the program is only a FeatureNode mutate_point is a Program
                mutate_father = mutate_point.program.father

        ''' To force the mutation in case, by chance, the same'''
        new_feature = offspring._generate_tree(
            depth=0, father=mutate_father)

        if mutate_father:
            mutate_father.operands[mutate_father.operands.index(
                mutate_point)] = new_feature
        else:
            offspring.program = new_feature

        if inplace:
            self.program = offspring.program
            return self

        new = Program(program=offspring.program, operations=self.operations,
                      features=self.features, const_range=self.const_range,
                      parsimony=self.parsimony, parsimony_decay=self.parsimony_decay)

        return new

    def mutate_operator(self, inplace: bool = False):
        offspring = deepcopy(self.program)

        mutate_point = self._select_random_node(root_node=offspring)

        if not mutate_point:  # Only a FeatureNode without any OperationNode
            new = Program(program=offspring, operations=self.operations,
                          features=self.features, const_range=self.const_range,
                          parsimony=self.parsimony, parsimony_decay=self.parsimony_decay)

            return new

        new_operation = random.choice(self.operations)
        while new_operation['arity'] != mutate_point.arity:
            new_operation = random.choice(self.operations)

        mutate_point.operation = new_operation['func']
        mutate_point.format_str = new_operation['format_str']
        mutate_point.format_tf = new_operation['format_tf']

        if inplace:
            self.program = offspring
            return self

        new = Program(program=offspring, operations=self.operations,
                      features=self.features, const_range=self.const_range,
                      parsimony=self.parsimony, parsimony_decay=self.parsimony_decay)

        return new
