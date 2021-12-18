from abc import ABC
from typing import Union

import pandas as pd
import tensorflow as tf


class Node(ABC):
    """ A node can represent an operation or a feature in a binary tree
    """

    depth: int

    def __init__(self, depth: int, father=None) -> None:
        self.depth = depth
        self.father = father


class OperationNode(Node):
    """ An OperationNode represents an arithmetic operation.
    It is characterized by the callable of the operation itself, the arity of the operation
    (i.e., the number of operands accepted), and the format to represent the operation in
    the formula.
    Finally there is the list with the operands; it must be long at most as arity.
    The operands can be OperationNode, if the formula continues deeply, or FeatureNode if
    the formula terminate and a feature is chosen.
    """

    def __init__(self, operation: callable, format_tf: str, arity: int, format_str: str, depth: int, father) -> None:
        """ To initialize an OperationNode

        Args:
            operation: The callable function of the operation
            format_tf: The string representation of the tensorflow equivalent for constants optimization
            arity: The number of arguments that the operation accepts
            format_str: The string representation of the operation
            depth: The depth in the program to which this node is
            father: The node above the current one (None for the root_node of the program)
        """
        super().__init__(depth, father)

        self.operation = operation
        self.arity = arity
        self.format_str = format_str
        self.format_tf = format_tf

        self.operands = []

    def __repr__(self) -> str:
        """ This call the render function of the operation to print it in a readable way
        """
        return self.render()

    def add_operand(self, operand: Node) -> None:
        """ Allow to ad a new operand, which can be any type of Node.

        An operand can be any type of node, another OperationNode or a FeatureNode.
        The number of operands must be consistent with the arity of the OperationNode.

        Args:
            operand: The generic node that will be child to the current one
        """
        if len(self.operands) > self.arity + 1:
            raise AttributeError(
                f'This operation support only {self.arity} operands: {self.arity} given.')

        self.operands.append(operand)
    
    def _get_complexity(self, base_complexity=0):
        """ This method recursively increment the complexity count of this program
        """
        
        base_complexity += 1  # Count for this operation contribution
        
        for child in self.operands:
            base_complexity = child._get_complexity(base_complexity=base_complexity)
        
        return base_complexity

    def render(self, data: Union[dict, pd.Series, pd.DataFrame, None] = None, format_tf: bool = False) -> str:
        """ This method render the string of the program according to the formatting rules of its operations

        This call recursively itself untile the terminal nodes are reached.
        """
        if format_tf:
            return self.format_tf.format(*[node.render(data=data, format_tf=True) for node in self.operands])
        return self.format_str.format(*[node.render(data=data) for node in self.operands])

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This method recursively calls the operations callable to evaluate the result of the program on data

        Each OperationNode has an operation callable function that receive operands as arguments.
        The operands can be other OperationNode in case the function goes deeply,
        or FeatureNode in case that branch of the tree terminate here.

        Args:
            data: The data on which to evaluate this node
        """
        result = None
        if isinstance(data, dict):
            result = self.operation(
                *[node.evaluate(data=data) for node in self.operands])

        elif isinstance(data, pd.Series):
            result = self.operation(
                *[node.evaluate(data=data) for node in self.operands])

        elif isinstance(data, pd.DataFrame):
            result = list()
            for _, row in data.iterrows():
                result.append(self.operation(
                    *[node.evaluate(data=row) for node in self.operands]))

        else:
            raise TypeError(
                f'Evaluation supports only data as dict, pd.Series or pd.DataFrame objects')

        return result

    def _get_constants(self, const_list: list):
        """
        """
        for child in self.operands:
            if isinstance(child, OperationNode):
                const_list = child._get_constants(const_list=const_list)
            elif child.is_constant:
                const_list += [child]

        return const_list

    def _get_features(self, features_list: list):
        """
        """
        for child in self.operands:
            if isinstance(child, OperationNode):
                features_list = child._get_features(features_list=features_list)
            elif not child.is_constant:
                features_list += [child.feature]

        return features_list


class FeatureNode(Node):
    """ A FeatureNode represent a terminal node of the binary tree of the program and is always a feature or a constant
    """

    def __init__(self, feature: Union[str, float], depth: int, father: Union[OperationNode, None] = None, is_constant: bool = False) -> None:
        """ To initalize this FeatureNode

        Args:
            feature: The name of the feature from the training dataset or the numerical value of a constant
            depth: The depth to which this node is in the program tree
            father: The father node of the current FeatureNode
            is_constant: To specify whether this is a feature from the training data or a numerical constant
        """
        super().__init__(depth, father)

        self.feature = feature
        self.arity = 0  # because it is a constand and not an operator
        self.is_constant = is_constant
        self.index = None

    def __repr__(self) -> str:
        """ To print the current node in a readable way
        """
        return f'FeatureNode({self.render()})'

    def _get_complexity(base_complexity=0):
        """ This method increase the complexity of the program by 1
        It is usually called by an OperationNode _get_complexity which
        accounts for the rest of the program.
        """
        return base_complexity + 1

    def render(self, data: Union[dict, pd.Series, None] = None, format_tf = False) -> str:
        """ This method render the string representation of this FeatureNode

        If data is provided, the rendering consist of the value of the datapoint of the feature of this
        FeatureNode.
        If data is not provided, the rendering will be the name of the feature.
        If this node is a constant value (is_constant==True) then that numerical value is returned
        """

        if self.is_constant:
            if format_tf:
                return f'constants[{self.index}]'
            return str(self.feature)

        if data is not None:  # Case in which I render the value of the feature in the datapoint instead of its name
            if format_tf:
                return f'X[{list(data.index).index(self.feature)}]'
            return self.evaluate(data=data)

        return self.feature

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame, None] = None) -> Union[int, float]:
        """ This function evaluate the value of a FeatureNode, which is the datapoint passed as argument

        The data argument needs to be accessible by the name of the feature of this node.
        This is a FeatureNode, the end of a branch of a tree, so its evaluation is simply the value of that
        feature in a specific datapoint.

        In the recursive call stack of the evaluation of a program, this is the terminal call and
        always return or the constant numerical value (if is_constant==True) or the value of the feature
        from the data passed as argument.

        Args:
            data: The data on which to evaluate the node
        """

        result = None

        if self.is_constant:
            result = self.feature

        elif data is not None and (isinstance(data, pd.Series) or isinstance(data, dict)):
            result = data[self.feature]

        else:
            raise TypeError(
                f'Non constants FeatureNodes need data to be evaluated')

        return result
