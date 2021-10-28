from abc import ABC
from typing import Union

import pandas as pd
from pandas.core.algorithms import isin


class Node(ABC):
    """ A node can represent an operation or a feature in a binary tree
    """

    depth: int

    def __init__(self, depth: int, father = None) -> None:
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

    def __init__(self, operation: callable, arity: int, format_str: str, depth: int, father) -> None:

        super().__init__(depth, father)

        self.operation = operation
        self.arity = arity
        self.format_str = format_str

        self.operands = []

    def __repr__(self) -> str:
        return self.render()

    def add_operand(self, operand: Node) -> None:
        """ Allow to ad a new operand, which can be any type of Node.
        """
        if len(self.operands) > self.arity + 1:
            raise AttributeError(
                f'This operation support only {self.arity} operands: {self.arity} given.')

        self.operands.append(operand)

    def render(self, data: Union[dict, None] = None) -> str:
        """ This method render the string of the program according to the formatting rules of its operations
        """
        return self.format_str.format(*[node.render(data=data) for node in self.operands])

    def evaluate(self, data: Union[dict, pd.Series, pd.DataFrame]) -> Union[int, float]:
        """ This method recursively calls the operations callable to evaluate the result of the program on data

        Each OperationNode has an operation callable function that receive operands as arguments.
        The operands can be other OperationNode in case the function goes deeply, or FeaturesNode in case
        that branch of the tree terminate here.
        """
        try:
            result = None
            if isinstance(data, dict):
                result = self.operation(*[node.evaluate(data=data) for node in self.operands])

            elif isinstance(data, pd.Series):
                result = self.operation(*[node.evaluate(data=data) for node in self.operands])

            elif isinstance(data, pd.DataFrame):
                result = list()
                for _, row in data.iterrows():
                    result.append(self.operation(*[node.evaluate(data=row) for node in self.operands]))

            else:
                raise TypeError(f'Evaluation supports only data as dict, pd.Series or pd.DataFrame objects')
        except ValueError:
            print(f'VALUE ERROR {self.operation}, {self.operands}')

        return result

class FeatureNode(Node):
    """ A FeatureNode represent a leaf of the binary tree of the program and is always a feature

    As weel as OperationNode, FeatureNodes have a depth.
    """

    def __init__(self, feature: Union[str, float], depth: int, father: Union[OperationNode, None] = None, is_constant: bool = False) -> None:
        """
        """
        super().__init__(depth , father)

        self.feature = feature
        self.arity = 0  # because it is a constand and not an operator
        self.is_constant = is_constant

    def __repr__(self) -> str:
        return self.render()

    def render(self, data: Union[dict, None] = None) -> str:
        """ This method render the string representation of this FeatureNode

        If data is provided, the rendering consist of the value of the datapoint of the feature of this
        FeatureNode.
        If data is not provided, the rendering will be the name of the feature.
        """
        if self.is_constant:
            return str(self.feature)

        if data:  # Case in which I render the value of the feature in the datapoint instead of its name
            return self.evaluate(data=data)
        return self.feature

    def evaluate(self, data) -> Union[int, float]:
        """ This function evaluate the value of a FeatureNode, which is the datapoint passed as argument
        
        The data argument needs to be accessible by the name of the feature of this node.
        This is a FeatureNode, the end of a branch of a tree, so its evaluation is simply the value of that
        feature in a specific datapoint.
        """
        if self.is_constant:
            return self.feature

        return data[self.feature]
