from typing import Dict, List

from symbolic_regression.federated.strategies.BaseStrategy import BaseStrategy
from symbolic_regression.Node import Node


class SymbolicFedAvg(BaseStrategy):
    def __init__(self, mode: str) -> None:
        super().__init__()

        self.mode = mode  # 'client' or 'serer'

    def execute(self, populations: Dict[str, List[Node]]):
        # TODO
        
        if self.mode == 'server':
            pass
        
        elif self.mode == 'client':
            pass
        
        ''' This returns a population object to be sent back to the clients
        '''
        return populations
