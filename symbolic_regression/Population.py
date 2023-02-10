import pickle
from typing import List

from symbolic_regression.Program import Program


class Population(List):

    def __init__(self, *args):
        super().__init__(*args)

    def as_binary(self):
        p = Population()
        for index, individual in enumerate(self):
            if isinstance(individual, Program):
                individual.programs_dominated_by = list()
                individual.programs_dominates = list()
                p.append(pickle.dumps(individual))
        return p
    
    def as_program(self):
        p = Population()
        for index, individual in enumerate(self):
            if isinstance(individual, bytes):
                p.append(pickle.loads(individual))
        return p