import pickle
from typing import List
import zlib

from symbolic_regression.Program import Program

def compress(object):
    serialized_data = pickle.dumps(object)
    compressed_data = zlib.compress(serialized_data)
    return compressed_data


def decompress(compressed_data):
    serialized_data = zlib.decompress(compressed_data)
    object = pickle.loads(serialized_data)
    return object


class Population(List):

    def __init__(self, *args):
        super().__init__(*args)

    def as_binary(self):
        p = Population()
        for index, individual in enumerate(self):
            if isinstance(individual, Program):
                individual.programs_dominated_by = list()
                individual.programs_dominates = list()
                p.append(compress(individual))
        return p
    
    def as_program(self):
        p = Population()
        for index, individual in enumerate(self):
            if isinstance(individual, bytes):
                p.append(decompress(individual))
        return p