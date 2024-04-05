import logging
import numpy as np
import pandas as pd


class SymbolicEnsembler:

    def __init__(self, programs_selection) -> None:
        self.programs_selection = programs_selection

    def predict(self, data):
        predictions = []
        for program in self.programs_selection:
            predictions.append(program.predict(data))
        return np.mean(predictions, axis=0)
    
    def compute_fitness(self, data, fitness_functions):
        predictions = pd.Series(self.predict(data))

        fitness_blueprint = {}

        for fitness_function in fitness_functions:
            try:
                if hasattr(fitness_function, 'threshold'):
                    predictions = np.where(predictions > fitness_function.threshold, 1, 0)

                fitness_blueprint[fitness_function.label] = fitness_function.evaluate(
                    program = None, data = data, validation = False, pred = predictions)
            except Exception as e:
                logging.warning(f'Unable to compute fitness {fitness_function.label} {e}')
            
        return fitness_blueprint