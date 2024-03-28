import copy
import time
import plotly.graph_objs as go
from symbolic_regression.SymbolicRegressor import SymbolicRegressor, compress

from symbolic_regression.callbacks.CallbackBase import MOSRCallbackBase


class MOSRStatisticsComputation(MOSRCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.statistics_computation_frequency = kwargs.get(
            'statistics_computation_frequency', -1)
        self.compute_hypervolume = kwargs.get('compute_hypervolume', False)
        self.compute_tree_diversity = kwargs.get(
            'compute_tree_diversity', False)
        self.compute_spearman_diversity = kwargs.get(
            'compute_spearman_diversity', False)

    def _hypervolume(self):
        before = time.perf_counter()
        self.sr.compute_hypervolume()
        self.sr.times.loc[self.sr.generation,
                          "time_hypervolume_computation"] = time.perf_counter() - before

    def _tree_diversity(self):
        before = time.perf_counter()
        self.sr.tree_diversity()
        self.sr.times.loc[self.sr.generation,
                          "time_tree_diversity_computation"] = time.perf_counter() - before

    def _spearman_diversity(self, data):
        before = time.perf_counter()
        self.sr.spearman_diversity(data)
        self.sr.times.loc[self.sr.generation,
                          "time_spearman_diversity_computation"] = time.perf_counter() - before

    def on_pareto_front_computation_end(self, **kwargs):
        self.sr: 'SymbolicRegressor'

        data = kwargs.get('data', None)

        if (self.sr.generation == 1) or \
            (self.statistics_computation_frequency == -1 and (self.sr.generation == self.sr.generations_to_train or self.sr.converged_generation)) or \
                (self.statistics_computation_frequency > 0 and self.sr.generation % self.statistics_computation_frequency == 0):

            if self.sr.verbose > 1:
                print(
                    f'Computing statistics for generation {self.sr.generation}')

            if self.compute_hypervolume:
                self._hypervolume()

            if self.compute_tree_diversity:
                self._tree_diversity()

            if self.compute_spearman_diversity:
                self._spearman_diversity(data)

        del data


class MOSRHistory(MOSRCallbackBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.history_fpf_frequency = kwargs.get('history_fpf_frequency', -1)
        self.history_best_fitness_frequency = kwargs.get(
            'history_best_fitness_frequency', -1)

    def on_pareto_front_computation_end(self, **kwargs):

        if (self.sr.generation == 1) or \
            (self.history_fpf_frequency == -1 and (self.sr.generation == self.sr.generations_to_train or self.sr.converged_generation)) or \
                (self.history_fpf_frequency > 0 and self.sr.generation % self.history_fpf_frequency == 0):

            val_data = kwargs.get('val_data', None)

            self.sr: 'SymbolicRegressor'

            if self.sr.verbose > 1:
                print(
                    f'Saving FPF history for generation {self.sr.generation}')

            if not self.sr.first_pareto_front_history:
                self.sr.first_pareto_front_history = dict()

            self.sr.first_pareto_front_history[self.sr.generation] = compress(
                [p for p in self.sr.first_pareto_front])

        if (self.sr.generation == 1) or \
            (self.history_best_fitness_frequency == -1 and (self.sr.generation == self.sr.generations_to_train or self.sr.converged_generation)) or \
                (self.history_best_fitness_frequency > 0 and self.sr.generation % self.history_best_fitness_frequency == 0):

            if not self.sr.best_history.get('training'):
                self.sr.best_history['training'] = dict()
            if not self.sr.best_history['training'].get(self.sr.generation):
                self.sr.best_history['training'][self.sr.generation] = dict()

            val_data = kwargs.get('val_data', None)

            if val_data is not None and not self.sr.best_history.get('validation'):
                self.sr.best_history['validation'] = dict()
            if val_data is not None and not self.sr.best_history['validation'].get(self.sr.generation):
                self.sr.best_history['validation'][self.sr.generation] = dict()

            for fitness in self.sr.fitness_functions:
                try:
                    if fitness.minimize:
                        best_p = min([p for p in self.sr.first_pareto_front if p.is_valid],
                                    key=lambda obj: obj.fitness.get(fitness.label, +float('inf')))
                        if val_data is not None:
                            best_p_val = min([p for p in self.sr.first_pareto_front if p.is_valid],
                                            key=lambda obj: obj.fitness_validation.get(fitness.label, +float('inf')))
                    else:
                        best_p = max([p for p in self.sr.first_pareto_front if p.is_valid],
                                    key=lambda obj: obj.fitness.get(fitness.label, -float('inf')))
                        if val_data is not None:
                            best_p_val = max([p for p in self.sr.first_pareto_front if p.is_valid],
                                            key=lambda obj: obj.fitness_validation.get(fitness.label, -float('inf')))

                    self.sr.best_history['training'][self.sr.generation][fitness.label] = compress(
                        best_p)
                    if val_data is not None:
                        self.sr.best_history['validation'][self.sr.generation][fitness.label] = compress(
                            best_p_val)
                except ValueError:  # Case of empty min()
                    pass
