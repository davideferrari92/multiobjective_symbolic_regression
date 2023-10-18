import time
import plotly.graph_objs as go
from symbolic_regression.SymbolicRegressor import SymbolicRegressor

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
        
        print(kwargs)

    def _hypervolume(self):
        before = time.perf_counter()
        self.sr.compute_hypervolume(exclusive=True)
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