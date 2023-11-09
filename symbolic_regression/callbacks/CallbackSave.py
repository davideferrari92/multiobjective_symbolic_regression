
from symbolic_regression.SymbolicRegressor import SymbolicRegressor
from symbolic_regression.callbacks.CallbackBase import MOSRCallbackBase


class MOSRCallbackSaveCheckpoint(MOSRCallbackBase):

    def __init__(self, checkpoint_file, checkpoint_frequency: int = -1, checkpoint_overwrite: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.checkpoint_file = checkpoint_file
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_overwrite = checkpoint_overwrite

    def on_generation_end(self):
        if not self.checkpoint_frequency or self.checkpoint_frequency == -1:
            return

        if self.sr.generation % self.checkpoint_frequency == 0:
            self.sr: SymbolicRegressor
            if self.sr.verbose > 1:
                print(
                    f'Saving checkpoint for generation {self.sr.generation}')
            self.sr.save_model(file=f'{self.checkpoint_file}.{self.sr.client_name}',
                               checkpoint_overwrite=self.checkpoint_overwrite)

    def on_training_completed(self):
        if self.checkpoint_frequency == -1 or self.checkpoint_frequency > 0:
            self.sr: SymbolicRegressor
            if self.sr.verbose > 1:
                print(
                    f'Saving checkpoint for training complete on generation {self.sr.generation}')
            self.sr.save_model(file=f'{self.checkpoint_file}.{self.sr.client_name}',
                               checkpoint_overwrite=self.checkpoint_overwrite)
