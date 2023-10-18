
from symbolic_regression.SymbolicRegressor import SymbolicRegressor
from symbolic_regression.callbacks.CallbackBase import MOSRCallbackBase


class MOSRCallbackSaveCheckpoint(MOSRCallbackBase):

    def __init__(self, checkpoint_file, checkpoint_frequency, checkpoint_overwrite, **kwargs):
        super().__init__(**kwargs)

        self.checkpoint_file = checkpoint_file
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_overwrite = checkpoint_overwrite

    def on_generation_end(self):
        if not self.checkpoint_frequency or self.checkpoint_frequency == -1:
            return

        if self.sr.generation % self.checkpoint_frequency == 0:
            self.sr: SymbolicRegressor
            self.sr.save_model(file=self.checkpoint_file,
                               checkpoint_overwrite=self.checkpoint_overwrite)

            print(f"Checkpoint saved at generation {self.sr.generation}.")

    def on_training_completed(self):
        if self.checkpoint_frequency == -1 or self.checkpoint_frequency > 0:
            self.sr: SymbolicRegressor
            self.sr.save_model(file=self.checkpoint_file,
                               checkpoint_overwrite=self.checkpoint_overwrite)

            print(f"Checkpoint saved at end of training.")
