from abc import abstractmethod


class MOSRCallbackBase:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    @abstractmethod
    def on_callback_set_init(self, **kwargs):
        pass

    @abstractmethod
    def on_initialization_start(self, **kwargs):
        pass

    @abstractmethod
    def on_initialization_end(self, **kwargs):
        pass

    @abstractmethod
    def on_generation_start(self, **kwargs):
        pass

    @abstractmethod
    def on_generation_end(self, **kwargs):
        pass

    @abstractmethod
    def on_offspring_generation_start(self, **kwargs):
        pass

    @abstractmethod
    def on_offspring_generation_end(self, **kwargs):
        pass

    @abstractmethod
    def on_refill_start(self, **kwargs):
        pass

    @abstractmethod
    def on_refill_end(self, **kwargs):
        pass

    @abstractmethod
    def on_pareto_front_computation_start(self, **kwargs):
        pass

    @abstractmethod
    def on_pareto_front_computation_end(self, **kwargs):
        pass

    @abstractmethod
    def on_convergence(self, **kwargs):
        pass

    @abstractmethod
    def on_training_completed(self, **kwargs):
        pass
