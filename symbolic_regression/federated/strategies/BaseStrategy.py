from abc import abstractmethod


class BaseStrategy:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def execute(self, populations: dict):
        pass