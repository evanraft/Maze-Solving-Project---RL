from abc import ABC, abstractmethod

class AbstractModel(ABC):
    def __init__(self, maze, **kwargs):
        self.environment = maze
        self.name = kwargs.get("name", "model")
