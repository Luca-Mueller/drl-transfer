from abc import ABC, abstractmethod
from torch_agents.replay_buffer import ReplayBuffer
import pandas as pd


"""
    Observer Class
    
    Collects transitions from a driver and saves them
"""


class Observer(ABC):
    def save(self, *args):
        self._save(*args)

    @abstractmethod
    def _save(self, *args):
        raise NotImplementedError


class DummyObserver(Observer):
    def _save(self, *args):
        pass


class BufferObserver(Observer):
    def __init__(self, replay_buffer: ReplayBuffer):
        self._buffer = replay_buffer

    def _save(self, *args):
        self._buffer.push(*args)


# Not efficient
class CSVObserver(Observer):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.df = pd.DataFrame()

    def _save(self, *args):
        args = [a.cpu().detach().numpy() for a in [*args] if a is not None]
        series = pd.Series([*args])
        self.df = self.df.append(series, ignore_index=True)
        self.df.to_csv(self.file_name)


class MultiObserver(Observer):
    def __init__(self, *args):
        self.observers = [*args]

    def _save(self, *args):
        for observer in self.observers:
            observer.save(*args)