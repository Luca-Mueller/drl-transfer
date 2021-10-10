from abc import ABC, abstractmethod
from collections import deque, namedtuple
import copy
import random
from typing import Union


"""
    ReplayBuffer Class
    
    Stores and samples transitions
"""


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))


class ReplayBuffer(ABC):
    def __init__(self, transition_type=Transition):
        self.transition_type = transition_type
        self.memory = None
        self.capacity = None
        self.name = "Base"

    def push(self, *args):
        return self._push(*args)

    def sample(self, batch_size: int) -> list:
        return self._sample(batch_size)

    def to(self, device: str):
        self.memory = deque([[tensor.to(device) if tensor is not None else None for tensor in transition]
                             for transition in self.memory], maxlen=self.memory.maxlen)

    def _push(self, *args):
        self.memory.append(self.transition_type(*args))

    def _sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    @abstractmethod
    def _len(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return self._len()

    def __repr__(self) -> str:
        return self.name


class SimpleReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, *args, **kwargs):
        super(SimpleReplayBuffer, self).__init__(*args, **kwargs)
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.name = "SimpleReplayBuffer"

    def _len(self):
        return len(self.memory)


class SplitReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, transfer_buffer: Union[ReplayBuffer, deque, list], *args, **kwargs):
        super(SplitReplayBuffer, self).__init__(*args, **kwargs)
        if isinstance(transfer_buffer, ReplayBuffer):
            self.transfer_memory = copy.deepcopy(list(transfer_buffer.memory))
        elif isinstance(transfer_buffer, deque):
            self.transfer_memory = copy.deepcopy(list(transfer_buffer))
        elif isinstance(transfer_buffer, list):
            self.transfer_memory = copy.deepcopy(transfer_buffer)
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity
        self.name = "SplitReplayBuffer"

    def _sample(self, batch_size: int) -> list:
        fill_size = self.memory.maxlen - len(self.memory)
        fill_sample = random.sample(self.transfer_memory, fill_size)
        return random.sample(list(self.memory) + fill_sample, batch_size)

    def _len(self):
        return len(self.memory) + len(self.transfer_memory)


class FilledReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, transfer_buffer: Union[ReplayBuffer, deque, list], *args, **kwargs):
        super(FilledReplayBuffer, self).__init__(*args, **kwargs)
        assert len(transfer_buffer) >= capacity, "transfer buffer too small: needs to be at least same as capacity"
        if isinstance(transfer_buffer, ReplayBuffer):
            transfer_memory = copy.deepcopy(list(transfer_buffer.memory))
        elif isinstance(transfer_buffer, deque):
            transfer_memory = copy.deepcopy(list(transfer_buffer))
        elif isinstance(transfer_buffer, list):
            transfer_memory = copy.deepcopy(transfer_buffer)
        random.shuffle(transfer_memory)
        self.memory = deque(transfer_memory[:capacity], maxlen=capacity)
        self.capacity = capacity
        self.name = "FilledReplayBuffer"

    def _len(self):
        return len(self.memory)


class LimitedReplayBuffer(SimpleReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(LimitedReplayBuffer, self).__init__(*args, **kwargs)
        self.name = "LimitedReplayBuffer"

    def _push(self, *args):
        if len(self.memory) != self.capacity:
            self.memory.append(self.transition_type(*args))


class LimitedSplitReplayBuffer(SplitReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(LimitedSplitReplayBuffer, self).__init__(*args, **kwargs)
        self.name = "LimitedSplitReplayBuffer"

    def _push(self, *args):
        if len(self.memory) != self.capacity:
            self.memory.append(self.transition_type(*args))


# TODO: fix push for full buffer
class LimitedFilledReplayBuffer(FilledReplayBuffer):
    def __init__(self, *args, **kwargs):
        super(LimitedFilledReplayBuffer, self).__init__(*args, **kwargs)
        self.name = "LimitedFilledReplayBuffer"

    def _push(self, *args):
        if len(self.memory) != self.capacity:
            self.memory.append(self.transition_type(*args))
