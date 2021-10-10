from abc import ABC, abstractmethod
import random
import torch


"""
    Policy Class
    
    Returns an action from the action space based on policy
"""


class Policy(ABC):
    def __init__(self, n_observations: int, n_actions: int, device: str):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device
        self.name = "Base"

    def select_action(self, state):
        return self._select_action(state)

    @abstractmethod
    def _select_action(self, state):
        raise NotImplementedError

    def __str__(self):
        return self.name


class RandomPolicy(Policy):
    def __init__(self, n_actions: int, device: str):
        super().__init__(0, n_actions, device)
        self.name = "Random"

    def _select_action(self, state=None):
        return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)


class GreedyPolicy(Policy):
    def __init__(self, model, device: str):
        super().__init__(model.shape[0], model.shape[1], device)
        self.model = model
        self.name = "Greedy"

    def _select_action(self, state):
        return self.model.predict(state, self.device)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, model, device: str, eps_start: float = 0.9,
                 eps_end: float = 0.01, eps_decay: float = 0.995):
        super().__init__(model.shape[0], model.shape[1], device)
        self.model = model
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.name = "Epsilon Greedy"

    def _select_action(self, state):
        self.eps = max(self.eps * self.eps_decay, self.eps_end)
        if random.random() >= self.eps:
            action = self.model.predict(state, self.device)
        else:
            action = torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        return action
