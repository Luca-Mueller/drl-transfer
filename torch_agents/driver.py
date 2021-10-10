from abc import ABC, abstractmethod
from torch_agents.observer import Observer
from torch_agents.policy import Policy

import torch


"""
    Driver Class

    Collects transitions from an environment based on policy and sends them to observer
"""


class Driver(ABC):
    def __init__(self, env, policy: Policy, observer: Observer, device: str):
        self.env = env
        self.policy = policy
        self.observer = observer
        self.device = device
        self.reward_history = []
        self._acc_reward = None
        self.reset()

    def step(self) -> bool:
        return self._step()

    def reset(self):
        self._reset()

    def append_rewards(self):
        self.reward_history.append(self._acc_reward)

    @abstractmethod
    def _step(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _reset(self):
        raise NotImplementedError


class SimpleDriver(Driver):
    def _step(self) -> bool:
        self.steps += 1
        action = self.policy.select_action(self.state)
        observation, reward, done, _ = self.env.step(action.item())
        self._acc_reward += reward
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

        if not done:
            next_state = torch.tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)
        else:
            next_state = None

        self.observer.save(self.state, action, reward, next_state)
        self.state = next_state

        if done:
            self.append_rewards()
            self.reset()
            return True

        return False

    def _reset(self):
        self.steps = 0
        self._acc_reward = 0
        observation = self.env.reset()
        self.state = torch.tensor(observation, device=self.device, dtype=torch.float32).unsqueeze(0)
