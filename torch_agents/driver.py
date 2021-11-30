from abc import ABC, abstractmethod
from torch_agents.observer import Observer
from torch_agents.policy import Policy
from torch_agents.utils import resize_frame, empty_frame

import torch
import numpy as np


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


class ALEDriver(Driver):
    def __init__(self, *args, skip=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip = skip

    def _step(self) -> bool:
        action = self.policy.select_action(self.state)
        self.steps += 1
        for _ in range(self.skip + 1):
            observation, reward, done, _ = self.env.step(action.item())
            self._acc_reward += reward
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

            if not done:
                next_state = torch.tensor(resize_frame(observation), device=self.device, dtype=torch.uint8).unsqueeze(0)
            else:
                next_state = None

            self.observer.save(self.state[-1, :, :], action, reward, next_state)
            if not done:
                self.state = torch.cat((self.state[1:, :, :], next_state), 0)

            if done:
                self.append_rewards()
                self.reset()
                return True

        return False

    def _reset(self):
        self.steps = 0
        self._acc_reward = 0
        observation = resize_frame(self.env.reset())
        states = np.array([empty_frame(), empty_frame(), empty_frame(), observation])
        self.state = torch.tensor(states, device=self.device, dtype=torch.uint8)
