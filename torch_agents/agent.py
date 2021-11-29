from abc import ABC, abstractmethod
from itertools import count
import copy
import numpy as np
import torch
from torch import optim

from torch_agents.models import VModel, V2Model
from torch_agents.replay_buffer import ReplayBuffer, Transition
from torch_agents.policy import Policy, GreedyPolicy, RandomPolicy
from torch_agents.driver import SimpleDriver, ALEDriver
from torch_agents.observer import Observer, BufferObserver, DummyObserver


"""
    Agent Class
    
    Train or play in an environment
"""


class Agent(ABC):
    def __init__(self, model, replay_buffer: ReplayBuffer, policy: Policy, optimizer, loss_function,
                 observer: Observer = None, gamma: float = 0.95, target_update_period: int = 10,
                 device: str = "cpu"):
        self.policy_model = model
        self.replay_buffer = replay_buffer
        self.driver_type = SimpleDriver
        if "Frame" in replay_buffer.name:
            self.driver_type = ALEDriver
        self.policy = policy
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.observer = BufferObserver(replay_buffer) if observer is None else observer
        self.gamma = gamma
        self.target_update_period = target_update_period
        self.device = device
        self.episodes_trained = 0
        self.name = "Base"

    def train(self, env, n_episodes: int, max_steps: int = None, batch_size: int = 32, warm_up_period: int = 0,
              visualize: bool = False) -> np.array:
        return self._train(env, n_episodes, max_steps, batch_size, warm_up_period, visualize)

    @torch.no_grad()
    def play(self, env, n_episodes: int, max_steps: int = None, observer: Observer = None,
             visualize: bool = False) -> np.array:
        if observer is None:
            observer = DummyObserver()
        return self._play(env, n_episodes, max_steps, observer, visualize)

    def _train(self, env, n_episodes: int, max_steps: int, batch_size: int, warm_up_period: int,
               visualize: bool) -> np.array:
        driver = self.driver_type(env, RandomPolicy(self.policy_model.shape[1], self.device), self.observer, self.device)
        while len(self.replay_buffer) < warm_up_period:
            print(f"\rBufferSize: {len(self.replay_buffer):>6}/{self.replay_buffer.capacity:<6}", end="")
            driver.step()

        driver = self.driver_type(env, self.policy, self.observer, self.device)
        history = []
        for episode in range(n_episodes):
            for step in count():
                # TODO: Epsilon cannot be printed for policies other than EpsilonGreedyPolicy
                print(f"\rEpisode: {episode + 1:>5}/{n_episodes:<5}\t BufferSize: {len(self.replay_buffer):>6}/"
                      f"{self.replay_buffer.capacity:<6}\t Epsilon: {self.policy.eps:.4f}", end="")

                if visualize:
                    env.render()

                done = driver.step()
                self._optimize(batch_size)

                if done:
                    break

                # episode ends before agent finishes
                if (step + 1) == max_steps:
                    driver.append_rewards()
                    driver.reset()
                    break

            history.append(driver.reward_history[-1])

            self.episodes_trained += 1
            if (self.episodes_trained + 1) % self.target_update_period == 0:
                self._update_target()

        env.close()
        print("")
        return np.array(history)

    def _play(self, env, n_episodes: int, max_steps: int, observer: Observer, visualize: bool) -> np.array:
        driver = self.driver_type(env, GreedyPolicy(self.policy_model, device=self.device), observer, self.device)
        history = []

        for episode in range(n_episodes):
            print(f"\rEpisode: {(episode + 1):>5} of {n_episodes:<5}"
                  f"\t Score: {np.mean(history) if len(history) else 0.0:.2f}", end="")

            for step in count():
                if visualize:
                    env.render()

                done = driver.step()

                if done:
                    break

                if (step + 1) == max_steps:
                    driver.append_rewards()
                    driver.reset()
                    break

            history.append(driver.reward_history[-1])

        env.close()
        print("")
        return np.array(history)

    @abstractmethod
    def _update_target(self):
        raise NotImplementedError

    @abstractmethod
    def _optimize(self, batch_size: int):
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)
        self.target_model = copy.deepcopy(self.policy_model).to(self.device)
        self.target_model.eval()
        self.name = "DQN"

    def _update_target(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def _optimize(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss_function(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


class DDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super(DDQNAgent, self).__init__(*args, **kwargs)
        self.name = "DDQN"

    def _optimize(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        with torch.no_grad():
            max_action_values = self.policy_model(non_final_next_states).max(1)[1].unsqueeze(1)

        next_state_values = torch.zeros(batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_action_values).reshape(-1)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.loss_function(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


class DQVAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQVAgent, self).__init__(*args, **kwargs)
        self.lr = self.optimizer.param_groups[0]['lr']
        self._build_v_model()
        self.name = "DQV"

    def _update_target(self):
        self.v_target_model.load_state_dict(self.v_policy_model.state_dict())

    def _optimize(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute TD targets
        y_t = torch.zeros(batch_size, device=self.device)
        y_t[non_final_mask] = self.v_target_model(non_final_next_states).squeeze(1).detach()
        y_t = (self.gamma * y_t) + reward_batch
        y_t = y_t.unsqueeze(1)

        # predict Q(s_t, a_t) and V(s_t)
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        state_values = self.v_policy_model(state_batch)

        # calculate loss
        l_theta = self.loss_function(state_action_values, y_t)
        l_phi = self.loss_function(state_values, y_t)

        # update Q model
        self.optimizer.zero_grad()
        l_theta.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # update V model
        self.v_optimizer.zero_grad()
        l_phi.backward()

        for param in self.v_policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.v_optimizer.step()

    def _build_v_model(self):
        n_observations = self.policy_model.shape[0]
        fc1_nodes = self.policy_model.fc1_nodes
        fc2_nodes = self.policy_model.fc2_nodes

        self.v_policy_model = VModel(n_observations, fc1_nodes, fc2_nodes).to(self.device)
        self.v_target_model = copy.deepcopy(self.v_policy_model).to(self.device)
        self.v_target_model.eval()
        self.v_optimizer = optim.Adam(self.v_policy_model.parameters(), lr=self.lr)


class DQV2Agent(DQVAgent):
    def __init__(self, *args, **kwargs):
        super(DQV2Agent, self).__init__(*args, **kwargs)
        self.name = "DQV2"

    def _optimize(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # compute TD targets
        y_t = torch.zeros(batch_size, device=self.device)
        y_t[non_final_mask] = self.v_target_model(non_final_next_states).squeeze(1).detach()
        y_t = (self.gamma * y_t) + reward_batch
        y_t = y_t.unsqueeze(1)

        # update Q model
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)
        l_theta = self.loss_function(state_action_values, y_t)
        self.optimizer.zero_grad()
        l_theta.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        # update V model
        self.v_policy_model.freeze()
        state_values = self.v_policy_model(state_batch)
        l_phi = self.loss_function(state_values, y_t)
        self.v_optimizer.zero_grad()
        l_phi.backward()

        for param in self.v_policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.v_optimizer.step()
        self.v_policy_model.unfreeze()

    def _build_v_model(self):
        self.v_policy_model = V2Model(self.policy_model).to(self.device)
        self.v_target_model = copy.deepcopy(self.v_policy_model).to(self.device)
        self.v_target_model.eval()
        self.v_optimizer = optim.Adam(self.v_policy_model.parameters(), lr=self.lr)
