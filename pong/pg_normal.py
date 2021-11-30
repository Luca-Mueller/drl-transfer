import gym
import numpy as np
import collections
from colorama import init, Fore, Style

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import ConvDQN
from torch_agents.replay_buffer import Transition, SimpleFrameBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent, DQV2Agent
from torch_agents.plotting import plot_scores
from torch_agents.utils import AgentArgParser, ArgPrinter


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
       super(FireResetEnv, self).__init__(env)
       assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
       assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
       return self.env.step(action)

    def reset(self):
       self.env.reset()
       obs, _, done, _ = self.env.step(1)
       if done:
          self.env.reset()
       obs, _, done, _ = self.env.step(2)
       if done:
          self.env.reset()
       return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
           obs, reward, done, info = self.env.step(action)
           self._obs_buffer.append(obs)
           total_reward += reward
           if done:
               break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
       self._obs_buffer.clear()
       obs = self.env.reset()
       self._obs_buffer.append(obs)
       return obs


# initialize color / gym / device
init()
env = gym.make('Pong-v0')#.unwrapped
env = FireResetEnv(MaxAndSkipEnv(env))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
arg_parser = AgentArgParser()
args = arg_parser.parse_args()

# agent type
assert args.agent in ["DQN", "DDQN", "DQV", "DQV2"], f"invalid agent type '{args.agent}'"

# agent type
if args.agent == "DQN":
    agent_type = DQNAgent
elif args.agent == "DDQN":
    agent_type = DDQNAgent
elif args.agent == "DQV":
    agent_type = DQVAgent
elif args.agent == "DQV2":
    agent_type = DQV2Agent

# device
if args.cpu:
    device = torch.device("cpu")

# hyperparameters
LR = args.lr
BATCH_SIZE = args.batch_size
BUFFER_SIZE = args.buffer_size
GAMMA = args.gamma
EPS_START = args.epsilon_start
EPS_END = args.epsilon_end
EPS_DECAY = args.epsilon_decay
N_EPISODES = args.episodes
MAX_STEPS = args.max_steps if args.max_steps else env.spec.max_episode_steps
args.max_steps = MAX_STEPS
WARM_UP = args.warm_up
TARGET_UPDATE = args.target_update

# visualization
VIS_TRAIN = args.vv
VIS_EVAL = args.v or args.vv

# print info
ArgPrinter.print_agent(str(args.agent))
ArgPrinter.print_device(str(device))
ArgPrinter.print_args(args)

# env changes

# state / action dims
n_actions = env.action_space.n

# build agent
model = ConvDQN(84, 84, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_function = nn.MSELoss()
replay_buffer = SimpleFrameBuffer(BUFFER_SIZE, device, Transition)
policy = EpsilonGreedyPolicy(model, device, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY)

agent = agent_type(model, replay_buffer, policy, optimizer, loss_function, gamma=GAMMA,
                   target_update_period=TARGET_UPDATE, device=device)

# train
print("Train...")
episode_scores = agent.train(env, N_EPISODES, MAX_STEPS, batch_size=BATCH_SIZE, warm_up_period=WARM_UP,
                             visualize=VIS_TRAIN)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(episode_scores, title=("Pong-v0 " + agent.name + " Training"))

# test
print("Evaluate...")
threshold = env.spec.reward_threshold
if threshold:
    print(f"Target Score: {threshold:.2f}")
test_scores = agent.play(env, 100, env.spec.max_episode_steps, visualize=VIS_EVAL)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(test_scores, title=("Pong-v0 " + agent.name + " Test"))
