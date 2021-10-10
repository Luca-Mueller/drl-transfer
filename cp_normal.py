import gym
from colorama import init, Fore, Style

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import DQN
from torch_agents.replay_buffer import Transition, SimpleReplayBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent
from torch_agents.plotting import plot_scores
from torch_agents.utils import AgentArgParser

# initialize color / gym / device
init()
env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
arg_parser = AgentArgParser()
arg_parser.add_cartpole_args()
args = arg_parser.parse_args()

# agent type
if args.agent == "DQN":
    agent_type = DQNAgent
    agent_color = Fore.LIGHTCYAN_EX
elif args.agent == "DDQN":
    agent_type = DDQNAgent
    agent_color = Fore.LIGHTMAGENTA_EX
elif args.agent == "DQV":
    agent_type = DQVAgent
    agent_color = Fore.LIGHTYELLOW_EX
else:
    print("Error: invalid agent type")
    exit(-1)

print("Agent:\t" + agent_color + args.agent + Style.RESET_ALL)

# device
if args.cpu:
    device = torch.device("cpu")

device_color = Fore.LIGHTBLUE_EX if str(device) == "cpu" else Fore.LIGHTGREEN_EX
print("Device:\t" + device_color + str(device).upper() + Style.RESET_ALL + "\n")

# hyperparameters
LR = args.lr
BATCH_SIZE = args.batch_size
BUFFER_SIZE = args.buffer_size
GAMMA = args.gamma
EPS_START = args.epsilon_start
EPS_END = args.epsilon_end
EPS_DECAY = args.epsilon_decay
N_EPISODES = args.episodes
MAX_STEPS = args.max_steps
WARM_UP = args.warm_up
TARGET_UPDATE = args.target_update

param_color = Fore.YELLOW
print("Agent Hyperparameters:")
print(f"* Learning Rate:  {param_color}{LR}{Style.RESET_ALL}")
print(f"* Batch Size:     {param_color}{BATCH_SIZE}{Style.RESET_ALL}")
print(f"* Buffer Size:    {param_color}{BUFFER_SIZE}{Style.RESET_ALL}")
print(f"* Gamma:          {param_color}{GAMMA}{Style.RESET_ALL}")
print(f"* Eps Start:      {param_color}{EPS_START}{Style.RESET_ALL}")
print(f"* Eps End:        {param_color}{EPS_END}{Style.RESET_ALL}")
print(f"* Eps Decay:      {param_color}{EPS_DECAY}{Style.RESET_ALL}")
print(f"* Episodes:       {param_color}{N_EPISODES}{Style.RESET_ALL}")
print(f"* Max Steps:      {param_color}{MAX_STEPS}{Style.RESET_ALL}")
print(f"* Warm Up:        {param_color}{WARM_UP}{Style.RESET_ALL}")
print(f"* Target Update:  {param_color}{TARGET_UPDATE}{Style.RESET_ALL}\n")

# visualization
VIS_TRAIN = args.vv
VIS_EVAL = args.v or args.vv

# env changes
print("CartPole Parameters:")
print("* Gravity:\t  ", end="")
print((Fore.GREEN + str(env.gravity)) if env.gravity == args.gravity else (Fore.RED + str(args.gravity)))
print(Style.RESET_ALL, end="")
print("* Mass Cart:\t  ", end="")
print((Fore.GREEN + str(env.masscart)) if env.masscart == args.mass_cart else (Fore.RED + str(args.mass_cart)))
print(Style.RESET_ALL, end="")
print("* Mass Pole:\t  ", end="")
print((Fore.GREEN + str(env.masspole)) if env.masspole == args.mass_pole else (Fore.RED + str(args.mass_pole)))
print(Style.RESET_ALL, end="")
print("* Pole Length:\t  ", end="")
print((Fore.GREEN + str(env.length)) if env.length == args.pole_length else (Fore.RED + str(args.pole_length)))
print(Style.RESET_ALL)

env.gravity = args.gravity
env.masscart = args.mass_cart
env.masspole = args.mass_pole
env.length = args.pole_length

# state / action dims
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# build agent
model = DQN(n_observations, n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_function = nn.MSELoss()
replay_buffer = SimpleReplayBuffer(BUFFER_SIZE, Transition)
policy = EpsilonGreedyPolicy(model, device, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY)

agent = agent_type(model, replay_buffer, policy, optimizer, loss_function, gamma=GAMMA,
                   target_update_period=TARGET_UPDATE, device=device)

# Train
print("Train...")
episode_scores = agent.train(env, N_EPISODES, MAX_STEPS, batch_size=BATCH_SIZE, warm_up_period=WARM_UP,
                             visualize=VIS_TRAIN)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(episode_scores, title=("CartPole-v0 " + agent.name + " Training"))

# Test
print("Evaluate...")
print(f"Target Score: {env.spec.reward_threshold:.2f}")
test_scores = agent.play(env, 100, env.spec.max_episode_steps, visualize=VIS_EVAL)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(test_scores, title=("CartPole-v0 " + agent.name + " Test"))
