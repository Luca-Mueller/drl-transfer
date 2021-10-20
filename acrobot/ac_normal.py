import gym
from colorama import init, Fore, Style

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import DQN
from torch_agents.replay_buffer import Transition, SimpleReplayBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent, DQV2Agent
from torch_agents.plotting import plot_scores
from torch_agents.utils import AgentArgParser, ArgPrinter

# initialize color / gym / device
init()
env = gym.make('Acrobot-v1').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
arg_parser = AgentArgParser()
arg_parser.add_acrobot_args()
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
ArgPrinter.print_acrobot_args(args)

# env changes
env.LINK_LENGTH_1 = args.link_len1
env.LINK_LENGTH_2 = args.link_len2
env.LINK_MASS_1 = args.link_mass1
env.LINK_MASS_2 = args.link_mass2
env.LINK_COM_POS_1 = args.link_com1
env.LINK_COM_POS_2 = args.link_com2
env.LINK_MOI = args.link_moi

# state / action dims
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# build agent
model = DQN(n_observations, n_actions, fc1_nodes=64, fc2_nodes=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_function = nn.MSELoss()
replay_buffer = SimpleReplayBuffer(BUFFER_SIZE, Transition)
policy = EpsilonGreedyPolicy(model, device, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY)

agent = agent_type(model, replay_buffer, policy, optimizer, loss_function, gamma=GAMMA,
                   target_update_period=TARGET_UPDATE, device=device)

# train
print("Train...")
episode_scores = agent.train(env, N_EPISODES, MAX_STEPS, batch_size=BATCH_SIZE, warm_up_period=WARM_UP,
                             visualize=VIS_TRAIN)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(episode_scores, title=("Acrobot-v1 " + agent.name + " Training"))

# test
print("Evaluate...")
print(f"Target Score: {env.spec.reward_threshold:.2f}")
test_scores = agent.play(env, 100, env.spec.max_episode_steps, visualize=VIS_EVAL)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(test_scores, title=("Acrobot-v1 " + agent.name + " Test"))
