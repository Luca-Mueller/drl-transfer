import gym
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
from colorama import init, Fore, Back, Style

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import DQN
from torch_agents.replay_buffer import Transition, SimpleReplayBuffer,  FilledReplayBuffer, LimitedReplayBuffer, \
    LimitedFilledReplayBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent, DQV2Agent
from torch_agents.plotting import plot_transfer_history
from torch_agents.utils import no_print, AgentArgParser, ArgPrinter

# initialize color / gym / device
init()
env = gym.make('Acrobot-v1').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
arg_parser = AgentArgParser()
arg_parser.add_acrobot_args()
arg_parser.add_transfer_args()
args = arg_parser.parse_args()

# task
TASK_NAME = args.task_name
assert len(TASK_NAME) == 5, "task name should be exactly 5 letters (experiments.md.g. cp_v0)"

# needs transfer buffer, transfer model or both
assert args.buffer_name or args.model_name, "no buffer or model specified for transfer"

# agent type
assert args.agent in ["DQN", "DDQN", "DQV", "DQV2"], f"invalid agent type '{args.agent}'"

if args.agent == "DQN":
    agent_type = DQNAgent
    agent_color = Fore.LIGHTCYAN_EX
elif args.agent == "DDQN":
    agent_type = DDQNAgent
    agent_color = Fore.LIGHTMAGENTA_EX
elif args.agent == "DQV":
    agent_type = DQVAgent
    agent_color = Fore.LIGHTYELLOW_EX
elif args.agent == "DQV2":
    agent_type = DQV2Agent
    agent_color = Fore.LIGHTRED_EX

# device
if args.cpu:
    device = torch.device("cpu")

# buffer options
BUFFER_NAME = args.buffer_name
if BUFFER_NAME and not BUFFER_NAME.endswith(".pickle"):
    BUFFER_NAME += ".pickle"
BUFFER_DIR = Path(args.buffer_dir)

# load transfer buffer
if BUFFER_NAME:
    with open(BUFFER_DIR / BUFFER_NAME[:5] / BUFFER_NAME, "rb") as f:
        transfer_buffer = pickle.load(f)
        transfer_buffer.to(device)

# model options
MODEL_NAME = args.model_name
if MODEL_NAME and not MODEL_NAME.endswith(".pth"):
    MODEL_NAME += ".pth"
MODEL_DIR = Path(args.model_dir)

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
REPETITIONS = args.repetitions
TEST_EVERY = args.test_every

# visualization
VIS_TRAIN = args.vv
VIS_EVAL = args.v or args.vv

# print info
ArgPrinter.print_agent(str(args.agent))
ArgPrinter.print_device(str(device))
ArgPrinter.print_transfer_args(BUFFER_NAME, MODEL_NAME)
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

# scores
default_agent_hist, buffer_transfer_agent_hist, model_transfer_agent_hist, double_transfer_agent_hist = [], [], [], []


# perform one training cycle for one agent
def train_agent(buffer_transfer: bool = False, model_transfer: bool = False) -> np.array:
    global default_agent_hist, buffer_transfer_agent_hist, model_transfer_agent_hist, double_transfer_agent_hist

    model = DQN(n_observations, n_actions).to(device)
    if model_transfer:
        model.load_state_dict(torch.load(MODEL_DIR / MODEL_NAME[:5] / MODEL_NAME))
        model.reset_head()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.MSELoss()
    if buffer_transfer:
        if args.limited_buffer:
            replay_buffer = LimitedFilledReplayBuffer(BUFFER_SIZE, transfer_buffer, Transition, limit=1000)
        else:
            replay_buffer = FilledReplayBuffer(BUFFER_SIZE, transfer_buffer, Transition)
    else:
        if args.limited_buffer:
            replay_buffer = LimitedReplayBuffer(1000, Transition)
        else:
            replay_buffer = SimpleReplayBuffer(BUFFER_SIZE, Transition)
    policy = EpsilonGreedyPolicy(model, device, eps_decay=EPS_DECAY)

    agent = agent_type(model, replay_buffer, policy, optimizer, loss_function, gamma=GAMMA,
                       target_update_period=TARGET_UPDATE, device=device)

    with no_print():
        local_hist = []
        episode_scores = agent.play(env, 1, MAX_STEPS, visualize=VIS_EVAL)
        local_hist.append(episode_scores[0])
        for _ in range(N_EPISODES // TEST_EVERY):
            agent.train(env, TEST_EVERY, MAX_STEPS, batch_size=BATCH_SIZE, warm_up_period=WARM_UP, visualize=VIS_TRAIN)
            episode_scores = agent.play(env, 10, MAX_STEPS, visualize=VIS_EVAL)
            local_hist.append(np.mean(episode_scores))

    return np.array(local_hist)


# train loop
print(agent_color + Back.BLACK, end="")
for i in tqdm(range(REPETITIONS)):
    # default agent
    default_agent_hist.append(train_agent())
    # buffer transfer
    if BUFFER_NAME:
        buffer_transfer_agent_hist.append(train_agent(buffer_transfer=True))
    # model transfer
    if MODEL_NAME:
        model_transfer_agent_hist.append(train_agent(model_transfer=True))
    # double transfer
    if BUFFER_NAME and MODEL_NAME:
        double_transfer_agent_hist.append(train_agent(buffer_transfer=True, model_transfer=True))
print(Style.RESET_ALL, end="")

# save history
hist = {"default": np.array(default_agent_hist), "x": np.arange(len(default_agent_hist[0])) * TEST_EVERY}
if len(buffer_transfer_agent_hist):
    hist["buffer_transfer"] = np.array(buffer_transfer_agent_hist)
if len(model_transfer_agent_hist):
    hist["model_transfer"] = np.array(model_transfer_agent_hist)
if len(double_transfer_agent_hist):
    hist["double_transfer"] = np.array(double_transfer_agent_hist)

history_dir = Path("history") / TASK_NAME
history_file = args.task_name + "_" + args.agent + "_" + str(N_EPISODES) + "eps_" + ("ltd_" if args.limited_buffer else
                                                                                     "") + "hist.pickle"
with open(history_dir / history_file, "wb") as f:
    pickle.dump(hist, f)

# plot history
plot_transfer_history(history_file, ylim=(-600, 0))
