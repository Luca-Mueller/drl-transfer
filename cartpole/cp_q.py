import gym
import numpy as np
from pathlib import Path
import pickle
from colorama import init, Fore
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import DQN
from torch_agents.replay_buffer import Transition, SimpleReplayBuffer, LimitedReplayBuffer, LimitedFilledReplayBuffer, \
    FilledReplayBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.observer import StateObserver
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent, DQV2Agent
from torch_agents.plotting import plot_transfer_history
from torch_agents.utils import no_print, AgentArgParser, ArgPrinter

# initialize color / gym / device
init()
env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
arg_parser = AgentArgParser()
arg_parser.add_cartpole_args()
arg_parser.add_transfer_args()
args = arg_parser.parse_args()

# task
TASK_NAME = args.task_name
assert len(TASK_NAME) == 5, "task name should be exactly 5 letters (e.g. cp_v0)"

# needs at least transfer model
assert args.model_name, "no model specified for transfer"

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
#REPETITIONS = args.repetitions
TEST_EVERY = args.test_every

# visualization
VIS_TRAIN = args.vv
VIS_EVAL = args.v or args.vv

# print info
ArgPrinter.print_agent(str(args.agent))
ArgPrinter.print_device(str(device))
ArgPrinter.print_transfer_args(BUFFER_NAME, MODEL_NAME)
ArgPrinter.print_args(args)
ArgPrinter.print_cp_args(args)

# env changes
env.gravity = args.gravity
env.masscart = args.mass_cart
env.masspole = args.mass_pole
env.length = args.pole_length

# state / action dims
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# load trained model to predict Q*
OPT_MODEL_NAME = TASK_NAME + MODEL_NAME[5:]
opt_model = DQN(n_observations, n_actions).to(device)
opt_model.load_state_dict(torch.load(MODEL_DIR / TASK_NAME / OPT_MODEL_NAME))


# perform one training cycle for one agent
def train_agent(buffer_transfer: bool = False, model_transfer: bool = False) -> Tuple[list]:
    q_pred, q_opt = [], []
    states_visited = []
    state_observer = StateObserver(states_visited)
    model = DQN(n_observations, n_actions).to(device)
    if model_transfer:
        model.load_state_dict(torch.load(MODEL_DIR / MODEL_NAME[:5] / MODEL_NAME))
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
    policy = EpsilonGreedyPolicy(model, device, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY)

    agent = agent_type(model, replay_buffer, policy, optimizer, loss_function, gamma=GAMMA,
                       target_update_period=TARGET_UPDATE, device=device)

    with no_print():
        local_hist = []
        episode_scores = agent.play(env, 1, observer=state_observer, visualize=VIS_EVAL)
        while states_visited:
            with torch.no_grad():
                s = states_visited.pop()
                q_pred.append(model(s).max(0)[0].squeeze())
                q_opt.append(opt_model(s).max(0)[0].squeeze())
        local_hist.extend(episode_scores)

        for _ in range(N_EPISODES // TEST_EVERY):
            agent.train(env, TEST_EVERY, MAX_STEPS, batch_size=BATCH_SIZE, warm_up_period=WARM_UP, visualize=VIS_TRAIN)
            episode_scores = agent.play(env, 1, MAX_STEPS, observer=state_observer, visualize=VIS_EVAL)
            while states_visited:
                with torch.no_grad():
                    s = states_visited.pop()
                    q_pred.append(model(s).max(0)[0].squeeze())
                    q_opt.append(opt_model(s).max(0)[0].squeeze())
            local_hist.extend(episode_scores)

    return local_hist, q_pred, q_opt


# train
default_agent_hist, default_agent_q_pred, default_agent_q_opt = train_agent()

if BUFFER_NAME:
    double_transfer_agent_hist, double_transfer_agent_q_pred, double_transfer_agent_q_opt = train_agent(True, True)
else:
    model_transfer_agent_hist, model_transfer_agent_q_pred, model_transfer_agent_q_opt = train_agent(model_transfer=True)

# plot delta Q
default_agent_delta_q = np.array(default_agent_q_pred) - np.array(default_agent_q_opt)
smoothed_default_agent_delta_q = savgol_filter(default_agent_delta_q, 101, 7)
plt.plot(smoothed_default_agent_delta_q, label="default", lw=2.5)

if BUFFER_NAME:
    double_transfer_delta_q = np.array(double_transfer_agent_q_pred) - np.array(double_transfer_agent_q_opt)
    smoothed_double_transfer_delta_q = savgol_filter(double_transfer_delta_q, 101, 7)
    plt.plot(smoothed_double_transfer_delta_q, label="double transfer", lw=2.5)
else:
    model_transfer_delta_q = np.array(model_transfer_agent_q_pred) - np.array(model_transfer_agent_q_opt)
    smoothed_model_transfer_delta_q = savgol_filter(model_transfer_delta_q, 101, 7)
    plt.plot(smoothed_model_transfer_delta_q, label="model transfer", lw=2.5)

plt.title("CartPole-mod: (Q - Q*)", fontsize="x-large")
plt.xlabel("Steps", fontsize="large")
plt.ylabel("ΔQ", fontsize="large")
plt.legend(loc="best")
plt.grid(alpha=0.7)
fig = plt.gcf()
plot_dir = Path("plots")
#fig.savefig(plot_dir / "delta_q" / (TASK_NAME + "_" + args.agent))
plt.show()

# save history
hist = {"default": np.array([default_agent_hist]), "x": np.arange(len(default_agent_hist)) * TEST_EVERY}
hist["q_default"] = default_agent_delta_q

if BUFFER_NAME:
    hist["double_transfer"] = np.array([double_transfer_agent_hist])
    hist["q_double_transfer"] = double_transfer_delta_q
else:
    hist["model_transfer"] = np.array([model_transfer_agent_hist])
    hist["q_model_transfer"] = model_transfer_delta_q

history_dir = Path("history") / TASK_NAME
history_file = args.task_name + "_" + args.agent + "_" + "q_" + ("ltd_" if args.limited_buffer else "") + "hist.pickle"
with open(history_dir / history_file, "wb") as f:
    pickle.dump(hist, f)

# plot history
plot_transfer_history(history_file, ylim=(0, 200), save=False)
