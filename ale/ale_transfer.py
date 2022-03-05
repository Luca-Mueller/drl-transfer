import gym
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from colorama import init, Fore, Back, Style

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import ConvDQN
from torch_agents.replay_buffer import Transition, SimpleFrameBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent, DQV2Agent
from torch_agents.plotting import plot_transfer_history
from torch_agents.utils import AgentArgParser, ArgPrinter, FireResetEnv, MaxAndSkipEnv, ReduceActionsFLR, \
    ClipRewardEnv, no_print

# parse args
arg_parser = AgentArgParser()
arg_parser.add_transfer_args()
args = arg_parser.parse_args()

# game env
ENV_NAMES = {"Pong": "PongDeterministic-v4", "Enduro": "EnduroDeterministic-v0", "Breakout": "BreakoutDeterministic-v0"}
ENV = args.task_name
if ENV.lower() in map(lambda k: k.lower(), ENV_NAMES.keys()):
    ENV = ENV.capitalize()
else:
    print(f"environment '{ENV}' invalid, default to Pong")
    ENV = "Pong"

# initialize color / gym / device
init()
env = gym.make(ENV_NAMES[ENV])

env = ReduceActionsFLR(env)
env = FireResetEnv(env)
env = MaxAndSkipEnv(env)
#env = ClipRewardEnv(env)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# agent type
assert args.agent in ["DQN", "DDQN", "DQV", "DQV2"], f"invalid agent type '{args.agent}'"

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
print(f"Env:\t{ENV}")
ArgPrinter.print_agent(str(args.agent))
ArgPrinter.print_device(str(device))
ArgPrinter.print_transfer_args(BUFFER_NAME, MODEL_NAME)
ArgPrinter.print_args(args)

# state / action dims
n_actions = env.action_space.n

# scores
default_agent_hist, buffer_transfer_agent_hist, model_transfer_agent_hist, double_transfer_agent_hist = [], [], [], []


# perform one training cycle for one agent
def train_agent(buffer_transfer: bool = False, model_transfer: bool = False) -> np.array:
    global default_agent_hist, buffer_transfer_agent_hist, model_transfer_agent_hist, double_transfer_agent_hist

    # make space for new buffer
    torch.cuda.empty_cache()

    model = ConvDQN(84, 84, n_actions).to(device)
    if model_transfer:
        model.load_state_dict(torch.load(MODEL_DIR / MODEL_NAME.split('_')[0] / MODEL_NAME))
        model.reset_head()
    optimizer = optim.RMSprop(model.parameters(), lr=LR, eps=0.01)
    loss_function = nn.MSELoss()
    if buffer_transfer:
        # load transfer buffer
        with open(BUFFER_DIR / BUFFER_NAME.split('_')[0] / BUFFER_NAME, "rb") as f:
            transfer_buffer = pickle.load(f)
            transfer_buffer.to(device)
        replay_buffer = transfer_buffer
    else:
        replay_buffer = SimpleFrameBuffer(BUFFER_SIZE, device, Transition)
    policy = EpsilonGreedyPolicy(model, device, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY)

    agent = agent_type(model, replay_buffer, policy, optimizer, loss_function, gamma=GAMMA,
                       target_update_period=TARGET_UPDATE, device=device)

    with no_print():
        local_hist = []
        episode_scores = agent.play(env, 1, visualize=VIS_EVAL)
        local_hist.append(episode_scores[0])
        for _ in range(N_EPISODES // TEST_EVERY):
            agent.train(env, TEST_EVERY, MAX_STEPS, batch_size=BATCH_SIZE, warm_up_period=WARM_UP, visualize=VIS_TRAIN)
            episode_scores = agent.play(env, 3, MAX_STEPS, visualize=VIS_EVAL)
            local_hist.append(np.mean(episode_scores))

    # free GPU memory
    replay_buffer.to(torch.device("cpu"))
    model.to("cpu")
    del replay_buffer
    del model

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

history_dir = Path("history") / ENV
if not history_dir.is_dir():
    history_dir.mkdir()
idx = len([c for c in history_dir.iterdir()])
history_file = args.task_name + "_" + args.agent + "_" + str(idx) + "_hist.pickle"
with open(history_dir / history_file, "wb") as f:
    pickle.dump(hist, f)

# plot history
plot_transfer_history(history_file, task_dir=ENV)
