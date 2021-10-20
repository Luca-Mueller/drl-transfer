import gym
from pathlib import Path
import pickle
from colorama import init, Fore, Style

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import DQN
from torch_agents.replay_buffer import Transition, SimpleReplayBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.observer import BufferObserver
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent, DQV2Agent
from torch_agents.plotting import plot_scores
from torch_agents.utils import no_print, AgentArgParser, ArgPrinter

# initialize color / gym / device
init()
env = gym.make('CartPole-v0').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
arg_parser = AgentArgParser()
arg_parser.add_cartpole_args()
arg_parser.add_collect_args()
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

# save options
SAVE_MODEL = args.save_model
SAVE_AGENT = args.save_agent
TASK_NAME = args.task_name

# print info
ArgPrinter.print_agent(str(args.agent))
ArgPrinter.print_device(str(device))
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

# build agent
model = DQN(n_observations, n_actions).to(device)
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

plot_scores(episode_scores, title=(TASK_NAME + " " + agent.name + " Training"))

# test
print("Evaluate...")
print(f"Target Score: {env.spec.reward_threshold:.2f}")
test_scores = agent.play(env, 100, env.spec.max_episode_steps, visualize=VIS_EVAL)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(test_scores, title=(TASK_NAME + " " + agent.name + " Test"))

# build transfer buffer
transfer_buffer = SimpleReplayBuffer(BUFFER_SIZE, Transition)
transfer_observer = BufferObserver(transfer_buffer)

print("Fill buffer...")
while len(transfer_buffer) < BUFFER_SIZE:
    with no_print():
        test_scores = agent.play(env, 1, env.spec.max_episode_steps, observer=transfer_observer, visualize=VIS_EVAL)
    print(f"\rBufferSize: {len(transfer_buffer):>6}/{BUFFER_SIZE:<6}", end="")
print(Fore.GREEN + "\nDone\n" + Style.RESET_ALL)

# save buffer
buffer_dir = Path("buffers") / TASK_NAME
buffer_file = TASK_NAME + "_" + agent.name + "_buffer.pickle"
print("Save buffer as:\t'" + buffer_file + "'")

with open(buffer_dir / buffer_file, "wb") as f:
    pickle.dump(transfer_buffer, f)

# save model
if SAVE_MODEL:
    model_dir = Path("models") / TASK_NAME
    model_file = f"{TASK_NAME}_{agent.name}_{model.shape[0]}x{model.fc1_nodes}x{model.fc2_nodes}x{model.shape[1]}.pth"
    print("Save model as:\t'" + model_file + "'")

    torch.save(model.state_dict(), model_dir / model_file)

# save agent
if SAVE_AGENT:
    agent_dir = Path("agents") / TASK_NAME
    agent_file = TASK_NAME + "_" + agent.name + "_agent.pickle"
    print("Save agent as:\t'" + agent_file + "'")

    with open(agent_dir / agent_file, "wb") as f:
        pickle.dump(agent, f)
