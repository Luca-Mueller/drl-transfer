import gym
import pickle
from pathlib import Path
from colorama import init, Fore, Style

import torch
import torch.nn as nn
import torch.optim as optim

from torch_agents.models import ConvDQN
from torch_agents.replay_buffer import Transition, SimpleFrameBuffer
from torch_agents.policy import EpsilonGreedyPolicy
from torch_agents.observer import BufferObserver
from torch_agents.agent import DQNAgent, DDQNAgent, DQVAgent, DQV2Agent
from torch_agents.plotting import plot_scores
from torch_agents.utils import AgentArgParser, ArgPrinter, FireResetEnv, MaxAndSkipEnv, PongReduceActions, \
    EnduroReduceActions, BreakoutReduceActions, ClipRewardEnv, no_print

# parse args
arg_parser = AgentArgParser()
arg_parser.add_collect_args()
args = arg_parser.parse_args()

# game env
ENV_NAMES = {"Pong": "PongDeterministic-v4", "Enduro": "EnduroDeterministic-v0", "Breakout": "BreakoutDeterministic-v0"}
ENV = args.task_name
if ENV.lower() in map(lambda k: k.lower(), ENV_NAMES.keys()):
    ENV = ENV.capitalize()
else:
    ENV = "Pong"

# initialize color / gym / device
init()
env = gym.make(ENV_NAMES[ENV])

if ENV == "Pong":
    env = PongReduceActions(env)
if ENV == "Enduro":
    env = EnduroReduceActions(env)
if ENV == "Breakout":
    env = BreakoutReduceActions(env)
env = FireResetEnv(env)
env = MaxAndSkipEnv(env)
#env = ClipRewardEnv(env)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
SAVE_BUFFER = args.save_buffer
SAVE_MODEL = args.save_model
SAVE_AGENT = args.save_agent

# print info
ArgPrinter.print_agent(str(args.agent))
ArgPrinter.print_device(str(device))
ArgPrinter.print_args(args)

# state / action dims
n_actions = env.action_space.n

# build agent
model = ConvDQN(84, 84, n_actions).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=LR, eps=0.01)
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
plot_scores(episode_scores, title=(ENV + agent.name + " Training"))

# test
# TODO: test in non-deterministic env?
print("Evaluate...")
threshold = env.spec.reward_threshold
if threshold:
    print(f"Target Score: {threshold:.2f}")
agent.load_best_model()
test_scores = agent.play(env, 5, env.spec.max_episode_steps, visualize=VIS_EVAL)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(test_scores, title=(ENV + agent.name + " Test"))

# build transfer buffer
transfer_buffer = SimpleFrameBuffer(BUFFER_SIZE, Transition)
transfer_observer = BufferObserver(transfer_buffer)

print("Fill buffer...")
while len(transfer_buffer) < BUFFER_SIZE:
    with no_print():
        test_scores = agent.play(env, 1, env.spec.max_episode_steps, observer=transfer_observer, visualize=VIS_EVAL)
    print(f"\rBufferSize: {len(transfer_buffer):>6}/{BUFFER_SIZE:<6}", end="")
print(Fore.GREEN + "\nDone\n" + Style.RESET_ALL)

# save buffer
if SAVE_BUFFER:
    buffer_dir = Path("buffers") / ENV
    if not buffer_dir.is_dir():
        buffer_dir.mkdir()
    idx = len([c for c in buffer_dir.iterdir()])
    buffer_file = f"{ENV}_{agent.name}_buffer_{idx}.pickle"
    print("Save buffer as:\t'" + buffer_file + "'")

    with open(buffer_dir / buffer_file, "wb") as f:
        pickle.dump(transfer_buffer, f)

# save model
if SAVE_MODEL:
    model_dir = Path("models") / ENV
    if not model_dir.is_dir():
        model_dir.mkdir()
    idx = len([c for c in model_dir.iterdir()])
    model_file = f"{ENV}_{agent.name}_{idx}.pth"
    print("Save model as:\t'" + model_file + "'")

    torch.save(model.state_dict(), model_dir / model_file)

# save agent
if SAVE_AGENT:
    agent_dir = Path("agents") / ENV
    if not agent_dir.is_dir():
        agent_dir.mkdir()
    idx = len([c for c in agent_dir.iterdir()])
    agent_file = f"{ENV}_{agent.name}_{agent}_{idx}.pickle"
    print("Save agent as:\t'" + agent_file + "'")

    with open(agent_dir / agent_file, "wb") as f:
        pickle.dump(agent, f)