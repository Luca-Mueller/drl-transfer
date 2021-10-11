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
env = gym.make('LunarLander-v2').unwrapped
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parse args
arg_parser = AgentArgParser()
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
plot_scores(episode_scores, title=("LunarLander-v2 " + agent.name + " Training"))

# Test
print("Evaluate...")
print(f"Target Score: {env.spec.reward_threshold:.2f}")
test_scores = agent.play(env, 100, env.spec.max_episode_steps, visualize=VIS_EVAL)

print(Fore.GREEN + "Done\n" + Style.RESET_ALL)
plot_scores(test_scores, title=("LunarLander-v2 " + agent.name + " Test"))
