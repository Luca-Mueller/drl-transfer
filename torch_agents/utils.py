from contextlib import contextmanager
from colorama import init, Fore, Style
import sys
import os
import argparse


@contextmanager
def no_print():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class AgentArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(AgentArgParser, self).__init__()
        init()  # colorama
        self.add_argument("-a", "--agent", type=str, default="DQN", help="the agent type (DQN, DDQN, DQV)")
        self.add_argument("-l", "--lr", type=float, default=0.001, help="learning rate (0.001)")
        self.add_argument("-b", "--batch-size", type=int, default=32, help="training batch size (32)")
        self.add_argument("-B", "--buffer-size", type=int, default=10_000, help="size of replay buffer (10,000)")
        self.add_argument("-g", "--gamma", type=float, default=0.95, help="discount factor (0.95)")
        self.add_argument("-S", "--epsilon-start", type=float, default=0.9, help="initial epsilon for EG policy (0.9)")
        self.add_argument("-E", "--epsilon-end", type=float, default=0.01, help="final epsilon for EG policy (0.01)")
        self.add_argument("-d", "--epsilon-decay", type=float, default=0.995, help="epsilon decay for EG policy (0.995)")
        self.add_argument("-e", "--episodes", type=int, default=200, help="N training episodes (200)")
        self.add_argument("-s", "--max-steps", type=int, default=None, help="max training steps per episode (inf)")
        self.add_argument("-w", "--warm-up", type=int, default=0, help="N training steps collected before training "
                                                                       "starts (0)")
        self.add_argument("-t", "--target-update", type=int, default=10, help="N episodes after which the target "
                                                                              "model is updated (10)")
        self.add_argument("--cpu", action="store_true", help="force CPU use")
        self.add_argument("-v", action="store_true", help="render evaluation")
        self.add_argument("-vv", action="store_true", help="render training and evaluation")

    def add_collect_args(self):
        self.add_argument("--task-name", type=str, default="cp_v0", help="5 letter task name (cp_v0)")
        self.add_argument("--save-model", action="store_true", help="store Q policy model")
        self.add_argument("--save-agent", action="store_true", help="store agent")

    def add_transfer_args(self):
        self.add_argument("--task-name", type=str, default="cp_v0", help="5 letter task name (cp_v0)")
        self.add_argument("--buffer-name", type=str, default=None, help="name of buffer file")
        self.add_argument("--buffer-dir", type=str, default="buffers", help="name of buffer directory ('buffers')")
        self.add_argument("--model-name", type=str, default=None, help="name of model parameter file")
        self.add_argument("--model-dir", type=str, default="models", help="name of model directory ('models')")
        self.add_argument("--limited-buffer", action="store_true", help="stop collecting after buffer is full")
        self.add_argument("-r", "--repetitions", type=int, default=5, help="number of training cycles per agent (5)")
        self.add_argument("-T", "--test-every", type=int, default=10, help="test interval (10)")
        self.add_argument("--max-eval", type=int, default=None, help="max number of steps during evaluation (None)")

    def add_cartpole_args(self):
        self.add_argument("-G", "--gravity", type=float, default=9.8, help="env gravity (9.8)")
        self.add_argument("-m", "--mass-cart", type=float, default=1.0, help="mass of the cart (1.0)")
        self.add_argument("-M", "--mass-pole", type=float, default=0.1, help="mass of the pole (0.1)")
        self.add_argument("-p", "--pole-length", type=float, default=0.5, help="length of the pole (default 0.5)")


class ArgPrinter:
    @staticmethod
    def print_agent(agent: str):
        if agent == "DQN":
            agent_color = Fore.LIGHTCYAN_EX
        elif agent == "DDQN":
            agent_color = Fore.LIGHTMAGENTA_EX
        elif agent == "DQV":
            agent_color = Fore.LIGHTYELLOW_EX
        else:
            return

        print("Agent:\t" + agent_color + agent + Style.RESET_ALL)

    @staticmethod
    def print_device(device: str):
        device_color = Fore.LIGHTBLUE_EX if device == "cpu" else Fore.LIGHTGREEN_EX
        print("Device:\t" + device_color + str(device).upper() + Style.RESET_ALL + "\n")

    @staticmethod
    def print_args(args):
        param_color = Fore.YELLOW
        print("Agent Hyperparameters:")
        print(f"* Learning Rate:  {param_color}{args.lr}{Style.RESET_ALL}")
        print(f"* Batch Size:     {param_color}{args.batch_size}{Style.RESET_ALL}")
        print(f"* Buffer Size:    {param_color}{args.buffer_size}{Style.RESET_ALL}")
        print(f"* Gamma:          {param_color}{args.gamma}{Style.RESET_ALL}")
        print(f"* Eps Start:      {param_color}{args.epsilon_start}{Style.RESET_ALL}")
        print(f"* Eps End:        {param_color}{args.epsilon_end}{Style.RESET_ALL}")
        print(f"* Eps Decay:      {param_color}{args.epsilon_decay}{Style.RESET_ALL}")
        print(f"* Episodes:       {param_color}{args.episodes}{Style.RESET_ALL}")
        print(f"* Max Steps:      {param_color}{args.max_steps}{Style.RESET_ALL}")
        print(f"* Warm Up:        {param_color}{args.warm_up}{Style.RESET_ALL}")
        print(f"* Target Update:  {param_color}{args.target_update}{Style.RESET_ALL}\n")

    @staticmethod
    def print_cp_args(args, env):
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

    @staticmethod
    def print_transfer_args(buffer_name: bool, model_name: bool):
        print("Buffer Transfer:  " + ((Fore.GREEN + "YES") if buffer_name else (Fore.RED + "NO")) + Style.RESET_ALL)
        print("Model  Transfer:  " + ((Fore.GREEN + "YES") if model_name else (Fore.RED + "NO")) + Style.RESET_ALL)
        print("Double Transfer:  " + ((Fore.GREEN + "YES") if buffer_name and model_name else (Fore.RED + "NO")))
        print(Style.RESET_ALL)
