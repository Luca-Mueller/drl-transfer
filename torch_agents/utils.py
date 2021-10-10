from contextlib import contextmanager
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
        self.add_argument("-a", "--agent", type=str, default="DQN", help="the agent type (DQN, DDQN, DQV)")
        self.add_argument("-l", "--lr", type=float, default=0.001, help="learning rate (0.001)")
        self.add_argument("-b", "--batch-size", type=int, default=32, help="training batch size (32)")
        self.add_argument("-B", "--buffer-size", type=int, default=10_000, help="size of replay buffer (10,000)")
        self.add_argument("-g", "--gamma", type=float, default=0.95, help="discount factor (0.95)")
        self.add_argument("-S", "--epsilon-start", type=float, default=0.9, help="initial epsilon for EG policy (0.9)")
        self.add_argument("-E", "--epsilon-end", type=float, default=0.01, help="final epsilon for EG policy (0.01)")
        self.add_argument("-d", "--epsilon-decay", type=float, default=0.995, help="epsilon decay for EG policy (0.995)")
        self.add_argument("-e", "--episodes", type=int, default=200, help="N training episodes (200)")
        self.add_argument("-s", "--max_steps", type=int, default=None, help="max training steps per episode (inf)")
        self.add_argument("-w", "--warm-up", type=int, default=0, help="N training steps collected before training "
                                                                       "starts (0)")
        self.add_argument("-t", "--target-update", type=int, default=10, help="N episodes after which the target "
                                                                              "model is updated (10)")
        self.add_argument("--cpu", action="store_true", help="force CPU use")
        self.add_argument("-v", action="store_true", help="render evaluation")
        self.add_argument("-vv", action="store_true", help="render training and evaluation")

    def add_collect_args(self):
        self.add_argument("--save-model", action="store_true", help="store Q policy model")
        self.add_argument("--save-agent", action="store_true", help="store agent")
        self.add_argument("--task-name", type=str, default="cp_v0", help="name of buffer file")

    def add_transfer_args(self):
        self.add_argument("--task-name", type=str, default="cp_v0", help="name of buffer file")
        self.add_argument("--buffer-name", type=str, default=None, help="name of buffer file")
        self.add_argument("--buffer-dir", type=str, default="buffers", help="name of buffer directory ('buffers')")
        self.add_argument("--model-name", type=str, default=None, help="name of model parameter file")
        self.add_argument("--model-dir", type=str, default="models", help="name of model directory ('models')")
        self.add_argument("--limited-buffer", action="store_true", help="stop collecting after buffer is full")
        self.add_argument("-r", "--repetitions", type=int, default=5, help="number of training cycles per agent (5)")
        self.add_argument("-T", "--test-every", type=int, default=10, help="test interval (10)")

    def add_cartpole_args(self):
        self.add_argument("-G", "--gravity", type=float, default=9.8, help="env gravity (9.8)")
        self.add_argument("-m", "--mass-cart", type=float, default=1.0, help="mass of the cart (1.0)")
        self.add_argument("-M", "--mass-pole", type=float, default=0.1, help="mass of the pole (0.1)")
        self.add_argument("-p", "--pole-length", type=float, default=0.5, help="length of the pole (default 0.5)")
