import pickle
import sys
import numpy as np
from pathlib import Path

HISTORY = Path("history")
KEYS = ["buffer_transfer", "model_transfer", "double_transfer"]


def calc_r(default_score, transfer_score):
    mean_default = np.mean(default_score, axis=0)
    mean_transfer = np.mean(transfer_score, axis=0)
    return (sum(mean_transfer) - sum(mean_default)) / sum(mean_transfer)

if __name__ == "__main__":
    eps = sys.argv[1]
    history = Path("history")

    print("Area Ratio Scores")

    with open("r_scores.txt", "w") as file:
        for task in ("cp_v0", "cp_vL"):
            for agent in ("DQN", "DDQN", "DQV"):
                try:
                    filename = f"{task}_{agent}_{eps}eps_hist.pickle"
                    path = history / task / filename

                    with open(path, "rb") as f:
                        data = pickle.load(f)

                    print(f"\n{filename[:-12]}")
                    file.write(filename[:-12] + '\n')

                    for key in KEYS:
                        line = f"{key:<20}:{round(calc_r(data['default'], data[key]), 2)}"
                        print(line)
                        file.write(line + '\n')

                    file.write('\n')
                except FileNotFoundError:
                    print(f"\nNo entry found for {task}_{agent}_{eps}eps")
