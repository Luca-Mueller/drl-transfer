import pickle
import sys
import numpy as np
from pathlib import Path

HISTORY = Path("history")
KEYS = ["buffer_transfer", "model_transfer", "double_transfer"]


def calc_r(default_score, transfer_score):
    # add 500 since rewards are in range [-500, 0]
    mean_default = np.mean(default_score, axis=0) + 500
    mean_transfer = np.mean(transfer_score, axis=0) + 500
    return (sum(mean_transfer) - sum(mean_default)) / sum(mean_transfer)

if __name__ == "__main__":
    filename = sys.argv[1]
    TASK = filename[:5]
    PATH = HISTORY / TASK / filename

    with open(PATH, "rb") as f:
        data = pickle.load(f)

    print(f"r Scores {filename}")

    for key in KEYS:
        print(f"{key:<20}:{calc_r(data['default'], data[key])}")