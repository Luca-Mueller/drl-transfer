import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from pathlib import Path


def plot_transfer_history(filename: str, hist_dir: Path = Path("history"), title: str = None, show: bool = True,
                          save: bool = True):
    if not filename.endswith(".pickle"):
        filename += ".pickle"

    with open(hist_dir / filename, "rb") as f:
        hist = pickle.load(f)

    if title is None:
        title = filename[:-12].replace("_", " ")

    x = hist.pop("x")
    x += 10
    print(x)
    print(hist["default"])

    for key in hist:
        mean = np.mean(hist[key], axis=0)
        sm_mean = savgol_filter(mean, 9, 5)
        std = savgol_filter(np.std(hist[key], axis=0), 9, 5)

        plt.plot(x, sm_mean, label=key.replace("_", " "))
        plt.fill_between(x, sm_mean - std, sm_mean + std, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.title(title)
    plt.legend(loc="lower right")

    fig = plt.gcf()
    if save:
        plot_dir = Path("plots")
        fig.savefig(plot_dir / (filename[:-12] + "_transfer"))

    if show:
        plt.show()

    return fig


plot_transfer_history("cp_v0_DQN_200x10_hist.pickle")
