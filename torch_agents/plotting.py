import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from pathlib import Path
from typing import Tuple, Union


def plot_scores(scores, title: str = None, show: bool = True):
    scores = np.array(scores, dtype=np.float32)
    if title is not None:
        plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores)

    if show:
        plt.show()

    return plt.gcf()


def plot_transfer_history(filename: str, hist_dir: Path = Path("history"), title: str = None,
                          ylim: Union[int, Tuple[int, int]] = None, show: bool = True, save: bool = True):
    if not filename.endswith(".pickle"):
        filename += ".pickle"

    task_dir = Path(filename[:5])

    with open(hist_dir / task_dir / filename, "rb") as f:
        hist = pickle.load(f)

    if title is None:
        title = filename[:-12].replace("_", " ")

    x = hist.pop("x")

    for key in hist:
        if key.startswith("q"):
            continue

        mean = np.mean(hist[key], axis=0)
        sm_mean = savgol_filter(mean, 27, 3)
        std = savgol_filter(np.std(hist[key], axis=0), 27, 3)

        plt.plot(x, sm_mean, label=key.replace("_", " "))
        if ylim:
            plt.ylim(ylim)
        plt.fill_between(x, sm_mean - std, sm_mean + std, alpha=0.2)

    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.title(title)
    plt.legend(loc="best")

    fig = plt.gcf()
    if save:
        plot_dir = Path("plots")
        fig.savefig(plot_dir / task_dir / (filename[:-12] + "_transfer"))

    if show:
        plt.show()

    return fig
