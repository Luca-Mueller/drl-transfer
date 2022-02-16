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
                          ylim: Union[int, Tuple[int, int]] = None, xlim: Union[int, Tuple[int, int]] = None,
                          show: bool = True, save: bool = True):
    if not filename.endswith(".pickle"):
        filename += ".pickle"

    task_dir = Path(filename[:5])

    with open(hist_dir / task_dir / filename, "rb") as f:
        hist = pickle.load(f)

    if title is None:
        title = filename[:10].replace("_", " ").replace("vL", "mod")
        title = title.replace("cp ", "CartPole-").replace("ac ", "Acrobot-")
        title = title[:-4] + "|" + title[-5:]

    x = hist.pop("x")

    for key in hist:
        if key.startswith("q_"):
            continue

        mean = np.mean(hist[key], axis=0)
        sm_mean = savgol_filter(mean, 21, 3)
        std = savgol_filter(np.std(hist[key], axis=0), 21, 3)

        plt.plot(x, mean, lw=5.0, label=key.replace("_", " "))
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.fill_between(x, sm_mean - std, sm_mean + std, alpha=0.15)

    #plt.xlabel("Episodes", fontsize="large")
    #plt.ylabel("Scores", fontsize="large")
    plt.title(title, fontsize="x-large")
    #plt.legend(loc="lower right", fontsize="x-large")
    plt.grid(alpha=0.7)

    print("save " + filename)

    fig = plt.gcf()
    if save:
        plot_dir = Path("plots")
        fig.savefig(plot_dir / task_dir / (filename[:-12] + "_transfer"))

    if show:
        plt.show()

    return fig
