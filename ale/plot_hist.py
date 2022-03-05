from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    filename = sys.argv[1]
    task_dir = Path(filename.split('_')[0].capitalize())

    if not filename.endswith(".pickle"):
        filename += ".pickle"

    with open(Path("history") / task_dir / filename, "rb") as f:
        hist = pickle.load(f)

    title = f"{task_dir} transfer"
    x = hist.pop("x")

    for key in hist:
        plt.plot(x, hist[key][0], lw=3.0, label=key.replace("_", " "))

    if task_dir == "Pong":
        plt.xlim(-10, 510)
        plt.ylim(-22, 22)

    if task_dir == "Enduro":
        plt.xlim(-10, 510)
        plt.ylim(-10, 510)

    plt.xlabel("Episodes", fontsize="x-large")
    plt.ylabel("Scores", fontsize="x-large")
    plt.title(title, fontsize="x-large")
    plt.legend(loc="lower right", fontsize="large")
    plt.grid(alpha=0.7)

    fig = plt.gcf()
    fig.savefig(Path("plots") / task_dir / f"{task_dir}_transfer")
    plt.show()
