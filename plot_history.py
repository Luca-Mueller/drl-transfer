import sys
from pathlib import Path
from torch_agents.plotting import plot_transfer_history

if __name__ == "__main__":
    filename = sys.argv[1]
    TASK = filename[:5]

    assert TASK in {"cp_v0", "cp_vL", "ac_v1", "ac_vL"}, "Task not recognized"

    if TASK.startswith("cp"):
        PATH = Path("cartpole")
        Y_LIM = (0, 200)
        X_LIM = (0, 200)
    if TASK.startswith("ac"):
        PATH = Path("acrobot")
        Y_LIM = (-500, 0)
        X_LIM = (0, 1000)

    HIST = PATH / "history"

    plot_transfer_history(filename, HIST, save=False, ylim=Y_LIM, xlim=X_LIM)
