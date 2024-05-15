import numpy as np


def plot_timings(timings, ax=None):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise KeyError("Matplotlib is needed for the plotting functionailites.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    summed = np.zeros_like(timings["total"])
    for key, val in timings.items():
        ax.plot(val, lw=1, marker="o", markevery=(1, 10), markersize=5, label=key)

        if key == "total":
            mean_total = np.mean(val)
            ax.set_ylim(0, mean_total)
        else:
            if None not in val:
                summed += np.array(val)

    # ax.plot(summed, lw=1, ls="--", color="grey", label="summed")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Runtime [sec]")
    ax.grid(axis="both", lw=0.2, ls="--", zorder=-10)
    fig.legend(loc="upper right")

    return fig, ax


def plot_metrics(metrics, log_every, ax=None):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        raise KeyError("Matplotlib is needed for the plotting functionailites.")

    fig, ax = plt.subplots(figsize=(6, 5))

    for key, val in metrics.items():
        ax.plot(
            np.arange(len(val)) * log_every,
            val,
            lw=2,
            marker="o",
            markevery=(-1, len(val)),
            markersize=7,
            label=key,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_yscale("log")
    ax.grid(axis="both", lw=0.2, ls="--", zorder=-10)
    fig.legend(loc="upper right")

    return fig, ax
