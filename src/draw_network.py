import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

FEATURES = ["Engine rpm", "Lub oil pressure", "Fuel pressure",
            "Coolant pressure", "lub oil temp", "Coolant temp"]


def main():
    sizes = [len(FEATURES), *config.HIDDEN, 1]
    labels = ["Вход", "Linear + ReLU", "Linear + ReLU", "Linear → sigmoid"]
    shown = [min(n, 8) for n in sizes]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pos = []
    for i, k in enumerate(shown):
        ys = [(j - (k - 1) / 2) * 0.8 for j in range(k)]
        pos.append([(i * 3, y) for y in ys])
        ax.scatter([i * 3] * k, ys, s=300, zorder=3, color=f"C{i}", edgecolor="k")
        extra = f" ({sizes[i]})" if sizes[i] > k else f" ({sizes[i]})"
        ax.text(i * 3, 3.7, labels[i] + extra, ha="center", fontsize=10)
        if sizes[i] > k:
            ax.text(i * 3, -(k / 2) * 0.8 - 0.3, "⋮", ha="center", fontsize=16)
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color="gray", lw=0.4, zorder=1)
    for (x, y), f in zip(pos[0], FEATURES[::-1]):
        ax.text(x - 0.3, y, f, ha="right", va="center", fontsize=9)
    ax.text(pos[-1][0][0] + 0.3, 0, "P(Engine Condition = 1)", va="center", fontsize=10)
    ax.set_xlim(-3.5, 10.5); ax.set_ylim(-3.6, 4.2); ax.axis("off")
    ax.set_title("Архитектура: 6 → 64 → 32 → 1")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "network.png", dpi=150)


if __name__ == "__main__":
    main()
