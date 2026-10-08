import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config


def curves(df, col, title, path, mean_std=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if mean_std:
        g = df.groupby("epoch")[col]
        m, s = g.mean(), g.std()
        ax.plot(m.index, m, color="C0", label="среднее")
        ax.fill_between(m.index, m - s, m + s, color="C0", alpha=0.25, label="±1 std")
    else:
        for r, d in df.groupby("run"):
            ax.plot(d.epoch, d[col], alpha=0.6, lw=1)
    ax.set_xlabel("Эпоха"); ax.set_ylabel("Ошибка (BCE)")
    ax.set_title(title); ax.grid(alpha=0.3)
    if mean_std: ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    config.FIGURES_DIR.mkdir(exist_ok=True)
    for mode, name in (("same", "один seed"), ("diff", "разные seed")):
        df = pd.read_csv(config.RESULTS_DIR / f"history_{mode}.csv")
        for col, kind in (("train_loss", "обучение"), ("val_loss", "валидация")):
            title = f"Ошибка на {kind}, 15 запусков ({name})"
            curves(df, col, title, config.FIGURES_DIR / f"{col}_{mode}.png")
            if mode == "diff":
                curves(df, col, title + ": среднее ± std",
                       config.FIGURES_DIR / f"{col}_{mode}_mean.png", mean_std=True)


if __name__ == "__main__":
    main()
