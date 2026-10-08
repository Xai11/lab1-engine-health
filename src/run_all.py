"""15 запусков: (same) один seed -> проверка повторяемости; (diff) разные seed -> разброс."""
import numpy as np
import pandas as pd

from . import config
from .train import run


def main():
    for mode, seeds in (("same", [config.SEED] * config.N_RUNS),
                        ("diff", [config.SEED + i for i in range(config.N_RUNS)])):
        hists, mets = [], []
        for i, s in enumerate(seeds, 1):
            h = run(s, config.RESULTS_DIR / mode / f"run_{i:02d}")
            h.insert(0, "run", i)
            hists.append(h); mets.append(h.attrs["metrics"] | {"run": i})
            print(mode, i, s, round(h.attrs["metrics"]["test_acc"], 4))
        pd.concat(hists).to_csv(config.RESULTS_DIR / f"history_{mode}.csv", index=False)
        pd.DataFrame(mets).to_csv(config.RESULTS_DIR / f"summary_{mode}.csv", index=False)

    # Анализ повторяемости: максимальное отклонение между запусками с одним seed
    h = pd.read_csv(config.RESULTS_DIR / "history_same.csv")
    cols = ["train_loss", "val_loss", "train_acc", "val_acc"]
    ref = h[h.run == 1][cols].to_numpy()
    dev = max(np.abs(h[h.run == r][cols].to_numpy() - ref).max()
              for r in range(2, config.N_RUNS + 1))
    print("max |delta| между запусками с одним seed:", dev)
    (config.RESULTS_DIR / "repro_check.txt").write_text(f"max_abs_delta={dev}\n")


if __name__ == "__main__":
    main()
