import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.environ.get("ENGINE_DATA", ROOT / "data" / "engine_data.csv"))
RESULTS_DIR = Path(os.environ.get("ENGINE_RESULTS", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("ENGINE_FIGURES", ROOT / "figures"))

TARGET = "Engine Condition"
SEED = 42            # seed разбиения данных и базовый seed обучения
SPLIT = (0.70, 0.15, 0.15)  # train / val / test

HIDDEN = (64, 32)
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3
N_RUNS = 15
