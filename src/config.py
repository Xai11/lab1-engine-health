import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Пути можно переопределить переменными окружения (удобно для тестов на других данных)
DATA_PATH = Path(os.environ.get("ENGINE_DATA", ROOT / "data" / "engine_data.csv"))
RESULTS_DIR = Path(os.environ.get("ENGINE_RESULTS", ROOT / "results"))
FIGURES_DIR = Path(os.environ.get("ENGINE_FIGURES", ROOT / "figures"))

TARGET = "Engine Condition"
SEED = 42                   # seed разбиения данных и базовый seed обучения
SPLIT = (0.70, 0.15, 0.15)  # train / val / test

# Два скрытых слоя (64 и 32): данных ~19.5k, признаков всего 6, поэтому большая сеть
# не нужна, а сужение к выходу - стандартная форма MLP.
HIDDEN = (64, 32)
# 50 эпох с запасом: val loss минимален около 10-й эпохи, дальше видно переобучение,
# и это специально оставлено на графиках для анализа.
EPOCHS = 50
BATCH_SIZE = 128  # компромисс: ~107 шагов на эпоху, градиент не слишком шумный
LR = 1e-3         # типичный шаг для Adam; при нём обучение стабильно без подбора
N_RUNS = 15       # по заданию
