import os
import random

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел и включает детерминированные алгоритмы."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)  # единый порядок суммирования на CPU
