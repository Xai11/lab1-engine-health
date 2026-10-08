import os
import random

import numpy as np
import torch


def set_determinism(seed: int) -> None:
    """Фиксирует все генераторы случайных чисел и включает детерминированные алгоритмы."""
    # Влияет на порядок хеширования в дочерних процессах и set/dict-зависимый код
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Нужна для детерминизма cuBLAS на GPU; на CPU не мешает
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Бросает ошибку, если операция не имеет детерминированной реализации,
    # так недетерминизм не пройдёт незамеченным
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # benchmark выбирает алгоритм по времени
    # Сложение чисел с плавающей точкой неассоциативно: при многопоточности порядок
    # суммирования может меняться, поэтому берём один поток
    torch.set_num_threads(1)
