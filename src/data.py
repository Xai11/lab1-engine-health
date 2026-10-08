import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config


def load_splits(split_seed: int = config.SEED):
    """Стратифицированное разбиение 70/15/15; seed разбиения фиксирован.

    Разбиение не зависит от seed обучения: все запуски видят одни и те же
    train/val/test, поэтому разброс результатов вызван только сетью и порядком батчей.
    """
    df = pd.read_csv(config.DATA_PATH)
    X = df.drop(columns=[config.TARGET]).to_numpy(dtype=np.float32)
    y = df[config.TARGET].to_numpy(dtype=np.float32)

    _, val_f, test_f = config.SPLIT
    # Два шага: сначала отделяем train, затем остаток делим на val и test.
    # stratify сохраняет долю классов (63/37) во всех трёх частях
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=val_f + test_f, stratify=y, random_state=split_seed)
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=test_f / (val_f + test_f),
        stratify=y_rest, random_state=split_seed)

    # Стандартизация: признаки в разных масштабах (rpm ~ сотни, давление ~ единицы).
    # Статистики считаем только по train, чтобы не было утечки из val/test
    scaler = StandardScaler().fit(X_tr)
    t = lambda a: torch.from_numpy(scaler.transform(a).astype(np.float32))
    # Цель имеет форму (N, 1), как и выход модели, для BCEWithLogitsLoss
    u = lambda a: torch.from_numpy(a).unsqueeze(1)
    return (t(X_tr), u(y_tr)), (t(X_va), u(y_va)), (t(X_te), u(y_te))


def make_train_loader(X, y, seed: int) -> DataLoader:
    """Перемешивание батчей управляется отдельным генератором с фиксированным seed.

    Свой генератор делает порядок батчей независимым от остальных вызовов RNG.
    """
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=config.BATCH_SIZE,
                      shuffle=True, generator=g)
