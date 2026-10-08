import torch
from torch import nn

from . import config


class MLP(nn.Module):
    """Полносвязная сеть из Linear-слоёв с ReLU между ними (6 -> 64 -> 32 -> 1)."""

    def __init__(self, n_in: int, hidden=config.HIDDEN):
        super().__init__()
        dims = [n_in, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        # Последний слой без активации: возвращает логит. Сигмоида встроена в
        # BCEWithLogitsLoss, что численно устойчивее, чем sigmoid + BCELoss
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_weights(model: nn.Module, seed: int) -> None:
    """Детерминированная инициализация: Kaiming normal из отдельного генератора, bias = 0.

    Kaiming (He) подходит для ReLU: std = sqrt(2 / fan_in) сохраняет дисперсию
    сигнала при прохождении через слои. Веса берутся из нормального распределения
    через собственный torch.Generator, поэтому не зависят от глобального RNG.
    """
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * std)
                m.bias.zero_()
