import json

import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn

from . import config
from .data import load_splits, make_train_loader
from .model import MLP, init_weights
from .seed import set_determinism


@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    """Loss и accuracy на всей выборке сразу (она мала, батчи не нужны)."""
    model.eval()
    logits = model(X)
    loss = loss_fn(logits, y).item()
    acc = ((logits > 0).float() == y).float().mean().item()
    return loss, acc, logits  # логит > 0 <=> вероятность класса 1 > 0.5


def run(seed: int, out_dir=None, split_seed: int = config.SEED) -> pd.DataFrame:
    # Порядок важен: сначала фиксируем RNG, затем строим данные, модель и оптимизатор
    set_determinism(seed)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_splits(split_seed)
    loader = make_train_loader(X_tr, y_tr, seed)

    model = MLP(X_tr.shape[1])
    init_weights(model, seed)
    loss_fn = nn.BCEWithLogitsLoss()  # бинарная классификация -> бинарная кросс-энтропия
    opt = torch.optim.Adam(model.parameters(), lr=config.LR)

    rows = []
    for epoch in range(1, config.EPOCHS + 1):
        # --- обучение: один проход по батчам ---
        model.train()
        total, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            # loss батча взвешиваем размером: последний батч может быть меньше
            total += loss.item() * len(xb)
            correct += ((logits.detach() > 0).float() == yb).sum().item()
            n += len(xb)
        # --- проверка на валидации после каждой эпохи (веса не меняются) ---
        val_loss, val_acc, _ = evaluate(model, X_va, y_va, loss_fn)
        rows.append(dict(epoch=epoch, train_loss=total / n, train_acc=correct / n,
                         val_loss=val_loss, val_acc=val_acc))
    hist = pd.DataFrame(rows)

    # Тест считаем один раз, после обучения; по нему модель не подбирается
    test_loss, test_acc, logits = evaluate(model, X_te, y_te, loss_fn)
    pred = (logits > 0).float().numpy().ravel()
    yt = y_te.numpy().ravel()
    metrics = dict(seed=seed, test_loss=test_loss, test_acc=test_acc,
                   test_f1=float(f1_score(yt, pred)),
                   test_auc=float(roc_auc_score(yt, logits.numpy().ravel())))
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        hist.to_csv(out_dir / "history.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        torch.save(model.state_dict(), out_dir / "model.pt")
    hist.attrs["metrics"] = metrics
    return hist


if __name__ == "__main__":
    h = run(config.SEED, config.RESULTS_DIR / "single")
    print(h.tail(3)); print(h.attrs["metrics"])
