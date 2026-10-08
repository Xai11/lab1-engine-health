"""Собирает report/Report.docx из results/ и figures/ (запускать после run_all, plots, draw_network)."""
import json

import pandas as pd
from docx import Document
from docx.shared import Cm

from src import config

R, F = config.RESULTS_DIR, config.FIGURES_DIR


def main():
    same = pd.read_csv(R / "summary_same.csv")
    diff = pd.read_csv(R / "summary_diff.csv")
    delta = (R / "repro_check.txt").read_text().strip()
    repro = delta.endswith("=0.0")

    d = Document()
    d.add_heading("Лабораторная работа №1. Нейросеть из Linear-слоёв", 0)
    d.add_heading("1. Цель и данные", 1)
    d.add_paragraph(
        "Бинарная классификация состояния двигателя (Engine Condition) по 6 признакам: "
        "Engine rpm, Lub oil pressure, Fuel pressure, Coolant pressure, lub oil temp, Coolant temp. "
        "Набор данных: Kaggle, Engine Health Prediction.")
    d.add_paragraph(
        f"Разбиение стратифицированное {config.SPLIT} (train/val/test), random_state={config.SEED}. "
        "Признаки стандартизованы по статистикам обучающей выборки.")

    d.add_heading("2. Модель и обучение", 1)
    d.add_paragraph(
        f"Архитектура: 6 → {config.HIDDEN[0]} → {config.HIDDEN[1]} → 1, Linear + ReLU. "
        "Функция потерь: BCEWithLogitsLoss (бинарная классификация). "
        f"Оптимизатор: Adam, lr={config.LR}. Эпох: {config.EPOCHS}, размер пакета: {config.BATCH_SIZE}.")
    d.add_picture(str(F / "network.png"), width=Cm(15))

    d.add_heading("3. Детерминированность", 1)
    for t in ("Фиксация random, numpy, torch и PYTHONHASHSEED (src/seed.py).",
              "torch.use_deterministic_algorithms(True), один поток CPU.",
              "Инициализация весов: Kaiming normal из отдельного torch.Generator с seed, bias = 0.",
              "Разбиение данных фиксировано; порядок батчей задаётся отдельным torch.Generator."):
        d.add_paragraph(t, style="List Bullet")

    d.add_heading("4. Результаты 15 запусков", 1)
    for mode, title in (("same", "Один и тот же seed"), ("diff", "Разные seed")):
        d.add_heading(title, 2)
        for col in ("train_loss", "val_loss"):
            d.add_picture(str(F / f"{col}_{mode}.png"), width=Cm(13))
    d.add_picture(str(F / "val_loss_diff_mean.png"), width=Cm(13))

    d.add_heading("5. Анализ повторяемости", 1)
    def stat(df, c): return f"{df[c].mean():.4f} ± {df[c].std():.4f}"
    d.add_paragraph(f"Один seed: test acc {stat(same, 'test_acc')}, test AUC {stat(same, 'test_auc')}. "
                    f"Максимальное отклонение кривых между запусками: {delta.split('=')[1]}.")
    d.add_paragraph(f"Разные seed: test acc {stat(diff, 'test_acc')}, test AUC {stat(diff, 'test_auc')}.")
    d.add_paragraph(
        "Вывод: при фиксированном seed обучение побитово повторяется на CPU, все 15 запусков идентичны. "
        if repro else
        "Вывод: полной побитовой повторяемости добиться не удалось, см. отклонение выше. ")
    d.add_paragraph("При смене seed (инициализация и порядок батчей) результаты немного различаются, "
                    "разброс указан выше.")
    h = pd.read_csv(R / "history_diff.csv")
    m = h.groupby("epoch")[["train_loss", "val_loss"]].mean()
    best = int(m.val_loss.idxmin())
    d.add_heading("6. Качество и переобучение", 1)
    d.add_paragraph(
        f"Средняя ошибка на валидации минимальна на эпохе {best} ({m.val_loss.min():.4f}), "
        f"к эпохе {config.EPOCHS} она растёт до {m.val_loss.iloc[-1]:.4f}, а ошибка на обучении "
        f"продолжает падать ({m.train_loss.iloc[0]:.4f} → {m.train_loss.iloc[-1]:.4f}). "
        "Это признак начавшегося переобучения; для практического применения следует "
        "ограничить число эпох или использовать early stopping.")
    d.add_paragraph(
        "Точность на тесте около 66% при доле класса 1 около 63%: модель лишь немного лучше "
        "тривиального предсказания, признаки датасета слабо разделяют классы (AUC около 0.69).")
    d.save(config.ROOT / "report" / "Report.docx")


if __name__ == "__main__":
    main()
