# ЛР №1: нейросеть из Linear-слоёв (Engine Health)

Бинарная классификация состояния двигателя, PyTorch (CPU).

1. Положить `engine_data.csv` (Kaggle, Engine Health) в `data/`.
2. `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`
3. Запуск из корня проекта:
   ```
   .venv/bin/python -m src.run_all        # 15 запусков (один seed) + 15 (разные seed)
   .venv/bin/python -m src.plots          # графики в figures/
   .venv/bin/python -m src.draw_network   # схема сети
   .venv/bin/python -m report.build_report  # report/Report.docx
   ```
Истории обучения: `results/*/run_XX/history.csv`, сводки: `results/summary_*.csv`.
