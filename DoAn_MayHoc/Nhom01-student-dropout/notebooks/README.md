# Notebook layout

- `02_EDA/`: data exploration; outputs are in `02_EDA/Experiments/Logs/Experiment_01_EDA/`.
- `03_Baseline/`: preprocessing and baseline; outputs are in `03_Baseline/Experiments/Logs/Experiment_02_Baseline/`.
- `Experiments/`: imbalance, LR/DT tuning, A/B/C and error analysis, with one `Logs/Experiment_*` folder per run.
- `models/`: a train/CV notebook for each of the five models.

Open notebooks with the project `.venv` kernel. Repository lookup works from the project root or a nested notebook folder. Data splitting stays centralized in `src/data.py`; final hold-out evidence stays in `reports/evaluation/holdout/`.
