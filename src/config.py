# src/config.py
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

# Create directories if they don't exist
for p in (RAW_DIR, PROC_DIR, RESULTS_DIR, MODELS_DIR, FIGURES_DIR):
    p.mkdir(parents = True, exist_ok = True)

# Reproducibility
SEED = 42
N_JOBS = -1  # for scikit-learn (use -1 to use all CPUs)

# CV / evaluation
CV_FOLDS = 5
RANDOM_STATE = SEED