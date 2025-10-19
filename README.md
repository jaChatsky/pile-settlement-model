# pile-settlement-model
Modeling and predicting pile settlement using sparse regression, kernel methods and Bayesian uncertainty quantification.

pile-settlement-prediction/
│
├── README.md                ← project overview, instructions, and references
├── requirements.txt         ← list of Python dependencies
├── LICENSE                  ← (MIT License from GitHub)
│
├── data/                    ← datasets (raw and processed)
│   ├── raw/                 ← original data (Nejad & Jaksa dataset)
│   └── processed/           ← cleaned/normalized CSV files
│
├── notebooks/               ← Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_sparse_models.ipynb
│   ├── 03_kernel_models.ipynb
│   └── 04_bayesian_uncertainty.ipynb
│
├── src/                     ← all Python source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── sparse_regression.py
│   ├── kernel_methods.py
│   ├── bayesian_model.py
│   └── evaluation.py
│
├── results/                 ← store model outputs, figures, metrics
│   ├── figures/
│   └── metrics/
│
├── report/                  ← NeurIPS-style report
│   ├── main.tex
│   ├── references.bib
│   └── appendix.tex
│
└── scripts/                 ← automation scripts for experiments
    └── run_all_experiments.py
