# Pile Settlement Prediction

This project models and predicts **pile settlement** using sparse regression and kernel-based machine learning methods.
Bayesian techniques are incorporated to quantify uncertainty in predictions, improving engineering decision-making under variable soil and loading conditions.                                                                             |

---

## Repository Structure

```
pile-settlement-model/
│
├── README.md                ← project overview, instructions, and references
├── requirements.txt         ← list of Python dependencies
├── LICENSE                  ← license information
│
├── data/
│   ├── raw/                 ← original datasets (Nejad & Jaksa)
│   └── processed/           ← cleaned and scaled data
│
├── notebooks/               ← Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_sparse_models.ipynb
│   ├── 03_kernel_models.ipynb
│   └── 04_bayesian_uncertainty.ipynb
│
├── src/                     ← Python source code
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── sparse_regression.py
│   ├── kernel_methods.py
│   ├── bayesian_model.py
│   └── evaluation.py
│
├── results/                 ← model outputs, figures, and metrics
│   ├── figures/
│   └── metrics/
│
├── report/                  ← academic report (LaTeX)
│   ├── main.tex
│   ├── references.bib
│   └── appendix.tex
│
└── scripts/                 ← automation and experiment scripts
    ├── run_preprocessing.py
    └── run_all_experiments.py
```

---

## Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/yourname/pile-settlement-prediction.git
   cd pile-settlement-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run preprocessing:

   ```bash
   python src/data_preprocessing.py
   ```

4. Generate EDA figures:

   ```bash
   python src/eda.py
   ```

---

## Notebooks

| Notebook                        | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| `01_data_exploration.ipynb`     | Exploratory Data Analysis (EDA): pairplots, correlation matrix, PCA, VIF |
| `02_sparse_models.ipynb`        | Sparse regression with Ridge, Lasso, PCA-based selection                 |
| `03_kernel_models.ipynb`        | Nonlinear kernel regression models (RBF, polynomial)                     |
| `04_bayesian_uncertainty.ipynb` | Bayesian regression and uncertainty quantification                       |

---

## Citation

Dataset reference:

> Nejad, F.P., & Jaksa, M.B. (Year). *Dataset on Pile Load Tests*. [Dataset reference PASTE!]

---

**Author:** Aleksandra Burdakova
**License:** MIT
