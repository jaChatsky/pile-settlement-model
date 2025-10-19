# Pile Settlement Prediction

This project models and predicts **pile settlement** using sparse regression and kernel-based methods.

Bayesian techniques are incorporated to quantify uncertainty in predictions, improving engineering decision-making under variable soil and loading conditions.                                                                           

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
│   └── 02_models_evaluation.ipynb
│
├── src/                     ← Python source code
│   ├── bayesian_model.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── kernel_methods.py
│   └── sparse_regression.py
│
├── results/                 ← model outputs, figures, and metrics
│   ├── figures/
│   └── models/
│
├── report/                  ← academic report
│   └── Project_SF2935_Group_19.pdf
│
└── scripts/                 ← automation and experiment scripts
    ├── run_eda.py
    ├── run_models.py
    └── run_preprocessing.py
```

---

## Setup Instructions

1. Clone this repository:

   ```bash
   git clone https://github.com/yourname/pile-settlement-model.git
   cd pile-settlement-model
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
   python src/run_eda.py
   ```

5. Generate models:

   ```bash
   python src/run_models.py
   ```

---

## Notebooks

| Notebook                        | Description                                                              |
| ------------------------------- | ------------------------------------------------------------------------ |
| `01_data_exploration.ipynb`     | Exploratory Data Analysis (EDA): pairplots, correlation matrix, PCA, VIF |
| `02_models_evaluation.ipynb`    | Sparse, kernel and Bayesian regressions with PCA-based selection         |

---

## Citation

Dataset reference:

> Nejad, F.P., & Jaksa, M.B. (Year). *Dataset on Pile Load Tests*. [LINK](https://doi.org/10.1016/J.COMPGEO.2017.04.003)

---

**Author:** Aleksandra Burdakova
**License:** MIT
