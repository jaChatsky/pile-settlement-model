# src/eda.py
"""
Exploratory Data Analysis (EDA) for pile settlement dataset.
Generates correlation plots, PCA scree plot, pairplots, and collinearity diagnostics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.config import PROC_DIR, FIGURES_DIR
from src.sparse_regression import pca_feature_selection

def load_processed_data(target_col):
    """Load cleaned data from PROC_DIR."""
    # Load feature data
    train_features = pd.read_csv(PROC_DIR / "train_cleaned.csv")
    test_features = pd.read_csv(PROC_DIR / "test_cleaned.csv")
    val_features = pd.read_csv(PROC_DIR / "val_cleaned.csv")
    
    # Load target data
    train_target = pd.read_csv(PROC_DIR / "train_target.csv", names=[target_col])
    test_target = pd.read_csv(PROC_DIR / "test_target.csv", names=[target_col])
    val_target = pd.read_csv(PROC_DIR / "val_target.csv", names=[target_col])
    
    # Combine features and target into single DataFrames
    train = pd.concat([train_features, train_target], axis=1)
    test = pd.concat([test_features, test_target], axis=1)
    val = pd.concat([val_features, val_target], axis=1)

    print("Loaded train columns:", train.columns.tolist())
    print("Loaded test columns:", test.columns.tolist())
    print("Loaded val columns:", val.columns.tolist())
    
    return train, test, val

def plot_pairwise(df, target_col):
    """Pairwise scatterplots of numeric features and target."""
    sns.pairplot(df.select_dtypes(include = np.number), diag_kind = "kde")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pairwise_scatterplots.png", dpi = 300)
    plt.close()

def plot_correlation_matrix(df):
    """Correlation heatmap of numeric features."""
    corr = df.select_dtypes(include = np.number).corr()
    plt.figure(figsize = (10, 8))
    sns.heatmap(corr, cmap="coolwarm", center = 0, annot = False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi = 300)
    plt.close()

def plot_marginals(df, target_col):
    """Marginal distributions of key features and target."""
    num_cols = df.select_dtypes(include = np.number).columns
    df[num_cols].hist(bins = 20, figsize = (14, 10))
    plt.suptitle("Marginal Distributions")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "marginal_distributions.png", dpi = 300)
    plt.close()

def plot_settlement_vs_features(df, target_col, top_n = 5):
    """Scatter plots of target vs most correlated features."""
    corr = df.corr(numeric_only = True)[target_col].abs().sort_values(ascending = False)
    top_features = corr.index[1 : top_n + 1]
    for feat in top_features:
        plt.figure()
        sns.scatterplot(x=df[feat], y=df[target_col])
        plt.title(f"{target_col} vs {feat}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"settlement_vs_{feat}.png", dpi = 300)
        plt.close()

def compute_vif(df):
    """Compute Variance Inflation Factor for each feature."""
    X = df.select_dtypes(include = np.number).dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data.to_csv(FIGURES_DIR / "vif_report.csv", index = False)
    return vif_data

def plot_pca_scree(df):
    """PCA scree plot showing explained variance ratio."""
    X = df.select_dtypes(include = np.number).dropna()
    X_pca, var_ratio = pca_feature_selection(X, n_components = 0.95)
    plt.figure(figsize = (8, 5))
    plt.plot(np.cumsum(var_ratio), marker="o")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_scree_plot.png", dpi = 300)
    plt.close()

def run_eda(target_col = "S-mm"):
    """Main function to run all EDA steps."""
    train, _, _ = load_processed_data(target_col)

    plot_pairwise(train, target_col)
    plot_correlation_matrix(train)
    plot_marginals(train, target_col)
    plot_settlement_vs_features(train, target_col)
    vif_data = compute_vif(train.drop(columns = [target_col]))
    plot_pca_scree(train.drop(columns = [target_col]))

    print("EDA complete. Figures saved to:", FIGURES_DIR)
    print("High VIF features (potential collinearity):")
    print(vif_data[vif_data["VIF"] > 10])

if __name__ == "__main__":
    run_eda()