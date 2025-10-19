# src/eda.py
"""
Exploratory Data Analysis (EDA) for pile settlement dataset.
Generates correlation plots, PCA scree plot, pairplots, and collinearity diagnostics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.config import PROC_DIR, FIGURES_DIR

def load_processed_data(target_col):
    """Load cleaned data from PROC_DIR."""
    # Load feature data
    train_features = pd.read_csv(PROC_DIR / "train_cleaned.csv")
    test_features = pd.read_csv(PROC_DIR / "test_cleaned.csv")
    val_features = pd.read_csv(PROC_DIR / "val_cleaned.csv")
    
    # Load target data
    try:
        train_target = pd.read_csv(PROC_DIR / "train_target.csv")
        test_target = pd.read_csv(PROC_DIR / "test_target.csv")
        val_target = pd.read_csv(PROC_DIR / "val_target.csv")

        # If the target file has only one column, it might not have a proper header
        # or the column name might be different
        print("Train target shape:", train_target.shape)
        print("Train target columns:", train_target.columns.tolist())
        
        # If there's only one column, assume it's the target
        if len(train_target.columns) == 1:
            # Rename the single column to our target_col
            train_target.columns = [target_col]
            test_target.columns = [target_col]
            val_target.columns = [target_col]
            print(f"Renamed single column to '{target_col}'")

    except FileNotFoundError as fnf_error:
        print(f"Target file not found error: {fnf_error}")
        raise
    
    # Combine features and target into single DataFrames
    train = pd.concat([train_features, train_target], axis=1)
    test = pd.concat([test_features, test_target], axis=1)
    val = pd.concat([val_features, val_target], axis=1)

    print("Final train columns:", train.columns.tolist())
    print(f"Target column '{target_col}' exists in final train: {target_col in train.columns}")
    
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
    plt.figure(figsize = (8, 6))
    sns.heatmap(corr, cmap="coolwarm", center = 0, annot = False)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    plt.savefig(FIGURES_DIR / "correlation_matrix.png", dpi = 300)
    plt.close()

def plot_marginals(df, target_col):
    """Marginal distributions of key features and target."""
    num_cols = df.select_dtypes(include = np.number).columns
    df[num_cols].hist(bins = 20, figsize = (14, 10))
    plt.suptitle("Marginal Distributions")
    plt.tight_layout()
    plt.show()
    plt.savefig(FIGURES_DIR / "marginal_distributions.png", dpi = 300)
    plt.close()

def plot_settlement_vs_features(df, target_col, top_n = 5):
    """Scatter plots of target vs most correlated features."""
    corr = df.corr(numeric_only = True)[target_col].abs().sort_values(ascending = False)
    top_features = corr.index[1 : top_n + 1]
    for feat in top_features:
        plt.figure(figsize = (8, 6))
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
    """Plot a PCA scree plot showing explained variance by component."""
    from sklearn.decomposition import PCA

    X = df.select_dtypes(include=np.number).dropna()
    pca = PCA()
    pca.fit(X)

    explained_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(explained_var)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_var) + 1), explained_var, marker='o', label='Individual variance')
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='s', linestyle='--', label='Cumulative variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Scree Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(FIGURES_DIR / 'pca_scree_plot.png', dpi=300)
    plt.close()

def load_and_inverse_scale(train_df, test_df, target_col):
    """Reverse scaling of processed datasets using the saved StandardScaler."""
    scaler_path = PROC_DIR / "feature_scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    scaler = joblib.load(scaler_path)
    num_cols = train_df.select_dtypes(include="number").columns.drop(target_col)

    train_unscaled = train_df.copy()
    test_unscaled = test_df.copy()
    train_unscaled[num_cols] = scaler.inverse_transform(train_df[num_cols])
    test_unscaled[num_cols] = scaler.inverse_transform(test_df[num_cols])

    return train_unscaled, test_unscaled