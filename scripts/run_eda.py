# scripts/run_eda.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.eda import load_processed_data, plot_correlation_matrix, plot_marginals, plot_pairwise, plot_pca_scree, plot_settlement_vs_features, compute_vif, load_and_inverse_scale
from src.config import PROC_DIR, FIGURES_DIR

def run_eda(target_col = "S-mm"):
    """Main function to run all EDA steps."""
    train, test, _ = load_processed_data(target_col)
    train_unscaled, test_unscaled = load_and_inverse_scale(train, test, target_col)
    print("\n=== Data Loaded ===")
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")

    # Separate features and target
    X_train, y_train = train.drop(columns = [target_col]), train[target_col]
    X_test, y_test = test.drop(columns = [target_col]), test[target_col]

    # Separate features and target
    X_train_unscaled, y_train_unscaled = train_unscaled.drop(columns = [target_col]), train_unscaled[target_col]
    X_test_unscaled, y_test_unscaled = test_unscaled.drop(columns = [target_col]), test_unscaled[target_col]

    # === 1. Target distribution comparison ===
    print("\n=== Target Distribution Comparison ===")
    print("Training target stats:")
    print(f" Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
    print("Test target stats:")
    print(f" Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
    
    plt.figure(figsize = (6,4))
    sns.kdeplot(y_train, label = "Train", fill = True, alpha = 0.5)
    sns.kdeplot(y_test, label = "Test", fill = True, alpha = 0.5)
    plt.title(f"Target Distribution Comparison ({target_col})")
    plt.xlabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(FIGURES_DIR / "target_distribution_comparison.png", dpi = 300)
    plt.close()

    # === 2. Feature distribution summary ===
    print("\n=== Feature Means Comparison (first 15 features) ===")
    feature_means_df = pd.DataFrame({
        "Train_Mean": X_train_unscaled.mean().round(4),
        "Test_Mean": X_test_unscaled.mean().round(4),
        "Mean_Difference": (X_train_unscaled.mean() - X_test_unscaled.mean()).round(4)
    }).head(15)
    display(feature_means_df)

    # === 3. Pairwise scatterplots ===
    print("\nGenerating pairwise scatterplots...")
    plot_pairwise(train, target_col)

    # === 4. Correlation matrix ===
    print("Generating correlation heatmap...")
    plot_correlation_matrix(train)

    # === 5. Marginal distributions ===
    print("Generating marginal histograms...")
    plot_marginals(train, target_col)

    # === 6. Settlement vs. top correlated features ===
    print("Generating settlement vs top correlated features...")
    plot_settlement_vs_features(train, target_col)

    # === 7. Variance Inflation Factor (VIF) ===
    print("Computing VIF to detect multicollinearity...")
    vif_data = compute_vif(train.drop(columns = [target_col]))
    print("\nHigh VIF features (potential collinearity):")
    print(vif_data[vif_data["VIF"] > 10])

    # === 8. PCA scree plot ===
    print("Generating PCA scree plot...")
    plot_pca_scree(train.drop(columns = [target_col]))

    print("EDA complete. All figures saved to:", FIGURES_DIR)

if __name__ == "__main__":
    run_eda()