# scripts/run_models.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import gpytorch
from xgboost import XGBRegressor

from src.config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, RANDOM_STATE, CV_FOLDS
from src.eda import load_processed_data
from src.sparse_regression import run_linear_models, run_pca_models
from src.evaluation import evaluate_model, cross_validate_model, bias_variance_decomposition
from src.bayesian_model import train_gp_model, predict_gp_model, train_gp_on_residuals
from src.kernel_methods import kernel_svr

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok = True)
FIGURES_DIR.mkdir(exist_ok = True)
RESULTS_DIR.mkdir(exist_ok = True)

def load_training_data():
    """Load and prepare training data for modeling."""
    train, test, val = load_processed_data("S-mm")
    
    # Prepare features and target
    X_train = train.drop(columns = ["S-mm"]).select_dtypes(include = [np.number])
    y_train = train["S-mm"]
    
    X_test = test.drop(columns = ["S-mm"]).select_dtypes(include = [np.number])
    y_test = test["S-mm"]
    
    X_val = val.drop(columns = ["S-mm"]).select_dtypes(include = [np.number])
    y_val = val["S-mm"]
    
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Validation data: {X_val.shape}")
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def main():
    """Run sparse linear feature selection + PCA-based regression models."""
    print("Loading and preparing data...")
    X_train, y_train, X_test, y_test, X_val, y_val = load_training_data()

    # === 1. Sparse Linear Models: Feature Selection (Ridge & Lasso) ===
    print("\n=== Sparse Linear Models for Feature Selection (Ridge, Lasso) ===")
    linear_results, linear_preds = run_linear_models(X_train, y_train, X_test, y_test)

    # Choose between Lasso and Ridge based on RMSE
    ridge_key = next((k for k in linear_results.keys() if "Ridge" in k), None)
    lasso_key = next((k for k in linear_results.keys() if "Lasso" in k), None)

    ridge_rmse = linear_results[ridge_key]['RMSE'] if ridge_key else np.inf
    lasso_rmse = linear_results[lasso_key]['RMSE'] if lasso_key else np.inf
    
    if lasso_rmse < ridge_rmse:
        best_model_type = "Lasso"
        print(f"\nLasso performs better (RMSE {lasso_rmse:.4f} < {ridge_rmse:.4f}) — using for feature selection.")
        from sklearn.linear_model import Lasso

        # Safely extract alpha value from model name
        lasso_key = next((k for k in linear_results.keys() if "Lasso" in k), None)
        if lasso_key and "alpha" in lasso_key:
            alpha_str = lasso_key.split("alpha=")[-1].split(")")[0].strip()
            alpha = float(alpha_str)
        else:
            alpha = 0.01  # fallback default

        selector = Lasso(alpha = alpha, random_state = RANDOM_STATE, max_iter = 10000)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[(selector.coef_ != 0)]

    else:
        best_model_type = "Ridge"
        print(f"\nRidge performs better (RMSE {ridge_rmse:.4f} <= {lasso_rmse:.4f}) — using for feature weighting.")
        from sklearn.linear_model import Ridge

        ridge_key = next((k for k in linear_results.keys() if "Ridge" in k), None)
        if ridge_key and "alpha" in ridge_key:
            alpha_str = ridge_key.split("alpha=")[-1].split(")")[0].strip()
            alpha = float(alpha_str)
        else:
            alpha = 1.0  # fallback default

        selector = Ridge(alpha = alpha, random_state = RANDOM_STATE)
        selector.fit(X_train, y_train)
        coef_mags = np.abs(selector.coef_)
        threshold = np.percentile(coef_mags, 70)
        selected_features = X_train.columns[coef_mags >= threshold]

    # Report and save selected features
    print(f"\nSelected {len(selected_features)} informative features out of {X_train.shape[1]} total.")
    print("Selected features:\n", ", ".join(selected_features))

    # Save selected feature names to a CSV file
    selected_features_path = RESULTS_DIR / f"selected_features_{best_model_type.lower()}.csv"
    pd.Series(selected_features, name="Selected_Features").to_csv(selected_features_path, index = False)
    print(f"Selected feature names saved to: {selected_features_path}")

    # Reduce datasets to selected features
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    # === 2. PCA on Selected Features ===
    print("\n=== PCA on Selected Features ===")
    # Run PCA-based models (tests 90%, 95%, 99% explained variance)
    pca_results, pca_preds = run_pca_models(X_train_sel, y_train, X_test_sel, y_test)
    
    # Select the best PCA variant by RMSE
    best_pca_model = min(pca_results, key=lambda k: pca_results[k]['RMSE'])
    best_var_ratio = int(best_pca_model.split("_")[1].replace("%", "")) / 100.0
    print(f"\nBest PCA configuration: {best_pca_model} (variance retained = {best_var_ratio*100:.0f}%)")
    
    # Refit PCA using the best variance ratio for downstream modeling
    from sklearn.decomposition import PCA
    pca = PCA(n_components = best_var_ratio, random_state = RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_sel)
    X_test_pca = pca.transform(X_test_sel)
    print(f"PCA reduced features from {X_train_sel.shape[1]} → {X_train_pca.shape[1]} components")

    # Inspect which features contribute to each component
    pca_loadings = pd.DataFrame(
        pca.components_.T,
        index = selected_features,
        columns = [f"PC{i+1}" for i in range(pca.n_components_)]
        )
    
    # Show top contributing features for the first few components
    for i in range(min(3, pca.n_components_)): # limit to first 3 components
        top_features = pca_loadings.iloc[:, i].abs().sort_values(ascending = False).head(5)
        print(f"\nTop features for PC{i+1}:")
        print(top_features)

    # Save PCA loadings for analysis
    pca_loadings_path = RESULTS_DIR / "pca_feature_loadings.csv"
    pca_loadings.to_csv(pca_loadings_path)
    print(f"\nPCA feature loadings saved to: {pca_loadings_path}")

    # === 3a. Regression on Best PCA Components ===
    print("\n=== Running Regression on Best PCA Components ===")
    from sklearn.linear_model import LinearRegression
    pca_lr = LinearRegression()
    pca_lr.fit(X_train_pca, y_train)

    best_pca_results, best_pca_preds = evaluate_model( pca_lr, X_test_pca, y_test, f"Best_PCA_Linear_{int(best_var_ratio*100)}%" )

    # === 3b. Nonlinear Modeling via Kernel Methods ===
    print("\n=== Running Kernel-Based Models on PCA Data ===")

    # Define parameter grids for kernel methods
    param_grids = {
        "rbf": {"kernel": ["rbf"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        "poly": {"kernel": ["poly"], "C": [0.1, 1, 10], "degree": [2, 3, 4]},
        "sigmoid": {"kernel": ["sigmoid"], "C": [0.1, 1, 10]}
    }

    kernel_results = {}
    kernel_preds = {}
    kernel_models = {}

    for kernel_name, grid in param_grids.items():
        print(f"\nTraining SVR with kernel = {kernel_name}")

        # Train SVR using your kernel_svr() with grid search
        model, preds, metrics = kernel_svr(
            X_train_pca, y_train,
            X_test_pca, y_test,
            param_grid = grid,  # pass grid for tuning
            cv = CV_FOLDS
        )

        # Evaluate the best model using your existing evaluate_model
        results, _ = evaluate_model(model, X_test_pca, y_test, f"PCA_SVR_{kernel_name}")

        # Merge evaluation metrics (evaluate_model covers RMSE/MAE/R2)
        results.update(metrics)
        kernel_results[f"PCA_SVR_{kernel_name}"] = results
        kernel_preds[f"PCA_SVR_{kernel_name}"] = preds
        kernel_models[kernel_name] = model

    # === 3c. Bayesian Regression via Gaussian Processes ===
    print("\n=== Running Gaussian Process Regression for Uncertainty Quantification ===")
    
    # Convert PCA features to torch tensors
    train_x = torch.tensor(X_train_pca, dtype = torch.float32)
    train_y = torch.tensor(
        y_train.values if hasattr(y_train, "values") else y_train,
        dtype=torch.float32
    )
    test_x = torch.tensor(X_test_pca, dtype=torch.float32)

    # Train GP model
    gp_model, gp_likelihood = train_gp_model(train_x, train_y)

    # Predict with uncertainty intervals
    mean, lower, upper = predict_gp_model(gp_model, gp_likelihood, test_x)

    # Evaluate predictions using your existing evaluation function
    # (wrap GP predictions into a small class with .predict() so evaluate_model works)
    class GPWrapper:
        def __init__(self, preds):
            self._preds = preds
        def predict(self, X):
            return self._preds

    gp_wrapper = GPWrapper(mean)
    gp_results, _ = evaluate_model(gp_wrapper, X_test_pca, y_test, "PCA_GaussianProcess")

    gp_preds = pd.DataFrame({
        "y_true": y_test,
        "y_pred_mean": mean,
        "y_pred_lower": lower,
        "y_pred_upper": upper,
    })

    # === 3d. Fit XGBoost on PCA features ===
    print("\n=== Running XGBoost + GP Residual Model ===")
    xgb = XGBRegressor(n_estimators = 200, max_depth = 6, learning_rate = 0.05, random_state = RANDOM_STATE)
    xgb.fit(X_train_pca, y_train)

    # Evaluate XGB normally
    xgb_results, xgb_preds = evaluate_model(xgb, X_test_pca, y_test, "PCA_XGBoost")
    kernel_results["PCA_XGBoost"] = xgb_results
    kernel_preds["PCA_XGBoost"] = xgb_preds
    kernel_models["xgb"] = xgb

    # Train GP on residuals and get combined predictions + intervals
    gp_model_res, gp_lik_res, comb_mean, comb_lower, comb_upper = train_gp_on_residuals(
        xgb, X_train_pca, y_train, X_test_pca, training_iter = 80, lr = 0.05, use_log_target = True)

    # Wrap to use evaluate_model (wrap only mean predictions)
    class TmpWrapper:
        def __init__(self, preds): self._preds = preds
        def predict(self, X): return self._preds

    tmp = TmpWrapper(comb_mean)
    comb_results, _ = evaluate_model(tmp, X_test_pca, y_test, "XGB_plus_GP_residuals")

    # Add combined results and predictions to dictionaries
    kernel_results["XGB_plus_GP_residuals"] = comb_results
    kernel_preds["XGB_plus_GP_residuals"] = comb_mean
    kernel_models["xgb_gp"] = (xgb, gp_model_res, gp_lik_res)

    # Save combined predictions as DataFrame
    comb_preds_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred_mean": comb_mean,
        "y_pred_lower": comb_lower,
        "y_pred_upper": comb_upper,
    })

    # === 4. Merge All Results ===
    print("\n=== Merging All Results ===")

    all_results = {
        **linear_results,
        **pca_results,
        "Best_PCA_Linear": best_pca_results,
        **kernel_results,
        "PCA_GaussianProcess": gp_results
    }

    all_predictions = {
        **linear_preds,
        **pca_preds,
        "Best_PCA_Linear": best_pca_preds,
        **kernel_preds,
        "PCA_GaussianProcess": gp_preds
    }

    # === 5. Save Results ===
    print("\n=== Saving Results ===")
    results_df = pd.DataFrame(all_results).T
    results_df.index.name = "Model"
    results_path = RESULTS_DIR / "model_results.csv"
    results_df.to_csv(results_path)
    print(f"Results saved to: {results_path}")

    # Save standard predictions
    preds_dict = {"y_true": y_test}
    preds_dict.update({
        k: v for k, v in all_predictions.items()
        if k not in ["PCA_GaussianProcess", "XGB_plus_GP_residuals"]
    })
    preds_df = pd.DataFrame(preds_dict)
    preds_path = RESULTS_DIR / "model_predictions.csv"
    preds_df.to_csv(preds_path, index = False)
    print(f"Predictions (with y_true) saved to: {preds_path}")

    # Save GP predictions separately (with uncertainty intervals)
    if 'gp_preds' in locals() and gp_preds is not None:
        gp_path = RESULTS_DIR / "gp_predictions.csv"
        gp_preds.to_csv(gp_path, index=False)
        print(f"GP predictions with uncertainty saved to: {gp_path}")
    else:
        print("No GP predictions to save.")

    # Save combined XGBoost + GP residual predictions
    if 'comb_preds_df' in locals() and not comb_preds_df.empty:
        comb_preds_path = RESULTS_DIR / "xgb_gp_residuals_predictions.csv"
        comb_preds_df.to_csv(comb_preds_path, index=False)
        print(f"Combined XGB+GP predictions saved to: {comb_preds_path}")
    else:
        print("No combined XGB+GP predictions to save.")

    # === 6. CV and bias-variance ===
    print("\n=== Cross-Validation and Bias-Variance Analysis ===")
    cv_summary = {}
    bv_summary = {}

    # CV for top models
    
    cv_summary["PCA+Linear"] = cross_validate_model(pca_lr, X_train_pca, y_train, "PCA+Linear", cv = CV_FOLDS)
    cv_summary["SVR (RBF)"] = cross_validate_model(kernel_models["rbf"], X_train_pca, y_train, "SVR (RBF)", cv = CV_FOLDS)
    cv_summary["XGBoost"] = cross_validate_model(kernel_models["xgb"], X_train_pca, y_train, "XGBoost", cv = CV_FOLDS)

    # Combine & save
    cv_concat = pd.concat(cv_summary, axis = 0)
    cv_path = RESULTS_DIR / "cv_results.csv"
    cv_concat.to_csv(cv_path)

    # Bias-variance decomposition for the best model
    bv_summary["XGBoost"] = bias_variance_decomposition(kernel_models["xgb"], X_train_pca, y_train, X_test_pca, y_test)
    bv_summary["PCA+Linear"] = bias_variance_decomposition(pca_lr, X_train_pca, y_train, X_test_pca, y_test)
    bv_df = pd.DataFrame(bv_summary).T
    bv_df.to_csv(RESULTS_DIR / "bias_variance.csv")
    print("Bias-variance results saved.")


if __name__ == "__main__":
    main()