"""
Model evaluation utilities: RMSE, MAE, R2, MSE. Predictions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from src.config import RANDOM_STATE, CV_FOLDS, N_JOBS, RESULTS_DIR


def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred)
    }
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    return metrics, y_pred

def cross_validate_model(model, X, y, model_name, cv = CV_FOLDS, scoring_metrics = None, save_results = True):
    """
    Perform K-fold cross-validation and compute mean ± std for metrics and optionally store results in RESULTS_DIR.

    Parameters
    ----------
    model : sklearn estimator
    X, y : data
    model_name : str
    cv : int, number of folds
    scoring_metrics : list of metrics to evaluate

    Returns
    -------
    DataFrame with mean and std for each metric.
    """
    if scoring_metrics is None:
        scoring_metrics = ["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"]

    kf = KFold(n_splits = cv, shuffle = True, random_state = RANDOM_STATE)
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, scoring = metric, cv = kf, n_jobs = N_JOBS)
        metric_name = metric.replace("neg_", "").replace("mean_", "").replace("root_", "").upper()
        results[metric_name] = [np.mean(np.abs(scores)), np.std(np.abs(scores))]

    df = pd.DataFrame(results, index = ["Mean", "Std"]).T
    print(f"\n=== {model_name} Cross-Validation ({cv}-fold) ===")
    print(df)

    # Save results
    if save_results:
        cv_path = RESULTS_DIR / "cv_results.csv"
        df_out = df.copy()
        df_out.insert(0, "Model", model_name)
        df_out.reset_index(inplace = True)
        df_out.rename(columns = {"index": "Metric"}, inplace = True)

        # Append or create file
        if cv_path.exists():
            existing = pd.read_csv(cv_path)
            combined = pd.concat([existing, df_out], ignore_index = True)
            combined.to_csv(cv_path, index = False)
        else:
            df_out.to_csv(cv_path, index = False)

        print(f"Cross-validation results saved to: {cv_path}")

    return df


def bias_variance_decomposition(model, X_train, y_train, X_test, y_test, n_iter = 20):
    """
    Approximate bias-variance decomposition using repeated train/test splits.
    """
    preds = []
    for _ in range(n_iter):
        model.fit(X_train, y_train)
        preds.append(model.predict(X_test))
    preds = np.array(preds)

    mean_pred = preds.mean(axis = 0)
    bias = np.mean((mean_pred - y_test) ** 2)
    variance = np.mean(np.var(preds, axis = 0))
    total_error = bias + variance

    return {"Bias": bias, "Variance": variance, "TotalError": total_error}