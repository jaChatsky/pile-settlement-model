"""
Model evaluation utilities: cross-validation, RMSE, bias-variance.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def k_fold_rmse(model_func, X, y, k = 5, **model_params):
    kf = KFold(n_splits = k, shuffle = True, random_state = 42)
    rmses = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        _, mse = model_func(X_train, y_train, X_test, y_test, **model_params)
        rmses.append(np.sqrt(mse))
    return np.mean(rmses), np.std(rmses)