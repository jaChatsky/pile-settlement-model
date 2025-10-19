"""
Kernel-based regression methods using Support Vector Regression (SVR)
and Gaussian kernel.
"""

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def kernel_svr(X_train, y_train, X_test, y_test,
               param_grid = None, cv = 5, scoring = 'neg_root_mean_squared_error'):
    """
    Fit an SVR model (optionally with grid search) and return performance metrics.
    """
    if param_grid is not None:
        model = GridSearchCV(SVR(), param_grid, cv = cv, scoring=scoring, n_jobs = -1)
        model.fit(X_train, y_train)
        best_model = model.best_estimator_
    else:
        best_model = SVR(kernel = 'rbf', C = 1.0, gamma = 'scale')
        best_model.fit(X_train, y_train)

    preds = best_model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    return best_model, preds, {"RMSE": rmse, "R2": r2}