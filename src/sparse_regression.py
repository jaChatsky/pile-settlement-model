"""
Sparse regression models: Ridge, Lasso, and feature selection using PCA.
"""

from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error


def ridge_regression(X_train, y_train, X_test, y_test, alpha = 1.0):
    model = Ridge(alpha = alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse


def lasso_regression(X_train, y_train, X_test, y_test, alpha = 0.01):
    model = Lasso(alpha = alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse


def pca_feature_selection(X, n_components = 0.95):
    pca = PCA(n_components = n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_