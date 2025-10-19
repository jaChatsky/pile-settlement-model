"""
Kernel-based regression methods using Support Vector Regression (SVR)
and Gaussian kernel.
"""

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def kernel_svr(X_train, y_train, X_test, y_test, kernel = "rbf", C = 1.0, gamma = "scale"):
    model = SVR(kernel = kernel, C = C, gamma = gamma)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse