"""
Bayesian regression using Gaussian Processes for uncertainty quantification.
"""

import numpy as np
import torch
import gpytorch


class GaussianProcessModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(train_x, train_y, training_iter = 50):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GaussianProcessModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood

def predict_gp_model(model, likelihood, X_test):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test))
        mean = preds.mean.detach().cpu().numpy()
        lower, upper = preds.confidence_region()
        lower = lower.detach().cpu().numpy()
        upper = upper.detach().cpu().numpy()

    return mean, lower, upper

def train_gp_on_residuals(base_model, X_train, y_train, X_test, training_iter = 50, lr = 0.05, use_log_target = True):
    """
    Train a Gaussian Process on the residuals of a base regression model (e.g., XGBoost).

    Parameters
    ----------
    base_model : fitted model with .predict()
    X_train, y_train : training data
    X_test : test data
    training_iter : number of GP training iterations
    lr : learning rate

    Optionally applying a log1p transform to stabilize variance.

    Returns
    -------
    model : trained GP model
    likelihood : fitted likelihood
    y_pred_comb_mean : combined mean predictions (base + GP correction)
    y_pred_lower, y_pred_upper : uncertainty bounds
    """

    # Log transform True
    if use_log_target:
        y_train_t = np.log1p(y_train)
    else:
        y_train_t = y_train

    # Compute residuals
    base_preds_train = base_model.predict(X_train)
    if use_log_target:
        base_preds_train = np.log1p(np.maximum(base_preds_train, 0))  # ensure non-negativity
    residuals = y_train_t - base_preds_train

    # Convert to torch tensors
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(residuals, dtype=torch.float32)
    test_x = torch.tensor(X_test, dtype=torch.float32)

    # Build GP model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GaussianProcessModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # Switch to evaluation mode
    model.eval()
    likelihood.eval()

    # Predict residual correction on test set
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(test_x))
        gp_mean = preds.mean.detach().numpy()
        gp_lower, gp_upper = preds.confidence_region()
        gp_lower = gp_lower.detach().numpy()
        gp_upper = gp_upper.detach().numpy()

    # Combine base model + GP correction (convert to NumPy explicitly)
    base_preds_test = np.asarray(base_model.predict(X_test))
    if use_log_target:
        base_preds_test = np.log1p(np.maximum(base_preds_test, 0))

    y_pred_comb_mean = base_preds_test + gp_mean
    y_pred_lower = base_preds_test + gp_lower
    y_pred_upper = base_preds_test + gp_upper

    # Inverse-transforms predictions
    if use_log_target:
        y_pred_comb_mean = np.expm1(y_pred_comb_mean)
        y_pred_lower = np.expm1(y_pred_lower)
        y_pred_upper = np.expm1(y_pred_upper)

    return model, likelihood, y_pred_comb_mean, y_pred_lower, y_pred_upper