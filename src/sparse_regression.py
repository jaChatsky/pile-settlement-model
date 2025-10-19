"""
Sparse regression models: Ridge, Lasso, and feature selection using PCA.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.config import FIGURES_DIR, RANDOM_STATE
from src.evaluation import evaluate_model
from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def run_linear_models(X_train, y_train, X_test, y_test):
    """Run baseline and regularized linear models."""
    results = {}
    predictions = {}
    
    # 1. Baseline: Linear Regression (OLS)
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    results['OLS'], predictions['OLS'] = evaluate_model(ols, X_test, y_test, "OLS Linear Regression")
    
    # 2. Ridge Regression (multiple alphas)
    alphas_ridge = [0.001, 0.01, 0.1, 1.0, 10.0]
    ridge_rmse_list = []
    best_ridge_rmse = float('inf')
    best_ridge = None
    
    for alpha in alphas_ridge:
        ridge = Ridge(alpha = alpha, random_state = RANDOM_STATE)
        ridge.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, ridge.predict(X_test)))
        ridge_rmse_list.append(rmse)
        
        if rmse < best_ridge_rmse:
            best_ridge_rmse = rmse
            best_ridge = ridge
            best_alpha_ridge = alpha

    # Visualization
    plt.figure(figsize = (6,4))
    plt.plot(alphas_ridge, ridge_rmse_list, marker = 'o')
    plt.xscale('log')
    plt.title('Ridge Regression RMSE vs Alpha')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'ridge_rmse_vs_alpha.png')
    plt.close()
    
    results['Ridge'], predictions['Ridge'] = evaluate_model(best_ridge, X_test, y_test, 
                                                           f"Ridge Regression (alpha = {best_alpha_ridge})")
    
    # 3. Lasso Regression (multiple alphas)
    alphas_lasso = [0.05, 0.1, 0.05, 0.08]
    lasso_rmse_list = []
    best_lasso_rmse = float('inf')
    best_lasso = None
    
    for alpha in alphas_lasso:
        lasso = Lasso(alpha = alpha, random_state = RANDOM_STATE, max_iter = 10000)
        lasso.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, lasso.predict(X_test)))
        lasso_rmse_list.append(rmse)
        
        if rmse < best_lasso_rmse:
            best_lasso_rmse = rmse
            best_lasso = lasso
            best_alpha_lasso = alpha

    # Visualization
    plt.figure(figsize = (6,4))
    plt.plot(alphas_lasso, lasso_rmse_list, marker = 'o', color='green')
    plt.xscale('log')
    plt.title('Lasso Regression RMSE vs Alpha')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'lasso_rmse_vs_alpha.png')
    plt.close()
    
    results['Lasso'], predictions['Lasso'] = evaluate_model(best_lasso, X_test, y_test, 
                                                           f"Lasso Regression (alpha = {best_alpha_lasso})")
    
    return results, predictions
    

def run_pca_models(X_train, y_train, X_test, y_test):
    """Run PCA-based feature reduction models."""
    results = {}
    predictions = {}
    
    variance_ratios = [0.90, 0.95, 0.99]
    pca_rmse_list = []
    
    for var_ratio in variance_ratios:
        # Apply PCA
        pca = PCA(n_components = var_ratio)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        print(f"\nPCA with {var_ratio*100}% variance - {X_train_pca.shape[1]} components")
        
        # Linear regression on PCA components
        from sklearn.linear_model import LinearRegression
        pca_lr = LinearRegression()
        pca_lr.fit(X_train_pca, y_train)

        y_pred = pca_lr.predict(X_test_pca)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        pca_rmse_list.append(rmse)
        
        model_name = f"PCA_{int(var_ratio*100)}%_Linear"
        results[model_name], predictions[model_name] = evaluate_model(pca_lr, X_test_pca, y_test, model_name)

    # Visualization
    plt.figure(figsize = (6,4))
    plt.plot([int(v*100) for v in variance_ratios], pca_rmse_list, marker = 'o', color = 'darkmagenta')
    plt.title('PCA Linear Regression RMSE vs Variance Retained')
    plt.xlabel('Variance Retained (%)')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pca_rmse_vs_variance.png')
    plt.close()
    
    return results, predictions
