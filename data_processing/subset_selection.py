"""
Hardest Subset Selection for Training Data

This module implements optimization-based subset selection to identify
the hardest/most informative training samples. Inspired by trimmed loss
optimization and hard example mining techniques.

Supports both Gurobi (if available) and scipy.optimize as fallback.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import warnings

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    warnings.warn("Gurobi not available. Using scipy.optimize fallback.")

try:
    from scipy.optimize import minimize, linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Some functions may not work.")


def identify_hard_samples(
    X: np.ndarray,
    y: np.ndarray,
    model: Optional[BaseEstimator] = None,
    method: str = 'prediction_error',
    top_k: Optional[int] = None,
    top_p: float = 0.7
) -> np.ndarray:
    """
    Identify hardest samples based on prediction errors or other criteria.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    model : BaseEstimator, optional
        Trained model to evaluate samples. If None, trains a simple LogisticRegression
    method : str
        Method to identify hard samples:
        - 'prediction_error': Samples with highest prediction error
        - 'confidence': Samples with lowest prediction confidence
        - 'margin': Samples closest to decision boundary
    top_k : int, optional
        Number of hardest samples to return. If None, uses top_p
    top_p : float
        Proportion of hardest samples to return (default: 0.7)
    
    Returns:
    --------
    np.ndarray
        Boolean mask indicating hardest samples
    """
    if model is None:
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    
    if method == 'prediction_error':
        # Samples with highest loss
        losses = []
        for i in range(len(X)):
            loss = log_loss([y[i]], [y_pred_proba[i]], labels=model.classes_)
            losses.append(loss)
        scores = np.array(losses)
    
    elif method == 'confidence':
        # Samples with lowest confidence (furthest from 0.5)
        # For binary classification, use distance from 0.5
        if y_pred_proba.shape[1] == 2:
            confidences = np.abs(y_pred_proba[:, 1] - 0.5)
        else:
            confidences = np.max(y_pred_proba, axis=1)
        scores = -confidences  # Negative because we want LOW confidence
    
    elif method == 'margin':
        # Samples closest to decision boundary
        if y_pred_proba.shape[1] == 2:
            margins = np.abs(y_pred_proba[:, 1] - 0.5)
        else:
            sorted_proba = np.sort(y_pred_proba, axis=1)
            margins = sorted_proba[:, -1] - sorted_proba[:, -2]
        scores = -margins  # Negative because we want SMALL margins
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Select top k or top p proportion
    if top_k is None:
        top_k = int(len(X) * top_p)
    
    # Get indices of hardest samples
    hardest_indices = np.argsort(scores)[-top_k:]
    mask = np.zeros(len(X), dtype=bool)
    mask[hardest_indices] = True
    
    return mask


def trimmed_loss_subset_selection_gurobi(
    X: np.ndarray,
    y: np.ndarray,
    m: Optional[int] = None,
    m_ratio: float = 0.7,
    lambda_reg: float = 0.1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select hardest subset using trimmed loss optimization with Gurobi.
    
    This implements a similar approach to the Julia code:
    - Selects m hardest samples (with trimming threshold t)
    - Minimizes: m*t + sum(s_i) + lambda*||beta||_1
    where s_i = max(0, |residual_i| - t)
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    m : int, optional
        Number of samples to select. If None, uses m_ratio
    m_ratio : float
        Proportion of samples to select (default: 0.7)
    lambda_reg : float
        L1 regularization strength
    verbose : bool
        Whether to print optimization details
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (selected_mask, beta_coefficients)
    """
    if not GUROBI_AVAILABLE:
        raise ImportError("Gurobi is required for this function. Install with: pip install gurobipy")
    
    n, p = X.shape
    if m is None:
        m = int(np.ceil(m_ratio * n))
    
    # Create model
    model = gp.Model("TrimmedLoss")
    model.setParam('OutputFlag', 1 if verbose else 0)
    
    # Decision variables
    beta = model.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
    a = model.addVars(n, lb=0, name="a")  # |residuals|
    b = model.addVars(p, lb=0, name="b")  # |beta|
    t = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="t")  # trimming threshold
    s = model.addVars(n, lb=0, name="s")  # hinge: max(0, a - t)
    z = model.addVars(n, vtype=GRB.BINARY, name="z")  # sample selection
    
    # Constraints: |residual| <= a
    residuals = [y[i] - sum(X[i, j] * beta[j] for j in range(p)) for i in range(n)]
    for i in range(n):
        model.addConstr(a[i] >= residuals[i], name=f"residual_pos_{i}")
        model.addConstr(a[i] >= -residuals[i], name=f"residual_neg_{i}")
    
    # Constraints: |beta| <= b
    for j in range(p):
        model.addConstr(b[j] >= beta[j], name=f"beta_pos_{j}")
        model.addConstr(b[j] >= -beta[j], name=f"beta_neg_{j}")
    
    # Constraints: hinge s >= a - t
    for i in range(n):
        model.addConstr(s[i] >= a[i] - t, name=f"hinge_{i}")
    
    # Constraint: select exactly m samples
    model.addConstr(sum(z[i] for i in range(n)) == m, name="select_m")
    
    # Big-M constraint: only count losses for selected samples
    M = 1000.0  # Large constant
    for i in range(n):
        model.addConstr(s[i] <= M * z[i], name=f"select_constraint_{i}")
    
    # Objective: m*t + sum(s_i) + lambda*sum(|beta|)
    model.setObjective(
        m * t + sum(s[i] for i in range(n)) + lambda_reg * sum(b[j] for j in range(p)),
        GRB.MINIMIZE
    )
    
    # Optimize
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        # Get selected samples
        selected_mask = np.array([z[i].X > 0.5 for i in range(n)])
        beta_values = np.array([beta[j].X for j in range(p)])
        return selected_mask, beta_values
    else:
        raise RuntimeError(f"Optimization failed with status: {model.status}")


def trimmed_loss_subset_selection_scipy(
    X: np.ndarray,
    y: np.ndarray,
    m: Optional[int] = None,
    m_ratio: float = 0.7,
    lambda_reg: float = 0.1,
    method: str = 'SLSQP'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select hardest subset using trimmed loss optimization with scipy.
    
    This is a simplified version that uses continuous relaxation and
    then rounds to binary selection.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    m : int, optional
        Number of samples to select. If None, uses m_ratio
    m_ratio : float
        Proportion of samples to select (default: 0.7)
    lambda_reg : float
        L1 regularization strength
    method : str
        Optimization method for scipy.minimize
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (selected_mask, beta_coefficients)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for this function.")
    
    n, p = X.shape
    if m is None:
        m = int(np.ceil(m_ratio * n))
    
    # Simplified approach: use iterative hard thresholding
    # Start with all samples, iteratively remove easiest ones
    
    # Initialize with simple model
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=lambda_reg, max_iter=2000, random_state=42)
    
    # Fit on all data first
    model.fit(X, y)
    residuals = np.abs(y - model.predict(X))
    
    # Select m hardest samples (highest residuals)
    hardest_indices = np.argsort(residuals)[-m:]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[hardest_indices] = True
    
    # Refit model on selected subset
    model.fit(X[selected_mask], y[selected_mask])
    
    return selected_mask, model.coef_


def select_hardest_subset(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'auto',
    m: Optional[int] = None,
    m_ratio: float = 0.7,
    lambda_reg: float = 0.1,
    use_gurobi: bool = False,
    verbose: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Main function to select hardest subset of training data.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    method : str
        Selection method:
        - 'auto': Automatically choose best available method
        - 'optimization': Use trimmed loss optimization (Gurobi or scipy)
        - 'prediction_error': Use simple prediction error ranking
        - 'confidence': Use confidence-based ranking
        - 'margin': Use margin-based ranking
    m : int, optional
        Number of samples to select. If None, uses m_ratio
    m_ratio : float
        Proportion of samples to select (default: 0.7)
    lambda_reg : float
        Regularization strength (for optimization methods)
    use_gurobi : bool
        Force use of Gurobi (if available)
    verbose : bool
        Print progress information
    
    Returns:
    --------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (selected_mask, beta_coefficients if optimization method used)
    """
    n = len(X)
    if m is None:
        m = int(np.ceil(m_ratio * n))
    
    if method == 'auto':
        # Choose method based on availability
        if use_gurobi and GUROBI_AVAILABLE:
            method = 'optimization'
        elif SCIPY_AVAILABLE:
            method = 'optimization'
        else:
            method = 'prediction_error'
    
    if method == 'optimization':
        if use_gurobi and GUROBI_AVAILABLE:
            if verbose:
                print("Using Gurobi for optimization-based subset selection...")
            selected_mask, beta = trimmed_loss_subset_selection_gurobi(
                X, y, m=m, m_ratio=m_ratio, lambda_reg=lambda_reg, verbose=verbose
            )
            return selected_mask, beta
        elif SCIPY_AVAILABLE:
            if verbose:
                print("Using scipy.optimize for optimization-based subset selection...")
            selected_mask, beta = trimmed_loss_subset_selection_scipy(
                X, y, m=m, m_ratio=m_ratio, lambda_reg=lambda_reg
            )
            return selected_mask, beta
        else:
            warnings.warn("Optimization methods not available. Falling back to prediction_error.")
            method = 'prediction_error'
    
    if method in ['prediction_error', 'confidence', 'margin']:
        if verbose:
            print(f"Using {method} method for subset selection...")
        selected_mask = identify_hard_samples(
            X, y, method=method, top_k=m, top_p=m_ratio
        )
        return selected_mask, None
    
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_subset_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    method: str = 'auto',
    m_ratio: float = 0.7,
    lambda_reg: float = 0.1,
    preserve_class_balance: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Apply subset selection and return filtered training data.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray, optional
        Validation features (returned unchanged)
    y_val : np.ndarray, optional
        Validation labels (returned unchanged)
    method : str
        Selection method (see select_hardest_subset)
    m_ratio : float
        Proportion of samples to select
    lambda_reg : float
        Regularization strength
    preserve_class_balance : bool
        If True, maintains class balance in selected subset
    verbose : bool
        Print information
    
    Returns:
    --------
    Tuple containing:
        - X_train_selected: Selected training features
        - y_train_selected: Selected training labels
        - X_val: Validation features (unchanged)
        - y_val: Validation labels (unchanged)
    """
    if preserve_class_balance:
        # Select hardest samples per class
        unique_classes = np.unique(y_train)
        selected_masks = []
        
        for cls in unique_classes:
            cls_mask = y_train == cls
            X_cls = X_train[cls_mask]
            y_cls = y_train[cls_mask]
            
            m_cls = int(np.ceil(m_ratio * np.sum(cls_mask)))
            
            mask_cls = identify_hard_samples(
                X_cls, y_cls, method='prediction_error' if method == 'auto' else method,
                top_k=m_cls
            )
            
            # Map back to original indices
            cls_indices = np.where(cls_mask)[0]
            full_mask = np.zeros(len(y_train), dtype=bool)
            full_mask[cls_indices[mask_cls]] = True
            selected_masks.append(full_mask)
        
        selected_mask = np.any(selected_masks, axis=0)
    else:
        # Select globally hardest samples
        selected_mask, _ = select_hardest_subset(
            X_train, y_train, method=method, m_ratio=m_ratio,
            lambda_reg=lambda_reg, verbose=verbose
        )
    
    if verbose:
        print(f"\nSubset Selection Summary:")
        print(f"  Original samples: {len(X_train)}")
        print(f"  Selected samples: {np.sum(selected_mask)} ({np.sum(selected_mask)/len(X_train)*100:.1f}%)")
        if preserve_class_balance:
            for cls in unique_classes:
                cls_selected = np.sum((y_train == cls) & selected_mask)
                cls_total = np.sum(y_train == cls)
                print(f"  Class {cls}: {cls_selected}/{cls_total} ({cls_selected/cls_total*100:.1f}%)")
    
    return (
        X_train[selected_mask],
        y_train[selected_mask],
        X_val,
        y_val
    )

