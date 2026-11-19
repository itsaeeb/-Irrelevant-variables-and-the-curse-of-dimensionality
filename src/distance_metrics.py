"""
Distance Metrics and High-Dimensional Concentration Calculations.

Computes pairwise Euclidean and feature-weighted distances between test and training
observations to quantify the curse of dimensionality and distance concentration:
- Nearest-neighbour distance (D_near)
- Farthest-neighbour distance (D_far)
- Distance contrast: C = (D_far - D_near) / D_near
- Distance ratio: R = D_near / D_far
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist


def compute_distance_matrix(
    X_test: np.ndarray,
    X_train: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute pairwise (weighted) Euclidean distances from test to train points.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_test, d)
        Test observations.
    X_train : np.ndarray of shape (n_train, d)
        Training observations.
    weights : np.ndarray of shape (d,), optional
        Non-negative feature weights. If provided, computes:
        D_W(x, x') = sqrt( sum_m w_m * (x_m - x'_m)^2 ).

    Returns
    -------
    dist_matrix : np.ndarray of shape (n_test, n_train)
        Pairwise distance matrix.
    """
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != X_test.shape[1]:
            raise ValueError(
                f"Weight length {weights.shape[0]} does not match feature dimension {X_test.shape[1]}"
            )
        if np.any(weights < 0):
            raise ValueError("Feature weights must be non-negative.")
        sqrt_w = np.sqrt(weights)
        X_test_scaled = X_test * sqrt_w
        X_train_scaled = X_train * sqrt_w
        return cdist(X_test_scaled, X_train_scaled, metric="euclidean")
    else:
        return cdist(X_test, X_train, metric="euclidean")


def compute_neighbour_distance_metrics(
    X_test: np.ndarray,
    X_train: np.ndarray,
    weights: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate nearest and farthest neighbour distances, contrast, and ratio for test points.

    For every test observation:
    - Calculates distances to all training observations.
    - Finds nearest distance D_near and farthest distance D_far.
    - Computes contrast C = (D_far - D_near) / D_near.
    - Computes ratio R = D_near / D_far.
    - Aggregates mean and standard deviation over all test observations.

    Parameters
    ----------
    X_test : np.ndarray of shape (n_test, d)
        Evaluation (test) observations.
    X_train : np.ndarray of shape (n_train, d)
        Reference (training) observations.
    weights : np.ndarray of shape (d,), optional
        Optional feature weights.
    eps : float, default=1e-12
        Small epsilon to prevent division by zero if D_near == 0.

    Returns
    -------
    metrics : Dict containing:
        - 'mean_d_near': Mean nearest distance across test observations
        - 'std_d_near': Std of nearest distance
        - 'mean_d_far': Mean farthest distance across test observations
        - 'std_d_far': Std of farthest distance
        - 'mean_contrast': Mean contrast C = (D_far - D_near) / D_near
        - 'std_contrast': Std of contrast
        - 'mean_ratio': Mean ratio R = D_near / D_far
        - 'std_ratio': Std of ratio
        - 'd_near': Raw per-sample nearest distances (n_test,)
        - 'd_far': Raw per-sample farthest distances (n_test,)
        - 'contrast': Raw per-sample contrast values (n_test,)
        - 'ratio': Raw per-sample ratio values (n_test,)
    """
    dist_matrix = compute_distance_matrix(X_test, X_train, weights=weights)

    # For each test observation (row), find min and max distance to train points
    d_near = np.min(dist_matrix, axis=1)
    d_far = np.max(dist_matrix, axis=1)

    # Calculate contrast and ratio safely
    denom_near = np.maximum(d_near, eps)
    denom_far = np.maximum(d_far, eps)

    contrast = (d_far - d_near) / denom_near
    ratio = d_near / denom_far

    return {
        "mean_d_near": float(np.mean(d_near)),
        "std_d_near": float(np.std(d_near)),
        "mean_d_far": float(np.mean(d_far)),
        "std_d_far": float(np.std(d_far)),
        "mean_contrast": float(np.mean(contrast)),
        "std_contrast": float(np.std(contrast)),
        "mean_ratio": float(np.mean(ratio)),
        "std_ratio": float(np.std(ratio)),
        "d_near": d_near,
        "d_far": d_far,
        "contrast": contrast,
        "ratio": ratio,
    }

