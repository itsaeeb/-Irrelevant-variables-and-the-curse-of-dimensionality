"""
Evaluation Metrics Module.

Computes:
1. Classification error rate
2. Predictive log-loss (with documented probability clipping)
3. Pairwise test-to-train distance metrics (D_near, D_far, contrast, ratio)
4. Comprehensive evaluation wrappers and summary aggregations
"""

from typing import Dict, Optional, Union
import numpy as np
from src.distance_metrics import compute_neighbour_distance_metrics


def classification_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate 0-1 classification error rate: (1/N) * sum(y_true != y_pred).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True binary class labels in {0, 1}.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted binary class labels in {0, 1}.

    Returns
    -------
    error_rate : float
        Proportion of misclassified instances in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return float(np.mean(y_true != y_pred))


def predictive_log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Calculate predictive binary cross-entropy (log-loss) with probability clipping.

    Log-loss formula:
    - (1/N) * sum_{i=1}^N [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

    Probability values are clipped to [eps, 1 - eps] to guarantee numerical
    stability and avoid log(0) undefined behavior.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth binary labels in {0, 1}.
    y_prob : np.ndarray of shape (n_samples,) or (n_samples, 2)
        Predicted probability of class 1, or 2D array of [P(y=0), P(y=1)].
    eps : float, default=1e-15
        Clipping threshold to prevent infinite penalties.

    Returns
    -------
    loss : float
        Mean cross-entropy loss.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Handle 2D probability matrices [P(c=0), P(c=1)]
    if y_prob.ndim == 2:
        if y_prob.shape[1] == 2:
            p1 = y_prob[:, 1]
        elif y_prob.shape[1] == 1:
            p1 = y_prob.ravel()
        else:
            raise ValueError(f"Unsupported probability array shape: {y_prob.shape}")
    elif y_prob.ndim == 1:
        p1 = y_prob
    else:
        raise ValueError(f"Unsupported probability array dimensions: {y_prob.ndim}")

    # Probability clipping
    p1_clipped = np.clip(p1, eps, 1.0 - eps)

    # Binary cross-entropy
    loss = -np.mean(y_true * np.log(p1_clipped) + (1.0 - y_true) * np.log(1.0 - p1_clipped))
    return float(loss)


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    X_train: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    eps_prob: float = 1e-15,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Comprehensive evaluation of classifier predictions on evaluation data.

    Calculates:
    - classification error
    - log-loss
    - accuracy
    - (optional) neighbour distance metrics if X_test and X_train are supplied

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray
        Predicted class-1 probabilities.
    X_test : np.ndarray, optional
        Test feature matrix.
    X_train : np.ndarray, optional
        Train feature matrix.
    weights : np.ndarray, optional
        Feature weights for distance computation.
    eps_prob : float, default=1e-15
        Clipping constant for log-loss.

    Returns
    -------
    eval_dict : Dict with error_rate, log_loss, accuracy, and distance metrics.
    """
    error = classification_error_rate(y_true, y_pred)
    loss = predictive_log_loss(y_true, y_prob, eps=eps_prob)

    results = {
        "error_rate": error,
        "accuracy": 1.0 - error,
        "log_loss": loss,
    }

    if X_test is not None and X_train is not None:
        dist_metrics = compute_neighbour_distance_metrics(X_test, X_train, weights=weights)
        results.update(dist_metrics)

    return results

