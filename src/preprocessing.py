"""
Preprocessing and Feature Manipulation Pipelines.

Implements training-isolated preprocessing transformations:
A. Raw (identity / no transform)
B. Standardized (StandardScaler fitted strictly on training data)
C. Feature-Weighted (weights derived from training data statistical relevance, e.g., ANOVA F-scores)
D. Feature-Selected (SelectKBest fitted strictly on training data without oracle knowledge)
E. Oracle (strictly retains ground-truth informative dimensions z in R^r)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


class RawTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer leaving features unscaled."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RawTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64).copy()


class StandardScalerWrapper(BaseEstimator, TransformerMixin):
    """
    Standardizes features by removing mean and scaling to unit variance.
    Strictly fitted on training data only to prevent data leakage.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "StandardScalerWrapper":
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)


class FeatureWeighter(BaseEstimator, TransformerMixin):
    """
    Feature relevance weighting based on training set statistics.

    Computes ANOVA F-statistic for each feature on training set:
        w_j = max(F_j, 0)
    Normalizes weights such that sum(w_j) = d (preserving overall scale).
    Transforms feature matrix by elementwise scaling:
        X_weighted = X * sqrt(w)
    This guarantees that standard Euclidean distance on X_weighted matches
    feature-weighted Euclidean distance:
        || X_weighted - X'_weighted ||^2 = sum_j w_j (X_j - X'_j)^2.

    Parameters
    ----------
    method : str, default='f_classif'
        Method for relevance scoring ('f_classif' or 'inverse_variance').
    """

    def __init__(self, method: str = "f_classif", min_weight: float = 1e-4):
        self.method = method
        self.min_weight = min_weight
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureWeighter":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        d = X.shape[1]

        if self.method == "f_classif":
            f_vals, _ = f_classif(X, y)
            # Replace NaNs/Infs if any
            f_vals = np.nan_to_num(f_vals, nan=self.min_weight, posinf=1e4, neginf=self.min_weight)
            f_vals = np.maximum(f_vals, self.min_weight)
            # Normalize so mean weight is 1.0 (sum is d)
            self.weights_ = (f_vals / np.sum(f_vals)) * float(d)
        elif self.method == "inverse_variance":
            variances = np.var(X, axis=0)
            variances = np.maximum(variances, 1e-8)
            inv_var = 1.0 / variances
            self.weights_ = (inv_var / np.sum(inv_var)) * float(d)
        else:
            raise ValueError(f"Unknown weighting method: {self.method}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("FeatureWeighter must be fitted first.")
        X = np.asarray(X, dtype=np.float64)
        return X * np.sqrt(self.weights_)


class FeatureSelectorWrapper(BaseEstimator, TransformerMixin):
    """
    Supervised feature selection using training set statistics only.
    Selects top k_features based on ANOVA F-scores without oracle knowledge.

    Parameters
    ----------
    k_features : int, default=2
        Number of top features to retain.
    """

    def __init__(self, k_features: int = 2):
        self.k_features = k_features
        self.selector = SelectKBest(score_func=f_classif, k=self.k_features)
        self.selected_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelectorWrapper":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        k = min(self.k_features, X.shape[1])
        self.selector.set_params(k=k)
        self.selector.fit(X, y)
        self.selected_indices_ = self.selector.get_support(indices=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)


class OracleSelector(BaseEstimator, TransformerMixin):
    """
    Oracle baseline transformer.
    Uses ground-truth knowledge of the informative feature indices (0 to r-1).
    Discards all irrelevant noise features.

    Parameters
    ----------
    r : int
        Number of ground-truth informative features.
    """

    def __init__(self, r: int = 2):
        self.r = r

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "OracleSelector":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X[:, : self.r].copy()

