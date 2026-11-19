import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler


class RawTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


class Standardizer:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)


class FeatureWeighter:
    """Standardize features, then weight each coordinate by its training ANOVA F-score."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.weights = None

    def fit(self, X, y):
        X_std = self.scaler.fit_transform(X)
        f_vals, _ = f_classif(X_std, y)
        f_vals = np.nan_to_num(f_vals, nan=1e-4, posinf=1e4, neginf=1e-4)
        f_vals = np.maximum(f_vals, 1e-4)
        # Normalize weights so average weight is 1
        self.weights = (f_vals / np.sum(f_vals)) * X.shape[1]
        return self

    def transform(self, X):
        return self.scaler.transform(X) * np.sqrt(self.weights)


class FeatureSelector:
    """Training-only top-k feature selection via ANOVA F-score (after standardization)."""

    def __init__(self, k_features=2):
        self.k_features = k_features
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=f_classif, k=k_features)

    def fit(self, X, y):
        X_std = self.scaler.fit_transform(X)
        self.selector.set_params(k=min(self.k_features, X.shape[1]))
        self.selector.fit(X_std, y)
        return self

    def transform(self, X):
        return self.selector.transform(self.scaler.transform(X))


class OracleSelector:
    """Diagnostic baseline using only the true informative coordinates z."""

    def __init__(self, r=2):
        self.r = r

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X[:, : self.r], dtype=float)
