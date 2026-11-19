"""
Classification Algorithms Module.

Implements four classifiers:
1. K-Nearest Neighbours (KNN) with configurable k, Euclidean or weighted distance.
2. Gaussian Kernel Density Classifier (GKDC) with log-density, logsumexp, Bayes' rule.
3. Gaussian Classifier (Generative QDA / regularized covariance with Cholesky factorization).
4. Logistic Regression (L2 regularized with scikit-learn backend).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


# =====================================================================
# A. K-Nearest Neighbours Classifier
# =====================================================================
class KNNClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbours binary classifier.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors (k).
    weights : np.ndarray of shape (d,), optional
        Feature weights for weighted Euclidean distance:
        D_W(x, x') = sqrt(sum_m w_m * (x_m - x'_m)^2).
    """

    def __init__(self, n_neighbors: int = 5, weights: Optional[np.ndarray] = None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)
        if len(self.classes_) > 2:
            raise ValueError("Only binary classification is supported.")

        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape[0] != X.shape[1]:
                raise ValueError("Weights dimension does not match feature dimension.")
            self.X_train_ = X * np.sqrt(weights)
        else:
            self.X_train_ = X

        self.y_train_ = y
        return self

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.weights is not None:
            X_eval = X * np.sqrt(self.weights)
        else:
            X_eval = X
        # Returns pairwise distances of shape (n_eval, n_train)
        return cdist(X_eval, self.X_train_, metric="euclidean")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute class posterior probabilities as neighbor class proportions.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            [P(y=0|x), P(y=1|x)].
        """
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("Classifier has not been fitted.")

        k = min(self.n_neighbors, len(self.y_train_))
        dists = self._compute_distances(X)  # (N_eval, N_train)

        # Find indices of k nearest neighbors
        # argpartition is O(N) instead of full sort O(N log N)
        knn_indices = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

        # Gather labels of neighbors: shape (N_eval, k)
        neighbor_labels = self.y_train_[knn_indices]

        # Class 1 proportion
        p1 = np.mean(neighbor_labels == 1, axis=1)
        p0 = 1.0 - p1

        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels in {0, 1}."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int32)


# =====================================================================
# B. Gaussian Kernel Density Classifier
# =====================================================================
class GaussianKDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Kernel Density Classifier for binary classification.

    Estimates class-conditional densities:
        p(x | y=c) = (1 / N_c) * sum_{i: y_i=c} K_h(x - x_i)
    using an isotropic Gaussian kernel with bandwidth h:
        K_h(u) = (2*pi)^(-d/2) * h^(-d) * exp(-||u||^2 / (2 * h^2))

    Calculates posterior probabilities via Bayes' rule using numerically stable
    log-density computations and logsumexp to prevent underflow in high dimensions.

    Parameters
    ----------
    bandwidth : float, default=1.0
        Kernel smoothing parameter h > 0.
    weights : np.ndarray of shape (d,), optional
        Feature weights for weighted Euclidean distance.
    prior : Optional[np.ndarray], default=None
        Class prior probabilities [P(y=0), P(y=1)]. If None, estimated from class frequencies.
    """

    def __init__(
        self,
        bandwidth: float = 1.0,
        weights: Optional[np.ndarray] = None,
        prior: Optional[np.ndarray] = None,
    ):
        self.bandwidth = bandwidth
        self.weights = weights
        self.prior = prior
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.X_by_class_: Dict[int, np.ndarray] = {}
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianKDEClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if self.bandwidth <= 0:
            raise ValueError(f"Bandwidth must be positive, got {self.bandwidth}")

        self.n_features_ = X.shape[1]
        self.classes_ = np.array([0, 1], dtype=np.int32)

        # Apply feature weights if supplied
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape[0] != self.n_features_:
                raise ValueError("Weights dimension does not match feature dimension.")
            X_proc = X * np.sqrt(weights)
        else:
            X_proc = X

        # Partition training data by class
        for c in [0, 1]:
            mask = y == c
            if np.sum(mask) == 0:
                raise ValueError(f"Class {c} has 0 training instances.")
            self.X_by_class_[c] = X_proc[mask]

        # Priors
        if self.prior is not None:
            self.priors_ = np.asarray(self.prior, dtype=np.float64)
        else:
            n_total = len(y)
            self.priors_ = np.array(
                [len(self.X_by_class_[0]) / n_total, len(self.X_by_class_[1]) / n_total],
                dtype=np.float64,
            )

        return self

    def _compute_class_log_density(self, X_eval: np.ndarray, c: int) -> np.ndarray:
        """
        Compute log p(x | y=c) for evaluation points.

        log p(x | y=c) = -log(N_c) - (d/2)*log(2*pi) - d*log(h)
                         + logsumexp_{i in I_c} ( - ||x - x_i||^2 / (2 * h^2) )
        """
        X_train_c = self.X_by_class_[c]
        N_c = len(X_train_c)
        d = self.n_features_
        h = self.bandwidth

        # Pairwise squared Euclidean distances between X_eval and X_train_c
        # cdist squared: (n_eval, N_c)
        dist_sq = cdist(X_eval, X_train_c, metric="sqeuclidean")

        # Exponent arguments: - dist_sq / (2 * h^2)
        exponent_args = -dist_sq / (2.0 * (h**2))

        # Logsumexp over training samples i in class c: shape (n_eval,)
        lse = logsumexp(exponent_args, axis=1)

        # Full log-density
        log_norm_const = - (d / 2.0) * np.log(2.0 * np.pi) - d * np.log(h)
        log_density = -np.log(N_c) + log_norm_const + lse
        return log_density

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior probabilities log P(y=c | x).

        Returns
        -------
        log_proba : np.ndarray of shape (n_samples, 2)
        """
        X = np.asarray(X, dtype=np.float64)
        if self.weights is not None:
            X_eval = X * np.sqrt(self.weights)
        else:
            X_eval = X

        # Joint log-probabilities log p(x, y=c) = log p(x | y=c) + log P(y=c)
        log_p0 = self._compute_class_log_density(X_eval, c=0) + np.log(self.priors_[0])
        log_p1 = self._compute_class_log_density(X_eval, c=1) + np.log(self.priors_[1])

        joint_log = np.column_stack([log_p0, log_p1])  # (n_samples, 2)

        # Normalizing marginal log p(x) = logsumexp([log p(x, 0), log p(x, 1)])
        log_px = logsumexp(joint_log, axis=1, keepdims=True)

        return joint_log - log_px

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities [P(y=0|x), P(y=1|x)]."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels in {0, 1}."""
        log_prob = self.predict_log_proba(X)
        return (log_prob[:, 1] >= log_prob[:, 0]).astype(np.int32)


# =====================================================================
# C. Gaussian Generative Classifier (Regularized QDA)
# =====================================================================
class GaussianClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian Generative Classifier with covariance regularization.

    Estimates:
    - Class priors: pi_c = N_c / N
    - Class means: mu_c = (1 / N_c) * sum_{i: y_i=c} x_i
    - Regularized class covariance matrices:
        Sigma_c^{reg} = (1 - alpha) * S_c + alpha * (Tr(S_c) / d) * I_d + reg_jitter * I_d
      where S_c is the sample covariance matrix and alpha in [0, 1] controls shrinkage.

    Evaluates log posterior probabilities using Cholesky decomposition for numerical stability:
        Sigma_c = L_c L_c^T
        log |Sigma_c| = 2 * sum(log(diag(L_c)))
        (x - mu_c)^T Sigma_c^{-1} (x - mu_c) = || L_c^{-1} (x - mu_c)^T ||^2

    Parameters
    ----------
    alpha : float, default=0.1
        Covariance shrinkage regularization parameter in [0, 1].
    reg_jitter : float, default=1e-6
        Small positive constant added to diagonal to guarantee strict positive definiteness.
    """

    def __init__(self, alpha: float = 0.1, reg_jitter: float = 1e-6):
        self.alpha = alpha
        self.reg_jitter = reg_jitter
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Dict[int, np.ndarray] = {}
        self.covs_: Dict[int, np.ndarray] = {}
        self.chol_L_: Dict[int, np.ndarray] = {}
        self.log_det_: Dict[int, float] = {}
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        n_samples, d = X.shape
        self.n_features_ = d
        self.classes_ = np.array([0, 1], dtype=np.int32)

        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        if n0 == 0 or n1 == 0:
            raise ValueError("Both classes must be present in training data.")

        self.priors_ = np.array([n0 / n_samples, n1 / n_samples], dtype=np.float64)

        for c in [0, 1]:
            Xc = X[y == c]
            Nc = len(Xc)
            mu_c = np.mean(Xc, axis=0)
            self.means_[c] = mu_c

            # Unbiased sample covariance
            diff = Xc - mu_c
            if Nc > 1:
                S_c = np.dot(diff.T, diff) / (Nc - 1.0)
            else:
                S_c = np.zeros((d, d), dtype=np.float64)

            # Shrinkage towards isotropic covariance:
            # (1 - alpha) * S_c + alpha * (Tr(S_c) / d) * I_d
            tr_scaled = np.trace(S_c) / max(d, 1)
            target = np.eye(d, dtype=np.float64) * tr_scaled
            cov_reg = (1.0 - self.alpha) * S_c + self.alpha * target
            # Add numerical jitter to ensure well-conditioned positive definiteness
            cov_reg += np.eye(d, dtype=np.float64) * self.reg_jitter

            self.covs_[c] = cov_reg

            # Stable Cholesky factorization: Sigma = L L^T
            # Fallback to increasing jitter if slightly non-positive definite
            current_jitter = self.reg_jitter
            success = False
            for _ in range(5):
                try:
                    L = np.linalg.cholesky(cov_reg)
                    success = True
                    break
                except np.linalg.LinAlgError:
                    current_jitter *= 10.0
                    cov_reg += np.eye(d, dtype=np.float64) * current_jitter

            if not success:
                # Eigenvalue clipping fallback
                eigvals, eigvecs = np.linalg.eigh(cov_reg)
                eigvals = np.maximum(eigvals, 1e-6)
                cov_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(cov_reg)

            self.chol_L_[c] = L
            # log |Sigma| = 2 * sum(log(diag(L)))
            self.log_det_[c] = 2.0 * float(np.sum(np.log(np.diag(L))))

        return self

    def _compute_log_likelihood(self, X: np.ndarray, c: int) -> np.ndarray:
        """
        Compute class log-likelihood log p(x | y=c) using Cholesky factor L.
        """
        d = self.n_features_
        diff = X - self.means_[c]  # (N, d)
        L = self.chol_L_[c]        # (d, d)

        # Solve L * v = diff.T for v: v shape (d, N)
        # scipy/numpy forward solve for lower triangular matrix
        from scipy.linalg import solve_triangular
        v = solve_triangular(L, diff.T, lower=True, check_finite=False)
        # Mahalanobis distance squared: sum_m v_m^2
        mahalanobis_sq = np.sum(v**2, axis=0)  # (N,)

        log_norm = - (d / 2.0) * np.log(2.0 * np.pi) - 0.5 * self.log_det_[c]
        log_prob = log_norm - 0.5 * mahalanobis_sq
        return log_prob

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior probabilities log P(y=c | x).
        """
        X = np.asarray(X, dtype=np.float64)
        log_p0 = self._compute_log_likelihood(X, c=0) + np.log(self.priors_[0])
        log_p1 = self._compute_log_likelihood(X, c=1) + np.log(self.priors_[1])

        joint = np.column_stack([log_p0, log_p1])
        norm = logsumexp(joint, axis=1, keepdims=True)
        return joint - norm

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute posterior probabilities [P(y=0|x), P(y=1|x)]."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels in {0, 1}."""
        log_prob = self.predict_log_proba(X)
        return (log_prob[:, 1] >= log_prob[:, 0]).astype(np.int32)


# =====================================================================
# D. Logistic Regression Classifier
# =====================================================================
class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary Logistic Regression with L2 Regularization.

    Optimizes cross-entropy with L2 penalty:
        min_w  C * sum_i Loss(w^T x_i, y_i) + 0.5 * ||w||_2^2

    Parameters
    ----------
    C : float, default=1.0
        Inverse regularization strength (smaller values specify stronger regularization).
    max_iter : int, default=1000
        Maximum iterations for L-BFGS optimizer.
    random_state : Optional[int], default=None
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: Optional[int] = None):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = SklearnLogisticRegression(
            penalty="l2",
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict(X).astype(np.int32)

