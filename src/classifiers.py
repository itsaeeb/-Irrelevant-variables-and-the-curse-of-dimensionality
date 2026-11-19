import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression


class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        k = min(self.n_neighbors, len(self.y_train))
        dists = cdist(X, self.X_train, metric="euclidean")
        nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
        p1 = np.mean(self.y_train[nn_idx] == 1, axis=1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class GaussianKDEClassifier:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.d = X.shape[1]
        self.X_class = [X[y == 0], X[y == 1]]
        self.priors = [len(self.X_class[0]) / len(y), len(self.X_class[1]) / len(y)]
        return self

    def _class_log_density(self, X, c):
        Xc = self.X_class[c]
        h = self.bandwidth
        dist_sq = cdist(X, Xc, metric="sqeuclidean")
        # Compute in log-space to avoid floating-point underflow in high d
        log_kernel_sum = logsumexp(-dist_sq / (2.0 * h * h), axis=1)
        log_norm = -np.log(len(Xc)) - 0.5 * self.d * np.log(2.0 * np.pi) - self.d * np.log(h)
        return log_norm + log_kernel_sum

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_p0 = self._class_log_density(X, 0) + np.log(self.priors[0])
        log_p1 = self._class_log_density(X, 1) + np.log(self.priors[1])
        joint = np.column_stack([log_p0, log_p1])
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        lp = self.predict_log_proba(X)
        return (lp[:, 1] >= lp[:, 0]).astype(int)


class GaussianClassifier:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        d = X.shape[1]
        self.priors = [np.mean(y == 0), np.mean(y == 1)]
        self.means = []
        self.covs = []

        for c in (0, 1):
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            S = np.cov(Xc, rowvar=False) if d > 1 else np.array([[np.var(Xc, ddof=1)]])
            # Shrink sample covariance toward an isotropic diagonal matrix
            avg_var = np.trace(S) / d
            cov = (1.0 - self.alpha) * S + self.alpha * avg_var * np.eye(d) + 1e-6 * np.eye(d)
            self.means.append(mu)
            self.covs.append(cov)
        return self

    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_p0 = multivariate_normal.logpdf(X, mean=self.means[0], cov=self.covs[0]) + np.log(self.priors[0])
        log_p1 = multivariate_normal.logpdf(X, mean=self.means[1], cov=self.covs[1]) + np.log(self.priors[1])
        joint = np.column_stack([log_p0, log_p1])
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        lp = self.predict_log_proba(X)
        return (lp[:, 1] >= lp[:, 0]).astype(int)


class LogisticRegressionClassifier:
    def __init__(self, C=1.0, max_iter=2000):
        self.C = C
        self.model = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X).astype(int)
