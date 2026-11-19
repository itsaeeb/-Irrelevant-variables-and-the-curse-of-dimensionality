import numpy as np


def error_rate(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) != np.asarray(y_pred)))


def log_loss(y_true, y_prob, eps=1e-15):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    p1 = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
    p1 = np.clip(p1, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p1) + (1.0 - y_true) * np.log(1.0 - p1)))
