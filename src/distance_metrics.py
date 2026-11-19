import numpy as np
from scipy.spatial.distance import cdist


def compute_distance_metrics(X_test, X_train, eps=1e-12):
    dists = cdist(X_test, X_train, metric="euclidean")
    d_near = np.min(dists, axis=1)
    d_far = np.max(dists, axis=1)

    contrast = (d_far - d_near) / np.maximum(d_near, eps)
    ratio = d_near / np.maximum(d_far, eps)

    return {
        "mean_d_near": float(np.mean(d_near)),
        "mean_d_far": float(np.mean(d_far)),
        "mean_contrast": float(np.mean(contrast)),
        "mean_ratio": float(np.mean(ratio)),
    }
