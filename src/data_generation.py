import numpy as np


def generate_data(
    n_samples,
    d,
    r,
    signal_norm=1.2,
    informative_scale=1.0,
    noise_scale=1.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    # Balanced binary labels: P(y=0) = P(y=1) = 0.5
    n0 = n_samples // 2
    y = np.array([0] * n0 + [1] * (n_samples - n0), dtype=int)
    rng.shuffle(y)

    # Scale per-coordinate shift by 1/sqrt(r) so total SNR (and Bayes error) stays
    # constant across informative dimension r and across rescaling of z
    delta = (signal_norm / np.sqrt(r)) * informative_scale
    sign = (2 * y - 1)[:, None]
    z = sign * delta + rng.normal(0.0, informative_scale, size=(n_samples, r))

    if d > r:
        u = rng.normal(0.0, noise_scale, size=(n_samples, d - r))
        X = np.hstack([z, u])
    else:
        X = z

    return X, y


def generate_splits(
    n_train,
    n_val,
    n_test,
    d,
    r,
    signal_norm=1.2,
    informative_scale=1.0,
    noise_scale=1.0,
    seed=42,
):
    # Separate RNGs so train, val, and test are independent and reproducible
    rng_train = np.random.default_rng(seed)
    rng_val = np.random.default_rng(seed + 10_000)
    rng_test = np.random.default_rng(seed + 20_000)

    X_train, y_train = generate_data(
        n_train, d, r, signal_norm, informative_scale, noise_scale, rng_train
    )
    X_val, y_val = generate_data(
        n_val, d, r, signal_norm, informative_scale, noise_scale, rng_val
    )
    X_test, y_test = generate_data(
        n_test, d, r, signal_norm, informative_scale, noise_scale, rng_test
    )

    return X_train, y_train, X_val, y_val, X_test, y_test
