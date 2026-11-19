"""
Synthetic Data Generation Module for Curse of Dimensionality Experiments.

Generates controlled synthetic binary classification datasets:
- Informative features: z in R^r drawn conditionally on y in {0, 1}.
- Irrelevant noise features: u in R^(d-r) drawn independently of z and y.
- Observed feature vector: X = [z, u] in R^d.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import os


def generate_synthetic_data(
    n_samples: int,
    d: int,
    r: int,
    signal_strength: float = 1.0,
    informative_scale: float = 1.0,
    noise_scale: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    balanced: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single synthetic binary classification dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    d : int
        Total ambient dimension (d >= r).
    r : int
        Number of informative features (1 <= r <= d).
    signal_strength : float, default=1.0
        Mean separation parameter delta.
        For class 0, mean = -delta; for class 1, mean = +delta.
        For baseline r=2, delta=1.0 yields:
        z | y=0 ~ N([-1, -1], I_2) and z | y=1 ~ N([1, 1], I_2).
    informative_scale : float, default=1.0
        Standard deviation of informative features (sigma_z).
    noise_scale : float or np.ndarray, default=1.0
        Standard deviation of irrelevant noise features (sigma_u).
        Can be a scalar or a vector of length (d - r).
    seed : int, optional
        Random seed for reproducibility. Used if rng is None.
    rng : np.random.Generator, optional
        NumPy random number generator instance.
    balanced : bool, default=True
        If True, ensures exactly balanced classes (n // 2 per class).

    Returns
    -------
    X : np.ndarray of shape (n_samples, d)
        Concatenated feature matrix [z, u].
    y : np.ndarray of shape (n_samples,)
        Binary class labels in {0, 1}.
    """
    if d < r:
        raise ValueError(f"Total dimension d ({d}) must be >= informative dimension r ({r}).")
    if r < 1:
        raise ValueError(f"Informative dimension r ({r}) must be >= 1.")

    if rng is None:
        rng = np.random.default_rng(seed)

    # Generate class labels P(y=0) = P(y=1) = 0.5
    if balanced:
        n_c0 = n_samples // 2
        n_c1 = n_samples - n_c0
        y = np.array([0] * n_c0 + [1] * n_c1, dtype=np.int32)
        rng.shuffle(y)
    else:
        y = rng.binomial(n=1, p=0.5, size=n_samples).astype(np.int32)

    # Generate informative features z in R^r
    # z_j | y=c ~ N((2c - 1) * delta, sigma_z^2)
    # y=0 -> -delta, y=1 -> +delta
    mean_shift = (2.0 * y - 1.0) * signal_strength  # shape (n_samples,)
    # Expand to (n_samples, r)
    z_means = np.outer(mean_shift, np.ones(r, dtype=np.float64))
    z = z_means + rng.normal(loc=0.0, scale=informative_scale, size=(n_samples, r))

    # Generate irrelevant noise features u in R^(d - r)
    n_noise = d - r
    if n_noise > 0:
        if isinstance(noise_scale, (int, float)):
            u = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, n_noise))
        else:
            noise_scale_arr = np.asarray(noise_scale, dtype=np.float64)
            if noise_scale_arr.shape != (n_noise,):
                raise ValueError(
                    f"noise_scale array shape {noise_scale_arr.shape} must match (d - r) = {n_noise}"
                )
            u = rng.normal(loc=0.0, scale=noise_scale_arr, size=(n_samples, n_noise))
        X = np.hstack([z, u])
    else:
        X = z

    return X, y


def generate_experiment_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    d: int,
    r: int,
    signal_strength: float = 1.0,
    informative_scale: float = 1.0,
    noise_scale: Union[float, np.ndarray] = 1.0,
    seed: Optional[int] = None,
    balanced: bool = True,
) -> Dict[str, Union[np.ndarray, int, float]]:
    """
    Generate independent training, validation, and test splits with strict isolation.

    Parameters
    ----------
    n_train : int
        Training set size.
    n_val : int
        Validation set size.
    n_test : int
        Test set size.
    d : int
        Total ambient dimension.
    r : int
        Informative dimension.
    signal_strength : float, default=1.0
        Mean separation parameter delta.
    informative_scale : float, default=1.0
        Standard deviation of informative features.
    noise_scale : float or np.ndarray, default=1.0
        Standard deviation of irrelevant features.
    seed : int, optional
        Master random seed. Independent child RNGs are spawned for each split.
    balanced : bool, default=True
        Whether to enforce balanced class proportions.

    Returns
    -------
    data_dict : Dict containing:
        - 'X_train', 'y_train'
        - 'X_val', 'y_val'
        - 'X_test', 'y_test'
        - 'd', 'r', 'n_train', 'n_val', 'n_test', 'seed'
        - 'informative_indices': np.ndarray of true informative feature indices
    """
    # Spawn independent generators using SeedSequence for rigorous statistical independence
    ss = np.random.SeedSequence(seed)
    train_ss, val_ss, test_ss = ss.spawn(3)

    rng_train = np.random.default_rng(train_ss)
    rng_val = np.random.default_rng(val_ss)
    rng_test = np.random.default_rng(test_ss)

    X_train, y_train = generate_synthetic_data(
        n_samples=n_train,
        d=d,
        r=r,
        signal_strength=signal_strength,
        informative_scale=informative_scale,
        noise_scale=noise_scale,
        rng=rng_train,
        balanced=balanced,
    )

    X_val, y_val = generate_synthetic_data(
        n_samples=n_val,
        d=d,
        r=r,
        signal_strength=signal_strength,
        informative_scale=informative_scale,
        noise_scale=noise_scale,
        rng=rng_val,
        balanced=balanced,
    )

    X_test, y_test = generate_synthetic_data(
        n_samples=n_test,
        d=d,
        r=r,
        signal_strength=signal_strength,
        informative_scale=informative_scale,
        noise_scale=noise_scale,
        rng=rng_test,
        balanced=balanced,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "d": d,
        "r": r,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "seed": seed if seed is not None else -1,
        "informative_indices": np.arange(r),
    }


def save_dataset(data: Dict[str, Union[np.ndarray, int, float]], filepath: str) -> None:
    """
    Save generated dataset dictionary to a compressed .npz archive.

    Parameters
    ----------
    data : Dict
        Dataset dictionary from generate_experiment_splits.
    filepath : str
        Target destination path (e.g., 'data/dataset_d10_seed42.npz').
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    save_kwargs = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            save_kwargs[k] = v
        else:
            save_kwargs[k] = np.array(v)
    np.savez_compressed(filepath, **save_kwargs)


def load_dataset(filepath: str) -> Dict[str, Union[np.ndarray, int, float]]:
    """
    Load dataset from a compressed .npz archive.

    Parameters
    ----------
    filepath : str
        Path to the saved .npz archive.

    Returns
    -------
    data : Dict
        Dataset dictionary matching generate_experiment_splits output.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")

    with np.load(filepath) as npz:
        loaded = {}
        for k in npz.files:
            val = npz[k]
            # Convert scalar arrays back to standard Python types
            if val.ndim == 0:
                loaded[k] = val.item()
            else:
                loaded[k] = val
    return loaded

