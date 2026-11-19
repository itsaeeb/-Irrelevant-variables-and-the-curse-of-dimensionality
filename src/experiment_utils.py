import os
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.classifiers import (
    KNNClassifier,
    GaussianKDEClassifier,
    GaussianClassifier,
    LogisticRegressionClassifier,
)
from src.distance_metrics import compute_distance_metrics
from src.evaluation import error_rate, log_loss


CLASSIFIERS = ["KNN", "GaussianKDE", "Gaussian", "LogisticRegression"]

PARAM_GRIDS = {
    "KNN": [1, 3, 5, 7, 11, 15, 21, 31, 51],
    "GaussianKDE": [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0],
    "Gaussian": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0],
    "LogisticRegression": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
}


def make_classifier(name, param):
    if name == "KNN":
        return KNNClassifier(n_neighbors=int(param))
    if name == "GaussianKDE":
        return GaussianKDEClassifier(bandwidth=float(param))
    if name == "Gaussian":
        return GaussianClassifier(alpha=float(param))
    if name == "LogisticRegression":
        return LogisticRegressionClassifier(C=float(param))
    raise ValueError(f"Unknown classifier: {name}")


def tune_classifier(name, X_train, y_train, X_val, y_val):
    best_model = None
    best_param = None
    best_val_err = float("inf")
    best_val_loss = float("inf")

    for val in PARAM_GRIDS[name]:
        param = min(int(val), len(y_train)) if name == "KNN" else val
        model = make_classifier(name, param).fit(X_train, y_train)

        val_err = error_rate(y_val, model.predict(X_val))
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        # Tune probabilistic models by validation log-loss for calibrated probabilities;
        # tune KNN by validation error (tie-breaking on log-loss) since 0/1 neighbor
        # counts make raw KNN probabilities uncalibrated at small k.
        if name == "KNN":
            better = (val_err < best_val_err - 1e-9) or (
                abs(val_err - best_val_err) <= 1e-9 and val_loss < best_val_loss
            )
        else:
            better = val_loss < best_val_loss

        if better:
            best_model = model
            best_param = param
            best_val_err = val_err
            best_val_loss = val_loss

    return best_model, best_param, best_val_err, best_val_loss


def evaluate_classifier(name, X_train, y_train, X_val, y_val, X_test, y_test, dist_metrics=None):
    model, best_param, val_err, val_loss = tune_classifier(name, X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    if dist_metrics is None:
        dist_metrics = compute_distance_metrics(X_test, X_train)

    return {
        "classifier": name,
        "error_rate": error_rate(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob),
        "best_param": best_param,
        "val_error": val_err,
        "val_loss": val_loss,
        **dist_metrics,
    }


def summarize_results(raw_df, group_cols):
    metrics = ["error_rate", "log_loss", "mean_d_near", "mean_d_far", "mean_contrast", "mean_ratio"]
    agg = {}
    for col in metrics:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_std"] = (col, "std")
    return raw_df.groupby(group_cols).agg(**agg).reset_index()


def save_results(raw_df, summary_df, exp_name, results_dir="results"):
    csv_dir = os.path.join(results_dir, "csv")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    raw_df.to_csv(os.path.join(csv_dir, f"{exp_name}_raw.csv"), index=False)
    summary_df.to_csv(os.path.join(csv_dir, f"{exp_name}_summary.csv"), index=False)

    md = tabulate(summary_df.round(4), headers="keys", tablefmt="pipe", showindex=False)
    with open(os.path.join(tables_dir, f"{exp_name}_table.md"), "w", encoding="utf-8") as f:
        f.write(f"# {exp_name}\n\n{md}\n")
