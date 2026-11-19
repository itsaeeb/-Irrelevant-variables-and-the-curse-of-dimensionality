"""
Experiment Utilities and Orchestration Module.

Provides:
- Hyperparameter tuning exclusively on validation data
- Multi-seed experiment execution harness
- Result logging, metrics aggregation, and table exports
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
from src.evaluation import (
    classification_error_rate,
    predictive_log_loss,
    evaluate_classifier_predictions,
)
from src.distance_metrics import compute_neighbour_distance_metrics


DEFAULT_HYPERPARAM_GRIDS = {
    "KNN": {
        "n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31, 51],
    },
    "GaussianKDE": {
        "bandwidth": [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0],
    },
    "Gaussian": {
        "alpha": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0],
    },
    "LogisticRegression": {
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    },
}


def tune_and_fit_classifier(
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    candidate_grid: Optional[Dict[str, List[Any]]] = None,
    random_state: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any]]:
    if candidate_grid is None:
        candidate_grid = DEFAULT_HYPERPARAM_GRIDS[classifier_name]

    param_name = list(candidate_grid.keys())[0]
    param_values = candidate_grid[param_name]

    best_score = float("inf")
    best_log_loss = float("inf")
    best_param_val = param_values[0]
    best_model = None

    for val in param_values:
        if classifier_name == "KNN":
            k_val = min(int(val), len(y_train))
            model = KNNClassifier(n_neighbors=k_val)
        elif classifier_name == "GaussianKDE":
            model = GaussianKDEClassifier(bandwidth=float(val))
        elif classifier_name == "Gaussian":
            model = GaussianClassifier(alpha=float(val))
        elif classifier_name == "LogisticRegression":
            model = LogisticRegressionClassifier(C=float(val), random_state=random_state)
        else:
            raise ValueError(f"Unknown classifier name: {classifier_name}")

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)

        val_err = classification_error_rate(y_val, y_val_pred)
        val_loss = predictive_log_loss(y_val, y_val_prob)

        is_better = False
        if val_err < best_score:
            is_better = True
        elif np.isclose(val_err, best_score, atol=1e-5) and val_loss < best_log_loss:
            is_better = True

        if is_better:
            best_score = val_err
            best_log_loss = val_loss
            best_param_val = val
            best_model = model

    if classifier_name == "KNN":
        best_model = KNNClassifier(n_neighbors=min(int(best_param_val), len(y_train)))
    elif classifier_name == "GaussianKDE":
        best_model = GaussianKDEClassifier(bandwidth=float(best_param_val))
    elif classifier_name == "Gaussian":
        best_model = GaussianClassifier(alpha=float(best_param_val))
    elif classifier_name == "LogisticRegression":
        best_model = LogisticRegressionClassifier(C=float(best_param_val), random_state=random_state)

    best_model.fit(X_train, y_train)
    return best_model, {param_name: best_param_val, "val_error": best_score, "val_loss": best_log_loss}


def run_single_classifier_experiment(
    classifier_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: Optional[int] = None,
    candidate_grid: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    model, best_params = tune_and_fit_classifier(
        classifier_name=classifier_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        candidate_grid=candidate_grid,
        random_state=random_state,
    )

    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)

    eval_results = evaluate_classifier_predictions(
        y_true=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
        X_test=X_test,
        X_train=X_train,
    )

    param_key = [k for k in best_params.keys() if k not in ["val_error", "val_loss"]][0]
    record = {
        "classifier": classifier_name,
        "error_rate": eval_results["error_rate"],
        "log_loss": eval_results["log_loss"],
        "mean_d_near": eval_results["mean_d_near"],
        "std_d_near": eval_results["std_d_near"],
        "mean_d_far": eval_results["mean_d_far"],
        "std_d_far": eval_results["std_d_far"],
        "mean_contrast": eval_results["mean_contrast"],
        "std_contrast": eval_results["std_contrast"],
        "mean_ratio": eval_results["mean_ratio"],
        "std_ratio": eval_results["std_ratio"],
        "best_param_name": param_key,
        "best_param_value": best_params[param_key],
        "val_error": best_params["val_error"],
        "val_loss": best_params["val_loss"],
    }
    return record


def aggregate_experiment_results(
    raw_df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    if metric_cols is None:
        metric_cols = [
            "error_rate",
            "log_loss",
            "mean_d_near",
            "mean_d_far",
            "mean_contrast",
            "mean_ratio",
        ]

    agg_dict = {}
    for col in metric_cols:
        if col in raw_df.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, "std")

    summary_df = raw_df.groupby(group_cols).agg(**agg_dict).reset_index()
    return summary_df


def save_experiment_outputs(
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    exp_name: str,
    base_dir: str = "results",
) -> Tuple[str, str, str]:
    csv_dir = os.path.join(base_dir, "csv")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    raw_csv_path = os.path.join(csv_dir, f"{exp_name}_raw.csv")
    summary_csv_path = os.path.join(csv_dir, f"{exp_name}_summary.csv")
    table_md_path = os.path.join(tables_dir, f"{exp_name}_table.md")

    raw_df.to_csv(raw_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    md_table = tabulate(summary_df.round(4), headers="keys", tablefmt="pipe", showindex=False)
    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Summary Table: {exp_name}\n\n")
        f.write(md_table)
        f.write("\n")

    return raw_csv_path, summary_csv_path, table_md_path
