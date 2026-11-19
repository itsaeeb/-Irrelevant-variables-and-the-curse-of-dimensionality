"""
Experiment 1: Ambient Dimension.

Investigates how increasing irrelevant ambient dimension d affects:
- Classification error rate across 4 classifiers
- Predictive log-loss across 4 classifiers
- Nearest-neighbor distance D_near
- Farthest-neighbor distance D_far
- Distance contrast C = (D_far - D_near) / D_near

Parameters:
- r = 2 (fixed informative dimension)
- d in [2, 5, 10, 20, 50, 100]
- n_train = 1000, n_val = 500, n_test = 1000
- 10 independent random seeds
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation import generate_experiment_splits
from src.experiment_utils import (
    run_single_classifier_experiment,
    aggregate_experiment_results,
    save_experiment_outputs,
)
from src.plotting import (
    plot_metric_vs_parameter,
    plot_distance_concentration_summary,
)


def run_experiment_1(
    d_list: Optional[List[int]] = None,
    n_seeds: int = 10,
    n_train: int = 1000,
    n_val: int = 500,
    n_test: int = 1000,
    base_seed: int = 42,
    results_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if d_list is None:
        d_list = [2, 5, 10, 20, 50, 100]

    classifiers = ["KNN", "GaussianKDE", "Gaussian", "LogisticRegression"]
    r = 2
    raw_records = []

    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT 1: Ambient Dimension")
    print(f"Dimensions d in {d_list}, Informative r = {r}")
    print(f"n_train = {n_train}, n_val = {n_val}, n_test = {n_test}, seeds = {n_seeds}")
    print(f"{'='*70}\n")

    for d in d_list:
        print(f"--> Processing dimension d = {d} (noise features = {d - r}) ...")
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            splits = generate_experiment_splits(
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                d=d,
                r=r,
                signal_strength=1.0,
                informative_scale=1.0,
                noise_scale=1.0,
                seed=seed,
            )

            X_train, y_train = splits["X_train"], splits["y_train"]
            X_val, y_val = splits["X_val"], splits["y_val"]
            X_test, y_test = splits["X_test"], splits["y_test"]

            for clf_name in classifiers:
                res = run_single_classifier_experiment(
                    classifier_name=clf_name,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=seed,
                )
                res["d"] = d
                res["r"] = r
                res["n_train"] = n_train
                res["seed"] = seed
                raw_records.append(res)

    raw_df = pd.DataFrame(raw_records)
    summary_df = aggregate_experiment_results(
        raw_df=raw_df,
        group_cols=["d", "classifier"],
    )

    # Save CSVs and Markdown table
    exp_name = "experiment_1_ambient_dimension"
    raw_path, summary_path, table_path = save_experiment_outputs(
        raw_df=raw_df,
        summary_df=summary_df,
        exp_name=exp_name,
        base_dir=results_dir,
    )
    print(f"Saved raw data: {raw_path}")
    print(f"Saved summary data: {summary_path}")
    print(f"Saved table: {table_path}")

    # Generate plots
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Error rate vs ambient dimension
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="d",
        y_mean_col="error_rate_mean",
        y_std_col="error_rate_std",
        hue_col="classifier",
        x_label="Total Dimension ($d$)",
        y_label="Classification Error Rate",
        title="Classification Error Rate vs Ambient Dimension ($r=2$)",
        save_path=os.path.join(plots_dir, "exp1_error_vs_ambient_dim.png"),
        ylim=(0.0, 0.55),
    )

    # 2. Log-loss vs ambient dimension
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="d",
        y_mean_col="log_loss_mean",
        y_std_col="log_loss_std",
        hue_col="classifier",
        x_label="Total Dimension ($d$)",
        y_label="Predictive Log-Loss",
        title="Predictive Log-Loss vs Ambient Dimension ($r=2$)",
        save_path=os.path.join(plots_dir, "exp1_log_loss_vs_ambient_dim.png"),
    )

    # 3. Nearest neighbor distance vs d
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="d",
        y_mean_col="mean_d_near_mean",
        y_std_col="mean_d_near_std",
        hue_col="classifier",
        x_label="Total Dimension ($d$)",
        y_label="Mean Nearest Distance ($D_{near}$)",
        title="Mean Nearest Neighbor Distance vs Ambient Dimension",
        save_path=os.path.join(plots_dir, "exp1_nearest_dist_vs_ambient_dim.png"),
    )

    # 4. Farthest neighbor distance vs d
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="d",
        y_mean_col="mean_d_far_mean",
        y_std_col="mean_d_far_std",
        hue_col="classifier",
        x_label="Total Dimension ($d$)",
        y_label="Mean Farthest Distance ($D_{far}$)",
        title="Mean Farthest Neighbor Distance vs Ambient Dimension",
        save_path=os.path.join(plots_dir, "exp1_farthest_dist_vs_ambient_dim.png"),
    )

    # 5. Distance contrast vs d
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="d",
        y_mean_col="mean_contrast_mean",
        y_std_col="mean_contrast_std",
        hue_col="classifier",
        x_label="Total Dimension ($d$)",
        y_label="Distance Contrast $C = (D_{far} - D_{near}) / D_{near}$",
        title="Distance Contrast vs Ambient Dimension",
        save_path=os.path.join(plots_dir, "exp1_contrast_vs_ambient_dim.png"),
    )

    # 6. Combined distance concentration summary
    plot_distance_concentration_summary(
        summary_df=summary_df,
        x_col="d",
        save_path=os.path.join(plots_dir, "exp1_distance_concentration_summary.png"),
        title_prefix="Experiment 1",
    )

    print("Experiment 1 complete! All plots and outputs saved successfully.\n")
    return raw_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Ambient Dimension")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    args = parser.parse_args()
    run_experiment_1(n_seeds=args.seeds)
