"""
Experiment 4: Informative Dimension.

Investigates the effect of increasing the number of informative features r
while keeping the ambient dimension d = 50 and sample size n = 1000 fixed.

Each added informative coordinate contributes the exact same signal strength:
    z_j | y=c ~ N((2c - 1) * delta, 1) with delta = 1.0.
Remaining (d - r) features are independent N(0, 1) noise.

Parameters:
- r in [1, 2, 4, 8]
- d = 50 (fixed total ambient dimension)
- delta = 1.0 (fixed signal separation per coordinate)
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


def run_experiment_4(
    r_list: Optional[List[int]] = None,
    n_seeds: int = 10,
    d: int = 50,
    delta: float = 1.0,
    n_train: int = 1000,
    n_val: int = 500,
    n_test: int = 1000,
    base_seed: int = 42,
    results_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if r_list is None:
        r_list = [1, 2, 4, 8]

    classifiers = ["KNN", "GaussianKDE", "Gaussian", "LogisticRegression"]
    raw_records = []

    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT 4: Informative Dimension")
    print(f"Fixed Total Dimension d = {d}, Signal Strength delta = {delta}")
    print(f"Informative Dimensions r in {r_list}")
    print(f"n_train = {n_train}, n_val = {n_val}, n_test = {n_test}, seeds = {n_seeds}")
    print(f"{'='*70}\n")

    for r in r_list:
        print(f"--> Processing informative dimension r = {r} (noise = {d - r}) ...")
        for seed_idx in range(n_seeds):
            seed = base_seed + seed_idx
            splits = generate_experiment_splits(
                n_train=n_train,
                n_val=n_val,
                n_test=n_test,
                d=d,
                r=r,
                signal_strength=delta,
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
                res["r"] = r
                res["d"] = d
                res["delta"] = delta
                res["seed"] = seed
                raw_records.append(res)

    raw_df = pd.DataFrame(raw_records)
    summary_df = aggregate_experiment_results(
        raw_df=raw_df,
        group_cols=["r", "classifier"],
    )

    # Save CSVs and Markdown table
    exp_name = "experiment_4_informative_dimension"
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

    # 1. Error rate vs informative dimension r
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="r",
        y_mean_col="error_rate_mean",
        y_std_col="error_rate_std",
        hue_col="classifier",
        x_label="Informative Dimension ($r$)",
        y_label="Classification Error Rate",
        title="Classification Error vs Informative Dimension ($d=50, \\delta=1.0$)",
        save_path=os.path.join(plots_dir, "exp4_error_vs_informative_dim.png"),
        ylim=(0.0, 0.40),
    )

    # 2. Log-loss vs informative dimension r
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="r",
        y_mean_col="log_loss_mean",
        y_std_col="log_loss_std",
        hue_col="classifier",
        x_label="Informative Dimension ($r$)",
        y_label="Predictive Log-Loss",
        title="Predictive Log-Loss vs Informative Dimension ($d=50, \\delta=1.0$)",
        save_path=os.path.join(plots_dir, "exp4_log_loss_vs_informative_dim.png"),
    )

    # 3. Distance concentration vs informative dimension r
    plot_distance_concentration_summary(
        summary_df=summary_df,
        x_col="r",
        save_path=os.path.join(plots_dir, "exp4_distance_vs_informative_dim.png"),
        title_prefix="Experiment 4",
    )

    print("Experiment 4 complete! All plots and outputs saved successfully.\n")
    return raw_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: Informative Dimension")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    args = parser.parse_args()
    run_experiment_4(n_seeds=args.seeds)

