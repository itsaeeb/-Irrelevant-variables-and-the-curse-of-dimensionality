"""
Experiment 2: Sample Size.

Investigates whether increasing the training sample size n can mitigate
the degradation caused by irrelevant ambient dimensions.

Parameters:
- r = 2 (fixed informative dimension)
- d = 50 (fixed ambient dimension: 2 informative + 48 noise features)
- n_train in [100, 250, 500, 1000, 2000, 5000]
- Fixed validation size n_val = 500, test size n_test = 1000
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


def run_experiment_2(
    n_train_list: Optional[List[int]] = None,
    n_seeds: int = 10,
    d: int = 50,
    r: int = 2,
    n_val: int = 500,
    n_test: int = 1000,
    base_seed: int = 42,
    results_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_train_list is None:
        n_train_list = [100, 250, 500, 1000, 2000, 5000]

    classifiers = ["KNN", "GaussianKDE", "Gaussian", "LogisticRegression"]
    raw_records = []

    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT 2: Sample Size")
    print(f"Ambient Dimension d = {d}, Informative r = {r}")
    print(f"Training Sample Sizes n_train in {n_train_list}")
    print(f"n_val = {n_val}, n_test = {n_test}, seeds = {n_seeds}")
    print(f"{'='*70}\n")

    for n_train in n_train_list:
        print(f"--> Processing training sample size n_train = {n_train} ...")
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
                res["n_train"] = n_train
                res["d"] = d
                res["r"] = r
                res["seed"] = seed
                raw_records.append(res)

    raw_df = pd.DataFrame(raw_records)
    summary_df = aggregate_experiment_results(
        raw_df=raw_df,
        group_cols=["n_train", "classifier"],
    )

    # Save CSVs and Markdown table
    exp_name = "experiment_2_sample_size"
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

    # 1. Error rate vs sample size
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="n_train",
        y_mean_col="error_rate_mean",
        y_std_col="error_rate_std",
        hue_col="classifier",
        x_label="Training Sample Size ($n$)",
        y_label="Classification Error Rate",
        title="Classification Error Rate vs Sample Size ($d=50, r=2$)",
        save_path=os.path.join(plots_dir, "exp2_error_vs_sample_size.png"),
        log_x=True,
    )

    # 2. Log-loss vs sample size
    plot_metric_vs_parameter(
        summary_df=summary_df,
        x_col="n_train",
        y_mean_col="log_loss_mean",
        y_std_col="log_loss_std",
        hue_col="classifier",
        x_label="Training Sample Size ($n$)",
        y_label="Predictive Log-Loss",
        title="Predictive Log-Loss vs Sample Size ($d=50, r=2$)",
        save_path=os.path.join(plots_dir, "exp2_log_loss_vs_sample_size.png"),
        log_x=True,
    )

    # 3. Distance concentration vs sample size
    plot_distance_concentration_summary(
        summary_df=summary_df,
        x_col="n_train",
        save_path=os.path.join(plots_dir, "exp2_distance_vs_sample_size.png"),
        title_prefix="Experiment 2",
    )

    print("Experiment 2 complete! All plots and outputs saved successfully.\n")
    return raw_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Sample Size")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    args = parser.parse_args()
    run_experiment_2(n_seeds=args.seeds)

