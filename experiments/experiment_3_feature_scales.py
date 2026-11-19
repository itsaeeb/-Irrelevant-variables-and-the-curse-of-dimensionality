"""
Experiment 3: Unequal Feature Scales and Mitigation Strategies.

Investigates the impact of large irrelevant variance (noise sigma = 10 vs informative sigma = 1),
and compares 5 preprocessing / distance configurations:
A. Raw (unscaled features)
B. Standardized (StandardScaler fitted exclusively on training set)
C. Feature-Weighted (weights derived from training data ANOVA F-statistics)
D. Feature-Selected (SelectKBest fitted on training set without oracle knowledge)
E. Oracle (strictly uses only true informative features z in R^r)

Parameters:
- r = 2, d = 50 (48 noise features with sigma_u = 10.0, 2 informative with sigma_z = 1.0)
- n_train = 1000, n_val = 500, n_test = 1000
- 10 independent random seeds
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation import generate_experiment_splits
from src.preprocessing import (
    RawTransformer,
    StandardScalerWrapper,
    FeatureWeighter,
    FeatureSelectorWrapper,
    OracleSelector,
)
from src.experiment_utils import (
    run_single_classifier_experiment,
    tune_and_fit_classifier,
    aggregate_experiment_results,
    save_experiment_outputs,
)
from src.evaluation import (
    classification_error_rate,
    predictive_log_loss,
    evaluate_classifier_predictions,
)
from src.plotting import (
    plot_preprocessing_comparison,
)


def run_experiment_3(
    n_seeds: int = 10,
    d: int = 50,
    r: int = 2,
    noise_scale: float = 10.0,
    n_train: int = 1000,
    n_val: int = 500,
    n_test: int = 1000,
    base_seed: int = 42,
    results_dir: str = "results",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    classifiers = ["KNN", "GaussianKDE", "Gaussian", "LogisticRegression"]
    settings = ["Raw", "Standardized", "Weighted", "Selected", "Oracle"]
    raw_records = []

    print(f"\n{'='*70}")
    print(f"RUNNING EXPERIMENT 3: Unequal Feature Scales & Mitigation Strategies")
    print(f"Ambient d = {d}, Informative r = {r}")
    print(f"Informative sigma_z = 1.0, Noise sigma_u = {noise_scale}")
    print(f"Settings: {settings}")
    print(f"n_train = {n_train}, n_val = {n_val}, n_test = {n_test}, seeds = {n_seeds}")
    print(f"{'='*70}\n")

    for seed_idx in range(n_seeds):
        seed = base_seed + seed_idx
        print(f"--> Processing seed {seed_idx + 1}/{n_seeds} (seed={seed}) ...")

        splits = generate_experiment_splits(
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            d=d,
            r=r,
            signal_strength=1.0,
            informative_scale=1.0,
            noise_scale=noise_scale,
            seed=seed,
        )

        X_train_raw, y_train = splits["X_train"], splits["y_train"]
        X_val_raw, y_val = splits["X_val"], splits["y_val"]
        X_test_raw, y_test = splits["X_test"], splits["y_test"]

        # Build transformers for each setting (strictly fitted on training data)
        # 1. Raw
        raw_trans = RawTransformer().fit(X_train_raw, y_train)

        # 2. Standardized
        std_trans = StandardScalerWrapper().fit(X_train_raw, y_train)

        # 3. Weighted
        weight_trans = FeatureWeighter(method="f_classif").fit(X_train_raw, y_train)

        # 4. Oracle
        oracle_trans = OracleSelector(r=r).fit(X_train_raw, y_train)

        # Transform data splits for static transformers
        data_by_setting = {
            "Raw": (
                raw_trans.transform(X_train_raw),
                raw_trans.transform(X_val_raw),
                raw_trans.transform(X_test_raw),
            ),
            "Standardized": (
                std_trans.transform(X_train_raw),
                std_trans.transform(X_val_raw),
                std_trans.transform(X_test_raw),
            ),
            "Weighted": (
                weight_trans.transform(X_train_raw),
                weight_trans.transform(X_val_raw),
                weight_trans.transform(X_test_raw),
            ),
            "Oracle": (
                oracle_trans.transform(X_train_raw),
                oracle_trans.transform(X_val_raw),
                oracle_trans.transform(X_test_raw),
            ),
        }

        # For Feature Selection: tune k_features on validation set per classifier
        k_candidates = [1, 2, 3, 5, 10, 20, 50]

        for setting in settings:
            for clf_name in classifiers:
                if setting == "Selected":
                    # Tune k_features and classifier hyperparameter jointly on validation set
                    best_val_err = float("inf")
                    best_val_loss = float("inf")
                    best_sel_k = 2
                    best_k_model = None
                    best_k_params = {}
                    best_sel_transformer = None

                    for k_cand in k_candidates:
                        sel = FeatureSelectorWrapper(k_features=k_cand).fit(X_train_raw, y_train)
                        X_tr_k = sel.transform(X_train_raw)
                        X_va_k = sel.transform(X_val_raw)

                        cand_model, cand_params = tune_and_fit_classifier(
                            classifier_name=clf_name,
                            X_train=X_tr_k,
                            y_train=y_train,
                            X_val=X_va_k,
                            y_val=y_val,
                            random_state=seed,
                        )

                        cand_val_err = cand_params["val_error"]
                        cand_val_loss = cand_params["val_loss"]

                        if cand_val_err < best_val_err or (
                            np.isclose(cand_val_err, best_val_err, atol=1e-5)
                            and cand_val_loss < best_val_loss
                        ):
                            best_val_err = cand_val_err
                            best_val_loss = cand_val_loss
                            best_sel_k = k_cand
                            best_k_model = cand_model
                            best_k_params = cand_params
                            best_sel_transformer = sel

                    # Evaluate best selected model on test set
                    X_tr_best = best_sel_transformer.transform(X_train_raw)
                    X_te_best = best_sel_transformer.transform(X_test_raw)

                    y_test_pred = best_k_model.predict(X_te_best)
                    y_test_prob = best_k_model.predict_proba(X_te_best)

                    eval_res = evaluate_classifier_predictions(
                        y_true=y_test,
                        y_pred=y_test_pred,
                        y_prob=y_test_prob,
                        X_test=X_te_best,
                        X_train=X_tr_best,
                    )

                    param_key = [k for k in best_k_params.keys() if k not in ["val_error", "val_loss"]][0]
                    record = {
                        "setting": setting,
                        "classifier": clf_name,
                        "error_rate": eval_res["error_rate"],
                        "log_loss": eval_res["log_loss"],
                        "mean_d_near": eval_res["mean_d_near"],
                        "std_d_near": eval_res["std_d_near"],
                        "mean_d_far": eval_res["mean_d_far"],
                        "std_d_far": eval_res["std_d_far"],
                        "mean_contrast": eval_res["mean_contrast"],
                        "std_contrast": eval_res["std_contrast"],
                        "mean_ratio": eval_res["mean_ratio"],
                        "std_ratio": eval_res["std_ratio"],
                        "best_param_name": f"{param_key}_and_k_feat",
                        "best_param_value": f"{best_k_params[param_key]}_k{best_sel_k}",
                        "val_error": best_val_err,
                        "val_loss": best_val_loss,
                        "seed": seed,
                        "selected_k": best_sel_k,
                    }
                    raw_records.append(record)

                else:
                    X_tr, X_va, X_te = data_by_setting[setting]
                    res = run_single_classifier_experiment(
                        classifier_name=clf_name,
                        X_train=X_tr,
                        y_train=y_train,
                        X_val=X_va,
                        y_val=y_val,
                        X_test=X_te,
                        y_test=y_test,
                        random_state=seed,
                    )
                    res["setting"] = setting
                    res["seed"] = seed
                    raw_records.append(res)

    raw_df = pd.DataFrame(raw_records)
    summary_df = aggregate_experiment_results(
        raw_df=raw_df,
        group_cols=["setting", "classifier"],
    )

    # Save CSVs and Markdown table
    exp_name = "experiment_3_feature_scales"
    raw_path, summary_path, table_path = save_experiment_outputs(
        raw_df=raw_df,
        summary_df=summary_df,
        exp_name=exp_name,
        base_dir=results_dir,
    )
    print(f"Saved raw data: {raw_path}")
    print(f"Saved summary data: {summary_path}")
    print(f"Saved table: {table_path}")

    # Generate comparative plots
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Error rate comparison across settings
    plot_preprocessing_comparison(
        summary_df=summary_df,
        metric_mean="error_rate_mean",
        metric_std="error_rate_std",
        y_label="Classification Error Rate",
        title="Impact of Feature Scaling & Selection on Error Rate (Noise $\\sigma=10$)",
        save_path=os.path.join(plots_dir, "exp3_error_rate_comparison.png"),
    )

    # 2. Log-loss comparison across settings
    plot_preprocessing_comparison(
        summary_df=summary_df,
        metric_mean="log_loss_mean",
        metric_std="log_loss_std",
        y_label="Predictive Log-Loss",
        title="Impact of Feature Scaling & Selection on Log-Loss (Noise $\\sigma=10$)",
        save_path=os.path.join(plots_dir, "exp3_log_loss_comparison.png"),
    )

    # 3. Distance contrast comparison across settings
    plot_preprocessing_comparison(
        summary_df=summary_df,
        metric_mean="mean_contrast_mean",
        metric_std="mean_contrast_std",
        y_label="Distance Contrast $C = (D_{far}-D_{near})/D_{near}$",
        title="Distance Contrast Across Mitigation Strategies (Noise $\\sigma=10$)",
        save_path=os.path.join(plots_dir, "exp3_contrast_comparison.png"),
    )

    print("Experiment 3 complete! All plots and outputs saved successfully.\n")
    return raw_df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: Feature Scales")
    parser.add_argument("--seeds", type=int, default=10, help="Number of random seeds")
    args = parser.parse_args()
    run_experiment_3(n_seeds=args.seeds)

