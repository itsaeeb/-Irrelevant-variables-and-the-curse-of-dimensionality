import os
import sys
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation import generate_splits
from src.distance_metrics import compute_distance_metrics
from src.experiment_utils import CLASSIFIERS, evaluate_classifier, summarize_results, save_results
from src.plotting import plot_curves, plot_distance_summary


def run_experiment_2(n_seeds=5, results_dir="results"):
    n_list = [100, 250, 500, 1000, 2500, 5000]
    d, r = 50, 2
    n_val, n_test = 500, 1000
    signal_norm = 1.2
    bayes_err = float(norm.cdf(-signal_norm))

    print(f"\n[Exp 2] Sample size n in {n_list} (d={d}, r={r}, seeds={n_seeds})")
    rows = []

    for n_train in n_list:
        print(f"  n_train = {n_train}")
        for s in range(n_seeds):
            seed = 42 + s
            X_tr, y_tr, X_va, y_va, X_te, y_te = generate_splits(
                n_train, n_val, n_test, d=d, r=r, signal_norm=signal_norm, seed=seed
            )
            dist_metrics = compute_distance_metrics(X_te, X_tr)

            for clf in CLASSIFIERS:
                res = evaluate_classifier(clf, X_tr, y_tr, X_va, y_va, X_te, y_te, dist_metrics)
                res.update({"n_train": n_train, "d": d, "r": r, "seed": seed})
                rows.append(res)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_results(raw_df, ["n_train", "classifier"])
    save_results(raw_df, summary_df, "experiment_2_sample_size", results_dir)

    plots_dir = os.path.join(results_dir, "plots")
    plot_curves(
        summary_df, "n_train", "error_rate_mean", "error_rate_std",
        x_label="Training Sample Size ($n$)",
        y_label="Classification Error Rate",
        title="Exp 2: Error Rate vs Sample Size ($d=50, r=2$)",
        save_path=os.path.join(plots_dir, "exp2_error_vs_sample_size.png"),
        log_x=True,
        hline=bayes_err, hline_label=f"Bayes Optimal ({bayes_err:.3f})",
    )
    plot_curves(
        summary_df, "n_train", "log_loss_mean", "log_loss_std",
        x_label="Training Sample Size ($n$)",
        y_label="Predictive Log-Loss",
        title="Exp 2: Predictive Log-Loss vs Sample Size ($d=50, r=2$)",
        save_path=os.path.join(plots_dir, "exp2_log_loss_vs_sample_size.png"),
        log_x=True,
    )
    plot_distance_summary(
        summary_df, "n_train",
        x_label="Training Sample Size ($n$)",
        title="Exp 2: Distance Concentration vs Sample Size",
        save_path=os.path.join(plots_dir, "exp2_distance_vs_sample_size.png"),
        log_x=True,
    )
    return raw_df, summary_df


if __name__ == "__main__":
    run_experiment_2()
