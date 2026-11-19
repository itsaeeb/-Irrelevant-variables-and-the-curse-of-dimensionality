import os
import sys
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation import generate_splits
from src.distance_metrics import compute_distance_metrics
from src.experiment_utils import CLASSIFIERS, evaluate_classifier, summarize_results, save_results
from src.plotting import plot_curves, plot_distance_summary


def run_experiment_1(n_seeds=5, results_dir="results"):
    d_list = [2, 5, 10, 20, 50, 100, 200]
    r = 2
    n_train, n_val, n_test = 500, 500, 1000
    signal_norm = 1.2
    bayes_err = float(norm.cdf(-signal_norm))

    print(f"\n[Exp 1] Ambient dimension d in {d_list} (r={r}, n={n_train}, seeds={n_seeds})")
    rows = []

    for d in d_list:
        print(f"  d = {d}")
        for s in range(n_seeds):
            seed = 42 + s
            X_tr, y_tr, X_va, y_va, X_te, y_te = generate_splits(
                n_train, n_val, n_test, d=d, r=r, signal_norm=signal_norm, seed=seed
            )
            dist_metrics = compute_distance_metrics(X_te, X_tr)

            for clf in CLASSIFIERS:
                res = evaluate_classifier(clf, X_tr, y_tr, X_va, y_va, X_te, y_te, dist_metrics)
                res.update({"d": d, "r": r, "n_train": n_train, "seed": seed})
                rows.append(res)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_results(raw_df, ["d", "classifier"])
    save_results(raw_df, summary_df, "experiment_1_ambient_dimension", results_dir)

    plots_dir = os.path.join(results_dir, "plots")
    plot_curves(
        summary_df, "d", "error_rate_mean", "error_rate_std",
        x_label="Ambient Dimension ($d$)",
        y_label="Classification Error Rate",
        title="Exp 1: Error Rate vs Ambient Dimension ($r=2, n=500$)",
        save_path=os.path.join(plots_dir, "exp1_error_vs_ambient_dim.png"),
        hline=bayes_err, hline_label=f"Bayes Optimal ({bayes_err:.3f})",
    )
    plot_curves(
        summary_df, "d", "log_loss_mean", "log_loss_std",
        x_label="Ambient Dimension ($d$)",
        y_label="Predictive Log-Loss",
        title="Exp 1: Predictive Log-Loss vs Ambient Dimension ($r=2, n=500$)",
        save_path=os.path.join(plots_dir, "exp1_log_loss_vs_ambient_dim.png"),
    )
    plot_distance_summary(
        summary_df, "d",
        x_label="Ambient Dimension ($d$)",
        title="Exp 1: Distance Concentration vs Ambient Dimension",
        save_path=os.path.join(plots_dir, "exp1_distance_concentration.png"),
    )
    return raw_df, summary_df


if __name__ == "__main__":
    run_experiment_1()
