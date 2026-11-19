import os
import sys
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation import generate_splits
from src.distance_metrics import compute_distance_metrics
from src.experiment_utils import CLASSIFIERS, evaluate_classifier, summarize_results, save_results
from src.plotting import plot_paired_informative_dim


def run_experiment_4(n_seeds=5, results_dir="results"):
    r_list = [1, 2, 4, 8, 16, 32]
    n_train, n_val, n_test = 500, 500, 1000
    # Total signal norm is fixed at 1.2 (delta = 1.2 / sqrt(r)) so Bayes error is
    # identical (~11.51%) for every r, isolating informative dimension fairly.
    signal_norm = 1.2
    bayes_err = float(norm.cdf(-signal_norm))

    modes = [
        ("Fixed Ambient (d=50)", lambda r: 50),
        ("No Noise (d=r)", lambda r: r),
    ]

    print(f"\n[Exp 4] Paired informative dimension r in {r_list} (constant SNR={signal_norm}, seeds={n_seeds})")
    rows = []

    for mode_name, get_d in modes:
        print(f"  mode = {mode_name}")
        for r in r_list:
            d = get_d(r)
            for s in range(n_seeds):
                seed = 42 + s
                X_tr, y_tr, X_va, y_va, X_te, y_te = generate_splits(
                    n_train, n_val, n_test, d=d, r=r, signal_norm=signal_norm, seed=seed
                )
                dist_metrics = compute_distance_metrics(X_te, X_tr)

                for clf in CLASSIFIERS:
                    res = evaluate_classifier(clf, X_tr, y_tr, X_va, y_va, X_te, y_te, dist_metrics)
                    res.update({"mode": mode_name, "r": r, "d": d, "seed": seed})
                    rows.append(res)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_results(raw_df, ["mode", "r", "classifier"])
    save_results(raw_df, summary_df, "experiment_4_informative_dimension", results_dir)

    plots_dir = os.path.join(results_dir, "plots")
    plot_paired_informative_dim(
        summary_df, "error_rate_mean", "error_rate_std",
        y_label="Classification Error Rate",
        title="Exp 4: Error Rate vs Informative Dimension (Constant Bayes Error)",
        save_path=os.path.join(plots_dir, "exp4_error_vs_informative_dim.png"),
        hline=bayes_err,
    )
    plot_paired_informative_dim(
        summary_df, "log_loss_mean", "log_loss_std",
        y_label="Predictive Log-Loss",
        title="Exp 4: Predictive Log-Loss vs Informative Dimension (Constant Bayes Error)",
        save_path=os.path.join(plots_dir, "exp4_log_loss_vs_informative_dim.png"),
    )
    return raw_df, summary_df


if __name__ == "__main__":
    run_experiment_4()
