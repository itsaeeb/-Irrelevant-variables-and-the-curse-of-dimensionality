import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generation import generate_splits
from src.distance_metrics import compute_distance_metrics
from src.evaluation import error_rate, log_loss
from src.experiment_utils import (
    CLASSIFIERS,
    tune_classifier,
    evaluate_classifier,
    summarize_results,
    save_results,
)
from src.plotting import plot_scale_regimes
from src.preprocessing import (
    RawTransformer,
    Standardizer,
    FeatureWeighter,
    FeatureSelector,
    OracleSelector,
)


def _run_feature_selection(clf, X_tr, y_tr, X_va, y_va, X_te, y_te, k_grid):
    best_model, best_sel, best_k, best_param = None, None, None, None
    best_val_err, best_val_loss = float("inf"), float("inf")

    for k in k_grid:
        sel = FeatureSelector(k_features=k).fit(X_tr, y_tr)
        model, param, val_err, val_loss = tune_classifier(
            clf, sel.transform(X_tr), y_tr, sel.transform(X_va), y_va
        )
        if clf == "KNN":
            better = (val_err < best_val_err - 1e-9) or (
                abs(val_err - best_val_err) <= 1e-9 and val_loss < best_val_loss
            )
        else:
            better = val_loss < best_val_loss

        if better:
            best_model, best_sel, best_k, best_param = model, sel, k, param
            best_val_err, best_val_loss = val_err, val_loss

    X_tr_sel = best_sel.transform(X_tr)
    X_te_sel = best_sel.transform(X_te)
    dist_metrics = compute_distance_metrics(X_te_sel, X_tr_sel)

    return {
        "classifier": clf,
        "error_rate": error_rate(y_te, best_model.predict(X_te_sel)),
        "log_loss": log_loss(y_te, best_model.predict_proba(X_te_sel)),
        "best_param": f"{best_param}_k{best_k}",
        "val_error": best_val_err,
        "val_loss": best_val_loss,
        **dist_metrics,
    }


def run_experiment_3(n_seeds=5, results_dir="results"):
    d, r = 50, 2
    n_train, n_val, n_test = 500, 500, 1000
    signal_norm = 1.2
    k_grid = [1, 2, 3, 5, 10, 20, 50]

    # Three regimes that show when standardization helps, fails to remove noise, or hurts
    regimes = [
        ("Noise-Dominated", 1.0, 10.0),
        ("Equal-Scale", 1.0, 1.0),
        ("Signal-Dominated", 10.0, 1.0),
    ]

    print(f"\n[Exp 3] Unequal feature scales across 3 regimes (d={d}, r={r}, n={n_train}, seeds={n_seeds})")
    rows = []

    for regime_name, sig_z, sig_u in regimes:
        print(f"  regime = {regime_name} (sigma_z={sig_z}, sigma_u={sig_u})")
        for s in range(n_seeds):
            seed = 42 + s
            X_tr, y_tr, X_va, y_va, X_te, y_te = generate_splits(
                n_train, n_val, n_test, d=d, r=r,
                signal_norm=signal_norm,
                informative_scale=sig_z,
                noise_scale=sig_u,
                seed=seed,
            )

            pipelines = {
                "Raw": RawTransformer().fit(X_tr, y_tr),
                "Standardized": Standardizer().fit(X_tr, y_tr),
                "Weighted": FeatureWeighter().fit(X_tr, y_tr),
                "Oracle": OracleSelector(r=r).fit(X_tr, y_tr),
            }

            for setting, trans in pipelines.items():
                Xt_tr = trans.transform(X_tr)
                Xt_va = trans.transform(X_va)
                Xt_te = trans.transform(X_te)
                dist_metrics = compute_distance_metrics(Xt_te, Xt_tr)

                for clf in CLASSIFIERS:
                    res = evaluate_classifier(clf, Xt_tr, y_tr, Xt_va, y_va, Xt_te, y_te, dist_metrics)
                    res.update({"regime": regime_name, "setting": setting, "seed": seed})
                    rows.append(res)

            for clf in CLASSIFIERS:
                res = _run_feature_selection(clf, X_tr, y_tr, X_va, y_va, X_te, y_te, k_grid)
                res.update({"regime": regime_name, "setting": "Selected", "seed": seed})
                rows.append(res)

    raw_df = pd.DataFrame(rows)
    summary_df = summarize_results(raw_df, ["regime", "setting", "classifier"])
    save_results(raw_df, summary_df, "experiment_3_feature_scales", results_dir)

    plots_dir = os.path.join(results_dir, "plots")
    plot_scale_regimes(
        summary_df, "error_rate_mean", "error_rate_std",
        y_label="Classification Error Rate",
        title="Exp 3: Error Rate Across Feature-Scale Regimes ($d=50, r=2$)",
        save_path=os.path.join(plots_dir, "exp3_error_rate_comparison.png"),
    )
    plot_scale_regimes(
        summary_df, "log_loss_mean", "log_loss_std",
        y_label="Predictive Log-Loss",
        title="Exp 3: Log-Loss Across Feature-Scale Regimes ($d=50, r=2$)",
        save_path=os.path.join(plots_dir, "exp3_log_loss_comparison.png"),
    )
    plot_scale_regimes(
        summary_df, "mean_contrast_mean", "mean_contrast_std",
        y_label="Distance Contrast $C = (D_{far}-D_{near})/D_{near}$",
        title="Exp 3: Distance Contrast Across Feature-Scale Regimes ($d=50, r=2$)",
        save_path=os.path.join(plots_dir, "exp3_contrast_comparison.png"),
    )
    return raw_df, summary_df


if __name__ == "__main__":
    run_experiment_3()
