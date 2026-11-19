"""
Publication-Quality Plotting and Visualization Utilities.

Generates plots with:
- Error bars and standard deviation bands
- Distinct, consistent color palettes for classifiers
- Clear labels, legends, and gridlines
- Matplotlib and Seaborn aesthetic configurations
"""

from typing import Dict, List, Optional
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set consistent style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)

CLASSIFIER_COLORS = {
    "KNN": "#1f77b4",                # Steel Blue
    "GaussianKDE": "#ff7f0e",        # Safety Orange
    "Gaussian": "#2ca02c",           # Forest Green
    "LogisticRegression": "#d62728", # Crimson Red
}

CLASSIFIER_MARKERS = {
    "KNN": "o",
    "GaussianKDE": "s",
    "Gaussian": "^",
    "LogisticRegression": "D",
}


def plot_metric_vs_parameter(
    summary_df: pd.DataFrame,
    x_col: str,
    y_mean_col: str,
    y_std_col: str,
    hue_col: str,
    x_label: str,
    y_label: str,
    title: str,
    save_path: str,
    log_x: bool = False,
    log_y: bool = False,
    ylim: Optional[tuple] = None,
) -> None:
    """
    Plot mean curves with shaded standard deviation bands for multiple classifiers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(8, 5), dpi=300)

    groups = summary_df.groupby(hue_col)
    for name, group in groups:
        sorted_group = group.sort_values(by=x_col)
        x_vals = sorted_group[x_col].values
        y_mean = sorted_group[y_mean_col].values
        y_std = sorted_group[y_std_col].values

        color = CLASSIFIER_COLORS.get(name, None)
        marker = CLASSIFIER_MARKERS.get(name, "o")

        plt.plot(
            x_vals,
            y_mean,
            marker=marker,
            label=name,
            color=color,
            linewidth=2.2,
            alpha=0.9,
        )
        plt.fill_between(
            x_vals,
            y_mean - y_std,
            y_mean + y_std,
            color=color,
            alpha=0.18,
        )

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(x_label, fontweight="bold")
    plt.ylabel(y_label, fontweight="bold")
    plt.title(title, fontweight="bold", pad=12)
    plt.legend(frameon=True, loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_distance_concentration_summary(
    summary_df: pd.DataFrame,
    x_col: str,
    save_path: str,
    title_prefix: str = "Experiment 1",
) -> None:
    """
    Plot 3-panel figure showing Nearest Distance, Farthest Distance, and Distance Contrast.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    first_clf = summary_df["classifier"].iloc[0]
    sub_df = summary_df[summary_df["classifier"] == first_clf].sort_values(by=x_col)
    x = sub_df[x_col].values

    # Panel 1: D_near
    axes[0].plot(x, sub_df["mean_d_near_mean"], marker="o", color="#1f77b4", linewidth=2.2)
    axes[0].fill_between(
        x,
        sub_df["mean_d_near_mean"] - sub_df["mean_d_near_std"],
        sub_df["mean_d_near_mean"] + sub_df["mean_d_near_std"],
        color="#1f77b4",
        alpha=0.2,
    )
    axes[0].set_xlabel(f"{x_col}", fontweight="bold")
    axes[0].set_ylabel("Mean Nearest Distance ($D_{near}$)", fontweight="bold")
    axes[0].set_title("Nearest Neighbor Distance", fontweight="bold")

    # Panel 2: D_far
    axes[1].plot(x, sub_df["mean_d_far_mean"], marker="s", color="#ff7f0e", linewidth=2.2)
    axes[1].fill_between(
        x,
        sub_df["mean_d_far_mean"] - sub_df["mean_d_far_std"],
        sub_df["mean_d_far_mean"] + sub_df["mean_d_far_std"],
        color="#ff7f0e",
        alpha=0.2,
    )
    axes[1].set_xlabel(f"{x_col}", fontweight="bold")
    axes[1].set_ylabel("Mean Farthest Distance ($D_{far}$)", fontweight="bold")
    axes[1].set_title("Farthest Neighbor Distance", fontweight="bold")

    # Panel 3: Contrast C = (D_far - D_near) / D_near
    axes[2].plot(x, sub_df["mean_contrast_mean"], marker="^", color="#2ca02c", linewidth=2.2)
    axes[2].fill_between(
        x,
        sub_df["mean_contrast_mean"] - sub_df["mean_contrast_std"],
        sub_df["mean_contrast_mean"] + sub_df["mean_contrast_std"],
        color="#2ca02c",
        alpha=0.2,
    )
    axes[2].set_xlabel(f"{x_col}", fontweight="bold")
    axes[2].set_ylabel("Distance Contrast $C = (D_{far}-D_{near})/D_{near}$", fontweight="bold")
    axes[2].set_title("Distance Contrast Degradation", fontweight="bold")

    fig.suptitle(f"{title_prefix}: High-Dimensional Distance Concentration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_preprocessing_comparison(
    summary_df: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    y_label: str,
    title: str,
    save_path: str,
) -> None:
    """
    Grouped bar plot comparing preprocessing pipelines (Raw, Standardized, Weighted, Selected, Oracle).
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(10, 6), dpi=300)

    # Order of settings
    settings_order = ["Raw", "Standardized", "Weighted", "Selected", "Oracle"]
    available_settings = [s for s in settings_order if s in summary_df["setting"].unique()]

    classifiers = sorted(summary_df["classifier"].unique())
    n_clfs = len(classifiers)
    n_settings = len(available_settings)

    x = np.arange(n_clfs)
    width = 0.8 / n_settings

    palette = sns.color_palette("muted", n_settings)

    for idx, setting in enumerate(available_settings):
        sub_df = summary_df[summary_df["setting"] == setting].set_index("classifier")
        means = [sub_df.loc[c, metric_mean] if c in sub_df.index else 0.0 for c in classifiers]
        stds = [sub_df.loc[c, metric_std] if c in sub_df.index else 0.0 for c in classifiers]

        offset = (idx - n_settings / 2.0 + 0.5) * width
        plt.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            label=setting,
            color=palette[idx],
            capsize=3,
            alpha=0.88,
            edgecolor="black",
            linewidth=0.8,
        )

    plt.xticks(x, classifiers, fontweight="bold")
    plt.ylabel(y_label, fontweight="bold")
    plt.title(title, fontweight="bold", pad=12)
    plt.legend(title="Pipeline Setting", frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

