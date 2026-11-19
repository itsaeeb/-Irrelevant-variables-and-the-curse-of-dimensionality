import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

COLORS = {
    "KNN": "#1f77b4",
    "GaussianKDE": "#ff7f0e",
    "Gaussian": "#2ca02c",
    "LogisticRegression": "#d62728",
}
MARKERS = {
    "KNN": "o",
    "GaussianKDE": "s",
    "Gaussian": "^",
    "LogisticRegression": "D",
}


def plot_curves(
    summary_df,
    x_col,
    y_mean,
    y_std,
    x_label,
    y_label,
    title,
    save_path,
    log_x=False,
    ylim=None,
    hline=None,
    hline_label=None,
):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(7.5, 4.8), dpi=200)

    for clf, group in summary_df.groupby("classifier"):
        g = group.sort_values(x_col)
        x = g[x_col].values
        mean = g[y_mean].values
        std = g[y_std].values
        c = COLORS.get(clf)
        plt.plot(x, mean, marker=MARKERS.get(clf, "o"), label=clf, color=c, linewidth=2)
        plt.fill_between(x, mean - std, mean + std, color=c, alpha=0.16)

    if hline is not None:
        plt.axhline(hline, color="black", linestyle="--", linewidth=1.2, label=hline_label)

    if log_x:
        plt.xscale("log")
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_distance_summary(summary_df, x_col, x_label, title, save_path, log_x=False):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=200)

    sub = summary_df[summary_df["classifier"] == summary_df["classifier"].iloc[0]].sort_values(x_col)
    x = sub[x_col].values

    panels = [
        ("mean_d_near", "Mean Nearest Distance ($D_{near}$)", "Nearest Distance", "#1f77b4", "o"),
        ("mean_d_far", "Mean Farthest Distance ($D_{far}$)", "Farthest Distance", "#ff7f0e", "s"),
        ("mean_contrast", "Contrast $(D_{far} - D_{near}) / D_{near}$", "Distance Contrast", "#2ca02c", "^"),
    ]
    for ax, (col, ylabel, ptitle, color, marker) in zip(axes, panels):
        m = sub[f"{col}_mean"].values
        s = sub[f"{col}_std"].values
        ax.plot(x, m, marker=marker, color=color, linewidth=2)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.18)
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.set_title(ptitle)

    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_scale_regimes(summary_df, metric_mean, metric_std, y_label, title, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    regimes = ["Noise-Dominated", "Equal-Scale", "Signal-Dominated"]
    settings = ["Raw", "Standardized", "Weighted", "Selected", "Oracle"]
    classifiers = ["KNN", "GaussianKDE", "Gaussian", "LogisticRegression"]
    palette = sns.color_palette("muted", len(settings))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=200, sharey=True)
    x = np.arange(len(classifiers))
    width = 0.8 / len(settings)

    for ax, regime in zip(axes, regimes):
        rdf = summary_df[summary_df["regime"] == regime]
        for i, setting in enumerate(settings):
            sdf = rdf[rdf["setting"] == setting].set_index("classifier")
            means = [sdf.loc[c, metric_mean] for c in classifiers]
            stds = [sdf.loc[c, metric_std] for c in classifiers]
            offset = (i - len(settings) / 2 + 0.5) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=stds,
                label=setting,
                color=palette[i],
                capsize=2.5,
                edgecolor="black",
                linewidth=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["KNN", "KDE", "Gaussian", "LogReg"])
        ax.set_title(regime)

    axes[0].set_ylabel(y_label)
    axes[-1].legend(title="Pipeline", loc="upper right", fontsize=8.5)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_paired_informative_dim(summary_df, y_mean, y_std, y_label, title, save_path, hline=None):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    modes = ["Fixed Ambient (d=50)", "No Noise (d=r)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=200, sharey=True)

    for ax, mode in zip(axes, modes):
        mdf = summary_df[summary_df["mode"] == mode]
        for clf, group in mdf.groupby("classifier"):
            g = group.sort_values("r")
            x = g["r"].values
            m = g[y_mean].values
            s = g[y_std].values
            c = COLORS.get(clf)
            ax.plot(x, m, marker=MARKERS.get(clf, "o"), label=clf, color=c, linewidth=2)
            ax.fill_between(x, m - s, m + s, color=c, alpha=0.16)
        if hline is not None:
            ax.axhline(hline, color="black", linestyle="--", linewidth=1.2, label="Bayes Optimal")
        ax.set_xlabel("Informative Dimension ($r$)")
        ax.set_title(mode)
        ax.legend(frameon=True, fontsize=8.5)

    axes[0].set_ylabel(y_label)
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
