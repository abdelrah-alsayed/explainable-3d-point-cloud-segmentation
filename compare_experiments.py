"""
Comparison plots between two filtered-part experiments for a single category.
Usage examples:
  python compare_experiments.py --category Table --exp_a parts2_cap312 --exp_b parts3_cap626
  python compare_experiments.py --category Chair --exp_a parts3_cap626 --exp_b parts4_cap596
"""
import argparse
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── palette (from dataviz skill validated palette) ──────────────────────────
C = {
    "blue":    "#2a78d6",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "surface": "#fcfcfb",
    "page":    "#f9f9f7",
    "text":    "#0b0b0b",
    "text2":   "#52514e",
    "muted":   "#898781",
    "grid":    "#e1e0d9",
    "axis":    "#c3c2b7",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": C["axis"],
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": C["grid"],
    "grid.linewidth": 0.5,
    "grid.alpha": 1.0,
    "xtick.color": C["muted"],
    "ytick.color": C["muted"],
    "figure.facecolor": C["page"],
    "axes.facecolor": C["surface"],
})

SEEDS = ["seed_1", "seed_7", "seed_42", "seed_99", "seed_123",
         "seed_256", "seed_512", "seed_1000", "seed_2024", "seed_31337"]


# ── data loading ─────────────────────────────────────────────────────────────
def load_all_seeds(exp_name, category):
    """Load metrics_per_epoch.csv for all seeds; return (epochs_array, matrix[seeds x epochs])."""
    base = f"log/part_seg/{exp_name}/{category}"
    matrices = {}
    for seed in SEEDS:
        path = os.path.join(base, seed, "metrics_per_epoch.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        matrices[seed] = df
    return matrices


def load_best_mious(exp_name, category):
    """
    Best mIoU per seed = max test_instance_avg_iou across all epochs
    (i.e. the value that triggered best_model.pth).
    """
    seed_data = load_all_seeds(exp_name, category)
    best = {}
    for seed, df in seed_data.items():
        best[seed] = df["test_instance_avg_iou"].max()
    return best


def load_config(exp_name, category):
    seed = "seed_42"
    path = f"log/part_seg/{exp_name}/{category}/{seed}/config.json"
    with open(path) as f:
        return json.load(f)


def num_parts_label(exp_name, category):
    """Return the num_parts value from config."""
    try:
        cfg = load_config(exp_name, category)
        return cfg.get("num_parts") or cfg["args"].get("num_parts", "?")
    except Exception:
        return "?"


def collect_epoch_matrix(exp_name, category, col):
    """Return (epochs 1..N, matrix shape [n_seeds, N])."""
    seed_data = load_all_seeds(exp_name, category)
    rows = []
    for seed in SEEDS:
        if seed in seed_data:
            rows.append(seed_data[seed][col].values)
    mat = np.array(rows)
    n_epochs = mat.shape[1]
    epochs = np.arange(1, n_epochs + 1)
    return epochs, mat


# ── helpers ──────────────────────────────────────────────────────────────────
def mean_std(mat):
    return mat.mean(axis=0), mat.std(axis=0)


def plot_band(ax, epochs, mat, color, label, alpha_band=0.15):
    mu, sd = mean_std(mat)
    ax.plot(epochs, mu, color=color, lw=2, label=label)
    ax.fill_between(epochs, mu - sd, mu + sd, color=color, alpha=alpha_band)
    return mu, sd


def style_ax(ax, xlabel=None, ylabel=None, title=None):
    if title:
        ax.set_title(title, color=C["text"], fontsize=10, fontweight="bold", pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["text2"], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["text2"], fontsize=9)
    ax.tick_params(colors=C["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(C["axis"])
    ax.set_facecolor(C["surface"])


def add_legend(ax):
    ax.legend(framealpha=0, fontsize=9, labelcolor=C["text2"])


# ── individual plots ─────────────────────────────────────────────────────────
def plot_learning_curves(ax, exp_a, exp_b, category, label_a, label_b):
    epochs_a, mat_a = collect_epoch_matrix(exp_a, category, "test_instance_avg_iou")
    epochs_b, mat_b = collect_epoch_matrix(exp_b, category, "test_instance_avg_iou")
    mu_a, sd_a = mean_std(mat_a)
    mu_b, sd_b = mean_std(mat_b)
    ax.plot(epochs_a, mu_a, color=C["blue"], lw=2, label=label_a)
    ax.fill_between(epochs_a, mu_a - sd_a, mu_a + sd_a, color=C["blue"], alpha=0.15)
    ax.plot(epochs_b, mu_b, color=C["aqua"], lw=2, label=label_b)
    ax.fill_between(epochs_b, mu_b - sd_b, mu_b + sd_b, color=C["aqua"], alpha=0.15)
    style_ax(ax, "Epoch", "Instance mIoU", "Test Instance mIoU (mean ± std, 10 seeds)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    add_legend(ax)


def plot_train_loss(ax, exp_a, exp_b, category, label_a, label_b):
    epochs_a, mat_a = collect_epoch_matrix(exp_a, category, "train_loss")
    epochs_b, mat_b = collect_epoch_matrix(exp_b, category, "train_loss")
    mu_a, _ = mean_std(mat_a)
    mu_b, _ = mean_std(mat_b)
    ax.plot(epochs_a, mu_a, color=C["blue"], lw=2, label=label_a)
    ax.plot(epochs_b, mu_b, color=C["aqua"], lw=2, label=label_b)
    style_ax(ax, "Epoch", "Cross-Entropy Loss", "Train Loss (mean, 10 seeds)")
    add_legend(ax)


def plot_boxplot(ax, exp_a, exp_b, category, label_a, label_b):
    best_a = list(load_best_mious(exp_a, category).values())
    best_b = list(load_best_mious(exp_b, category).values())

    bp = ax.boxplot(
        [best_a, best_b],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color=C["text"], lw=2),
        whiskerprops=dict(color=C["muted"], lw=1),
        capprops=dict(color=C["muted"], lw=1),
        flierprops=dict(marker="o", markerfacecolor=C["muted"], markersize=5, linestyle="none"),
        boxprops=dict(linewidth=0),
    )
    colors = [C["blue"], C["aqua"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # overlay individual points with jitter
    rng = np.random.default_rng(0)
    for xi, (vals, color) in enumerate(zip([best_a, best_b], colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=28, zorder=5, alpha=0.85,
                   edgecolors=C["surface"], linewidths=0.8)

    mu_a, mu_b = np.mean(best_a), np.mean(best_b)
    delta = mu_b - mu_a
    sign = "+" if delta >= 0 else ""
    ax.set_xticks([1, 2])
    ax.set_xticklabels([label_a, label_b], color=C["text2"])
    style_ax(ax, ylabel="Best mIoU", title="Best mIoU Distribution (10 seeds)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    # annotate delta
    ymax = max(max(best_a), max(best_b))
    ax.annotate(f"Δ = {sign}{delta:.3f}", xy=(1.5, ymax + 0.005),
                ha="center", fontsize=9, color=C["text2"])


def plot_violin(ax, exp_a, exp_b, category, label_a, label_b):
    best_a = list(load_best_mious(exp_a, category).values())
    best_b = list(load_best_mious(exp_b, category).values())

    parts = ax.violinplot([best_a, best_b], positions=[1, 2], showmeans=True,
                          showmedians=False, showextrema=True)
    colors = [C["blue"], C["aqua"]]
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor(color)
    for key in ["cmeans", "cmaxes", "cmins", "cbars"]:
        if key in parts:
            parts[key].set_color(C["muted"])
            parts[key].set_linewidth(1)

    rng = np.random.default_rng(1)
    for xi, (vals, color) in enumerate(zip([best_a, best_b], colors), start=1):
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=28, zorder=5, alpha=0.9,
                   edgecolors=C["surface"], linewidths=0.8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels([label_a, label_b], color=C["text2"])
    style_ax(ax, ylabel="Best mIoU", title="mIoU Distribution (violin + seeds)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))


def plot_seed_scatter(ax, exp_a, exp_b, category, label_a, label_b):
    """Paired seed comparison: each seed is a point (x=exp_a, y=exp_b)."""
    dict_a = load_best_mious(exp_a, category)
    dict_b = load_best_mious(exp_b, category)
    common = sorted(set(dict_a) & set(dict_b))
    vals_a = [dict_a[s] for s in common]
    vals_b = [dict_b[s] for s in common]

    ax.scatter(vals_a, vals_b, color=C["yellow"], s=50, zorder=5,
               edgecolors=C["text"], linewidths=0.6)
    # diagonal
    lo = min(min(vals_a), min(vals_b)) - 0.01
    hi = max(max(vals_a), max(vals_b)) + 0.01
    ax.plot([lo, hi], [lo, hi], color=C["muted"], lw=1, linestyle="--", zorder=2)

    # label seeds
    seed_labels = [s.replace("seed_", "") for s in common]
    for x, y, lbl in zip(vals_a, vals_b, seed_labels):
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color=C["text2"])

    ax.set_xlabel(label_a, color=C["text2"], fontsize=9)
    ax.set_ylabel(label_b, color=C["text2"], fontsize=9)
    style_ax(ax, title="Seed-Paired mIoU Comparison")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))


def plot_cumulative_best(ax, exp_a, exp_b, category, label_a, label_b):
    """Mean cumulative best mIoU — running max over epochs."""
    epochs_a, mat_a = collect_epoch_matrix(exp_a, category, "test_instance_avg_iou")
    epochs_b, mat_b = collect_epoch_matrix(exp_b, category, "test_instance_avg_iou")

    cum_a = np.maximum.accumulate(mat_a, axis=1)
    cum_b = np.maximum.accumulate(mat_b, axis=1)

    mu_a, sd_a = mean_std(cum_a)
    mu_b, sd_b = mean_std(cum_b)

    ax.plot(epochs_a, mu_a, color=C["blue"], lw=2, label=label_a)
    ax.fill_between(epochs_a, mu_a - sd_a, mu_a + sd_a, color=C["blue"], alpha=0.15)
    ax.plot(epochs_b, mu_b, color=C["aqua"], lw=2, label=label_b)
    ax.fill_between(epochs_b, mu_b - sd_b, mu_b + sd_b, color=C["aqua"], alpha=0.15)

    style_ax(ax, "Epoch", "Cumulative Best mIoU", "Cumulative Best mIoU over Training")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    add_legend(ax)


def plot_convergence_speed(ax, exp_a, exp_b, category, label_a, label_b, thresholds=(0.80, 0.90, 0.95, 0.99)):
    """
    For each seed, find the first epoch where cumulative best >= X% of that seed's best.
    Plot mean epochs-to-threshold for each experiment.
    """
    def epochs_to(exp):
        seed_data = load_all_seeds(exp, category)
        results = {t: [] for t in thresholds}
        for seed, df in seed_data.items():
            series = df["test_instance_avg_iou"].values
            best = series.max()
            cum = np.maximum.accumulate(series)
            for t in thresholds:
                target = t * best
                hits = np.where(cum >= target)[0]
                results[t].append(hits[0] + 1 if len(hits) else len(series))
        return {t: np.array(v) for t, v in results.items()}

    res_a = epochs_to(exp_a)
    res_b = epochs_to(exp_b)

    x = np.arange(len(thresholds))
    w = 0.32
    bars_a = [res_a[t].mean() for t in thresholds]
    bars_b = [res_b[t].mean() for t in thresholds]
    err_a = [res_a[t].std() for t in thresholds]
    err_b = [res_b[t].std() for t in thresholds]

    ax.bar(x - w / 2, bars_a, width=w, color=C["blue"], alpha=0.8, label=label_a,
           yerr=err_a, error_kw=dict(ecolor=C["muted"], lw=1, capsize=3))
    ax.bar(x + w / 2, bars_b, width=w, color=C["aqua"], alpha=0.8, label=label_b,
           yerr=err_b, error_kw=dict(ecolor=C["muted"], lw=1, capsize=3))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t*100)}%" for t in thresholds], color=C["text2"])
    style_ax(ax, "Threshold (% of best mIoU)", "Epochs", "Convergence Speed (epochs to threshold)")
    add_legend(ax)


def plot_generalisation_gap(ax, exp_a, exp_b, category, label_a, label_b):
    """Train accuracy minus test accuracy over epochs (generalisation gap)."""
    for exp, label, color in [(exp_a, label_a, C["blue"]), (exp_b, label_b, C["aqua"])]:
        epochs, mat_tr = collect_epoch_matrix(exp, category, "train_accuracy")
        _, mat_te = collect_epoch_matrix(exp, category, "test_accuracy")
        gap = mat_tr - mat_te
        mu, sd = mean_std(gap)
        ax.plot(epochs, mu, color=color, lw=2, label=label)
        ax.fill_between(epochs, mu - sd, mu + sd, color=color, alpha=0.15)

    ax.axhline(0, color=C["muted"], lw=0.8, linestyle="--")
    style_ax(ax, "Epoch", "Train acc − Test acc", "Generalisation Gap (mean ± std)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))
    add_legend(ax)


def plot_best_epoch_dist(ax, exp_a, exp_b, category, label_a, label_b):
    """Histogram of the epoch where best mIoU was first achieved."""
    def best_epochs(exp):
        seed_data = load_all_seeds(exp, category)
        bests = []
        for df in seed_data.values():
            bests.append(df["test_instance_avg_iou"].idxmax() + 1)  # 1-indexed epoch
        return bests

    be_a = best_epochs(exp_a)
    be_b = best_epochs(exp_b)

    bins = np.linspace(0, 100, 21)
    ax.hist(be_a, bins=bins, color=C["blue"], alpha=0.6, label=label_a, edgecolor=C["surface"], lw=0.5)
    ax.hist(be_b, bins=bins, color=C["aqua"], alpha=0.6, label=label_b, edgecolor=C["surface"], lw=0.5)
    style_ax(ax, "Epoch of Best Checkpoint", "Count", "Best Checkpoint Epoch")
    add_legend(ax)


# ── main dashboard ────────────────────────────────────────────────────────────
def build_dashboard(category, exp_a, exp_b, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    np_a = num_parts_label(exp_a, category)
    np_b = num_parts_label(exp_b, category)
    label_a = f"{exp_a}\n({np_a}-part)"
    label_b = f"{exp_b}\n({np_b}-part)"
    label_a_short = f"{exp_a} ({np_a}-part)"
    label_b_short = f"{exp_b} ({np_b}-part)"

    fig = plt.figure(figsize=(18, 22), facecolor=C["page"])
    fig.suptitle(
        f"{category} — {label_a_short}  vs  {label_b_short}",
        fontsize=14, fontweight="bold", color=C["text"], y=0.995
    )

    gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38,
                  left=0.07, right=0.97, top=0.96, bottom=0.04)

    # Row 0: learning curve (wide), train loss
    ax0 = fig.add_subplot(gs[0, :2])
    ax1 = fig.add_subplot(gs[0, 2])

    # Row 1: box, violin, seed scatter
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    # Row 2: cumulative best (wide), convergence speed
    ax5 = fig.add_subplot(gs[2, :2])
    ax6 = fig.add_subplot(gs[2, 2])

    # Row 3: generalisation gap (wide), best epoch dist
    ax7 = fig.add_subplot(gs[3, :2])
    ax8 = fig.add_subplot(gs[3, 2])

    plot_learning_curves(ax0, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_train_loss(ax1, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_boxplot(ax2, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_violin(ax3, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_seed_scatter(ax4, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_cumulative_best(ax5, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_convergence_speed(ax6, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_generalisation_gap(ax7, exp_a, exp_b, category, label_a_short, label_b_short)
    plot_best_epoch_dist(ax8, exp_a, exp_b, category, label_a_short, label_b_short)

    out_path = os.path.join(out_dir, f"comparison_{category}_{exp_a}_vs_{exp_b}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C["page"])
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ── individual saved plots ────────────────────────────────────────────────────
def save_individual_plots(category, exp_a, exp_b, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    np_a = num_parts_label(exp_a, category)
    np_b = num_parts_label(exp_b, category)
    label_a = f"{exp_a} ({np_a}-part)"
    label_b = f"{exp_b} ({np_b}-part)"

    plots = [
        ("01_learning_curves",  plot_learning_curves,   (8, 4)),
        ("02_train_loss",       plot_train_loss,         (8, 4)),
        ("03_boxplot",          plot_boxplot,            (6, 5)),
        ("04_violin",           plot_violin,             (6, 5)),
        ("05_seed_scatter",     plot_seed_scatter,       (5, 5)),
        ("06_cumulative_best",  plot_cumulative_best,    (8, 4)),
        ("07_convergence_speed",plot_convergence_speed,  (7, 4)),
        ("08_generalisation",   plot_generalisation_gap, (8, 4)),
        ("09_best_epoch",       plot_best_epoch_dist,    (7, 4)),
    ]

    paths = []
    for name, fn, size in plots:
        fig, ax = plt.subplots(figsize=size, facecolor=C["page"])
        fig.patch.set_facecolor(C["page"])
        fn(ax, exp_a, exp_b, category, label_a, label_b)
        p = os.path.join(out_dir, f"{name}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C["page"])
        plt.close(fig)
        print(f"  {p}")
        paths.append(p)
    return paths


# ── entry point ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--exp_a",    required=True, help="e.g. parts2_cap312")
    p.add_argument("--exp_b",    required=True, help="e.g. parts3_cap626")
    p.add_argument("--out_dir",  default=None,
                   help="Output directory (default: comparison_plots/<category>_<expA>_vs_<expB>/)")
    p.add_argument("--individual", action="store_true",
                   help="Also save each plot individually")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = args.out_dir or f"comparison_plots/{args.category}_{args.exp_a}_vs_{args.exp_b}"
    dashboard = build_dashboard(args.category, args.exp_a, args.exp_b, out)
    if args.individual:
        print("Individual plots:")
        save_individual_plots(args.category, args.exp_a, args.exp_b, out)
