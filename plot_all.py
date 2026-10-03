"""
Comprehensive visualization for a single-category × seed experiment.

Auto-discovers the categories and seeds under log/part_seg/<exp_name>/ and writes
the full 20-figure plot set into plots_<exp_name>/. Works for any number of
categories (the subplot grids adapt) and any seed list (<= tab10's 10 colors).

Usage:
  python plot_all.py --exp_name parts4_cap596
  python plot_all.py --exp_name parts3_cap626 --outdir plots_parts3_cap626
"""

import argparse
import csv
import math
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

THESIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("Comprehensive per-experiment plot set")
    p.add_argument("--exp_name", required=True,
                   help="experiment folder name under log/part_seg/")
    p.add_argument("--base", default=None,
                   help="path to the experiment dir (default: log/part_seg/<exp_name>)")
    p.add_argument("--outdir", default=None,
                   help="output folder for PNGs (default: plots_<exp_name>)")
    return p.parse_args()

args = parse_args()
EXP_NAME = args.exp_name
BASE   = args.base   or os.path.join(THESIS_DIR, "log", "part_seg", EXP_NAME)
OUTDIR = args.outdir or os.path.join(THESIS_DIR, f"plots_{EXP_NAME}")
BASE   = BASE.replace("\\", "/")
OUTDIR = OUTDIR.replace("\\", "/")

if not os.path.isdir(BASE):
    raise SystemExit(f"Experiment dir not found: {BASE}")
os.makedirs(OUTDIR, exist_ok=True)


# ── discovery ─────────────────────────────────────────────────────────────────
def _seed_dirs(path):
    return [d for d in os.listdir(path)
            if d.startswith("seed_") and os.path.isdir(os.path.join(path, d))]

def discover_classes(base):
    out = []
    for d in sorted(os.listdir(base)):
        cdir = os.path.join(base, d)
        if os.path.isdir(cdir) and _seed_dirs(cdir):
            out.append(d)
    return out

def discover_seeds(base, classes):
    seeds = set()
    for cls in classes:
        for d in _seed_dirs(os.path.join(base, cls)):
            try:
                seeds.add(int(d[len("seed_"):]))
            except ValueError:
                pass
    return [f"seed_{s}" for s in sorted(seeds)]

CLASSES = discover_classes(BASE)
if not CLASSES:
    raise SystemExit(f"No category subfolders with seed_* runs under {BASE}")
SEEDS = discover_seeds(BASE, CLASSES)
if not SEEDS:
    raise SystemExit(f"No seed_* runs found under {BASE}")
SEED_LABELS = [s.replace("seed_", "") for s in SEEDS]

# Cap parsed from names like 'parts4_cap596' (for titles only; optional).
_cap_m = re.search(r"cap(\d+)", EXP_NAME)
CAP = _cap_m.group(1) if _cap_m else None

print(f'Experiment "{EXP_NAME}": classes={CLASSES}  seeds={SEED_LABELS}')

# ── colors ────────────────────────────────────────────────────────────────────
# Keep the original palette for the four canonical classes; fall back to tab10
# (by discovery order) for any other category name.
_BASE_COLORS = {
    "Airplane": "#4C72B0",
    "Car":      "#DD8452",
    "Chair":    "#55A868",
    "Lamp":     "#C44E52",
}
SEED_CMAP = plt.cm.tab10
CLASS_COLORS = {
    cls: _BASE_COLORS.get(cls, matplotlib.colors.to_hex(plt.cm.tab10(i % 10)))
    for i, cls in enumerate(CLASSES)
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":   120,
    "savefig.dpi":  150,
    "savefig.bbox": "tight",
})


# ── layout helper ─────────────────────────────────────────────────────────────
def grid_axes(n, figsize_per=(7, 5), sharex=False, sharey=False, ncols=2):
    """Create an n-panel figure on a 2-col grid; return (fig, axes_list[:n]) with
    any leftover axes hidden. Used where the original code assumed a 2x2 grid."""
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]

N_CLS = len(CLASSES)

# ── data loading ──────────────────────────────────────────────────────────────
def load_metrics(cls, seed):
    path = f"{BASE}/{cls}/{seed}/metrics_per_epoch.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # normalise header keys (strip spaces)
    rows = [{k.strip(): v.strip() for k, v in r.items()} for r in rows]
    def col(key):
        return np.array([float(r[key]) for r in rows])
    return {
        "epoch":     col("epoch").astype(int),
        "loss":      col("train_loss"),
        "train_acc": col("train_accuracy"),
        "test_acc":  col("test_accuracy"),
        "miou":      col("test_instance_avg_iou"),
        "lr":        col("learning_rate"),
        "time":      col("epoch_time_seconds"),
    }

def load_best_miou(cls, seed):
    path = f"{BASE}/{cls}/{seed}/per_category_miou.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return float(row["miou"])

# pre-load everything
data = {}
for cls in CLASSES:
    data[cls] = {}
    for seed in SEEDS:
        data[cls][seed] = load_metrics(cls, seed)

best_miou = {cls: np.array([load_best_miou(cls, s) for s in SEEDS]) for cls in CLASSES}

# Detect epoch count from the data (don't assume 100).
N_EPOCHS = len(data[CLASSES[0]][SEEDS[0]]["epoch"])
epochs = np.arange(1, N_EPOCHS + 1)

def mean_std(cls, key):
    mat = np.stack([data[cls][s][key] for s in SEEDS])
    return mat.mean(0), mat.std(0)

def save(fig, name):
    path = f"{OUTDIR}/{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved -> {name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LEARNING CURVES — mean ± 1 std (4 metrics, 4 classes, one figure)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(N_CLS, 4, figsize=(20, 4 * N_CLS), sharex=True)
axes = np.atleast_2d(axes)
fig.suptitle(f"Learning Curves — mean ± 1 std across {len(SEEDS)} seeds",
             fontsize=15, y=1.01)

metrics_cfg = [
    ("loss",      "Train Loss",          None),
    ("miou",      "Test mIoU",           None),
    ("train_acc", "Train Accuracy",      None),
    ("test_acc",  "Test Accuracy",       None),
]

for col_i, (key, ylabel, _) in enumerate(metrics_cfg):
    for row_i, cls in enumerate(CLASSES):
        ax = axes[row_i][col_i]
        mu, sd = mean_std(cls, key)
        color = CLASS_COLORS[cls]
        ax.plot(epochs, mu, color=color, lw=2)
        ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.25, color=color)
        # LR step markers
        lr_arr = mean_std(cls, "lr")[0]
        steps = np.where(np.diff(lr_arr) < 0)[0] + 2
        for s in steps:
            ax.axvline(s, color="grey", lw=0.8, ls="--", alpha=0.5)
        if row_i == 0:
            ax.set_title(ylabel)
        if col_i == 0:
            ax.set_ylabel(cls, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, N_EPOCHS)

for ax in axes[-1]:
    ax.set_xlabel("Epoch")

fig.tight_layout()
save(fig, "01_learning_curves_mean_std")

# ══════════════════════════════════════════════════════════════════════════════
# 2. INDIVIDUAL SEED CURVES (spaghetti) — mIoU
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = grid_axes(N_CLS, figsize_per=(7, 5), sharex=True)
fig.suptitle("Test mIoU per Seed (spaghetti plot)", fontsize=14)

for i, cls in enumerate(CLASSES):
    ax = axes[i]
    for j, seed in enumerate(SEEDS):
        miou = data[cls][seed]["miou"]
        ax.plot(epochs, miou, color=SEED_CMAP(j), lw=1.0, alpha=0.75,
                label=SEED_LABELS[j])
    # overlay mean
    mu = np.stack([data[cls][s]["miou"] for s in SEEDS]).mean(0)
    ax.plot(epochs, mu, color="black", lw=2.5, label="mean")
    ax.set_title(cls, color=CLASS_COLORS[cls])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mIoU")
    ax.grid(True, alpha=0.3)

handles = [Line2D([0], [0], color=SEED_CMAP(j), lw=1.5, label=SEED_LABELS[j])
           for j in range(len(SEEDS))]
handles.append(Line2D([0], [0], color="black", lw=2.5, label="mean"))
fig.legend(handles=handles, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.03), title="Seed")
fig.tight_layout()
save(fig, "02_spaghetti_miou")

# ══════════════════════════════════════════════════════════════════════════════
# 3. BOX + STRIP PLOT — best checkpoint mIoU per class
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
positions = np.arange(len(CLASSES))
for i, cls in enumerate(CLASSES):
    vals = best_miou[cls]
    bp = ax.boxplot(vals, positions=[i], widths=0.4,
                    patch_artist=True,
                    medianprops=dict(color="black", lw=2),
                    boxprops=dict(facecolor=CLASS_COLORS[cls], alpha=0.5),
                    whiskerprops=dict(lw=1.2),
                    capprops=dict(lw=1.2),
                    flierprops=dict(marker="o", ms=4))
    # strip
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(vals))
    ax.scatter(np.full(len(vals), i) + jitter, vals,
               color=CLASS_COLORS[cls], edgecolors="black",
               s=40, zorder=3, alpha=0.9)

ax.set_xticks(positions)
ax.set_xticklabels(CLASSES)
ax.set_ylabel("Best Checkpoint mIoU")
ax.set_title("mIoU Distribution across 10 Seeds (best checkpoint)")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "03_boxplot_best_miou")

# ══════════════════════════════════════════════════════════════════════════════
# 4. HEATMAP — mIoU per class × seed
# ══════════════════════════════════════════════════════════════════════════════
mat = np.array([best_miou[cls] for cls in CLASSES])  # (4, 10)

fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(mat, aspect="auto", cmap="YlGn",
               vmin=mat.min() - 0.005, vmax=mat.max() + 0.005)
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels(SEED_LABELS, rotation=45, ha="right")
ax.set_yticks(range(len(CLASSES)))
ax.set_yticklabels(CLASSES)
ax.set_title("Best Checkpoint mIoU — Class × Seed Heatmap")
ax.set_xlabel("Seed")

for i in range(len(CLASSES)):
    for j in range(len(SEEDS)):
        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                fontsize=8.5, color="black")

plt.colorbar(im, ax=ax, label="mIoU")
fig.tight_layout()
save(fig, "04_heatmap_class_seed")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BEST EPOCH DISTRIBUTION — histogram per class
# ══════════════════════════════════════════════════════════════════════════════
def best_epoch(cls, seed):
    miou = data[cls][seed]["miou"]
    return int(np.argmax(miou)) + 1

fig, axes = plt.subplots(1, N_CLS, figsize=(4 * N_CLS, 4), sharey=True)
axes = np.atleast_1d(axes)
fig.suptitle("Epoch at Which Best mIoU Was Achieved (across seeds)", fontsize=13)
for i, cls in enumerate(CLASSES):
    bepochs = [best_epoch(cls, s) for s in SEEDS]
    ax = axes[i]
    ax.bar(range(len(SEEDS)), bepochs, color=CLASS_COLORS[cls], alpha=0.8,
           edgecolor="black", lw=0.6)
    ax.axhline(np.mean(bepochs), color="black", lw=1.5, ls="--",
               label=f"mean={np.mean(bepochs):.1f}")
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels(SEED_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_title(cls, color=CLASS_COLORS[cls])
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    if i == 0:
        ax.set_ylabel("Best Epoch")
    ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "05_best_epoch_distribution")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONVERGENCE GAP — best vs final mIoU per class/seed
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, N_CLS, figsize=(4 * N_CLS, 5), sharey=False)
axes = np.atleast_1d(axes)
fig.suptitle("Convergence Gap: Best Checkpoint mIoU vs Final Epoch mIoU", fontsize=13)

for i, cls in enumerate(CLASSES):
    ax = axes[i]
    best_vals  = [data[cls][s]["miou"].max() for s in SEEDS]
    final_vals = [data[cls][s]["miou"][-1]   for s in SEEDS]
    x = np.arange(len(SEEDS))
    ax.bar(x - 0.2, best_vals,  0.35, label="Best",  color=CLASS_COLORS[cls], alpha=0.85, edgecolor="black", lw=0.5)
    ax.bar(x + 0.2, final_vals, 0.35, label="Final", color=CLASS_COLORS[cls], alpha=0.40, edgecolor="black", lw=0.5)
    # gap arrows
    for j, (b, f) in enumerate(zip(best_vals, final_vals)):
        ax.annotate("", xy=(j + 0.2, f), xytext=(j + 0.2, b),
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(SEED_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_title(cls, color=CLASS_COLORS[cls])
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(True, axis="y", alpha=0.3)
    ymin = min(min(best_vals), min(final_vals)) - 0.005
    ymax = max(max(best_vals), max(final_vals)) + 0.005
    ax.set_ylim(ymin, ymax)

fig.tight_layout()
save(fig, "06_convergence_gap")

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN LOSS CURVES — mean ± std, all 4 classes overlaid
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for cls in CLASSES:
    mu, sd = mean_std(cls, "loss")
    c = CLASS_COLORS[cls]
    ax.plot(epochs, mu, color=c, lw=2, label=cls)
    ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.2, color=c)
ax.set_xlabel("Epoch")
ax.set_ylabel("Train Loss")
ax.set_title("Train Loss — all classes (mean ± 1 std)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "07_train_loss_overlay")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TEST mIoU CURVES — mean ± std, all 4 classes overlaid
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for cls in CLASSES:
    mu, sd = mean_std(cls, "miou")
    c = CLASS_COLORS[cls]
    ax.plot(epochs, mu, color=c, lw=2, label=cls)
    ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.2, color=c)
# LR step markers for reference (use the first class as representative)
lr_arr = mean_std(CLASSES[0], "lr")[0]
steps = np.where(np.diff(lr_arr) < 0)[0] + 2
for s in steps:
    ax.axvline(s, color="grey", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("Epoch")
ax.set_ylabel("Test mIoU")
ax.set_title("Test mIoU — all classes (mean ± 1 std)\n(dashed lines = LR decay steps)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "08_test_miou_overlay")

# ══════════════════════════════════════════════════════════════════════════════
# 9. GENERALISATION GAP — train_acc vs test_acc per class
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = grid_axes(N_CLS, figsize_per=(7, 5), sharex=True)
fig.suptitle("Generalisation Gap: Train vs Test Accuracy (mean ± 1 std)", fontsize=13)
for i, cls in enumerate(CLASSES):
    ax = axes[i]
    tr_mu, tr_sd = mean_std(cls, "train_acc")
    te_mu, te_sd = mean_std(cls, "test_acc")
    c = CLASS_COLORS[cls]
    ax.plot(epochs, tr_mu, color=c, lw=2, label="Train")
    ax.fill_between(epochs, tr_mu - tr_sd, tr_mu + tr_sd, alpha=0.2, color=c)
    ax.plot(epochs, te_mu, color=c, lw=2, ls="--", label="Test")
    ax.fill_between(epochs, te_mu - te_sd, te_mu + te_sd, alpha=0.1, color=c)
    ax.fill_between(epochs, te_mu, tr_mu, alpha=0.12, color="red", label="gap")
    ax.set_title(cls, color=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "09_generalisation_gap")

# ══════════════════════════════════════════════════════════════════════════════
# 10. mIoU AT KEY TRAINING STAGES — grouped bar
# ══════════════════════════════════════════════════════════════════════════════
stages = [e for e in [10, 20, 40, 60, 80, 100] if e <= N_EPOCHS]
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(stages))
width = 0.8 / N_CLS
for i, cls in enumerate(CLASSES):
    stage_means = []
    for ep in stages:
        vals = np.array([data[cls][s]["miou"][ep - 1] for s in SEEDS])
        stage_means.append(vals.mean())
    ax.bar(x + (i - (N_CLS - 1) / 2) * width, stage_means, width,
           label=cls, color=CLASS_COLORS[cls], alpha=0.85, edgecolor="black", lw=0.5)

ax.set_xticks(x)
ax.set_xticklabels([f"Epoch {e}" for e in stages])
ax.set_ylabel("Mean mIoU (across seeds)")
ax.set_title("Mean Test mIoU at Key Training Stages")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
fig.tight_layout()
save(fig, "10_miou_at_stages")

# ══════════════════════════════════════════════════════════════════════════════
# 11. SEED RANKING — per class, sorted bar chart
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = grid_axes(N_CLS, figsize_per=(7, 5))
fig.suptitle("Seed Ranking by Best Checkpoint mIoU", fontsize=13)
for i, cls in enumerate(CLASSES):
    ax = axes[i]
    vals = best_miou[cls]
    sorted_idx = np.argsort(vals)[::-1]
    sorted_labels = [SEED_LABELS[j] for j in sorted_idx]
    sorted_vals   = vals[sorted_idx]
    colors_bar = [CLASS_COLORS[cls]] * len(sorted_vals)
    colors_bar[0] = "gold"    # best
    colors_bar[-1] = "silver" # worst
    bars = ax.bar(range(len(SEEDS)), sorted_vals, color=colors_bar,
                  edgecolor="black", lw=0.6)
    ax.set_xticks(range(len(SEEDS)))
    ax.set_xticklabels(sorted_labels)
    ax.set_title(cls, color=CLASS_COLORS[cls])
    ax.set_xlabel("Seed (ranked)")
    ax.set_ylabel("Best mIoU")
    ymin = sorted_vals.min() - 0.003
    ymax = sorted_vals.max() + 0.003
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.grid(True, axis="y", alpha=0.3)
    # value labels
    for bar, v in zip(bars, sorted_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
fig.tight_layout()
save(fig, "11_seed_ranking")

# ══════════════════════════════════════════════════════════════════════════════
# 12. LEARNING RATE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4))
lr = data[CLASSES[0]][SEEDS[0]]["lr"]
ax.step(epochs, lr, where="post", color="#4C72B0", lw=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Learning Rate")
ax.set_title("Learning Rate Schedule (step decay ×0.5 every 20 epochs)")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "12_lr_schedule")

# ══════════════════════════════════════════════════════════════════════════════
# 13. EPOCH TIME — mean ± std per class
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for cls in CLASSES:
    mu, sd = mean_std(cls, "time")
    c = CLASS_COLORS[cls]
    ax.plot(epochs, mu, color=c, lw=1.5, label=cls)
    ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.2, color=c)
ax.set_xlabel("Epoch")
ax.set_ylabel("Wall-clock Time (seconds)")
ax.set_title("Epoch Training Time (mean ± 1 std)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "13_epoch_time")

# ══════════════════════════════════════════════════════════════════════════════
# 14. CUMULATIVE BEST mIoU — tracking best-so-far over epochs
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = grid_axes(N_CLS, figsize_per=(7, 5), sharex=True)
fig.suptitle("Cumulative Best mIoU (best-so-far over epochs)", fontsize=13)
for i, cls in enumerate(CLASSES):
    ax = axes[i]
    cummax_mat = []
    for seed in SEEDS:
        miou = data[cls][seed]["miou"]
        cummax = np.maximum.accumulate(miou)
        cummax_mat.append(cummax)
    cummax_mat = np.array(cummax_mat)
    mu = cummax_mat.mean(0)
    sd = cummax_mat.std(0)
    c = CLASS_COLORS[cls]
    for j, cm in enumerate(cummax_mat):
        ax.plot(epochs, cm, color=SEED_CMAP(j), lw=0.8, alpha=0.5)
    ax.plot(epochs, mu, color="black", lw=2.5, label="mean")
    ax.fill_between(epochs, mu - sd, mu + sd, alpha=0.2, color=c)
    ax.set_title(cls, color=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best mIoU so far")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
save(fig, "14_cumulative_best_miou")

# ══════════════════════════════════════════════════════════════════════════════
# 15. EPOCHS TO REACH X% OF FINAL BEST mIoU
# ══════════════════════════════════════════════════════════════════════════════
thresholds = [0.90, 0.95, 0.99]
threshold_labels = ["90%", "95%", "99%"]

def epochs_to_threshold(cls, seed, frac):
    miou = data[cls][seed]["miou"]
    target = miou.max() * frac
    for ep, v in enumerate(miou):
        if v >= target:
            return ep + 1
    return N_EPOCHS

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(CLASSES))
width = 0.22
for ti, (frac, tlabel) in enumerate(zip(thresholds, threshold_labels)):
    means = []
    stds  = []
    for cls in CLASSES:
        vals = [epochs_to_threshold(cls, s, frac) for s in SEEDS]
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    means = np.array(means)
    stds  = np.array(stds)
    bars = ax.bar(x + ti * width - width, means, width,
                  label=tlabel, alpha=0.82, edgecolor="black", lw=0.5)
    ax.errorbar(x + ti * width - width, means, yerr=stds,
                fmt="none", ecolor="black", capsize=3, lw=1.2)

ax.set_xticks(x)
ax.set_xticklabels(CLASSES)
ax.set_ylabel("Epoch")
ax.set_title("Epochs to Reach X% of Best mIoU (mean ± std across seeds)")
ax.legend(title="Threshold")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "15_epochs_to_threshold")

# ══════════════════════════════════════════════════════════════════════════════
# 16. VIOLIN PLOT — best mIoU distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
parts = ax.violinplot([best_miou[cls] for cls in CLASSES],
                      positions=range(len(CLASSES)),
                      showmeans=True, showmedians=True, showextrema=True)
for i, (pc, cls) in enumerate(zip(parts["bodies"], CLASSES)):
    pc.set_facecolor(CLASS_COLORS[cls])
    pc.set_alpha(0.6)
for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
    parts[key].set_color("black")
    parts[key].set_linewidth(1.2)
# overlay points
for i, cls in enumerate(CLASSES):
    jitter = np.random.default_rng(1).uniform(-0.08, 0.08, len(SEEDS))
    ax.scatter(np.full(len(SEEDS), i) + jitter, best_miou[cls],
               color=CLASS_COLORS[cls], edgecolors="black", s=45, zorder=3)
ax.set_xticks(range(len(CLASSES)))
ax.set_xticklabels(CLASSES)
ax.set_ylabel("Best Checkpoint mIoU")
ax.set_title("Violin Plot — mIoU Distribution per Class")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "16_violin_best_miou")

# ══════════════════════════════════════════════════════════════════════════════
# 17. FINAL SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
gs  = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
_cap_txt = f"{CAP} samples cap, " if CAP else ""
fig.suptitle(f"Experiment Summary Dashboard — {EXP_NAME}\n"
             f"{N_CLS} classes, {_cap_txt}{len(SEEDS)} seeds, {N_EPOCHS} epochs",
             fontsize=14, y=1.01)

# (A) mean ± std mIoU bar
ax_a = fig.add_subplot(gs[0, :2])
means = [best_miou[cls].mean() for cls in CLASSES]
stds  = [best_miou[cls].std()  for cls in CLASSES]
bars = ax_a.bar(CLASSES, means, yerr=stds, capsize=5,
                color=[CLASS_COLORS[c] for c in CLASSES],
                alpha=0.85, edgecolor="black", lw=0.7, error_kw=dict(lw=1.5))
for bar, m in zip(bars, means):
    ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
              f"{m:.4f}", ha="center", va="bottom", fontsize=9)
ymin_a = min(means) - 0.02
ax_a.set_ylim(ymin_a, max(means) + 0.015)
ax_a.set_ylabel("Mean Best mIoU")
ax_a.set_title("(A) Performance per Class (mean ± 1 std)")
ax_a.grid(True, axis="y", alpha=0.3)

# (B) std bar
ax_b = fig.add_subplot(gs[0, 2:])
stds_pt = [best_miou[cls].std() * 100 for cls in CLASSES]
ax_b.bar(CLASSES, stds_pt, color=[CLASS_COLORS[c] for c in CLASSES],
         alpha=0.75, edgecolor="black", lw=0.7)
ax_b.set_ylabel("Std (mIoU points × 100)")
ax_b.set_title("(B) Seed-to-Seed Variability")
ax_b.grid(True, axis="y", alpha=0.3)

# (C) mIoU over epochs
ax_c = fig.add_subplot(gs[1, :2])
for cls in CLASSES:
    mu, sd = mean_std(cls, "miou")
    ax_c.plot(epochs, mu, color=CLASS_COLORS[cls], lw=1.8, label=cls)
    ax_c.fill_between(epochs, mu - sd, mu + sd, alpha=0.15, color=CLASS_COLORS[cls])
ax_c.set_xlabel("Epoch")
ax_c.set_ylabel("Test mIoU")
ax_c.set_title("(C) Test mIoU Convergence")
ax_c.legend(fontsize=8)
ax_c.grid(True, alpha=0.3)

# (D) train loss
ax_d = fig.add_subplot(gs[1, 2:])
for cls in CLASSES:
    mu, sd = mean_std(cls, "loss")
    ax_d.plot(epochs, mu, color=CLASS_COLORS[cls], lw=1.8, label=cls)
    ax_d.fill_between(epochs, mu - sd, mu + sd, alpha=0.15, color=CLASS_COLORS[cls])
ax_d.set_xlabel("Epoch")
ax_d.set_ylabel("Train Loss")
ax_d.set_title("(D) Train Loss")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.3)

# (E) best epoch
ax_e = fig.add_subplot(gs[2, :2])
be_means = []
be_stds  = []
for cls in CLASSES:
    be = [best_epoch(cls, s) for s in SEEDS]
    be_means.append(np.mean(be))
    be_stds.append(np.std(be))
ax_e.bar(CLASSES, be_means, yerr=be_stds, capsize=5,
         color=[CLASS_COLORS[c] for c in CLASSES],
         alpha=0.8, edgecolor="black", lw=0.7, error_kw=dict(lw=1.5))
ax_e.set_ylabel("Epoch")
ax_e.set_title("(E) Mean Epoch of Best Model")
ax_e.grid(True, axis="y", alpha=0.3)
ax_e.set_ylim(0, 115)

# (F) convergence gap
ax_f = fig.add_subplot(gs[2, 2:])
gap_means = []
gap_stds  = []
for cls in CLASSES:
    gaps = [data[cls][s]["miou"].max() - data[cls][s]["miou"][-1] for s in SEEDS]
    gap_means.append(np.mean(gaps) * 100)
    gap_stds.append(np.std(gaps) * 100)
ax_f.bar(CLASSES, gap_means, yerr=gap_stds, capsize=5,
         color=[CLASS_COLORS[c] for c in CLASSES],
         alpha=0.8, edgecolor="black", lw=0.7, error_kw=dict(lw=1.5))
ax_f.set_ylabel("Gap (mIoU pts × 100)")
ax_f.set_title("(F) Convergence Gap: Best − Final Epoch")
ax_f.grid(True, axis="y", alpha=0.3)

save(fig, "00_summary_dashboard")

# ══════════════════════════════════════════════════════════════════════════════
# 18. mIoU IMPROVEMENT SPEED — delta mIoU between consecutive epochs
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = grid_axes(N_CLS, figsize_per=(7, 5), sharex=True)
fig.suptitle("Epoch-over-Epoch mIoU Improvement (mean ± 1 std)", fontsize=13)
for i, cls in enumerate(CLASSES):
    ax = axes[i]
    deltas = []
    for seed in SEEDS:
        miou = data[cls][seed]["miou"]
        deltas.append(np.diff(miou))
    deltas = np.array(deltas)
    mu = deltas.mean(0)
    sd = deltas.std(0)
    c = CLASS_COLORS[cls]
    ax.plot(epochs[1:], mu, color=c, lw=1.5)
    ax.fill_between(epochs[1:], mu - sd, mu + sd, alpha=0.25, color=c)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(cls, color=c)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ΔmIoU")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "17_miou_delta")

# ══════════════════════════════════════════════════════════════════════════════
# 19. TRAIN ACCURACY vs TEST mIoU SCATTER — all seeds, all epochs sampled
# ══════════════════════════════════════════════════════════════════════════════
sample_epochs = [e for e in [20, 40, 60, 80, 100] if e <= N_EPOCHS]
fig, axes = plt.subplots(1, N_CLS, figsize=(4.5 * N_CLS, 5))
axes = np.atleast_1d(axes)
fig.suptitle("Train Accuracy vs Test mIoU (sampled epochs, all seeds)", fontsize=13)
for i, cls in enumerate(CLASSES):
    ax = axes[i]
    for j, seed in enumerate(SEEDS):
        tr_acc = data[cls][seed]["train_acc"]
        miou   = data[cls][seed]["miou"]
        for ep in sample_epochs:
            ax.scatter(tr_acc[ep - 1], miou[ep - 1],
                       color=SEED_CMAP(j), s=30 + ep * 0.4,
                       alpha=0.8, edgecolors="black", lw=0.3)
    ax.set_title(cls, color=CLASS_COLORS[cls])
    ax.set_xlabel("Train Accuracy")
    ax.set_ylabel("Test mIoU")
    ax.grid(True, alpha=0.3)

# legend for epochs
for ep in sample_epochs:
    ax.scatter([], [], c="grey", s=30 + ep * 0.4, label=f"E{ep}", edgecolors="black", lw=0.3)
axes[-1].legend(title="Epoch", fontsize=8, loc="lower right")
fig.tight_layout()
save(fig, "18_scatter_trainacc_vs_miou")

# ══════════════════════════════════════════════════════════════════════════════
# 20. PER-CLASS METRICS TABLE (printed + saved as PNG)
# ══════════════════════════════════════════════════════════════════════════════
col_labels = ["Class", "Mean mIoU", "Std", "Min", "Max", "Range",
              "Avg Best Epoch", "Avg Gap (pts)"]
rows_data = []
for cls in CLASSES:
    arr = best_miou[cls]
    gap = np.mean([data[cls][s]["miou"].max() - data[cls][s]["miou"][-1] for s in SEEDS])
    be  = np.mean([best_epoch(cls, s) for s in SEEDS])
    rows_data.append([
        cls,
        f"{arr.mean():.4f}",
        f"{arr.std():.4f}",
        f"{arr.min():.4f}",
        f"{arr.max():.4f}",
        f"{(arr.max()-arr.min()):.4f}",
        f"{be:.1f}",
        f"{gap*100:.2f}",
    ])

fig, ax = plt.subplots(figsize=(14, 2.5))
ax.axis("off")
tbl = ax.table(cellText=rows_data, colLabels=col_labels,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.1, 2.0)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0:
        cell.set_facecolor(CLASS_COLORS[CLASSES[row - 1]])
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")
ax.set_title(f"Summary Metrics Table — {EXP_NAME}", fontsize=12, pad=20)
fig.tight_layout()
save(fig, "19_metrics_table")

print("\nAll plots saved to:", OUTDIR)
