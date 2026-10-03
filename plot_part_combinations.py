"""Per-category part-combination analysis of ShapeNetPart.

For every category this script:
  1. counts which distinct sets of part IDs actually occur across all splits
     and saves a combination-frequency bar chart, and
  2. saves an interactive 3D point-cloud plot of one sample object for each
     combination that appears.

Outputs go to <out_dir>/<Category>/ as both PNG (kaleido) and HTML.

Usage:
    python plot_part_combinations.py
    python plot_part_combinations.py --out_dir figures --max_points 4096
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
import plotly.graph_objects as go

from data_utils.ShapeNetDataLoader import SEG_CLASSES

# Semantic name of each global part ID (Pistol 38-40 absent from this dataset).
PART_ID_NAMES = {
    0: "Body", 1: "Wing", 2: "Tail", 3: "Engine",            # Airplane
    4: "Handle", 5: "Body",                                   # Bag
    6: "Panels", 7: "Peak",                                   # Cap
    8: "Roof", 9: "Hood", 10: "Wheel", 11: "Body",            # Car
    12: "Back", 13: "Seat", 14: "Leg", 15: "Armrest",         # Chair
    16: "Headband", 17: "Earcup", 18: "Cord",                 # Earphone
    19: "Head", 20: "Neck", 21: "Body",                       # Guitar
    22: "Blade", 23: "Handle",                                # Knife
    24: "Base", 25: "Shade", 26: "Canopy", 27: "Pole",        # Lamp
    28: "Keyboard", 29: "Screen",                             # Laptop
    30: "Gas tank", 31: "Seat", 32: "Wheel",                  # Motorbike
    33: "Handle", 34: "Light", 35: "Frame",
    36: "Handle", 37: "Cup",                                  # Mug
    41: "Body", 42: "Fin", 43: "Nose",                        # Rocket
    44: "Wheel", 45: "Deck", 46: "Belt",                      # Skateboard
    47: "Top", 48: "Leg", 49: "Support",                      # Table
}

# Ordinal blue ramp: combination size 1..6, light -> dark.
ORDINAL_RAMP = {
    1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6",
    4: "#1c5cab", 5: "#184f95", 6: "#0d366b",
}
# Categorical palette for part identity in the 3D plots (fixed slot order;
# a part keeps its slot even when other parts are missing from the combo).
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

FONT = dict(family='system-ui, -apple-system, "Segoe UI", sans-serif')


def combo_label(part_ids):
    return " + ".join(PART_ID_NAMES.get(p, f"p{p}") for p in sorted(part_ids))


def combo_filename(part_ids):
    ids = "-".join(str(p) for p in sorted(part_ids))
    names = "+".join(PART_ID_NAMES.get(p, f"p{p}") for p in sorted(part_ids))
    return f"sample_{ids}_" + re.sub(r"[^A-Za-z0-9+\-]", "_", names)


def scan_files(paths):
    """Return (Counter{frozenset: count}, {frozenset: first sample .txt path})."""
    combos, sample_file = Counter(), {}
    for path in paths:
        with open(path) as f:
            raw = {line.rsplit(None, 1)[-1] for line in f if line.strip()}
        parts = frozenset(int(float(s)) for s in raw)
        combos[parts] += 1
        sample_file.setdefault(parts, path)
    return combos, sample_file


def scan_category(synset_dir):
    return scan_files(
        os.path.join(synset_dir, f)
        for f in sorted(os.listdir(synset_dir)) if f.endswith(".txt"))


def build_histogram(cat, combos):
    total = sum(combos.values())
    nominal = len(SEG_CLASSES[cat])
    ordered = sorted(combos.items(), key=lambda x: x[1])  # smallest first -> top bar largest

    labels = [combo_label(ids) for ids, _ in ordered]
    counts = [cnt for _, cnt in ordered]

    fig = go.Figure()
    shown_sizes = set()
    for (ids, cnt), label in zip(ordered, labels):
        k = len(ids)
        fig.add_trace(go.Bar(
            y=[label], x=[cnt], orientation="h",
            name=f"{k}-part combination",
            legendgroup=f"{k}", legendrank=k,
            showlegend=k not in shown_sizes,
            marker=dict(color=ORDINAL_RAMP.get(k, "#0d366b"),
                        cornerradius=4),
            text=[f"{cnt:,} ({100 * cnt / total:.0f}%)"],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=11, color=TEXT2),
            hovertemplate=(f"{label}<br>%{{x:,}} objects "
                           f"({100 * cnt / total:.1f}%)<extra></extra>"),
        ))
        shown_sizes.add(k)

    fig.update_layout(
        title=dict(text=f"{cat} — part combinations  (nominal = {nominal}, "
                        f"{total:,} objects)",
                   font=dict(size=15, color=TEXT)),
        font={**FONT, "color": TEXT2},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        barmode="overlay",
        xaxis=dict(title="Objects", gridcolor=GRID, zerolinecolor=AXIS,
                   linecolor=AXIS, tickfont=dict(color=MUTED),
                   range=[0, max(counts) * 1.22]),
        yaxis=dict(categoryorder="array", categoryarray=labels,
                   tickfont=dict(size=11, color=TEXT2)),
        legend=dict(title=dict(text="Parts in combination", font=dict(size=11)),
                    orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(size=11), traceorder="grouped"),
        height=max(360, 170 + 34 * len(labels)), width=900,
        margin=dict(l=10, r=30, t=110, b=50),
    )
    return fig


def build_sample(cat, part_ids, count, total, path, max_points=None):
    data = np.loadtxt(path)
    if max_points and len(data) > max_points:
        idx = np.random.default_rng(0).choice(len(data), max_points, replace=False)
        data = data[idx]
    xyz, labels = data[:, :3], data[:, -1].astype(int)

    slots = {pid: i for i, pid in enumerate(SEG_CLASSES[cat])}
    fig = go.Figure()
    for pid in sorted(part_ids):
        m = labels == pid
        fig.add_trace(go.Scatter3d(
            x=xyz[m, 0], y=xyz[m, 1], z=xyz[m, 2],
            mode="markers",
            marker=dict(size=2.5, color=CATEGORICAL[slots.get(pid, 0)]),
            name=f"{PART_ID_NAMES.get(pid, f'p{pid}')} ({pid})",
            hovertemplate=(f"{PART_ID_NAMES.get(pid, f'p{pid}')} ({pid})"
                           "<extra></extra>"),
        ))

    obj_id = os.path.splitext(os.path.basename(path))[0]
    hidden_axis = dict(visible=False)
    fig.update_layout(
        title=dict(text=f"{cat} — {combo_label(part_ids)}<br>"
                        f"<sup>{count:,} objects ({100 * count / total:.1f}%) · "
                        f"sample {obj_id}</sup>",
                   font=dict(size=14, color=TEXT)),
        font={**FONT, "color": TEXT2},
        paper_bgcolor=SURFACE,
        scene=dict(xaxis=hidden_axis, yaxis=hidden_axis, zaxis=hidden_axis,
                   aspectmode="data", bgcolor=SURFACE),
        legend=dict(orientation="h", yanchor="bottom", y=0, x=0,
                    itemsizing="constant", font=dict(size=12)),
        width=900, height=700, margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def save_figure(fig, base):
    """Write <base>.html (interactive) and <base>.png (kaleido, 2x)."""
    fig.write_html(base + ".html", include_plotlyjs="cdn")
    fig.write_image(base + ".png", scale=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root",
        default="data/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    parser.add_argument("--out_dir", default="figures")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Subsample point clouds for the 3D plots")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Restrict to these categories (default: all)")
    args = parser.parse_args()

    with open(os.path.join(args.data_root, "synsetoffset2category.txt")) as f:
        cat2synset = dict(line.split() for line in f if line.strip())
    if args.categories:
        cat2synset = {c: s for c, s in cat2synset.items()
                      if c in args.categories}

    summary = []
    for cat in sorted(cat2synset):
        synset_dir = os.path.join(args.data_root, cat2synset[cat])
        combos, sample_file = scan_category(synset_dir)
        total = sum(combos.values())

        out_dir = os.path.join(args.out_dir, cat)
        os.makedirs(out_dir, exist_ok=True)

        save_figure(build_histogram(cat, combos),
                    os.path.join(out_dir, "combination_histogram"))
        for ids, cnt in sorted(combos.items(), key=lambda x: -x[1]):
            save_figure(build_sample(cat, ids, cnt, total, sample_file[ids],
                                     args.max_points),
                        os.path.join(out_dir, combo_filename(ids)))

        summary.append((cat, total, len(combos), out_dir))
        print(f"{cat:<12} {total:>6,} objects  {len(combos):>3} combinations "
              f"-> {out_dir}/")

    print(f"\n{'Category':<12} {'Objects':>8} {'Combos':>7}  Folder")
    for cat, total, n, out_dir in summary:
        print(f"{cat:<12} {total:>8,} {n:>7}  {out_dir}/")


if __name__ == "__main__":
    main()
