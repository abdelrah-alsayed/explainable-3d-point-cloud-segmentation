"""Cross-category numbers for the explorer's Overview tab, from the saved results only.

Writes data/overview.js (window.XAI_OVERVIEW). For every category, model, mask and score:
medians of mask / model / seed / budget / replacement-seed agreement (with quartiles),
AOPC gap and "SHAP beats random", additive R^2, groups for 80 %, mIoU, and study counts.
Same definitions as the Desktop preview (one-time/all_categories/make_all_categories.py).
Run: ~/miniconda3/envs/pointnet2/bin/python build_overview.py
"""
import csv
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
PS = HERE.parents[2]                                   # .../log/part_seg
OFFLINE = HERE.parent/"offline_metrics.csv"
CATS = ["Airplane", "Car", "Laptop", "Motorbike", "Rocket", "Skateboard"]
MODELS, MASKS, SCORES = ["single", "joint"], ["centroid", "retained_point"], ["iou", "logprob_mean"]


def root(cat, model):
    return PS/(f"full_parts_per_category/{cat}/seed_42/explain/full-exp" if model == "single"
               else f"full_parts_joint_model/explain/full-exp/{cat}")


def rho(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


def stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if not len(v):
        return None
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    return {"med": round(float(md), 4), "q1": round(float(q1), 4), "q3": round(float(q3), 4), "n": int(len(v))}


def aopc(c):
    c = np.asarray(c, float)
    return float(np.mean(c[0]-c))


def g80(v):
    a = np.sort(np.abs(v))[::-1]
    return int(np.searchsorted(np.cumsum(a)/a.sum(), 0.8)+1) if a.sum() > 0 else 0


def main():
    r2 = {}
    with open(OFFLINE) as fh:
        for r in csv.DictReader(fh):
            if r["cohort"] == "main":
                r2.setdefault((r["category"], r["model"], r["mask"], r["score"]), []).append(float(r["additive_r2"]))
    train = {}
    with open(PS/"full_parts_joint_model/sample_counts.csv") as fh:
        for r in csv.DictReader(fh):
            train[r["category"].strip()] = int(r["trainval_count"])
    out = {"categories": CATS, "train": {c: train[c] for c in CATS}, "rows": [], "cats": {}}
    for cat in CATS:
        cfg = json.loads((root(cat, "joint")/"manifest.json").read_text())["identity"]["config"]
        names = cfg["object_names"]
        seen = {names.index(n) for n in cfg["training_seen_objects"]}
        objs = [i for i in range(len(names)) if i not in seen]
        trial = cfg["trial_objects"]
        parts = [int(p) for p in np.load(root(cat, "joint")/"objects/obj0000.npz")["parts"]]
        P = len(parts)
        full = {(m, k, i): np.load(root(cat, m)/f"full/obj{i:04d}_{k}_m8192_s0_r0.npz") for m in MODELS for k in MASKS for i in objs}
        miou = {m: float(np.mean([np.load(root(cat, m)/f"objects/obj{i:04d}.npz")["score_iou"].mean() for i in objs])) for m in MODELS}
        tr = lambda m, k, i, key: np.load(root(cat, m)/f"trial/obj{i:04d}_{k}_{key}.npz")
        out["cats"][cat] = {"objects": len(names), "main": len(objs), "seen": len(seen), "trial": len(trial), "parts": P,
                            "exact": len(cfg["exact_objects"]), "miou": {m: round(miou[m], 4) for m in MODELS}}
        for m in MODELS:
            for k in MASKS:
                for s in SCORES:
                    ph = lambda d: d[f"phi_{s}"]
                    mask_a = [rho(ph(full[m, "centroid", i])[:, p], ph(full[m, "retained_point", i])[:, p]) for i in objs for p in range(P)]
                    model_a = [rho(ph(full["single", k, i])[:, p], ph(full["joint", k, i])[:, p]) for i in objs for p in range(P)]
                    seed_a = [rho(ph(tr(m, k, i, f"m8192_s{a}_r0"))[:, p], ph(tr(m, k, i, f"m8192_s{b}_r0"))[:, p])
                              for i in trial for p in range(P) for a, b in combinations(cfg["kernel_seeds"], 2)]
                    budget_a = [rho(ph(tr(m, k, i, f"m4096_s{a}_r0"))[:, p], ph(tr(m, k, i, f"m8192_s{a}_r0"))[:, p])
                                for i in trial for p in range(P) for a in cfg["kernel_seeds"]]
                    rep_a = ([rho(ph(tr(m, k, i, f"m8192_s0_r{a}"))[:, p], ph(tr(m, k, i, f"m8192_s0_r{b}"))[:, p])
                              for i in trial for p in range(P) for a, b in combinations(cfg["replacement_seeds"], 2)]
                             if k == "retained_point" else [])
                    gaps = []
                    for i in trial:
                        d = tr(m, k, i, "m8192_s0_r0")
                        for pid in parts:
                            key = f"removal_{s}_part{pid}_"
                            if key+"helpful_scores" in d.files and len(d[key+"helpful_scores"]) > 1:
                                gaps.append(aopc(d[key+"helpful_scores"]) - np.mean([aopc(x) for x in d[key+"random_scores"]]))
                    out["rows"].append({"category": cat, "model": m, "mask": k, "score": s,
                                        "mask_agreement": stats(mask_a), "model_agreement": stats(model_a),
                                        "seed_agreement": stats(seed_a), "budget_agreement": stats(budget_a),
                                        "replacement_agreement": stats(rep_a),
                                        "aopc_gap": stats(gaps), "beats": int(sum(g > 0 for g in gaps)), "beats_of": len(gaps),
                                        "additive_r2": stats(r2.get((cat, m, k, s), [])),
                                        "groups80": stats([g80(ph(full[m, k, i])[:, p]) for i in objs for p in range(P)])})
        print(cat, "done")
    js = "window.XAI_OVERVIEW=" + json.dumps(out, separators=(",", ":")) + ";\n"
    (HERE/"data/overview.js").write_text(js)
    print("wrote data/overview.js", round(len(js)/1e3, 1), "kB")


if __name__ == "__main__":
    main()
