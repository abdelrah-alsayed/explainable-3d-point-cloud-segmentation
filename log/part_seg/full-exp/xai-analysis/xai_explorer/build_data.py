"""Build the data files of the XAI explorer from the raw saved results only.

Reads, for each category and model:
    objects/objNNNN.npz   points, true labels, prediction, part scores
    exact/  trial/  full/ SHAP values, v_empty / v_full, group labels, removal curves
    offline_metrics.csv   additive R^2 (full stage only)
and writes data/<Category>.js, which sets window.XAI_DATA[<Category>].
Arrays are stored as base64: points int16 (x 32767), labels uint8, SHAP values float32.
Run: ~/miniconda3/envs/pointnet2/bin/python build_data.py
"""
import base64
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PART_SEG = HERE.parents[2]           # .../log/part_seg
OUT = HERE/"data"
sys.path.insert(0, str(PART_SEG.parents[1]))   # the thesis folder, for the study's own masking code
from xai.masking import apply_mask            # noqa: E402
CATEGORIES = ["Airplane", "Car", "Laptop", "Motorbike", "Rocket", "Skateboard"]
MODELS = {"single": "Single model (one category)", "joint": "Joint model (14 categories)"}
MASKS = ["centroid", "retained_point"]
SCORES = ["iou", "logprob_mean"]
PART_NAMES = {0: "Body", 1: "Wing", 2: "Tail", 3: "Engine", 8: "Roof", 9: "Hood", 10: "Wheel",
              11: "Body", 28: "Keyboard", 29: "Screen", 30: "Gas tank", 31: "Seat", 32: "Wheel",
              33: "Handle", 34: "Light", 35: "Frame", 41: "Body", 42: "Fin", 43: "Nose",
              44: "Wheel", 45: "Deck", 46: "Belt"}


def root(cat, model):
    if model == "single":
        return PART_SEG/f"full_parts_per_category/{cat}/seed_42/explain/full-exp"
    return PART_SEG/f"full_parts_joint_model/explain/full-exp/{cat}"


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def rnd(a, n=5):
    return np.round(np.asarray(a, dtype=float), n).tolist()


def shap_pair(d, prefix="phi_"):
    return {s: b64(d[prefix+s], "<f4") for s in SCORES}


def to_index(labels, parts):
    lut = np.full(256, 255, dtype=np.uint8)
    for i, p in enumerate(parts):
        lut[p] = i
    return lut[labels]


def coalition_rng(seed, row):
    """The study's per-coalition random generator (xai/game.py, RegionGame._coalition_rng)."""
    bits = np.packbits(np.asarray(row, dtype=np.uint8)).tobytes()
    digest = hashlib.sha256(str(seed).encode()+b"|"+bits).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def preview_donors(pts, groups, seed):
    """Masked-input preview: the odd-numbered groups are hidden, the even-numbered ones stay.
    Returns, for each hidden point row (in row order), the visible row it copies under the
    retained-point mask. Checked against the study's apply_mask with the same generator."""
    K = int(groups.max())+1
    row = np.arange(K) % 2 == 0
    visible = row[groups]
    donors = coalition_rng(seed, row).choice(np.flatnonzero(visible), size=int((~visible).sum()), replace=True)
    out = apply_mask(pts, groups, row, "retained_point", coalition_rng(seed, row))
    ref = np.asarray(pts, dtype=np.float32)
    assert np.array_equal(out[~visible], ref[donors]) and np.array_equal(out[visible], ref[visible])
    origin = apply_mask(pts, groups, row, "centroid", None)
    assert np.all(origin[~visible] == 0) and np.array_equal(origin[visible], ref[visible])
    return b64(donors, "<u2")


def load_r2():
    table = {}
    with open(HERE.parent/"offline_metrics.csv") as fh:
        for r in csv.DictReader(fh):
            key = (r["category"], r["model"], int(r["object"]), r["mask"], r["score"])
            table.setdefault(key, {})[int(r["part_id"])] = round(float(r["additive_r2"]), 4)
    return table


def build(cat, r2):
    cfg = json.loads((root(cat, "joint")/"manifest.json").read_text())["identity"]["config"]
    for m in MODELS:  # both studies must describe the same objects
        other = json.loads((root(cat, m)/"manifest.json").read_text())["identity"]["config"]
        assert other["object_names"] == cfg["object_names"], (cat, m)
    names = cfg["object_names"]
    seen = {names.index(n) for n in cfg["training_seen_objects"]}
    trial_objs, exact_objs = list(cfg["trial_objects"]), list(cfg["exact_objects"])
    first = np.load(root(cat, "joint")/"objects/obj0000.npz")
    parts = [int(p) for p in first["parts"]]
    data = {"category": cat, "parts": [{"id": p, "name": PART_NAMES.get(p, str(p))} for p in parts],
            "groups": cfg["groups"], "exactGroups": cfg["exact_groups"], "trialObjects": trial_objs,
            "exactObjects": exact_objs, "exactBudgets": cfg["exact_budgets"], "trialBudgets": cfg["trial_budgets"],
            "shapSeeds": cfg["kernel_seeds"], "replacementSeeds": cfg["replacement_seeds"],
            "models": {m: {"label": MODELS[m], "miou": None} for m in MODELS}, "objects": [],
            "full": {m: {k: {} for k in MASKS} for m in MODELS},
            "trial": {m: {k: {} for k in MASKS} for m in MODELS},
            "removal": {m: {k: {} for k in MASKS} for m in MODELS},
            "exact": {m: {k: {} for k in MASKS} for m in MODELS}}
    n_obj = len(names)
    for i in range(n_obj):
        base = np.load(root(cat, "joint")/f"objects/obj{i:04d}.npz")
        pts = base["points"].astype(np.float64)
        assert np.abs(pts).max() <= 1.0
        g_full = np.load(root(cat, "joint")/f"full/obj{i:04d}_centroid_m8192_s0_r0.npz")["region_labels"]
        obj = {"i": i, "seen": i in seen, "trial": i in trial_objs, "exact": i in exact_objs,
               "points": b64(np.round(pts*32767), "<i2"), "truth": b64(to_index(base["truth"], parts), "u1"),
               "groups": b64(g_full, "u1"), "models": {},
               # retained-point donors for the preview; replacement seeds 0-2 exist for trial objects only
               "preview": {str(r): preview_donors(base["points"], g_full, r)
                           for r in (cfg["replacement_seeds"] if i in trial_objs else [0])}}
        for m in MODELS:
            o = np.load(root(cat, m)/f"objects/obj{i:04d}.npz")
            assert np.array_equal(o["points"], base["points"]) and np.array_equal(o["truth"], base["truth"])
            obj["models"][m] = {"pred": b64(to_index(o["prediction"], parts), "u1"),
                                "iou": rnd(o["score_iou"], 4), "logprob": rnd(o["score_logprob_mean"], 4)}
            for mask in MASKS:
                d = np.load(root(cat, m)/f"full/obj{i:04d}_{mask}_m8192_s0_r0.npz")
                assert np.array_equal(d["region_labels"], g_full)
                data["full"][m][mask][i] = {**shap_pair(d), "vEmpty": rnd(d["v_empty"]), "vFull": rnd(d["v_full"]),
                                            "r2": {s: [r2.get((cat, m, i, mask, s), {}).get(p) for p in parts]
                                                   for s in SCORES}}
        data["objects"].append(obj)
    for m in MODELS:
        main = [o["models"][m]["iou"] for o in data["objects"] if not o["seen"]]
        data["models"][m]["miou"] = round(float(np.mean([np.mean(v) for v in main])), 4)
        for mask in MASKS:
            for i in trial_objs:
                runs = {}
                for b in cfg["trial_budgets"]:
                    for s in cfg["kernel_seeds"]:
                        reps = cfg["replacement_seeds"] if (mask == "retained_point" and b == max(cfg["trial_budgets"])
                                                            and s == cfg["kernel_seeds"][0]) else [0]
                        for rs in reps:
                            f = root(cat, m)/f"trial/obj{i:04d}_{mask}_m{b}_s{s}_r{rs}.npz"
                            d = np.load(f)
                            runs[f"{b}_{s}_{rs}"] = {**shap_pair(d), "vEmpty": rnd(d["v_empty"]), "vFull": rnd(d["v_full"])}
                            if (b, s, rs) == (max(cfg["trial_budgets"]), cfg["kernel_seeds"][0], 0):
                                rem = {}
                                for sc in SCORES:
                                    rem[sc] = {}
                                    for p in parts:
                                        k = f"removal_{sc}_part{p}_"
                                        if k+"helpful_scores" in d.files:
                                            rem[sc][p] = {"helpful": rnd(d[k+"helpful_scores"], 4),
                                                          "random": rnd(d[k+"random_scores"], 4),
                                                          "helpfulFrac": rnd(d[k+"helpful_hidden_point_fraction"], 4),
                                                          "randomFrac": rnd(d[k+"random_hidden_point_fraction"], 4),
                                                          "order": d[k+"positive_order"].astype(int).tolist()}
                                            # same ranking, groups hidden with the other mask
                                            other = [x for x in MASKS if x != mask][0]
                                            c = f"cross_removal_{other}_{sc}_part{p}_"
                                            if c+"helpful_scores" in d.files:
                                                rem[sc][p]["cross"] = {
                                                    "mask": other, "helpful": rnd(d[c+"helpful_scores"], 4),
                                                    "random": rnd(d[c+"random_scores"], 4),
                                                    "helpfulFrac": rnd(d[c+"helpful_hidden_point_fraction"], 4),
                                                    "randomFrac": rnd(d[c+"random_hidden_point_fraction"], 4)}
                                data["removal"][m][mask][i] = rem
                data["trial"][m][mask][i] = runs
            for i in exact_objs:
                d = np.load(root(cat, m)/f"exact/obj{i:04d}_{mask}_K8.npz")
                runs = {f"{b}_{s}": {sc: b64(d[f"phi_{sc}_m{b}_s{s}"], "<f4") for sc in SCORES}
                        for b in cfg["exact_budgets"] for s in cfg["kernel_seeds"]}
                obj_pts = np.load(root(cat, m)/f"objects/obj{i:04d}.npz")["points"]
                data["exact"][m][mask][i] = {"groups": b64(d["region_labels"], "u1"),
                                             "preview": preview_donors(obj_pts, d["region_labels"], 0),
                                             "exact": {sc: b64(d[f"exact_phi_{sc}"], "<f4") for sc in SCORES},
                                             "runs": runs,
                                             "vEmpty": [rnd(d[f"table_{sc}"][0]) for sc in SCORES],
                                             "vFull": [rnd(d[f"table_{sc}"][-1]) for sc in SCORES]}
    return data


def main():
    OUT.mkdir(exist_ok=True)
    r2 = load_r2()
    index = []
    for cat in CATEGORIES:
        data = build(cat, r2)
        js = "window.XAI_DATA=window.XAI_DATA||{};window.XAI_DATA[" + json.dumps(cat) + "]=" + \
             json.dumps(data, separators=(",", ":")) + ";\n"
        (OUT/f"{cat}.js").write_text(js)
        index.append({"name": cat, "objects": len(data["objects"]), "parts": [p["name"] for p in data["parts"]]})
        print(f"{cat}: {len(data['objects'])} objects, {len(js)/1e6:.1f} MB")
    (OUT/"index.js").write_text("window.XAI_INDEX=" + json.dumps(index) + ";\n")


if __name__ == "__main__":
    main()
