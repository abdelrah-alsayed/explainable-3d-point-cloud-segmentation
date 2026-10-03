"""Build the data file of the training explorer from the saved training results only.

Reads, for every training run under log/part_seg (a folder with config.json and metrics_per_epoch.csv):
    config.json             settings of the run
    metrics_per_epoch.csv   loss, accuracy, mIoU and time of every epoch
    per_category_miou.csv   test mIoU per category of the saved (best) checkpoint
    sample_counts.csv       training and test shapes per category
    logs/*.txt              test mIoU per category of every epoch (only needed for runs with several categories)
and, for the 6 categories of the XAI study, the per-object test results of both models:
    .../explain/full-exp/report/prediction_details.json   IoU per object and part
    .../explain/full-exp/objects/objNNNN.npz              true and predicted part of every point (for the confusion matrices)
and, for the Dataset tab, every object of ShapeNetPart (data/shapenetcore_partanno_segmentation_benchmark_v0_normal):
    its split (train / val / test, chosen as in data_utils/ShapeNetDataLoader.py), its raw number of points
    and its number of distinct part labels
It writes data/train.js, which sets window.TRAIN_DATA. Nothing is trained or evaluated again.
Run: python3 build_data.py      (standard library only)
"""
import array
import ast
import csv
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PART_SEG = HERE.parent                      # .../log/part_seg
DATA = PART_SEG.parents[1]/"data"/"shapenetcore_partanno_segmentation_benchmark_v0_normal"
OUT = HERE/"data"/"train.js"

SEG_CLASSES = {'Airplane': [0, 1, 2, 3], 'Bag': [4, 5], 'Cap': [6, 7], 'Car': [8, 9, 10, 11], 'Chair': [12, 13, 14, 15],
               'Earphone': [16, 17, 18], 'Guitar': [19, 20, 21], 'Knife': [22, 23], 'Lamp': [24, 25, 26, 27],
               'Laptop': [28, 29], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Mug': [36, 37], 'Pistol': [38, 39, 40],
               'Rocket': [41, 42, 43], 'Skateboard': [44, 45, 46], 'Table': [47, 48, 49]}
PART_NAMES = {0: "Body", 1: "Wing", 2: "Tail", 3: "Engine", 4: "Handle", 5: "Body", 6: "Panels", 7: "Peak",
              8: "Roof", 9: "Hood", 10: "Wheel", 11: "Body", 12: "Back", 13: "Seat", 14: "Leg", 15: "Armrest",
              16: "Headband", 17: "Earcup", 18: "Cord", 19: "Head", 20: "Neck", 21: "Body", 22: "Blade", 23: "Handle",
              24: "Base", 25: "Shade", 26: "Canopy", 27: "Pole", 28: "Keyboard", 29: "Screen", 30: "Gas tank", 31: "Seat",
              32: "Wheel", 33: "Handle", 34: "Light", 35: "Frame", 36: "Handle", 37: "Cup", 41: "Body", 42: "Fin", 43: "Nose",
              44: "Wheel", 45: "Deck", 46: "Belt", 47: "Top", 48: "Leg", 49: "Support"}     # as in plot_part_combinations.py
XAI_CATEGORIES = ["Airplane", "Car", "Laptop", "Motorbike", "Rocket", "Skateboard"]

# the experiments, in the order the page shows them; text in simple English
EXPERIMENTS = [
    ("full_parts_per_category", "Single models", "One model per category, 14 categories. Only objects with all parts of their category. Seed 42. Used in the XAI study."),
    ("full_parts_joint_model", "Joint model", "One model for 14 categories together. Only objects with all parts. Seed 42. Used in the XAI study."),
    ("full-dataset", "Full dataset", "One model for all 15 categories and all objects. 251 epochs. An earlier run."),
    ("mechanical_joint", "Mechanical joint", "One model for 4 mechanical categories (Airplane, Car, Motorbike, Rocket), all objects."),
    ("parts2_cap312", "2 parts, 312 shapes", "Only objects with exactly 2 parts. 312 training and 80 test shapes per category. 10 seeds."),
    ("parts3_cap626", "3 parts, 626 shapes", "Only objects with exactly 3 parts. 626 training and 159 test shapes per category. 10 seeds."),
    ("parts4_cap596", "4 parts, 596 shapes", "Only objects with exactly 4 parts. 596 training and 118 test shapes per category. 10 seeds."),
    ("parts4_cap740", "740 shapes", "740 training shapes per category, no part filter, all test shapes. 10 seeds."),
]
COLS = ["train_loss", "train_accuracy", "test_accuracy", "test_class_avg_iou", "test_instance_avg_iou", "learning_rate", "epoch_time_seconds"]


def read_csv(path):
    """The csv files are written with padding spaces in some runs: strip every key and value."""
    with open(path, newline="") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def epoch_logs(path, cats):
    """test mIoU of every category at every epoch, from the text log (the last block of an epoch wins)"""
    txt = path.read_text(errors="replace")
    blocks = re.split(r"Epoch (\d+) \(\d+/\d+\):", txt)
    per = {}
    for k in range(1, len(blocks), 2):
        per[int(blocks[k])] = {m.group(1): float(m.group(2)) for m in re.finditer(r"eval mIoU of (\S+)\s+([0-9.]+)", blocks[k + 1])}
    n = max(per) if per else 0
    return {c: [round(per.get(e, {}).get(c, float("nan")), 5) for e in range(1, n + 1)] for c in cats}


def run_record(d, exp):
    cfg = json.load(open(d/"config.json"))
    a = cfg.get("args", {})
    rows = read_csv(d/"metrics_per_epoch.csv")
    series = {c: [round(float(r[c]), 6 if c == "learning_rate" else 5) for r in rows] for c in COLS}
    epochs = [int(float(r["epoch"])) for r in rows]
    assert epochs == list(range(1, len(epochs) + 1)), d
    cats = cfg.get("categories") or a.get("categories")
    counts = {r["category"]: [int(r["trainval_count"]), int(r["test_count"])] for r in read_csv(d/"sample_counts.csv")}
    best_cat = {r["category"]: round(float(r["miou"]), 5) for r in read_csv(d/"per_category_miou.csv")}
    inst = series["test_instance_avg_iou"]
    best = max(i for i, v in enumerate(inst) if v == max(inst))        # the code saves on ">=": the last maximum
    rec = {
        "id": str(d.relative_to(PART_SEG)), "exp": exp, "categories": sorted(cats),
        "seed": cfg.get("seed", a.get("seed")), "epochs": len(epochs), "best_epoch": best + 1,
        "counts": counts, "best_cat": best_cat, "series": series,
        "config": {k: a.get(k) for k in ["batch_size", "epoch", "learning_rate", "optimizer", "decay_rate", "npoint", "step_size", "lr_decay",
                                         "samples_per_category", "test_samples_per_category", "full_parts_only", "num_parts", "subset_seed"]},
        "env": {"gpu": cfg.get("gpu_name"), "torch": cfg.get("torch_version"), "date": (cfg.get("timestamp") or "")[:10]},
        "complete": (d/"RUN_COMPLETE").exists() or exp in ("full-dataset", "mechanical_joint", "full_parts_joint_model"),
    }
    if len(cats) > 1:
        rec["cat_series"] = epoch_logs(next((d/"logs").glob("*.txt")), sorted(cats))
        # the saved checkpoint = the best epoch; its per-category values must match per_category_miou.csv
        for c in cats:
            assert abs(rec["cat_series"][c][best] - best_cat[c]) < 1e-4, (d, c, rec["cat_series"][c][best], best_cat[c])
    else:
        assert abs(inst[best] - best_cat[cats[0]]) < 2e-3, (d, inst[best], best_cat)
    return rec


def load_npz(path, keys):
    """the integer arrays of an .npz file, with the standard library only (no numpy needed)"""
    out = {}
    with zipfile.ZipFile(path) as z:
        for k in keys:
            b = z.read(k + ".npy")
            major = b[6]
            hl = int.from_bytes(b[8:10] if major == 1 else b[8:12], "little")
            start = 10 if major == 1 else 12
            head = ast.literal_eval(b[start:start + hl].decode("latin1"))
            assert head["descr"] == "<i8" and not head["fortran_order"], (path, k, head)
            a = array.array("q"); a.frombytes(b[start + hl:])
            out[k] = list(a)
    return out


def part_iou(truth, pred, part):
    """IoU of one part, with the rule of the training code: a part that is in neither truth nor prediction counts as 1"""
    inter = sum(1 for t, q in zip(truth, pred) if t == part and q == part)
    union = sum(1 for t, q in zip(truth, pred) if t == part or q == part)
    return 1.0 if union == 0 else inter / union


def test_results():
    out = {}
    for cat in XAI_CATEGORIES:
        rec = {"parts": [{"id": p, "name": PART_NAMES.get(p, str(p))} for p in SEG_CLASSES[cat]]}
        for model, root in [("single", PART_SEG/f"full_parts_per_category/{cat}/seed_42/explain/full-exp"),
                            ("joint", PART_SEG/f"full_parts_joint_model/explain/full-exp/{cat}")]:
            det = json.load(open(root/"report"/"prediction_details.json"))
            seen = root/"report"/"training_seen_prediction_details.json"
            rec[model] = [{"o": x["object"], "miou": round(x["miou"], 5), "parts": [round(x["part_iou"][str(p)], 5) for p in SEG_CLASSES[cat]]}
                          for x in det if x["cohort"] == "main"]
            rec[model + "_seen"] = len(json.load(open(seen))) if seen.exists() else 0
            # confusion matrix over all points of the main objects: rows = true part, columns = predicted part
            parts = SEG_CLASSES[cat]
            conf = [[0] * len(parts) for _ in parts]
            for x in det:
                if x["cohort"] != "main":
                    continue
                d = load_npz(root/"objects"/f"obj{x['object']:04d}.npz", ["truth", "prediction"])
                t, q = d["truth"], d["prediction"]
                assert set(t) <= set(parts) and set(q) <= set(parts), (cat, model, x["object"])
                for a, b in zip(t, q):
                    conf[parts.index(a)][parts.index(b)] += 1
                for p in parts:      # the saved points reproduce the saved IoU of the report
                    assert abs(part_iou(t, q, p) - x["part_iou"][str(p)]) < 1e-6, (cat, model, x["object"], p)
            rec[model + "_conf"] = conf
        assert [x["o"] for x in rec["single"]] == [x["o"] for x in rec["joint"]], cat
        out[cat] = rec
    return out


def dataset_stats():
    """split, raw point count and part combination of every object, per category (same rules as the data loader).
    A few files are listed in two split lists; the loader then uses them in both splits, so they get two letters.
    combo: the parts that occur in the object, as bits over the category's parts (bit j = SEG_CLASSES[name][j])."""
    cats = [line.split() for line in (DATA/"synsetoffset2category.txt").read_text().splitlines() if line.strip()]
    ids = {s: {d.split("/")[2] for d in json.load(open(DATA/"train_test_split"/f"shuffled_{s}_file_list.json"))} for s in ("train", "val", "test")}
    out = []
    for name, folder in cats:
        split, points, combos, slot = [], [], [], {p: j for j, p in enumerate(SEG_CLASSES[name])}
        for fn in sorted((DATA/folder).iterdir()):
            token = fn.name[:-4]
            s = "".join(letter for sp, letter in (("train", "t"), ("val", "v"), ("test", "e")) if token in ids[sp])    # e = test
            if not s:
                continue
            lines = [l for l in fn.read_bytes().split(b"\n") if l.strip()]
            split.append(s)
            points.append(len(lines))
            labels = {int(float(l.rsplit(None, 1)[-1])) for l in lines}      # last column = part label
            assert labels <= set(slot), (fn, labels)
            combos.append(sum(1 << slot[p] for p in labels))
        out.append({"name": name, "folder": folder, "nominal": len(SEG_CLASSES[name]), "part_names": [PART_NAMES[p] for p in SEG_CLASSES[name]],
                    "split": split, "points": points, "combo": combos})
    used = {f for _, f in cats}
    unused = [{"folder": d.name, "files": sum(1 for _ in d.glob("*.txt"))} for d in sorted(DATA.iterdir()) if d.is_dir() and d.name.isdigit() and d.name not in used]
    return {"categories": out, "unused_folders": unused}


def main():
    runs = []
    for exp, _, _ in EXPERIMENTS:
        for cfg in sorted((PART_SEG/exp).rglob("config.json")):
            if "explain" in cfg.parts or not (cfg.parent/"metrics_per_epoch.csv").exists():
                continue
            runs.append(run_record(cfg.parent, exp))
    data = {
        "experiments": [{"id": e, "label": l, "about": t} for e, l, t in EXPERIMENTS],
        "parts": {c: len(v) for c, v in SEG_CLASSES.items()},
        "runs": runs,
        "test": test_results(),
        "dataset": dataset_stats(),
    }
    OUT.parent.mkdir(exist_ok=True)
    txt = json.dumps(data, separators=(",", ":")).replace("NaN", "null")
    OUT.write_text("window.TRAIN_DATA = " + txt + ";\n")
    print(f"{len(runs)} runs, {OUT.stat().st_size / 1e6:.2f} MB -> {OUT}")


if __name__ == "__main__":
    main()
