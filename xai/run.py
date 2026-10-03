"""Small, explicit stages: plan, check, smoke, exact, trial, full, report.

Run from the thesis folder: python -m xai.run --help
The full-study budget is chosen by the researcher after reviewing the trial.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch

from xai.config import ROOT, StudyConfig
from xai.data import DATA_ROOT, load_for_config, object_paths
from xai.estimators.exact import enumerate_all
from xai.estimators.kernel import kernel_shap
from xai.game import RegionGame, BatchScorer, SCORES
from xai.masking import MASK_OPS
from xai.metrics import helpful_removal
from xai.model_io import load_model, category_onehot, SEG_CLASSES
from xai.provenance import run_provenance
from xai.regions import build_regions
from xai.storage import (sha256_file, atomic_json, save_npz, load_npz,
                         result_id, ensure_manifest)
from xai.validation import known_answer_checks, compare_reference
from xai.integrity import check_result, check_prediction, check_exact

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["plan","check","smoke","exact","trial","full","report","replay"])
    p.add_argument("--config", type=Path, help="Defaults to the agreed Airplane study")
    p.add_argument("--budget", type=int, help="Required for full, chosen after the trial")
    p.add_argument("--replacement-seed", type=int, default=0, help="Fixed full-study replacement seed")
    p.add_argument("--kernel-seed", type=int, default=0, help="Fixed full-study sampling seed")
    p.add_argument("--tables-dir", type=Path, help="For replay: existing trained-model exact tables")
    p.add_argument("--reference-groups", type=int, default=16, help="Group count of existing replay tables")
    return p.parse_args()

def planned_jobs(cfg, stage, budget=None, kernel_seed=0, replacement_seed=0):
    if stage == "trial":
        jobs = [(i, op, m, s, cfg.replacement_seeds[0])
                for i in cfg.trial_objects for op in MASK_OPS
                for m in cfg.trial_budgets for s in cfg.kernel_seeds]
        jobs += [(i, "retained_point", max(cfg.trial_budgets), cfg.kernel_seeds[0], r)
                 for i in cfg.trial_objects for r in cfg.replacement_seeds[1:]]
        return jobs
    if stage == "full":
        if budget is None or budget < cfg.groups+1:
            raise ValueError("Choose --budget after reviewing the trial; no automatic cutoff is used")
        return [(i, op, budget, kernel_seed, replacement_seed) for i in range(cfg.test_count)
                for op in MASK_OPS]
    if stage == "smoke":
        return [(cfg.trial_objects[0], op, max(cfg.groups+1, 64), 0, 0) for op in MASK_OPS]
    raise ValueError(stage)

def plan(cfg):
    jobs = planned_jobs(cfg, "trial")
    return {
        "category": cfg.category, "model": str(cfg.resolve(cfg.log_dir)),
        "final_objects": cfg.test_count, "groups": cfg.groups,
        "masking_methods": list(MASK_OPS), "scores": list(SCORES),
        "trial_objects": cfg.trial_objects, "trial_runs": len(jobs),
        "trial_clouds_upper_bound_before_cache_and_removal": sum(j[2]+2 for j in jobs),
        "exact_reference": {"groups": cfg.exact_groups, "objects": cfg.exact_objects,
                            "clouds": len(cfg.exact_objects)*len(MASK_OPS)*2**cfg.exact_groups},
        "full_budget": "Choose after viewing trial results",
        "output": str(cfg.resolve(cfg.output_dir)),
    }

def setup_manifest(cfg):
    paths = object_paths(cfg)
    checkpoint = cfg.resolve(cfg.log_dir)/"checkpoints"/"best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    # Hash actual computation sources, not report layout or legacy summaries.
    code_files = [ROOT/"xai"/name for name in (
        "run.py","config.py","data.py","game.py","masking.py","metrics.py","regions.py",
        "model_io.py","storage.py","validation.py","integrity.py","estimators/kernel.py","estimators/exact.py")]
    code_files += list((ROOT/"models").glob("*.py"))
    code_files += [ROOT/"data_utils"/"ShapeNetDataLoader.py"]
    identity = {
        "schema": 2, "config": cfg.as_dict(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_files": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(code_files)},
        "dependencies": {n: importlib.metadata.version(n) for n in
                         ("shap","numpy","scipy","torch")},
        "objects": [{"index": i, "name": p.name, "sha256": sha256_file(p)}
                    for i, p in enumerate(paths)],
        "category_map_sha256": sha256_file(DATA_ROOT/"synsetoffset2category.txt"),
        "split_files": {p.name: sha256_file(p) for p in
                        sorted((DATA_ROOT/"train_test_split").glob("*.json"))},
        "fps": "initial point index 0 for every cloud",
        "scores": list(SCORES), "logprob_mode": cfg.logprob_mode, "masking_methods": list(MASK_OPS),
        "cohorts": {p.name: cfg.cohort(i) for i,p in enumerate(paths)},
    }
    return ensure_manifest(cfg.resolve(cfg.output_dir)/"manifest.json", identity,
                           {"provenance": run_provenance(cfg.resolve(cfg.log_dir)),
                            "notes": ["Masking changes xyz; original point labels and row indices stay fixed.",
                                      f"Mean log-probability uses original true-part rows; mode={cfg.logprob_mode}.",
                                      "IoU predictions are restricted to the category label set.",
                                      "Retained-point has an origin fallback when all groups are hidden.",
                                      "Trial examples were selected before the new SHAP runs.",
                                      "No fixed scientific pass/fail gates."]})

def new_scorer(cfg):
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    model = load_model(cfg.resolve(cfg.log_dir), device=torch.device(device))
    return BatchScorer(model, device, category_onehot(cfg.category), cfg.batch_size)

def object_game(cfg, index, K, op, seed, scorer):
    obj = load_for_config(cfg, index)
    labels = build_regions(obj.pts, K, cfg.region_seed)
    # Always compare predictions over the category's complete label set.
    parts = list(SEG_CLASSES[cfg.category])
    if any(not np.any(obj.seg == p) for p in parts):
        raise ValueError(f"Object {index} lacks a target part after sampling. "
                         "Mean log-probability is undefined; revise the explicit object/point policy.")
    return RegionGame(obj, labels, scorer, op, parts, SCORES, mask_seed=seed,
                      chunk_size=cfg.score_chunk_size, logprob_mode=cfg.logprob_mode)

def make_spec(stage, index, K, op, budget=None, kernel_seed=0, replacement_seed=0):
    return {"stage": stage, "object": index, "groups": K, "mask_op": op,
            "budget": budget, "kernel_seed": kernel_seed, "replacement_seed": replacement_seed}

def save_prediction(cfg, index, game, manifest):
    path = cfg.resolve(cfg.output_dir)/"objects"/f"obj{index:04d}.npz"
    spec = {"object": index, "kind": "unmasked prediction"}
    rid = result_id(manifest["study_id"], spec)
    if path.exists():
        check_prediction(path, cfg, manifest, index)
        return
    log_probs = game.scorer.score(game.obj.pts[None])[0]
    prediction = np.asarray(game.target_parts)[log_probs[:, game.target_parts].argmax(axis=1)]
    values = game._reduce_single(log_probs)
    save_npz(path, {"points": game.obj.pts, "truth": game.obj.seg, "prediction": prediction,
                    "parts": game.target_parts, **{f"score_{s}": v for s,v in values.items()}},
             {"result_id": rid, "source_path": game.obj.source_path, **spec})

def run_sampled(cfg, stage, manifest, budget=None, kernel_seed=0, replacement_seed=0):
    jobs = planned_jobs(cfg, stage, budget, kernel_seed, replacement_seed)
    root = cfg.resolve(cfg.output_dir)/stage
    if stage == "full":
        settings = {"budget": budget, "kernel_seed": kernel_seed,
                    "replacement_seed": replacement_seed}
        selection_path = root/"settings.json"
        if selection_path.exists() and json.loads(selection_path.read_text()) != settings:
            raise ValueError("A different full-study setting already exists. Use a new output_dir.")
        atomic_json(selection_path, settings)
    scorer = None
    # Grouping loops reuse reduced scores across budgets/seeds within each game.
    jobs = sorted(jobs, key=lambda j: (j[0], j[1], j[4], j[2], j[3]))
    current, game = None, None
    for number, (index, op, m, seed, rseed) in enumerate(jobs, 1):
        spec = make_spec(stage, index, cfg.groups, op, m, seed, rseed)
        rid = result_id(manifest["study_id"], spec)
        path = root/f"obj{index:04d}_{op}_m{m}_s{seed}_r{rseed}.npz"
        if path.exists():
            check_result(path, cfg, manifest, spec)
            print(f"[{number}/{len(jobs)}] Already complete: {path.name}", flush=True)
            continue
        if scorer is None:
            scorer = new_scorer(cfg)
        key = (index, op, rseed)
        if key != current:
            game = object_game(cfg, index, cfg.groups, op, rseed, scorer)
            current = key
            save_prediction(cfg, index, game, manifest)
        started = time.monotonic()
        phi, diag = kernel_shap(game, cfg.groups, m, seed)
        arrays = {f"phi_{s}": v for s,v in phi.items()}
        arrays.update(diag)
        arrays["region_labels"] = game.region_labels
        # Main removal check is at the largest trial budget and first seeds.
        # Full runs repeat it only on the same preselected trial objects.
        removal = index in cfg.trial_objects and (
            stage == "full" or stage == "smoke" or
            (m == max(cfg.trial_budgets) and seed == cfg.kernel_seeds[0]
             and rseed == cfg.replacement_seeds[0]))
        if removal:
            evaluators = {op: game}
            if cfg.cross_mask_removal:
                for evaluation_op in MASK_OPS:
                    if evaluation_op != op:
                        evaluators[evaluation_op] = object_game(cfg,index,cfg.groups,evaluation_op,rseed,scorer)
            for score in SCORES:
                for j, part in enumerate(game.target_parts):
                    values = helpful_removal(game, phi[score][:,j], score, j,
                                             cfg.random_removal_orders, seed=0)
                    arrays.update({f"removal_{score}_part{part}_{k}": v for k,v in values.items()})
                    if cfg.cross_mask_removal:
                        for evaluation_op, evaluation_game in evaluators.items():
                            cross = values if evaluation_op == op else helpful_removal(
                                evaluation_game, phi[score][:,j], score, j, cfg.random_removal_orders, seed=0)
                            arrays.update({f"cross_removal_{evaluation_op}_{score}_part{part}_{k}": v
                                           for k,v in cross.items()})
        meta = {"result_id": rid, "study_id": manifest["study_id"], **spec,
                "parts": game.target_parts, "scores": list(SCORES), "logprob_mode": cfg.logprob_mode,
                "cohort": cfg.cohort(index),
                "source_path": game.obj.source_path, "estimator": "shap.KernelExplainer",
                "shap_version": importlib.metadata.version("shap"),
                "l1_reg": 0.0, "link": "identity",
                "seconds": time.monotonic()-started, "actual_combinations": len(diag["coalitions"]),
                "batch_size_used": scorer.batch_size}
        save_npz(path, arrays, meta)
        print(f"[{number}/{len(jobs)}] Saved {path.name} ({meta['seconds']:.1f}s)", flush=True)

def run_exact(cfg, manifest, smoke=False):
    scorer = None
    indices = cfg.exact_objects[:1] if smoke else cfg.exact_objects
    K = min(cfg.exact_groups, 4) if smoke else cfg.exact_groups
    budgets = [2**K-2] if smoke else sorted(set(cfg.exact_budgets+[2**K-2]))
    stage = "smoke_exact" if smoke else "exact"
    for index in indices:
        for op in MASK_OPS:
            spec = make_spec(stage, index, K, op, replacement_seed=cfg.replacement_seeds[0])
            spec["budgets"], spec["seeds"] = budgets, cfg.kernel_seeds
            rid = result_id(manifest["study_id"], spec)
            path = cfg.resolve(cfg.output_dir)/stage/f"obj{index:04d}_{op}_K{K}.npz"
            if path.exists():
                check_exact(path, cfg, manifest, spec)
                print(f"Already complete: {path.name}", flush=True)
                continue
            if scorer is None:
                scorer = new_scorer(cfg)
            game = object_game(cfg, index, K, op, cfg.replacement_seeds[0], scorer)
            tables = enumerate_all(game, K, cfg.score_chunk_size)
            rows, arrays = compare_reference(tables, game.target_parts, K, budgets, cfg.kernel_seeds)
            arrays.update({f"table_{s}": v for s,v in tables.items()})
            arrays["region_labels"] = game.region_labels
            save_npz(path, arrays, {"result_id": rid, "study_id": manifest["study_id"], **spec,
                     "parts": game.target_parts, "comparisons": rows, "source": "live model",
                     "logprob_mode": cfg.logprob_mode})
            print(f"Exact reference saved: {path.name}", flush=True)

def replay(cfg, manifest, tables_dir, K):
    if tables_dir is None:
        raise ValueError("replay requires --tables-dir")
    if not 2 <= K <= 20:
        raise ValueError("Replay supports 2–20 groups")
    for index in cfg.exact_objects:
        labels = build_regions(load_for_config(cfg, index).pts, K, cfg.region_seed)
        for op in MASK_OPS:
            source = tables_dir/f"{cfg.category}_K{K}_{op}_trained_obj{index:04d}.npz"
            arrays, meta = load_npz(source)
            expected = {"category": cfg.category, "obj_index": index, "K": K, "mask_op": op,
                        "model_variant": "trained", "target_parts": list(SEG_CLASSES[cfg.category]),
                        "npoints": cfg.npoints, "sample_seed": cfg.sample_seed,
                        "region_seed": cfg.region_seed, "mask_seed": cfg.replacement_seeds[0],
                        "deterministic_fps": True}
            for key, value in expected.items():
                if meta.get(key) != value:
                    raise ValueError(f"Replay metadata differs: {source}: {key}")
            if meta.get("provenance", {}).get("checkpoint_sha256") != manifest["identity"]["checkpoint_sha256"]:
                raise ValueError(f"Replay model does not match: {source}")
            if Path(meta["source_path"]).name != manifest["identity"]["objects"][index]["name"]:
                raise ValueError(f"Replay object does not match: {source}")
            np.testing.assert_array_equal(labels, arrays["region_labels"])
            tables = {s: arrays[f"vf_{s}"] for s in SCORES}
            budgets = sorted(set(min(m, 2**K-2) for m in cfg.trial_budgets))
            rows, values = compare_reference(tables, expected["target_parts"], K, budgets, cfg.kernel_seeds)
            spec = {"stage": "replay", "object": index, "groups": K, "mask_op": op,
                    "table_sha256": sha256_file(source), "budgets": budgets}
            values["region_labels"] = labels
            save_npz(cfg.resolve(cfg.output_dir)/"replay"/source.name, values,
                     {"result_id": result_id(manifest["study_id"], spec), **spec,
                      "parts": expected["target_parts"], "comparisons": rows,
                      "source": str(source), "source_provenance": meta["provenance"],
                      "limitation": "Checks the estimator against saved table values. "
                                    "Does not verify how those old tables were generated or validate 32 groups."})
            print(f"Replayed ExactSHAP comparison: {source.name}", flush=True)

def main():
    args = parse_args()
    cfg = StudyConfig.load(args.config)
    if args.stage == "plan":
        print(json.dumps(plan(cfg), indent=2))
        return
    if args.stage == "report":
        from xai.report import build_report
        print(build_report(cfg))
        return
    if args.stage == "full":
        planned_jobs(cfg, "full", args.budget, args.kernel_seed, args.replacement_seed)
    manifest = setup_manifest(cfg)
    if args.stage == "check":
        result = known_answer_checks()
        atomic_json(cfg.resolve(cfg.output_dir)/"correctness.json", result)
        print(json.dumps(result, indent=2))
    elif args.stage in ("trial", "full", "smoke"):
        if args.stage == "smoke":
            atomic_json(cfg.resolve(cfg.output_dir)/"correctness.json", known_answer_checks())
            run_exact(cfg, manifest, smoke=True)
        run_sampled(cfg, args.stage, manifest, args.budget, args.kernel_seed, args.replacement_seed)
    elif args.stage == "exact":
        run_exact(cfg, manifest)
    elif args.stage == "replay":
        replay(cfg, manifest, args.tables_dir, args.reference_groups)

if __name__ == "__main__":
    main()
