"""Frozen test objects, loaded from source; no stale cross-project object cache."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from data_utils.ShapeNetDataLoader import PartNormalDataset, pc_normalize

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT/"data"/"shapenetcore_partanno_segmentation_benchmark_v0_normal"

class _ReadOnlyPartDataset(PartNormalDataset):
    # Preflight and XAI may read an existing part-count cache but never rewrite it.
    def _save_part_cache(self, cache):
        pass

@dataclass(frozen=True)
class LoadedObject:
    pts: np.ndarray
    seg: np.ndarray
    cls_index: int
    source_path: str

@lru_cache(maxsize=8)
def get_dataset(category, test_count=118, num_parts=4, subset_seed=0):
    return _ReadOnlyPartDataset(root=str(DATA_ROOT), npoints=1, split="test",
        class_choice=[category], normal_channel=False, samples_per_category=test_count,
        num_parts=num_parts, subset_seed=subset_seed, verbose=False)

def object_paths(cfg):
    dataset = get_dataset(cfg.category, cfg.test_count, cfg.num_parts_filter, cfg.subset_seed)
    paths = [Path(p) for _, p in dataset.datapath]
    if len(paths) != cfg.test_count:
        raise ValueError(f"Requested {cfg.test_count} objects but found {len(paths)}")
    if cfg.object_names and [p.name for p in paths] != cfg.object_names:
        raise ValueError("Selected source objects or their order changed")
    # Explicitly check the source split, including a future joint-model comparison.
    import json
    split = DATA_ROOT/"train_test_split"
    trainval = set()
    for name in ("shuffled_train_file_list.json", "shuffled_val_file_list.json"):
        trainval.update(Path(p).name for p in json.loads((split/name).read_text()))
    overlaps = {p.name for p in paths if p.stem in trainval}
    if overlaps != set(cfg.training_seen_objects):
        raise ValueError("Training/test overlap differs from the explicit training-seen list: "
                         f"observed={sorted(overlaps)}, allowed={cfg.training_seen_objects}")
    return paths

def load_object(category, obj_index, npoints, sample_seed, test_count=118,
                num_parts=4, subset_seed=0):
    dataset = get_dataset(category, test_count, num_parts, subset_seed)
    category, path = dataset.datapath[obj_index]
    raw = np.loadtxt(path).astype(np.float32)
    pts = pc_normalize(raw[:, :3].copy())
    labels = raw[:, -1].astype(np.int64)
    choices = np.random.default_rng(sample_seed).choice(len(labels), npoints, replace=True)
    return LoadedObject(pts[choices].astype(np.float32), labels[choices],
                        int(dataset.classes[category]), str(path))

def load_for_config(cfg, index):
    return load_object(cfg.category, index, cfg.npoints, cfg.sample_seed,
                       cfg.test_count, cfg.num_parts_filter, cfg.subset_seed)
