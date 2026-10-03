"""Model + checkpoint loading for the point-cloud Shapley pipeline.

Mirrors exactly what `test_partseg.py`/`train_partseg.py` do at inference
time, reusing the existing model and category bookkeeping rather than
duplicating it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # models/pointnet2_part_seg_msg.py does `from models.pointnet2_utils import
    # ...`, which requires the repo root (not just `models/`) on sys.path.
    # This also makes `xai` itself importable regardless of invocation cwd.
    sys.path.insert(0, str(REPO_ROOT))

from models.pointnet2_part_seg_msg import get_model  # noqa: E402
from data_utils.ShapeNetDataLoader import SEG_CLASSES  # noqa: E402,F401  (re-exported)

NUM_PART = 50
NUM_CATEGORIES = 16

_CATFILE = (
    REPO_ROOT
    / "data"
    / "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    / "synsetoffset2category.txt"
)

_category_index_cache = None


def _category_index_map():
    global _category_index_cache
    if _category_index_cache is None:
        mapping = {}
        with open(_CATFILE, "r") as f:
            for i, line in enumerate(f):
                ls = line.strip().split()
                if not ls:
                    continue
                mapping[ls[0]] = i
        _category_index_cache = mapping
    return _category_index_cache


#: Category indices this pipeline has verified against the checkpoint's own
#: training run. The index comes from line order in
#: synsetoffset2category.txt, so a differently-populated copy of that file on
#: another machine would silently shift it and produce a wrong cls_label
#: one-hot -- no error, just quietly wrong attributions. (This repo's copy has
#: 15 lines: Pistol is absent, while NUM_CATEGORIES is 16.) Asserting the
#: handful of categories actually explained here is cheap insurance.
EXPECTED_CATEGORY_INDEX = {"Airplane": 0, "Car": 3, "Chair": 4, "Laptop": 9,
                           "Motorbike": 10, "Rocket": 12, "Skateboard": 13}


def category_onehot(category: str) -> torch.Tensor:
    """(1, 16) float one-hot vector, index = line order in
    synsetoffset2category.txt (Airplane == 0), matching to_categorical(...)
    in train_partseg.py/test_partseg.py."""
    idx = _category_index_map()[category]
    expected = EXPECTED_CATEGORY_INDEX.get(category)
    if expected is not None and idx != expected:
        raise RuntimeError(
            f"category {category!r} maps to index {idx}, expected {expected}. "
            f"{_CATFILE} differs from the copy this checkpoint was trained "
            f"with; the cls_label one-hot would be wrong."
        )
    onehot = torch.zeros(1, NUM_CATEGORIES, dtype=torch.float32)
    onehot[0, idx] = 1.0
    return onehot


def load_model(log_dir, num_part: int = NUM_PART, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_part, normal_channel=False).to(device)

    ckpt_path = Path(log_dir) / "checkpoints" / "best_model.pth"
    # weights_only=False is REQUIRED on torch>=2.6 (default flipped to True,
    # which rejects this checkpoint's non-tensor keys like epoch/test_acc).
    # Harmless on torch 2.1.2, which predates that default.
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
