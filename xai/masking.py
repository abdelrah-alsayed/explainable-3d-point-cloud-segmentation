"""Two equally used masking methods. Point rows and original labels stay fixed."""
import numpy as np

MASK_OPS = ("centroid", "retained_point")
MASK_NAMES = {"centroid": "Origin collapse", "retained_point": "Retained-point masking"}

def apply_mask(pts, region_labels, coalition, op, rng):
    if op not in MASK_OPS:
        raise ValueError(f"Unknown masking method {op!r}; choose from {MASK_OPS}")
    pts = np.asarray(pts, dtype=np.float32)
    labels = np.asarray(region_labels, dtype=np.int64)
    coalition = np.asarray(coalition, dtype=bool)
    visible = coalition[labels]
    out = pts.copy()
    if op == "centroid" or not visible.any():
        # Shared all-hidden baseline: retained-point has no visible donor.
        out[~visible] = 0.0
    else:
        donors = rng.choice(np.flatnonzero(visible), size=int((~visible).sum()), replace=True)
        out[~visible] = pts[donors]
    return out
