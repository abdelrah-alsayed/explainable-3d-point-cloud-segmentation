"""Stream masked clouds through the model and keep only the two part scores."""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import numpy as np
import torch
from models import pointnet2_utils
from xai.masking import MASK_OPS, apply_mask

SCORES = ("iou", "logprob_mean")

def _deterministic_farthest_point_sample(xyz, npoint):
    """Original FPS algorithm with initial point 0, independent of batching."""
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, dtype=xyz.dtype, device=xyz.device)
    farthest = torch.zeros(B, dtype=torch.long, device=xyz.device)
    rows = torch.arange(B, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[rows, farthest].view(B, 1, 3)
        dist = torch.sum((xyz-centroid)**2, -1)
        distance = torch.minimum(distance, dist)
        farthest = distance.max(-1)[1]
    return centroids

@contextmanager
def deterministic_fps():
    # Restore the original implementation even if inference fails.
    original = pointnet2_utils.farthest_point_sample
    pointnet2_utils.farthest_point_sample = _deterministic_farthest_point_sample
    try:
        yield
    finally:
        pointnet2_utils.farthest_point_sample = original

class BatchScorer:
    """Bounded inference with OOM retry; inference never accumulates all logits."""
    def __init__(self, model, device, cls_onehot, batch_size=8):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model.eval()
        self.device = torch.device(device)
        self.cls_onehot = cls_onehot.to(self.device)
        self.batch_size = int(batch_size)
        self.n_clouds = 0
        self.deterministic_fps = True

    def iter_scores(self, clouds):
        clouds = np.asarray(clouds, dtype=np.float32)
        if clouds.ndim != 3 or clouds.shape[2] != 3:
            raise ValueError("Expected clouds with shape (M,N,3)")
        start = 0
        with deterministic_fps(), torch.inference_mode():
            while start < len(clouds):
                end = min(start+self.batch_size, len(clouds))
                batch = log_probs = onehot = None
                try:
                    batch = torch.from_numpy(clouds[start:end]).to(self.device).transpose(2,1)
                    onehot = self.cls_onehot.repeat(end-start, 1)
                    log_probs, _ = self.model(batch, onehot)
                    values = log_probs.detach().cpu().numpy()
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or end-start == 1:
                        raise
                    self.batch_size = max(1, (end-start)//2)
                    del batch, log_probs, onehot
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(f"GPU memory full; retrying with batch size {self.batch_size}", flush=True)
                    continue
                self.n_clouds += end-start
                yield start, values
                start = end
                del batch, log_probs, onehot, values

    def score(self, clouds):
        """For small uses such as the single unmasked prediction."""
        return np.concatenate([v.copy() for _, v in self.iter_scores(clouds)], axis=0)

class RegionGame:
    """One fixed object, grouping, masking seed, model, and target label set.

    Mean log-probability reads original GT-part rows. Config selects raw
    50-class or category-normalised probabilities. IoU uses category-restricted predictions and all original rows.
    Their meanings differ; scores are never combined or renormalized.
    """
    VALUE_FNS = SCORES

    def __init__(self, obj, region_labels, scorer, mask_op, target_parts,
                 value_fns=SCORES, mask_seed=0, chunk_size=64, logprob_mode="raw50"):
        if mask_op not in MASK_OPS:
            raise ValueError(f"Unsupported masking method: {mask_op}")
        self.obj = obj
        self.region_labels = np.asarray(region_labels, dtype=np.int64)
        self.K = int(self.region_labels.max())+1
        if self.region_labels.shape != obj.seg.shape or not np.array_equal(
                np.unique(self.region_labels), np.arange(self.K)):
            raise ValueError("Every group must contain points and labels must be 0..K-1")
        self.scorer, self.mask_op = scorer, mask_op
        self.target_parts, self.value_fns = list(target_parts), list(value_fns)
        if logprob_mode not in ("raw50", "category"):
            raise ValueError("Unknown log-probability definition")
        self.logprob_mode = logprob_mode
        if set(self.value_fns)-set(SCORES) or not self.value_fns:
            raise ValueError("Use IoU and/or mean log-probability")
        self.mask_seed, self.chunk_size = int(mask_seed), int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._Q = {p: np.flatnonzero(obj.seg == p) for p in self.target_parts}
        if "logprob_mean" in self.value_fns and any(not len(q) for q in self._Q.values()):
            raise ValueError("Cannot explain mean log-probability for an absent true part")
        # Only reduced scores are cached, in this game instance. No disk cache
        # can leak across models, objects, seeds, or score definitions.
        self._cache = {}

    def _coalition_rng(self, row):
        bits = np.packbits(np.asarray(row, dtype=np.uint8)).tobytes()
        digest = hashlib.sha256(str(self.mask_seed).encode()+b"|" + bits).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "little"))

    def _reduce_single(self, log_probs):
        log_probs = np.asarray(log_probs)
        if self.logprob_mode == "category":
            log_probs = log_probs.astype(np.float64)
            # Stable log-softmax within the category. IoU argmax is unchanged.
            local = log_probs[:, self.target_parts]
            peak = local.max(axis=1, keepdims=True)
            normalizer = peak + np.log(np.exp(local-peak).sum(axis=1, keepdims=True))
            log_probs = log_probs-normalizer
        predicted = np.asarray(self.target_parts)[log_probs[:, self.target_parts].argmax(axis=1)]
        out = {s: [] for s in self.value_fns}
        for p in self.target_parts:
            if "logprob_mean" in out:
                out["logprob_mean"].append(float(log_probs[self._Q[p], p].mean()))
            if "iou" in out:
                pred, truth = predicted == p, self.obj.seg == p
                union = np.count_nonzero(pred | truth)
                out["iou"].append(float(np.count_nonzero(pred & truth)/union) if union else 1.0)
        return {s: np.asarray(v, dtype=np.float64) for s, v in out.items()}

    def v(self, coalitions):
        coalitions = np.asarray(coalitions, dtype=bool)
        if coalitions.ndim != 2 or coalitions.shape[1] != self.K:
            raise ValueError(f"Expected (M,{self.K}) coalitions")
        keys = [np.packbits(row).tobytes() for row in coalitions]
        missing = {}
        for key, row in zip(keys, coalitions):
            if key not in self._cache:
                missing.setdefault(key, row)
        work = list(missing.items())
        for start in range(0, len(work), self.chunk_size):
            chunk = work[start:start+self.chunk_size]
            clouds = np.stack([apply_mask(self.obj.pts, self.region_labels, row,
                               self.mask_op, self._coalition_rng(row)) for _, row in chunk])
            for offset, log_probs in self.scorer.iter_scores(clouds):
                for j, values in enumerate(log_probs):
                    self._cache[chunk[offset+j][0]] = self._reduce_single(values)
        return {s: np.asarray([self._cache[key][s] for key in keys], dtype=np.float64)
                    .reshape(len(keys), len(self.target_parts)) for s in self.value_fns}

class TableGame:
    """Replay complete exact tables without model inference."""
    def __init__(self, tables, target_parts, K, value_fns=None):
        self.K, self.target_parts = int(K), list(target_parts)
        self.value_fns = list(value_fns) if value_fns is not None else list(tables)
        self.tables = {s: np.asarray(tables[s], dtype=np.float64) for s in self.value_fns}
        for score, table in self.tables.items():
            if table.shape != (2**K, len(target_parts)) or not np.isfinite(table).all():
                raise ValueError(f"Incomplete or invalid table: {score}")

    def v(self, coalitions):
        coalitions = np.asarray(coalitions, dtype=bool)
        if coalitions.ndim != 2 or coalitions.shape[1] != self.K:
            raise ValueError(f"Expected (M,{self.K}) coalitions")
        indices = (coalitions.astype(np.int64) << np.arange(self.K)).sum(axis=1)
        return {s: table[indices] for s, table in self.tables.items()}
