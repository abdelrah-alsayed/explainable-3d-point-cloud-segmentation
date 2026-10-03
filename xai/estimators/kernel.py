"""Official KernelSHAP over binary groups; all scores share model runs."""
from __future__ import annotations
import numpy as np

def kernel_shap(game, K: int, nsamples: int, seed: int = 0):
    """Return {score: (K, parts) importance} and the sampled game.

    The background is the all-hidden membership vector, not a background
    point cloud. RegionGame applies the geometry masking. No extra weights
    or feature selection are applied here. Do not call in parallel threads:
    SHAP uses the global NumPy RNG, whose state is restored after this call.
    """
    import shap
    if K != game.K or K < 2:
        raise ValueError("Use the game's group count, at least two")
    if nsamples < min(K + 1, 2**K-2):
        raise ValueError("Use at least K+1 combinations, or all interior combinations for K=2")
    scores, n_parts = list(game.value_fns), len(game.target_parts)

    def model_fn(membership):
        membership = np.asarray(membership)
        if not np.all((membership == 0) | (membership == 1)):
            raise ValueError("Group membership must be binary")
        values = game.v(membership.astype(bool))
        out = np.concatenate([values[s] for s in scores], axis=1)
        if not np.isfinite(out).all():
            raise ValueError("Non-finite scores; check absent parts and model output")
        return out

    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        explainer = shap.KernelExplainer(
            model_fn, np.zeros((1, K)), link="identity",
            feature_names=[f"group_{i}" for i in range(K)],
        )
        raw = explainer.shap_values(
            np.ones((1, K)), nsamples=int(nsamples), l1_reg=0.0, silent=True,
        )
    finally:
        np.random.set_state(rng_state)
    raw = np.asarray(raw, dtype=np.float64)
    expected = (1, K, len(scores) * n_parts)
    if raw.shape != expected:
        raise RuntimeError(f"Unexpected SHAP shape {raw.shape}; expected {expected}")
    phi = {s: raw[0, :, i*n_parts:(i+1)*n_parts] for i, s in enumerate(scores)}
    # These inspection attributes are covered by tests and a pinned SHAP version.
    n = int(explainer.nsamplesAdded)
    diag = {
        "coalitions": np.asarray(explainer.maskMatrix[:n], dtype=bool),
        "weights": np.asarray(explainer.kernelWeights[:n], dtype=np.float64),
        "values": np.asarray(explainer.ey[:n], dtype=np.float64).reshape(n, len(scores), n_parts),
        "v_empty": np.asarray(explainer.expected_value).reshape(len(scores), n_parts),
        "v_full": np.asarray(explainer.fx).reshape(len(scores), n_parts),
    }
    stacked = np.stack([phi[s] for s in scores], axis=1)
    # Bookkeeping only: efficiency is imposed by SHAP, not proof of accuracy.
    np.testing.assert_allclose(stacked.sum(axis=0), diag["v_full"]-diag["v_empty"],
                               rtol=1e-7, atol=1e-8)
    return phi, diag
