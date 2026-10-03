"""Known-answer checks and ExactSHAP comparisons; no scientific pass/fail gates."""
import numpy as np
from xai.estimators.exact import coalitions_for_range, exact_shapley
from xai.estimators.kernel import kernel_shap
from xai.game import TableGame, SCORES
from xai.metrics import compare_maps

def known_answer_checks():
    K = 8
    z = coalitions_for_range(K, 0, 2**K).astype(float)
    # One helpful group, one harmful group, irrelevant groups, and a
    # three-group interaction. An additive-only test misses weighting bugs.
    y = np.column_stack((2*z[:,0], -3*z[:,1], np.prod(z[:,:3], axis=1),
                         z[:,0]-2*z[:,1]+4*np.prod(z[:,2:5], axis=1),
                         np.full(len(z), 5.0)))
    expected = np.zeros((K, 5))
    expected[0,0], expected[1,1] = 2, -3
    expected[:3,2] = 1/3
    expected[0,3], expected[1,3] = 1, -2
    expected[2:5,3] = 4/3
    exact = exact_shapley(y, K)
    np.testing.assert_allclose(exact, expected, atol=1e-12)
    game = TableGame({"known": y, "scaled": 2*y+7}, list(range(5)), K)
    phi, diag = kernel_shap(game, K, 2**K-2, seed=0)
    np.testing.assert_allclose(phi["known"], expected, atol=1e-10)
    np.testing.assert_allclose(phi["scaled"], 2*expected, atol=1e-10)
    # A sampled additive game also has known answers.
    additive = TableGame({"known": y[:,:2]}, [0,1], K)
    sampled, _ = kernel_shap(additive, K, 64, seed=2)
    np.testing.assert_allclose(sampled["known"], expected[:,:2], atol=1e-10)
    return {"cases": ["single helpful group", "harmful group", "irrelevant groups",
                       "three-group interaction", "mixed interactions", "constant output",
                       "multiple outputs", "sampled additive game"],
            "max_exact_error": float(np.max(np.abs(exact-expected))),
            "max_library_error_full_enumeration": float(np.max(np.abs(phi["known"]-expected))),
            "note": "Numerical correctness checks; these do not prove 32-group sampling accuracy."}

def compare_reference(tables, parts, K, budgets, seeds):
    game = TableGame(tables, parts, K, SCORES)
    exact = {s: exact_shapley(tables[s], K) for s in SCORES}
    rows, saved = [], {f"exact_phi_{s}": exact[s] for s in SCORES}
    for budget in budgets:
        for seed in seeds:
            phi, _ = kernel_shap(game, K, budget, seed=seed)
            for score in SCORES:
                saved[f"phi_{score}_m{budget}_s{seed}"] = phi[score]
                for j, part in enumerate(parts):
                    rows.append({"score": score, "part": int(part), "budget": budget,
                                 "kernel_seed": seed, **compare_maps(phi[score][:,j], exact[score][:,j])})
                if budget >= 2**K-2:
                    np.testing.assert_allclose(phi[score], exact[score], rtol=1e-7, atol=1e-8)
    return rows, saved
