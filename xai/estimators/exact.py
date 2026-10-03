"""ExactSHAP reference for small games, never the 32-group main study."""
import math
import numpy as np

def coalitions_for_range(K, start, end):
    return ((np.arange(start, end, dtype=np.int64)[:, None] >> np.arange(K)) & 1).astype(bool)

def enumerate_all(game, K, chunk_size=64):
    if K != game.K or not 2 <= K <= 12:
        raise ValueError("Live exact reference is limited to 2–12 groups")
    tables = {s: np.empty((2**K, len(game.target_parts))) for s in game.value_fns}
    for start in range(0, 2**K, chunk_size):
        end = min(start+chunk_size, 2**K)
        values = game.v(coalitions_for_range(K, start, end))
        for score in tables:
            tables[score][start:end] = values[score]
    return tables

def exact_shapley(table, K):
    """Exact weighted marginal contributions from a complete bitmask table."""
    table = np.asarray(table, dtype=np.float64)
    if not 1 <= K <= 20:
        raise ValueError("ExactSHAP reference supports at most 20 groups")
    if table.ndim != 2 or table.shape[0] != 2**K or not np.isfinite(table).all():
        raise ValueError("ExactSHAP needs a finite complete (2**K, parts) table")
    sizes = np.array([bin(i).count("1") for i in range(2**K)])
    weights = np.array([1.0/(K*math.comb(K-1, s)) for s in range(K)])
    indices = np.arange(2**K)
    phi = np.empty((K, table.shape[1]))
    for i in range(K):
        absent = indices[(indices & (1 << i)) == 0]
        phi[i] = np.sum(weights[sizes[absent], None] *
                       (table[absent | (1 << i)]-table[absent]), axis=0)
    np.testing.assert_allclose(phi.sum(axis=0), table[-1]-table[0], rtol=1e-8, atol=1e-8)
    return phi
