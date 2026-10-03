"""Three evidence checks; no scientific pass/fail thresholds."""
import numpy as np
from scipy.stats import spearmanr

def compare_maps(a, b):
    """Compare actual values and signed ranks; constant ranks are undefined."""
    a, b = np.asarray(a), np.asarray(b)
    rho = float(spearmanr(a, b).statistic) if np.ptp(a) > 0 and np.ptp(b) > 0 else None
    return {"mean_absolute_difference": float(np.abs(a-b).mean()), "rank_agreement": rho}

def helpful_removal(game, phi, score, part_index, n_random=5, seed=0):
    """Remove positive groups only; compare equal group counts to random orders.

    Shapley contributions average over contexts. Removing a positive group
    from the full object need not lower its score. These are observations,
    not pass/fail claims. Group sizes can differ; save hidden point fractions.
    """
    if n_random < 1:
        raise ValueError("At least one random order is needed")
    order = np.argsort(-np.asarray(phi), kind="stable")
    order = order[np.asarray(phi)[order] > 0]
    steps = len(order)
    def curve(indices):
        coalitions = np.ones((steps+1, game.K), dtype=bool)
        for n in range(1, steps+1):
            coalitions[n, indices[:n]] = False
        scores = game.v(coalitions)[score][:, part_index]
        if hasattr(game, "region_labels"):
            fractions = (~coalitions[:, game.region_labels]).mean(axis=1)
        else:
            fractions = np.full(steps+1, np.nan)
        return scores, fractions
    helpful, hidden = curve(order)
    rng = np.random.default_rng(seed)
    random = [curve(rng.permutation(game.K)) for _ in range(n_random)]
    return {
        "positive_order": order, "groups_hidden": np.arange(steps+1),
        "helpful_scores": helpful,
        "random_scores": np.stack([r[0] for r in random]),
        "helpful_hidden_point_fraction": hidden,
        "random_hidden_point_fraction": np.stack([r[1] for r in random]),
    }


def matched_cross_curves(pair, score, part):
    """Interpolate saved curves on one shared point-fraction range; never extrapolate.

    Whole groups cannot generally hit identical point fractions. Linear interpolation
    compares saved curves, not newly evaluated point masks. If any positive order is
    empty there is no common nonzero range, so no area comparison is reported.
    """
    from xai.masking import MASK_OPS
    curves = {}
    ends = []
    for source in MASK_OPS:
        arrays = pair[source][0]
        for evaluation in MASK_OPS:
            prefix = f'cross_removal_{evaluation}_{score}_part{part}_'
            if prefix+'helpful_scores' not in arrays:
                return None
            x = arrays[prefix+'helpful_hidden_point_fraction']
            y = arrays[prefix+'helpful_scores']
            rx = arrays[prefix+'random_hidden_point_fraction']
            ry = arrays[prefix+'random_scores']
            curves[source,evaluation] = (x,y,rx,ry)
            ends.extend([float(x[-1]), *rx[:,-1].tolist()])
    end = min(ends)
    if end <= 0:
        return {'end_fraction': 0., 'rows': [], 'reason': 'No shared nonzero range of positive-group removal'}
    grid = np.linspace(0,end,101)
    rows = []
    for (source,evaluation),(x,y,rx,ry) in curves.items():
        helpful = np.interp(grid,x,y)
        random = np.stack([np.interp(grid,a,b) for a,b in zip(rx,ry)])
        # Units: score times hidden-point fraction. No division by endpoint delta.
        area = lambda v: float(np.trapezoid(v,grid))
        rows.append({'source_mask':source,'evaluation_mask':evaluation,
                     'grid':grid.tolist(),'helpful':helpful.tolist(),'random':random.tolist(),
                     'helpful_drop_area':area(helpful[0]-helpful),
                     'random_drop_area_mean':area((random[:,[0]]-random).mean(axis=0)),
                     'extra_drop_area':area(random.mean(axis=0)-helpful)})
    return {'end_fraction':end,'rows':rows,
            'note':'Linear interpolation of whole-group steps on a shared range; no extrapolation or new inference.'}
