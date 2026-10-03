"""Shared read-only validation for runners, reports and comparisons."""
import hashlib
import json
import numpy as np
from xai.config import StudyConfig
from xai.game import SCORES
from xai.masking import MASK_OPS
from xai.model_io import SEG_CLASSES
from xai.storage import load_npz, result_id, digest_json

def checked_array(arrays, key, shape):
    value = arrays[key]
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{key}: wrong shape or non-finite values")
    return value


def check_result(path, cfg, manifest, spec):
    arrays, meta = load_npz(path, result_id(manifest['study_id'], spec))
    if any(meta.get(k) != v for k, v in spec.items()):
        raise ValueError('Saved run settings differ from the planned run')
    parts = list(SEG_CLASSES[cfg.category])
    if meta.get('parts') != parts or meta.get('scores') != list(SCORES):
        raise ValueError('Wrong target parts or missing score names')
    if meta.get('logprob_mode', 'raw50') != cfg.logprob_mode:
        raise ValueError('Wrong log-probability definition')
    if meta.get('cohort',cfg.cohort(spec['object'])) != cfg.cohort(spec['object']):
        raise ValueError('Wrong object cohort')
    for score in SCORES:
        checked_array(arrays, 'phi_'+score, (cfg.groups, len(parts)))
    for key in ('v_full', 'v_empty'):
        checked_array(arrays, key, (len(SCORES), len(parts)))
    labels = checked_array(arrays, 'region_labels', (cfg.npoints,))
    if labels.dtype.kind not in 'iu' or not np.array_equal(np.unique(labels), np.arange(cfg.groups)):
        raise ValueError('Missing or invalid group labels')
    n = len(arrays['coalitions'])
    z = checked_array(arrays, 'coalitions', (n, cfg.groups))
    if n < 1 or not np.isin(z, [0, 1]).all():
        raise ValueError('Invalid saved group combinations')
    if n > min(spec['budget'],2**cfg.groups-2) or len(np.unique(z,axis=0)) != n:
        raise ValueError('Too many or duplicate saved combinations')
    if np.any(z.sum(axis=1)==0) or np.any(z.sum(axis=1)==cfg.groups):
        raise ValueError('Endpoints must be stored separately from interior coalitions')
    weights = checked_array(arrays, 'weights', (n,))
    if np.any(weights <= 0):
        raise ValueError('Coalition weights must be positive')
    checked_array(arrays, 'values', (n, len(SCORES), len(parts)))
    for si, score in enumerate(SCORES):
        np.testing.assert_allclose(arrays['phi_'+score].sum(axis=0),
                                   arrays['v_full'][si]-arrays['v_empty'][si], rtol=1e-7, atol=1e-8)
    removal = spec['object'] in cfg.trial_objects and (
        spec['stage'] in ('full', 'smoke') or
        (spec['budget'] == max(cfg.trial_budgets)
         and spec['kernel_seed'] == cfg.kernel_seeds[0]
         and spec['replacement_seed'] == cfg.replacement_seeds[0]))
    if removal:
        for score in SCORES:
            for part in parts:
                prefix = f'removal_{score}_part{part}_'
                order = arrays[prefix+'positive_order']
                phi = arrays['phi_'+score][:,parts.index(part)]
                expected_order = np.argsort(-phi,kind='stable')
                expected_order = expected_order[phi[expected_order] > 0]
                if not np.array_equal(order,expected_order):
                    raise ValueError('Removal order does not match the positive SHAP ranking')
                steps = len(order)+1
                checked_array(arrays, prefix+'positive_order', (steps-1,))
                checked_array(arrays, prefix+'groups_hidden', (steps,))
                for suffix in ('helpful_scores', 'helpful_hidden_point_fraction'):
                    checked_array(arrays, prefix+suffix, (steps,))
                for suffix in ('random_scores', 'random_hidden_point_fraction'):
                    checked_array(arrays, prefix+suffix, (cfg.random_removal_orders, steps))
                check_curve(arrays, prefix, cfg, order)
                expected_fraction = np.array([np.isin(labels,order[:n]).mean() for n in range(steps)])
                np.testing.assert_allclose(arrays[prefix+'helpful_hidden_point_fraction'],expected_fraction,atol=1e-12)
                if cfg.cross_mask_removal:
                    for op in MASK_OPS:
                        cross_prefix = f'cross_removal_{op}_{score}_part{part}_'
                        check_curve(arrays, cross_prefix, cfg, order)
                        np.testing.assert_allclose(arrays[cross_prefix+'helpful_hidden_point_fraction'],expected_fraction,atol=1e-12)
    return {'groups': hashlib.sha256(labels.tobytes()).hexdigest(),
            'combinations': hashlib.sha256(z.tobytes()).hexdigest(),
            'full': arrays['v_full'].tolist()}


def check_prediction(path, cfg, manifest, index):
    spec = {'object': index, 'kind': 'unmasked prediction'}
    a, _ = load_npz(path, result_id(manifest['study_id'], spec))
    parts = list(SEG_CLASSES[cfg.category])
    checked_array(a, 'points', (cfg.npoints, 3))
    for key in ('truth', 'prediction'):
        v = checked_array(a, key, (cfg.npoints,))
        if not np.isin(v, parts).all():
            raise ValueError(f'Unexpected {key} labels')
    if a['parts'].tolist() != parts:
        raise ValueError('Wrong prediction part list')
    for s in SCORES:
        checked_array(a, 'score_'+s, (len(parts),))
    return {}


def check_curve(arrays, prefix, cfg, expected_order):
    order = arrays[prefix+'positive_order']
    if (order.dtype.kind not in 'iu' or len(set(order.tolist())) != len(order) or
            np.any(order < 0) or np.any(order >= cfg.groups) or
            not np.array_equal(order, expected_order)):
        raise ValueError('Invalid removal order')
    steps = len(order)+1
    np.testing.assert_array_equal(arrays[prefix+'groups_hidden'], np.arange(steps))
    for suffix, shape in [('helpful_scores',(steps,)),('random_scores',(cfg.random_removal_orders,steps)),
                          ('helpful_hidden_point_fraction',(steps,)),
                          ('random_hidden_point_fraction',(cfg.random_removal_orders,steps))]:
        a = checked_array(arrays, prefix+suffix, shape)
        if 'fraction' in suffix and (np.any(a < 0) or np.any(a > 1) or
                                     np.any(np.diff(a,axis=-1) <= 0) or np.any(a[...,0] != 0)):
            raise ValueError('Invalid hidden-point fractions')


def check_manifest(cfg, manifest):
    if digest_json(manifest['identity']) != manifest['study_id']:
        raise ValueError('Manifest identity is inconsistent')
    # Permit reading old studies whose omitted fields have legacy defaults.
    if StudyConfig(**manifest['identity']['config']).as_dict() != cfg.as_dict():
        raise ValueError('Configuration differs from the saved study')


def check_exact(path, cfg, manifest, spec):
    from xai.estimators.exact import exact_shapley
    a, m = load_npz(path, result_id(manifest['study_id'], spec))
    if any(m.get(k) != v for k,v in spec.items()):
        raise ValueError('Wrong exact-check settings')
    parts = list(SEG_CLASSES[cfg.category]); K = spec['groups']
    if m.get('parts') != parts or m.get('logprob_mode','raw50') != cfg.logprob_mode:
        raise ValueError('Wrong exact-check score or parts')
    labels = checked_array(a,'region_labels',(cfg.npoints,))
    if labels.dtype.kind not in 'iu' or not np.array_equal(np.unique(labels),np.arange(K)):
        raise ValueError('Invalid exact-check groups')
    for score in SCORES:
        table = checked_array(a,'table_'+score,(2**K,len(parts)))
        phi = checked_array(a,'exact_phi_'+score,(K,len(parts)))
        np.testing.assert_allclose(phi,exact_shapley(table,K),rtol=1e-7,atol=1e-8)
        for budget in spec['budgets']:
            for seed in spec['seeds']:
                sampled=checked_array(a,f'phi_{score}_m{budget}_s{seed}',(K,len(parts)))
                np.testing.assert_allclose(sampled.sum(axis=0),table[-1]-table[0],rtol=1e-7,atol=1e-8)
                if budget>=2**K-2:
                    np.testing.assert_allclose(sampled,phi,rtol=1e-7,atol=1e-8)
    if not isinstance(m.get('comparisons'),list) or not m['comparisons']:
        raise ValueError('Missing exact-check comparisons')
    return a,m
