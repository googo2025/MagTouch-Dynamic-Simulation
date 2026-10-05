"""Repeat virtual-only inverse fitting; do not tune on any test result.

Seeds are declared before this run. Virtual generator, calibration, basis,
memory hyperparameters, architecture, supervision and all splits are fixed.
This checks optimizer-initialization sensitivity, not generator realism.
"""
import json
import numpy as np
from training.simulation_discovery_r367 import (
    OUT, GEN, AxisForward, ForwardAtlas, make_trials, prepare,
    guard_and_features, mlp_data, MLP, evaluate, metrics, dump, sha,
)

SEEDS = (7, 17, 27, 37, 47)
NAMES = ('causal_no_static_anchors', 'causal_static_anchors',
         'current_only_static_anchors')


def main():
    target = OUT / 'initialization_replications.json'
    if target.exists():
        raise FileExistsError('preserve completed replication experiment')
    generator = AxisForward.load(GEN / 'models/generator.json')
    manifest = json.loads((GEN / 'synthetic_manifest.json').read_text(encoding='utf8'))
    sigma = np.array(manifest['noise_sigma_uT'])
    cal = np.load(OUT / 'virtual_static_calibration.npz')
    atlas = ForwardAtlas(cal['q'], cal['b'], 'unit_logamp')
    basis = np.load(OUT / 'virtual_memory_basis.npz')['basis']
    pick = json.loads((OUT / 'memory_selection.json').read_text(encoding='utf8'))['selected']
    train, valid, test, stationary = make_trials(generator, sigma)
    for a in train + valid + test + [stationary]:
        prepare(a, atlas, basis)
        guard_and_features(a, atlas, basis, pick['retention'], pick['strength'])
    x = np.vstack([a['x'][np.flatnonzero(a['primary'] & a['gate'])[::2]] for a in train])
    y = np.vstack([np.clip(a['q'][ix, :2] - a['guarded'][ix, :2], -3, 3)
                   for a in train
                   for ix in [np.flatnonzero(a['primary'] & a['gate'])[::2]]])
    anchors = np.column_stack((cal['q'][::3] / [10, 10, 3],
                              atlas.features(cal['b'][::3]),
                              np.zeros((len(cal['q'][::3]), 34))))
    np.savez_compressed(OUT / 'virtual_training_features.npz',
                        x=x, y=y, anchors=anchors)
    dump(OUT / 'replication_protocol.json', {
        'seeds_declared_before_execution': list(SEEDS),
        'all_seeds_reported': True, 'select_seed_on_test': False,
        'training_ids': [a['id'] for a in train],
        'validation_ids': [a['id'] for a in valid],
        'standard_test_ids': [a['id'] for a in test],
        'complex_test_ids': [a['id'] for a in manifest['trials']],
        'motion_samples': len(x), 'static_anchor_samples': len(anchors),
        'shared_model_architecture': '53-64-64-2 tanh',
        'current_only_ablation': 'zero ONLY h and lag input blocks; shared upstream M/G retained',
        'anchor_ablation': 'same recipe except static training samples; preprocessing changes with sample distribution',
        'learned_inverse_real_samples': 0,
        'virtual_generator_sha256': sha(GEN / 'models/generator.json'),
        'calibration_sha256': sha(OUT / 'virtual_static_calibration.npz'),
    })
    tests = [(a, 'history_on') for a in test + [stationary]]
    for a in test:
        off = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in a.items()
               if k in ('id', 'family', 'q', 't', 'primary')}
        off['b'] = a['off']; off['gate'] = np.linalg.norm(off['b'], axis=1) >= 70
        prepare(off, atlas, basis)
        guard_and_features(off, atlas, basis, pick['retention'], pick['strength'])
        tests.append((off, 'history_off'))
    data = np.load(GEN / 'complex_synthetic.npz')
    for i, info in enumerate(manifest['trials']):
        lo, hi = data['offsets'][i:i+2]
        q, t, b = data['q'][lo:hi], data['t'][lo:hi], data['b'][lo:hi]
        a = {'id': info['id'], 'family': info['family'], 'q': q, 't': t, 'b': b,
             'primary': (q[:, 2] >= .4) & (q[:, 2] <= 3),
             'gate': np.linalg.norm(b, axis=1) >= 70}
        prepare(a, atlas, basis)
        guard_and_features(a, atlas, basis, pick['retention'], pick['strength'])
        tests.append((a, 'complex_heldout'))
    rows = []
    for seed in SEEDS:
        for name in NAMES:
            anchor = name != 'causal_no_static_anchors'
            instant = name == 'current_only_static_anchors'
            xx = np.vstack((x, anchors)) if anchor else x.copy()
            yy = np.vstack((y, np.zeros((len(anchors), 2)))) if anchor else y.copy()
            if instant:
                xx[:, 19:] = 0
            if seed == 7:
                model = MLP(64, seed=seed)
                with np.load(OUT / (name + '.npz')) as weights:
                    for key in weights.files:
                        setattr(model, key, weights[key])
                info = json.loads((OUT / (name + '.json')).read_text(encoding='utf8'))
                epoch, objective = info['epoch'], info['objective']
            else:
                model = MLP(64, seed=seed).fit(xx, yy, mlp_data(valid, instant), atlas)
                epoch, objective = model.best_epoch, model.best_validation
                dest = OUT / 'replicate_models'; dest.mkdir(exist_ok=True)
                np.savez_compressed(dest / f'{name}_seed{seed}.npz',
                    **{k: getattr(model, k) for k in ('mean','scale','w1','b1','w2','b2','w3','b3')})
            scores = []
            for a, world in tests:
                p = evaluate(a, model, instant)
                scores.append({'id': a['id'], 'world': world, 'family': a['family'],
                               'scores': metrics(p, a['q'], a['t'], a['primary'], a['gate'])})
            rows.append({'seed': seed, 'model': name, 'epoch': epoch,
                         'validation_xy_mm': objective, 'tests': scores})
            print('virtual-only replication', seed, name, epoch, flush=True)
    dump(target, {'seeds': list(SEEDS), 'rows': rows,
                  'test_used_for_selection': False,
                  'boundary': 'five optimizer initializations of one fixed empirical generator; not five independent physical devices'})


if __name__ == '__main__':
    main()
