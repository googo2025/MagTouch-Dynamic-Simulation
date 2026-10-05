"""Causal sequence confusers, source-only candidate ranking and two-tip hypotheses.

Synthetic hypotheses are kept separate from actual measured-field searches.
Two-tip superposition is explicitly NOT a coupled-mechanics simulation.
"""
from __future__ import annotations
import json
import numpy as np
from training.run_forward_r361 import ROOT, load_row, dump, sha
from training.forward_complex_r363 import AxisForward
from training.magnetic_forward_r361 import MagneticForward

OUT = ROOT / 'analysis/r366_sequence_ambiguity_20261001'
VIRTUAL = ROOT / 'analysis/r363_complex_forward_20261001'
WINDOWS = (0., .1, .4, 1., 2.)


def causal_windows(b, t, reset, endpoints, window):
    """Last available samples at fixed causal lags. Never cross reset/gap."""
    lags = np.arange(round(window * 10) + 1) * .1
    indices = np.searchsorted(t, t[endpoints, None] - lags[None, :] + 1e-8, side='right') - 1
    clipped = np.maximum(indices, 0)
    token = np.cumsum(np.asarray(reset, bool) | np.r_[False, np.diff(t) > .25])
    good = (indices >= 0).all(1)
    good &= (token[clipped] == token[endpoints, None]).all(1)
    good &= ((t[endpoints, None] - lags - t[clipped]) <= .2).all(1)
    good &= np.isfinite(b[clipped]).all((1, 2))
    values = b[clipped].reshape(len(endpoints), -1)
    return values, good


def bank(rows, sigma, maximum=2200):
    records = []; features = {str(w): [] for w in WINDOWS}; positions = []
    for trial, a in enumerate(rows):
        mask = a['primary'] & np.isfinite(a['b']).all(1)
        endpoints = np.flatnonzero(mask)[::4]
        if len(endpoints) == 0: continue
        _, good = causal_windows(a['b'], a['t'], a['reset'], endpoints, 2.)
        endpoints = endpoints[good]
        if not len(endpoints): continue
        positions.append(a['q'][endpoints])
        for w in WINDOWS:
            values, valid = causal_windows(a['b'], a['t'], a['reset'], endpoints, w)
            if not valid.all(): raise ValueError('window validity must share endpoint pool')
            features[str(w)].append(values / np.tile(sigma, round(w * 10) + 1))
        records.extend({'trial': trial, 'id': a['id'], 'index': int(i), 'time_s': float(a['t'][i])}
                       for i in endpoints)
    if not records: raise ValueError('empty sequence bank')
    q = np.vstack(positions)
    x = {key: np.vstack(values) for key, values in features.items()}
    choose = np.arange(len(q)) if len(q) <= maximum else np.linspace(0, len(q)-1, maximum).astype(int)
    return {'q': q[choose], 'x': {key: value[choose] for key, value in x.items()},
            'records': [records[i] for i in choose], 'before_cap': len(q)}


def nearest_confusers(b, window, within_trial=False):
    x, q = b['x'][str(window)], b['q']
    n = len(q); trial = np.array([r['trial'] for r in b['records']]); t = np.array([r['time_s'] for r in b['records']])
    norm = np.sum(x*x, 1); best = np.full(n, np.inf); other = np.full(n, -1, int)
    for start in range(0, n, 96):
        end = min(start+96, n)
        distance = np.maximum(norm[start:end, None] + norm[None, :] - 2*x[start:end]@x.T, 0) / x.shape[1]
        xy = np.linalg.norm(q[start:end, None, :2] - q[None, :, :2], axis=2)
        z = np.abs(q[start:end, None, 2] - q[None, :, 2])
        separated = (xy >= 1.) | (z >= .3)
        same = trial[start:end, None] == trial[None, :]
        separated &= ~same | (np.abs(t[start:end, None] - t[None, :]) >= 3.)
        if within_trial: separated &= same
        distance[~separated] = np.inf
        idx = np.argmin(distance, axis=1)
        other[start:end] = idx; best[start:end] = np.sqrt(distance[np.arange(end-start), idx])
    pairs = sorted({tuple(sorted((i, int(j)))) for i, j in enumerate(other) if np.isfinite(best[i])})
    return pairs, best


def pair_distance(b, pairs, window):
    if not pairs: return np.empty(0)
    i, j = np.asarray(pairs, int).T
    return np.sqrt(np.mean((b['x'][str(window)][i]-b['x'][str(window)][j])**2, axis=1))


def peak_pair_distance(b, pairs, window):
    if not pairs: return np.empty(0)
    i, j = np.asarray(pairs, int).T
    delta = (b['x'][str(window)][i]-b['x'][str(window)][j]).reshape(len(i), -1, 15)
    return np.sqrt(np.mean(delta**2, axis=2)).max(axis=1)


def search(b, name):
    pairs, nearest = nearest_confusers(b, 0.)
    current = pair_distance(b, pairs, 0.)
    close = [p for p, d in zip(pairs, current) if d <= 2.]
    # A whole-window average can dilute a distinguishable CURRENT frame with
    # similar old frames. Primary rule keeps each lag as evidence: max of the
    # per-lag 15-channel RMS distances. This is monotone for nested histories.
    q = b['q']; n = len(q)
    trials = np.array([r['trial'] for r in b['records']]); times = np.array([r['time_s'] for r in b['records']])
    same = trials[:, None] == trials[None, :]
    allowed = (np.linalg.norm(q[:, None, :2]-q[None, :, :2], axis=2) >= 1.) | (np.abs(q[:, None, 2]-q[None, :, 2]) >= .3)
    allowed &= ~same | (np.abs(times[:, None]-times[None, :]) >= 3.)
    history = b['x']['2.0'].reshape(n, 21, 15)
    cumulative = np.zeros((n, n))
    hist = {}; rng = np.random.default_rng(366)
    noise = rng.normal(0, np.sqrt(2), (12000, 21, 15))
    noise_distances = np.sqrt(np.mean(noise**2, axis=2))
    noise_distances = noise_distances[noise_distances[:, 0] <= 2.]
    for lag in range(21):
        x = history[:, lag]; norm = np.sum(x*x, 1)
        distance = np.maximum(norm[:, None]+norm[None, :]-2*x@x.T, 0) / 15
        cumulative = np.maximum(cumulative, distance)
        w = lag / 10
        if w not in WINDOWS: continue
        d = peak_pair_distance(b, close, w)
        peak = np.sqrt(cumulative)
        best = np.min(np.where(allowed, peak, np.inf), axis=1)
        best_within = np.min(np.where(allowed & same, peak, np.inf), axis=1)
        finite = np.isfinite(best)
        noise_threshold = float(np.quantile(noise_distances[:, :lag+1].max(axis=1), .95))
        hist[str(w)] = {'fixed_current_confuser_pairs': len(close),
            'fixed_pairs_median_distance': float(np.median(d)) if len(d) else None,
            'fixed_pairs_still_close_at_2': int((d <= 2.).sum()),
            'fixed_within_trial_pairs': int(sum(trials[i] == trials[j] for i, j in close)),
            'fixed_within_trial_still_close_at_2': sum(bool(value <= 2.) for (i, j), value in zip(close, d) if trials[i] == trials[j]),
            'fixed_pairs_separated_fraction': float((d > 2.).mean()) if len(d) else None,
            'white_noise_only_max_distance_95': noise_threshold,
            'fixed_pairs_above_window_noise_95': int((d > noise_threshold).sum()),
            'distance_rule': 'max per-lag whitened 15D RMS; contains all shorter-window evidence',
            'fresh_nearest_fraction_at_1p414': float((best[finite] <= np.sqrt(2)).mean()),
            'fresh_nearest_fraction_at_2': float((best[finite] <= 2.).mean()),
            'fresh_nearest_fraction_at_3': float((best[finite] <= 3.).mean()),
            'within_trial_eligible_endpoints': int(np.isfinite(best_within).sum()),
            'within_trial_near_fraction_at_2': float((best_within[np.isfinite(best_within)] <= 2.).mean())
                if np.isfinite(best_within).any() else None}
    ordered = sorted(close, key=lambda p: pair_distance(b, [p], 0.)[0])
    examples = []
    for i, j in ordered:
        examples.append({'a': b['records'][i], 'b': b['records'][j],
            'q_a_mm': b['q'][i].tolist(), 'q_b_mm': b['q'][j].tolist(),
            'same_trial': b['records'][i]['trial'] == b['records'][j]['trial'],
            'window_distance': {str(w): float(peak_pair_distance(b, [(i, j)], w)[0]) for w in WINDOWS},
            'window_mean_distance_exploratory': {str(w): float(pair_distance(b, [(i, j)], w)[0]) for w in WINDOWS}})
    # Same current nominal position, different preceding histories; maximum
    # field disparity, with within-trial evidence separated from cross-trial.
    same_position = []
    q = b['q']; field = b['x']['0.0']; trials = np.array([r['trial'] for r in b['records']]); times = np.array([r['time_s'] for r in b['records']])
    for i in range(len(q)):
        near = np.flatnonzero((np.linalg.norm(q-q[i], axis=1) <= .15) &
            ((trials != trials[i]) | (np.abs(times-times[i]) >= 3.)))
        if not len(near): continue
        d = np.sqrt(np.mean((field[near]-field[i])**2, 1)); j = int(near[np.argmax(d)])
        if i < j: same_position.append({'a': b['records'][i], 'b': b['records'][j],
            'q_a_mm': q[i].tolist(), 'q_b_mm': q[j].tolist(), 'distance': float(d.max()),
            'same_trial': int(trials[i]) == int(trials[j])})
    same_position.sort(key=lambda r: r['distance'], reverse=True)
    np.savez_compressed(OUT / f'{name}_bank.npz', q=q, **{'w'+k.replace('.', 'p'): x for k, x in b['x'].items()})
    dump(OUT / f'{name}_bank_metadata.json', {'records': b['records'], 'source_noise_whitening': True})
    return {'endpoints': len(q), 'before_cap': b['before_cap'], 'windows': hist,
            'examples': examples[:20], 'all_fixed_current_pairs': examples,
            'same_position_history_candidates': same_position[:20],
            'same_trial_history_candidates': [r for r in same_position if r['same_trial']][:20]}


def candidates(data, manifest, rows, sigma):
    models = {'static': MagneticForward.load(ROOT / 'analysis/r361_command_forward_20261001/models/instantaneous.json'),
              'joint': MagneticForward.load(ROOT / 'analysis/r361_command_forward_20261001/models/dynamic.json'),
              'axis': AxisForward.load(VIRTUAL / 'models/generator.json')}
    reference = np.vstack([a['q'][a['primary']][::8] for a in rows])
    result = []
    for i, info in enumerate(manifest['trials']):
        start, end = data['offsets'][i:i+2]; q = data['q'][start:end]; t = data['t'][start:end]
        primary = q[:, 2] >= .4
        prediction = np.stack([m.rollout(q, t) for m in models.values()])
        disagreement = np.sqrt(np.mean(np.var(prediction / sigma, axis=0), axis=1))
        inside = models['axis'].support(q)
        speed = np.linalg.norm(np.diff(q, axis=0), axis=1) / np.diff(t)
        dist = np.concatenate([np.min(np.linalg.norm((part[:, None, :]-reference[None, :, :]) / [1, 1, .6], axis=2), axis=1)
                               for part in np.array_split(q[primary], max(1, int(primary.sum() / 128)))])
        qualified = bool(inside[primary].mean() >= .9 and speed.max() <= 2.0001 and np.abs(q[:, :2]).max() <= 4.5 and q[:, 2].max() <= 2.3)
        result.append({**info, 'candidate_index': i, 'qualified_screen': qualified,
            'inside_training_fraction': float(inside[primary].mean()), 'max_speed_mm_s': float(speed.max()),
            'model_disagreement_p95_noise_units': float(np.quantile(disagreement[primary], .95)),
            'model_disagreement_mean_noise_units': float(disagreement[primary].mean()),
            'source_position_novelty_p95': float(np.quantile(dist, .95)),
            'max_depth_mm': float(q[:, 2].max())})
    ranked = sorted(result, key=lambda r: r['model_disagreement_p95_noise_units'], reverse=True)
    selected = []; counts = {}
    for r in ranked:
        if r['qualified_screen'] and counts.get(r['family'], 0) < 2:
            selected.append(r); counts[r['family']] = counts.get(r['family'], 0) + 1
    dump(OUT / 'candidate_experiments.json', {'ranked': ranked, 'selected': selected,
        'criterion': 'Source-only model-hypothesis disagreement P95, then at most two per action family; no target/test outcomes',
        'uncertainty_claim': 'Mechanism disagreement, NOT calibrated posterior uncertainty or confidence',
        'executable_stage_plan': False, 'contact_z_and_hardware_envelope_still_required': True,
        'hardware_operated': False})


def two_contact(sigma):
    m = AxisForward.load(VIRTUAL / 'models/generator.json')
    rows = []
    for separation in (0., .5, 1., 2., 4., 6.):
        for angle in ((0.,) if separation == 0 else (0., 45., 90.)):
            offset = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]) * separation / 2
            for z1 in (.6, 1.4, 2.2):
                for z2 in (.6, 1.4, 2.2):
                    if separation == 0 and z1 != z2: continue
                    p1 = np.r_[-offset, z1]; p2 = np.r_[offset, z2]
                    derivatives = []
                    for q in (p1, p2):
                        for axis in range(3):
                            d = np.zeros(3); d[axis] = .02
                            derivatives.append((m.static([q+d])[0]-m.static([q-d])[0]) / .04 / sigma)
                    jac = np.column_stack(derivatives); singular = np.linalg.svd(jac, compute_uv=False)
                    rank = int((singular > max(singular[0]*1e-8, 1e-9)).sum())
                    rows.append({'separation_mm': separation, 'angle_deg': angle,
                        'geometry_status': 'coincident analytical limit' if separation == 0 else
                            'nominal 5mm tip-body overlap risk; mathematical diagnostic only' if separation < 5 else
                            'nominal tip diameters separated; actual mounting clearance unverified',
                        'depths_mm': [z1, z2], 'rank': rank, 'singular_values_noise_per_mm': singular.tolist(),
                        'weakest_sensitivity_noise_per_mm': float(singular[-1]),
                        'condition_number': float(singular[0]/singular[-1]) if rank == 6 else None})
    grouped = {str(s): {'cases': len(ss := [r for r in rows if r['separation_mm'] == s]),
        'rank6_cases': sum(r['rank'] == 6 for r in ss),
        'weakest_sensitivity_median': float(np.median([r['weakest_sensitivity_noise_per_mm'] for r in ss]))}
        for s in (0., .5, 1., 2., 4., 6.)}
    dump(OUT / 'two_contact_hypothesis.json', {'rows': rows, 'by_separation': grouped,
        'construction': 'additive_single_contact_sensitivity_hypothesis',
        'coupled_mechanics_simulation': False, 'physical_validation': False,
        'multi_contact_training_labels_created': False,
        'nominal_indenter_diameter_mm': 5., 'small_separations_are_not_executable_two_tip_experiments': True,
        'interpretation': 'Only a distinguishability screening prior. Full-rank additive Jacobian does not prove mechanically coupled two-touch identifiability; contact permutation, history and sensor noise still matter.'})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((VIRTUAL / 'synthetic_manifest.json').read_text(encoding='utf8'))
    sigma = np.asarray(manifest['noise_sigma_uT'])
    data = np.load(VIRTUAL / 'complex_synthetic.npz', allow_pickle=False)
    virtual = []
    for i, info in enumerate(manifest['trials']):
        start, end = data['offsets'][i:i+2]; q = data['q'][start:end]
        virtual.append({**info, 'q': q, 'b': data['b'][start:end], 't': data['t'][start:end],
            'primary': (q[:, 2] >= .4), 'reset': np.r_[True, np.zeros(end-start-1, bool)]})
    inventory = json.loads((ROOT / 'analysis/r361_command_forward_20261001/data_manifest.json').read_text(encoding='utf8'))
    real = [load_row({**r, 'path': ROOT / r['path']}) for r in inventory if r['role'] in ('train', 'validation', 'test')]
    groups = {'synthetic': virtual,
        'source_training': [a for a in real if a['device'] == 'source' and a['role'] == 'train'],
        'source_validation': [a for a in real if a['device'] == 'source' and a['role'] == 'validation'],
        'source_heldout': [a for a in real if a['device'] == 'source' and a['role'] == 'test'],
        'target_heldout': [a for a in real if a['device'] == 'target' and a['role'] == 'test']}
    dump(OUT / 'protocol.json', {'windows_s': WINDOWS, 'noise_sigma_source_uT': sigma.tolist(),
        'distance': 'MAX across causal lags of the 15-channel noise-whitened RMS; whole-window RMS secondary only. No DTW, future frames or pairwise time shifts.',
        'position_separation': 'XY >=1 mm OR |Z difference| >=0.3 mm',
        'thresholds': [float(np.sqrt(2)), 2., 3.], 'primary_threshold': 2.,
        'same_trial_time_exclusion_s': 3., 'fixed_pair_pool': 'same endpoints valid through 2 s; same current-confuser pairs across all window comparisons',
        'interpretation': 'Nominal command-position ambiguity, not independently measured material state. Source validation participated in forward model choice and is reported separately, never called new blind data. Cross-trial candidates may include origin/B0/timing differences. Gaussian independent white-noise window-length control is illustrative, not calibrated to colored material/sensor noise.',
        'target_parameter_selection': False, 'hardware': False})
    result = {}
    for name, rows in groups.items():
        result[name] = search(bank(rows, sigma), name)
        print('sequence search', name, result[name]['endpoints'], result[name]['windows']['0.0']['fixed_current_confuser_pairs'], flush=True)
    dump(OUT / 'report.json', {'datasets': result, 'models_retrained': False,
        'true_physical_nonidentifiability_proven': False, 'virtual_pairs_require_real_confirmation': True})
    candidates(data, manifest, groups['source_training'], sigma)
    two_contact(sigma)
    print('candidate ranking and two-contact screening completed', flush=True)


if __name__ == '__main__': main()
