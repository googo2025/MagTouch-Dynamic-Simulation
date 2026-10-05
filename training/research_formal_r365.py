"""Frozen MagTouch module chain on synthetic fields; missing-node real replay.

No alternative inverse is trained, no hardware is imported or operated, and
no command or reference position is passed into a reconstruction engine.
10 Hz is a controlled research replay rate, not saved-online parity.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from training.run_forward_r361 import ROOT, load_row, dump, sha
from training.forward_complex_r363 import AxisForward
from training.sparse_field_r364 import completion_features

OUT = ROOT / 'analysis/r365_formal_closed_loop_20261001'
VIRTUAL = ROOT / 'analysis/r363_complex_forward_20261001'
SPARSE = ROOT / 'analysis/r364_sparse_field_20261001'
sys.path.insert(0, str(ROOT / 'code'))
from live_fast_slow_reconstruction_r348 import FastSlowReconstructionEngineR348
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from live_algorithm_ui_r126 import PROFILE_KEY
from replay_backtest import load_profile
from training.had239_field_memory_depth_guard_runtime_r126 import DepthGuardFieldMemoryDecoder
from training.had239_guarded_neural_runtime_r166 import GuardedBranchMemoryDecoder
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder

LABELS = {'F': 'Static magnetic atlas', 'M': 'Atlas + slow field memory',
          'G': '+ magnetic-evidence branch guard', 'N': '+ frozen causal XY',
          'U': 'Full system, incremental Z disabled', 'Final': 'Frozen full system'}


def new_engine():
    p = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
    e = FastSlowReconstructionEngineR348(p, p['defaults']['threshold'],
        p['defaults']['min_cosine'], p['defaults']['smoothing_ms'], 70.)
    e.set_baseline(np.zeros(15))
    return e


def replay_fields(b, t, reset=None, eligible=None, components=False, disable_increment=False):
    """Only B, timestamps, baseline epochs and input validity enter inference."""
    b, t = np.asarray(b, float), np.asarray(t, float)
    if b.shape != (len(t), 15) or not len(t) or np.any(np.diff(t) <= 0):
        raise ValueError('increasing time and 15-channel field required')
    reset = np.zeros(len(t), bool) if reset is None else np.asarray(reset, bool)
    eligible = np.isfinite(b).all(1) if eligible is None else np.asarray(eligible, bool)
    engine = new_engine()
    if disable_increment:
        engine.final_refiner.selection = dict(engine.final_refiner.selection, gain=0.)
    names = ['Final'] if not components else list(LABELS)
    predictions = {k: np.full((len(t), 3), np.nan) for k in names}
    accepted = np.zeros(len(t), bool)
    states = np.zeros(len(t), np.int16)
    veto = np.zeros(len(t), bool)
    if components:
        no_increment = new_engine()
        no_increment.final_refiner.selection = dict(no_increment.final_refiner.selection, gain=0.)
        decoders = {'F': DepthGuardFieldMemoryDecoder.from_report(),
                    'M': DepthGuardFieldMemoryDecoder.from_report(),
                    'G': GuardedBranchMemoryDecoder.from_report(),
                    'N': CoverageGuardedNeuralDecoder.from_report()}
        decoders['F'].strength = 0.
    epoch = 0
    state_names = {}
    for i, now in enumerate(t):
        if reset[i]:
            epoch += 1
            engine.set_baseline(np.zeros(15))
            if components:
                no_increment.set_baseline(np.zeros(15))
                for decoder in decoders.values(): decoder.reset()
        result = engine.push(int(round(now * 1e9)) + 10_000_000_000,
                             b[i] if eligible[i] else np.zeros(15), eligible=bool(eligible[i]))
        accepted[i] = bool(result['valid'])
        if components:
            without = no_increment.push(int(round(now * 1e9)) + 10_000_000_000,
                b[i] if eligible[i] else np.zeros(15), eligible=bool(eligible[i]))
            assert bool(without['valid']) == bool(result['valid'])
        name = str(result['extra'].get('system_state', 'unknown'))
        if name not in state_names: state_names[name] = len(state_names)
        states[i] = state_names[name]
        if accepted[i]:
            predictions['Final'][i] = result['xyz']
            veto[i] = bool(result['extra'].get('branch_local_vetoed_by_field', False))
            if components:
                predictions['U'][i] = without['xyz']
                np.testing.assert_array_equal(predictions['U'][i, :2], predictions['Final'][i, :2])
        if components:
            # Fixed FINAL output permission is shared by all position ablations.
            # This isolates decoder components, not historical gate versions.
            for key, decoder in decoders.items():
                row = decoder.predict_frame(b[i] if eligible[i] else np.zeros(15),
                    time_ns=int(round(now * 1e9)) + 10_000_000_000,
                    b0_id=epoch, stream_id=0, contact_active=bool(accepted[i]))
                if row is not None: predictions[key][i] = row['xyz_mm']
    return {'valid': accepted, 'states': states, 'state_names': state_names,
            'veto': veto, **predictions}


def metrics(pred, q, t, mask, accepted):
    mask = np.asarray(mask, bool) & np.isfinite(q).all(1)
    use = mask & accepted & np.isfinite(pred).all(1)
    result = {'reference_frames': int(mask.sum()), 'output_frames': int(use.sum()),
              'coverage': float(use.sum() / mask.sum()) if mask.any() else None}
    if not use.any(): return result
    e = pred[use] - q[use]
    bias = e.mean(0)
    centered = e - bias
    result.update(xyz_mean_mm=float(np.linalg.norm(e, axis=1).mean()),
        xy_mean_mm=float(np.linalg.norm(e[:, :2], axis=1).mean()),
        xy_p95_mm=float(np.quantile(np.linalg.norm(e[:, :2], axis=1), .95)),
        z_mae_mm=float(np.abs(e[:, 2]).mean()), bias_xyz_mm=bias.tolist(),
        xyz_translation_removed_mm=float(np.linalg.norm(centered, axis=1).mean()),
        xy_translation_removed_mm=float(np.linalg.norm(centered[:, :2], axis=1).mean()),
        z_translation_removed_mm=float(np.abs(centered[:, 2]).mean()))
    pairs = use[1:] & use[:-1] & (np.diff(t) <= .25) & (np.abs(np.diff(q[:, 2])) < .002)
    jumps = np.abs(np.diff(pred[:, 2]))[pairs]
    result['constant_reference_z_steps'] = {'pairs': int(pairs.sum()),
        'p95_mm': float(np.quantile(jumps, .95)) if len(jumps) else None,
        'over_0p2_mm': int((jumps > .2).sum())}
    return result


def macro(rows, names):
    return {name: {k: float(np.mean([r[name][k] for r in rows if r[name].get(k) is not None]))
                  for k in ('coverage', 'xyz_mean_mm', 'xy_mean_mm', 'z_mae_mm',
                            'xyz_translation_removed_mm', 'xy_translation_removed_mm', 'z_translation_removed_mm')
                  if any(r[name].get(k) is not None for r in rows)} for name in names}


def synthetic():
    data = np.load(VIRTUAL / 'complex_synthetic.npz', allow_pickle=False)
    manifest = json.loads((VIRTUAL / 'synthetic_manifest.json').read_text(encoding='utf8'))
    generator = AxisForward.load(VIRTUAL / 'models/generator.json')
    atlas = CoverageGuardedNeuralDecoder.from_report().atlas
    rows = []
    fixed = {next(i for i, r in enumerate(manifest['trials']) if r['family'] == f)
             for f in {r['family'] for r in manifest['trials']}}
    for i, info in enumerate(manifest['trials']):
        a, stop = data['offsets'][i:i+2]
        q, t, empirical = data['q'][a:stop], data['t'][a:stop], data['b'][a:stop]
        destination = OUT / 'virtual_streams' / (info['id'] + '.npz')
        if destination.exists():
            rows.append(json.loads(destination.with_suffix('.json').read_text(encoding='utf8'))); continue
        contact = (q[:, 2] >= .4) & (q[:, 2] <= 3.)
        core = contact & generator.support(q)
        safe = q[:, 2] <= -.5
        row = {**info, 'primary': 'command >=0.4 mm and source training box; no gate conditioning',
               'scores': {}, 'safe_reference_frames': int(safe.sum())}
        saved = {'q': q, 't': t, 'empirical_field': empirical, 'core': core}
        arms = {'empirical_dynamic': empirical}
        if i in fixed:
            rollout = generator.rollout(q, t)
            residual = rollout - generator.static(q)
            noise = empirical - rollout
            atlas_field = np.zeros_like(empirical)
            atlas_field[contact] = atlas.forward(q[contact])
            arms.update(atlas_static_control=atlas_field + noise,
                        atlas_plus_history_control=atlas_field + residual + noise)
        for arm, b in arms.items():
            result = replay_fields(b, t, components=i in fixed)
            names = list(LABELS) if i in fixed else ['Final']
            row['scores'][arm] = {key: metrics(result[key], q, t, core, result['valid']) for key in names}
            row['scores'][arm]['safe_output_frames'] = int((safe & result['valid']).sum())
            saved.update({arm + '_' + key: result[key] for key in names})
            saved[arm + '_valid'] = result['valid']
        np.savez_compressed(destination, **saved)
        dump(destination.with_suffix('.json'), row); rows.append(row)
        print('virtual', i + 1, '/', len(manifest['trials']), info['family'], flush=True)
    dump(OUT / 'virtual_report.json', {'rows': rows, 'component_labels': LABELS,
        'macro_final': macro([{'Final': r['scores']['empirical_dynamic']['Final']} for r in rows], ['Final']),
        'fixed_component_cases': sorted(fixed), 'formal_model_retrained': False,
        'atlas_control_is_independent_physical_validation': False,
        'interpretation': 'Empirical generator is independent of frozen inverse atlas. Atlas+residual control isolates static-map mismatch but is an internal consistency check, not a new measurement.'})


def completion():
    manifest = json.loads((ROOT / 'analysis/r361_command_forward_20261001/data_manifest.json').read_text(encoding='utf8'))
    choices = json.loads((SPARSE / 'selection.json').read_text(encoding='utf8'))['source_selected_methods']
    rows = []
    for rec in manifest:
        if rec['role'] != 'test': continue
        destination = OUT / 'completion_streams' / (rec['id'] + '.npz')
        if destination.exists():
            rows.append(json.loads(destination.with_suffix('.json').read_text(encoding='utf8'))); continue
        a = load_row({**rec, 'path': ROOT / rec['path']})
        b, t = a['b'], a['t']
        reference = replay_fields(b, t, a['reset'], a['valid'])
        safe = a['valid'] & np.isfinite(a['q']).all(1) & (a['q'][:, 2] <= -.5)
        row = {k: rec[k] for k in ('id', 'device', 'mode')}
        row.update(full=metrics(reference['Final'], a['q'], t, a['primary'], reference['valid']),
                   full_safe_outputs=int((safe & reference['valid']).sum()), safe_frames=int(safe.sum()), nodes=[])
        saved = {'q': a['q'], 't': t, 'primary': a['primary'], 'b': b,
                 'full': reference['Final'], 'full_valid': reference['valid']}
        for node in range(5):
            hidden = np.arange(3 * node, 3 * node + 3)
            observed = np.setdiff1d(np.arange(15), hidden)
            method = choices[str(node)]
            file = SPARSE / 'models' / f'S{node+1}_{method}.npz'
            weights = np.load(file, allow_pickle=False)
            if method == 'lowrank':
                predicted = b[:, observed] @ weights['mapping'].T + weights['intercept']
            else:
                predicted = completion_features(b, t, a['reset'], observed,
                    method == 'nonlinear_causal', method != 'ridge_current') @ weights['coef']
            filled = b.copy(); filled[:, hidden] = predicted
            completed = replay_fields(filled, t, a['reset'], a['valid'])
            # Use the existing acquisition-valid flags for paired comparison;
            # no hidden magnetic values or reference XYZ are inference inputs.
            paired = a['primary'] & reference['valid'] & completed['valid']
            item = {'held_node': node+1, 'method': method,
                'filled': metrics(completed['Final'], a['q'], t, a['primary'], completed['valid']),
                'paired_full': metrics(reference['Final'], a['q'], t, paired, reference['valid']),
                'paired_filled': metrics(completed['Final'], a['q'], t, paired, completed['valid']),
                'lost_reference_outputs': int((a['primary'] & reference['valid'] & ~completed['valid']).sum()),
                'new_outputs': int((a['primary'] & ~reference['valid'] & completed['valid']).sum()),
                'safe_output_frames': int((safe & completed['valid']).sum())}
            row['nodes'].append(item)
            saved[f'S{node+1}'] = completed['Final']; saved[f'S{node+1}_valid'] = completed['valid']
        np.savez_compressed(destination, **saved)
        dump(destination.with_suffix('.json'), row); rows.append(row)
        print('completion', len(rows), rec['id'], rec['mode'], flush=True)
    grouped = {}
    for device in ('source', 'target'):
        selected = [r for r in rows if r['device'] == device]
        grouped[device] = {'full': macro([{'full': r['full']} for r in selected], ['full'])['full'],
            'nodes': {str(n): macro([{'filled': item['filled'], 'paired_full': item['paired_full'],
                'paired_filled': item['paired_filled']} for r in selected for item in r['nodes']
                if item['held_node'] == n], ['filled', 'paired_full', 'paired_filled']) for n in range(1, 6)}}
    dump(OUT / 'completion_report.json', {'rows': rows, 'macro': grouped,
        'formal_model_retrained': False, 'target_fit_or_selection': False,
        'sampling': 'Same causal cached 10 Hz frames for full and completed field, not saved online parity',
        'missing_node_values_enter_completion': False, 'reference_xyz_enters_inference': False,
        'no_safe_stage_hint_provided': True, 'rate_transfer_and_completion_errors_both_possible': True})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--task', choices=['virtual', 'completion', 'all'], default='all')
    args = p.parse_args()
    for sub in ('virtual_streams', 'completion_streams'): (OUT / sub).mkdir(parents=True, exist_ok=True)
    freeze = ROOT / 'analysis/r348_fast_slow_residual_20260929/freeze.json'
    guard = {'formal_freeze': sha(freeze), 'dependencies':
        {d['path']: sha(ROOT / d['path']) for d in json.loads(freeze.read_text(encoding='utf8'))['dependencies']},
        'forward': sha(VIRTUAL / 'models/generator.json'), 'completion_selection': sha(SPARSE / 'selection.json')}
    dump(OUT / 'protocol.json', {'schema': 'MagTouch-R365-FormalFrozen-1', 'guard': guard,
        'component_comparison': 'F/M/G/N share final output permission. U uses the identical full engine with only incremental-Z gain disabled in memory; XY and permission equality asserted. Frozen artifact files unchanged.',
        'rate_hz': 10, 'training': False, 'hardware': False,
        'offsets': 'Raw metrics primary; one XYZ translation per record diagnostic only; no time shift or smoothing'})
    started = time.perf_counter()
    if args.task in ('virtual', 'all'): synthetic()
    if args.task in ('completion', 'all'): completion()
    assert guard['formal_freeze'] == sha(freeze)
    for name, digest in guard['dependencies'].items(): assert sha(ROOT / name) == digest
    dump(OUT / f'verification_{args.task}.json', {'formal_freeze_unchanged': True,
        'formal_dependencies_unchanged': True, 'elapsed_s': time.perf_counter() - started,
        'hardware_operated': False, 'paper_modified': False})


if __name__ == '__main__': main()
