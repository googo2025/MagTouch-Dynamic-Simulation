"""Two frozen-weight arms on existing real streams; no fitting or selection.

The original full engine and the same engine with only the directly retrained
XY and incremental-Z weights. All other algorithms and coefficients stay fixed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/r372_direct_source_plateau_20261001'


def dependencies():
    base = Path(os.environ.get('MAGTOUCH_REPLAY_BASE', str(ROOT)))
    sys.path.insert(0, str(base/'tmp/r360_math_packages'))
    sys.path.insert(0, str(base))
    # Research dependencies resolve the old Windows manifest without editing it.
    if os.name != 'nt':
        from pathlib import PosixPath
        PosixPath.__fspath__ = lambda self: str(self).replace('\\', '/')
    return base


def freeze_checks():
    from training.run_forward_r361 import sha
    for component in ('xy', 'z'):
        f = json.loads((OUT/component/'freeze.json').read_text(encoding='utf8'))
        if f['status'] != 'COMPLETED_PLATEAU':
            raise ValueError('Both source models must have stopped before test access')
        if f['test_used_for_fitting_or_selection']:
            raise ValueError('Test-free selection required')
        if sha(OUT/component/f['model']) != f['model_sha256']:
            raise ValueError('Frozen new weight hash mismatch')


def replay(a, arm):
    from training.domain_randomization_r369 import ResearchXY
    from training.research_formal_r365 import new_engine
    from training.causal_depth_network_r337 import DepthMLP
    e = new_engine()
    if arm == 'new':
        model = ResearchXY.load(OUT/'xy/best_model.npz')
        e.memory_decoder.xy_model = ResearchXY(model)
        e.candidate_decoder.xy_model = ResearchXY(model)
        e.final_refiner.model = DepthMLP.load(OUT/'z/best_model.npz')
    pred = np.full((len(a['t']), 3), np.nan)
    accepted = np.zeros(len(a['t']), bool)
    for i, (now, b, eligible, reset) in enumerate(zip(a['t'], a['b'], a['valid'], a['reset'])):
        if reset:
            e.set_baseline(np.zeros(15))
        # No command, position target, path type or future field enters push.
        result = e.push(int(round(now*1e9))+10_000_000_000,
                        b if eligible else np.zeros(15), eligible=bool(eligible))
        accepted[i] = bool(result['valid'])
        if accepted[i]:
            pred[i] = result['xyz']
    return pred, accepted


def trial(rec):
    base = dependencies()
    from training.run_forward_r361 import load_row, sha, dump
    from training.research_formal_r365 import metrics
    for name, expected in rec['hashes'].items():
        if sha(base/rec['path']/name) != expected:
            raise ValueError('Original recording hash mismatch: '+rec['id'])
    a = load_row({**rec, 'path': base/rec['path']})
    started = time.perf_counter()
    old, old_ok = replay(a, 'old')
    new, new_ok = replay(a, 'new')
    common = old_ok & new_ok
    r = {k: rec[k] for k in ('id', 'device', 'mode', 'path')}
    r['seconds'] = time.perf_counter()-started
    r['old'] = metrics(old, a['q'], a['t'], a['primary'], old_ok)
    r['new'] = metrics(new, a['q'], a['t'], a['primary'], new_ok)
    r['old_common'] = metrics(old, a['q'], a['t'], a['primary'], common)
    r['new_common'] = metrics(new, a['q'], a['t'], a['primary'], common)
    r['safe_frames'] = int(a['safe'].sum())
    r['old_safe_output_frames'] = int((a['safe'] & old_ok).sum())
    r['new_safe_output_frames'] = int((a['safe'] & new_ok).sum())
    np.savez_compressed(OUT/'replay'/f"{rec['id']}.npz", t=a['t'], q=a['q'],
                        primary=a['primary'], safe=a['safe'], old=old, new=new,
                        old_ok=old_ok, new_ok=new_ok)
    dump(OUT/'replay'/f"{rec['id']}.json", r)
    print(rec['id'], 'old/new XYZ', r['old'].get('xyz_mean_mm'), r['new'].get('xyz_mean_mm'),
          'seconds', round(r['seconds'], 1), flush=True)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--legacy-root', type=Path)
    args = parser.parse_args()
    if args.legacy_root:
        os.environ['MAGTOUCH_REPLAY_BASE'] = str(args.legacy_root)
    base = dependencies()
    from training.run_forward_r361 import dump, sha
    from training.research_formal_r365 import macro
    freeze_checks()
    manifest = base/'analysis/r361_command_forward_20261001/data_manifest.json'
    rows = [r for r in json.loads(manifest.read_text(encoding='utf8')) if r['role'] == 'test']
    if len(rows) != 30:
        raise ValueError('Expected 10 source and 20 target retrospective test records')
    (OUT/'replay').mkdir(exist_ok=True)
    dump(OUT/'replay_protocol.json', {
        'arms': ['old frozen full system', 'same full system with retrained XY and Z increment'],
        'manifest_sha256': sha(manifest), 'records': [r['id'] for r in rows],
        'sampling': 'existing 10Hz causal last-available observation replay; not new online acquisition',
        'source_model_selection_completed_before_test_access': True,
        'test_used_to_select': False, 'new_models': {
            c: sha(OUT/c/'best_model.npz') for c in ('xy', 'z')},
        'translation': 'per-Trial constant XYZ translation diagnostics only; raw scores are primary',
        'no_rotation_scaling_time_alignment_or_smoothing': True,
        'target_used_to_fit_or_select': False, 'hardware_or_formal_UI_changed': False})
    started = time.perf_counter()
    if args.workers == 1:
        result = [trial(r) for r in rows]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            result = list(pool.map(trial, rows))
    report = {'status': 'COMPLETE', 'records': result, 'seconds': time.perf_counter()-started,
              'groups': {}, 'test_used_to_fit_or_select': False}
    for device in ('source', 'target'):
        group = [r for r in result if r['device'] == device]
        report['groups'][device] = {'trials': len(group),
            **macro(group, ['old', 'new', 'old_common', 'new_common']),
            'safe_frames': sum(r['safe_frames'] for r in group),
            'old_safe_outputs': sum(r['old_safe_output_frames'] for r in group),
            'new_safe_outputs': sum(r['new_safe_output_frames'] for r in group),
            'by_mode': {mode: macro([r for r in group if r['mode'] == mode], ['old', 'new'])
                        for mode in sorted(set(r['mode'] for r in group))}}
    dump(OUT/'replay_report.json', report)
    print(json.dumps(report['groups'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
