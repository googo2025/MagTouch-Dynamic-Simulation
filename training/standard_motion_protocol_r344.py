"""Whole-Trial roles: complex paths and Target never enter fitting/selection."""
import argparse
import json
from pathlib import Path
from training.analyze_z_evidence_r334 import ROOT, SOURCE_D, VALIDATION_IDS, dump, sha

OUT = ROOT/'analysis/r344_standard_motion_20260929'
STANDARD = {'angle_scan', 'circle_scan', 'grid_points', 'compound', 'z_path'}


def is_complex(path, meta):
    return (meta.get('trial_parameters', {}).get('command_mode') == 'waypoint_xyz'
            or 'waypoint_xyz_n' in str(path) or 'R136_' in str(path) or 'R208_' in str(path))


def assert_fit_allowed(path, role):
    path = Path(path).resolve()
    meta = json.loads((path/'meta.json').read_text(encoding='utf-8'))
    if role not in ('train', 'selection'):
        raise ValueError('Explicit fitting role required')
    if 'had-239' not in str(path) or 'Yelow' in str(path) or is_complex(path, meta):
        raise ValueError('Complex-family/Target fitting is forbidden: '+str(path))
    if meta['trial_parameters']['command_mode'] not in STANDARD:
        raise ValueError('Only declared Source standard motion may fit')
    return meta


def build():
    destination = OUT/'dataset_roles.json'
    if destination.exists():
        raise FileExistsError('Preserve declared roles')
    rows = []
    for p in sorted((ROOT/'data').rglob('meta.json')):
        m = json.loads(p.read_text(encoding='utf-8')); tp = m.get('trial_parameters', {})
        mode = tp.get('command_mode'); path = p.parent
        if 'had-239' not in str(path) and 'Yelow' not in str(path): continue
        target = 'Yelow' in str(path); number = int(tp.get('trial_id', 0))
        if mode == 'waypoint_xyz':
            role = 'complex_final_evaluation'
        elif mode not in STANDARD or not (path/'live_predictions.csv').exists():
            continue
        elif is_complex(path, m):
            role = 'complex_family_excluded'
        elif target:
            role = 'target_evaluation_only'
        elif path.parent == SOURCE_D:
            role = 'selection' if number in VALIDATION_IDS else 'train'
        elif 'R332_SOURCE_UI_ACCEPTANCE' in str(path):
            role = 'selection'
        elif ('R132_A_HAD239_R130_NEURAL_BLIND_DAY' in str(path)
              or 'R132_C_HAD239_HISTORY_CARDINAL' in str(path)
              or 'R133_RETRAIN_HAD239_ONLINE_REPLICATION' in str(path)):
            role = 'train'
        else:
            role = 'other_source_not_fit'
        raw_names = ['meta.json', 'magnetic_raw.csv', 'live_predictions.csv', 'events.csv', 'stage_commands.csv']
        raw = [{'path': (path/n).relative_to(ROOT).as_posix(), 'sha256': sha(path/n)}
               for n in raw_names if (path/n).exists()]
        row = {'path':path.relative_to(ROOT).as_posix(), 'trial':number,
               'session':tp.get('session_id'), 'mode':mode, 'target':target,
               'role':role, 'raw_manifest':raw}
        if role in ('train', 'selection'): assert_fit_allowed(path, role)
        rows.append(row)
    from collections import Counter
    document = {'schema':'R344-WholeTrialRoles-1', 'records':rows,
        'counts':dict(Counter(r['role'] for r in rows)),
        'complex_entire_trial_excluded':True, 'complex_family_excluded':True,
        'complex_normalizer_support_epoch_threshold_selection_rows':0,
        'target_fit_rows':0, 'target_selection_rows':0,
        'qualification':'Source standard motion only; freeze before complex and Target scores',
        'historical_complex_response_layer_exposure':True,
        'prospective_blind':False,
        'scope':'new localization layer held out; historical response/gating not path-blind'}
    dump(destination, document)
    # A raw dependency list is not by itself proof of training. Inspect the
    # explicit old response partition, and record overlap without rewriting it.
    old = ROOT/'analysis/r300_source_strength_conditioned_direction_20260929/source_partition.json'
    partition = json.loads(old.read_text(encoding='utf-8'))
    complex_paths = {r['path'] for r in rows if r['role']=='complex_final_evaluation'}
    overlap = [r for r in partition['records'] if r['path'] in complex_paths]
    dump(OUT/'historical_response_exposure.json', {
        'partition':old.relative_to(ROOT).as_posix(), 'sha256':sha(old),
        'complex_overlap':overlap, 'new_localization_fit_overlap':0,
        'interpretation':'Old response layer has path exposure; no complete-system unseen claim.'})
    print(document['counts'], 'complex overlap with old response:', len(overlap), flush=True)
    return document


def load(role):
    document = json.loads((OUT/'dataset_roles.json').read_text(encoding='utf-8'))
    values = [r for r in document['records'] if r['role']==role]
    for r in values:
        if role in ('train', 'selection'): assert_fit_allowed(ROOT/r['path'], role)
        for item in r['raw_manifest']:
            if sha(ROOT/item['path']) != item['sha256']:
                raise ValueError('Raw dependency changed: '+item['path'])
    return values


if __name__ == '__main__':
    build()
