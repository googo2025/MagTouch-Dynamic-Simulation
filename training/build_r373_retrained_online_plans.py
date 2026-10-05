"""File-only matched online plans for the direct source retraining."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from training.waypoint_xyz_protocol_r136 import compile_waypoint_path, motion_sha256, path_sha256

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT/'config/paper_plans_r136'
OUT = ROOT/'analysis/r373_retrained_ui_online_20261001'
PRESET = '5×5｜源5 mm模型｜充分训练XY＋Z R373（实测）'
FAMILY = 'DirectPlateauReconstructionR373'
BOUNDS = {'x': (0., 305.), 'y': (0., 305.), 'z': (0., 320.)}
EXACT_GEOMETRY = '9c27fe50d6d0433fe1085b204780a9f5e3563e3e9326f0cc99b6cf5e8b6cbaa9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    return json.loads((DIR/name).read_text(encoding='utf8'))


def make(side):
    if side not in ('SOURCE', 'TARGET'):
        raise ValueError(side)
    standard = load(f'R354_{side}_R348_FINAL_ONLINE.json')
    exact = load(f'R355_{side}_R348_EXACT_SPACE_DEPTH_SLOW_ONLINE.json')
    xf = ROOT/'analysis/r372_direct_source_plateau_20261001/xy/freeze.json'
    zf = ROOT/'analysis/r372_direct_source_plateau_20261001/z/freeze.json'
    xy, z = json.loads(xf.read_text(encoding='utf8')), json.loads(zf.read_text(encoding='utf8'))
    if xy['status'] != 'COMPLETED_PLATEAU' or z['status'] != 'COMPLETED_PLATEAU':
        raise ValueError('Do not issue online plans for unfinished weights')
    plan = {k: copy.deepcopy(v) for k, v in standard.items() if k != 'items'}
    plan.pop('paper_protocol_r348', None)
    plan.update(plan_name=f'R373_{side}_R372_PLATEAU_ONLINE', paper_protocol_r373='plateau_trained_online_v1',
        paper_model_preset=PRESET, paper_data_role='new_weights_online_test_not_training',
        paper_purpose='Original-data plateau-trained XY/Z full system: shallow/static/pureZ/radial/circle/compound and the exact 29-waypoint 3D path. Save every Trial. Compare old/new on the same raw stream later.',
        paper_selected_waypoint_sha256=EXACT_GEOMETRY, paper_geometry_id='SPACE_DEPTH',
        original_motion_templates=[standard['plan_name'], exact['plan_name']])
    original = standard['items'][:7]+[exact['items'][2]]+[standard['items'][-1]]
    rows = []
    for i, item in enumerate(original, 1):
        row = copy.deepcopy(item)
        row.pop('paper_protocol_r348', None)
        row.pop('paper_protocol_r355', None)
        row.update(group=f'R373_{side}_{i:02d}_{row["action"].upper()}', status='PENDING',
            required_model_preset=PRESET, required_model_family=FAMILY,
            required_xy_mlp_artifact_sha256=xy['model_sha256'],
            required_depth_increment_freeze_sha256=sha(zf),
            required_r372_xy_freeze_sha256=sha(xf), required_r372_z_freeze_sha256=sha(zf),
            required_z_increment_artifact_sha256=z['model_sha256'],
            paper_protocol_r373='plateau_trained_online_v1',
            prediction_folder_tag='R373_R372_plateau_trained_online',
            research_role='new_weights_online_test_not_training')
        if row['action'] == 'waypoint_xyz':
            if path_sha256(row['waypoints_xyz_mm']) != EXACT_GEOMETRY:
                raise ValueError('Wrong user-selected complex 3D path')
            row['indent_mm'] = max(p['depth_mm'] for p in row['waypoints_xyz_mm'])
            if motion_sha256(row) != row['required_motion_sha256']:
                raise ValueError('Original motion changed')
            compile_waypoint_path(row, machine_bounds=BOUNDS)
        rows.append(row)
    plan['items'] = rows
    return plan


def save_new(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf8')
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError('Do not replace an issued online plan: '+str(path))
    if not path.exists():
        path.write_bytes(raw)


def main():
    OUT.mkdir(exist_ok=True)
    entries = {}
    for side in ('SOURCE', 'TARGET'):
        plan = make(side)
        filename = plan['plan_name']+'.json'
        save_new(DIR/filename, plan)
        dry = compile_waypoint_path(plan['items'][7], machine_bounds=BOUNDS)
        entries[plan['plan_name']] = {'filename': filename, 'sha256': sha(DIR/filename),
            'rows': len(plan['items']), 'profile': plan['paper_target_profile_id'],
            'center_printer_mm': plan['reference_center_printer_mm'],
            'contact_z_mm': plan['reference_contact_z_mm'], 'complex_dry_run': dry,
            'standard_motion_unchanged': True}
        print(filename, 'rows', len(plan['items']), 'path duration s', round(dry['estimated_seconds_all_repeats_excluding_approach'], 1), flush=True)
    manifest = {'schema': 'MagTouch-R373-OnlinePlans-1', 'plans': entries,
                'hardware_operated': False, 'old_models_and_plans_changed': False}
    save_new(DIR/'r373_plan_manifest.json', manifest)
    save_new(OUT/'plan_manifest.json', manifest)


if __name__ == '__main__':
    main()
