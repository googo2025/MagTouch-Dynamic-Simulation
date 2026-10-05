"""Post-freeze three-arm full-chain evaluation, not a training selector.

R373 -> selected XY only -> the same selected XY plus a qualified Z head.
The first two streams must already exist in the R379 evaluation cache. This
avoids changing input/permission conventions between the two training stages.
No held-out stream is opened before both source-only freezes pass validation.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse, html, json, os, time
import numpy as np
from training.virtual_depth_expansion_r382 import OUT, ROOT, XYOUT, OLD, engine
from training.evaluate_virtual_expansion_r379 import REAL, quantile
from training.causal_depth_network_r337 import DepthMLP
from training.research_formal_r365 import metrics, macro
from training.train_source_to_plateau_r372 import sha, dump
from live_retrained_reconstruction_r373 import frozen_components, COMPONENTS

ARMS = ('baseline', 'xy_only', 'xy_z')
LABELS = {'baseline': '冻结原完整模型 R373', 'xy_only': '仅扩集 XY',
          'xy_z': '同一扩集 XY＋重训 Z'}


def check_freezes(xy, depth):
    """Pure gate; budget stops and an unqualified depth head are not usable."""
    assert xy['status'] == depth['status'] == 'COMPLETED_PLATEAU'
    assert not xy['target_fit_or_selection'] and not depth['target_fit_or_selection']
    assert not xy['production_replaced'] and not depth['production_replaced']
    assert depth['source_only'] and depth['qualified_candidate']
    assert depth['selected_xy_sha256'] == xy['model_sha256']
    assert depth['model_sha256']


def freezes(count):
    frozen_components()
    folder = OUT/f'arm_{count}'
    xyfolder = XYOUT/f'arm_{count}'
    xy = json.loads((xyfolder/'freeze.json').read_text(encoding='utf8'))
    depth = json.loads((folder/'freeze.json').read_text(encoding='utf8'))
    check_freezes(xy, depth)
    assert sha(xyfolder/'best_model.npz') == xy['model_sha256']
    assert sha(folder/'best_model.npz') == depth['model_sha256']
    assert sha(folder/'protocol.json') == depth['protocol_sha256']
    assert sha(folder/'feature_manifest.json') == depth['feature_manifest_sha256']
    p = json.loads((folder/'protocol.json').read_text(encoding='utf8'))
    assert p['xy_freeze_sha256'] == sha(xyfolder/'freeze.json')
    assert p['selected_xy_sha256'] == xy['model_sha256']
    assert not p['target_or_real_reserved_complex_opened_for_preparation']
    prior = json.loads((xyfolder/'evaluation_status.json').read_text(encoding='utf8'))
    assert prior['status'] == 'COMPLETE' and prior['trials'] == 82
    prior_protocol = json.loads((xyfolder/'evaluation_protocol.json').read_text(encoding='utf8'))
    assert prior_protocol['candidate_sha256'] == xy['model_sha256']
    assert prior_protocol['freeze_sha256'] == sha(xyfolder/'freeze.json')
    return xy, depth


def replay_input(a, e):
    """This inference function accepts no reference position or action label."""
    t = np.asarray(a['t'], float)
    b = np.asarray(a['b'], float)
    eligible = np.asarray(a['input_eligible'], bool)
    reset = np.asarray(a['reset'], bool)
    assert b.shape == (len(t), 15) and len(t) > 0 and np.all(np.diff(t) > 0)
    assert eligible.shape == reset.shape == t.shape
    assert np.isfinite(b[eligible]).all()
    e.set_baseline(np.zeros(15))
    pred = np.full((len(t), 3), np.nan)
    ok = np.zeros(len(t), bool)
    permission, support = [np.full(len(t), np.nan) for _ in range(2)]
    route = np.empty(len(t), dtype='U32')
    for i, now in enumerate(t):
        if reset[i]:
            e.set_baseline(np.zeros(15))
        r = e.push(int(round(now*1e9))+10_000_000_000,
                   b[i] if eligible[i] else np.zeros(15), eligible=bool(eligible[i]))
        ok[i] = bool(r['valid'])
        if ok[i]:
            pred[i] = r['xyz']
        extra = r['extra']
        route[i] = extra.get('localization_path', 'unknown')
        v = extra.get('localization_permission_score')
        permission[i] = np.nan if v is None else v
        v = extra.get('final_depth_evidence', {}).get('source_support_distance')
        support[i] = np.nan if v is None else v
    return {'pred': pred, 'ok': ok, 'permission': permission, 'support': support, 'route': route}


def new_candidate(count):
    e = engine(count, recorded=False)
    model = DepthMLP.load(OUT/f'arm_{count}/best_model.npz')
    np.testing.assert_array_equal(model.mean, e.final_refiner.model.mean)
    np.testing.assert_array_equal(model.scale, e.final_refiner.model.scale)
    for new, original in zip(model.params, e.final_refiner.model.params):
        assert new.shape == original.shape and np.isfinite(new).all()
        new.flags.writeable = False
    assert len(model.params) == 6
    e.final_refiner.model = model
    return e


def pair_changes(a, before, after):
    mask = a['primary']
    change = {'primary_outputs_lost': int((mask & before['ok'] & ~after['ok']).sum()),
              'primary_outputs_added': int((mask & ~before['ok'] & after['ok']).sum())}
    for key in ('permission', 'support'):
        use = mask & np.isfinite(before[key]) & np.isfinite(after[key])
        change[key+'_absolute_change'] = quantile(np.abs(after[key][use]-before[key][use]))
    return change


def trial(task):
    count, kind, entry = task
    xy, depth = freezes(count)
    folder = OUT/f'arm_{count}'
    dest = folder/'evaluation'/f"{kind}_{entry['id']}.npz"
    side = dest.with_suffix('.json')
    prior_file = XYOUT/f"arm_{count}/evaluation/{kind}_{entry['id']}.npz"
    prior_meta = json.loads(prior_file.with_suffix('.json').read_text(encoding='utf8'))
    path = REAL/'streams'/f"{entry['id']}_matched.npz" if kind == 'real' else OLD/'virtual'/f"{entry['id']}.npz"
    assert sha(path) == prior_meta['input_sha256']
    assert prior_meta['candidate_sha256'] == xy['model_sha256']
    if kind == 'real':
        for name, expected in entry['hashes'].items():
            assert sha(ROOT/entry['path']/name) == expected
    else:
        assert sha(path) == entry['sha256']
    if side.exists():
        record = json.loads(side.read_text(encoding='utf8'))
        assert record['xy_sha256'] == xy['model_sha256'] and record['z_sha256'] == depth['model_sha256']
        assert record['input_sha256'] == sha(path) and record['prior_stream_sha256'] == sha(prior_file)
        assert record['output_sha256'] == sha(dest)
        return record
    start = time.perf_counter()
    with np.load(path, allow_pickle=False) as f:
        a = {k: f[k].copy() for k in f.files}
    # Preserve the exact earlier R374/R379 controlled input contract. q is never
    # passed to the decoder; finite-reference eligibility is a logged convention,
    # explicitly retained here rather than silently changing the matched replay.
    if kind == 'real':
        a['input_eligible'] = a['valid'] & np.isfinite(a['q']).all(1)
    else:
        a['input_eligible'] = a['valid']
        a['safe'] = a['valid'] & np.isfinite(a['q']).all(1) & (a['q'][:, 2] <= -.2)
    models = {}
    with np.load(prior_file, allow_pickle=False) as f:
        for key in ('t', 'q', 'primary', 'safe', 'reset'):
            np.testing.assert_array_equal(a[key], f[key])
        for name, old in (('baseline', 'baseline'), ('xy_only', 'candidate')):
            models[name] = {key: f[old+'_'+key].copy() for key in ('pred', 'ok', 'permission', 'support')}
    inputs = {k: a[k] for k in ('t', 'b', 'input_eligible', 'reset')}
    models['xy_z'] = replay_input(inputs, new_candidate(count))
    all_common = np.logical_and.reduce([models[name]['ok'] for name in ARMS])
    scores, changes = {}, {}
    for name, v in models.items():
        scores[name] = metrics(v['pred'], a['q'], a['t'], a['primary'], v['ok'])
        scores[name]['safe_frames'] = int(a['safe'].sum())
        scores[name]['safe_outputs'] = int((a['safe'] & v['ok']).sum())
        for key, label in (('permission', 'permission_primary'), ('support', 'z_support_primary')):
            scores[name][label] = quantile(v[key][a['primary']])
        scores[name+'_all_common'] = metrics(v['pred'], a['q'], a['t'], a['primary'], all_common)
    for before, after in (('baseline', 'xy_only'), ('xy_only', 'xy_z'), ('baseline', 'xy_z')):
        key = before+'_to_'+after
        changes[key] = pair_changes(a, models[before], models[after])
        common = models[before]['ok'] & models[after]['ok']
        changes[key]['paired_scores'] = {
            name: metrics(models[name]['pred'], a['q'], a['t'], a['primary'], common)
            for name in (before, after)}
    shallow = np.isfinite(a['q']).all(1) & (a['q'][:, 2] > 0) & (a['q'][:, 2] < .4)
    err = np.linalg.norm(models['xy_z']['pred']-a['q'], axis=1)
    ix = np.flatnonzero(a['primary'] & models['xy_z']['ok'] & np.isfinite(err))
    ix = ix[np.argsort(err[ix])[-10:][::-1]]
    worst = [{'index': int(i), 'time_s': float(a['t'][i]), 'reference_xyz_mm': a['q'][i].tolist(),
              'outputs_mm': {name: models[name]['pred'][i].tolist() if models[name]['ok'][i] else None for name in ARMS},
              'permission': float(models['xy_z']['permission'][i]) if np.isfinite(models['xy_z']['permission'][i]) else None,
              'support': float(models['xy_z']['support'][i]) if np.isfinite(models['xy_z']['support'][i]) else None,
              'route': str(models['xy_z']['route'][i])} for i in ix]
    stream = {k: a[k] for k in ('t', 'q', 'primary', 'safe', 'reset')}
    for name, v in models.items():
        for key in ('pred', 'ok', 'permission', 'support'):
            stream[name+'_'+key] = v[key]
    np.savez_compressed(dest, **stream)
    record = {'id': entry['id'], 'kind': kind, 'device': entry.get('device', 'virtual'),
              'mode': entry.get('mode', entry.get('style')), 'scores': scores, 'changes': changes,
              'nominal_shallow': {'frames': int(shallow.sum()), 'outputs': {name: int((shallow & models[name]['ok']).sum()) for name in ARMS}},
              'worst_accepted_primary_frames': worst, 'xy_sha256': xy['model_sha256'],
              'z_sha256': depth['model_sha256'], 'input_sha256': sha(path),
              'prior_stream_sha256': sha(prior_file), 'output_sha256': sha(dest),
              'seconds': time.perf_counter()-start}
    dump(side, record)
    print('R382', count, kind, entry['id'], 'Z MAE', scores['xy_z'].get('z_mae_mm'), flush=True)
    return record


def grid_diagnostics(folder, rows):
    result = {}
    for device, ident in (('source', 'source_000002'), ('target', 'target_000011')):
        grid = next(r for r in rows if r['id'] == ident)
        biases = {m: grid['scores'][m]['bias_xyz_mm'] for m in ARMS}
        records = []
        for row in rows:
            if row['device'] != device or row['mode'] == 'none':
                continue
            with np.load(folder/'evaluation'/f"real_{row['id']}.npz", allow_pickle=False) as s:
                scores = {name: metrics(s[name+'_pred']-np.asarray(biases[name]), s['q'], s['t'],
                                       s['primary'], s[name+'_ok']) for name in ARMS}
            records.append({'id': row['id'], 'scores': scores})
        result[device] = {'calibration_trial_id': ident, 'offsets_mm': biases, 'records': records,
                          'macro': macro([r['scores'] for r in records], ARMS),
                          'diagnostic_only_not_used_for_inference_or_selection': True}
    return result


def render(count, report):
    body = ['<!doctype html><meta charset="utf-8"><title>虚拟扩集 XYZ 三臂回测</title>',
            '<style>body{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto;padding:24px;color:#17364c}table{width:100%;border-collapse:collapse;margin:18px 0}th,td{border:1px solid #cad6df;padding:10px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf1f7}.note{background:#fff3d9;padding:16px}pre{white-space:pre-wrap}</style>',
            f'<h1>{count}条自由三维虚拟路径：XY与Z扩集贡献分开核对</h1>',
            '<p>三臂沿用同一磁场与10 Hz受控输入约定：原完整R373、只换研究XY、固定同一研究XY再换研究Z。门控、磁图、历史状态与支持保护参数不变，但数值耦合按实际输出检验。</p>',
            '<p class="note">以下是冻结后的已有留出回放，不是新的原生在线实验，也不是新的盲测。Yellow不用于训练、阈值与检查点选择。逐Trial去平移与同场网格偏移均是独立诊断，不用于美化主要误差；浅层平台指令不是实际接触真值。</p>']
    for device, group in report['groups'].items():
        body.append('<h2>'+html.escape(device)+'</h2><table><tr><th>完整Trial等权指标</th>'+''.join('<th>'+LABELS[m]+'</th>' for m in ARMS)+'</tr>')
        for key, label in (('coverage', '输出覆盖率'), ('xy_mean_mm', '原始XY平均误差 mm'),
                           ('z_mae_mm', '原始Z MAE mm'), ('xyz_mean_mm', '原始XYZ平均误差 mm'),
                           ('xy_translation_removed_mm', '去常值XY诊断 mm')):
            cells = []
            for m in ARMS:
                v = group['macro'][m].get(key)
                cells.append('—' if v is None else f'{v*100:.2f}%' if key == 'coverage' else f'{v:.4f}')
            body.append('<tr><td>'+label+'</td>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
        body.append('<tr><td>安全抬离输出／总帧</td>'+''.join(f'<td>{group["safe_outputs"][m]} / {group["safe_frames"]}</td>' for m in ARMS)+'</tr></table>')
    body += ['<p>恒深Z步长、跳变次数、许可／支持度变化、浅层数量、共同输出误差、逐动作整体偏移与最差帧均保留在完整JSON，不以缺失输出记零误差。</p>',
             '<p><a href="evaluation_report.json">全部指标与失败片段</a> · <a href="evaluation_protocol.json">回测协议</a> · <a href="freeze.json">Z源侧冻结</a> · <a href="protocol.json">源侧完整训练协议</a></p>']
    (OUT/f'arm_{count}/index.html').write_text(''.join(body), encoding='utf8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, choices=(100, 500, 1000), default=100)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()
    xy, depth = freezes(args.count)  # Before any heldout records are opened.
    folder = OUT/f'arm_{args.count}'
    (folder/'evaluation').mkdir(exist_ok=True)
    real = json.loads((REAL/'manifest.json').read_text(encoding='utf8'))
    catalog = json.loads((OLD/'catalog.json').read_text(encoding='utf8'))
    virtual = [r for r in catalog['trials'] if r['role'] == 'test']
    assert len(real) == 18 and len(virtual) == 64
    protocol = {'schema': 'R382-ThreeArm-FrozenEvaluation-1', 'count': args.count,
                'xy_sha256': xy['model_sha256'], 'z_sha256': depth['model_sha256'],
                'xy_freeze_sha256': sha(XYOUT/f'arm_{args.count}/freeze.json'),
                'z_freeze_sha256': sha(folder/'freeze.json'),
                'prior_evaluation_protocol_sha256': sha(XYOUT/f'arm_{args.count}/evaluation_protocol.json'),
                'real_ids': [r['id'] for r in real], 'virtual_ids': [r['id'] for r in virtual],
                'arms': LABELS, 'same_magnetic_stream_and_input_contract': True,
                'input': 'Only time, B, explicit baseline resets and input eligibility enter inference; exact prior R374 real validity-and-finite-reference convention retained, virtual magnetic validity only',
                'cadence': '10Hz controlled replay; NOT native online outputs',
                'unchanged': ['action head', 'permission policy', 'magnetic atlas', 'slow state', 'branch guard', 'Z protection policy'],
                'no_coordinate_smoothing_or_posthoc_time_alignment': True,
                'raw_errors_and_translation_diagnostics_separate': True,
                'selection': 'original SOURCE-only 3 XY and 12 Z whole-Trial validation splits; never heldout evaluation',
                'baseline_xy_z_hashes': {k: v['model_sha256'] for k, v in COMPONENTS.items()},
                'target_tuning': False, 'production_replaced': False,
                'scope': 'Previously inspected real records remain excluded from fit/selection; not new blind acquisitions. Virtual paths share a fitted empirical generator.'}
    dest = folder/'evaluation_protocol.json'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf8')) == protocol
    else:
        dump(dest, protocol)
    dump(folder/'evaluation_status.json', {'status': 'RUNNING', 'pid': os.getpid()})
    tasks = [(args.count, 'real', r) for r in real]+[(args.count, 'virtual', r) for r in virtual]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(trial, tasks))
    groups = {}
    for device in ('source', 'target', 'virtual'):
        contact = [r for r in rows if r['device'] == device and r['mode'] != 'none']
        all_rows = [r for r in rows if r['device'] == device]
        groups[device] = {'trials': len(contact),
                          'macro': macro([r['scores'] for r in contact], (*ARMS, *(m+'_all_common' for m in ARMS))),
                          'safe_frames': sum(r['scores']['baseline']['safe_frames'] for r in all_rows),
                          'safe_outputs': {m: sum(r['scores'][m]['safe_outputs'] for r in all_rows) for m in ARMS},
                          'constant_reference_z_steps': {m: {
                              'pairs': sum(r['scores'][m].get('constant_reference_z_steps', {}).get('pairs', 0) for r in contact),
                              'over_0p2_mm': sum(r['scores'][m].get('constant_reference_z_steps', {}).get('over_0p2_mm', 0) for r in contact)} for m in ARMS}}
    report = {'status': 'COMPLETE', 'xy_freeze': xy, 'z_freeze': depth, 'groups': groups,
              'independent_grid_offset_diagnostic': grid_diagnostics(folder, rows), 'records': rows,
              'target_tuned': False, 'production_replaced': False, 'native_online': False}
    dump(folder/'evaluation_report.json', report)
    render(args.count, report)
    dump(folder/'evaluation_status.json', {'status': 'COMPLETE', 'pid': os.getpid(), 'trials': len(rows)})
    print(json.dumps(groups, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
