"""Recapture selected-XY-dependent Z features and train a research Z head.

No production runtime is edited. Read-only instrumentation wraps its real
predict/adjudication calls, and compact validation is checked against those
outputs. All source Trial roles precede recapture; no target/test input is read.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse, json, os, sys, time
import numpy as np
from training.train_source_to_plateau_r372 import (
    Plateau, forward, gradients_z, checkpoint, save_model, dump, sha, val_z,
    score_z, score_phases, fast_depth,
)
from training.virtual_expansion_r379 import ROOT, OUT as XYOUT, OLD, REAL, ResearchXY
from training.standard_motion_protocol_r344 import assert_fit_allowed
from training.fast_slow_residual_r348 import phase_masks
from training.unified_depth_runtime_r341 import UnifiedDepthAdjudicator
from training.build_trajectory_dataset_v1 import (
    assemble, command_segments, load_magnetic, read_rows, reference_origin,
)
sys.path.insert(0, str(ROOT/'code'))
from live_retrained_reconstruction_r373 import DirectTrainedEngineR373, COMPONENTS, frozen_components
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from live_algorithm_ui_r126 import PROFILE_KEY
from replay_backtest import load_profile

OUT = ROOT/'analysis/r382_xy_conditioned_depth_expansion_20261002'

class RecordingModel:
    def __init__(self, model):
        self.model = model
        self.last = None
    def predict(self, x):
        self.last = np.asarray(x, float).copy()
        return self.model.predict(x)

class RecordingAdjudicator:
    def __init__(self, tracker):
        self.tracker = tracker
        self.last = None
    def reset(self):
        self.last = None
        return self.tracker.reset()
    def step(self, time_ns, z, cost, phi, proposal, support_distance, *, token=0):
        self.last = (int(time_ns), np.asarray(z).copy(), np.asarray(cost).copy(),
                     np.asarray(phi).copy(), float(proposal), float(support_distance), token)
        return self.tracker.step(time_ns, z, cost, phi, proposal, support_distance, token=token)

def engine(count, recorded=True):
    xyfile = XYOUT/f'arm_{count}/best_model.npz'
    profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
    e = DirectTrainedEngineR373(profile, profile['defaults']['threshold'],
                               profile['defaults']['min_cosine'], profile['defaults']['smoothing_ms'], 70.)
    e.memory_decoder.xy_model = ResearchXY.load(xyfile)
    e.candidate_decoder.xy_model = ResearchXY.load(xyfile)
    if recorded:
        e.final_refiner.model = RecordingModel(e.final_refiner.model)
        e.final_refiner.anchor.final = RecordingAdjudicator(e.final_refiner.anchor.final)
    e.set_baseline(np.zeros(15))
    return e

def protocol(count):
    frozen_components()
    xyfolder = XYOUT/f'arm_{count}'
    f = json.loads((xyfolder/'freeze.json').read_text(encoding='utf8'))
    assert f['status'] == 'COMPLETED_PLATEAU' and not f['target_fit_or_selection']
    assert sha(xyfolder/'best_model.npz') == f['model_sha256']
    manifest = json.loads((REAL/'data_manifest.json').read_text(encoding='utf8'))
    assert not manifest['target_measurements_included'] and not manifest['test_measurements_included']
    records = []
    for r in manifest['z']['records']:
        meta = assert_fit_allowed(ROOT/r['path'], r['role'])
        for item in r['raw_manifest']:
            assert sha(ROOT/item['path']) == item['sha256']
        records.append({**r, 'role': 'validation' if r['role'] == 'selection' else 'train',
                        'id': 'R382_SOURCE_'+sha(ROOT/r['path']/'meta.json')[:16],
                        'mode': meta['trial_parameters']['command_mode'], 'device': 'source'})
    assert sum(r['role'] == 'train' for r in records) == 45
    assert sum(r['role'] == 'validation' for r in records) == 12
    assert all(r['mode'] != 'waypoint_xyz' for r in records)
    p = {'schema': 'R382-SelectedXY-DepthExpansion-1', 'count': count, 'seed': 348,
         'architecture': 'unchanged 59-32-32-1 tanh bounded incremental Z',
         'initial_z_sha256': COMPONENTS['z']['model_sha256'], 'selected_xy_sha256': f['model_sha256'],
         'xy_freeze_sha256': sha(xyfolder/'freeze.json'), 'source_manifest_sha256': sha(REAL/'data_manifest.json'),
         'source_records': records, 'virtual_ids': [f'V370_train_{i:04d}' for i in range(count)],
         'selected_xy_stays_fixed': True, 'z_normalization': 'frozen R373; not refit',
         'source_input_sampling': '10Hz causal last-available magnetic assembly; all eligible frames passed through, not conditioned on command or saved XYZ',
         'training_mask': 'same nominal calibrated range0.4-3mm; requires real permitted Z-feature call; not a contact-recall label',
         'training_stride': 2, 'validation_stride': 1, 'static_samples': 565,
         'source_targets': 'command Z - no-increment frozen teacher adjudicated Z, subtract mean on each TRAIN Trial only, clip+-0.4mm',
         'virtual_targets': 'known generated Z - no-increment frozen teacher adjudicated Z, no per-Trial centering, clip+-0.4mm',
         'loss': 'weighted residual MSE +0.5 zero-increment MSE +0.5 within-Trial increment-difference MSE +0.0005 weight L2',
         'loss_mass': {'source_dynamic': .4, 'virtual_dynamic': .4, 'static': .2},
         'dynamic_weighting': 'equal total mass per complete Trial within each domain',
         'pairs': 'adjacent retained training frames, same baseline epoch, 0<gap<=0.35s',
         'initial_lr': .0002, 'minimum_lr': 1e-6, 'patience': 50, 'lr_factor': .5,
         'relative_min_delta': 1e-4, 'initial_max_epochs': 1000,
         'true_epoch': 'complete shuffled traversal without replacement of every prepared training sample',
         'selection': 'original12 SOURCE validation Trials, fixed R372 Z objective/phase protections, benchmark selectedXY+original R373 Z on these same recaptured streams',
         'validation_every_epoch': True,
         'qualification': 'same R372 category/phase raw and centered MAE protections; objective>=1% improvement; else no qualified Z candidate',
         'normalization_threshold_or_generator_selected_with_target': False,
         'target_or_real_reserved_complex_opened_for_preparation': False,
         'generation': 'already frozen R370 rich paths; not real path templates',
         'baseline_control': 'selectedXY+frozen R373 Z, no new real-only optimization arm',
         'production_replaced': False, 'hardware_operated': False,
         'scope': 'this predeclared Z stage applies to each100/500/1000 XY arm; no arm chosen using target outcome'}
    folder = OUT/f'arm_{count}'
    folder.mkdir(parents=True, exist_ok=True)
    for sub in ('features', 'inputs'):
        (folder/sub).mkdir(exist_ok=True)
    file = folder/'protocol.json'
    if file.exists():
        assert json.loads(file.read_text(encoding='utf8')) == p
    else:
        dump(file, p)
    return p

def source_input(count, row):
    folder = OUT/f'arm_{count}/inputs'
    dest = folder/(row['id']+'.npz')
    if dest.exists():
        with np.load(dest, allow_pickle=False) as f:
            return {k: f[k].copy() for k in f.files}, dest
    path = ROOT/row['path']
    meta = json.loads((path/'meta.json').read_text(encoding='utf8'))
    segments = command_segments(read_rows(path/'stage_commands.csv'))
    full, _ = assemble(load_magnetic(path/'magnetic_raw.csv'), segments,
                       read_rows(path/'events.csv'), reference_origin(meta))
    ix = np.arange(0, len(full['t_ns']), 4)
    q = full['xyz_mm'][ix]
    valid = full['input_valid'][ix]
    a = {'q': q, 'b': full['delta_b_uT'][ix], 'valid': valid,
         't': (full['t_ns'][ix]-full['t_ns'][0])/1e9,
         'reset': np.r_[True, np.diff(full['b0_id'][ix]) != 0],
         'primary': valid & np.isfinite(q).all(1) & (q[:, 2] >= .4) & (q[:, 2] <= 3.)}
    np.savez_compressed(dest, **a)
    return a, dest

def capture_arrays(a, count, check_uninstrumented=False):
    e = engine(count)
    control = engine(count, recorded=False) if check_uninstrumented else None
    no_increment = UnifiedDepthAdjudicator(e.final_refiner.anchor.policy)
    n = len(a['t'])
    x = np.full((n, 59), np.nan)
    old_z, teacher_z, proposal = [np.full(n, np.nan) for _ in range(3)]
    valid = np.zeros(n, bool)
    accepted = valid.copy()
    xyz = np.full((n, 3), np.nan)
    costs, phis, distances = [], [], []
    epoch = np.cumsum(a['reset']).astype(np.int64)
    times = np.rint(a['t']*1e9).astype(np.int64)+10_000_000_000
    zgrid = None
    for i, now in enumerate(times):
        if a['reset'][i]:
            e.set_baseline(np.zeros(15)); no_increment.reset()
            if control is not None:
                control.set_baseline(np.zeros(15))
        e.final_refiner.model.last = None
        e.final_refiner.anchor.final.last = None
        eligible = bool(a['valid'][i])
        b = a['b'][i] if eligible else np.zeros(15)
        r = e.push(int(now), b, eligible=eligible)
        if control is not None:
            other = control.push(int(now), b, eligible=eligible)
            assert r['valid'] == other['valid']
            if r['valid']:
                np.testing.assert_allclose(r['xyz'], other['xyz'], rtol=0, atol=1e-12)
        accepted[i] = r['valid']
        if r['valid']:
            xyz[i] = r['xyz']
        recorded = e.final_refiner.anchor.final.last
        feature = e.final_refiner.model.last
        if recorded is None or feature is None:
            no_increment.reset()
            continue
        assert r['valid'] and feature.shape == (1, 59)
        t, z, cost, phi, prop, distance, token = recorded
        assert t == now
        if zgrid is None:
            zgrid = z.copy()
        else:
            np.testing.assert_array_equal(zgrid, z)
        info = r['extra']['final_depth_evidence']
        teach_prop = info['teacher_proposal_z_mm']
        teach = no_increment.step(t, z, cost, phi, teach_prop, distance, token=token)
        x[i] = feature[0]
        old_z[i] = r['extra']['original_z_mm']
        proposal[i] = teach_prop
        teacher_z[i] = teach['z_mm']
        valid[i] = True
        costs.append(cost); phis.append(phi); distances.append(distance)
        np.testing.assert_allclose(x[i, :3]*[10., 10., 3.], [*xyz[i, :2], old_z[i]], rtol=0, atol=1e-10)
    if zgrid is None:
        raise ValueError('No current-field Z-feature calls in this complete stream')
    policy = e.final_refiner.anchor.policy
    policy_dict = {'neural_strength': policy.neural_strength, 'continuity': policy.continuity,
                   'support_limit': policy.support_limit, 'veto': policy.field_veto_ratio,
                   'innovation_gain': policy.innovation_gain, 'max_gap': policy.max_gap_s}
    result = {'time': times, 'valid': valid, 'accepted': accepted, 'q': a['q'], 'x': x,
              'old_z': old_z, 'old_output_z': xyz[:, 2], 'old_output_xyz': xyz,
              'epoch': epoch, 'z': zgrid, 'cost': np.asarray(costs), 'phi': np.asarray(phis),
              'proposal': proposal[valid], 'distance': np.asarray(distances),
              'teacher_z': teacher_z, 'calibrated_action': a['primary']}
    result.update({'phase_'+k: v for k, v in phase_masks(result).items()})
    # Run the real policy in the compact kernel at the exact captured calls.
    kernel_teacher = fast_depth(times, valid, old_z, epoch, zgrid, result['cost'], result['phi'],
                                result['proposal'], result['distance'], **policy_dict)
    np.testing.assert_allclose(kernel_teacher[valid], teacher_z[valid], rtol=0, atol=1e-9)
    delta = np.clip(e.final_refiner.model.model.predict(x[valid]).reshape(-1), -.4, .4)
    old_proposal = np.clip(result['proposal']+delta, zgrid[0], zgrid[-1])
    kernel_old = fast_depth(times, valid, old_z, epoch, zgrid, result['cost'], result['phi'],
                           old_proposal, result['distance'], **policy_dict)
    np.testing.assert_allclose(kernel_old[valid], xyz[valid, 2], rtol=0, atol=1e-9)
    return result, policy_dict

def capture(task):
    count, kind, row, smoke = task
    folder = OUT/f'arm_{count}'
    ident = row['id']
    dest = folder/'features'/f'{kind}_{ident}.npz'
    side = dest.with_suffix('.json')
    xyhash = sha(XYOUT/f'arm_{count}/best_model.npz')
    if side.exists():
        record = json.loads(side.read_text(encoding='utf8'))
        assert record['xy_sha256'] == xyhash and record['sha256'] == sha(dest)
        return record
    if kind == 'source':
        a, source = source_input(count, row)
    else:
        source = OLD/'virtual'/f'{ident}.npz'
        with np.load(source, allow_pickle=False) as f:
            a = {k: f[k].copy() for k in f.files}
    start = time.perf_counter()
    arrays, policy = capture_arrays(a, count, smoke)
    np.savez_compressed(dest, **arrays)
    record = {'id': ident, 'kind': kind, 'role': row['role'], 'mode': row['mode'],
              'source': source.relative_to(ROOT).as_posix(), 'source_sha256': sha(source),
              'file': dest.relative_to(folder).as_posix(), 'sha256': sha(dest),
              'xy_sha256': xyhash, 'z_sha256': COMPONENTS['z']['model_sha256'],
              'depth_feature_frames': int(arrays['valid'].sum()),
              'nominal_calibrated_feature_frames': int((arrays['valid'] & arrays['calibrated_action']).sum()),
              'frame_count': len(arrays['time']), 'policy': policy,
              'runtime_kernel_parity_atol_mm': 1e-9, 'uninstrumented_full_chain_parity': smoke,
              'seconds': time.perf_counter()-start, 'target_loaded': False}
    dump(side, record)
    print(count, kind, ident, 'Z frames', record['depth_feature_frames'], 's', round(record['seconds'], 1), flush=True)
    return record

def prepare(count, workers, smoke=False):
    p = protocol(count)
    folder = OUT/f'arm_{count}'
    catalog = json.loads((OLD/'catalog.json').read_text(encoding='utf8'))
    virtual = [{'id': r['id'], 'role': 'train', 'mode': r['style']}
               for r in catalog['trials'] if r['id'] in p['virtual_ids']]
    assert len(virtual) == count and all(r['id'].startswith('V370_train_') for r in virtual)
    tasks = [(count, 'source', r, smoke) for r in p['source_records']]
    tasks += [(count, 'virtual', r, smoke) for r in virtual]
    if smoke:
        rows = [capture(tasks[0]), capture(tasks[-count])]
        assert all(r['uninstrumented_full_chain_parity'] for r in rows), 'Cached non-smoke output is not instrumentation parity proof'
        dump(folder/'smoke_validation.json', {'status': 'PASSED', 'records': rows,
             'read_only_instrumentation_full_chain_parity_atol_mm': 1e-12,
             'compact_validation_runtime_parity_atol_mm': 1e-9, 'production_replaced': False})
        return
    dump(folder/'status.json', {'status': 'PREPARING', 'pid': os.getpid(), 'count': count})
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(capture, tasks):
            rows.append(r)
            dump(folder/'status.json', {'status': 'PREPARING', 'pid': os.getpid(), 'count': count,
                                       'complete_streams': len(rows), 'total_streams': len(tasks)})
    policies = [r['policy'] for r in rows]
    assert all(policy == policies[0] for policy in policies)
    dump(folder/'feature_manifest.json', {'records': rows, 'policy': policies[0],
                                        'protocol_sha256': sha(folder/'protocol.json')})
    dump(folder/'status.json', {'status': 'PREPARED', 'pid': os.getpid(), 'count': count, 'complete_streams': len(rows)})

def training_data(count):
    folder = OUT/f'arm_{count}'
    p = json.loads((folder/'protocol.json').read_text(encoding='utf8'))
    manifest = json.loads((folder/'feature_manifest.json').read_text(encoding='utf8'))
    assert manifest['protocol_sha256'] == sha(folder/'protocol.json')
    blocks, targets, weights, pairs, vals, baselines, nuisances, cursor = [], [], [], [], [], [], [], 0
    for r in manifest['records']:
        assert sha(folder/r['file']) == r['sha256']
        assert sha(ROOT/r['source']) == r['source_sha256']
        with np.load(folder/r['file'], allow_pickle=False) as f:
            a = {k: f[k].copy() for k in f.files}
        if r['role'] == 'validation':
            assert r['kind'] == 'source'
            vals.append(({'trial': r['id'], 'mode': r['mode']}, a))
            baselines.append({'trial': r['id'], 'mode': r['mode'],
                              'metrics': score_z(a, a['old_output_z']),
                              'phases': score_phases(a, a['old_output_z'])})
            continue
        ids = np.flatnonzero(a['valid'] & a['calibrated_action'])[::p['training_stride']]
        if not len(ids):
            raise ValueError('No depth training samples: '+r['id'])
        residual = a['q'][ids, 2]-a['teacher_z'][ids]
        nuisance = float(residual.mean()) if r['kind'] == 'source' else 0.
        y = np.clip(residual-nuisance, -.4, .4)[:, None]
        domain_trials = 45 if r['kind'] == 'source' else count
        mass = p['loss_mass']['source_dynamic' if r['kind'] == 'source' else 'virtual_dynamic']
        blocks.append(a['x'][ids]); targets.append(y)
        weights.append(np.full(len(ids), mass/domain_trials/len(ids)))
        dt = np.diff(a['time'][ids])/1e9
        good = (dt > 0) & (dt <= .35) & (a['epoch'][ids[1:]] == a['epoch'][ids[:-1]])
        pp = np.flatnonzero(good)
        pairs.extend(zip((cursor+pp).tolist(), (cursor+pp+1).tolist()))
        cursor += len(ids)
        nuisances.append({'id': r['id'], 'kind': r['kind'], 'samples': len(ids), 'train_nuisance_z_mm': nuisance})
    assert len(vals) == 12
    with np.load(REAL/'inputs/z_train.npz', allow_pickle=False) as f:
        static = f['x'][f['owners'] == -1].copy()
    assert len(static) == 565
    blocks.append(static); targets.append(np.zeros((len(static), 1)))
    weights.append(np.full(len(static), p['loss_mass']['static']/len(static)))
    x, y, w = np.vstack(blocks), np.vstack(targets), np.concatenate(weights)
    assert np.isfinite(np.r_[x.ravel(), y.ravel(), w]).all()
    np.testing.assert_allclose(w.sum(), 1., rtol=0, atol=1e-12)
    assert len(pairs)
    w *= len(w)
    with np.load(REAL/'z/best_model.npz', allow_pickle=False) as f:
        mean, scale = f['mean'].copy(), f['scale'].copy()
        initial = [f[f'p{i}'].copy() for i in range(6)]
    return p, manifest, x, y, w, np.asarray(pairs, int), vals, baselines, nuisances, mean, scale, initial

def train(count, resume=False, max_epochs=1000):
    folder = OUT/f'arm_{count}'
    p, manifest, x, y, w, pairs, vals, baselines, nuisances, mean, scale, params = training_data(count)
    assert max_epochs >= p['initial_max_epochs']
    u = np.clip((x-mean)/scale, -8., 8.)
    statefile = folder/'last_state.npz'
    if (folder/'freeze.json').exists():
        previous = json.loads((folder/'freeze.json').read_text(encoding='utf8'))
        assert resume and previous['status'] == 'BUDGET_REACHED_NOT_CONVERGED' and max_epochs > previous['epochs_run']
    elif statefile.exists() and not resume:
        raise FileExistsError('Use --resume; preserve existing optimizer/RNG')
    base, _, _ = val_z(params, vals, mean, scale, manifest['policy'], None, None)
    ctrl = Plateau(lr=p['initial_lr'])
    rng = np.random.default_rng(p['seed'])
    m = [np.zeros_like(v) for v in params]
    v = [a.copy() for a in m]
    best_params, best, best_epoch, updates, start_epoch, history = None, None, None, 0, 0, []
    if resume:
        with np.load(statefile, allow_pickle=False) as f:
            s = json.loads(str(f['state_json']))
            assert s['protocol_sha256'] == sha(folder/'protocol.json')
            assert s['feature_manifest_sha256'] == sha(folder/'feature_manifest.json')
            params = [f[f'p{i}'].copy() for i in range(6)]
            m = [f[f'm{i}'].copy() for i in range(6)]
            v = [f[f'v{i}'].copy() for i in range(6)]
            best_params = [f[f'best{i}'].copy() for i in range(6)] if s['best_epoch'] is not None else None
        ctrl = Plateau(**s['controller']); rng.bit_generator.state = s['rng_state']
        best, best_epoch, updates, start_epoch, history = s['best_metric'], s['best_epoch'], s['updates'], s['epoch'], s['history']
    dump(folder/'training_data_summary.json', {'samples': len(x), 'static': 565,
         'validation_trials': 12, 'baseline_validation_objective': base, 'nuisances': nuisances,
         'source_only': True, 'target_loaded': False, 'feature_manifest_sha256': sha(folder/'feature_manifest.json')})
    stop = 'initial_budget_limit'
    for epoch in range(start_epoch+1, max_epochs+1):
        started = time.perf_counter(); lr = ctrl.lr; order = rng.permutation(len(u))
        for ids in np.array_split(order, max(1, (len(u)+255)//256)):
            pp = pairs[rng.integers(0, len(pairs), size=len(ids))]
            g = gradients_z(params, u[ids], y[ids], w[ids], pp[:, 0], pp[:, 1], u, y, w)
            updates += 1
            for par, grad, first, second in zip(params, g, m, v):
                first *= .9; first += .1*grad
                second *= .999; second += .001*grad*grad
                par -= lr*(first/(1-.9**updates))/(np.sqrt(second/(1-.999**updates))+1e-8)
        if not all(np.isfinite(par).all() for par in params):
            raise FloatingPointError('Nonfinite research Z weight')
        pred = forward(params, u)[2]
        mse = float(np.mean((pred-y)**2*w[:, None]))
        diff = pred[pairs[:, 1]]-pred[pairs[:, 0]]-(y[pairs[:, 1]]-y[pairs[:, 0]])
        pw = ((w[pairs[:, 0]]+w[pairs[:, 1]])/2)[:, None]
        loss = mse+.5*float(np.mean(pred*pred*w[:, None]))+.5*float(np.mean(diff*diff*pw))
        loss += .0005*sum(float(np.sum(params[k]**2)) for k in (0, 2, 4))
        obj, qualified, rows = val_z(params, vals, mean, scale, manifest['policy'], baselines, base)
        if qualified and (best is None or obj < best):
            best, best_epoch, best_params = obj, epoch, [par.copy() for par in params]
            save_model(folder/'best_model.npz', best_params, mean, scale, 'z')
        event = ctrl.observe(obj)
        rec = {'epoch': epoch, 'updates': updates, 'lr_used': lr, 'next_lr': ctrl.lr,
               'training_samples': len(u), 'unique_samples_visited': len(np.unique(order)),
               'train_weighted_mse': mse, 'train_regularized_objective': loss,
               'validation_objective': obj, 'qualified': bool(qualified), 'best_epoch': best_epoch,
               'best_objective': best, 'scheduler_event': event, 'bad_epochs': ctrl.bad_epochs,
               'validation': rows, 'seconds': time.perf_counter()-started}
        history.append(rec)
        s = {'epoch': epoch, 'updates': updates, 'history': history, 'controller': asdict(ctrl),
             'rng_state': rng.bit_generator.state, 'best_epoch': best_epoch, 'best_metric': best,
             'protocol_sha256': sha(folder/'protocol.json'), 'feature_manifest_sha256': sha(folder/'feature_manifest.json')}
        checkpoint(statefile, params, m, v, best_params, s)
        dump(folder/'status.json', {'status': 'TRAINING', 'pid': os.getpid(), **rec})
        if epoch % 10 == 0 or event != 'continue':
            print('R382', count, epoch, obj, best_epoch, lr, event, flush=True)
        if event == 'early_stop_min_lr_plateau':
            stop = event
            break
    status = 'COMPLETED_PLATEAU' if stop == 'early_stop_min_lr_plateau' else 'BUDGET_REACHED_NOT_CONVERGED'
    dump(folder/'history.json', history)
    dump(folder/'freeze.json', {'status': status, 'count': count, 'component': 'Z increment conditioned on fixed selected XY',
         'epochs_run': history[-1]['epoch'], 'best_epoch': best_epoch, 'qualified_candidate': best_params is not None,
         'model_sha256': sha(folder/'best_model.npz') if best_params is not None else None,
         'selected_xy_sha256': p['selected_xy_sha256'], 'source_only': True, 'target_fit_or_selection': False,
         'production_replaced': False, 'feature_manifest_sha256': sha(folder/'feature_manifest.json'),
         'protocol_sha256': sha(folder/'protocol.json'), 'training_samples': len(u),
         'optimizer_updates': updates, 'final_learning_rate': ctrl.lr, 'lr_reductions': ctrl.reductions,
         'stop_reason': stop, 'best_validation_objective': best, 'baseline_validation_objective': base})
    dump(folder/'status.json', {'status': status, 'pid': os.getpid(), 'epochs_run': history[-1]['epoch'],
         'qualified_candidate': best_params is not None, 'production_replaced': False})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=('smoke','prepare','train','all'), default='prepare')
    ap.add_argument('--count', type=int, choices=(100,500,1000), default=100)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-epochs', type=int, default=1000)
    a = ap.parse_args()
    if a.phase in ('smoke','prepare','all'):
        prepare(a.count, a.workers, a.phase == 'smoke')
    if a.phase in ('train','all'):
        train(a.count, a.resume, a.max_epochs)

if __name__ == '__main__':
    main()
