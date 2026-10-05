"""Research-only source-real refinement of the already trained R373 XY head.

Keep its 53 inputs/normalization/weights and full system. Append three causal
EMA residuals of the corrected 16D magnetic feature. Zero input rows preserve
the exact original function at initialization. No acquisition/UI imports.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('MAGTOUCH_R460_SOURCE', str(HERE)))
sys.path[:0] = [str(SOURCE), str(SOURCE/'code'), str(SOURCE/'tmp/r360_math_packages')]
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory
from training.train_source_to_plateau_r372 import Plateau, dump, sha, forward

OUT = HERE/'analysis/r460_source_real_history_refinement_20261004'
REAL = SOURCE/'analysis/r372_direct_source_plateau_20261001'
KEYS = ('w1', 'b1', 'w2', 'b2', 'w3', 'b3')
ARMS = ('history_position', 'history_motion_hold_anchor')
TAUS = np.array([.3, 1., 4.])


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


class MultiscaleXYHistory(CausalXYFeatureHistory):
    """Existing history plus current-minus-EMA; no future or XYZ averaging."""
    def reset(self):
        super().reset()
        self.ema = None

    def push(self, **kw):
        old_t = self.last_time_ns
        t = int(kw['time_ns'])
        reset = old_t is None or t <= old_t or t-old_t > self.max_gap_ns
        reset |= (old_t is not None and (int(kw['b0_id']) != self.last_b0_id
                  or int(kw['stream_id']) != self.last_stream_id))
        original = super().push(**kw)
        phi = np.asarray(kw['phi'], float)
        if reset or self.ema is None:
            self.ema = np.tile(phi, (3, 1))
        else:
            decay = np.exp(-(t-old_t)/1e9/TAUS)[:, None]
            self.ema = decay*self.ema+(1-decay)*phi
        return np.r_[original, (phi[None, :]-self.ema).ravel()]


def augment_existing(x, t, valid):
    """Replay entire accepted-frame prefix BEFORE selecting supervised rows."""
    h = MultiscaleXYHistory()
    out = np.zeros((len(t), 101))
    for i in range(len(t)):
        if not valid[i]:
            h.reset()
            continue
        out[i] = h.push(time_ns=int(t[i]), b0_id=0, stream_id=0,
                       xyz=x[i, :3]*np.array([10., 10., 3.]),
                       phi=x[i, 3:19], memory=x[i, 19:21]*1.5)
        np.testing.assert_allclose(out[i, :53], x[i], atol=1e-10, rtol=0)
    return out


def initialize(mean_extra, scale_extra):
    base = load(REAL/'xy/best_model.npz')
    params = [base[k] for k in KEYS]
    params[0] = np.vstack((params[0], np.zeros((48, 64))))
    return params, np.r_[base['mean'], mean_extra], np.r_[base['scale'], scale_extra]


class ResearchXY:
    def __init__(self, artifact):
        a = load(artifact)
        self.params = [a[k] for k in KEYS]
        self.mean, self.scale = a['mean'], a['scale']
        if self.mean.shape != (101,) or self.params[0].shape != (101, 64):
            raise ValueError('Expected R460 extended research model')
        if not all(np.isfinite(v).all() for v in self.params+[self.mean, self.scale]) or np.any(self.scale <= 0):
            raise ValueError('Invalid research model')

    def predict(self, feature):
        u = (np.asarray(feature, float).reshape(-1, 101)-self.mean)/self.scale
        return forward(self.params, u)[2]


def install_research_head(engine, artifact):
    """For frozen offline replay only; Z/gating/slow memory stay unchanged."""
    for decoder in (engine.memory_decoder, engine.candidate_decoder):
        decoder.xy_features = MultiscaleXYHistory()
        decoder.xy_model = ResearchXY(artifact)
    return engine


def protocol():
    p = read(OUT/'protocol.json')
    if p['baseline_xy_sha256'] != sha(REAL/'xy/best_model.npz'):
        raise ValueError('R373 baseline hash changed')
    if p['baseline_z_sha256'] != sha(REAL/'z/best_model.npz'):
        raise ValueError('R373 Z hash changed')
    if p['source_manifest_sha256'] != sha(REAL/'data_manifest.json'):
        raise ValueError('Original source contract changed')
    return p


def server_only():
    if socket.gethostname() != 'googo-Super-Server':
        raise RuntimeError('Scientific preparation/training runs on authorized server')


def prepare():
    server_only()
    p = protocol()
    if (OUT/'data_manifest.json').exists():
        raise FileExistsError('Preserve prepared study inputs')
    from training import retrain_r133_coverage_xy as old
    from training.probe_causal_xy_neural_r164 import guarded_positions
    from training.probe_had239_local_forward_r104 import ForwardAtlas, load_profile
    from training.build_trajectory_dataset_v1 import command_reference, command_segments, read_rows, reference_origin
    original = read(REAL/'data_manifest.json')['xy']
    excluded = read(HERE/'analysis/r387_soft_state_research_map_20261002/exclusions.json')['excluded_path_prefixes']
    profile = load_profile(old.FIVE_CONFIG, old.FIVE_KEY)
    samples = profile['samples']
    xyz = np.array([[s['x'], s['y'], s['depth']] for s in samples])
    delta = np.array([s['delta'] for s in samples])
    atlas = ForwardAtlas(xyz, delta, 'unit_logamp')
    memory = read(old.FIVE_REPORT)
    basis = np.asarray(memory['basis'], float)
    strength = float(memory['selection']['selected']['strength'])
    core_x, core_y, core_owners = [], [], []
    blocks, val_files, raw_hashes, summaries = [], [], [], []
    for rec in original['records']:
        if any(rec['path'].startswith(prefix) for prefix in excluded):
            raise ValueError('Excluded recording in inherited source split')
        for entry in rec['raw_manifest']:
            if sha(SOURCE/entry['path']) != entry['sha256']:
                raise ValueError('Raw input changed: '+entry['path'])
            raw_hashes.append(entry)
        n = rec['trial']
        if n in old.BASE_TRAIN+old.BASE_VALID:
            row = old.load_old(n, atlas, basis, strength)
            row['guarded'] = guarded_positions(row, atlas)
        else:
            row = old.load_new(SOURCE/rec['path'], atlas, basis, strength)
        x = old.features(row, atlas)
        ext = augment_existing(x, row['time'], row['valid'])
        segments = command_segments(read_rows(row['path']/'stage_commands.csv'))
        ref = command_reference(row['time'], segments, reference_origin(row['meta']))
        np.testing.assert_allclose(ref['xyz_mm'], row['truth'], atol=0, rtol=0, equal_nan=True)
        ids = ref['command_segment_id']
        known_est = np.array([j >= 0 and segments[j]['duration_source'] == 'logged_EST' for j in ids])
        reliable = (row['valid'] & np.isfinite(row['truth']).all(1) & known_est
                    & (row['truth'][:, 2] >= p['minimum_nominal_depth_mm'])
                    & ~ref['command_moving'] & ~ref['command_boundary']
                    & (ref['hold_age_s'] >= .2) & (ref['hold_age_s'] <= 1.5))
        hold_ids = np.flatnonzero(reliable & ~row['core'])[::4]
        pair_ok = np.zeros(len(x), bool)
        dt = np.diff(row['time'])/1e9
        pair_ok[1:] = (row['valid'][1:] & row['valid'][:-1] & (dt > 0) & (dt <= .25)
                       & np.isfinite(row['truth'][1:]).all(1) & np.isfinite(row['truth'][:-1]).all(1)
                       & (row['truth'][1:, 2] >= .45) & (row['truth'][:-1, 2] >= .45)
                       & ~ref['command_boundary'][1:] & ~ref['command_boundary'][:-1]
                       & (ids[1:] == ids[:-1]) & known_est[1:])
        def arrays(choose):
            previous = np.maximum(choose-1, 0)
            return dict(x=ext[choose], y=np.clip(row['truth'][choose, :2]-row['guarded'][choose, :2], -3., 3.),
                        guard=row['guarded'][choose, :2], q=row['truth'][choose, :2], time=row['time'][choose],
                        prev_x=ext[previous], prev_guard=row['guarded'][previous, :2],
                        dq=row['truth'][choose, :2]-row['truth'][previous, :2], pair=pair_ok[choose],
                        hold_pair=(reliable[choose] & reliable[previous] & pair_ok[choose]),
                        hold=reliable[choose], owners=np.full(len(choose), n),
                        source_index=choose)
        if rec['role'] == 'train':
            selected = np.flatnonzero(row['core'])[::4]
            core_x.append(x[selected]); core_y.append(np.clip(row['truth'][selected, :2]-row['guarded'][selected, :2], -3., 3.))
            core_owners.extend([n]*len(selected))
            choose = np.unique(np.r_[selected, hold_ids])
            a = arrays(choose)
            a['group'] = np.where(row['core'][choose], 0, 1)
            blocks.append(a)
        else:
            selected = np.flatnonzero(row['core'])
            old_val = load(REAL/f'inputs/xy_validation_{n}.npz')
            np.testing.assert_allclose(x[selected], old_val['x'], atol=1e-10, rtol=0)
            a = arrays(np.unique(np.r_[selected, np.flatnonzero(reliable & ~row['core'])]))
            a['core'] = row['core'][a['source_index']]
            file = OUT/'inputs'/f'validation_{n}.npz'
            file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, **a)
            val_files.append(dict(trial=n, mode=row['mode'], file=file.relative_to(OUT).as_posix(), sha256=sha(file)))
        summaries.append(dict(trial=n, role=rec['role'], mode=row['mode'], path=rec['path'],
                              original_core_rows=int(row['core'].sum()), reliable_hold_rows=int(reliable.sum()),
                              known_EST_rows=int(known_est.sum()), motion_pair_rows=int(pair_ok.sum())))
        print('PREPARED', rec['role'], n, 'core', int(row['core'].sum()), 'hold', int(reliable.sum()), flush=True)
    old_train = load(REAL/'inputs/xy_train.npz')
    static = old_train['x'][-original['static_samples']:]
    rebuilt_x = np.vstack(core_x+[static])
    rebuilt_y = np.vstack(core_y+[np.zeros((len(static), 2))])
    np.testing.assert_allclose(rebuilt_x, old_train['x'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(rebuilt_y, old_train['y'], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(np.r_[core_owners, np.full(len(static), -1)], old_train['owners'])
    sx = np.c_[static, np.zeros((len(static), 48))]
    anchor = dict(x=sx, y=np.zeros((len(sx), 2)), guard=static[:, :2]*10, q=static[:, :2]*10,
                  time=np.zeros(len(sx), dtype=np.int64), prev_x=sx.copy(), prev_guard=static[:, :2]*10,
                  dq=np.zeros((len(sx), 2)), pair=np.zeros(len(sx), bool), hold_pair=np.zeros(len(sx), bool),
                  hold=np.zeros(len(sx), bool), owners=np.full(len(sx), -1), source_index=np.full(len(sx), -1),
                  group=np.full(len(sx), 2))
    blocks.append(anchor)
    data = {k: np.concatenate([a[k] for a in blocks]) for k in blocks[0]}
    counts = [int((data['group'] == k).sum()) for k in range(3)]
    if not all(counts):
        raise ValueError('Position/hold/anchor groups must all exist')
    weight = np.empty(len(data['x']))
    for k, mass in enumerate(p['position_loss_group_masses']):
        weight[data['group'] == k] = len(weight)*mass/counts[k]
    mean_extra = data['x'][:, 53:].mean(0)
    scale_extra = np.maximum(data['x'][:, 53:].std(0), .25)
    params, mean, scale = initialize(mean_extra, scale_extra)
    np.testing.assert_allclose(forward(params, (data['x']-mean)/scale)[2],
        forward([load(REAL/'xy/best_model.npz')[k] for k in KEYS],
                (data['x'][:, :53]-mean[:53])/scale[:53])[2], atol=1e-12, rtol=0)
    for k in ('prev_x', 'prev_guard', 'dq'):
        data[k][~data['pair']] = 0
    file = OUT/'inputs/train.npz'
    np.savez_compressed(file, **data, weight=weight, mean=mean, scale=scale,
                        **dict(zip(KEYS, params)))
    manifest = dict(schema='R460-Source-Whole-Trial-1', protocol_sha256=sha(OUT/'protocol.json'),
        baseline_xy_sha256=sha(REAL/'xy/best_model.npz'), baseline_z_sha256=sha(REAL/'z/best_model.npz'),
        train=dict(file='inputs/train.npz', sha256=sha(file), samples=len(weight), group_counts=counts,
                   motion_pairs=int(data['pair'].sum()), hold_pairs=int(data['hold_pair'].sum())),
        validation=val_files, records=summaries, raw_inputs=raw_hashes,
        original_xy_7782_rows_reconstructed=True, original_input_parity_max=float(np.max(np.abs(rebuilt_x-old_train['x']))),
        normalization='first53 R373 immutable, added48 from source TRAIN only',
        runtime_initialization_parity_max=0.,
        held_out_or_Yellow_read_for_preparation=False,
        domain='original rounded-flat D5/topD3/R1 x source had239; historical label aliases retained')
    dump(OUT/'data_manifest.json', manifest)
    return manifest


def torch_prediction(params, x):
    import torch
    return torch.tanh(torch.tanh(x@params[0]+params[1])@params[2]+params[3])@params[4]+params[5]


def batch_loss(params, a, arm, weights):
    """Match predicted increments to reference, not to zero during movement."""
    import torch
    pred = torch_prediction(params, a['u'])
    position = ((pred-a['y']).square().sum(1)*a['weight']).mean()
    regularization = .0005*sum(v.square().sum() for v in params[::2])
    result = position+regularization
    if arm == ARMS[1]:
        previous = torch_prediction(params, a['prev_u'])
        increment_error = a['guard']+pred-a['prev_guard']-previous-a['dq']
        result = result+weights['increment']*(increment_error.square().sum(1)*a['pair_weight']).mean()
        result = result+weights['hold']*(increment_error.square().sum(1)*a['hold_weight']).mean()
        result = result+weights['anchor']*(pred.square().sum(1)*a['anchor_weight']).mean()
    return result


def val_metrics(params, vals, mean, scale, anchors):
    rows = []
    for entry, a in vals:
        correction = forward(params, (a['x']-mean)/scale)[2]
        pred = a['guard']+correction
        err = pred-a['q']
        use = a['core']
        residual = err[use]
        core_ids = np.flatnonzero(use)
        near = ((np.diff(a['time'][core_ids]) > 0) & (np.diff(a['time'][core_ids]) <= 250_000_000)
                & (np.diff(a['source_index'][core_ids]) == 1))
        steps = np.linalg.norm(np.diff(residual, axis=0)[near], axis=1)
        hold_ids = np.flatnonzero(a['hold'])
        groups = np.split(hold_ids, np.flatnonzero((np.diff(hold_ids) != 1)
                            | (np.diff(a['time'][hold_ids]) > 250_000_000))+1) if len(hold_ids) else []
        hold_values = []
        for g in groups:
            if len(g) >= 3 and a['time'][g[-1]]-a['time'][g[0]] >= 200_000_000:
                centered = pred[g]-pred[g].mean(0)
                hold_values.append(float(np.sqrt(np.mean(np.sum(centered**2, axis=1)))))
        rows.append(dict(trial=entry['trial'], mode=entry['mode'], frames=int(use.sum()),
                         raw_xy_mm=float(np.linalg.norm(residual, axis=1).mean()),
                         centered_xy_mm=float(np.linalg.norm(residual-residual.mean(0), axis=1).mean()),
                         bias_xy_mm=residual.mean(0).tolist(),
                         residual_step_rms_mm=float(np.sqrt(np.mean(steps**2))) if len(steps) else 0.,
                         jump_over_0p2_percent=float(100*np.mean(steps > .2)) if len(steps) else 0.,
                         hold_xy_rms_mm=float(np.mean(hold_values)) if hold_values else 0., hold_blocks=len(hold_values)))
    correction = forward(params, (anchors-mean)/scale)[2]
    anchor_rms = float(np.sqrt(np.mean(np.sum(correction**2, axis=1))))
    objective = float(np.mean([r['raw_xy_mm']+.1*r['residual_step_rms_mm']+.1*r['hold_xy_rms_mm'] for r in rows])+.1*anchor_rms)
    return objective, rows, anchor_rms


def save_model(file, params, mean, scale):
    np.savez_compressed(file, mean=mean, scale=scale, **dict(zip(KEYS, params)))


def train(arm, resume=False, max_epochs=None):
    server_only()
    import torch
    p = protocol()
    manifest = read(OUT/'data_manifest.json')
    if manifest['protocol_sha256'] != sha(OUT/'protocol.json'):
        raise ValueError('Study protocol changed after data preparation')
    for entry in [manifest['train']]+manifest['validation']:
        if sha(OUT/entry['file']) != entry['sha256']:
            raise ValueError('Study data changed')
    folder = OUT/arm
    folder.mkdir(exist_ok=True)
    if (folder/'freeze.json').exists():
        fr = read(folder/'freeze.json')
        if not (resume and fr['status'] == 'BUDGET_TRUNCATED' and max_epochs and max_epochs > fr['epochs']):
            raise FileExistsError('Frozen arm already exists')
    if not torch.cuda.is_available():
        raise RuntimeError('Prechecked isolated GPU required')
    torch.set_num_threads(2)
    torch.manual_seed(7)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    a = load(OUT/manifest['train']['file'])
    mean, scale = a['mean'], a['scale']
    initial = [a[k] for k in KEYS]
    vals = [(r, load(OUT/r['file'])) for r in manifest['validation']]
    anchors = a['x'][a['group'] == 2]
    baseline_obj, baseline_rows, baseline_anchor = val_metrics(initial, vals, mean, scale, anchors)
    data = dict(u=(a['x']-mean)/scale, prev_u=(a['prev_x']-mean)/scale,
                y=a['y'], guard=a['guard'], prev_guard=a['prev_guard'], dq=a['dq'], weight=a['weight'])
    for name, mask in (('pair_weight', a['pair']), ('hold_weight', a['hold_pair']), ('anchor_weight', a['group'] == 2)):
        data[name] = mask.astype(float)*len(mask)/max(int(mask.sum()), 1)
    tx = {k: torch.as_tensor(v, dtype=torch.float64, device=device) for k, v in data.items()}
    params = [torch.nn.Parameter(torch.as_tensor(v.copy(), dtype=torch.float64, device=device)) for v in initial]
    with torch.no_grad():
        np.testing.assert_allclose(torch_prediction(params, tx['u'][:64]).cpu().numpy(),
                                   forward(initial, data['u'][:64])[2], atol=1e-11, rtol=1e-11)
    optimizer = torch.optim.Adam(params, lr=p['initial_lr'], eps=1e-8)
    ctrl = Plateau(lr=p['initial_lr'])
    rng = np.random.default_rng(7)
    best, best_obj, best_epoch = [v.copy() for v in initial], baseline_obj, 0
    free_best, free_best_obj, free_best_epoch = [v.copy() for v in initial], baseline_obj, 0
    history, updates, start = [], 0, 0
    checkpoint = folder/'last_state.pt'
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['protocol_sha256'] != sha(OUT/'protocol.json') or state['manifest_sha256'] != sha(OUT/'data_manifest.json'):
            raise ValueError('Resume contract changed')
        with torch.no_grad():
            for v, value in zip(params, state['params']): v.copy_(value)
        optimizer.load_state_dict(state['optimizer'])
        ctrl = Plateau(**state['controller'])
        rng.bit_generator.state = state['rng']
        history, updates, start = state['history'], state['updates'], state['epoch']
        best, best_obj, best_epoch = state['best'], state['best_obj'], state['best_epoch']
        free_best, free_best_obj, free_best_epoch = state['free_best'], state['free_best_obj'], state['free_best_epoch']
    elif checkpoint.exists():
        raise FileExistsError('Training checkpoint already exists; explicit resume required')
    save_model(folder/'initial_model.npz', initial, mean, scale)
    dump(folder/'baseline_validation.json', dict(objective=baseline_obj, rows=baseline_rows, anchor_rms_mm=baseline_anchor,
        note='Frozen feature-domain source validation, NOT native full-chain evaluation'))
    cap = max_epochs or p['max_epochs']
    started = time.perf_counter()
    stop = 'max_epochs_budget_truncated'
    dump(folder/'status.json', dict(status='TRAINING', pid=os.getpid(), arm=arm, epoch=start, updates=updates, lr=ctrl.lr))
    for epoch in range(start+1, cap+1):
        lr = ctrl.lr
        for group in optimizer.param_groups: group['lr'] = lr
        order = rng.permutation(len(a['x']))
        seen, loss_sum = 0, 0.
        for ids in np.array_split(order, (len(order)+255)//256):
            ix = torch.as_tensor(ids, device=device)
            batch = {k: v[ix] for k, v in tx.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(params, batch, arm, p['constraint_weights'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward(); optimizer.step()
            updates += 1; seen += len(ids); loss_sum += float(loss.detach())*len(ids)
        if seen != len(a['x']) or len(np.unique(order)) != len(a['x']):
            raise ValueError('Incomplete epoch')
        current = [v.detach().cpu().numpy().copy() for v in params]
        objective, rows, anchor_rms = val_metrics(current, vals, mean, scale, anchors)
        qualified = (np.mean([r['raw_xy_mm'] for r in rows]) <= np.mean([r['raw_xy_mm'] for r in baseline_rows])+.005
                     and all(r['raw_xy_mm'] <= b['raw_xy_mm']+.02 for r, b in zip(rows, baseline_rows))
                     and anchor_rms <= baseline_anchor+.02)
        if objective < free_best_obj:
            free_best, free_best_obj, free_best_epoch = current, objective, epoch
            save_model(folder/'unconstrained_best_model.npz', current, mean, scale)
        if qualified and objective < best_obj:
            best, best_obj, best_epoch = current, objective, epoch
            save_model(folder/'best_model.npz', current, mean, scale)
        event = ctrl.observe(objective)
        rec = dict(epoch=epoch, updates=updates, lr_used=lr, next_lr=ctrl.lr, unique_samples=seen,
                   train_objective=loss_sum/seen, validation_objective=objective, validation=rows,
                   qualified=bool(qualified), anchor_rms_mm=anchor_rms, best_epoch=best_epoch, best_objective=best_obj,
                   unconstrained_best_epoch=free_best_epoch, bad_epochs=ctrl.bad_epochs, scheduler_event=event)
        history.append(rec)
        state = dict(epoch=epoch, updates=updates, history=history, params=[v.detach().cpu() for v in params],
                     optimizer=optimizer.state_dict(), controller=asdict(ctrl), rng=rng.bit_generator.state,
                     best=best, best_obj=best_obj, best_epoch=best_epoch,
                     free_best=free_best, free_best_obj=free_best_obj, free_best_epoch=free_best_epoch,
                     protocol_sha256=sha(OUT/'protocol.json'), manifest_sha256=sha(OUT/'data_manifest.json'))
        temporary = folder/'last_state.writing.pt'
        torch.save(state, temporary); temporary.replace(checkpoint)
        dump(folder/'status.json', dict(status='TRAINING', pid=os.getpid(), arm=arm, epoch=epoch, updates=updates,
                                       lr=ctrl.lr, best_epoch=best_epoch, best_objective=best_obj,
                                       scheduler_event=event, elapsed_seconds=time.perf_counter()-started))
        if epoch % 10 == 0 or event != 'continue':
            print(arm, epoch, updates, 'val', objective, 'best', best_epoch, 'lr', lr, event, flush=True)
        if event == 'early_stop_min_lr_plateau':
            stop = event
            break
    save_model(folder/'best_model.npz', best, mean, scale)
    save_model(folder/'unconstrained_best_model.npz', free_best, mean, scale)
    dump(folder/'history.json', history)
    freeze = dict(status='COMPLETED_PLATEAU' if stop == 'early_stop_min_lr_plateau' else 'BUDGET_TRUNCATED',
        arm=arm, epochs=history[-1]['epoch'], updates=updates, best_epoch=best_epoch, best_objective=best_obj,
        unconstrained_best_epoch=free_best_epoch, unconstrained_best_objective=free_best_obj,
        model_sha256=sha(folder/'best_model.npz'), unconstrained_model_sha256=sha(folder/'unconstrained_best_model.npz'),
        baseline_objective=baseline_obj, stop_reason=stop, final_lr=ctrl.lr, reductions=ctrl.reductions,
        protocol_sha256=sha(OUT/'protocol.json'), manifest_sha256=sha(OUT/'data_manifest.json'), trainer_sha256=sha(Path(__file__)),
        samples=len(a['x']), parameters=sum(v.size for v in best), dtype='float64', seed=7,
        selected_initial_function_if_best_epoch_zero=best_epoch == 0,
        true_epoch='all supervised endpoints once; previous frames are causal contexts, not new sample coverage',
        new_trajectory_or_Yellow_used_for_training_or_selection=False, no_XYZ_output_smoothing_added=True,
        feature_domain_only=True, native_full_chain_evaluation_pending=True, elapsed_seconds=time.perf_counter()-started)
    dump(folder/'freeze.json', freeze)
    dump(folder/'status.json', freeze)
    print('FROZEN', arm, json.dumps(freeze), flush=True)
    return freeze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('prepare', 'train', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-epochs', type=int)
    args = parser.parse_args()
    server_only()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase in ('prepare', 'all') and not (OUT/'data_manifest.json').exists(): prepare()
    if args.phase in ('train', 'all'):
        for arm in ARMS:
            if (OUT/arm/'freeze.json').exists() and read(OUT/arm/'freeze.json')['status'] == 'COMPLETED_PLATEAU': continue
            result = train(arm, args.resume, args.max_epochs)
            if result['status'] != 'COMPLETED_PLATEAU':
                raise RuntimeError('Budget truncated: extend same optimizer run before continuing queue')
        dump(OUT/'training_complete.json', dict(status='BOTH_ARMS_FROZEN', arms=list(ARMS), native_real_evaluation='PENDING'))


if __name__ == '__main__': main()
