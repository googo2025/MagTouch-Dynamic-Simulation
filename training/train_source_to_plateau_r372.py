"""Direct source-only retraining, original initialization, plateau LR schedule.

Pure NumPy training matches the historical objectives and seed streams.
No historical training reproduction, acquisition imports or model deployment.
Prepared validation contains source records only; test records arrive after freeze.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/r372_direct_source_plateau_20261001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.writing')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


@dataclass
class Plateau:
    lr: float
    patience: int = 50
    factor: float = .5
    min_lr: float = 1e-6
    min_delta: float = 1e-4
    reference: float | None = None
    bad_epochs: int = 0
    reductions: int = 0

    def observe(self, metric):
        if not math.isfinite(metric) or metric < 0:
            raise ValueError('Finite nonnegative validation metric required')
        improved = self.reference is None or metric < self.reference * (1-self.min_delta)
        if improved:
            self.reference = float(metric)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        event = 'continue'
        if self.bad_epochs >= self.patience:
            if self.lr > self.min_lr:
                self.lr = max(self.lr*self.factor, self.min_lr)
                self.bad_epochs = 0
                self.reference = None
                self.reductions += 1
                event = 'reduce_lr'
            else:
                event = 'early_stop_min_lr_plateau'
        return event


def forward(params, x):
    h1 = np.tanh(x@params[0]+params[1])
    h2 = np.tanh(h1@params[2]+params[3])
    return h1, h2, h2@params[4]+params[5]


def backward(params, x, gout):
    h1, h2, _ = forward(params, x)
    g2 = (gout@params[4].T)*(1-h2*h2)
    g1 = (g2@params[2].T)*(1-h1*h1)
    return [x.T@g1, g1.sum(0), h1.T@g2, g2.sum(0), h2.T@gout, gout.sum(0)]


def gradients_xy(params, x, y):
    pred = forward(params, x)[2]
    grads = backward(params, x, 2*(pred-y)/len(x))
    for k in (0, 2, 4):
        grads[k] += .001*params[k]
    return grads


def gradients_z(params, x, y, w, ia, ib, all_x, all_y, all_w):
    pred = forward(params, x)[2]
    grads = backward(params, x, (2*(pred-y)+pred)*w[:, None]/len(x))
    diff = forward(params, all_x[ib])[2]-forward(params, all_x[ia])[2]-(all_y[ib]-all_y[ia])
    gout = diff*((all_w[ia]+all_w[ib])/2)[:, None]/len(x)
    ga = backward(params, all_x[ia], -gout)
    gb = backward(params, all_x[ib], gout)
    grads = [g+a+b for g, a, b in zip(grads, ga, gb)]
    for k in (0, 2, 4):
        grads[k] += .001*params[k]
    return grads


def depth_kernel(time_ns, valid, old_z, token, z, cost, phi, proposal, distance,
                 neural_strength, continuity, support_limit, veto, innovation_gain, max_gap):
    """Same causal current-field adjudication, optionally JIT-compiled, no fastmath."""
    out = old_z.copy()
    previous_z = 0.
    previous_t = 0
    previous_token = 0
    previous_phi = np.zeros(16)
    active = False
    j = 0
    for i in range(len(time_ns)):
        if not valid[i]:
            active = False
            continue
        jj = j
        j += 1
        if not z[0] <= old_z[i] <= z[-1]:
            active = False
            continue
        c = cost[jj]
        global_id = np.argmin(c)
        minimum = max(c[global_id], 1e-8)
        centre = z[global_id]
        curvature = max((np.interp(min(centre+.05, z[-1]), z, c)
                         +np.interp(max(centre-.05, z[0]), z, c)-2*c[global_id])/.005, 1e-5)
        support = (support_limit/(support_limit+distance[jj]))**2
        finite = c+neural_strength*curvature*support*(z-proposal[jj])**2
        dt = (time_ns[i]-previous_t)/1e9
        same = active and token[i] == previous_token and 0 < dt <= max_gap
        if same and continuity:
            innovation = np.sqrt(np.sum((phi[jj]-previous_phi)**2))
            scale = (max(dt, .025)/.1)**2*(1+innovation_gain*innovation)**2
            finite = finite+continuity*(z-previous_z)**2/scale
        constrained = np.where(c <= veto*minimum, finite, np.inf)
        selected = np.argmin(constrained)
        value = z[selected]
        if 0 < selected < len(z)-1:
            a, b, cc = finite[selected-1:selected+2]
            curvature2 = a-2*b+cc
            shift = min(max(.5*(a-cc)/max(curvature2, 1e-15), -.5), .5) if curvature2 > 0 else 0.
            value = z[selected]+shift*(z[selected+1]-z[selected])
        if np.interp(value, z, c) > veto*minimum:
            value = z[selected]
        out[i] = value
        previous_z = value
        previous_phi = phi[jj].copy()
        previous_t = time_ns[i]
        previous_token = token[i]
        active = True
    return out


try:
    from numba import njit
    fast_depth = njit(cache=True, fastmath=False)(depth_kernel)
    JIT = True
except ImportError:
    fast_depth = depth_kernel
    JIT = False


def score_z(a, predicted):
    good = a['valid'] & a['calibrated_action'] & np.isfinite(a['q']).all(1)
    error = predicted-a['q'][:, 2]
    residual = error[good]
    dt = np.diff(a['time'])/1e9
    pairs = good[1:] & good[:-1] & (dt > 0) & (dt <= .25)
    steps = np.abs(np.diff(error))[pairs]
    return {'output_frames': int(good.sum()), 'z_raw_mae_mm': float(np.mean(np.abs(residual))),
            'z_centered_mae_mm': float(np.mean(np.abs(residual-residual.mean()))),
            'z_step_p95_mm': float(np.quantile(steps, .95)) if len(steps) else 0.,
            'z_step_max_mm': float(steps.max()) if len(steps) else 0.,
            'z_bias_mm': float(residual.mean())}


def score_phases(a, predicted):
    good = a['valid'] & a['calibrated_action'] & np.isfinite(a['q']).all(1)
    error = predicted-a['q'][:, 2]
    bias = float(error[good].mean())
    results = {}
    for name in ('fixed_hold', 'pure_z_press', 'pure_z_withdraw', 'constant_z_lateral', 'xyz_simultaneous'):
        mask = a['phase_'+name] & a['valid']
        results[name] = {'output_frames': int(mask.sum()),
                         'z_raw_mae_mm': float(np.abs(error[mask]).mean()) if mask.any() else None,
                         'z_global_centered_mae_mm': float(np.abs(error[mask]-bias).mean()) if mask.any() else None}
    return results


def val_xy(params, vals, mean, scale):
    rows = []
    for entry, a in vals:
        pred = a['guard']+forward(params, (a['x']-mean)/scale)[2]
        error = pred-a['q']
        bias = error.mean(0)
        rows.append({'trial': entry['trial'], 'mode': entry['mode'], 'frames': len(error),
                     'xy_raw_mean_mm': float(np.linalg.norm(error, axis=1).mean()),
                     'xy_centered_mean_mm': float(np.linalg.norm(error-bias, axis=1).mean()),
                     'bias_xy_mm': bias.tolist()})
    return float(np.mean([r['xy_raw_mean_mm'] for r in rows])), True, rows


def val_z(params, vals, mean, scale, policy, baselines, baseline_score):
    rows, failures = [], []
    for entry, a in vals:
        u = np.clip((a['x'][a['valid']]-mean)/scale, -8., 8.)
        delta = np.clip(forward(params, u)[2].reshape(-1), -.4, .4)
        prop = np.clip(a['proposal']+delta, a['z'][0], a['z'][-1])
        pred = fast_depth(a['time'], a['valid'], a['old_z'], a['epoch'], a['z'], a['cost'],
                          a['phi'], prop, a['distance'], **policy)
        rows.append({'trial': entry['trial'], 'mode': entry['mode'],
                     'metrics': score_z(a, pred), 'phases': score_phases(a, pred)})
    modes = sorted({r['mode'] for r in rows})
    def macro(items):
        scores = []
        for mode in modes:
            vals_mode = [r['metrics'] for r in items if r['mode'] == mode]
            scores.append(np.mean([r['z_centered_mae_mm']+.2*r['z_raw_mae_mm']
                                   +.4*r['z_step_p95_mm']+.1*r['z_step_max_mm'] for r in vals_mode]))
        return float(np.mean(scores))
    if baselines is not None:
        for mode in modes:
            new = [r['metrics'] for r in rows if r['mode'] == mode]
            old = [r['metrics'] for r in baselines if r['mode'] == mode]
            for key, tol in (('z_raw_mae_mm', .005), ('z_centered_mae_mm', .005),
                             ('z_step_p95_mm', .005), ('z_step_max_mm', .05)):
                difference = float(np.mean([r[key] for r in new])-np.mean([r[key] for r in old]))
                if difference > tol:
                    failures.append({'mode': mode, 'metric': key, 'delta_mm': difference})
        for phase in rows[0]['phases']:
            count = sum(r['phases'][phase]['output_frames'] for r in rows)
            if count < 100:
                continue
            for key in ('z_raw_mae_mm', 'z_global_centered_mae_mm'):
                new = [r['phases'][phase][key] for r in rows if r['phases'][phase][key] is not None]
                old = [r['phases'][phase][key] for r in baselines if r['phases'][phase][key] is not None]
                difference = float(np.mean(new)-np.mean(old))
                if difference > .005:
                    failures.append({'phase': phase, 'metric': key, 'delta_mm': difference})
    result = macro(rows)
    qualified = baselines is None or (not failures and result < .99*baseline_score)
    return result, qualified, {'trials': rows, 'failures': failures}


def save_model(path, params, mean, scale, component):
    names = ('w1', 'b1', 'w2', 'b2', 'w3', 'b3') if component == 'xy' else tuple(f'p{i}' for i in range(6))
    np.savez_compressed(path, mean=mean, scale=scale, **dict(zip(names, params)))


def checkpoint(path, params, first, second, best_params, state):
    arrays = {**{f'p{i}': p for i, p in enumerate(params)},
              **{f'm{i}': p for i, p in enumerate(first)}, **{f'v{i}': p for i, p in enumerate(second)},
              **{f'best{i}': p for i, p in enumerate(best_params or params)},
              'state_json': np.array(json.dumps(state, allow_nan=False))}
    temporary = path.with_suffix('.writing.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def train(component, resume=False, max_epochs=None):
    bundle = json.loads((OUT/'data_manifest.json').read_text(encoding='utf8'))
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    spec = bundle[component]
    folder = OUT/component
    folder.mkdir(parents=True, exist_ok=True)
    effective_max = protocol['max_true_epochs'] if max_epochs is None else int(max_epochs)
    if effective_max < protocol['max_true_epochs']:
        raise ValueError('Do not shorten the declared training cap')
    if (folder/'freeze.json').exists():
        previous = json.loads((folder/'freeze.json').read_text(encoding='utf8'))
        if not (resume and previous['status'] == 'BUDGET_TRUNCATED'
                and effective_max > previous['epochs_run']):
            raise FileExistsError('Preserve already frozen training result')
        cutoff = folder/f"budget_cutoff_{previous['epochs_run']}.json"
        if not cutoff.exists():
            dump(cutoff, previous)
    for r in spec['files']:
        if sha(OUT/r['file']) != r['sha256']:
            raise ValueError('Prepared source data changed: '+r['file'])
    with np.load(OUT/spec['training_file'], allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    mean, scale = data['mean'], data['scale']
    u = (data['x']-mean)/scale
    if component == 'z':
        u = np.clip(u, -8., 8.)
    params = [data[f'init{i}'].copy() for i in range(6)]
    first = [np.zeros_like(v) for v in params]
    second = [v.copy() for v in first]
    rng = np.random.default_rng(spec['seed'])
    rng.bit_generator.state = json.loads(str(data['rng_json']))
    vals = []
    for r in spec['validation']:
        with np.load(OUT/r['file'], allow_pickle=False) as f:
            vals.append((r, {k: f[k].copy() for k in f.files}))
    controller = Plateau(lr=spec['initial_lr'])
    baseline_rows = spec.get('baseline_validation')
    baseline_score = spec.get('teacher_validation_objective')
    policy = bundle.get('depth_policy')
    history, best_params, best_metric, best_epoch, updates, start_epoch = [], None, None, None, 0, 0
    state_file = folder/'last_state.npz'
    if resume:
        with np.load(state_file, allow_pickle=False) as f:
            state = json.loads(str(f['state_json']))
            if state['protocol_sha256'] != sha(OUT/'protocol.json') or state['data_manifest_sha256'] != sha(OUT/'data_manifest.json'):
                raise ValueError('Resume protocol/data changed')
            params = [f[f'p{i}'].copy() for i in range(6)]
            first = [f[f'm{i}'].copy() for i in range(6)]
            second = [f[f'v{i}'].copy() for i in range(6)]
            best_params = [f[f'best{i}'].copy() for i in range(6)] if state['best_epoch'] is not None else None
        controller = Plateau(**state['controller'])
        rng.bit_generator.state = state['rng_state']
        history, updates, start_epoch = state['history'], state['updates'], state['epoch']
        best_metric, best_epoch = state['best_metric'], state['best_epoch']
    elif state_file.exists():
        raise FileExistsError('Training state exists; explicitly use --resume')
    beginning = time.perf_counter()
    stop_reason = 'max_epochs_budget_truncated'
    dump(folder/'status.json', {'status': 'TRAINING', 'component': component, 'seed': spec['seed'],
                               'pid': os.getpid(), 'epoch': start_epoch, 'jit_depth': JIT})
    old = [data[f'old{i}'].copy() for i in range(6)]
    if component == 'xy':
        old_score, _, old_rows = val_xy(old, vals, mean, scale)
    else:
        old_score, _, old_rows = val_z(old, vals, mean, scale, policy, baseline_rows, baseline_score)
    dump(folder/'old_validation.json', {'objective': old_score, 'rows': old_rows, 'weights': spec['old_model'],
                                      'training_performed': False})
    print(component, 'training starts', len(u), 'samples', len(vals), 'validation Trials', 'old', old_score, flush=True)
    for epoch in range(start_epoch+1, effective_max+1):
        epoch_start = time.perf_counter()
        lr_used = controller.lr
        order = rng.permutation(len(u))
        for ids in np.array_split(order, max(1, (len(u)+255)//256)):
            if component == 'xy':
                grads = gradients_xy(params, u[ids], data['y'][ids])
            else:
                pair = data['pair'][rng.integers(0, len(data['pair']), size=len(ids))]
                grads = gradients_z(params, u[ids], data['y'][ids], data['weight'][ids],
                                    pair[:, 0], pair[:, 1], u, data['y'], data['weight'])
            updates += 1
            for p, g, m, v in zip(params, grads, first, second):
                m *= .9
                m += .1*g
                v *= .999
                v += .001*g*g
                p -= lr_used*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
        if not all(np.isfinite(p).all() for p in params):
            raise FloatingPointError('Nonfinite training weights')
        pred = forward(params, u)[2]
        if component == 'xy':
            mse = float(np.mean(np.sum((pred-data['y'])**2, axis=1)))
            loss = mse
            metric, qualified, rows = val_xy(params, vals, mean, scale)
        else:
            w = data['weight'][:, None]
            mse = float(np.mean((pred-data['y'])**2*w))
            pair = data['pair']
            diff = pred[pair[:, 1]]-pred[pair[:, 0]]-(data['y'][pair[:, 1]]-data['y'][pair[:, 0]])
            pw = ((data['weight'][pair[:, 0]]+data['weight'][pair[:, 1]])/2)[:, None]
            loss = mse+.5*float(np.mean(pred*pred*w))+.5*float(np.mean(diff*diff*pw))
            metric, qualified, rows = val_z(params, vals, mean, scale, policy, baseline_rows, baseline_score)
        loss += .0005*sum(float(np.sum(params[k]**2)) for k in (0, 2, 4))
        if qualified and (best_metric is None or metric < best_metric):
            best_metric, best_epoch = float(metric), epoch
            best_params = [p.copy() for p in params]
            save_model(folder/'best_model.npz', best_params, mean, scale, component)
        event = controller.observe(metric)
        record = {'epoch': epoch, 'updates': updates, 'lr_used': lr_used, 'next_lr': controller.lr,
                  'train_mse': mse, 'train_regularized_objective': loss, 'validation_objective': metric,
                  'qualified': bool(qualified), 'best_epoch': best_epoch, 'best_objective': best_metric,
                  'bad_epochs': controller.bad_epochs, 'scheduler_event': event,
                  'unique_samples_visited': len(np.unique(order)), 'training_samples': len(u),
                  'epoch_seconds': time.perf_counter()-epoch_start, 'validation': rows}
        history.append(record)
        state = {'epoch': epoch, 'updates': updates, 'history': history, 'controller': asdict(controller),
                 'rng_state': rng.bit_generator.state, 'best_epoch': best_epoch, 'best_metric': best_metric,
                 'protocol_sha256': sha(OUT/'protocol.json'), 'data_manifest_sha256': sha(OUT/'data_manifest.json'),
                 'effective_max_epochs': effective_max}
        checkpoint(state_file, params, first, second, best_params, state)
        dump(folder/'status.json', {'status': 'TRAINING', 'component': component, 'pid': os.getpid(),
                                   **{k: record[k] for k in ('epoch', 'updates', 'lr_used', 'next_lr', 'validation_objective',
                                                            'best_epoch', 'best_objective', 'bad_epochs', 'scheduler_event')},
                                   'elapsed_seconds_this_session': time.perf_counter()-beginning})
        if epoch % 10 == 0 or event != 'continue':
            print(component, 'epoch', epoch, 'val', round(metric, 7), 'best', best_epoch,
                  'lr', lr_used, 'bad', controller.bad_epochs, event, flush=True)
        if event == 'early_stop_min_lr_plateau':
            stop_reason = event
            break
    if best_params is None:
        raise RuntimeError('No qualified validation checkpoint: do not publish deployable model')
    save_model(folder/'best_model.npz', best_params, mean, scale, component)
    status = 'COMPLETED_PLATEAU' if stop_reason == 'early_stop_min_lr_plateau' else 'BUDGET_TRUNCATED'
    report = {'status': status, 'component': component, 'seed': spec['seed'], 'epochs_run': history[-1]['epoch'],
              'best_epoch': best_epoch, 'best_validation_objective': best_metric,
              'old_validation_objective': old_score, 'validation_change_percent': 100*(best_metric-old_score)/old_score,
              'stop_reason': stop_reason, 'final_learning_rate': controller.lr,
              'lr_reductions': controller.reductions, 'optimizer_updates': updates,
              'training_samples': len(u), 'validation_trials': len(vals), 'history': history,
              'test_used_for_fitting_or_selection': False, 'historical_training_reproduction_performed': False,
              'hardware_or_UI_changed': False, 'wall_seconds_this_session': time.perf_counter()-beginning}
    report.update(effective_max_epochs=effective_max, resumed_from_epoch=start_epoch,
                  extension_is_same_optimizer_run_not_additional_experimental_arm=effective_max > protocol['max_true_epochs'])
    dump(folder/'report.json', report)
    freeze = {k: v for k, v in report.items() if k != 'history'}
    freeze.update(model='best_model.npz', model_sha256=sha(folder/'best_model.npz'),
                  protocol_sha256=sha(OUT/'protocol.json'), data_manifest_sha256=sha(OUT/'data_manifest.json'),
                  trainer_sha256=sha(Path(__file__)))
    dump(folder/'freeze.json', freeze)
    dump(folder/'status.json', freeze)
    print(component, status, 'best', best_epoch, best_metric, 'old', old_score, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('xy', 'z'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-epochs', type=int, help='Continue an unfinished budget cutoff using the complete saved state')
    args = parser.parse_args()
    train(args.component, args.resume, args.max_epochs)
