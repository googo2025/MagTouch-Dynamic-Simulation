"""Whole-category Source-only depth training and frozen paired evaluation.

No hardware/UI imports. Raw files and previous freezes are read-only. The
complex paths are additional final evaluation, not the sole benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import numpy as np

from training.analyze_z_evidence_r334 import ROOT, dump, sha, summary, objective
from training.standard_motion_protocol_r344 import OUT as ROLE_OUT, load
from training.develop_unified_depth_r341 import models, cache, apply, data_for_summary
from training.causal_depth_network_r337 import DepthMLP
from training.unified_depth_runtime_r341 import SourceFeatureSupport, UnifiedDepthPolicy
from training.fast_depth_features_r344 import augment
from training.train_standard_depth_r344 import anchors
from training.evaluate_standard_depth_r346 import frozen as r346_frozen, proposal as r346_proposal
from training.analyze_r332_online_acceptance_r333 import scoring

OUT = ROOT/'analysis/r347_all_motion_20260929'
SUPPORTED = {'GuardedNeuralFieldMemoryR130', 'GuardedNeuralFieldMemoryR133', 'UnifiedEvidenceTactileR332'}
METRICS = ('xy_raw_mean_mm', 'xy_centered_mean_mm', 'z_raw_mae_mm',
           'z_centered_mae_mm', 'xyz_raw_rmse_mm', 'xyz_centered_rmse_mm',
           'z_step_p95_mm', 'z_step_max_mm', 'coverage')


def baseline():
    decoder, network, depth_policy = models()
    f = json.loads((ROOT/'analysis/r341_unified_depth_20260929/freeze.json').read_text(encoding='utf-8'))
    for dep in f['dependencies']:
        if sha(ROOT/dep['path']) != dep['sha256']:
            raise ValueError('Changed frozen baseline dependency: '+dep['path'])
    return decoder, network, depth_policy, f, UnifiedDepthPolicy(**f['policy']), SourceFeatureSupport.load(ROOT/f['support_file'])


def declare():
    file = OUT/'protocol.json'
    if file.exists():
        return json.loads(file.read_text(encoding='utf-8'))
    roles = json.loads((ROLE_OUT/'dataset_roles.json').read_text(encoding='utf-8'))
    accepted = []; excluded = []
    for row in roles['records']:
        m = json.loads((ROOT/row['path']/'meta.json').read_text(encoding='utf-8'))
        family = m.get('live_prediction_recording', {}).get('model_family')
        record = {**row, 'recorded_family': family, 'parameters': m['trial_parameters']}
        if family not in SUPPORTED:
            excluded.append({**record, 'reason': 'Saved stream has no compatible causal XY/history. Requires separate raw-stream replay, not silently zero-scored.'})
        else:
            accepted.append(record)
    document = {'schema': 'R347-AllMotionProtocol-1', 'records': accepted, 'excluded': excluded,
        'counts': dict(Counter((r['role']+' / '+r['mode']) for r in accepted)),
        'target_fit_rows': 0, 'target_selection_rows': 0, 'complex_fit_rows': 0, 'complex_selection_rows': 0,
        'fit_roles': 'Existing 45 Source standard training Trials; 12 Source selection Trials. Whole Trial roles preserved.',
        'training': 'R340 warm start; 56D and 59D (causal magnetic depth tangent) controlled comparison; absolute and differential Z supervision; measured static zero-correction anchors.',
        'selection': 'Every standard category protects BOTH raw and translation-removed Z versus R341; no Target or complex selection.',
        'evaluation': 'Training seen-fit, selection, other Source standards, Target standards and complex additional evaluation reported separately.',
        'translation': 'One constant XYZ mean residual per Trial, diagnostics only. No rotation, scaling, time shift, template fit or deployment calibration.',
        'reference': 'Nominal >=0.4 mm action is depth calibration reference; extreme shallow actions reported as output counts, not contact recall or forced XYZ labels.',
        'scope': 'Conditional depth module on recorded XY, memory and permissions. Not a new full-system online experiment.',
        'historical_state_complex_exposure': True, 'prospective_blind': False,
        'roles_file_sha256': sha(ROLE_OUT/'dataset_roles.json')}
    dump(file, document)
    print('Protocol compatible', len(accepted), 'separate raw replay required', len(excluded), document['counts'], flush=True)
    return document


def net_proposal(a, model, x):
    b = dict(a); v = a['valid']
    b['proposal'] = np.clip(a['base'][v, 2]+np.clip(model.predict(x[v]).reshape(-1), -1, 1), a['z'][0], a['z'][-1])
    return b


def train():
    protocol = declare()
    if (OUT/'choice.json').exists():
        raise FileExistsError('Do not retrain after final evaluation choice.')
    decoder, teacher, dp, oldf, policy, support = baseline()
    records = load('train'); selection = load('selection')
    train_data = []; val_data = []
    for row in records:
        if 'old1mm' in row['path'].lower() or 'yelow' in row['path'].lower():
            raise ValueError('Wrong assembly in training')
        a, s = cache(ROOT/row['path'], decoder, teacher, dp)
        ids = np.flatnonzero(a['valid'] & a['calibrated_action'])[::8]
        if not len(ids):
            raise ValueError('No reference training frames: '+row['path'])
        train_data.append((a, s, ids, augment(decoder.atlas, a['time'], a['x'], a['valid'], a['epoch'])))
        print('Source fit', s['trial'], s['mode'], len(ids), flush=True)
    for row in selection:
        a, s = cache(ROOT/row['path'], decoder, teacher, dp)
        distance = support.distance(a['x'][a['valid']]); previous = apply(a, policy, distance)[0]
        val_data.append((a, s, distance, summary(data_for_summary(a, s), previous),
                         augment(decoder.atlas, a['time'], a['x'], a['valid'], a['epoch'])))
    modes = sorted(set(s['mode'] for a, s, *_ in val_data))
    def macro(rows):
        return float(np.mean([np.mean([objective(r) for r, s in rows if s['mode'] == mode]) for mode in modes]))
    baseline_score = macro([(r, s) for a, s, d, r, xx in val_data])
    all_results = []
    anchor = anchors(decoder)
    declared = {'warm_start': 'Frozen R340 weights and 56D normalization', 'dimensions': [56, 59],
        'epochs': 60, 'batch': 256, 'seed': 347, 'learning_rate': 0.0002, 'weight_decay': 0.001,
        'loss': 'Trial-equal absolute residual MSE + 0.5 teacher residual MSE + 0.5 within-Trial differential residual MSE + weight L2.',
        'anchor_trial_weight': 4., 'stride': 8, 'target_and_complex_training_or_selection': 0,
        'differential_pairs': 'Consecutive subsampled frames of same B0 epoch with positive <=0.35 s gap. Actual commanded depth change retained.',
        'criterion': 'Each mode raw/centered mean <= R341+0.005mm; each mode P95 step <=R341+0.005mm; max-step mean <=R341+0.05mm; objective improves at least 1%.',
        'policy': asdict(policy), 'source_support': 'Frozen R341 Source-only support; unchanged to isolate retraining.'}
    dump(OUT/'declared_training.json', declared)
    for dimension in declared['dimensions']:
        xlist=[]; ylist=[]; wlist=[]; pairs=[]; cursor=0; owners=[]
        for owner, (a, s, ids, xx) in enumerate(train_data):
            xlist.append(xx[ids, :dimension]); ylist.append(np.clip(a['q'][ids, 2]-a['base'][ids, 2], -1, 1)[:, None])
            wlist.append(np.full(len(ids), 1/len(ids))); owners.extend([owner]*len(ids))
            dt=np.diff(a['time'][ids])/1e9
            pair_ok=(dt>0)&(dt<=.35)&(a['epoch'][ids[1:]]==a['epoch'][ids[:-1]])
            pp=np.flatnonzero(pair_ok); pairs.extend(zip((cursor+pp).tolist(), (cursor+pp+1).tolist()))
            cursor+=len(ids)
        xlist.append(anchor if dimension==56 else np.column_stack((anchor, np.zeros((len(anchor), 3)))))
        ylist.append(np.zeros((len(anchor), 1))); wlist.append(np.full(len(anchor), 4/len(anchor)))
        x=np.vstack(xlist); y=np.vstack(ylist); weight=np.concatenate(wlist); weight*=len(weight)/weight.sum()
        pair=np.asarray(pairs, int)
        model=DepthMLP(32, 347); model.mean=teacher.mean.copy(); model.scale=teacher.scale.copy()
        if dimension==59:
            extra_mean=np.average(x[:, 56:], axis=0, weights=weight)
            extra_scale=np.maximum(np.sqrt(np.average((x[:, 56:]-extra_mean)**2, axis=0, weights=weight)), .1)
            model.mean=np.r_[model.mean, extra_mean]; model.scale=np.r_[model.scale, extra_scale]
        u=np.clip((x-model.mean)/model.scale, -8, 8)
        params=[p.copy() for p in teacher.params]
        if dimension==59:
            params[0]=np.vstack((params[0], np.zeros((3, 32))))
        teach=teacher.predict(x[:, :56]).reshape(-1, 1); teach[-len(anchor):]=0
        moment=[np.zeros_like(p) for p in params]; square=[p.copy() for p in moment]
        rng=np.random.default_rng(347); iteration=0; history=[]; best=None
        def forward(idx):
            h1=np.tanh(u[idx]@params[0]+params[1]); h2=np.tanh(h1@params[2]+params[3]); pred=h2@params[4]+params[5]
            return h1, h2, pred
        def gradient(idx, output_gradient):
            h1, h2, _=forward(idx); g2=(output_gradient@params[4].T)*(1-h2*h2); g1=(g2@params[2].T)*(1-h1*h1)
            return [u[idx].T@g1, g1.sum(0), h1.T@g2, g2.sum(0), h2.T@output_gradient, output_gradient.sum(0)]
        for epoch in range(1, declared['epochs']+1):
            for idx in np.array_split(rng.permutation(len(x)), max(1, (len(x)+255)//256)):
                _, _, pred=forward(idx)
                gout=(2*(pred-y[idx])+(pred-teach[idx]))*weight[idx, None]/len(idx)
                grad=gradient(idx, gout)
                if len(pair):
                    pp=pair[rng.integers(0, len(pair), size=len(idx))]
                    ia, ib=pp[:, 0], pp[:, 1]
                    pa=forward(ia)[2]; pb=forward(ib)[2]
                    pw=(weight[ia]+weight[ib])/2
                    gpair=((pb-pa)-(y[ib]-y[ia]))*pw[:, None]/len(idx)
                    ga, gb=gradient(ia, -gpair), gradient(ib, gpair)
                    grad=[g+a+b for g,a,b in zip(grad, ga, gb)]
                for k in (0, 2, 4): grad[k]+=.001*params[k]
                iteration+=1
                for p, g, m, v in zip(params, grad, moment, square):
                    m*=.9; m+=.1*g; v*=.999; v+=.001*g*g
                    p-=.0002*(m/(1-.9**iteration))/(np.sqrt(v/(1-.999**iteration))+1e-8)
            if epoch%10: continue
            model.params=params; result=[]; failures=[]
            for a, s, distance, previous, xx in val_data:
                xyz=apply(net_proposal(a, model, xx[:, :dimension]), policy, distance)[0]
                row=summary(data_for_summary(a, s), xyz); result.append((row, s))
            for mode in modes:
                new=[r for r,s in result if s['mode']==mode]
                previous=[r for a,s,d,r,xx in val_data if s['mode']==mode]
                for key, tol in (('z_raw_mae_mm', .005), ('z_centered_mae_mm', .005), ('z_step_p95_mm', .005), ('z_step_max_mm', .05)):
                    delta=float(np.mean([r[key] for r in new])-np.mean([r[key] for r in previous]))
                    if delta>tol: failures.append({'mode':mode, 'metric':key, 'delta_mm':delta, 'tolerance_mm':tol})
            score=macro(result); qualified=not failures and score<.99*baseline_score
            item={'epoch':epoch, 'objective':score, 'qualified':qualified, 'failures':failures,
                  'trials':[{'path':s['path'], 'mode':s['mode'], **r} for r,s in result]}
            history.append(item)
            print('Source all-category', dimension, epoch, round(score, 6), 'qualified', qualified, 'failed',len(failures), flush=True)
            if qualified and (best is None or score<best['objective']):
                best={**item}; best_params=[p.copy() for p in params]
        if best:
            model.params=best_params
        file=OUT/f'warm_z_mlp_{dimension}.npz'; model.save(file)
        all_results.append({'dimension':dimension, 'file':file.relative_to(ROOT).as_posix(), 'selected':best, 'history':history,
                            'training_samples':len(x), 'training_pair_count':len(pair)})
    qualified=[r for r in all_results if r['selected'] is not None]
    chosen=min(qualified, key=lambda r:r['selected']['objective']) if qualified else None
    dump(OUT/'source_selection.json', {'declared':declared, 'baseline_objective':baseline_score, 'models':all_results,
                                      'selected':chosen, 'target_selection_rows':0, 'complex_selection_rows':0})
    dependencies=[OUT/'protocol.json', OUT/'declared_training.json', OUT/'source_selection.json', ROOT/'training/all_motion_depth_r347.py',
                  ROOT/'training/fast_depth_features_r344.py', ROOT/'analysis/r341_unified_depth_20260929/freeze.json']
    dependencies += [ROOT/r['path'] for r in oldf['dependencies']]
    if chosen: dependencies.append(ROOT/chosen['file'])
    choice={'schema':'R347-FrozenBeforeTargetAndComplex-1', 'selected':chosen, 'fallback':'R341' if chosen is None else None,
            'policy':asdict(policy), 'support_file':oldf['support_file'], 'target_fit_rows':0, 'target_selection_rows':0,
            'complex_fit_rows':0, 'complex_selection_rows':0,
            'dependencies':[{'path':p.relative_to(ROOT).as_posix(), 'sha256':sha(p)} for p in dict.fromkeys(dependencies)]}
    dump(OUT/'choice.json', choice)
    print('Frozen final choice', 'baseline retained' if chosen is None else (chosen['dimension'],chosen['selected']['epoch']), flush=True)


def slices(a, xyz):
    dt=np.diff(a['time'])/1e9
    velocity=np.zeros(len(dt)); good=(dt>0)&(dt<=.25)
    velocity[good]=np.diff(a['q'][:,2])[good]/dt[good]
    depth_velocity=np.r_[0., velocity]
    spatial=np.r_[0., np.linalg.norm(np.diff(a['q'][:,:2],axis=0),axis=1)]
    deep=a['calibrated_action']; result={}
    for name, mask in (('pressing',deep&(depth_velocity>.005)), ('withdrawing',deep&(depth_velocity<-.005)),
                      ('hold',deep&(np.abs(depth_velocity)<=.005)&(spatial<1e-4))):
        result[name]=scoring({'q':a['q'],'xyz':xyz,'valid':a['valid']}, mask)
    shallow=a['action']&(a['q'][:,2]>=0)&(a['q'][:,2]<.4)
    result['nominal_shallow']={'action_frames':int(shallow.sum()), 'output_frames':int((shallow&a['valid']).sum()),
                             'meaning':'Output count only, not verified contact recall or XYZ accuracy.'}
    return result


def evaluate():
    protocol=declare(); choice_file=OUT/'choice.json'
    choice=json.loads(choice_file.read_text(encoding='utf-8'))
    for dep in choice['dependencies']:
        if sha(ROOT/dep['path'])!=dep['sha256']:raise ValueError('Frozen dependency changed: '+dep['path'])
    if any(choice[k] for k in ('target_fit_rows','target_selection_rows','complex_fit_rows','complex_selection_rows')):
        raise ValueError('Heldout fitting/selection forbidden')
    decoder, teacher, dp, oldf, policy, support=baseline()
    rf, rm, rs=r346_frozen(); rp=UnifiedDepthPolicy(**rf['policy'])
    selected=choice['selected']; model=DepthMLP.load(ROOT/selected['file']) if selected else None
    rows=[]; excluded=list(protocol['excluded']); total=0
    for record in protocol['records']:
        key=sha(ROOT/record['path']/'meta.json')[:16]; output=OUT/'trial_reports'/f'{key}.json'
        if output.exists():
            saved=json.loads(output.read_text(encoding='utf-8'))
            if saved['choice_sha256']!=sha(choice_file):raise ValueError('Stale trial report')
            rows.append(saved);continue
        try:
            for raw in record['raw_manifest']:
                if sha(ROOT/raw['path'])!=raw['sha256']:raise ValueError('Changed raw dependency')
            a,s=cache(ROOT/record['path'],decoder,teacher,dp)
            distance=support.distance(a['x'][a['valid']]); previous=apply(a,policy,distance)[0]
            alternative=apply(r346_proposal(a,decoder,rm,rf),rp,rs.distance(a['x'][a['valid']]))[0]
            if model:
                x=a['x'] if selected['dimension']==56 else augment(decoder.atlas,a['time'],a['x'],a['valid'],a['epoch'])
                new=apply(net_proposal(a,model,x),policy,distance)[0]
            else:new=previous.copy()
            d=data_for_summary(a,s); metrics={name:summary(d,xyz) for name,xyz in (('original',a['old']),('R341',previous),('R346',alternative),('selected',new))}
            stream=OUT/'streams'/f'{key}.npz';stream.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(stream,time=a['time'],q=a['q'],valid=a['valid'],action=a['action'],
                                calibrated=a['calibrated_action'],original=a['old'],R341=previous,R346=alternative,selected=new)
            saved={**record, 'choice_sha256':sha(choice_file),'stream':stream.relative_to(ROOT).as_posix(),
                   'metrics':metrics,'phase_slices':{name:slices(a,xyz) for name,xyz in (('original',a['old']),('R341',previous),('selected',new))},
                   'conditional_permission_xy_history':True,'raw_frames':len(a['time']), 'valid_frames':int(a['valid'].sum()),
                   'translation_for_diagnostics_only':True,'b0_amplitude_check':s['b0_amplitude_check']}
            dump(output,saved);rows.append(saved);total+=len(a['time'])
            print('All motion audit',record['role'], 'Target' if record['target'] else 'Source',record['trial'], record['mode'],
                  'outputs',saved['valid_frames'], 'Z', {n:round(v.get('z_raw_mae_mm',0),4) for n,v in metrics.items()}, flush=True)
        except (ValueError, KeyError, AssertionError) as exc:
            excluded.append({**record,'reason':'Data compatibility failure: '+str(exc)})
            print('EXPLICIT UNAVAILABLE',record['path'],str(exc),flush=True)
    groups={}
    for role,target,mode in sorted(set((r['role'],r['target'],r['mode']) for r in rows)):
        subset=[r for r in rows if (r['role'],r['target'],r['mode'])==(role,target,mode)]
        groups[f'{role}/{"Target" if target else "Source"}/{mode}']={'trials':len(subset),
            'zero_reference_trials':sum(r['metrics']['original']['reference_frames']==0 for r in subset),
            'macro':{k:{n:float(np.mean(v)) if (v:=[r['metrics'][n][k] for r in subset if r['metrics'][n].get(k) is not None]) else None
                        for n in ('original','R341','R346','selected')} for k in METRICS}}
    dump(OUT/'report.json',{'protocol_sha256':sha(OUT/'protocol.json'), 'choice':choice,'rows':rows,'groups':groups,'excluded':excluded,
                          'raw_frame_total':sum(r['raw_frames'] for r in rows), 'accepted_frame_total':sum(r['valid_frames'] for r in rows),
                          'target_fit_rows':0,'complex_fit_rows':0,'fresh_online_test':False,
                          'scope':protocol['scope']})
    print('Completed',len(rows),'Trials, exclusions',len(excluded),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('declare','train','evaluate'));args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'declare':declare,'train':train,'evaluate':evaluate}[args.task]()
