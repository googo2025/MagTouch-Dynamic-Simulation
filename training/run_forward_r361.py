"""Source-only command-to-magnetic-field experiment. Run as a module.

Reads native raw magnetic/command/B0 records; never uses live XYZ as targets.
All waypoint paths and all target-device records are held out from fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from training.build_trajectory_dataset_v1 import (assemble, command_segments, load_magnetic,
                                                read_rows, reference_origin)
from training.magnetic_forward_r361 import MagneticForward, design, fit_readout, spatial_features

ROOT = Path(__file__).resolve().parents[1]
D, E, F = tuple(range(86, 110)), tuple(range(112, 122)), tuple(range(124, 134))
GROUP = {
    'D': 'Plan_20260927_213250_R133_D_HAD239_COVERAGE_TRAIN__mag_5x5-had-239',
    'E': 'Plan_20260927_222827_R133_E_HAD239_UNSEEN_COMBINATIONS__mag_5x5-had-239',
    'F': 'Plan_20260927_230418_R133_F_YELLOW_UNSEEN_COMBINATIONS__mag_5x5-had-Yelow',
}


def new_path(group, number):
    matches = list((ROOT/'data/S20260927_A'/GROUP[group]).glob(f'Trial_{number:06d}_*'))
    if len(matches) != 1:
        raise ValueError((group, number, len(matches)))
    return matches[0]


OUT = ROOT/'analysis/r361_command_forward_20261001'
MODEL_DIR = OUT/'models'
CHANNELS = [f'S{s}_B{a}_uT' for s in range(1, 6) for a in 'xyz']
TAUS = [(), (.12, .5, 2.), (.3, 1., 4.)]
ALPHAS = [.0001, .001, .01, .1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def dump(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def inventory():
    rows=[]
    for group, ids, role, device in [('D', D, 'train', 'source'), ('E', E, 'validation', 'source'),
                                    ('F', F, 'test', 'target')]:
        rows += [{'path':new_path(group,n), 'role':role, 'device':device,
                  'id':f'{group}_{n:06d}'} for n in ids]
    p=ROOT/'data/S20260927_A/Plan_20260927_161540_R132_A_HAD239_R130_NEURAL_BLIND_DAY__mag_5x5-had-239'
    for n in (12,13,14):
        rows.append({'path':next(p.glob(f'Trial_{n:06d}_*')), 'role':'static_calibration',
                     'device':'source', 'id':f'CAL_{n:06d}'})
    for name in ('r354_final_online_20260930', 'r355_exact_space_depth_online_20260930'):
        report=json.loads((ROOT/'analysis'/name/'report.json').read_text(encoding='utf8'))
        for device, plan in report['plans'].items():
            for trial in plan['trials']:
                if trial['mode']=='none': continue
                rows.append({'path':ROOT/trial['path'], 'role':'test', 'device':device,
                             'id':name[:4]+'_'+device+'_'+trial['trial_id']})
    # Immutable per-role manifest precedes numerical fitting or outcome inspection.
    for r in rows:
        meta=json.loads((r['path']/'meta.json').read_text(encoding='utf8'))
        r['mode']=meta['trial_parameters']['command_mode']
        r['indenter']=meta['trial_parameters'].get('indenter_id','')
        if not r['indenter'].startswith('final_5mm'):
            raise ValueError('non-5mm recording in new forward experiment: '+str(r['path']))
        if r['role'] in ('train','validation','static_calibration'):
            assert r['device']=='source' and r['mode']!='waypoint_xyz'
        r['hashes']={fn:sha(r['path']/fn) for fn in ('magnetic_raw.csv','stage_commands.csv','events.csv','meta.json')}
    serial=[{**r,'path':r['path'].relative_to(ROOT).as_posix()} for r in rows]
    manifest=OUT/'data_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf8'))!=serial:
        raise ValueError('frozen data manifest changed')
    dump(manifest,serial)
    return rows


def load_row(row):
    dest=OUT/'cache'/f"{row['id']}.npz"
    if dest.exists():
        with np.load(dest, allow_pickle=False) as f: a={k:f[k].copy() for k in f.files}
    else:
        path=row['path']; meta=json.loads((path/'meta.json').read_text(encoding='utf8'))
        segments=command_segments(read_rows(path/'stage_commands.csv'))
        full, _=assemble(load_magnetic(path/'magnetic_raw.csv'),segments,
                         read_rows(path/'events.csv'),reference_origin(meta))
        # 10 Hz causal last-available samples. No temporal interpolation of B.
        ix=np.arange(0,len(full['t_ns']),4)
        keys={'q':'xyz_mm','b':'delta_b_uT','valid':'input_valid','b0':'b0_id',
              'moving':'command_moving','age':'hold_age_s','boundary':'command_boundary',
              'segment':'command_segment_id'}
        a={k:full[v][ix] for k,v in keys.items()}
        a['t']=(full['t_ns'][ix]-full['t_ns'][0])/1e9
        a['reset']=np.r_[True,np.diff(a['b0'])!=0]
        np.savez_compressed(dest,**a)
    a.update(row)
    known=np.isfinite(a['q']).all(1)
    # All valid magnetic frames; NEVER condition on inverse model acceptance.
    a['primary']=a['valid']&known&(a['q'][:,2]>=.4)&(a['q'][:,2]<=3.)
    a['safe']=a['valid']&known&(a['q'][:,2]<=-.2)
    a['shallow']=a['valid']&known&(a['q'][:,2]>0)&(a['q'][:,2]<.4)
    a['stable']=a['primary']&~a['moving']&(a['age']>=.2)&(a['age']<=.5)
    a['fit']=a['primary']|a['safe']
    if row['role']=='static_calibration': a['fit']=a['stable']
    return a


def metric(pred, a, mask=None):
    m=a['primary'] if mask is None else mask
    if not m.any(): return {'frames':0}
    e=pred[m]-a['b'][m]; y=a['b'][m]
    rmse=float(np.sqrt(np.mean(e*e)))
    denom=max(float(np.sqrt(np.mean(y*y))),1e-6)
    pred5=pred[m].reshape(-1,5,3); real5=y.reshape(-1,5,3)
    pn=np.linalg.norm(pred5,axis=2); rn=np.linalg.norm(real5,axis=2)
    eligible=(pn>10)&(rn>10)
    cosine=np.sum(pred5*real5,2)/np.maximum(pn*rn,1e-9)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    return {'frames':int(m.sum()),'rmse_uT':rmse,'nrmse':rmse/denom,
            'channel_rmse_uT':np.sqrt(np.mean(e*e,0)).tolist(),
            'sensor_vector_rmse_uT':np.sqrt(np.mean(np.sum(e.reshape(-1,5,3)**2,2),0)).tolist(),
            'vector_error_p95_uT':float(np.quantile(np.linalg.norm(e,axis=1),.95)),
            'field_constant_bias_uT':e.mean(0).tolist(),
            'constant_field_bias_removed_rmse_uT':float(np.sqrt(np.mean((e-e.mean(0))**2))),
            'sensor_direction_median_deg':float(np.median(angle[eligible])) if eligible.any() else None,
            'sensor_amplitude_mae_uT':float(np.mean(np.abs(pn-rn)))}


def macro(rows, name):
    selected=[r[name] for r in rows if r[name].get('frames',0)]
    return {k:float(np.mean([r[k] for r in selected])) for k in ('rmse_uT','nrmse')} if selected else {}


def matrices(rows, taus):
    xs=[]; ys=[]; ws=[]
    for a in rows:
        mask=a['fit']
        if not mask.any(): continue
        xx=design(a['q'],a['t'],taus,a['reset'])
        if a['role']=='static_calibration':
            # Calibrated held-contact anchors have zero dynamic features.
            xx[:,110:]=0.
        w=np.where(a['boundary'][mask],.5,1.)
        w*= (2. if a['role']=='static_calibration' else 1.)/w.sum()
        xs.append(xx[mask]); ys.append(a['b'][mask]); ws.append(w)
    return np.vstack(xs),np.vstack(ys),np.concatenate(ws)


def train(rows):
    fitting=[a for a in rows if a['role'] in ('train','static_calibration')]
    valid=[a for a in rows if a['role']=='validation']
    bounds=np.vstack([a['q'][a['primary']] for a in fitting])
    metadata={'training_ids':[a['id'] for a in fitting], 'validation_ids':[a['id'] for a in valid],
              'target_fit_rows':0,'waypoint_fit_rows':0,
              'training_bounds_mm':[bounds.min(0).tolist(),bounds.max(0).tolist()],
              'initial_state':'zero relative to an unloaded baseline; unknown prehistory is not inferred',
              'tip':'explicit final_5mm records only; no legacy atlas used in fitting',
              'reference':'command log + EST; source legacy ACK, final test serial-dispatch convention',
              'status':'research empirical surrogate; not finite-element simulation'}
    all_candidates=[]; chosen={}
    for taus in TAUS:
        x,y,w=matrices(fitting,taus)
        for alpha in ALPHAS:
            model=MagneticForward(fit_readout(x,y,w,alpha),taus,{**metadata,'alpha':alpha})
            val=[{'id':a['id'],'score':metric(model.rollout(a['q'],a['t'],a['reset']),a)} for a in valid]
            objective=macro(val,'score')['nrmse']
            c={'taus_s':list(taus),'alpha':alpha,'validation_nrmse_macro':objective,'validation':val}
            all_candidates.append(c)
            key='dynamic' if taus else 'instantaneous'
            if key not in chosen or objective<chosen[key][0]: chosen[key]=(objective,model)
            print('candidate',key,taus,alpha,round(objective,4),flush=True)
    models={k:v[1] for k,v in chosen.items()}
    for k,m in models.items(): m.save(MODEL_DIR/(k+'.json'))
    dump(OUT/'selection.json',{'objective':'source E whole-Trial macro normalized 15-channel RMSE',
                             'candidates':all_candidates,'selected':{k:{'taus':list(v.taus),'alpha':v.metadata['alpha']} for k,v in models.items()},
                             'frozen_model_hashes':{k:sha(MODEL_DIR/(k+'.json')) for k in models},
                             'target_used_for_selection':False,'waypoint_used_for_selection':False})
    return models


def translation_diagnostic(model, a, pred):
    ix=np.flatnonzero(a['primary'])[::5]
    if len(ix)<10: return None
    q=a['q'][ix].astype(float); jac=[]
    for k in range(3):
        d=np.zeros(3);d[k]=.02
        jac.append((model.static(q+d)-model.static(q-d))/.04)
    j=np.stack(jac,axis=2).reshape(-1,3)
    dq=np.linalg.lstsq(j,(a['b'][ix]-pred[ix]).reshape(-1),rcond=None)[0]
    # First-order common-coordinate offset estimate: a diagnostic, not calibration.
    shifted=model.rollout(a['q']+dq,a['t'],a['reset'])
    return {'estimated_delta_xyz_mm':dq.tolist(),'rmse_after_posthoc_shift_uT':metric(shifted,a)['rmse_uT'],
            'large_shift_warning':bool(np.linalg.norm(dq)>2),
            'applied_to_primary':False,'method':'single first-order XYZ offset fitted on this Trial; optimistic diagnostic only'}


def evaluate(rows,models):
    result=[]
    for a in rows:
        if a['role']!='test': continue
        item={k:a[k] for k in ('id','device','mode')}
        saved={'t':a['t'],'q':a['q'],'b':a['b'],'primary':a['primary'],'safe':a['safe'],'shallow':a['shallow']}
        for name,m in models.items():
            p=m.rollout(a['q'],a['t'],a['reset']);saved[name]=p
            item[name]=metric(p,a)
            item[name+'_safe']=metric(p,a,a['safe'])
            item[name+'_shallow']=metric(p,a,a['shallow'])
            item[name+'_stable']=metric(p,a,a['stable'])
            item[name+'_inside_training_box']=metric(p,a,a['primary']&m.support(a['q']))
            item[name+'_outside_training_box']=metric(p,a,a['primary']&~m.support(a['q']))
            if name=='dynamic': item['xyz_offset_diagnostic']=translation_diagnostic(m,a,p)
        result.append(item);np.savez_compressed(OUT/'predictions'/(a['id']+'.npz'),**saved)
        print('test',a['id'],a['mode'],round(item['instantaneous'].get('nrmse',0),3),round(item['dynamic'].get('nrmse',0),3),flush=True)
    aggregates={}
    for device in sorted(set(r['device'] for r in result)):
        subset=[r for r in result if r['device']==device]
        aggregates[device]={k:macro(subset,k) for k in models}
    report={'schema':'MagTouch-R361-ForwardEvaluation-v1','test':result,'macro_by_device':aggregates,
            'primary_mask':'valid 5-sensor acquisition and nominal 0.4 <= z <= 3.0; no inverse acceptance mask',
            'sampling':'10 Hz causal last-available magnetic frames; no B interpolation or output smoothing',
            'no_measured_B_in_rollout':True,'no_test_time_alignment':True,
            'target_zero_shot':True,'complex_waypoints_excluded_from_fit':True,
            'evidence':'retrospective held-out for this new fit; recordings previously inspected for inverse research',
            'physics_multi_contact_status':'interface only; unavailable validated mechanical solver and dual-contact recordings',
            'current_paper_or_ui_changed':False}
    dump(OUT/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','prepare','fit','evaluate'],default='all')
    args=parser.parse_args()
    for p in (OUT,MODEL_DIR,OUT/'cache',OUT/'predictions'): p.mkdir(parents=True,exist_ok=True)
    inv=inventory()
    if args.stage=='prepare':
        for r in inv:
            a=load_row(r);print('loaded',r['id'],r['role'],a['primary'].sum(),flush=True)
        return
    # Training and source validation are loaded before any held-out test arrays.
    rows=[]
    for r in inv:
        if r['role']!='test': rows.append(load_row(r))
    models=train(rows) if args.stage in ('all','fit') else {k:MagneticForward.load(MODEL_DIR/(k+'.json')) for k in ('instantaneous','dynamic')}
    if args.stage=='fit': return
    rows.extend(load_row(r) for r in inv if r['role']=='test')
    evaluate(rows,models)
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__': main()
