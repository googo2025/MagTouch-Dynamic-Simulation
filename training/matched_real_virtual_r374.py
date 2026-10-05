"""Matched real/virtual fields, frozen inverse, new saved-online R373 audit.

Only B/time/explicit baseline resets/validity enter inference. Command XYZ is
the generator input and evaluation reference, not inverse input or encoder truth.
Raw files and all deployed weights remain read-only. Run with python -m.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from training.run_forward_r361 import ROOT, sha, dump, metric
from training.build_trajectory_dataset_v1 import assemble, load_magnetic, read_rows, reference_origin, command_segments, command_reference
from training.analyze_r354_final_online import mask_for, score, steps
from training.retrain_r133_coverage_xy import recorded_bool_mask
from training.forward_complex_r363 import AxisForward
from training.research_formal_r365 import new_engine, metrics, macro
sys.path.insert(0,str(ROOT/'code'))
from live_retrained_reconstruction_r373 import DirectTrainedEngineR373, COMPONENTS
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from live_algorithm_ui_r126 import PROFILE_KEY
from replay_backtest import load_profile

OUT=ROOT/'analysis/r374_matched_real_virtual_20261002'
GEN=ROOT/'analysis/r363_complex_forward_20261001/models/generator.json'

def finite_quantiles(v):
    v=np.asarray(v,float); v=v[np.isfinite(v)]
    return {'n':len(v),'p05':float(np.quantile(v,.05)),'median':float(np.median(v)),'p95':float(np.quantile(v,.95))} if len(v) else {'n':0}

def inventory():
    rows=[]
    for plan in sorted((ROOT/'data/S20261001_A').glob('Plan_*R373_*')):
        prog=json.loads((plan/'plan_progress.json').read_text(encoding='utf8'))
        if prog['status']!='COMPLETED' or prog['completed_trials']!=9:raise ValueError('Plan incomplete')
        for path in sorted(plan.glob('Trial_*')):
            m=json.loads((path/'meta.json').read_text(encoding='utf8'));alg=m['live_prediction_recording']['algorithm_metadata']
            assert m['live_prediction_recording']['model_family']=='DirectPlateauReconstructionR373'
            assert alg['xy_mlp_artifact_sha256']==COMPONENTS['xy']['model_sha256']
            assert alg['depth_increment_artifact_sha256']==COMPONENTS['z']['model_sha256']
            assert not alg['target_fit_rows'] and not alg['cross_device_output_registration']
            d='target' if 'TARGET' in plan.name else 'source'; tp=m['trial_parameters'];tid=str(tp['trial_id'])
            rows.append({'id':f'{d}_{tid}','path':path.relative_to(ROOT).as_posix(),'device':d,'mode':tp['command_mode'],
                'hashes':{f:sha(path/f) for f in ['meta.json','magnetic_raw.csv','stage_commands.csv','events.csv','live_predictions.csv']}})
    assert len(rows)==18
    dest=OUT/'manifest.json'
    if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==rows
    else:dump(dest,rows)
    return rows

def saved_online(row):
    path=ROOT/row['path'];m=json.loads((path/'meta.json').read_text(encoding='utf8'))
    df=pd.read_csv(path/'live_predictions.csv',low_memory=False)
    ex=[json.loads(x) for x in df.extra_outputs_json]
    tn=df.t_prediction_ns.to_numpy(np.int64);assert (np.diff(tn)>0).all()
    xyz=df[['visual_x_mm','visual_y_mm','visual_z_mm']].to_numpy(float)
    valid=recorded_bool_mask(df.interaction_valid)&np.isfinite(xyz).all(1)
    assert np.array_equal(valid,np.array([bool(x.get('xyz_valid')) for x in ex]))
    seg=command_segments(read_rows(path/'stage_commands.csv'));ref=command_reference(tn,seg,reference_origin(m))
    phase=np.array([seg[j]['phase'] if j>=0 else 'before_stage' for j in ref['command_segment_id']])
    primary,action=mask_for(row['mode'],phase,ref)
    safe=np.isfinite(ref['xyz_mm'][:,2])&(ref['xyz_mm'][:,2]<=-.2)
    if row['mode']=='none':safe[:]=True
    hint=np.array([bool(x.get('external_no_contact_hint')) for x in ex])
    a={'time':tn,'xyz':xyz,'valid':valid,'q':ref['xyz_mm'],'primary':primary,'action':action,'safe':safe,'phase':phase}
    r={'frames':len(tn),'primary':score(a,primary),'whole_action':score(a,action&(a['q'][:,2]>=.4)),
       'safe_frames':int(safe.sum()),'safe_outputs':int((safe&valid).sum()),
       'safe_without_hint_frames':int((safe&~hint).sum()),'safe_without_hint_outputs':int((safe&~hint&valid).sum()),
       'queue_drops':m['live_prediction_recording'].get('input_groups_dropped_by_live_queue'),
       'prediction_dt_s':finite_quantiles(np.diff(tn)/1e9)}
    permission=np.array([x.get('localization_permission_score') if x.get('localization_permission_score') is not None else np.nan for x in ex])
    ev=[x.get('final_depth_evidence',{}) for x in ex]
    fields=sorted(set(k for e in ev for k,v in e.items() if isinstance(v,(float,int)) and not isinstance(v,bool)))
    r['depth_evidence_fields']=fields
    r['permission_score']={k:finite_quantiles(permission[v]) for k,v in [('primary',primary),('shallow',action&(a['q'][:,2]>0)&(a['q'][:,2]<.4)),('safe',safe)]}
    r['depth_evidence_quantiles']={k:finite_quantiles([e.get(k,np.nan) for e in ev]) for k in fields}
    r['routes']=dict(pd.Series([x.get('localization_path','unknown') for x in ex]).value_counts().items())
    r['shallow_nominal_bins']={}
    for lo,hi in [(0,.2),(.2,.4),(.4,.6),(.6,1.),(1.,2.)]:
        mask=action&(a['q'][:,2]>=lo)&(a['q'][:,2]<hi)
        r['shallow_nominal_bins'][f'{lo}-{hi}']=score(a,mask)
    if row['mode'] in ('angle_scan','circle_scan'):
        const=action&np.isclose(a['q'][:,2],float(m['trial_parameters']['z_indent_mm']),atol=.01)
        r['constant_depth_steps']=steps(a,const,True)
    # Worst accepted primary frames retain timestamp, reference, output and actual
    # evidence values. Rejected frames are separately counted, never zero error.
    take=np.flatnonzero(primary&valid)
    if len(take):take=take[np.argsort(np.linalg.norm(xyz[take]-a['q'][take],axis=1))[-12:][::-1]]
    r['worst_accepted_frames']=[{'time_ns':int(tn[i]),'q_command_mm':a['q'][i].tolist(),'xyz_mm':xyz[i].tolist(),
        'permission_score':float(permission[i]) if np.isfinite(permission[i]) else None,'depth':ev[i],'route':ex[i].get('localization_path')} for i in take]
    np.savez_compressed(OUT/'streams'/f"{row['id']}_online.npz",**a)
    return r

def load_controlled(row):
    path=ROOT/row['path'];m=json.loads((path/'meta.json').read_text(encoding='utf8'))
    seg=command_segments(read_rows(path/'stage_commands.csv'))
    full,_=assemble(load_magnetic(path/'magnetic_raw.csv'),seg,read_rows(path/'events.csv'),reference_origin(m))
    ix=np.arange(0,len(full['t_ns']),4)
    a={k:full[v][ix] for k,v in {'q':'xyz_mm','b':'delta_b_uT','valid':'input_valid','b0':'b0_id'}.items()}
    a['t']=(full['t_ns'][ix]-full['t_ns'][0])/1e9;a['reset']=np.r_[True,np.diff(a['b0'])!=0]
    phase=np.array([seg[j]['phase'] if j>=0 else 'before_stage' for j in full['command_segment_id'][ix]])
    ref={k:full[k][ix] for k in ('xyz_mm','command_moving','hold_age_s')}
    a['primary'],a['action']=mask_for(row['mode'],phase,ref)
    a['primary']&=a['valid'];a['safe']=a['valid']&np.isfinite(a['q']).all(1)&(a['q'][:,2]<=-.2)
    a['matched_eligible']=a['valid']&np.isfinite(a['q']).all(1)
    return a

def replay(a,b,old=False):
    if old:e=new_engine()
    else:
        p=load_profile(R92_PROFILE_PATH,PROFILE_KEY)
        e=DirectTrainedEngineR373(p,p['defaults']['threshold'],p['defaults']['min_cosine'],p['defaults']['smoothing_ms'],70.)
        e.set_baseline(np.zeros(15))
    pred=np.full((len(b),3),np.nan);ok=np.zeros(len(b),bool);perm=np.full(len(b),np.nan);support=np.full(len(b),np.nan)
    for i,now in enumerate(a['t']):
        if a['reset'][i]:e.set_baseline(np.zeros(15))
        eligible=bool(a['matched_eligible'][i]);r=e.push(int(round(now*1e9))+10_000_000_000,b[i] if eligible else np.zeros(15),eligible=eligible)
        ok[i]=bool(r['valid'])
        if ok[i]:pred[i]=r['xyz']
        ps=r['extra'].get('localization_permission_score');perm[i]=ps if ps is not None else np.nan
        support[i]=r['extra'].get('final_depth_evidence',{}).get('source_support_distance',np.nan)
    return pred,ok,perm,support

def trial(row):
    start=time.perf_counter()
    for f,h in row['hashes'].items():assert sha(ROOT/row['path']/f)==h
    online=saved_online(row);a=load_controlled(row);g=AxisForward.load(GEN)
    vb=g.rollout(a['q'],a['t'],a['reset'])
    r={**row,'online':online,'forward_field':metric(vb,a),'replay':{}}
    stream={k:a[k] for k in ('t','q','b','primary','action','safe','reset','valid')};stream['virtual_b']=vb
    results={}
    for name,b,old in [('real_old',a['b'],True),('real_new',a['b'],False),('virtual_new',vb,False)]:
        pred,ok,perm,support=replay(a,b,old); results[name]=(pred,ok)
        stream[name]=pred;stream[name+'_ok']=ok;stream[name+'_permission']=perm;stream[name+'_support']=support
        r['replay'][name]=metrics(pred,a['q'],a['t'],a['primary'],ok)
        r['replay'][name]['safe_outputs']=int((a['safe']&ok).sum())
    common=results['real_old'][1]&results['real_new'][1]
    r['replay']['real_old_common']=metrics(results['real_old'][0],a['q'],a['t'],a['primary'],common)
    r['replay']['real_new_common']=metrics(results['real_new'][0],a['q'],a['t'],a['primary'],common)
    rv=results['real_new'][1]&results['virtual_new'][1]&a['primary']
    r['matched_output_comparison']=metrics(results['virtual_new'][0],results['real_new'][0],a['t'],rv,rv)
    r['old_new_acceptance_changes']={'lost':int((a['primary']&results['real_old'][1]&~results['real_new'][1]).sum()),'added':int((a['primary']&~results['real_old'][1]&results['real_new'][1]).sum())}
    r['seconds']=time.perf_counter()-start
    np.savez_compressed(OUT/'streams'/f"{row['id']}_matched.npz",**stream)
    dump(OUT/'records'/f"{row['id']}.json",r)
    print(row['id'],row['mode'],'online',online['primary'].get('xy_raw_mean_mm'),'matched XYZ',r['replay']['real_new'].get('xyz_mean_mm'),r['replay']['virtual_new'].get('xyz_mean_mm'),'sec',round(r['seconds'],1),flush=True)
    return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    for d in ('streams','records'): (OUT/d).mkdir(parents=True,exist_ok=True)
    rows=inventory();dump(OUT/'protocol.json',{'generator_sha256':sha(GEN),'inverse':COMPONENTS,
        'controlled_replay_hz':10,'saved_online':'original native saved outputs, separate from controlled replay',
        'generator_fit':'pre-existing source-only R363; no latest recordings or Yellow used to refit',
        'virtual_noise':'none added, deterministic forward response; not simulated measurement-noise validation',
        'reference':'serial-dispatch + logged EST duration, not encoder ground truth',
        'primary':'predefined action masks and nominal depth >= 0.4 mm, not physical contact recall',
        'alignment':'no coordinate or time registration, no smoothing; per-Trial translation removal is diagnostic only',
        'target_selection':False,'downstream_parameters_changed':False,'raw_data_modified':False})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:result=list(pool.map(trial,rows))
    groups={}
    for d in ('source','target'):
        rr=[r for r in result if r['device']==d];active=[r for r in rr if r['mode']!='none']
        groups[d]={'trials':len(rr),'replay_macro':macro([dict(r['replay']) for r in active],['real_old','real_new','virtual_new','real_old_common','real_new_common']),
           'online_safe_frames':sum(r['online']['safe_frames'] for r in rr),'online_safe_outputs':sum(r['online']['safe_outputs'] for r in rr),
           'online_safe_without_hint_frames':sum(r['online']['safe_without_hint_frames'] for r in rr),'online_safe_without_hint_outputs':sum(r['online']['safe_without_hint_outputs'] for r in rr),
           'forward_macro_nrmse':float(np.mean([r['forward_field']['nrmse'] for r in active if r['forward_field']['frames']]))}
    dump(OUT/'report.json',{'status':'COMPLETE','groups':groups,'records':result})
    print(json.dumps(groups,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
