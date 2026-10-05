"""Add existing vertical histories to the known cone-domain forward model.

Static calibration stays fixed. Source-only source-day grouped validation
selects ridge strength, then unchanged next-day trajectories evaluate it.
No geometry shape is inferred from a legacy label and no target data are fit.
Normal-equation statistics are accumulated per record to bound memory use.
"""
from pathlib import Path
import json,time,os
import numpy as np
from training.two_tip_forward_r376 import manifest as previous_manifest,load,PROFILE,TAUS
from training.magnetic_forward_r361 import MagneticForward
from training.forward_complex_r363 import AxisForward,axis_history
from training.run_forward_r361 import ROOT,metric,dump,sha
from training.build_trajectory_dataset_v1 import reference_origin
OUT=ROOT/'analysis/r378_cone_vertical_history_20261002'
OLD=ROOT/'analysis/r376_two_tip_forward_20261002'

def manifest():
    rows=previous_manifest()
    paths=[]
    for n in (1,4,9):
        p=list((ROOT/'data/S20260924_A').glob(f'Trial_{n:06d}_*'));assert len(p)==1
        paths.append((p[0],'train'))
    plans=list((ROOT/'data/S20260924_A').glob('Plan_20260924_184901_*'));assert len(plans)==1
    for n in (29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46):
        p=list(plans[0].glob(f'Trial_{n:06d}_*'));assert len(p)==1
        paths.append((p[0],'validation' if n in (31,37,43) else 'train'))
    for p,role in paths:
        m=json.loads((p/'meta.json').read_text(encoding='utf8'));assert m['status']=='completed'
        assert m['acquisition']['magnetic_layer_profile_id']==PROFILE
        assert .4<=m['trial_parameters']['z_indent_mm']<=1.6
        rows.append({'id':p.parent.name+'_'+p.name.split('_')[1],'path':p.relative_to(ROOT).as_posix(),
            'role':role,'mode':'grid_points','profile':PROFILE,'contact_z_mm':float(reference_origin(m)[2]),
            'geometry':'known old1mm/cone profile; exact cone mesh unavailable',
            'hashes':{f:sha(p/f) for f in ('meta.json','magnetic_raw.csv','stage_commands.csv','events.csv')}})
    assert len({r['path'] for r in rows})==len(rows)
    return rows

def cached(row):
    p=OUT/'cache'/f"{row['id']}.npz"
    if p.exists():
        with np.load(p,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        return dict(a,**row)
    for f,h in row['hashes'].items():assert sha(ROOT/row['path']/f)==h
    a=load(row);a['fit']&=(np.abs(a['q'][:,:2])<=8+1e-5).all(1)
    np.savez_compressed(p,**{k:a[k] for k in ('q','b','t','reset','primary','fit')})
    return a

def grouped(model,rows):
    result=[]
    for row in rows:
        a=cached(row);v=metric(model.rollout(a['q'],a['t'],a['reset']),a)
        result.append({'id':row['id'],'mode':row['mode'],**v})
    families={m:float(np.mean([r['nrmse'] for r in result if r['mode']==m])) for m in sorted({r['mode'] for r in result})}
    return {'family_macro_nrmse':float(np.mean(list(families.values()))),'by_family':families,'trials':result}

def main():
    (OUT/'cache').mkdir(parents=True,exist_ok=True)
    dump(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'phase':'manifest_and_load'})
    rows=manifest();dump(OUT/'manifest.json',rows)
    # R376 static coefficients stay byte-for-byte fixed in the new generator.
    old=AxisForward.load(OLD/'generator.json');static=old.base
    train=[r for r in rows if r['role']=='train'];val=[r for r in rows if r['role']=='validation'];test=[r for r in rows if r['role']=='test']
    assert {r['id'] for r in test}=={r['id'] for r in previous_manifest() if r['role']=='test'}
    g=np.zeros((660,660));rhs=np.zeros((660,15));counts={}
    for row in train:
        a=cached(row);take=a['fit'];x=axis_history(a['q'],a['t'],TAUS,a['reset'])[take]
        y=(a['b']-static.rollout(a['q'],a['t'],a['reset']))[take]
        n=len(x);assert n>0
        g+=x.T@x/n;rhs+=x.T@y/n;counts[row['id']]=n
        print('sufficient_stats',row['id'],n,flush=True)
    g/=len(train);rhs/=len(train);scale=np.maximum(np.sqrt(np.diag(g)),.02)
    lhs=g/(scale[:,None]*scale[None,:]);right=rhs/scale[:,None]
    dump(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'phase':'source_validation'})
    candidates=[{'id':'static','model':static,'alpha':None},{'id':'previous','model':old,'alpha':None}]
    for alpha in (.001,.01,.1,1.):
        coef=np.linalg.solve(lhs+alpha*np.eye(660),right)/scale[:,None]
        candidates.append({'id':str(alpha),'model':AxisForward(static,coef,TAUS,True,dict(old.metadata,alpha=alpha,vertical_histories_added=True)),'alpha':alpha})
    validation=[]
    for c in candidates:
        r=grouped(c['model'],val);validation.append({'id':c['id'],'alpha':c['alpha'],**r});print('validation',c['id'],r['family_macro_nrmse'],flush=True)
    pick=min(validation,key=lambda r:r['family_macro_nrmse']);model=next(c['model'] for c in candidates if c['id']==pick['id'])
    model.save(OUT/'generator.json')
    dump(OUT/'freeze.json',{'selected':pick['id'],'selection':'source-day equal-family whole-record validation NRMSE',
        'validation':validation,'generator_sha256':sha(OUT/'generator.json'),'fit_method':'exact weighted ridge normal equations accumulated per Trial',
        'train_ids':[r['id'] for r in train],'validation_ids':[r['id'] for r in val],'fit_samples_by_trial':counts,
        'heldout_test_fit_or_selection':False,'target_fit_rows':0,'static_coefficients_unchanged':True,
        'static_calibration_exposure':'source-day static calibration may contain positions from grid validation; only new dynamic coefficient fitting excludes these complete histories'})
    # Open and score next-day tests only after writing the model freeze.
    result={'status':'COMPLETE','selected':pick['id'],'test':{'static':grouped(static,test),'previous':grouped(old,test),'updated':grouped(model,test)},
        'scope':'empirical known-geometry forward fit, not unseen-tip prediction or validated FEM','production_changed':False}
    dump(OUT/'report.json',result);dump(OUT/'status.json',{'status':'COMPLETE','phase':'heldout_evaluated','pid':os.getpid()})
    print({k:v['family_macro_nrmse'] for k,v in result['test'].items()},flush=True)
if __name__=='__main__':main()
