"""General bounded XYZ motion data and same-architecture XY research training.

No real waypoint or target observations enter generation, fitting or selection.
Only research XY weights change. The empirical forward model is frozen, not FEM.
"""
from __future__ import annotations
import argparse, json, os, time, shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from training import domain_randomization_r369 as prior
from training.run_forward_r361 import ROOT, dump, sha
from training.forward_complex_r363 import AxisForward
from training.augment_forward_r361 import append_segment
from training.research_formal_r365 import metrics

OUT=ROOT/'analysis/r370_broad_xyz_training_20261001'
GEN=prior.GEN
MANIFEST=prior.MANIFEST
SOURCE_DISTRIBUTION=ROOT/'analysis/r369_source_randomized_augmentation_20261001/source_parameter_distribution.json'
STYLES=('bezier_space','inertial_walk','free_waypoints','multiscale_curve')
SEEDS=(7,17,27)
COUNTS=(100,500,1000)
MASSES=(.25,.5)
prior.OUT=OUT


def protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads(MANIFEST.read_text(encoding='utf8'))
    p={'schema':'MagTouch-BroadXYZ-R370-1','goal':'new broadly varied motion experience, not standard-path patching',
       'architecture':'same 53->64->64->2 causal XY residual MLP, same frozen initial checkpoint and feature scaling',
       'changed':['research XY weights only'],'unchanged':['formal model','static atlas','slow-memory basis','branch rules','action permission','Z heads','UI','hardware','paper'],
       'source_train_ids':[r['id'] for r in rows if r['role']=='train'],
       'source_validation_ids':[r['id'] for r in rows if r['role']=='validation'],
       'real_test_ids':[r['id'] for r in rows if r['role']=='test'],
       'train_geometries':1000,'virtual_validation_geometries':64,'virtual_test_geometries':64,
       'construction_styles':list(STYLES),'randomness':'one independent seed per whole geometry; no real path copying',
       'seeds_by_role':{'train':3701000,'validation':3702000,'test':3703000},
       'xyz_contact_box_mm':[[-5.,-5.,.45],[5.,5.,2.4]],'sampling_hz':10,
       'nominal_motion_speed_mm_s':[.5,2.],'dwell_s':[0.,2.5],'contact_body_max_frames':450,
       'forward_model':'fixed source-fitted empirical XYZ-history-to-15D delta field; not FEM or independently validated mechanics',
       'domain_randomization':False,'noise':'same source-only unloaded sigma; not target residual noise',
       'counts':list(COUNTS),'optimizer_seeds':list(SEEDS),'virtual_dynamic_mass_candidates':list(MASSES),
       'epochs':40,'batches_per_epoch':24,'batch_size':256,'static_anchor_frames_per_batch':51,'learning_rate':.0002,'weight_l2':.001,
       'training_unit':'whole-geometry equal weight, features obtained by causal full-stream replay BEFORE minibatch sampling',
       'selection':'epoch >=20 every5; minimize 0.5*(source real validation XY/original)+0.5*(new virtual validation XY/original)',
       'source_guard':'real macro <=105% original, each original motion-family macro <=115% original; static anchor XY RMS <=max(0.12mm,120% original)',
       'fallback':'if no checkpoint satisfies safeguards, retain initial checkpoint and explicitly record rejection',
       'selection_scope':'shared initial-model features and permissions; full-stack seed7 rerun after freeze',
       'quantity_scope':'same optimizer budget across100/500/1000; not convergence-matched',
       'target_fit_rows':0,'target_selection_rows':0,'actual_complex_fit_rows':0,'actual_complex_selection_rows':0,
       'test_scope':'real recordings previously examined: retrospective fitting holdout, not new blind acquisition',
       'metric_scope':'raw coordinates primary; constant whole-Trial translation diagnostic separate; no smoothing or time alignment',
       'hashes':{'generator':sha(GEN),'initial_xy':sha(prior.ARTIFACT),'manifest':sha(MANIFEST),'source_noise':sha(SOURCE_DISTRIBUTION)}}
    dest=OUT/'protocol.json'
    if dest.exists():
        if json.loads(dest.read_text(encoding='utf8'))!=p:raise ValueError('Immutable protocol changed')
    else:dump(dest,p)
    return p,rows


def resample_curve(curve,rng):
    """Arc-length resampling with bounded speed; each output step <=0.2mm."""
    curve=np.asarray(curve,float)
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
    unique=np.r_[True,np.diff(distance)>1e-9]
    distance,curve=distance[unique],curve[unique]
    if len(curve)<2:return curve
    base=float(rng.uniform(.5,2.));phase=float(rng.uniform(0,2*np.pi));s=0.;samples=[]
    while s<distance[-1]:
        speed=np.clip(base*(.75+.3*np.sin(.35*len(samples)+phase)),.5,2.)
        s=min(s+.1*speed,distance[-1]);samples.append([np.interp(s,distance,curve[:,d]) for d in range(3)])
    return np.asarray(samples)


def geometry(role,index):
    seed={'train':3701000,'validation':3702000,'test':3703000}[role]+index
    rng=np.random.default_rng(seed);style=STYLES[index%4]
    def point():return rng.uniform([-4.9,-4.9,.5],[4.9,4.9,2.35])
    first=point();points=[[*first[:2],-1.]]*12
    append_segment(points,first,float(rng.uniform(.5,2.)),rng.uniform(0,1.))
    body_begin=len(points)-1
    if style=='bezier_space':
        current=first.copy()
        for k in range(int(rng.integers(4,8))):
            control=np.array([current,point(),point(),point()]);u=np.linspace(0,1,160)[:,None]
            curve=(1-u)**3*control[0]+3*(1-u)**2*u*control[1]+3*(1-u)*u**2*control[2]+u**3*control[3]
            part=resample_curve(curve,rng);points.extend(part.tolist());current=control[-1]
            if rng.random()<.35:points.extend([points[-1]]*int(rng.integers(2,26)))
            if rng.random()<.25:points.extend(resample_curve(curve[::-1],rng).tolist());current=control[0]
    elif style=='inertial_walk':
        current=first.copy();direction=rng.normal(size=3);direction[2]*=.35
        pause=0
        for k in range(450):
            if pause:points.append(current.tolist());pause-=1;continue
            if rng.random()<.018:pause=int(rng.integers(2,26))
            direction=.88*direction+.3*rng.normal(size=3)*[1,1,.35]
            if rng.random()<.015:direction=-direction
            step=.1*rng.uniform(.5,2.)*direction/max(np.linalg.norm(direction),1e-8)
            proposed=current+step
            lo=np.array([-4.95,-4.95,.5]);hi=np.array([4.95,4.95,2.35])
            hit=(proposed<lo)|(proposed>hi);direction[hit]*=-1
            current=np.clip(proposed,lo,hi);points.append(current.tolist())
    elif style=='free_waypoints':
        visits=[first.copy()]
        for k in range(int(rng.integers(6,11))):
            target=visits[int(rng.integers(len(visits)))] if k>1 and rng.random()<.3 else point()
            append_segment(points,target,float(rng.uniform(.5,2.)),float(rng.choice([0.,0.,.3,1.,2.5])))
            visits.append(target.copy())
    else:
        u=np.linspace(0,rng.uniform(2.5,5.)*np.pi,700)
        harmonics=rng.uniform(.45,3.2,(3,3));phases=rng.uniform(0,2*np.pi,(3,3))
        weights=rng.uniform(.2,1.,(3,3));waves=np.array([sum(w*np.sin(f*u+p) for w,f,p in zip(weights[d],harmonics[d],phases[d])) for d in range(3)]).T
        waves/=np.maximum(np.max(np.abs(waves),axis=0),1e-8)
        center=np.r_[rng.uniform(-.6,.6,2),rng.uniform(1.15,1.65)]
        scale=np.r_[rng.uniform(1.5,4.3,2),rng.uniform(.25,.65)]
        curve=center+waves*scale
        append_segment(points,curve[0],rng.uniform(.5,2.),rng.uniform(0,.8))
        points.extend(resample_curve(curve,rng).tolist())
    # A truncated stochastic path remains a complete Trial: final hold/withdraw
    # follow the last actual point, never splice in a future target.
    points=points[:body_begin+451]
    points.extend([points[-1]]*int(rng.integers(0,21)))
    body_end=len(points)-1
    append_segment(points,[*points[-1][:2],-1.],1.,2.5)
    q=np.asarray(points,float);t=np.arange(len(q))*.1
    contact=q[:,2]>=.45
    delta=np.diff(q,axis=0);speed=np.linalg.norm(delta,axis=1)/.1
    info={'seed':seed,'style':style,'body_begin':body_begin,'body_end':body_end,'frames':len(t),
          'duration_s':float(t[-1]),'contact_xyz_min_mm':q[contact].min(0).tolist(),'contact_xyz_max_mm':q[contact].max(0).tolist(),
          'max_speed_mm_s':float(speed.max()),'xy_extent_mm':np.ptp(q[contact,:2],axis=0).tolist(),'z_extent_mm':float(np.ptp(q[contact,2])),
          'role':role,'geometry_id':f'{role}_{index:04d}'}
    return q,t,info


def generated_job(job):
    role,index=job;ident=f'V370_{role}_{index:04d}'
    dest=OUT/'virtual'/f'{ident}.npz'
    q,t,info=geometry(role,index)
    if not dest.exists():
        g=AxisForward.load(GEN);d=json.loads(SOURCE_DISTRIBUTION.read_text(encoding='utf8'))
        b=g.rollout(q,t)+np.random.default_rng(info['seed']+400000).normal(size=(len(q),15))*np.array(d['sigma_uT'])
        a={'id':ident,'q':q,'t':t,'b':b,'reset':np.r_[True,np.zeros(len(t)-1,bool)],'valid':np.ones(len(t),bool),
           'primary':(q[:,2]>=.4)&(q[:,2]<=3)&g.support(q),'safe':q[:,2]<=-.5}
        np.savez_compressed(dest,**a)
    with np.load(dest,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    a['id']=str(a['id']);features=prior.cache_features(a)
    return {'id':ident,**info,'sha256':sha(dest),'eligible_training_frames':int(features['train_mask'].sum()),
            'reference_contact_frames':int(a['primary'].sum()),'original_coverage':float(features['train_mask'].sum()/max(a['primary'].sum(),1))}


def prepare(workers=1):
    p,rows=protocol()
    for sub in ('virtual','features','models','predictions'):(OUT/sub).mkdir(exist_ok=True)
    for r in rows:
        if r['role'] not in ('train','validation'):continue
        dest=OUT/'features'/f"{r['id']}.npz"
        old=ROOT/'analysis/r369_source_randomized_augmentation_20261001/features'/dest.name
        if not dest.exists() and old.exists():shutil.copy2(old,dest)
        if not dest.exists():prior.cache_features(prior.measured(r))
    jobs=[(role,i) for role,n in [('train',p['train_geometries']),('validation',p['virtual_validation_geometries']),('test',p['virtual_test_geometries'])] for i in range(n)]
    results=[]
    pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    iterator=pool.map(generated_job,jobs) if pool else map(generated_job,jobs)
    for r in iterator:
        results.append(r)
        if len(results)%20==0:print('broad XYZ prepared',len(results),'/',len(jobs),flush=True)
    if pool:pool.shutdown()
    dump(OUT/'catalog.json',{'trials':results,'source_only':True,'generator_sha256':sha(GEN),'not_independent_physical_devices':True})


def grouped_validation(model,items):
    values=[]
    for a in items:
        m=a['train_mask']
        if not m.any():raise ValueError('validation Trial has no supported output')
        pred=a['guard_xy'][m]+model.predict(a['x'][m])
        values.append(float(np.linalg.norm(pred-a['q'][m,:2],axis=1).mean()))
    return float(np.mean(values)),values


def train_job(job):
    seed,count,mass=job;p,rows=protocol();rng=np.random.default_rng(seed)
    real=[prior.load_features(r['id']) for r in rows if r['role']=='train']
    real_info=[r for r in rows if r['role']=='validation']
    valid=[prior.load_features(r['id']) for r in real_info]
    catalog=json.loads((OUT/'catalog.json').read_text(encoding='utf8'))['trials']
    vvalid=[prior.load_features(r['id']) for r in catalog if r['role']=='validation']
    virtual=[prior.load_features(r['id']) for r in catalog if r['role']=='train'][:count]
    model=prior.ResearchXY();base=prior.ResearchXY();rx,ry,rw,_=prior.sampled_training(real)
    vx,vy,vw,_=prior.sampled_training(virtual) if virtual else (rx,ry,rw,None)
    ax,ay=prior.anchors();rprob=rw/rw.sum();vprob=vw/vw.sum()
    real0,rv0=grouped_validation(base,valid);virtual0,_=grouped_validation(base,vvalid)
    static0=float(np.sqrt(np.mean(np.sum(base.predict(ax)**2,axis=1))))
    modes=sorted({r['mode'] for r in real_info});ix={m:[i for i,r in enumerate(real_info) if r['mode']==m] for m in modes}
    family0={m:float(np.mean(np.array(rv0)[ii])) for m,ii in ix.items()}
    params=[getattr(model,k) for k in prior.KEYS[2:]];first=[np.zeros_like(a) for a in params];second=[np.zeros_like(a) for a in params]
    initial=[a.copy() for a in params];chosen=initial;best=np.inf;selected_epoch=0;step=0;history=[]
    for epoch in range(1,p['epochs']+1):
        for _ in range(p['batches_per_epoch']):
            na=p['static_anchor_frames_per_batch'];nv=round((p['batch_size']-na)*mass) if count else 0;nr=p['batch_size']-na-nv
            ri=rng.choice(len(rx),nr,p=rprob);vi=rng.choice(len(vx),nv,p=vprob);ai=rng.integers(len(ax),size=na)
            xx=np.vstack((rx[ri],vx[vi],ax[ai]));yy=np.vstack((ry[ri],vy[vi],ay[ai]));xx=(xx-model.mean)/model.scale
            w1,b1,w2,b2,w3,b3=params;h1=np.tanh(xx@w1+b1);h2=np.tanh(h1@w2+b2);out=h2@w3+b3
            d3=2*(out-yy)/len(xx);g3=h2.T@d3+.001*w3;gb3=d3.sum(0)
            d2=(d3@w3.T)*(1-h2*h2);g2=h1.T@d2+.001*w2;gb2=d2.sum(0)
            d1=(d2@w2.T)*(1-h1*h1);g1=xx.T@d1+.001*w1;gb1=d1.sum(0);step+=1
            for a,grad,m,v in zip(params,(g1,gb1,g2,gb2,g3,gb3),first,second):
                m*=.9;m+=.1*grad;v*=.999;v+=.001*grad*grad
                a-=.0002*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if epoch%5:continue
        re,rv=grouped_validation(model,valid);ve,_=grouped_validation(model,vvalid)
        sr=float(np.sqrt(np.mean(np.sum(model.predict(ax)**2,axis=1))))
        family={m:float(np.mean(np.array(rv)[ii])) for m,ii in ix.items()}
        feasible=re<=1.05*real0 and all(family[m]<=1.15*family0[m] for m in modes) and sr<=max(.12,1.2*static0)
        objective=.5*(re/real0)+.5*(ve/virtual0)
        record={'epoch':epoch,'source_real_xy_mm':re,'new_virtual_xy_mm':ve,'normalized_objective':objective,'static_anchor_rms_mm':sr,'real_by_family':family,'eligible':bool(feasible)}
        history.append(record)
        if epoch>=20 and feasible and objective<best:best=objective;selected_epoch=epoch;chosen=[a.copy() for a in params]
    for k,v in zip(prior.KEYS[2:],chosen):setattr(model,k,v)
    name=f'broad_{count}_seed{seed}_mass{mass:.2f}' if count else f'real_control_seed{seed}'
    path=OUT/'models'/f'{name}.npz';model.save(path)
    re,rv=grouped_validation(model,valid);ve,_=grouped_validation(model,vvalid)
    result={'name':name,'arm':f'broad_{count}' if count else 'real_control','count':count,'mass':mass,'seed':seed,'epoch':selected_epoch,
            'eligible_checkpoint_found':bool(selected_epoch),'selection_objective':best if np.isfinite(best) else 1.,
            'source_validation_xy_mm':re,'new_virtual_validation_xy_mm':ve,'original_source_validation_xy_mm':real0,'original_virtual_validation_xy_mm':virtual0,
            'file':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'history':history,'optimizer_updates':step,
            'dynamic_training_samples':len(rx),'virtual_training_samples':len(vx) if count else 0}
    print('trained broad',name,'epoch',selected_epoch,'real',round(re,4),'newvirtual',round(ve,4),flush=True)
    return result


def fit(workers=1):
    p,_=protocol()
    if (OUT/'freeze.json').exists() or (OUT/'report.json').exists():raise FileExistsError('No refit after freeze/test; use a new study')
    jobs=[(s,0,0.) for s in SEEDS]+[(s,n,m) for s in SEEDS for n in COUNTS for m in MASSES]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:choices=list(pool.map(train_job,jobs))
    else:choices=list(map(train_job,jobs))
    selected=[]
    for s in SEEDS:
        for n in (0,)+COUNTS:
            options=[r for r in choices if r['seed']==s and r['count']==n]
            selected.append(min(options,key=lambda r:r['selection_objective']))
    dump(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'catalog_sha256':sha(OUT/'catalog.json'),
         'selected':selected,'choices':choices,'target_fit_rows':0,'target_selection_rows':0,'complex_fit_rows':0,'complex_selection_rows':0,
         'frozen_before_real_tests':True,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def virtual_test():
    f=json.loads((OUT/'freeze.json').read_text(encoding='utf8'));catalog=json.loads((OUT/'catalog.json').read_text(encoding='utf8'))['trials']
    selected=f['selected'];results=[]
    for r in catalog:
        if r['role']!='test':continue
        a=prior.load_features(r['id']);row={'id':r['id'],'style':r['style'],'models':{'original':prior.direct_score(prior.ResearchXY(),a)}}
        for choice in selected:row['models'][choice['name']]=prior.direct_score(prior.ResearchXY.load(ROOT/choice['file']),a)
        results.append(row)
    dump(OUT/'virtual_test.json',{'freeze_sha256':sha(OUT/'freeze.json'),'fixed_features':True,'rows':results,'not_real_validation':True})


def real_trial(rec):
    f=json.loads((OUT/'freeze.json').read_text(encoding='utf8'));a=prior.measured(rec);features=prior.cache_features(a)
    row={k:rec[k] for k in ('id','device','mode')};row['models']={'original':prior.direct_score(prior.ResearchXY(),features)}
    for choice in f['selected']:row['models'][choice['name']]=prior.direct_score(prior.ResearchXY.load(ROOT/choice['file']),features)
    full=[]
    for choice in [None]+[r for r in f['selected'] if r['seed']==7]:
        arm='original' if choice is None else choice['arm'];dest=OUT/'predictions'/f"{rec['id']}_{arm}.npz"
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:pred,accepted=z['pred'],z['accepted']
        else:
            output={'pred':features['old_pred'],'accepted':features['accepted']} if choice is None else prior.replay(a,prior.ResearchXY.load(ROOT/choice['file']),False)
            pred,accepted=output['pred'],output['accepted']
            np.savez_compressed(dest,pred=pred,accepted=accepted,t=a['t'],q=a['q'],primary=a['primary'],safe=a['safe'])
        common=accepted&features['accepted'];safe=a['safe']&(a['q'][:,2]<=-.5)
        full.append({**{k:rec[k] for k in ('id','device','mode')},'arm':arm,'own_mask':metrics(pred,a['q'],a['t'],a['primary'],accepted),
                     'common_mask':metrics(pred,a['q'],a['t'],a['primary'],common),'original_on_common':metrics(features['old_pred'],a['q'],a['t'],a['primary'],common),
                     'safe_reference_frames':int(safe.sum()),'safe_output_frames':int((safe&accepted).sum()),
                     'permission_changes':int(np.count_nonzero(accepted!=features['accepted']))})
    print('broad real test',rec['id'],flush=True);return row,full


def evaluate(workers=1):
    p,rows=protocol();f=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    assert f['protocol_sha256']==sha(OUT/'protocol.json') and f['catalog_sha256']==sha(OUT/'catalog.json')
    for r in f['selected']:assert sha(ROOT/r['file'])==r['sha256']
    fixed=[];full=[];tests=[r for r in rows if r['role']=='test']
    pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    for row,fr in (pool.map(real_trial,tests) if pool else map(real_trial,tests)):
        fixed.append(row);full.extend(fr);dump(OUT/'partial_real_test.json',{'rows':fixed,'full_stack_seed7':full})
    if pool:pool.shutdown()
    aggregate={}
    arms=('original','real_control')+tuple(f'broad_{n}' for n in COUNTS)
    for device in ('source','target'):
        aggregate[device]={}
        for arm in arms:
            rr=[r['own_mask'] for r in full if r['device']==device and r['arm']==arm]
            aggregate[device][arm]={k:float(np.mean([r[k] for r in rr if k in r])) for k in ('xy_mean_mm','z_mae_mm','xyz_mean_mm','coverage','xy_translation_removed_mm','z_translation_removed_mm')}
    dump(OUT/'report.json',{'freeze_sha256':sha(OUT/'freeze.json'),'fixed_features':fixed,'full_stack_seed7':full,'full_stack_macro':aggregate,
         'formal_model_replaced':False,'target_zero_shot_fit_selection':True,'changed':'research XY weights only','warning':'retrospective real test; empirical generator, not FEM; synthetic test uses same generator'})
    print(json.dumps(aggregate),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['protocol','prepare','fit','virtual-test','evaluate']);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args()
    if a.stage=='protocol':protocol()
    elif a.stage=='prepare':prepare(a.workers)
    elif a.stage=='fit':fit(a.workers)
    elif a.stage=='virtual-test':virtual_test()
    else:evaluate(a.workers)
if __name__=='__main__':main()
