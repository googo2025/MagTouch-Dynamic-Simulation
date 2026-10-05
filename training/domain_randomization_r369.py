"""Source-only virtual augmentation of the ACTUAL frozen causal XY network.

Research copies only. The static map, slow-memory basis, branch rules, action
heads and both Z networks are frozen. References enter training/scoring, never
an online decoder. Target outcome loading is guarded by a source-only freeze.
"""
from __future__ import annotations
import argparse, copy, csv, json, time, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
# Historical immutable freeze JSONs contain Windows separators. Preserve their
# bytes/hashes and normalize only filesystem resolution in the Linux worker.
if os.name!='nt':
    from pathlib import PosixPath
    PosixPath.__fspath__=lambda self:str(self).replace('\\','/')
from training.run_forward_r361 import ROOT, load_row, dump, sha
from training.forward_complex_r363 import AxisForward
from training.augment_forward_r361 import append_segment
from training.research_formal_r365 import new_engine, metrics
from training.had239_r133_coverage_xy_runtime import CoverageCausalXYMLP, ARTIFACT
from training.probe_had239_local_forward_r104 import CONFIG, PROFILE_KEY, load_profile

OUT = ROOT/'analysis/r369_source_randomized_augmentation_20261001'
GEN = ROOT/'analysis/r363_complex_forward_20261001/models/generator.json'
MANIFEST = ROOT/'analysis/r361_command_forward_20261001/data_manifest.json'
KEYS = ('mean','scale','w1','b1','w2','b2','w3','b3')
SEEDS = (7,17,27)
ARMS = ('real_control','nominal_virtual','randomized_virtual','curriculum_hard')


def protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=json.loads(MANIFEST.read_text(encoding='utf8'))
    p={'schema':'MagTouch-R369-SourceRandomizedXY-1',
       'architecture':'existing 53->64->64->2 tanh residual XY MLP; no alternative inverse',
       'initialization':'same frozen R133 XY checkpoint, including feature mean/scale, in every arm',
       'source_train_ids':[r['id'] for r in rows if r['role']=='train'],
       'source_validation_ids':[r['id'] for r in rows if r['role']=='validation'],
       'test_ids':[r['id'] for r in rows if r['role']=='test'],
       'virtual_geometry_count':48,'virtual_families':['circle','radial','pure_z','compound','depth_loop','stop_reverse','lissajous_xyz','piecewise_xyz'],
       'geometry_seed':369100,'domain_seed':369200,'optimizer_seeds':list(SEEDS),
       'virtual_mass_candidates':[.05,.15], 'randomization_shrinkage':.5,
       'static_anchor_mass':.2,'epochs':40,'batches_per_epoch':24,'batch_size':256,'learning_rate':.0002,'weight_decay':.001,
       'select':'epoch >=20, every 5 epochs; equal source-validation Trial XY error, then mix mass',
       'quantity_control':'equal optimizer updates/batch size; real-only arm samples more real repeats',
       'hard_mining':'top nominal-VIRTUAL-training quartile by original frozen XY error; never real test error',
       'curriculum':'epochs1-10 static/pureZ/radial;11-20 all standard;21-40 all trajectories including hard tail',
       'target_fit_rows':0,'target_distribution_rows':0,'target_selection_rows':0,
       'complex_real_fit_rows':0,'complex_real_selection_rows':0,
       'randomization_scope':'source empirical per-Trial field nuisance variability, not identified material constants',
       'zero_shot_scope':'no target measurements or scores used in fitting/selection; target was reviewed in earlier project work',
       'test_timing':'weights and source-selected settings frozen before this run opens test outcomes',
       'common_feature_test':'all seeds/arms, frozen reference permission; isolates XY training effect',
       'end_to_end_test':'predeclared optimizer seed7, every arm, actual full engine with both XY routes changed identically',
       'sampling':'10Hz last-available observations, not native-rate online or a new blind experiment',
       'changed':['research XY weights only'],'unchanged':['raw measurements','formal weights','UI presets','manuscript','hardware'],
       'hashes':{'generator':sha(GEN),'xy_checkpoint':sha(ARTIFACT),'real_manifest':sha(MANIFEST)}}
    dest=OUT/'protocol.json'
    if dest.exists() and json.loads(dest.read_text(encoding='utf8'))!=p:
        raise ValueError('Existing preregistered protocol differs')
    dump(dest,p)
    return p,rows


class ResearchXY:
    """Same runtime predict contract; optionally record its exact causal input."""
    def __init__(self,model=None):
        if model is None:model=CoverageCausalXYMLP()
        for k in KEYS:setattr(self,k,np.array(getattr(model,k),copy=True))
        self.last=None
    def predict(self,x):
        x=np.asarray(x,float).reshape(-1,53)
        self.last=x[-1].copy()
        h1=np.tanh(((x-self.mean)/self.scale)@self.w1+self.b1)
        return np.tanh(h1@self.w2+self.b2)@self.w3+self.b3
    def save(self,path):np.savez_compressed(path,**{k:getattr(self,k) for k in KEYS})
    @classmethod
    def load(cls,path):
        o=cls.__new__(cls)
        with np.load(path,allow_pickle=False) as a:
            for k in KEYS:setattr(o,k,a[k].copy())
        o.last=None;return o


def replay(a,model=None,capture=True):
    """No reference, mode, path or target identity is passed to the engine."""
    e=new_engine()
    main,provisional=ResearchXY(model),ResearchXY(model)
    e.memory_decoder.xy_model=main;e.candidate_decoder.xy_model=provisional
    n=len(a['t']);pred=np.full((n,3),np.nan);x=np.full((n,53),np.nan)
    accepted=np.zeros(n,bool);route=np.zeros(n,np.int8)
    start=time.perf_counter()
    for i,(now,b,eligible,reset) in enumerate(zip(a['t'],a['b'],a['valid'],a['reset'])):
        if reset:e.set_baseline(np.zeros(15))
        main.last=provisional.last=None
        result=e.push(int(round(now*1e9))+10_000_000_000,b if eligible else np.zeros(15),eligible=bool(eligible))
        accepted[i]=bool(result['valid'])
        if not accepted[i]:continue
        pred[i]=result['xyz']
        chosen=provisional if result['extra']['localization_path']=='provisional_permission' else main
        route[i]=2 if chosen is provisional else 1
        if capture:
            if chosen.last is None:raise AssertionError('Permitted output missing same-frame XY input')
            x[i]=chosen.last
            xy=x[i,:2]*10+chosen.predict(x[i:i+1])[0]
            np.testing.assert_allclose(xy,pred[i,:2],atol=1e-10)
    return {'pred':pred,'x':x,'accepted':accepted,'route':route,'seconds':time.perf_counter()-start}


def measured(rec):
    return load_row({**rec,'path':ROOT/rec['path']})


def cache_features(a):
    dest=OUT/'features'/f"{a['id']}.npz"
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        with np.load(dest,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
    r=replay(a)
    m=a['primary']&r['accepted']&np.isfinite(r['x']).all(1)
    out={'x':r['x'],'q':a['q'],'t':a['t'],'primary':a['primary'],'train_mask':m,
         'old_pred':r['pred'],'accepted':r['accepted'],'route':r['route'],
         'guard_xy':r['x'][:,:2]*10,'safe':a['safe']}
    np.savez_compressed(dest,**out)
    print('features',a['id'],len(a['t']),'accepted_train',int(m.sum()),'s',round(r['seconds'],2),flush=True)
    return out


def calibrate(rows):
    """Fit correlated field variability vectors using SOURCE TRAIN only.

    Every node has one vector-amplitude static gain and one dynamic gain.
    Unloaded baseline residuals are estimated separately, not fitted from
    contact positions. Changes are nuisance fits, not proof of fabrication
    or mechanical parameter variation. Reuse a whole row to retain correlation.
    """
    g=AxisForward.load(GEN);samples=[];sigmas=[]
    for rec in rows:
        if rec['role']!='train':continue
        assert rec['device']=='source' and rec['mode']!='waypoint_xyz'
        a=measured(rec);mask=a['primary']&g.support(a['q'])
        safe=a['safe'];bias=np.median(a['b'][safe],axis=0) if safe.any() else np.zeros(15)
        diff=np.diff(a['b'][safe],axis=0)
        sigma=np.maximum(np.median(np.abs(diff-np.median(diff,axis=0)),axis=0)/.67449/np.sqrt(2),.1) if len(diff)>10 else np.ones(15)*.1
        qsafe=np.where(np.isfinite(a['q']).all(1)[:,None],a['q'],[0.,0.,-1.])
        static=g.static(qsafe);hist=g.rollout(a['q'],a['t'],a['reset'])-static
        gs=[];gh=[]
        for node in range(5):
            sl=slice(3*node,3*node+3)
            xx=np.column_stack((static[mask,sl].reshape(-1),hist[mask,sl].reshape(-1)))
            yy=(a['b'][mask,sl]-bias[sl]).reshape(-1)
            coef=np.linalg.solve(xx.T@xx+np.eye(2)*1.,xx.T@yy)
            gs.append(float(coef[0]));gh.append(float(coef[1]))
        samples.append({'id':rec['id'],'static_gain':gs,'history_gain':gh,'baseline_uT':bias.tolist(),'sigma_uT':sigma.tolist(),'fit_frames':int(mask.sum())})
        sigmas.append(sigma)
    flat=np.array([s['static_gain']+s['history_gain']+s['baseline_uT'] for s in samples])
    lo,hi=np.quantile(flat,[.1,.9],axis=0);clipped=np.clip(flat,lo,hi)
    d={'fit_ids':[s['id'] for s in samples],'source_only':True,'target_fit_rows':0,'samples':samples,
       'clipped_parameters':clipped.tolist(),'quantiles':[.1,.9],'bounds':[lo.tolist(),hi.tolist()],
       'sigma_uT':np.median(sigmas,axis=0).tolist(),
       'sampling':'bootstrap correlated whole-Trial vector, shrink by 0.5 toward static/history gains1 and baseline0',
       'interpretation':'empirical nuisance variations include generator residual mismatch; not constitutive parameters'}
    dump(OUT/'source_parameter_distribution.json',d);return d


def geometry(i,rng):
    family=('circle','radial','pure_z','compound','depth_loop','stop_reverse','lissajous_xyz','piecewise_xyz')[i%8]
    z=float(rng.uniform(.6,2.2));v=float(rng.uniform(.5,2.));hold=float(rng.choice([0.,.4,1.2]))
    radius=float(rng.uniform(2.,4.7));direction=int(rng.choice([-1,1]));p=[[0.,0.,-1.]]*12
    center=rng.uniform(-.6,.6,2)
    def move(q,dwell=hold,speed=v):append_segment(p,q,speed,dwell)
    move([*center,z])
    if family=='circle':
        for a in np.linspace(0,direction*2*np.pi,73):move([*(center+radius*np.array([np.cos(a),np.sin(a)])),z],0)
    elif family=='radial':
        for a in np.arange(8)*np.pi/4*direction:
            move([*(center+radius*np.array([np.cos(a),np.sin(a)])),z]);move([*center,z])
    elif family=='pure_z':
        for zz in (.45,z,2.3,.6,z):move([*center,zz])
    elif family=='compound':
        for k in range(8):
            a=k*np.pi/4*direction;zz=np.clip(z+.45*np.sin(2*a),.45,2.4)
            move([*(center+radius*.7*np.array([np.cos(a),np.sin(a)])),zz])
    elif family in ('depth_loop','lissajous_xyz'):
        for a in np.linspace(0,direction*2*np.pi,85):
            xx=radius*np.cos(a) if family=='depth_loop' else radius*np.sin(2*a+.3)
            move([center[0]+xx,center[1]+radius*.65*np.sin(a),float(np.clip(z+.4*np.sin(a+.7),.45,2.4))],0)
    elif family=='stop_reverse':
        for target in ([radius,0,z],[-radius,0,z],[radius,0,max(.45,z-.4)],[-radius,0,z]):move(target,rng.uniform(.2,1.2))
    else:
        for k in range(10):move([*rng.uniform(-4.3,4.3,2),float(rng.uniform(.45,2.3))],rng.uniform(0,.5))
    move([*p[-1][:2],-1.],2.,1.)
    q=np.array(p);t=np.arange(len(p))*.1
    return q,t,{'family':family,'nominal_depth_mm':z,'speed_mm_s':v,'dwell_s':hold,'radius_mm':radius,'direction':direction}


def generate(distribution,p,workers=1):
    g=AxisForward.load(GEN);rng=np.random.default_rng(p['geometry_seed']);dr=np.random.default_rng(p['domain_seed'])
    sigma=np.array(distribution['sigma_uT']);vectors=np.array(distribution['clipped_parameters']);entries=[]
    for i in range(p['virtual_geometry_count']):
        q,t,info=geometry(i,rng);static=g.static(q);hist=g.rollout(q,t)-static
        noise=np.random.default_rng(369300+i).normal(size=static.shape)*sigma
        draw=int(dr.integers(len(vectors)));u=vectors[draw];gamma=p['randomization_shrinkage']
        gs=1+gamma*(u[:5]-1);gh=1+gamma*(u[5:10]-1);bias=gamma*u[10:]
        for view in ('nominal','randomized'):
            b=static+hist+noise if view=='nominal' else static*np.repeat(gs,3)+hist*np.repeat(gh,3)+bias+noise
            ident=f'V369_{i:03d}_{view}';reset=np.r_[True,np.zeros(len(t)-1,bool)]
            primary=(q[:,2]>=.4)&(q[:,2]<=3)&g.support(q)
            a={'id':ident,'q':q,'t':t,'b':b,'reset':reset,'valid':np.ones(len(t),bool),'primary':primary,'safe':q[:,2]<=-.5}
            np.savez_compressed(OUT/'virtual'/f'{ident}.npz',**a)
            ent={'id':ident,'geometry_id':i,'role':'train','view':view,**info,'frames':len(t),'duration_s':float(t[-1]),'parameter_source_trial':distribution['fit_ids'][draw] if view=='randomized' else None,
                 'sha256':sha(OUT/'virtual'/f'{ident}.npz')}
            entries.append(ent)
        print('virtual pair',i+1,'/',p['virtual_geometry_count'],flush=True)
    dump(OUT/'virtual_catalog.json',{'geometries':p['virtual_geometry_count'],'views':2,'trials':entries,'target_used':False,'held_out_real_geometry_used':False,'independent_trial_note':'two field views of one geometry are paired, not independent physical devices'})
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(cache_virtual, [r['id'] for r in entries]))
    else:
        for r in entries:cache_virtual(r['id'])


def cache_virtual(ident):
    with np.load(OUT/'virtual'/f'{ident}.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    a['id']=str(a['id']);return cache_features(a)['train_mask'].sum().item()


def cache_measured(rec):
    return cache_features(measured(rec))['train_mask'].sum().item()


def load_features(ident):
    with np.load(OUT/'features'/f'{ident}.npz',allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}


def sampled_training(items):
    xs=[];ys=[];ws=[];ids=[]
    for i,a in enumerate(items):
        ix=np.flatnonzero(a['train_mask'])[::2]
        if not len(ix):continue
        xs.append(a['x'][ix]);ys.append(np.clip(a['q'][ix,:2]-a['guard_xy'][ix],-3,3))
        ws.append(np.full(len(ix),1/len(ix)));ids.extend([i]*len(ix))
    return np.vstack(xs),np.vstack(ys),np.concatenate(ws),np.array(ids)


def anchors():
    decoder=new_engine().memory_decoder;atlas=decoder.atlas
    samples=load_profile(CONFIG,PROFILE_KEY)['samples']
    q=np.array([[s['x'],s['y'],s['depth']] for s in samples])[::3]
    b=np.array([s['delta'] for s in samples])[::3]
    return np.column_stack((q/np.array([10,10,3]),atlas.features(b),np.zeros((len(q),34)))),np.zeros((len(q),2))


def validation(model,items):
    errors=[]
    for a in items:
        m=a['train_mask'];pred=a['guard_xy'][m]+model.predict(a['x'][m])
        errors.append(float(np.linalg.norm(pred-a['q'][m,:2],axis=1).mean()))
    return float(np.mean(errors))


def train_one(seed,arm,mass,real,virtual,valid,catalog):
    model=ResearchXY();rng=np.random.default_rng(seed);rx,ry,rw,_=sampled_training(real)
    vx,vy,vw,groups=sampled_training(virtual) if virtual else (rx,ry,rw,np.zeros(len(rx),int))
    ax,ay=anchors();params=[getattr(model,k) for k in KEYS[2:]]
    first=[np.zeros_like(a) for a in params];second=[np.zeros_like(a) for a in params];step=0;best=np.inf;history=[];selected_epoch=None
    hard=np.array([np.linalg.norm(a['old_pred'][a['train_mask'],:2]-a['q'][a['train_mask'],:2],axis=1).mean() for a in virtual]) if virtual else np.zeros(1)
    cutoff=np.quantile(hard,.75)
    for epoch in range(1,41):
        vprob=vw.copy()
        if arm=='curriculum_hard':
            families=np.array([catalog[i]['family'] for i in range(len(virtual))])
            allowed=np.isin(families,['pure_z','radial']) if epoch<=10 else np.isin(families,['circle','radial','pure_z','compound']) if epoch<=20 else np.ones(len(virtual),bool)
            hardweight=np.where(hard>=cutoff,2.,1.) if epoch>20 else np.ones(len(virtual))
            vprob*=allowed[groups]*hardweight[groups]
        vprob/=vprob.sum();rprob=rw/rw.sum()
        for _ in range(24):
            na=51;nv=round((256-na)*mass) if virtual else 0;nr=256-na-nv
            ri=rng.choice(len(rx),nr,p=rprob);vi=rng.choice(len(vx),nv,p=vprob);ai=rng.integers(len(ax),size=na)
            xx=np.vstack((rx[ri],vx[vi],ax[ai]));yy=np.vstack((ry[ri],vy[vi],ay[ai]))
            xx=(xx-model.mean)/model.scale;w1,b1,w2,b2,w3,b3=params
            h1=np.tanh(xx@w1+b1);h2=np.tanh(h1@w2+b2);out=h2@w3+b3
            delta3=2*(out-yy)/len(xx);g3=h2.T@delta3+.001*w3;gb3=delta3.sum(0)
            delta2=(delta3@w3.T)*(1-h2*h2);g2=h1.T@delta2+.001*w2;gb2=delta2.sum(0)
            delta1=(delta2@w2.T)*(1-h1*h1);g1=xx.T@delta1+.001*w1;gb1=delta1.sum(0)
            step+=1
            for a,grad,m,v in zip(params,(g1,gb1,g2,gb2,g3,gb3),first,second):
                m*=.9;m+=.1*grad;v*=.999;v+=.001*grad*grad
                a-=.0002*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        if epoch%5==0:
            value=validation(model,valid);history.append({'epoch':epoch,'source_validation_xy_mm':value})
            if epoch>=20 and value<best:
                best=value;selected_epoch=epoch;chosen=[a.copy() for a in params]
    for k,v in zip(KEYS[2:],chosen):setattr(model,k,v)
    name=f'{arm}_seed{seed}_mass{mass:.2f}';dest=OUT/'models'/f'{name}.npz';model.save(dest)
    row={'seed':seed,'arm':arm,'virtual_mass':mass,'epoch':selected_epoch,'source_validation_xy_mm':best,
         'file':dest.relative_to(ROOT).as_posix(),'sha256':sha(dest),'history':history,'optimizer_updates':step,
         'source_train_samples':len(rx),'virtual_samples':len(vx) if virtual else 0,'static_anchors':len(ax)}
    print('trained',name,'val',round(best,4),'epoch',selected_epoch,flush=True);return row


def prepare(workers=1):
    p,rows=protocol()
    for sub in ('features','virtual','models','predictions'): (OUT/sub).mkdir(exist_ok=True)
    source=[r for r in rows if r['role'] in ('train','validation')]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(cache_measured,source))
    else:
        for r in source:cache_measured(r)
    dist=calibrate(rows)
    if not (OUT/'virtual_catalog.json').exists():generate(dist,p,workers)
    else:
        catalog=json.loads((OUT/'virtual_catalog.json').read_text(encoding='utf8'))['trials']
        with ProcessPoolExecutor(max_workers=workers) as pool:list(pool.map(cache_virtual,[r['id'] for r in catalog]))


def fit_job(job):
    seed,arm,mass=job
    rows=json.loads(MANIFEST.read_text(encoding='utf8'))
    real=[load_features(r['id']) for r in rows if r['role']=='train']
    valid=[load_features(r['id']) for r in rows if r['role']=='validation']
    catalog=json.loads((OUT/'virtual_catalog.json').read_text(encoding='utf8'))['trials']
    view='nominal' if arm=='nominal_virtual' else 'randomized'
    entries=[r for r in catalog if r['view']==view]
    subset=[] if arm=='real_control' else [load_features(r['id']) for r in entries]
    return train_one(seed,arm,mass,real,subset,valid,entries)


def fit(workers=1):
    p,rows=protocol()
    if (OUT/'freeze.json').exists():raise FileExistsError('Preserve source-only freeze; start a separately declared study to refit')
    assert not (OUT/'partial_evaluation.json').exists() and not (OUT/'report.json').exists(),'No refit after target outcomes'
    jobs=[(seed,arm,mass) for seed in SEEDS for arm in ARMS for mass in ([0.] if arm=='real_control' else p['virtual_mass_candidates'])]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:choices=list(pool.map(fit_job,jobs))
    else:choices=[fit_job(j) for j in jobs]
    selected=[]
    for seed in SEEDS:
        for arm in ARMS:
            variants=[r for r in choices if r['seed']==seed and r['arm']==arm]
            selected.append(min(variants,key=lambda r:r['source_validation_xy_mm']))
    dump(OUT/'freeze.json',{'protocol_sha256':sha(OUT/'protocol.json'),'parameter_distribution_sha256':sha(OUT/'source_parameter_distribution.json'),
         'catalog_sha256':sha(OUT/'virtual_catalog.json'),'choices':choices,'selected':selected,
         'target_fit_rows':0,'target_selection_rows':0,'complex_fit_rows':0,'complex_selection_rows':0,
         'frozen_before_test_outcomes':True,'original_xy_checkpoint_sha256':sha(ARTIFACT),'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})


def direct_score(model,a):
    pred=a['old_pred'].copy();m=a['accepted']&np.isfinite(a['x']).all(1)
    pred[m,:2]=a['guard_xy'][m]+model.predict(a['x'][m])
    # Z is intentionally not evaluated here: it must be rerun on changed XY.
    result=metrics(pred,a['q'],a['t'],a['primary'],a['accepted'])
    for k in ('xyz_mean_mm','z_mae_mm','xyz_translation_removed_mm','z_translation_removed_mm','constant_reference_z_steps'):result.pop(k,None)
    return result


def evaluate_trial(rec):
    f=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    a=measured(rec);features=cache_features(a);old=ResearchXY()
    row={k:rec[k] for k in ('id','device','mode')};row['models']={'original':direct_score(old,features)}
    for r in f['selected']:
        name=f"{r['arm']}_seed{r['seed']}"
        row['models'][name]=direct_score(ResearchXY.load(ROOT/r['file']),features)
    original_pred=features['old_pred'];original_valid=features['accepted'];safe=a['safe'];full=[]
    for item in [None]+[r for r in f['selected'] if r['seed']==7]:
        arm='original' if item is None else item['arm']
        dest=OUT/'predictions'/f"{rec['id']}_{arm}.npz"
        if dest.exists():
            with np.load(dest,allow_pickle=False) as arr:pred,accepted=arr['pred'],arr['accepted']
        else:
            output={'pred':original_pred,'accepted':original_valid} if item is None else replay(a,ResearchXY.load(ROOT/item['file']),False)
            pred,accepted=output['pred'],output['accepted']
            np.savez_compressed(dest,pred=pred,accepted=accepted,t=a['t'],q=a['q'],primary=a['primary'],safe=safe)
        common=accepted&original_valid
        rr={k:rec[k] for k in ('id','device','mode')}
        rr.update(arm=arm,own_mask=metrics(pred,a['q'],a['t'],a['primary'],accepted),
            common_mask=metrics(pred,a['q'],a['t'],a['primary'],common),
            original_on_common=metrics(original_pred,a['q'],a['t'],a['primary'],common),
            safe_reference_frames=int(safe.sum()),safe_output_frames=int((safe&accepted).sum()),
            output_permission_changes=int(np.count_nonzero(accepted!=original_valid)))
        full.append(rr)
    print('test done',rec['id'],rec['device'],rec['mode'],flush=True)
    return row,full


def evaluate(workers=1):
    p,rows=protocol();f=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    assert f['protocol_sha256']==sha(OUT/'protocol.json') and f['frozen_before_test_outcomes']
    for k in ('target_fit_rows','target_selection_rows','complex_fit_rows','complex_selection_rows'):assert f[k]==0
    for r in f['selected']:assert sha(ROOT/r['file'])==r['sha256']
    scores=[];full=[];tests=[r for r in rows if r['role']=='test']
    pool=ProcessPoolExecutor(max_workers=workers) if workers>1 else None
    iterator=pool.map(evaluate_trial,tests) if pool else map(evaluate_trial,tests)
    for row,full_rows in iterator:
        scores.append(row);full.extend(full_rows)
        dump(OUT/'partial_evaluation.json',{'fixed_feature':scores,'full_stack_seed7':full,'freeze_sha256':sha(OUT/'freeze.json')})
    if pool:pool.shutdown()
    aggregate={}
    for device in ('source','target'):
        subset=[r for r in scores if r['device']==device]
        aggregate[device]={}
        for arm in ('original',)+ARMS:
            names=['original'] if arm=='original' else [f'{arm}_seed{s}' for s in SEEDS]
            seed_rows=[]
            for name in names:
                val=[r['models'][name] for r in subset]
                seed_rows.append({k:float(np.mean([v[k] for v in val if k in v])) for k in ('xy_mean_mm','xy_translation_removed_mm','coverage')})
            aggregate[device][arm]={'mean':{k:float(np.mean([r[k] for r in seed_rows])) for k in seed_rows[0]},'by_seed':seed_rows}
    end={}
    for device in ('source','target'):
        end[device]={}
        for arm in ('original',)+ARMS:
            subset=[r for r in full if r['device']==device and r['arm']==arm]
            end[device][arm]={key:{k:float(np.mean([r[key][k] for r in subset if k in r[key]])) for k in ('xy_mean_mm','z_mae_mm','xyz_mean_mm','coverage','xy_translation_removed_mm','z_translation_removed_mm')} for key in ('own_mask','common_mask')}
            end[device][arm]['safe_output_frames']=sum(r['safe_output_frames'] for r in subset)
    dump(OUT/'report.json',{'fixed_features':scores,'full_stack_seed7':full,'macro_xy_by_device':aggregate,'full_stack_macro':end,
         'formal_model_replaced':False,'source_only_freeze':True,'target_zero_shot':True,
         'warning':'Retrospective test for the new source-only fine-tuning study; these recordings were previously examined. Domain distribution is inferred from one source device, not proven target-manufacturing variation.',
         'full_stack_seed':7,'freeze_sha256':sha(OUT/'freeze.json')})
    print(json.dumps(end,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','evaluate']);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if a.stage=='prepare':prepare(a.workers)
    elif a.stage=='fit':fit(a.workers)
    else:evaluate(a.workers)
if __name__=='__main__':main()
