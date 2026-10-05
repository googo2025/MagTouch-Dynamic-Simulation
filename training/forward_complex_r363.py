"""Source-only axis-separated magnetic history and complex synthetic trials.

Empirical command-to-field surrogate, NOT FEM. Held-out waypoints/target device
are not opened until generator and inverse-readout choices have been frozen.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from training.magnetic_forward_r361 import spatial_features, fit_readout, MagneticForward
from training.run_forward_r361 import ROOT, MODEL_DIR, inventory, load_row, metric, dump, sha, translation_diagnostic
from training.augment_forward_r361 import (append_segment, arrays, hidden_features,
    magnetic_features, score_prediction)

OUT=ROOT/'analysis/r363_complex_forward_20261001'


def axis_history(q,t,taus,reset=None,split=True):
    q=np.asarray(q,float);t=np.asarray(t,float)
    if q.shape!=(len(t),3) or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('matching XYZ and increasing finite time required')
    known=np.isfinite(q).all(1);qs=np.where(known[:,None],q,0)
    p=spatial_features(qs);previous=np.vstack((qs[0],qs[:-1]))
    a=previous.copy();a[:,:2]=qs[:,:2]
    c=qs.copy();c[:,:2]=previous[:,:2]
    pp=np.vstack((p[0],p[:-1]));total=p-pp
    xy=.5*(spatial_features(a)-pp+p-spatial_features(c));z=total-xy
    components=(xy,z) if split else (total,)
    resets=np.zeros(len(t),bool) if reset is None else np.asarray(reset,bool)
    parts=[]
    for d in components:
        for tau in taus:
            if tau<=0:raise ValueError('positive time constant required')
            state=np.zeros(110);h=np.zeros_like(p)
            for i in range(len(t)):
                invalid=resets[i] or not known[i] or (i and (not known[i-1] or t[i]-t[i-1]>.5))
                if invalid:state[:]=0
                elif i:state=np.exp(-(t[i]-t[i-1])/tau)*(state+d[i])
                h[i]=state
            parts.append(h)
    return np.column_stack(parts)


class AxisForward:
    def __init__(self,static,coef,taus,split=True,metadata=None):
        self.base=static;self.coef=np.asarray(coef,float);self.taus=tuple(taus)
        self.split=split;self.metadata=metadata or {}
    def rollout(self,q,t,reset=None):
        q=np.asarray(q,float);known=np.isfinite(q).all(1)
        return self.base.static(np.where(known[:,None],q,0))+axis_history(q,t,self.taus,reset,self.split)@self.coef
    def static(self,q):return self.base.static(q)
    def support(self,q):return self.base.support(q)
    def save(self,p):
        dump(p,{'schema':'MagTouch-AxisForward-R363-v1','base':self.base.to_dict(),
          'coef':self.coef.tolist(),'taus_s':list(self.taus),'split':self.split,'metadata':self.metadata,
          'physics_validated':False,'contact_count':1})
    @classmethod
    def load(cls,p):
        d=json.loads(Path(p).read_text(encoding='utf8'))
        if d['schema']=='MagTouch-CommandForward-R361-v1':return MagneticForward.load(p)
        if d['schema']!='MagTouch-AxisForward-R363-v1':raise ValueError('unknown schema')
        b=d['base'];return cls(MagneticForward(b['coef'],b['taus_s'],b['metadata']),d['coef'],d['taus_s'],d['split'],d['metadata'])


def grouped_scores(model,rows):
    scores=[{'id':a['id'],'mode':a['mode'],**metric(model.rollout(a['q'],a['t'],a['reset']),a)} for a in rows]
    modes=sorted({r['mode'] for r in scores})
    groups={m:float(np.mean([r['nrmse'] for r in scores if r['mode']==m])) for m in modes}
    return {'trial_macro_nrmse':float(np.mean([r['nrmse'] for r in scores])),
            'family_macro_nrmse':float(np.mean(list(groups.values()))),'by_family':groups,'trials':scores}


def fit_generator(train,valid):
    static=MagneticForward.load(MODEL_DIR/'instantaneous.json')
    old=MagneticForward.load(MODEL_DIR/'dynamic.json')
    baseline=grouped_scores(static,valid);candidates=[];models={}
    # Fix static map first; dynamic fitting cannot move its equilibrium solution.
    for split in (False,True):
        for taus in ((.15,.6,2.),(.3,1.,4.)):
            xx=[];yy=[];ww=[]
            for a in train:
                if a['role']=='static_calibration':continue
                mask=a['fit'];x=axis_history(a['q'],a['t'],taus,a['reset'],split)
                w=np.where(a['boundary'][mask],.5,1.);w/=w.sum()
                xx.append(x[mask]);yy.append(a['b'][mask]-static.static(np.where(np.isfinite(a['q'][mask]),a['q'][mask],0)));ww.append(w)
            x,y,w=np.vstack(xx),np.vstack(yy),np.concatenate(ww)
            for alpha in (.01,.1,1.):
                name=('axis' if split else 'joint')+'_'+str(taus[0])+'_'+str(alpha)
                m=AxisForward(static,fit_readout(x,y,w,alpha),taus,split)
                score=grouped_scores(m,valid)
                # Fixed source-only pure-Z safeguard; test outcomes cannot change it.
                guarded=[f for f in baseline['by_family'] if f in ('z_path','grid_points')]
                eligible=all(score['by_family'][f]<=1.10*baseline['by_family'][f] for f in guarded)
                candidates.append({'id':name,'alpha':alpha,'taus':list(taus),'split':split,'eligible':eligible,**score})
                models[name]=m
                print('generator',name,round(score['family_macro_nrmse'],4),'eligible',eligible,flush=True)
    eligible=[r for r in candidates if r['eligible']]
    # Include static/no-memory fallback in the selection, not a forced improvement.
    pick=min(eligible+[{'id':'static','family_macro_nrmse':baseline['family_macro_nrmse']}],key=lambda r:r['family_macro_nrmse'])
    model=static if pick['id']=='static' else models[pick['id']]
    metadata={'train_ids':[a['id'] for a in train],'validation_ids':[a['id'] for a in valid],
              'target_used':False,'waypoint_used':False,'static_base_sha256':sha(MODEL_DIR/'instantaneous.json')}
    model.metadata.update(metadata);model.save(OUT/'models/generator.json')
    dump(OUT/'generator_selection.json',{'baseline_static':baseline,'baseline_R361':grouped_scores(old,valid),
       'objective':'equal family source-validation NRMSE; pure-Z/grid <= 110% of fixed static baseline',
       'selected':pick['id'],'candidates':candidates,'target_selection':False,'waypoint_selection':False,
       'model_sha256':sha(OUT/'models/generator.json')})
    return model


def synthesize(model,train,seed=363,count=60):
    rng=np.random.default_rng(seed);trials=[]
    real=[a for a in train if a['role']=='train']
    dif=np.vstack([np.diff(a['b'],axis=0)[a['safe'][1:]&a['safe'][:-1]] for a in real])
    sigma=np.maximum(1.4826*np.median(np.abs(dif-np.median(dif,0)),0)/np.sqrt(2),.1)
    ref=np.vstack([a['q'][a['primary']][::5] for a in real])
    if len(ref)>1800:ref=ref[np.linspace(0,len(ref)-1,1800).astype(int)]
    families=('lissajous_xyz','variable_depth_loop','spiral_xyz','piecewise_xyz','stop_reverse','press_slide_release')
    for i in range(count):
        family=families[i%len(families)];points=[[0.,0.,-1.]]*10
        xy=rng.uniform(-1,1,2);depth=float(rng.uniform(.8,1.9));speed=float(rng.uniform(.5,2.))
        append_segment(points,[*xy,depth],1,.6)
        phase=rng.uniform(0,2*np.pi);amp=rng.uniform(1.5,4.4);zamp=min(depth-.45,2.25-depth,.55)
        if family in ('lissajous_xyz','variable_depth_loop','spiral_xyz'):
            for u in np.linspace(0,2*np.pi,100)[1:]:
                if family=='lissajous_xyz':target=[amp*np.sin(u+phase),.8*amp*np.sin(2*u+phase),depth+zamp*np.sin(3*u+.7)]
                elif family=='variable_depth_loop':target=[amp*np.cos(u+phase),.65*amp*np.sin(u+phase),depth+zamp*np.sin(2*u+1.1)]
                else:
                    r=amp*(.35+.65*u/(2*np.pi));target=[r*np.cos(2*u+phase),r*np.sin(2*u+phase),depth+zamp*np.sin(u)]
                append_segment(points,target,speed,0)
        elif family=='piecewise_xyz':
            for _ in range(10):append_segment(points,[*rng.uniform(-4.5,4.5,2),rng.uniform(.45,2.2)],speed,rng.choice([0.,.3,1.]))
        elif family=='stop_reverse':
            target=[*rng.uniform(-4.4,4.4,2),rng.uniform(.6,2.1)]
            for _ in range(3):
                append_segment(points,target,speed,rng.uniform(.5,2.))
                append_segment(points,[*xy,depth],speed*.6,rng.uniform(.2,1.))
        else:
            for _ in range(4):
                p=[*rng.uniform(-4.4,4.4,2),rng.uniform(.45,2.2)]
                append_segment(points,p,speed,rng.uniform(.2,1.))
                append_segment(points,[*p[:2],-1],1,1.)
                append_segment(points,[*p[:2],depth],1,.5)
        append_segment(points,[*points[-1][:2],-1],1,2.)
        q=np.asarray(points);t=np.arange(len(q))*.1
        b=model.rollout(q,t)+rng.normal(size=(len(q),15))*sigma
        mask=(q[:,2]>=.4)&model.support(q)
        distance=np.concatenate([np.sqrt(np.min(np.sum(((block[:,None,:]-ref[None,:,:])/np.array([1.,1.,.6]))**2,axis=2),axis=1)) for block in np.array_split(q,max(1,int(np.ceil(len(q)/256))))])
        support=1/(1+distance**2)
        trials.append({'id':f'SYN_R363_{i:03d}','family':family,'q':q,'t':t,'b':b,'primary':mask,
          'reset':np.r_[True,np.zeros(len(q)-1,bool)],'support_weight':support})
    offsets=np.cumsum([0]+[len(a['t']) for a in trials])
    np.savez_compressed(OUT/'complex_synthetic.npz',q=np.vstack([a['q'] for a in trials]),b=np.vstack([a['b'] for a in trials]),
       t=np.concatenate([a['t'] for a in trials]),offsets=offsets,support_weight=np.concatenate([a['support_weight'] for a in trials]))
    dump(OUT/'synthetic_manifest.json',{'seed':seed,'contact_count':1,'physics_validated':False,
       'generator_sha256':sha(OUT/'models/generator.json'),'heldout_waypoint_geometry_used':False,'target_used':False,
       'noise_sigma_uT':sigma.tolist(),'support_weight':'source-training XYZ nearest-distance heuristic, not confidence probability',
       'trials':[{'id':a['id'],'family':a['family'],'frames':len(a['t'])} for a in trials]})
    return trials


def covariances(rows,seed,weighted=False):
    x,y,w=arrays(rows,seed)
    if weighted:
        start=0
        for a in rows:
            mask=a['primary']&np.isfinite(a['b']).all(1);n=int(mask.sum())
            if n:
                ww=a['support_weight'][mask];w[start:start+n]=ww/ww.sum()
            start+=n
    w/=w.sum()
    return x.T@(x*w[:,None]),x.T@(y*w[:,None])


def fit_cov(cov,rhs,alpha):
    scale=np.maximum(np.sqrt(np.diag(cov)),.02)
    return np.linalg.solve(cov/scale[:,None]/scale[None,:]+alpha*np.eye(len(scale)),rhs/scale[:,None])/scale[:,None]


def train_probes(train,valid,synth):
    choices=[];models={}
    train=[a for a in train if a['role']=='train']
    for seed in (361,362,363):
        rc,ry=covariances(train,seed);sc,sy=covariances(synth,seed,True)
        xv=[hidden_features(magnetic_features(a['b'],a['reset']),seed) for a in valid]
        best_real=None;best_aug=None;best_syn=None
        for mass in (0.,.02,.05,.1,.2,1.):
            cov=(1-mass)*rc+mass*sc;rhs=(1-mass)*ry+mass*sy
            for alpha in (.0001,.001,.01,.1):
                coef=fit_cov(cov,rhs,alpha)
                scores=[score_prediction(x@coef*np.array([6,6,3]),a) for x,a in zip(xv,valid)]
                obj=float(np.mean([r['xyz_mean_mm'] for r in scores]))
                item={'seed':seed,'mass':mass,'alpha':alpha,'validation_xyz_mm':obj}
                choices.append(item)
                target='real' if mass==0 else 'synthetic' if mass==1 else 'augmentation'
                key=(seed,target)
                if key not in models or obj<models[key][0]:models[key]=(obj,coef,item)
        # Select among augmentation and zero synthetic mass using SOURCE validation.
        real=models[(seed,'real')];aug=models[(seed,'augmentation')]
        models[(seed,'selected')]=min((real,aug),key=lambda v:v[0])
        for arm in ('real','augmentation','synthetic','selected'):
            obj,coef,item=models[(seed,arm)]
            np.savez_compressed(OUT/'models'/f'probe_{seed}_{arm}.npz',coef=coef,seed=seed)
            print('probe',seed,arm,item['mass'],round(obj,4),flush=True)
    dump(OUT/'probe_selection.json',{'architecture':'same R361 48 causal features / random tanh 192 / linear skip / ridge XYZ',
       'choices':choices,'selected':[models[(s,a)][2] for s in (361,362,363) for a in ('real','augmentation','synthetic','selected')],
       'selection_source_only':True,'formal_decoder_modified':False,'includes_contact_gate':False,
       'zero_mass_fallback_included':True})
    return {k:v[1] for k,v in models.items()}


def main():
    for p in (OUT,OUT/'models',OUT/'predictions'):p.mkdir(parents=True,exist_ok=True)
    inv=inventory();train=[load_row(r) for r in inv if r['role'] in ('train','static_calibration')]
    valid=[load_row(r) for r in inv if r['role']=='validation']
    dump(OUT/'protocol.json',{'training':[a['id'] for a in train],'validation':[a['id'] for a in valid],
      'test_ids':[r['id'] for r in inv if r['role']=='test'],'target_train_count':0,'waypoint_train_count':0,
      'base_manifest_sha256':sha(ROOT/'analysis/r361_command_forward_20261001/data_manifest.json'),
      'test_data_previously_inspected':True,'interpretation':'new-fit heldout, not newly blind physical data'})
    model=fit_generator(train,valid);synth=synthesize(model,train);probes=train_probes(train,valid,synth)
    dump(OUT/'freeze.json',{'files':{p.name:sha(p) for p in sorted((OUT/'models').iterdir())},'test_opened_after_freeze':True})
    static=MagneticForward.load(MODEL_DIR/'instantaneous.json');old=MagneticForward.load(MODEL_DIR/'dynamic.json')
    result=[]
    for r in inv:
        if r['role']!='test':continue
        a=load_row(r);pred=model.rollout(a['q'],a['t'],a['reset'])
        item={k:a[k] for k in ('id','device','mode')}
        item.update({'static':metric(static.rollout(a['q'],a['t'],a['reset']),a),
         'old_history':metric(old.rollout(a['q'],a['t'],a['reset']),a),'new_history':metric(pred,a),
         'new_shallow':metric(pred,a,a['shallow']),'new_safe':metric(pred,a,a['safe'])})
        item['xyz_offset_diagnostic']=translation_diagnostic(model,a,pred)
        # Posthoc common field bias is diagnostic only, never used in scores.
        item['probes']=[];saved={'t':a['t'],'q':a['q'],'b':a['b'],'primary':a['primary'],'new_history':pred}
        for seed in (361,362,363):
            x=hidden_features(magnetic_features(a['b'],a['reset']),seed)
            rr={'seed':seed}
            for arm in ('real','augmentation','synthetic','selected'):
                pp=x@probes[(seed,arm)]*np.array([6,6,3]);rr[arm]=score_prediction(pp,a)
                if seed==361:saved[arm]=pp
            item['probes'].append(rr)
        result.append(item);np.savez_compressed(OUT/'predictions'/(a['id']+'.npz'),**saved)
    macro={}
    for device in ('source','target'):
        ss=[r for r in result if r['device']==device]
        macro[device]={'forward':{k:{metric_name:float(np.mean([r[k][metric_name] for r in ss])) for metric_name in ('rmse_uT','nrmse')} for k in ('static','old_history','new_history')},
          'inverse_probe':{arm:{k:float(np.mean([p[arm][k] for r in ss for p in r['probes']])) for k in ('xyz_mean_mm','xy_mean_mm','z_mae_mm','xy_translation_removed_mm','z_translation_removed_mm')} for arm in ('real','augmentation','synthetic','selected')}}
    dump(OUT/'report.json',{'test':result,'macro':macro,'primary_scores_unregistered':True,
       'synthetic_and_probe_research_only':True,'multi_contact_simulation':False,'paper_or_acquisition_UI_changed':False})
    print(json.dumps(macro,indent=2),flush=True)


if __name__=='__main__':main()
