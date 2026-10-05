"""Three-arm, same-architecture research readout experiment on REAL held-out B.

Not the deployed full MagTouch decoder: there is no contact/output-permission
layer in this probe. It tests usefulness of synthetic data, not product quality.
"""
from __future__ import annotations
import json
import numpy as np
from training.magnetic_forward_r361 import MagneticForward,fit_readout
from training.run_forward_r361 import OUT,MODEL_DIR,inventory,load_row,dump,sha


def append_segment(points,target,speed=1.5,hold=.4):
    a=np.array(points[-1]);b=np.array(target);duration=max(.1,float(np.linalg.norm(b-a)/speed))
    count=max(1,int(np.ceil(duration/.1)))
    for u in np.linspace(0,1,count+1)[1:]: points.append((a+u*(b-a)).tolist())
    points.extend([b.tolist() for _ in range(round(hold/.1))])


def synthetic_trials(model,noise_sigma,seed=361,count=36):
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        depth=float(rng.uniform(.6,2.2));speed=float(rng.uniform(.6,2.4))
        points=[[0.,0.,-1.]]*10; family=('circle','radial','vertical')[i%3]
        append_segment(points,[0,0,depth],1.,.5)
        if family=='circle':
            radius=float(rng.uniform(1.5,5.));start=rng.uniform(0,2*np.pi)
            append_segment(points,[radius*np.cos(start),radius*np.sin(start),depth],speed)
            count_arc=int(np.ceil(2*np.pi*radius/speed/.1))
            for angle in start+np.linspace(0,2*np.pi,count_arc+1)[1:]*(-1 if i%2 else 1):
                points.append([radius*np.cos(angle),radius*np.sin(angle),depth])
        elif family=='radial':
            for angle in rng.uniform(0,2*np.pi,6):
                radius=float(rng.uniform(1.5,5.))
                append_segment(points,[radius*np.cos(angle),radius*np.sin(angle),depth],speed,rng.uniform(.2,1.))
                append_segment(points,[0,0,depth],speed,rng.uniform(.2,1.))
        else:
            xy=rng.uniform(-3,3,2)
            append_segment(points,[*xy,depth],speed)
            for z in [.5,1.,1.8,2.3,1.8,1.,.5]:append_segment(points,[*xy,z],speed,rng.uniform(.2,1.))
        append_segment(points,[0,0,depth],speed)
        append_segment(points,[0,0,-1],1.,2.)
        q=np.array(points);t=np.arange(len(q))*.1;b=model.rollout(q,t)
        # Source-only unloaded sensor noise. No target residuals or biases used.
        b+=rng.normal(size=b.shape)*noise_sigma
        m=(q[:,2]>=.4)&model.support(q)
        out.append({'id':f'SYN_{seed}_{i:03d}','q':q,'t':t,'b':b,'primary':m,'fit':m,
                    'reset':np.r_[True,np.zeros(len(q)-1,bool)],'family':family})
    return out


def magnetic_features(b,reset):
    b=np.asarray(b,float);finite=np.isfinite(b).all(1);safe=np.where(finite[:,None],b,0.)
    amplitude=np.linalg.norm(safe,axis=1)
    phi=np.column_stack((safe/np.maximum(amplitude[:,None],1.),.5*np.log1p(amplitude/100.)))
    stream=np.cumsum(np.asarray(reset,bool)|~finite)
    out=[phi]
    for lag in (1,4):
        d=np.zeros_like(phi);m=stream[lag:]==stream[:-lag]
        d[lag:]=np.where(m[:,None],phi[lag:]-phi[:-lag],0.)
        out.append(d)
    return np.column_stack(out)


def hidden_features(x,seed):
    rng=np.random.default_rng(seed)
    w=rng.normal(0,.55,(48,192));bias=rng.normal(0,.4,192)
    return np.column_stack((np.ones(len(x)),x,np.tanh(x@w+bias)))


def arrays(rows,seed):
    x=[];y=[];w=[]
    for a in rows:
        m=a['primary']&np.isfinite(a['b']).all(1)
        if not m.any():continue
        x.append(hidden_features(magnetic_features(a['b'],a['reset']),seed)[m])
        y.append(a['q'][m]/np.array([6.,6.,3.]))
        w.append(np.full(m.sum(),1/m.sum()))
    return np.vstack(x),np.vstack(y),np.concatenate(w)


def score_prediction(pred,a):
    e=(pred-a['q'])[a['primary']];b=e.mean(0);center=e-b
    return {'frames':len(e),'xy_mean_mm':float(np.linalg.norm(e[:,:2],axis=1).mean()),
            'z_mae_mm':float(np.abs(e[:,2]).mean()),'xyz_mean_mm':float(np.linalg.norm(e,axis=1).mean()),
            'bias_xyz_mm':b.tolist(),'xy_translation_removed_mm':float(np.linalg.norm(center[:,:2],axis=1).mean()),
            'z_translation_removed_mm':float(np.abs(center[:,2]).mean())}


def main():
    inv=inventory();train=[load_row(r) for r in inv if r['role']=='train']
    valid=[load_row(r) for r in inv if r['role']=='validation']
    model=MagneticForward.load(MODEL_DIR/'dynamic.json')
    differences=[]
    for a in train:
        m=a['safe'][1:]&a['safe'][:-1]
        differences.append(np.diff(a['b'],axis=0)[m])
    dif=np.vstack(differences)
    sigma=np.maximum(1.4826*np.median(np.abs(dif-np.median(dif,0)),axis=0)/np.sqrt(2),.1)
    synth=synthetic_trials(model,sigma)
    offsets=np.cumsum([0]+[len(a['t']) for a in synth])
    np.savez_compressed(OUT/'synthetic_research_data.npz',q=np.vstack([a['q'] for a in synth]),
                        b=np.vstack([a['b'] for a in synth]),t=np.concatenate([a['t'] for a in synth]),
                        primary=np.concatenate([a['primary'] for a in synth]),offsets=offsets)
    dump(OUT/'synthetic_manifest.json',{'generated':True,'physics_validated':False,'training_release':'research_only',
        'generator_sha256':sha(MODEL_DIR/'dynamic.json'),'seed':361,'source_only_noise_sigma_uT':sigma.tolist(),
        'contact_count':1,'trials':[{'id':a['id'],'family':a['family'],'frames':len(a['t'])} for a in synth],
        'held_out_waypoint_geometry_used':False,'target_used':False,
        'warning':'virtual labels are prescribed commands, not independent physical truth'})
    # Freeze every arm using source E before loading any target/test outcome.
    fitted={};selection=[]
    for seed in (361,362,363):
        xv=[hidden_features(magnetic_features(a['b'],a['reset']),seed) for a in valid]
        for arm,rows in [('real',train),('synthetic',synth),('mixed',train+synth)]:
            x,y,w=arrays(rows,seed)
            if arm=='mixed':
                nreal=sum(a['primary'].sum() for a in train)
                w[:nreal]*=.5/len(train);w[nreal:]*=.5/len(synth)
            choices=[]
            for alpha in (.0001,.001,.01,.1):
                coef=fit_readout(x,y,w,alpha)
                val=[score_prediction(xx@coef*np.array([6,6,3]),a) for xx,a in zip(xv,valid)]
                obj=float(np.mean([r['xyz_mean_mm'] for r in val]))
                choices.append((obj,alpha,coef))
            obj,alpha,coef=min(choices,key=lambda r:r[0]);fitted[(seed,arm)]=coef
            np.savez_compressed(MODEL_DIR/f'probe_{seed}_{arm}.npz',coef=coef,seed=seed)
            selection.append({'seed':seed,'arm':arm,'alpha':alpha,'validation_xyz_mm':obj})
            print('inverse probe',seed,arm,round(obj,4),flush=True)
    dump(OUT/'augmentation_selection.json',{'architecture':'48 causal magnetic features -> fixed random tanh 192 + linear skip -> ridge XYZ',
        'deployed_decoder':False,'contact_gate_included':False,'source_validation_only':True,'selections':selection})
    tests=[load_row(r) for r in inv if r['role']=='test'];results=[]
    for a in tests:
        if not a['primary'].any():continue
        for seed in (361,362,363):
            xx=hidden_features(magnetic_features(a['b'],a['reset']),seed)
            row={k:a[k] for k in ('id','device','mode')};row['seed']=seed
            for arm in ('real','synthetic','mixed'):
                row[arm]=score_prediction(xx@fitted[(seed,arm)]*np.array([6,6,3]),a)
            results.append(row)
    macro={}
    for device in sorted(set(a['device'] for a in tests)):
        ss=[r for r in results if r['device']==device]
        macro[device]={arm:{k:float(np.mean([r[arm][k] for r in ss])) for k in
                              ('xy_mean_mm','z_mae_mm','xyz_mean_mm','xy_translation_removed_mm','z_translation_removed_mm')}
                       for arm in ('real','synthetic','mixed')}
    dump(OUT/'augmentation_report.json',{'test':results,'macro_by_device':macro,'seeds':[361,362,363],
        'macro':'equal trial, equal seed; seeds are not independent physical experiments',
        'scope':'capacity-matched research readout probe, NOT full deployed decoder or contact recognition',
        'synthetic_only_note':'supervised fitting uses only synthetic pairs; alpha is selected using real source validation',
        'target_fit_or_selection':False,'waypoint_fit_or_selection':False})
    print(json.dumps(macro,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
