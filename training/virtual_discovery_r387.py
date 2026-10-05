"""Finite virtual-only representation experiment; no production/real holdout writes.

Matched direct XYZ neural readouts use current field and optional causal EMA
residuals, never commanded motion or hidden generator states as input. Whole
virtual geometries are disjoint by role. This is a simulator research probe,
not a qualified replacement for the full magnetic-map/permission algorithm.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from training.train_source_to_plateau_r372 import Plateau

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/r387_virtual_discovery_20261002'
OLD=ROOT/'analysis/r370_broad_xyz_training_20261001'
GEN=ROOT/'analysis/r363_complex_forward_20261001/models/generator.json'
ARMS=('current','short_history','multiscale_history')
COUNTS={'train':32,'validation':16,'test':64}
TAUS=(.3,1.,4.)
YSCALE=np.array([5.,5.,2.],np.float32)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def path(role,i):return OLD/'virtual'/f'V370_{role}_{i:04d}.npz'

def representation(b,t,reset,arm):
    """Every state is computed on the full prefix; no future motion is read."""
    b=np.asarray(b,float);t=np.asarray(t,float);reset=np.asarray(reset,bool)
    if b.shape!=(len(t),15) or np.any(np.diff(t)<=0):raise ValueError('Invalid field/time')
    parts=[b];state=np.tile(b[0],(3,1));lag=np.zeros((len(t),45))
    for i in range(len(t)):
        if reset[i] or (i and t[i]-t[i-1]>.5):state[:]=b[i]
        elif i:
            rho=np.exp(-(t[i]-t[i-1])/np.array(TAUS))[:,None]
            state=rho*state+(1-rho)*b[i]
        lag[i]=(b[i]-state).reshape(-1)
    if arm=='current':lag[:]=0
    elif arm=='short_history':lag[:,15:]=0
    elif arm!='multiscale_history':raise ValueError(arm)
    return np.column_stack((b,lag)).astype(np.float32)

def kinematics(q,t):
    dt=np.r_[.1,np.diff(t)];v=np.vstack((np.zeros(3),np.diff(q,axis=0)))/dt[:,None]
    speed=np.linalg.norm(v,axis=1);prev=np.vstack((np.zeros(3),v[:-1]))
    moving=(speed>.05)&(np.linalg.norm(prev,axis=1)>.05)
    cosine=np.sum(v*prev,axis=1)/np.maximum(speed*np.linalg.norm(prev,axis=1),1e-10)
    recent_turn=np.zeros(len(q),bool)
    turns=np.flatnonzero(moving&(cosine<0.5))
    for i in turns:recent_turn[i:min(i+11,len(q))]=True
    dwell=speed<.02
    return {'moving':speed>.05,'dwell':dwell,'turn_or_reverse_recent_1s':recent_turn,
            'coupled_xyz':(np.linalg.norm(v[:,:2],axis=1)>.05)&(np.abs(v[:,2])>.03)}

def load(role,arm):
    arrays=[]
    for i in range(COUNTS[role]):
        with np.load(path(role,i),allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        x=representation(a['b'],a['t'],a['reset'],arm)
        m=a['primary']&np.isfinite(a['q']).all(1)&(a['q'][:,2]>=.45)
        arrays.append({'id':f'V370_{role}_{i:04d}','x':x[m],'q':a['q'][m].astype(np.float32),
          'masks':{k:v[m] for k,v in kinematics(a['q'],a['t']).items()},'indices':np.flatnonzero(m)})
    return arrays

def protocol():
    OUT.mkdir(parents=True,exist_ok=True)
    meta=json.loads(GEN.read_text(encoding='utf8'))['metadata']
    assert meta['target_used'] is False and meta['waypoint_used'] is False
    rows=[{'role':role,'id':f'V370_{role}_{i:04d}','file':path(role,i).relative_to(ROOT).as_posix(),
           'sha256':sha(path(role,i))} for role,n in COUNTS.items() for i in range(n)]
    p={'schema':'R387-VirtualDiscovery-1','hypothesis':'causal multiscale field representation reduces novel virtual whole-geometry XYZ error versus current and short-only representation',
      'arms':list(ARMS),'virtual_geometries_by_role':COUNTS,'seed':7,'architecture':'identical 60->32 tanh->3 XYZ; all 2051 trainable parameters and initialization identical',
      'representation':'current15 + (B_t-EMA_tau(B))15 at .3/1/4 s; disabled slots zero; prefix causal full Trial',
      'input_excludes':['XYZ commands','velocities','generator latent states','real measurements','future frames'],
      'normalization':'each input slot mean/std from virtual TRAIN only; target fixed units[5,5,2]mm',
      'loss':'equal-geometry normalized XYZ MSE, all eligible frames visited exactly once per epoch',
      'selection':'equal-geometry raw XYZ mean error on 16 virtual VAL whole geometries every epoch; same rules each arm',
      'max_epochs':1800,'batch_size':512,'initial_lr':.001,'minimum_lr':1e-6,'patience':50,'factor':.5,'relative_min_delta':1e-4,'l2':1e-4,
      'test':'64 independent-seed virtual geometries opened for metrics only AFTER all arm freezes',
      'predeclared_condition_masks':{'moving':'speed>.05 mm/s','dwell':'speed<.02 mm/s','turn_or_reverse_recent_1s':'moving heading cosine<.5, then1s causal tail','coupled_xyz':'XYspeed>.05 and absZspeed>.03 mm/s'},
      'generator_sha256':sha(GEN),'catalog_sha256':sha(OLD/'catalog.json'),'script_sha256':sha(__file__),
      'real_fit_rows':0,'Yellow_fit_or_selection_rows':0,'real_complex_fit_or_selection_rows':0,
      'production_changed':False,'source_generator_physics_validated':False,'inference_permission_gate':False,
      'boundary':'virtual-only direct XYZ research readout within teacher-supported contact; no live performance or physical-law claim'}
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==p
    else:dump(dest,p)
    manifest=OUT/'input_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text(encoding='utf8'))==rows
    else:dump(manifest,rows)
    return p

def initial():
    rng=np.random.default_rng(7)
    return [rng.normal(0,1/np.sqrt(60),(60,32)).astype(np.float32),np.zeros(32,np.float32),
            rng.normal(0,1/np.sqrt(32),(32,3)).astype(np.float32),np.zeros(3,np.float32)]
def predict(params,x):return np.tanh(x@params[0]+params[1])@params[2]+params[3]
def metrics(pred,q):
    e=pred-q;off=e.mean(0)
    return {'xyz_mean_mm':float(np.linalg.norm(e,axis=1).mean()),'xy_mean_mm':float(np.linalg.norm(e[:,:2],axis=1).mean()),
      'z_mae_mm':float(np.abs(e[:,2]).mean()),'bias_xyz_mm':off.tolist(),
      'translation_removed_xyz_mean_mm':float(np.linalg.norm(e-off,axis=1).mean()),'frames':len(q)}
def validation(params,rows,mean,scale):
    return float(np.mean([metrics(predict(params,(a['x']-mean)/scale)*YSCALE,a['q'])['xyz_mean_mm'] for a in rows]))
def save_model(p,params,mean,scale):np.savez_compressed(p,**{f'p{i}':v for i,v in enumerate(params)},mean=mean,scale=scale,yscale=YSCALE)

def train(arm):
    p=protocol();folder=OUT/arm;folder.mkdir(exist_ok=True)
    if (folder/'freeze.json').exists():return
    if (folder/'last_state.npz').exists():raise RuntimeError('Existing incomplete run; no silent restart')
    rows=load('train',arm);vals=load('validation',arm)
    x=np.vstack([a['x'] for a in rows]);y=np.vstack([a['q'] for a in rows])/YSCALE
    n=len(x);w=np.concatenate([np.full(len(a['x']),n/len(rows)/len(a['x']),np.float32) for a in rows])
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-3);u=(x-mean)/scale
    params=initial();best=[v.copy() for v in params];m=[np.zeros_like(v) for v in params];s=[v.copy() for v in m]
    initial_sha=hashlib.sha256(b''.join(v.tobytes() for v in params)).hexdigest()
    init=validation(params,vals,mean,scale);best_metric=init;best_epoch=0;updates=0;history=[];ctrl=Plateau(.001)
    rng=np.random.default_rng(7);stop='max_epochs_budget_truncated'
    dump(folder/'training_inputs.json',{'samples':n,'whole_trial_ids':[a['id'] for a in rows],
      'validation_ids':[a['id'] for a in vals],'initial_parameter_bytes_sha256':initial_sha,
      'input_normalization_sha256':hashlib.sha256(mean.tobytes()+scale.tobytes()).hexdigest(),'validation_initial_xyz_mm':init})
    for epoch in range(1,p['max_epochs']+1):
        begin=time.perf_counter();order=rng.permutation(n);lr=ctrl.lr
        for ids in np.array_split(order,math.ceil(n/p['batch_size'])):
            xx=u[ids];h=np.tanh(xx@params[0]+params[1]);pred=h@params[2]+params[3]
            d=2*(pred-y[ids])*w[ids,None]/len(ids);dh=(d@params[2].T)*(1-h*h)
            g=[xx.T@dh+p['l2']*params[0],dh.sum(0),h.T@d+p['l2']*params[2],d.sum(0)];updates+=1
            for a,grad,mi,si in zip(params,g,m,s):
                mi*=.9;mi+=.1*grad;si*=.999;si+=.001*grad*grad
                a-=lr*(mi/(1-.9**updates))/(np.sqrt(si/(1-.999**updates))+1e-8)
        val=validation(params,vals,mean,scale)
        if val<best_metric:best_metric=val;best_epoch=epoch;best=[v.copy() for v in params];save_model(folder/'best_model.npz',best,mean,scale)
        event=ctrl.observe(val);loss=float(np.mean(np.sum((predict(params,u)-y)**2,axis=1)*w))
        record={'epoch':epoch,'optimizer_updates':updates,'unique_samples_visited':n,'train_weighted_mse':loss,
          'validation_xyz_mean_mm':val,'best_validation_xyz_mean_mm':best_metric,'best_epoch':best_epoch,
          'lr_used':lr,'next_lr':ctrl.lr,'scheduler_event':event,'epoch_seconds':time.perf_counter()-begin}
        history.append(record)
        state={'epoch':epoch,'updates':updates,'rng_state':rng.bit_generator.state,'controller':asdict(ctrl),
          'history':history,'protocol_sha256':sha(OUT/'protocol.json'),'best_epoch':best_epoch,'best_metric':best_metric}
        np.savez_compressed(folder/'last_state.npz',**{f'p{i}':v for i,v in enumerate(params)},
          **{f'm{i}':v for i,v in enumerate(m)},**{f'v{i}':v for i,v in enumerate(s)},**{f'best{i}':v for i,v in enumerate(best)},
          state_json=json.dumps(state),mean=mean,scale=scale)
        dump(folder/'history.json',history);dump(folder/'status.json',{'status':'TRAINING','pid':os.getpid(),'arm':arm,**record})
        if epoch%25==0 or event!='continue':print(arm,epoch,round(val,5),best_epoch,lr,event,flush=True)
        if event=='early_stop_min_lr_plateau':stop=event;break
    save_model(folder/'best_model.npz',best,mean,scale)
    freeze={'status':'COMPLETED_PLATEAU' if stop=='early_stop_min_lr_plateau' else 'BUDGET_REACHED_NOT_CONVERGED',
      'arm':arm,'epochs_run':epoch,'best_epoch':best_epoch,'optimizer_updates':updates,'training_samples_per_epoch':n,
      'stop_reason':stop,'final_lr':ctrl.lr,'lr_reductions':ctrl.reductions,'final_bad_epochs':ctrl.bad_epochs,
      'best_validation_xyz_mean_mm':best_metric,'initial_xyz_mean_mm':init,'model_sha256':sha(folder/'best_model.npz'),
      'initial_parameter_bytes_sha256':initial_sha,'protocol_sha256':sha(OUT/'protocol.json'),'test_opened':False,'production_changed':False}
    dump(folder/'freeze.json',freeze);dump(folder/'status.json',{'pid':os.getpid(),**freeze})

def evaluate():
    p=protocol();freezes={a:json.loads((OUT/a/'freeze.json').read_text(encoding='utf8')) for a in ARMS}
    for a,f in freezes.items():
        assert f['status']=='COMPLETED_PLATEAU','No test of unfinished optimization'
        assert f['model_sha256']==sha(OUT/a/'best_model.npz')
    assert len({f['initial_parameter_bytes_sha256'] for f in freezes.values()})==1
    result=[];allpred={}
    for arm in ARMS:
        with np.load(OUT/arm/'best_model.npz',allow_pickle=False) as f:
            params=[f[f'p{i}'].copy() for i in range(4)];mean=f['mean'];scale=f['scale']
        rr=load('test',arm);armrows=[];allpred[arm]=[]
        for a in rr:
            pred=predict(params,(a['x']-mean)/scale)*YSCALE
            row={'id':a['id'],'style':('bezier_space','inertial_walk','free_waypoints','multiscale_curve')[int(a['id'][-4:])%4],
              **metrics(pred,a['q']),'conditions':{k:metrics(pred[m],a['q'][m]) for k,m in a['masks'].items() if m.any()}}
            armrows.append(row);allpred[arm].append((pred,a['q'],a['indices']))
        result.append({'arm':arm,'macro':{k:float(np.mean([r[k] for r in armrows])) for k in ('xyz_mean_mm','xy_mean_mm','z_mae_mm','translation_removed_xyz_mean_mm')},
          'by_style':{style:float(np.mean([r['xyz_mean_mm'] for r in armrows if r['style']==style])) for style in sorted({r['style'] for r in armrows})},
          'by_condition':{k:{'trial_macro_xyz_mm':float(np.mean([r['conditions'][k]['xyz_mean_mm'] for r in armrows if k in r['conditions']])),
                            'eligible_trials':sum(k in r['conditions'] for r in armrows),'frames':sum(r['conditions'].get(k,{}).get('frames',0) for r in armrows)} for k in ('moving','dwell','turn_or_reverse_recent_1s','coupled_xyz')},'trials':armrows})
    improvements=[];base=result[0]
    for row in result[1:]:
        improvements.append({'arm':row['arm'],'raw_xyz_change_percent':100*(row['macro']['xyz_mean_mm']/base['macro']['xyz_mean_mm']-1),
          'better_test_trials':sum(b['xyz_mean_mm']<a['xyz_mean_mm'] for a,b in zip(base['trials'],row['trials'])),'total_test_trials':64})
    dump(OUT/'report.json',{'protocol_sha256':sha(OUT/'protocol.json'),'freezes':freezes,'test_roles_opened_after_all_freezes':True,
      'models':result,'change_vs_current':improvements,'scope':'virtual-only directXYZ research readout; not production whole chain; no real test opened',
      'contact_coverage':'all teacher-supported nominal contact frames, no model permission gate or inferred contact recall',
      'real_transfer_proven':False,'physical_discovery_proven':False,'selected_based_on_test':False})
    np.savez_compressed(OUT/'example_test_predictions.npz',q=allpred['current'][0][1],indices=allpred['current'][0][2],
      **{arm:allpred[arm][0][0] for arm in ARMS})
    print(json.dumps(improvements,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('train','evaluate','all'),default='all');ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.phase in ('train','all'):
        for arm in ([a.arm] if a.arm else ARMS):train(arm)
    if a.phase in ('evaluate','all'):evaluate()
if __name__=='__main__':main()
