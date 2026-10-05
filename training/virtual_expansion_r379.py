"""Fully traversed rich-trajectory XY expansion from frozen R373.

Uses the already generated R370 free-XYZ magnetic trials, not real test paths.
Their features are recaptured through the actual R373 chain. The baseline is
frozen, not an extra real-only training arm. Z is deliberately not retrained
here: it must be recaptured with the selected XY before the later Z experiment.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import argparse,json,os,time,sys
import numpy as np
from training.train_source_to_plateau_r372 import Plateau,forward,backward,checkpoint,save_model,dump,sha,val_xy
from training.domain_randomization_r369 import ResearchXY,KEYS
from training.run_forward_r361 import ROOT
sys.path.insert(0,str(ROOT/'code'))
from live_retrained_reconstruction_r373 import DirectTrainedEngineR373,PlateauCausalXY,COMPONENTS
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from live_algorithm_ui_r126 import PROFILE_KEY
from replay_backtest import load_profile

OUT=ROOT/'analysis/r379_full_virtual_expansion_20261002'
OLD=ROOT/'analysis/r370_broad_xyz_training_20261001'
REAL=ROOT/'analysis/r372_direct_source_plateau_20261001'
GEN=ROOT/'analysis/r363_complex_forward_20261001/models/generator.json'

def weighted_gradients(params,x,y,weight):
    pred=forward(params,x)[2]
    g=backward(params,x,2*(pred-y)*weight[:,None]/len(x))
    for i in (0,2,4):g[i]+=.001*params[i]
    return g

def capture(index):
    id=f'V370_train_{index:04d}';dest=OUT/'features'/f'{id}.npz';source=OLD/'virtual'/f'{id}.npz'
    if dest.exists():
        with np.load(dest,allow_pickle=False) as f:n=int(f['samples'])
        return {'id':id,'source_sha256':sha(source),'features_sha256':sha(dest),'samples':n}
    with np.load(source,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    profile=load_profile(R92_PROFILE_PATH,PROFILE_KEY)
    e=DirectTrainedEngineR373(profile,profile['defaults']['threshold'],profile['defaults']['min_cosine'],profile['defaults']['smoothing_ms'],70.)
    main=ResearchXY(PlateauCausalXY());provisional=ResearchXY(PlateauCausalXY())
    e.memory_decoder.xy_model=main;e.candidate_decoder.xy_model=provisional;e.set_baseline(np.zeros(15))
    xx=[];yy=[];guard=[];q=[];pred=[];ids=[]
    for i,now in enumerate(a['t']):
        if a['reset'][i]:e.set_baseline(np.zeros(15))
        main.last=provisional.last=None
        r=e.push(int(round(now*1e9))+10_000_000_000,a['b'][i],eligible=bool(a['valid'][i]))
        if not (r['valid'] and a['primary'][i]):continue
        chosen=provisional if r['extra']['localization_path']=='provisional_permission' else main
        if chosen.last is None:raise ValueError('Accepted output lacks same-frame XY features')
        x=chosen.last.copy();g=x[:2]*10
        np.testing.assert_allclose(g+chosen.predict(x[None])[0],np.asarray(r['xyz'])[:2],atol=1e-9)
        xx.append(x);yy.append(np.clip(a['q'][i,:2]-g,-3,3));guard.append(g);q.append(a['q'][i,:2]);pred.append(np.asarray(r['xyz'])[:2]);ids.append(i)
    assert len(xx)>0
    np.savez_compressed(dest,x=np.array(xx),y=np.array(yy),guard=np.array(guard),q=np.array(q),initial_pred=np.array(pred),indices=np.array(ids),samples=len(xx))
    return {'id':id,'source_sha256':sha(source),'features_sha256':sha(dest),'samples':len(xx)}

def protocol(count):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'features').mkdir(exist_ok=True)
    p={'schema':'R379-FullTraversal-VirtualExpansion-1','component':'XY first; not a completed combined XYZ retraining',
       'virtual_count':count,'generated_paths':'four R370 free-XYZ procedural families; no measured trajectory templates',
       'feature_capture':'actual R373 full causal chain, both main and provisional XY routes; recaptured rather than old R133 cache',
       'architecture':'unchanged 53-64-64-2 tanh; frozen R373 initial weights and normalization',
       'seed':7,'initial_lr':.0002,'min_lr':1e-6,'patience':50,'factor':.5,'relative_min_delta':1e-4,
       'max_epochs_initial_budget':1000,'true_epoch':'shuffled complete traversal without replacement of every real/static/virtual sample',
       'loss_mass':{'original_source_dynamic':.4,'virtual_dynamic':.4,'static_anchors':.2},
       'virtual_weighting':'each complete synthetic trajectory receives equal loss mass',
       'source_weighting':'original source dynamic samples share their mass uniformly; original static anchors share theirs uniformly',
       'validation':'original three R372 source-only XY whole-Trial splits and raw equal-Trial error, evaluated every epoch',
       'baseline_control':'frozen R373, zero further updates, no repeated real-only training arm',
       'target_fit_or_selection':False,'latest_R373_real_used_in_fit_or_selection':False,
       'post_freeze':'real full-chain replay, coverage/offset/Z support/jumps; do not assume downstream invariance',
       'z_next':'recapture Z candidate-dependent features with selected XY before optional incremental-Z training',
       'hashes':{'generator':sha(GEN),'virtual_catalog':sha(OLD/'catalog.json'),'initial_xy':COMPONENTS['xy']['model_sha256'],'real_manifest':sha(REAL/'data_manifest.json')}}
    dest=OUT/f'protocol_{count}.json'
    if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==p
    else:dump(dest,p)
    return p

def prepare(count,workers):
    p=protocol(count);folder=OUT/f'arm_{count}';folder.mkdir(exist_ok=True)
    dump(folder/'status.json',{'status':'PREPARING','pid':os.getpid(),'count':count})
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(capture,range(count)):
            rows.append(r)
            if len(rows)%10==0:print('captured',len(rows),'/',count,flush=True)
    dump(folder/'feature_manifest.json',{'rows':rows,'samples':sum(r['samples'] for r in rows),'baseline_xy_sha256':COMPONENTS['xy']['model_sha256']})
    dump(folder/'status.json',{'status':'PREPARED','pid':os.getpid(),'samples':sum(r['samples'] for r in rows)})

def train(count,resume,max_epochs):
    p=protocol(count);folder=OUT/f'arm_{count}';manifest=json.loads((folder/'feature_manifest.json').read_text(encoding='utf8'))
    if (folder/'freeze.json').exists():
        previous=json.loads((folder/'freeze.json').read_text(encoding='utf8'))
        if not (resume and previous['status']=='BUDGET_REACHED_NOT_CONVERGED' and max_epochs>previous['epochs_run']):raise FileExistsError('Preserve frozen run')
    with np.load(REAL/'inputs/xy_train.npz',allow_pickle=False) as f:r={k:f[k].copy() for k in f.files}
    assert len(r['x'])==7782
    # Original preparation appends 565 static anchors after all dynamic data.
    xs=[r['x']];ys=[r['y']];w=[np.r_[np.full(7217,.4/7217),np.full(565,.2/565)]]
    for entry in manifest['rows']:
        path=OUT/'features'/f"{entry['id']}.npz";assert sha(path)==entry['features_sha256']
        with np.load(path,allow_pickle=False) as f:xs.append(f['x'].copy());ys.append(f['y'].copy())
        w.append(np.full(entry['samples'],.4/count/entry['samples']))
    x=np.vstack(xs);y=np.vstack(ys);weight=np.concatenate(w);assert np.isclose(weight.sum(),1)
    model=PlateauCausalXY();mean=model.mean.copy();scale=model.scale.copy();params=[getattr(model,k).copy() for k in KEYS[2:]]
    u=(x-mean)/scale;weight*=len(u)
    data=json.loads((REAL/'data_manifest.json').read_text(encoding='utf8'))['xy'];vals=[]
    assert np.all(r['y'][-565:]==0), 'Original static anchor layout changed'
    for entry in data['files']:assert sha(REAL/entry['file'])==entry['sha256']
    for entry in data['validation']:
        with np.load(REAL/entry['file'],allow_pickle=False) as f:vals.append((entry,{k:f[k].copy() for k in f.files}))
    baseline,_,_=val_xy(params,vals,mean,scale);rng=np.random.default_rng(7);ctrl=Plateau(.0002)
    first=[np.zeros_like(v) for v in params];second=[np.zeros_like(v) for v in params]
    history=[];updates=0;start=0;best_params=[v.copy() for v in params];best=baseline;best_epoch=0
    statefile=folder/'last_state.npz'
    if resume:
        with np.load(statefile,allow_pickle=False) as f:
            state=json.loads(str(f['state_json']));assert state['protocol_sha256']==sha(OUT/f'protocol_{count}.json') and state['feature_manifest_sha256']==sha(folder/'feature_manifest.json')
            params=[f[f'p{i}'].copy() for i in range(6)];first=[f[f'm{i}'].copy() for i in range(6)];second=[f[f'v{i}'].copy() for i in range(6)];best_params=[f[f'best{i}'].copy() for i in range(6)]
        ctrl=Plateau(**state['controller']);rng.bit_generator.state=state['rng_state'];history=state['history'];updates=state['updates'];start=state['epoch'];best=state['best_metric'];best_epoch=state['best_epoch']
    elif statefile.exists():raise FileExistsError('Existing checkpoint, explicitly resume')
    stop='max_epochs_budget_truncated'
    for epoch in range(start+1,max_epochs+1):
        begin=time.perf_counter();order=rng.permutation(len(u));lr=ctrl.lr
        for ids in np.array_split(order,(len(u)+255)//256):
            g=weighted_gradients(params,u[ids],y[ids],weight[ids])
            updates+=1
            for a,grad,m,v in zip(params,g,first,second):
                m*=.9;m+=.1*grad;v*=.999;v+=.001*grad*grad;a-=lr*(m/(1-.9**updates))/(np.sqrt(v/(1-.999**updates))+1e-8)
        obj,_,vrows=val_xy(params,vals,mean,scale)
        if obj<best:best=obj;best_epoch=epoch;best_params=[v.copy() for v in params];save_model(folder/'best_model.npz',best_params,mean,scale,'xy')
        event=ctrl.observe(obj);pred=forward(params,u)[2];loss=float(np.mean(np.sum((pred-y)**2,axis=1)*weight))
        record={'epoch':epoch,'updates':updates,'train_weighted_mse':loss,'validation_objective':obj,'baseline_validation_objective':baseline,
            'unique_samples_visited':int(len(order)),'training_samples':len(u),'lr_used':lr,'next_lr':ctrl.lr,'best_epoch':best_epoch,'best_objective':best,
            'scheduler_event':event,'epoch_seconds':time.perf_counter()-begin}
        history.append(record)
        state={'epoch':epoch,'updates':updates,'history':history,'controller':asdict(ctrl),'rng_state':rng.bit_generator.state,'best_metric':best,'best_epoch':best_epoch,
            'protocol_sha256':sha(OUT/f'protocol_{count}.json'),'feature_manifest_sha256':sha(folder/'feature_manifest.json')}
        checkpoint(statefile,params,first,second,best_params,state)
        dump(folder/'status.json',{'status':'TRAINING','pid':os.getpid(),**record})
        if epoch%10==0 or event!='continue':print(count,epoch,obj,best_epoch,lr,event,flush=True)
        if event=='early_stop_min_lr_plateau':stop=event;break
    save_model(folder/'best_model.npz',best_params,mean,scale,'xy')
    dump(folder/'freeze.json',{'status':'COMPLETED_PLATEAU' if stop=='early_stop_min_lr_plateau' else 'BUDGET_REACHED_NOT_CONVERGED','component':'XY only',
        'epochs_run':history[-1]['epoch'],'best_epoch':best_epoch,'baseline_validation_objective':baseline,'best_validation_objective':best,'optimizer_updates':updates,
        'final_learning_rate':ctrl.lr,'lr_reductions':ctrl.reductions,'stop_reason':stop,'training_samples':len(u),'virtual_samples':manifest['samples'],
        'model_sha256':sha(folder/'best_model.npz'),'feature_manifest_sha256':sha(folder/'feature_manifest.json'),
        'target_fit_or_selection':False,'test_opened':False,'production_replaced':False,'full_xyz_retrained':False})
    dump(folder/'history.json',history)
    dump(folder/'status.json',{'status':'COMPLETED_PLATEAU' if stop=='early_stop_min_lr_plateau' else 'BUDGET_REACHED_NOT_CONVERGED',
        'pid':os.getpid(),'epochs_run':history[-1]['epoch'],'best_epoch':best_epoch,'best_validation_objective':best,
        'optimizer_updates':updates,'component':'XY only','production_replaced':False})
    print('freeze',count,stop,'best',best_epoch,best,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('prepare','train','all'),default='all');ap.add_argument('--count',type=int,choices=(100,500,1000),default=100);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--resume',action='store_true');ap.add_argument('--max-epochs',type=int,default=1000);a=ap.parse_args()
    if a.phase in ('prepare','all'):prepare(a.count,a.workers)
    if a.phase in ('train','all'):train(a.count,a.resume,a.max_epochs)
if __name__=='__main__':main()
