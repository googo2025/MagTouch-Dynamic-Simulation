"""Train-pool hard mining and independently generated trajectory evaluation.

Same R373 XY head, initialization and normalization. No Yellow/new real test
labels in mining or selection; no production writes. Server-only computation.
All epoch counts mean full traversal. Fixed-feature and full-chain evidence are
separate: the former is not silently advertised as an operational XYZ result.
"""
from __future__ import annotations
import argparse, json, os, socket, sys, time, math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
from training.train_source_to_plateau_r372 import Plateau, forward, backward, dump, sha, save_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/r397_virtual_breakthrough_20261003'
REAL=ROOT/'analysis/r372_direct_source_plateau_20261001'
POOL=ROOT/'analysis/r379_full_virtual_expansion_20261002/features'
GEN=ROOT/'analysis/r363_complex_forward_20261001/models/generator.json'
NOISE=ROOT/'analysis/r369_source_randomized_augmentation_20261001/source_parameter_distribution.json'
KEYS=('w1','b1','w2','b2','w3','b3')
ARMS=('common64','uniform128','hard128')
STYLES=('bezier_space','inertial_walk','free_waypoints','multiscale_curve')


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def server_only():
    if socket.gethostname()!='googo-Super-Server':raise RuntimeError('Research computation runs on the authorized server only')


def protocol():
    p=dict(schema='MagTouch-R397-TargetedVirtualExpansion-1',
        purpose='Locate existing R373 weaknesses in virtual TRAIN; compare ordinary and targeted expansion on new whole geometries',
        arms=list(ARMS),optimizer_seed=7,initial_xy_sha256=sha(REAL/'xy/best_model.npz'),
        generator_sha256=sha(GEN),source_manifest_sha256=sha(REAL/'data_manifest.json'),
        architecture='unchanged 53-64-64-2 tanh XY residual; both full-chain XY routes replaced only in research replay',
        normalization='frozen original R373 input mean/scale',
        pool_role='original V370_train_0000..0999 only; never old validation/test paths',
        mining_score='whole-trajectory original R373 raw XY mean; centered XY also diagnostic; 16 highest per construction family',
        common_train_ids='original first 64 train geometries, 16/family',
        uniform_rule='seed3977 select16/family from remaining TRAIN pool',
        hard_rule='top16/family from remaining TRAIN pool by baseline error; no candidate/test outcomes',
        new_validation_geometries=32,new_test_geometries=64,
        new_seed_bases=dict(validation=3972000,test=3973000),
        generation='R370 bounded procedural generator; independently seeded whole geometry, new noise realization',
        source_loss_mass=.4,virtual_loss_mass=.4,static_loss_mass=.2,
        loss='equal virtual-trajectory weighted clipped residual-label MSE; original source labels unchanged',
        initial_lr=.0002,min_lr=1e-6,patience=50,relative_min_delta=1e-4,lr_factor=.5,
        batch_size=256,l2_weights=.001,max_epochs=2000,true_epoch='every training frame once, without replacement',
        selection='equal average of source-only validation ratio and new virtual validation ratio; no test data',
        source_validation_guard='source macro <=1.05 baseline and each source validation Trial <=1.15 baseline',
        anchor_guard='static head RMS <=max(.12mm,1.2*baseline RMS)',
        scheduler='all validation ratios, not test; minimumLR then50 consecutive plateau epochs',
        backend='torch float64 Adam on one GPU; NumPy forward/gradient parity checked before training',
        test='all arms must finish plateau and freeze before any test scoring; all64 test geometries report',
        hard_test_subset='predeclared top baseline-error quartile within each of four test families; includes all such paths, not candidate-picked',
        interpretation='simulator-internal new geometry evidence first; real migration is an additional distinct test',
        real_test_fit_or_selection=False,Yellow_fit_or_selection=False,production_changed=False,hardware_operated=False)
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'protocol.json'
    if dest.exists():assert read(dest)==p,'Do not overwrite preregistration'
    else:dump(dest,p)
    return p


def mine():
    p=protocol();rows=[]
    for i in range(1000):
        path=POOL/f'V370_train_{i:04d}.npz';a=load(path)
        error=a['initial_pred']-a['q'];bias=error.mean(0)
        rows.append(dict(id=f'V370_train_{i:04d}',index=i,style=STYLES[i%4],
            samples=len(error),raw_xy_mm=float(np.linalg.norm(error,axis=1).mean()),
            centered_xy_mm=float(np.linalg.norm(error-bias,axis=1).mean()),bias_xy_mm=bias.tolist(),
            feature_sha256=sha(path)))
    common=[r for r in rows if r['index']<64];remaining=[r for r in rows if r['index']>=64]
    rng=np.random.default_rng(3977);uniform=[];hard=[]
    for style in STYLES:
        group=[r for r in remaining if r['style']==style]
        uniform.extend(group[int(i)] for i in rng.choice(len(group),16,replace=False))
        hard.extend(sorted(group,key=lambda r:r['raw_xy_mm'],reverse=True)[:16])
    by_arm=dict(common64=common,uniform128=common+uniform,hard128=common+hard)
    report=dict(protocol_sha256=sha(OUT/'protocol.json'),pool_count=1000,
        source_only=True,no_test_used=True,rows=rows,
        arms={name:dict(ids=[r['id'] for r in group],paths=len(group),
            baseline_macro_xy_mm=float(np.mean([r['raw_xy_mm'] for r in group])),
            baseline_centered_macro_xy_mm=float(np.mean([r['centered_xy_mm'] for r in group])),
            samples=sum(r['samples'] for r in group)) for name,group in by_arm.items()},
        added_group_baseline_xy_mm=dict(uniform=float(np.mean([r['raw_xy_mm'] for r in uniform])),
            hard=float(np.mean([r['raw_xy_mm'] for r in hard]))))
    dump(OUT/'mining.json',report)
    print('TRAIN-POOL MINING',json.dumps(report['added_group_baseline_xy_mm']),flush=True)
    return report


def capture(job):
    role,index=job;id=f'V397_{role}_{index:04d}'
    dest=OUT/'features'/f'{id}.npz';stream=OUT/'virtual'/f'{id}.npz'
    if dest.exists():return dict(id=id,role=role,seed={'validation':3972000,'test':3973000}[role]+index,
        style=STYLES[index%4],feature_sha256=sha(dest),stream_sha256=sha(stream),samples=int(load(dest)['samples']))
    # Normalize only historical Windows separators, never change source bytes.
    from training.broad_trajectory_r370 import geometry
    from training.forward_complex_r363 import AxisForward
    seed_offset={'validation':270000,'test':270000}[role]
    q,t,info=geometry(role,seed_offset+index)
    assert info['seed']=={'validation':3972000,'test':3973000}[role]+index
    g=AxisForward.load(GEN);sigma=np.array(read(NOISE)['sigma_uT'])
    b=g.rollout(q,t)+np.random.default_rng(info['seed']+400000).normal(size=(len(q),15))*sigma
    primary=(q[:,2]>=.4)&(q[:,2]<=3.)&g.support(q)
    sys.path.insert(0,str(ROOT/'code'))
    from live_retrained_reconstruction_r373 import DirectTrainedEngineR373,PlateauCausalXY
    from training.domain_randomization_r369 import ResearchXY
    from live_algorithm_ui_r92 import R92_PROFILE_PATH
    from live_algorithm_ui_r126 import PROFILE_KEY
    from replay_backtest import load_profile
    profile=load_profile(R92_PROFILE_PATH,PROFILE_KEY)
    e=DirectTrainedEngineR373(profile,profile['defaults']['threshold'],profile['defaults']['min_cosine'],profile['defaults']['smoothing_ms'],70.)
    main=ResearchXY(PlateauCausalXY());provisional=ResearchXY(PlateauCausalXY())
    e.memory_decoder.xy_model=main;e.candidate_decoder.xy_model=provisional;e.set_baseline(np.zeros(15))
    xx=[];yy=[];guard=[];qq=[];pred=[];ids=[];fullpred=np.full((len(t),3),np.nan);ok=np.zeros(len(t),bool)
    for i,now in enumerate(t):
        main.last=provisional.last=None
        r=e.push(int(round(now*1e9))+10_000_000_000,b[i],eligible=True)
        ok[i]=bool(r['valid'])
        if ok[i]:fullpred[i]=r['xyz']
        if not (ok[i] and primary[i]):continue
        chosen=provisional if r['extra']['localization_path']=='provisional_permission' else main
        if chosen.last is None:raise ValueError('Missing same-frame causal features')
        x=chosen.last.copy();gd=x[:2]*10
        np.testing.assert_allclose(gd+chosen.predict(x[None])[0],np.array(r['xyz'])[:2],atol=1e-9)
        xx.append(x);yy.append(np.clip(q[i,:2]-gd,-3,3));guard.append(gd);qq.append(q[i,:2]);pred.append(fullpred[i,:2]);ids.append(i)
    (OUT/'features').mkdir(exist_ok=True);(OUT/'virtual').mkdir(exist_ok=True)
    np.savez_compressed(stream,q=q,t=t,b=b,primary=primary,valid=np.ones(len(q),bool),
        reset=np.r_[True,np.zeros(len(t)-1,bool)],baseline_pred=fullpred,baseline_ok=ok)
    np.savez_compressed(dest,x=np.array(xx),y=np.array(yy),guard=np.array(guard),q=np.array(qq),
        initial_pred=np.array(pred),indices=np.array(ids),samples=len(xx))
    assert len(xx)>10 and np.isfinite(xx).all(), 'Supported causal features required'
    return dict(id=id,role=role,seed=info['seed'],style=info['style'],samples=len(xx),
        feature_sha256=sha(dest),stream_sha256=sha(stream))


def prepare():
    p=protocol();mine();rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(capture,[('validation',i) for i in range(p['new_validation_geometries'])]):
            rows.append(row);print('PREPARED NEW VALIDATION',len(rows),'/',len(rows)+p['new_validation_geometries']-len(rows),flush=True)
    dump(OUT/'validation_manifest.json',rows)
    dump(OUT/'work_state.json',dict(status='PREPARED_NEW_VALIDATION',protocol='protocol.json',
        next='Train all three matched XY arms; test still sealed',real_or_target_loaded=False,production_changed=False))


def numpy_params():
    a=load(REAL/'xy/best_model.npz')
    return [a[k] for k in KEYS],a['mean'],a['scale']


def train(arm,max_epochs=None):
    import torch
    p=protocol();folder=OUT/arm;folder.mkdir(exist_ok=True)
    effective_max=p['max_epochs'] if max_epochs is None else int(max_epochs)
    assert effective_max>=p['max_epochs'], 'Do not shorten the preregistered training budget'
    if (folder/'freeze.json').exists() and read(folder/'freeze.json')['status']=='COMPLETED_PLATEAU':return
    mining=read(OUT/'mining.json');real=load(REAL/'inputs/xy_train.npz')
    dynamic=7217;anchors=565;assert len(real['x'])==dynamic+anchors
    ids=mining['arms'][arm]['ids'];xs=[real['x']];ys=[real['y']]
    ww=[np.r_[np.full(dynamic,.4/dynamic),np.full(anchors,.2/anchors)]]
    for id in ids:
        a=load(POOL/f'{id}.npz');xs.append(a['x']);ys.append(a['y']);ww.append(np.full(len(a['x']),.4/len(ids)/len(a['x'])))
    x=np.vstack(xs);y=np.vstack(ys);weights=np.concatenate(ww)*len(x);n=len(x)
    source=[]
    for entry in read(REAL/'data_manifest.json')['xy']['validation']:
        source.append(load(REAL/entry['file']))
    virtual=[load(OUT/'features'/f"{r['id']}.npz") for r in read(OUT/'validation_manifest.json')]
    initial,mean,scale=numpy_params();norm=(x-mean)/scale
    torch.set_num_threads(2);torch.manual_seed(7)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':raise RuntimeError('Use an idle CUDA GPU, not overloaded shared CPU')
    params=[torch.nn.Parameter(torch.as_tensor(v.copy(),dtype=torch.float64,device=device)) for v in initial]
    def f(u):return torch.tanh(torch.tanh(u@params[0]+params[1])@params[2]+params[3])@params[4]+params[5]
    tx=torch.as_tensor(norm,dtype=torch.float64,device=device)
    ty=torch.as_tensor(y,dtype=torch.float64,device=device);tw=torch.as_tensor(weights,dtype=torch.float64,device=device)
    # Exact objective derivative check against the existing NumPy implementation.
    fi=f(tx[:32]);np.testing.assert_allclose(fi.detach().cpu(),forward(initial,norm[:32])[2],atol=1e-10,rtol=1e-10)
    checkloss=((fi-ty[:32]).square().sum(1)*tw[:32]).mean()+.0005*sum(v.square().sum() for v in params[::2])
    checkloss.backward();ng=backward(initial,norm[:32],2*(forward(initial,norm[:32])[2]-y[:32])*weights[:32,None]/32)
    for k in (0,2,4):ng[k]+=.001*initial[k]
    for v,g in zip(params,ng):np.testing.assert_allclose(v.grad.cpu(),g,atol=1e-9,rtol=1e-9)
    for v in params:v.grad=None
    dump(folder/'backend_parity.json',dict(status='PASSED',device=str(device),torch_version=torch.__version__,dtype='float64',numpy_forward_and_gradients_matched=True))
    val_rows=[(kind,torch.as_tensor((a['x']-mean)/scale,dtype=torch.float64,device=device),
        torch.as_tensor(a['guard'],device=device),torch.as_tensor(a['q'],device=device))
        for kind,rows in [('source',source),('virtual',virtual)] for a in rows]
    anchor=torch.as_tensor((real['x'][-anchors:]-mean)/scale,dtype=torch.float64,device=device)
    def score():
        with torch.no_grad():
            sr=[];vr=[]
            for kind,u,guard,q in val_rows:
                value=float(torch.linalg.vector_norm(guard+f(u)-q,dim=1).mean())
                (sr if kind=='source' else vr).append(value)
            rms=float(torch.sqrt(f(anchor).square().sum(1).mean()))
        return np.array(sr),np.array(vr),rms
    source0,virtual0,static0=score();base_source=float(source0.mean());base_virtual=float(virtual0.mean())
    optimizer=torch.optim.Adam(params,lr=p['initial_lr'],eps=1e-8)
    ctrl=Plateau(p['initial_lr']);rng=np.random.default_rng(7);history=[];updates=0;start_epoch=0
    best=[v.copy() for v in initial];bestobj=1.;best_epoch=0;checkpoint=folder/'last_state.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        assert state['protocol_sha256']==sha(OUT/'protocol.json') and state['mining_sha256']==sha(OUT/'mining.json')
        with torch.no_grad():
            for v,value in zip(params,state['params']):v.copy_(value)
        optimizer.load_state_dict(state['optimizer']);ctrl=Plateau(**state['controller'])
        rng.bit_generator.state=state['rng_state'];history=state['history'];updates=state['updates'];start_epoch=state['epoch']
        best=state['best'];bestobj=state['bestobj'];best_epoch=state['best_epoch']
    if start_epoch>=effective_max:raise RuntimeError('Unfinished cutoff: extend this same optimizer run, not a new epoch-comparison arm')
    stop='BUDGET_REACHED_NOT_CONVERGED'
    for epoch in range(start_epoch+1,effective_max+1):
        begin=time.monotonic();order=rng.permutation(n);lr=ctrl.lr
        for group in optimizer.param_groups:group['lr']=lr
        for batch in np.array_split(order,math.ceil(n/p['batch_size'])):
            ix=torch.as_tensor(batch,device=device)
            optimizer.zero_grad(set_to_none=True)
            loss=((f(tx[ix])-ty[ix]).square().sum(1)*tw[ix]).mean()+.0005*sum(v.square().sum() for v in params[::2])
            loss.backward();optimizer.step();updates+=1
        sr,vr,rms=score();obj=.5*(float(sr.mean())/base_source+float(vr.mean())/base_virtual)
        qualified=bool(sr.mean()<=1.05*base_source and np.all(sr<=1.15*source0) and rms<=max(.12,1.2*static0))
        if qualified and obj<bestobj:
            bestobj=obj;best_epoch=epoch;best=[v.detach().cpu().numpy().copy() for v in params]
            save_model(folder/'best_model.npz',best,mean,scale,'xy')
        event=ctrl.observe(obj)
        row=dict(epoch=epoch,updates=updates,unique_samples_visited=n,training_samples=n,lr_used=lr,next_lr=ctrl.lr,
            validation_source_xy_mm=float(sr.mean()),validation_virtual_xy_mm=float(vr.mean()),
            validation_source_per_trial=sr.tolist(),static_anchor_rms_mm=rms,objective=obj,qualified=qualified,
            best_epoch=best_epoch,best_qualified_objective=bestobj,scheduler_event=event,epoch_seconds=time.monotonic()-begin)
        history.append(row);dump(folder/'status.json',dict(status='TRAINING',pid=os.getpid(),arm=arm,**row))
        state=dict(epoch=epoch,updates=updates,history=history,params=[v.detach().cpu() for v in params],
            optimizer=optimizer.state_dict(),controller=asdict(ctrl),rng_state=rng.bit_generator.state,
            best=best,bestobj=bestobj,best_epoch=best_epoch,protocol_sha256=sha(OUT/'protocol.json'),mining_sha256=sha(OUT/'mining.json'))
        temporary=folder/'last_state.writing.pt';torch.save(state,temporary);temporary.replace(checkpoint)
        if epoch%25==0 or event!='continue':
            dump(folder/'history.json',history)
            print('TRAIN',arm,epoch,'source',round(float(sr.mean()),4),'virtual',round(float(vr.mean()),4),'best',best_epoch,event,flush=True)
        if event=='early_stop_min_lr_plateau':stop='COMPLETED_PLATEAU';break
    save_model(folder/'best_model.npz',best,mean,scale,'xy');dump(folder/'history.json',history)
    freeze=dict(status=stop,arm=arm,epochs=epoch,best_epoch=best_epoch,updates=updates,training_samples_per_epoch=n,
        virtual_paths=len(ids),model_sha256=sha(folder/'best_model.npz'),protocol_sha256=sha(OUT/'protocol.json'),
        baseline_source_xy_mm=base_source,baseline_virtual_xy_mm=base_virtual,best_qualified_objective=bestobj,
        min_lr=ctrl.lr,final_bad_epochs=ctrl.bad_epochs,lr_reductions=ctrl.reductions,
        effective_max_epochs=effective_max,selected_training_candidate=best_epoch>0,test_opened=False,production_changed=False)
    dump(folder/'freeze.json',freeze);dump(folder/'status.json',freeze)


def evaluate():
    p=protocol();freezes={arm:read(OUT/arm/'freeze.json') for arm in ARMS}
    for arm,f in freezes.items():
        assert f['status']=='COMPLETED_PLATEAU','Do not open test of unfinished optimization'
        assert f['model_sha256']==sha(OUT/arm/'best_model.npz')
    test=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for a in pool.map(capture,[('test',i) for i in range(p['new_test_geometries'])]):
            test.append(a);print('NEW TEST CAPTURE',len(test),'/',p['new_test_geometries'],flush=True)
    dump(OUT/'test_manifest.json',test)
    models={'R373':REAL/'xy/best_model.npz',**{arm:OUT/arm/'best_model.npz' for arm in ARMS}}
    rows=[]
    for entry in test:
        a=load(OUT/'features'/f"{entry['id']}.npz")
        scores={}
        for name,path in models.items():
            model=load(path);params=[model[k] for k in KEYS]
            pred=a['guard']+forward(params,(a['x']-model['mean'])/model['scale'])[2]
            error=pred-a['q'];bias=error.mean(0)
            scores[name]=dict(raw_xy_mm=float(np.linalg.norm(error,axis=1).mean()),
                centered_xy_mm=float(np.linalg.norm(error-bias,axis=1).mean()),bias_xy_mm=bias.tolist())
        rows.append(dict(**entry,scores=scores))
    hard_ids=set()
    for style in STYLES:
        group=[r for r in rows if r['style']==style]
        hard_ids.update(r['id'] for r in sorted(group,key=lambda r:r['scores']['R373']['raw_xy_mm'],reverse=True)[:4])
    summaries={}
    for name in models:
        summaries[name]={scope:{metric:float(np.mean([r['scores'][name][metric] for r in selected]))
            for metric in ('raw_xy_mm','centered_xy_mm')} for scope,selected in
            [('all64',rows),('baseline_weakest_quartile16',[r for r in rows if r['id'] in hard_ids])]}
    report=dict(status='COMPLETED_FIXED_FEATURE_TEST',scope='same frozen-baseline causal features/masks; full operational chain still required',
        new_test_seed=3973000,new_test_paths=64,predeclared_hard_subset_ids=sorted(hard_ids),
        all_arms_frozen_before_test=True,models=summaries,rows=rows,freezes=freezes,
        real_transfer_proven=False,production_changed=False)
    dump(OUT/'feature_test_report.json',report)
    dump(OUT/'work_state.json',dict(status='COMPLETED_FIXED_FEATURE_TEST',next='Native full-chain replays of frozen research weights, then independent arithmetic audit',
        feature_report='feature_test_report.json',real_transfer_proven=False,hardware_operated=False))
    print(json.dumps(summaries,indent=2),flush=True)


def native_job(job):
    """Actual unchanged causal chain; no XYZ command/reference enters inference."""
    id,arm=job;dest=OUT/'native'/arm/f'{id}.npz'
    if dest.exists():return dict(id=id,arm=arm,sha256=sha(dest),cached=True)
    sys.path.insert(0,str(ROOT/'code'))
    from training.domain_randomization_r369 import ResearchXY
    from live_retrained_reconstruction_r373 import DirectTrainedEngineR373
    from live_algorithm_ui_r92 import R92_PROFILE_PATH
    from live_algorithm_ui_r126 import PROFILE_KEY
    from replay_backtest import load_profile
    profile=load_profile(R92_PROFILE_PATH,PROFILE_KEY)
    engine=DirectTrainedEngineR373(profile,profile['defaults']['threshold'],profile['defaults']['min_cosine'],profile['defaults']['smoothing_ms'],70.)
    model=REAL/'xy/best_model.npz' if arm=='R373' else OUT/arm/'best_model.npz'
    engine.memory_decoder.xy_model=ResearchXY.load(model)
    engine.candidate_decoder.xy_model=ResearchXY.load(model)
    a=load(OUT/'virtual'/f'{id}.npz');pred=np.full_like(a['q'],np.nan);ok=np.zeros(len(pred),bool)
    for i in range(len(pred)):
        if a['reset'][i]:engine.set_baseline(np.zeros(15))
        r=engine.push(int(round(a['t'][i]*1e9))+10_000_000_000,a['b'][i],eligible=bool(a['valid'][i]))
        ok[i]=bool(r['valid'])
        if ok[i]:pred[i]=r['xyz']
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,pred=pred,ok=ok)
    return dict(id=id,arm=arm,sha256=sha(dest),cached=False)


def native_metrics(pred,q,ok,mask,t):
    eligible=mask&ok&np.isfinite(pred).all(1);error=pred-q
    if not eligible.any():return dict(reference_frames=int(mask.sum()),output_frames=0,coverage=0.)
    e=error[eligible];bias=e.mean(0);dt=np.diff(t)
    pairs=eligible[1:]&eligible[:-1]&(dt>0)&(dt<=.25)
    residual_steps=np.diff(error,axis=0)[pairs]
    xy_steps=np.linalg.norm(residual_steps[:,:2],axis=1);z_steps=np.abs(residual_steps[:,2])
    return dict(reference_frames=int(mask.sum()),output_frames=int(eligible.sum()),coverage=float(eligible.sum()/max(mask.sum(),1)),
        raw_xy_mm=float(np.linalg.norm(e[:,:2],axis=1).mean()),raw_xyz_mm=float(np.linalg.norm(e,axis=1).mean()),
        centered_xy_mm=float(np.linalg.norm(e[:,:2]-bias[:2],axis=1).mean()),
        centered_xyz_mm=float(np.linalg.norm(e-bias,axis=1).mean()),z_mae_mm=float(np.abs(e[:,2]).mean()),
        bias_xyz_mm=bias.tolist(),xy_residual_step_p95_mm=float(np.quantile(xy_steps,.95)) if len(xy_steps) else None,
        z_residual_step_p95_mm=float(np.quantile(z_steps,.95)) if len(z_steps) else None,
        xy_jump_gt_point2mm_count=int((xy_steps>.2).sum()),z_jump_gt_point2mm_count=int((z_steps>.2).sum()),
        contiguous_pairs=int(pairs.sum()))


def full_chain():
    protocol();feature=read(OUT/'feature_test_report.json');test=read(OUT/'test_manifest.json')
    arms=('R373',)+ARMS;done=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(native_job,[(r['id'],arm) for r in test for arm in arms]):
            done.append(row)
            if len(done)%8==0:print('FULL CHAIN',len(done),'/',len(test)*len(arms),flush=True)
    rows=[]
    for entry in test:
        id=entry['id'];a=load(OUT/'virtual'/f'{id}.npz')
        results={arm:load(OUT/'native'/arm/f'{id}.npz') for arm in arms}
        common=a['primary'].copy()
        for result in results.values():common &= result['ok']&np.isfinite(result['pred']).all(1)
        per={}
        for arm,result in results.items():
            per[arm]=dict(own_output=native_metrics(result['pred'],a['q'],result['ok'],a['primary'],a['t']),
                common_output=native_metrics(result['pred'],a['q'],result['ok'],common,a['t']),
                safe_no_contact_output_frames=int((result['ok']&(a['q'][:,2]<=-.5)).sum()))
        rows.append(dict(**entry,models=per))
    summaries={};hard=set(feature['predeclared_hard_subset_ids'])
    for arm in arms:
        summaries[arm]={}
        for scope,selected in [('all64',rows),('predeclared_baseline_weakest16',[r for r in rows if r['id'] in hard])]:
            m=[r['models'][arm] for r in selected]
            summaries[arm][scope]=dict(trials=len(selected),
                mean_own_coverage=float(np.mean([v['own_output']['coverage'] for v in m])),
                safe_output_frames=sum(v['safe_no_contact_output_frames'] for v in m),
                common={k:float(np.mean([v['common_output'][k] for v in m])) for k in
                    ('raw_xy_mm','raw_xyz_mm','centered_xy_mm','centered_xyz_mm','z_mae_mm','xy_residual_step_p95_mm','z_residual_step_p95_mm')},
                common_xy_jump_count=sum(v['common_output']['xy_jump_gt_point2mm_count'] for v in m),
                common_z_jump_count=sum(v['common_output']['z_jump_gt_point2mm_count'] for v in m))
    dump(OUT/'full_chain_report.json',dict(status='COMPLETED_INDEPENDENT_VIRTUAL_CHAIN',models=summaries,rows=rows,
        replay_outputs=done,test_manifest_sha256=sha(OUT/'test_manifest.json'),
        scope='simulator-internal independent whole geometries; real transfer not proven',
        masks_are_common_to_all_arms=True,only_research_xy_changed=True,real_transfer_proven=False))
    dump(OUT/'work_state.json',dict(status='COMPLETED_INDEPENDENT_VIRTUAL_CHAIN',next='Audit all conditions and uncertainty; extend source-supported simulator variation or causal recognition as warranted',
        report='full_chain_report.json',real_transfer_proven=False,production_changed=False,hardware_operated=False))
    print('INDEPENDENT FULL CHAIN',json.dumps(summaries,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','mine','train','evaluate','full-chain','all'),default='all')
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--max-epochs',type=int)
    a=parser.parse_args();server_only()
    if a.phase in ('mine',):mine()
    if a.phase in ('prepare','all'):prepare()
    if a.phase in ('train','all'):
        for arm in ([a.arm] if a.arm else ARMS):train(arm,a.max_epochs)
    if a.phase in ('evaluate','all'):evaluate()
    if a.phase in ('full-chain','all'):full_chain()


if __name__=='__main__':main()
