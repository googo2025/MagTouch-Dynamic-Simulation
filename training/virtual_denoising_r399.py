"""Matched virtual TRAIN noise/causal-filter controls, separate from R397.

Reuses the checked R397 training implementation in this isolated process only.
The running R397 process and its protocol are not modified. Source real features
remain unchanged. Removing injected noise is not knowing the real clean field.
"""
from __future__ import annotations
import argparse, json, os, socket, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from training import virtual_breakthrough_r397 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/r399_virtual_denoising_20261003'
STUDY397=ROOT/'analysis/r397_virtual_breakthrough_20261003'
ARMS=('raw128','clean128','causal128')
MODE={'raw128':'raw','clean128':'clean','causal128':'causal'}
TAU=.1


def causal_lowpass(b,t,tau=TAU,reset=None,valid=None):
    b=np.asarray(b,float);t=np.asarray(t,float)
    if tau<=0 or not np.isfinite(t).all() or b.shape!=(len(t),15) or np.any(np.diff(t)<=0):raise ValueError('Valid tau,15D and increasing times required')
    reset=np.zeros(len(t),bool) if reset is None else np.asarray(reset,bool)
    valid=np.isfinite(b).all(1) if valid is None else np.asarray(valid,bool)&np.isfinite(b).all(1)
    out=np.full_like(b,np.nan);state=None;previous=None
    for i in range(len(t)):
        if reset[i]:state=None;previous=None
        if not valid[i]:continue
        dt=t[i]-previous if previous is not None else None
        if state is None or dt>.25:state=b[i].copy()
        else:state=state+(-np.expm1(-dt/tau))*(b[i]-state)
        out[i]=state;previous=t[i]
    return out


def protocol():
    p=base.read(STUDY397/'protocol.json').copy()
    p.update(schema='R399-MatchedVirtualDenoising-1',purpose='Separate injected-noise removal and causal magnetic-input filtering from data expansion',
        arms=list(ARMS),virtual_training_geometries=128,
        identical_train_geometry_selection='R397 uniform128 TRAIN IDs; no model/test outcomes used',
        input_modes=MODE,causal_filter=dict(kind='first-order exponential lowpass',tau_seconds=TAU,
            coefficient='1-exp(-actual_dt/tau)',initialization='first valid field at each B0 reset',
            maximum_contiguous_gap_seconds=.25,future_frames_used=False),
        clean_mode='remove only extra injected Gaussian sensor noise, retain static map and all axis-separated physical-history surrogate terms',
        source_features='original real TRAIN/validation and static anchors unchanged, explicitly testing only virtual-training preprocessing',
        reconstruction_train_features='recapture complete native causal XY features from each field mode; never smooth cached features or XYZ labels',
        new_seed_bases=dict(validation=3992000,test=3993000),
        common_test='unfiltered independently generated noisy field, all models; fixed features then native full chain',
        matched_filter_test='same frozen models also all tested with identical causal input filter, including frozen R373 baseline; report filter and model effects separately',
        delay='explicit residual-step and turning-response diagnostics; no time alignment or future-frame filtering',
        delay_diagnostic_rule='command-axis reversals with signed velocities >=.03mm/s, cosine<-.5; deduplicate1s; first3 valid contiguous signed output velocity pairs>=.03mm/s within1.5s, otherwise censored. No future information enters inference.',
        filter_step_characteristics=dict(t50_seconds=TAU*np.log(2),t90_seconds=TAU*np.log(10),t95_seconds=TAU*np.log(20),
            meaning='ideal first-order field-filter step response only, not measured sensor/mechanical or XYZ latency'),
        generator_clean_field_is_independent_truth=False,real_transfer_proven=False)
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
    if path.exists():assert base.read(path)==p,'Preserve preregistered denoising protocol'
    else:base.dump(path,p)
    return p


def setup():
    base.OUT=OUT;base.ARMS=ARMS;base.protocol=protocol


def field(role,index,mode='raw'):
    if role not in ('train','validation','test') or mode not in ('raw','clean','causal'):raise ValueError('Declared role/input mode required')
    from training.broad_trajectory_r370 import geometry
    from training.forward_complex_r363 import AxisForward
    offset=0 if role=='train' else 290000
    q,t,info=geometry(role,offset+index);g=AxisForward.load(base.GEN)
    clean=g.rollout(q,t);noise=np.random.default_rng(info['seed']+400000).normal(size=clean.shape)*np.array(base.read(base.NOISE)['sigma_uT'])
    raw=clean+noise;b=clean if mode=='clean' else causal_lowpass(raw,t) if mode=='causal' else raw
    return q,t,b,g,info,clean,raw


def capture_features(q,t,b,primary,dest,stream=None):
    sys.path.insert(0,str(ROOT/'code'))
    from training.domain_randomization_r369 import ResearchXY
    from live_retrained_reconstruction_r373 import DirectTrainedEngineR373,PlateauCausalXY
    from live_algorithm_ui_r92 import R92_PROFILE_PATH
    from live_algorithm_ui_r126 import PROFILE_KEY
    from replay_backtest import load_profile
    profile=load_profile(R92_PROFILE_PATH,PROFILE_KEY)
    e=DirectTrainedEngineR373(profile,profile['defaults']['threshold'],profile['defaults']['min_cosine'],profile['defaults']['smoothing_ms'],70.)
    main=ResearchXY(PlateauCausalXY());provisional=ResearchXY(PlateauCausalXY())
    e.memory_decoder.xy_model=main;e.candidate_decoder.xy_model=provisional;e.set_baseline(np.zeros(15))
    xx=[];yy=[];guard=[];qq=[];pred=[];ids=[];full=np.full_like(q,np.nan);ok=np.zeros(len(q),bool)
    for i,now in enumerate(t):
        main.last=provisional.last=None;r=e.push(int(round(now*1e9))+10_000_000_000,b[i],eligible=True)
        ok[i]=bool(r['valid'])
        if ok[i]:full[i]=r['xyz']
        if not (ok[i] and primary[i]):continue
        chosen=provisional if r['extra']['localization_path']=='provisional_permission' else main
        assert chosen.last is not None;x=chosen.last.copy();gd=x[:2]*10
        np.testing.assert_allclose(gd+chosen.predict(x[None])[0],full[i,:2],atol=1e-9)
        xx.append(x);yy.append(np.clip(q[i,:2]-gd,-3,3));guard.append(gd);qq.append(q[i,:2]);pred.append(full[i,:2]);ids.append(i)
    assert len(xx)>10 and np.isfinite(xx).all()
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,x=np.array(xx),y=np.array(yy),guard=np.array(guard),q=np.array(qq),initial_pred=np.array(pred),indices=np.array(ids),samples=len(xx))
    if stream is not None:
        stream.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(stream,q=q,t=t,b=b,primary=primary,valid=np.ones(len(q),bool),reset=np.r_[True,np.zeros(len(q)-1,bool)],baseline_pred=full,baseline_ok=ok)


def capture(job):
    role,index,mode=job
    id=f'V370_train_{index:04d}' if role=='train' else f'V399_{role}_{index:04d}'
    dest=OUT/'features'/mode/'train'/f'{id}.npz' if role=='train' else OUT/'features'/f'{id}.npz'
    stream=None if role=='train' else OUT/'virtual'/f'{id}.npz'
    if not dest.exists():
        q,t,b,g,info,clean,raw=field(role,index,mode)
        primary=(q[:,2]>=.4)&(q[:,2]<=3)&g.support(q)
        capture_features(q,t,b,primary,dest,stream)
        if role=='train' and mode=='raw':
            prior=base.load(ROOT/'analysis/r379_full_virtual_expansion_20261002/features'/f'{id}.npz')
            now=base.load(dest)
            np.testing.assert_array_equal(now['indices'],prior['indices'])
            np.testing.assert_allclose(now['initial_pred'],prior['initial_pred'],atol=1e-8,rtol=1e-8)
        noise_metrics=dict(injected_noise_rms_uT=float(np.sqrt(np.mean((raw-clean)**2))),
            chosen_vs_clean_rms_uT=float(np.sqrt(np.mean((b-clean)**2))),
            causal_vs_clean_rms_uT=float(np.sqrt(np.mean((causal_lowpass(raw,t)-clean)**2))),
            clean_history_retained=True)
        base.dump(dest.with_suffix('.metadata.json'),noise_metrics)
    return dict(id=id,role=role,mode=mode,seed=({'train':3701000,'validation':3992000,'test':3993000}[role]+index),
        style=base.STYLES[index%4],samples=int(base.load(dest)['samples']),feature_sha256=base.sha(dest),
        **({'stream_sha256':base.sha(stream)} if stream else {}))


def prepare():
    p=protocol();mine=base.read(STUDY397/'mining.json');ids=mine['arms']['uniform128']['ids']
    assert len(ids)==128 and all(id.startswith('V370_train_') for id in ids)
    jobs=[('validation',i,'raw') for i in range(p['new_validation_geometries'])]
    jobs += [('train',int(id.rsplit('_',1)[-1]),mode) for mode in MODE.values() for id in ids]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for row in pool.map(capture,jobs):
            rows.append(row)
            if len(rows)%8==0:print('R399 PREPARED',len(rows),'/',len(jobs),flush=True)
    base.dump(OUT/'validation_manifest.json',[r for r in rows if r['role']=='validation'])
    base.dump(OUT/'feature_manifest.json',rows)
    base.dump(OUT/'mining.json',dict(arms={arm:dict(ids=ids,paths=len(ids)) for arm in ARMS},
        identical_geometry_ids=True,source_mining_sha256=base.sha(STUDY397/'mining.json'),no_test_used=True))
    base.dump(OUT/'work_state.json',dict(status='PREPARED_DENOISING_CONTROLS',next='Fully train3 matched arms; independent test still sealed',hardware_operated=False))


def evaluate():
    p=protocol()
    for arm in ARMS:
        f=base.read(OUT/arm/'freeze.json');assert f['status']=='COMPLETED_PLATEAU'
        assert f['model_sha256']==base.sha(OUT/arm/'best_model.npz')
    with ProcessPoolExecutor(max_workers=2) as pool:test=list(pool.map(capture,[('test',i,'raw') for i in range(p['new_test_geometries'])]))
    base.dump(OUT/'test_manifest.json',test);rows=[]
    for entry in test:
        a=base.load(OUT/'features'/f"{entry['id']}.npz");scores={}
        for arm in ('R373',)+ARMS:
            m=base.load(base.REAL/'xy/best_model.npz' if arm=='R373' else OUT/arm/'best_model.npz')
            pred=a['guard']+base.forward([m[k] for k in base.KEYS],(a['x']-m['mean'])/m['scale'])[2]
            error=pred-a['q'];bias=error.mean(0)
            scores[arm]=dict(raw_xy_mm=float(np.linalg.norm(error,axis=1).mean()),centered_xy_mm=float(np.linalg.norm(error-bias,axis=1).mean()),bias_xy_mm=bias.tolist())
        rows.append(dict(**entry,scores=scores))
    hard=[]
    for style in base.STYLES:
        group=[r for r in rows if r['style']==style]
        hard.extend(r['id'] for r in sorted(group,key=lambda r:r['scores']['R373']['raw_xy_mm'],reverse=True)[:4])
    base.dump(OUT/'feature_test_report.json',dict(new_test_seed=3993000,rows=rows,predeclared_hard_subset_ids=hard,
        scope='fixed R373 causal features and masks, unfiltered input; full chain follows',all_arms_frozen_before_test=True,real_transfer_proven=False))


def wait_for_previous(pid):
    if not pid:return
    base.dump(OUT/'work_state.json',dict(status='QUEUED_AFTER_R397',waiting_for_pid=pid,own_pid=os.getpid(),hardware_operated=False))
    begin=time.monotonic()
    while True:
        try:os.kill(pid,0)
        except ProcessLookupError:break
        if time.monotonic()-begin>43200:raise RuntimeError('R397 still active after12hours; inspect rather than duplicate compute')
        time.sleep(30)


def filtered_native_job(job):
    id,arm=job;dest=OUT/'native_causal_input'/arm/f'{id}.npz'
    if dest.exists():return dict(id=id,arm=arm,sha256=base.sha(dest),cached=True)
    sys.path.insert(0,str(ROOT/'code'))
    from training.domain_randomization_r369 import ResearchXY
    from live_retrained_reconstruction_r373 import DirectTrainedEngineR373
    from live_algorithm_ui_r92 import R92_PROFILE_PATH
    from live_algorithm_ui_r126 import PROFILE_KEY
    from replay_backtest import load_profile
    profile=load_profile(R92_PROFILE_PATH,PROFILE_KEY)
    e=DirectTrainedEngineR373(profile,profile['defaults']['threshold'],profile['defaults']['min_cosine'],profile['defaults']['smoothing_ms'],70.)
    model=base.REAL/'xy/best_model.npz' if arm=='R373' else OUT/arm/'best_model.npz'
    e.memory_decoder.xy_model=ResearchXY.load(model);e.candidate_decoder.xy_model=ResearchXY.load(model)
    a=base.load(OUT/'virtual'/f'{id}.npz')
    b=causal_lowpass(a['b'],a['t'],reset=a['reset'],valid=a['valid'])
    pred=np.full_like(a['q'],np.nan);ok=np.zeros(len(pred),bool)
    for i in range(len(pred)):
        if a['reset'][i]:e.set_baseline(np.zeros(15))
        r=e.push(int(round(a['t'][i]*1e9))+10_000_000_000,b[i],eligible=bool(a['valid'][i]))
        ok[i]=bool(r['valid'])
        if ok[i]:pred[i]=r['xyz']
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,pred=pred,ok=ok)
    return dict(id=id,arm=arm,sha256=base.sha(dest),cached=False)


def reversal_delays(pred,ok,q,t):
    """Offline latency diagnostic, not time-shifted performance or training signal."""
    dt=np.diff(t);v=np.diff(q,axis=0)/dt[:,None]
    speed=np.linalg.norm(v,axis=1);cos=np.sum(v[1:]*v[:-1],axis=1)/np.maximum(speed[1:]*speed[:-1],1e-12)
    candidates=np.flatnonzero((cos<-.5)&(speed[1:]>.05)&(speed[:-1]>.05))+1
    events=[];last=-np.inf
    pv=np.diff(pred,axis=0)/dt[:,None]
    pair=ok[:-1]&ok[1:]&np.isfinite(pv).all(1)&(dt>0)&(dt<=.25)
    for i in candidates:
        if t[i]-last<1.:continue
        axes=np.flatnonzero((v[i]*v[i-1]<0)&(np.abs(v[i])>=.03)&(np.abs(v[i-1])>=.03))
        if not len(axes):continue
        axis=int(axes[np.argmax(np.abs(v[i,axes]))]);sign=np.sign(v[i,axis]);delay=None
        for j in range(i,min(len(pv)-2,np.searchsorted(t,t[i]+1.5))):
            if pair[j:j+3].all() and np.all(pv[j:j+3,axis]*sign>=.03):
                delay=float(t[j]-t[i]);break
        events.append(dict(event_time_s=float(t[i]),axis=axis,delay_s=delay,censored=delay is None));last=t[i]
    return events


def matched_filter_test():
    for arm in ARMS:assert base.read(OUT/arm/'freeze.json')['status']=='COMPLETED_PLATEAU'
    test=base.read(OUT/'test_manifest.json');arms=('R373',)+ARMS;done=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(filtered_native_job,[(e['id'],a) for e in test for a in arms]):
            done.append(result)
            if len(done)%8==0:print('R399 MATCHED INPUT FILTER',len(done),'/',len(test)*len(arms),flush=True)
    rows=[]
    for e in test:
        a=base.load(OUT/'virtual'/f"{e['id']}.npz");results={}
        for m in arms:
            for mode,folder in [('raw','native'),('causal','native_causal_input')]:results[(m,mode)]=base.load(OUT/folder/m/f"{e['id']}.npz")
        common=a['primary'].copy()
        for v in results.values():common &= v['ok']&np.isfinite(v['pred']).all(1)
        models={}
        for m in arms:
            modes={}
            for mode in ('raw','causal'):
                v=results[(m,mode)];delays=reversal_delays(v['pred'],v['ok'],a['q'],a['t'])
                modes[mode]=dict(own=base.native_metrics(v['pred'],a['q'],v['ok'],a['primary'],a['t']),
                    shared8=base.native_metrics(v['pred'],a['q'],v['ok'],common,a['t']),reversal_delays=delays,
                    safe_output_frames=int((v['ok']&(a['q'][:,2]<=-.5)).sum()))
            models[m]=modes
        rows.append(dict(**e,models=models))
    summary={}
    for m in arms:
        summary[m]={}
        for mode in ('raw','causal'):
            values=[r['models'][m][mode] for r in rows];metrics=[v['shared8'] for v in values]
            delays=[d for v in values for d in v['reversal_delays']];observed=[d['delay_s'] for d in delays if not d['censored']]
            summary[m][mode]=dict(trials=len(rows),mean_own_coverage=float(np.mean([v['own']['coverage'] for v in values])),
                shared8={k:float(np.mean([v[k] for v in metrics if v.get(k) is not None])) if any(v.get(k) is not None for v in metrics) else None for k in
                    ('raw_xy_mm','raw_xyz_mm','centered_xy_mm','z_mae_mm','xy_residual_step_p95_mm','z_residual_step_p95_mm')},
                xy_jumps=sum(v.get('xy_jump_gt_point2mm_count',0) for v in metrics),
                contiguous_pairs=sum(v.get('contiguous_pairs',0) for v in metrics),
                safe_output_frames=sum(v['safe_output_frames'] for v in values),
                reversal_events=len(delays),reversal_censored=sum(d['censored'] for d in delays),
                observed_reversal_delay_median_s=float(np.median(observed)) if observed else None)
    base.dump(OUT/'matched_filter_report.json',dict(status='COMPLETED_MATCHED_CAUSAL_INPUT_FILTER',models=summary,rows=rows,
        test_manifest_sha256=base.sha(OUT/'test_manifest.json'),filtered_outputs=done,
        tau_seconds=TAU,no_output_smoothing_or_time_alignment=True,real_transfer_proven=False,
        interpretation='Separate model contribution on raw independent field from input-filter contribution.8-way commonmask plus owncoverage; all64 geometries. Signed-threshold reversal delay is a diagnostic, not measured physicallatency.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-pid',type=int);args=parser.parse_args()
    base.server_only();protocol();wait_for_previous(args.wait_for_pid);setup();prepare()
    for arm in ARMS:
        base.POOL=OUT/'features'/MODE[arm]/'train';base.train(arm)
    evaluate();base.full_chain();matched_filter_test()
    base.dump(OUT/'work_state.json',dict(status='COMPLETED_RAW_AND_MATCHED_FILTER_INDEPENDENT_TEST',report='full_chain_report.json',matched_filter_report='matched_filter_report.json',
        next='Audit all64 raw/matched-filter, coverage and reversal delays; simulator-internal evidence, independent real transfer still separate',
        real_transfer_proven=False,hardware_operated=False))


if __name__=='__main__':main()
