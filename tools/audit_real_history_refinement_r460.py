"""Read-only independent reductions after the finite frozen native replay.

No fitting, output filtering, registration, or test-based checkpoint selection.
Reference construction is the original recorded-command implementation, not an
independently measured physical trajectory. Diagnostic windows do not alter any
saved model output. They estimate response timing relative to the EST reference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/r460_source_real_history_refinement_20261004'
EVAL = OUT/'new_trajectory_native_replay'
DEST = EVAL/'audit'
ARMS = ('R373', 'history_position', 'history_motion_hold_anchor')
CONTRACT = dict(schema='R460-Frozen-Reduction-Audit-1', all_36_complex_trials=True,
    jump_threshold_mm=.2, max_pair_gap_s=.25, minimum_dwell_s=1.,
    amplitude='Centered per-axis covariance gain and raw coordinate excursion on identical eligible frames; diagnostic only, not applied calibration',
    reversal=dict(reference_window_s=.35, separation_s=.75, minimum_speed_mm_s=.1,
        maximum_direction_dot=-.6, scan_start_s=-.3, scan_end_s=1.5, scan_step_s=.02,
        derivative_windows_s=[[-.12,-.03],[.03,.12]], minimum_frames_per_window=3,
        threshold_fraction=.25, consecutive_negative_windows=3),
    latency_reference='Window/threshold diagnostic relative to recorded command EST timing, not exact physical delay or encoder ground truth; paired differences and unresolved events retained',
    head_sensitivity='Not evaluated by this reduction; no causal-mechanism claim',
    no_training_or_model_selection=True, no_XYZ_output_filter_added=True,
    native_outputs_immutable=True, no_hardware=True)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def independent_score(x,q,ok,mask,t):
    eligible=np.asarray(mask,bool)&np.isfinite(q).all(axis=1)
    selected=eligible&ok&np.isfinite(x).all(axis=1)
    pairs=selected[:-1]&selected[1:]&(np.diff(t)>0)&(np.diff(t)<=250_000_000)
    r=dict(reference_frames=int(np.count_nonzero(eligible)),output_frames=int(np.count_nonzero(selected)),
           coverage=float(np.count_nonzero(selected)/np.count_nonzero(eligible)) if eligible.any() else None,
           adjacent_pairs=int(np.count_nonzero(pairs)))
    if selected.any():
        e=x[selected]-q[selected];c=e-np.average(e,axis=0)
        r.update(raw_xy_mean_mm=float(np.average(np.sqrt(np.sum(e[:,:2]**2,axis=1)))),
                 centered_xy_diagnostic_mm=float(np.average(np.sqrt(np.sum(c[:,:2]**2,axis=1)))),
                 raw_z_mae_mm=float(np.average(np.abs(e[:,2]))),bias_xyz_mm=np.average(e,axis=0).tolist())
    if pairs.any():
        d=(x[1:]-x[:-1])-(q[1:]-q[:-1]);length=np.sqrt(np.sum(d[pairs,:2]**2,axis=1))
        r.update(xy_residual_step_p95_mm=float(np.percentile(length,95)),
                 xy_residual_step_rms_mm=float(np.sqrt(np.average(length**2))),
                 xy_residual_jumps_over_0p2=int(np.count_nonzero(length>.2)),
                 xy_residual_jump_rate_percent=float(100*np.count_nonzero(length>.2)/len(length)))
    return r


def independent_dwell(x,ok,mask,t,phase):
    ids=np.nonzero(mask&ok&np.isfinite(x).all(axis=1))[0];blocks=[]
    current=[]
    for i in ids:
        if current and (i!=current[-1]+1 or t[i]-t[current[-1]]>250_000_000 or phase[i]!=phase[current[-1]]):
            blocks.append(current);current=[]
        current.append(int(i))
    if current: blocks.append(current)
    values=[]
    for ix in blocks:
        if len(ix)<3 or t[ix[-1]]-t[ix[0]]<1_000_000_000:continue
        v=x[ix,:2];center=v-np.average(v,axis=0)
        values.append(float(np.sqrt(np.average(np.sum(center*center,axis=1)))))
    return dict(blocks_count=len(values), mean_xy_wobble_rms_mm=float(np.average(values)) if values else None)


def amplitude(x,q,selected):
    rows=[]
    for axis in range(3):
        a=q[selected,axis];b=x[selected,axis]
        if not len(a): rows.append(None);continue
        a0=a-a.mean();b0=b-b.mean();energy=float(a0@a0)
        rows.append(dict(reference_span_mm=float(np.ptp(a)),output_span_mm=float(np.ptp(b)),
            centered_projection_gain=float((a0@b0)/energy) if energy>len(a)*.01 else None))
    return rows


def reversal_events(t,q,eligible):
    s=(t-t[0])/1e9;dt=np.diff(s);v=np.divide(np.diff(q[:,:2],axis=0),dt[:,None],
        out=np.zeros((len(dt),2)),where=dt[:,None]>0)
    mids=(s[:-1]+s[1:])/2;events=[]
    for i in range(1,len(s)-1):
        if not eligible[i] or (events and s[i]-events[-1]['time_s']<.75):continue
        pre=(mids>=s[i]-.35)&(mids<s[i]-.03)&(dt<=.25)
        post=(mids>s[i]+.03)&(mids<=s[i]+.35)&(dt<=.25)
        if pre.sum()<3 or post.sum()<3:continue
        before=np.median(v[pre],axis=0);after=np.median(v[post],axis=0)
        nb,na=np.linalg.norm(before),np.linalg.norm(after)
        if nb<=.1 or na<=.1 or (before@after)/(nb*na)>=-.6:continue
        # Require stable reference direction on both sides, excluding rounded
        # curves and transitions that are not a well-defined sharp reversal.
        if np.max(np.linalg.norm(v[pre]-before,axis=1))>.15*nb:continue
        if np.max(np.linalg.norm(v[post]-after,axis=1))>.15*na:continue
        events.append(dict(index=i,time_s=float(s[i]),incoming_unit=(before/nb).tolist(),speed=float(nb)))
    return events


def delay(x,t,selected,event):
    s=(t-t[0])/1e9;direction=np.array(event['incoming_unit']);zero=event['time_s']
    def projected_velocity(center):
        left=selected&(s>=center-.12)&(s<=center-.03)
        right=selected&(s>=center+.03)&(s<=center+.12)
        if left.sum()<3 or right.sum()<3:return None
        denom=float(s[right].mean()-s[left].mean())
        return float((x[right,:2].mean(0)-x[left,:2].mean(0))@direction/denom)
    pre=[projected_velocity(zero+u) for u in [-.3,-.25,-.2,-.15]]
    threshold=.25*event['speed']
    if not any(a is not None and a>threshold for a in pre):return dict(status='NO_STABLE_INCOMING_RESPONSE',delay_s=None)
    count=0;start=None
    for offset in np.arange(-.3,1.5+.001,.02):
        v=projected_velocity(zero+offset)
        if v is not None and v < -threshold:
            if count==0:start=float(offset)
            count+=1
            if count>=3:return dict(status='RESOLVED',delay_s=start)
        else:count=0;start=None
    return dict(status='NO_CONFIRMED_REVERSAL_WITHIN_WINDOW',delay_s=None)


def equal(a,b):
    if a is None or b is None:assert a is None and b is None;return
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)


def run():
    assert socket.gethostname()=='googo-Super-Server'
    native=read(EVAL/'work_state.json')
    assert native['status']=='COMPLETED_FROZEN_RETROSPECTIVE_REAL_TEST'
    report=read(EVAL/'report.json');assert sha(EVAL/'report.json')==native['report_sha256']
    spec=read(OUT/'frozen_real_replay_spec.json')
    for rel,h in report['output_hashes'].items():assert sha(ROOT/rel)==h
    sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'training')]
    import training.domain_randomization_r369
    from training.build_trajectory_dataset_v1 import command_segments,command_reference,reference_origin,read_rows
    from training.analyze_r354_final_online import mask_for
    indexed={(r['batch'],r['id'],r['arm']):r for r in report['rows']};rows=[]
    for item in spec['entries']:
        streams={}
        for arm in ARMS:
            with np.load(EVAL/'native'/item['device']/arm/(item['id']+'.npz'),allow_pickle=False) as f:
                streams[arm]={k:f[k].copy() for k in f.files}
        a=streams['R373'];t=a['t_ns'];trial=ROOT/item['path']
        for b in list(streams.values())[1:]:
            np.testing.assert_array_equal(t,b['t_ns']);np.testing.assert_array_equal(a['b0_epoch'],b['b0_epoch'])
            np.testing.assert_allclose(a['b'],b['b'],atol=0,rtol=0,equal_nan=True)
        seg=command_segments(read_rows(trial/'stage_commands.csv'))
        ref=command_reference(t,seg,reference_origin(read(trial/'meta.json')));q=ref['xyz_mm']
        phase=np.array([seg[j]['phase'] if j>=0 else 'before_stage' for j in ref['command_segment_id']])
        primary,action=mask_for(item['mode'],phase,ref);finite=np.isfinite(a['b']).all(axis=1)
        primary=primary&finite;action=action&(q[:,2]>=.4)&finite
        shared=np.logical_and.reduce([b['ok']&np.isfinite(b['xyz']).all(axis=1) for b in streams.values()])
        dt=np.r_[1.,np.diff(t)/1e9];v=np.vstack((np.zeros(3),np.diff(q,axis=0)))/dt[:,None]
        dwell=action&(np.linalg.norm(v,axis=1)<.02)
        events=reversal_events(t,q,action) if item['mode']=='waypoint_xyz' else []
        for arm,b in streams.items():
            previous=indexed[(item['device'],item['id'],arm)]
            computed=independent_score(b['xyz'],q,b['ok'],primary&shared,t)
            for key,val in computed.items():equal(val,previous['moving_shared'].get(key))
            own=independent_score(b['xyz'],q,b['ok'],primary,t)
            for key,val in own.items():equal(val,previous['moving_own'].get(key))
            held=independent_dwell(b['xyz'],b['ok'],dwell,t,phase)
            for key,val in held.items():equal(val,previous['dwell'][key])
            rows.append(dict(batch=item['device'],id=item['id'],mode=item['mode'],group=item['group'],arm=arm,
                movement=computed,own=own,dwell_own=held,dwell_shared=independent_dwell(b['xyz'],b['ok'],dwell&shared,t,phase),
                amplitude_axes=amplitude(b['xyz'],q,primary&shared),
                reversal=[dict(reference_event=e,response=delay(b['xyz'],t,action&shared,e)) for e in events],
                eligible_zero_output_frames=int(np.count_nonzero(primary&b['ok']&np.isfinite(b['xyz']).all(1)&(np.linalg.norm(b['xyz'],axis=1)==0)))))
    summaries=[]
    for batch in spec['journal_prefixes']:
        for arm in ARMS:
            rr=[r for r in rows if r['batch']==batch and r['arm']==arm and r['mode']=='waypoint_xyz']
            summary=dict(batch=batch,arm=arm,trials=len(rr))
            for key in ['raw_xy_mean_mm','centered_xy_diagnostic_mm','raw_z_mae_mm','xy_residual_jump_rate_percent','coverage']:
                values=[r['movement'][key] for r in rr if r['movement'].get(key) is not None]
                summary[key]=float(np.mean(values)) if values else None
                equal(summary[key],next(s for s in report['summaries'] if s['batch']==batch and s['arm']==arm)[key])
            for kind in ['dwell_own','dwell_shared']:
                vals=[r[kind]['mean_xy_wobble_rms_mm'] for r in rr if r[kind]['mean_xy_wobble_rms_mm'] is not None]
                summary[kind+'_RMS_mm']=float(np.mean(vals)) if vals else None
            rev=[e for r in rr for e in r['reversal']];resolved=[e['response']['delay_s'] for e in rev if e['response']['delay_s'] is not None]
            summary['reversal']=dict(events=len(rev),resolved=len(resolved),unresolved=len(rev)-len(resolved),
                mean_resolved_delay_s=float(np.mean(resolved)) if resolved else None)
            summaries.append(summary)
    paired_delays=[]
    baseline={(r['batch'],r['id']):r for r in rows if r['arm']=='R373'}
    for r in rows:
        if r['arm']=='R373':continue
        old=baseline[(r['batch'],r['id'])]['reversal']
        assert len(old)==len(r['reversal'])
        for a,b in zip(old,r['reversal']):
            assert a['reference_event']==b['reference_event']
            x,y=a['response']['delay_s'],b['response']['delay_s']
            paired_delays.append(dict(batch=r['batch'],id=r['id'],arm=r['arm'],reference_event=a['reference_event'],
                baseline=a['response'],candidate=b['response'],difference_vs_R373_s=y-x if x is not None and y is not None else None))
    dump(DEST/'report.json',dict(status='INDEPENDENT_NUMERIC_REDUCTION_PASSED',contract=CONTRACT,
        native_report_sha256=sha(EVAL/'report.json'),auditor_sha256=sha(__file__),rows=rows,summaries=summaries,
        paired_reversal_diagnostics=paired_delays,
        independent_metrics=True,shared_reference_implementation=True,independent_physical_truth=False,
        head_sensitivity_completed=False,model_training_or_selection=False,production_changed=False,hardware_operated=False))
    dump(DEST/'work_state.json',dict(status='COMPLETE_NUMERIC_AMPLITUDE_AND_REVERSAL_AUDIT',report_sha256=sha(DEST/'report.json'),
        sensitivity_not_completed=True))


def wait_and_run():
    assert socket.gethostname()=='googo-Super-Server';deadline=time.monotonic()+12*3600
    dump(DEST/'work_state.json',dict(status='WAITING_FOR_FROZEN_NATIVE_REPLAY',pid=os.getpid(),maximum_wait_hours=12))
    while time.monotonic()<deadline:
        if (EVAL/'work_state.json').exists() and read(EVAL/'work_state.json')['status']=='COMPLETED_FROZEN_RETROSPECTIVE_REAL_TEST':
            run();return
        parent=read(EVAL/'job.json')['pid']
        try:os.kill(parent,0)
        except ProcessLookupError:raise RuntimeError('Native replay ended without complete receipt; retain original inputs and inspect logs')
        time.sleep(20)
    raise TimeoutError('Finite audit wait expired; no new scoring or model selection performed')


def launch():
    assert socket.gethostname()=='googo-Super-Server';DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'job.json').exists():raise FileExistsError('Preserve existing audit job')
    dump(DEST/'diagnostic_contract.json',CONTRACT)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
    with (DEST/'stdout.log').open('ab') as out,(DEST/'stderr.log').open('ab') as err:
        job=subprocess.Popen([sys.executable,'-u',__file__,'--wait-and-run'],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    record=dict(pid=job.pid,status='WAITING_FOR_NATIVE_REPLAY_THEN_ONE_AUDIT',auditor_sha256=sha(__file__),
        diagnostic_contract_sha256=sha(DEST/'diagnostic_contract.json'),cpu_threads=2,gpu=False,scientific_training=False)
    dump(DEST/'job.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--wait-and-run',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.launch:launch()
    elif a.wait_and_run:wait_and_run()
    elif a.run:run()
    else:p.error('Choose mode')
