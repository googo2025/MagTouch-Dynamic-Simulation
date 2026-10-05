"""All frozen R397 XY heads on complete real ordered journals, never fitting.

No hardware, production writes, inferred B0, time alignment or output smoothing.
R373 parity is a prerequisite. This is retrospective testing, not a blind test.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
from pathlib import Path, PureWindowsPath
import socket
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/r405_frozen_virtual_real_replay_20261003'
R397 = ROOT / 'analysis/r397_virtual_breakthrough_20261003'
PRIOR = ROOT / 'analysis/r395_new_real_simulation_paper_20261003'
ARMS = ('R373', 'common64', 'uniform128', 'hard128')
KEYS = ('mean', 'scale', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3')
SHAPES = ((53,), (53,), (53,64), (64,), (64,64), (64,), (64,2), (2,))


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def dump(p,v):
    def finite(x):
        if isinstance(x,dict): return {k:finite(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [finite(v) for v in x]
        if isinstance(x,np.ndarray): return finite(x.tolist())
        if isinstance(x,np.generic): return finite(x.item())
        if isinstance(x,float) and not np.isfinite(x): return None
        return x
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(finite(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def arrays(path):
    with np.load(path,allow_pickle=False) as f:
        if set(f.files)!=set(KEYS): raise ValueError('Unexpected model arrays')
        a={k:f[k].copy() for k in KEYS}
    for key,shape in zip(KEYS,SHAPES):
        if a[key].shape!=shape or not np.isfinite(a[key]).all():
            raise ValueError('Invalid array '+key)
    if np.any(a['scale']<=0): raise ValueError('Nonpositive scale')
    return a


class FrozenXY:
    def __init__(self, values):
        for k in KEYS:
            v=values[k].copy();v.flags.writeable=False;setattr(self,k,v)

    def predict(self,x):
        x=np.asarray(x,float).reshape(-1,53)
        a=np.tanh(((x-self.mean)/self.scale)@self.w1+self.b1)
        return np.tanh(a@self.w2+self.b2)@self.w3+self.b3


def install(predictor, values):
    """Retain the original adapter/engine/B0; replace both XY routes only."""
    e=predictor.engine
    for decoder in (e.memory_decoder,e.candidate_decoder):
        for k in ('mean','scale'):
            np.testing.assert_array_equal(values[k],getattr(decoder.xy_model,k))
    e.memory_decoder.xy_model=FrozenXY(values)
    e.candidate_decoder.xy_model=FrozenXY(values)
    # The original begin/reset/set_baseline clear state, not these model objects.
    return e.memory_decoder.xy_model,e.candidate_decoder.xy_model


def resolve_config(spec, recorded_hash):
    selected=copy.deepcopy(spec)
    path=Path(selected['path'])
    if not path.is_file():
        name=PureWindowsPath(selected['path']).name
        path=ROOT/'config'/name
    path=path.resolve()
    if not path.is_relative_to((ROOT/'config').resolve()) or sha(path)!=recorded_hash:
        raise ValueError('Recorded configuration unresolved or SHA differs')
    selected['path']=str(path)
    return selected


def frozen_inputs():
    spec=read(R397/'frozen_real_replay_spec.json')
    for rel,h in spec['protocols'].items():
        if sha(ROOT/rel)!=h: raise ValueError('Frozen prior protocol changed')
    entries=[]
    for rel in spec['protocols']:
        entries.extend(read(ROOT/rel)['raw_inputs'])
    if len(entries)!=48 or len({e['id'] for e in entries})!=48:
        raise ValueError('Exactly three complete16-Trial batches required')
    for batch in spec['journal_prefixes']:
        group=[e for e in entries if e['device']==batch]
        if len(group)!=16 or sum(e['mode']=='waypoint_xyz' for e in group)!=12:
            raise ValueError('Frozen batch composition changed')
        record=spec['journal_prefixes'][batch]
        if sha(ROOT/record['path'])!=record['sha256']:
            raise ValueError('Ordered journal prefix changed')
    for arm,h in spec['models'].items():
        p=(ROOT/'analysis/r372_direct_source_plateau_20261001/xy/best_model.npz'
           if arm=='R373' else R397/arm/'best_model.npz')
        if sha(p)!=h: raise ValueError('Frozen XY weights changed: '+arm)
        if arm!='R373':
            f=read(R397/arm/'freeze.json')
            if (f['status']!='COMPLETED_PLATEAU' or f['model_sha256']!=h or
                f['protocol_sha256']!=sha(R397/'protocol.json')):
                raise ValueError('Not a sufficiently trained frozen R397 arm')
    manifest=read(OUT/'input_manifest.json')
    for rel,h in manifest['files'].items():
        if sha(ROOT/rel)!=h: raise ValueError('Transported input changed: '+rel)
    return spec,entries


def replay(batch, arm, spec, entries):
    from training import real_simulation_paper_r395 as base
    from general_model_selection_r387 import load, describe
    journal=ROOT/spec['journal_prefixes'][batch]['path']
    header=next(base.native._read_journal(journal))
    selection,target,same=base.replay_selection(header,'R373')
    selection=resolve_config(selection,header['models'][target]['checkpoint_or_profile_sha256'])
    dest=OUT/'native'/batch/arm
    completed=dest/'completed.json'
    contract=dict(journal_sha256=sha(journal),model_sha256=spec['models'][arm],
                  spec_sha256=sha(R397/'frozen_real_replay_spec.json'),script_sha256=sha(__file__))
    if completed.exists():
        r=read(completed)
        if r['contract']!=contract: raise ValueError('Existing replay differs; preserve, do not overwrite')
        for rel,h in r['output_hashes'].items():
            if sha(dest/rel)!=h: raise ValueError('Completed output changed')
        return r
    dest.mkdir(parents=True,exist_ok=True)
    predictor=load(selection)
    description=describe(predictor,selection)
    if same:
        before=base.native._hash_fields(header['models'][target])
        now=base.native._hash_fields(description)
        if any(now.get(k)!=v for k,v in before.items()):
            raise ValueError('Original recorded numerical dependencies changed')
        hashes=base.native.implementation_sources(predictor)
        if any(hashes.get(k)!=v for k,v in header['implementation_sources'][target].items()):
            raise ValueError('Original native implementation differs')
    head_refs=None
    if arm!='R373': head_refs=install(predictor,arrays(R397/arm/'best_model.npz'))
    session=base.native.NativeReplaySession(predictor,target,annotate_display=target=='A',
        honor_output_record_markers=(arm=='R373' and same),allow_unsupported_controls=False)
    output=[];fields={};epochs={};b0_log=[];epoch=0;groups=0;started=time.perf_counter()
    for record in base.native._read_journal(journal):
        was_ready=bool(predictor.ready)
        rr=session.consume(record)
        for row in rr:
            base.compact(row)
            if arm!='R373':
                row['extra_outputs'].update(xy_model_sha256=spec['models'][arm],
                    research_arm=arm,counterfactual_frozen_R397=True)
            output.append(row)
        if head_refs is not None:
            if (predictor.engine.memory_decoder.xy_model is not head_refs[0] or
                predictor.engine.candidate_decoder.xy_model is not head_refs[1]):
                raise ValueError('A native reset replaced the frozen research head')
        if record['op']=='group':
            groups+=1
            if predictor.ready:
                if not was_ready:
                    epoch+=1;b0_log.append(dict(t_ns=record['t_ns'],epoch=epoch,b0_uT=predictor.b0.reshape(-1).tolist()))
                from live_normalized_fit import vector
                v=vector({int(k):v for k,v in record['data'].items()})
                if v is not None:
                    fields[int(record['t_ns'])]=v-predictor.b0.reshape(-1)
                    epochs[int(record['t_ns'])]=epoch
            if groups%4000==0:
                print('R405',batch,arm,groups,'groups',round(time.perf_counter()-started,1),'s',flush=True)
    parity=[];out_hashes={}
    for item in entries:
        trial=ROOT/item['path'];base.comparison._assert_not_excluded(trial)
        meta=read(trial/'meta.json')
        if meta['status']!='completed': raise ValueError('Active/incomplete Trial not eligible')
        if arm=='R373' and same:
            rows=base.selected(output,trial)
            check=base.native.compare_online_csv(rows,trial/('live_predictions.csv' if target=='A' else 'live_predictions_virtual1000.csv'))
            if not check['xyz_and_validity_parity']: raise ValueError('Recorded R373 parity failed')
            parity.append(dict(id=item['id'],**check))
        else:
            with (trial/'live_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
                stamps={int(r['t_prediction_ns']) for r in csv.DictReader(f)}
            rows=[r for r in output if int(r['t_ns']) in stamps]
            if len(rows)!=len(stamps): raise ValueError('Not all native emitted timestamps preserved')
        t=np.array([r['t_ns'] for r in rows],np.int64)
        if not len(t) or not (np.diff(t)>0).all(): raise ValueError('Empty/reordered emissions')
        x=np.array([r['visual_xyz_mm'] for r in rows],float)
        ok=np.array([r['interaction_valid'] for r in rows],bool)&np.isfinite(x).all(1)
        b=np.array([fields.get(int(tn),np.full(15,np.nan)) for tn in t])
        b0_epochs=np.array([epochs.get(int(tn),-1) for tn in t],np.int64)
        # Compare the entire old frozen native adapter, not an invented CSV replay.
        with np.load(PRIOR/'native'/batch/'R373'/(item['id']+'.npz'),allow_pickle=False) as reference:
            np.testing.assert_array_equal(t,reference['t_ns'])
            np.testing.assert_array_equal(b0_epochs,reference['b0_epoch'])
            np.testing.assert_allclose(b,reference['b'],atol=0,rtol=0,equal_nan=True)
            if arm=='R373':
                np.testing.assert_array_equal(ok,reference['ok'])
                np.testing.assert_allclose(x,reference['xyz'],atol=1e-12,rtol=0,equal_nan=True)
        if arm=='R373':
            saved=np.asarray(meta['dynamic_b0_calibration']['b0_5x3_uT']).reshape(-1)
            before=[w for w in b0_log if w['t_ns']<=int(meta['live_prediction_recording']['started_perf_ns'])]
            if not before or np.max(np.abs(np.array(before[-1]['b0_uT'])-saved))>=1e-10:
                raise ValueError('Initial B0 differs from actual acquisition metadata')
        extra=[r['extra_outputs'] for r in rows]
        path=dest/(item['id']+'.npz')
        np.savez_compressed(path,t_ns=t,xyz=x,ok=ok,b=b,b0_epoch=b0_epochs,
            hint=np.array([e.get('external_no_contact_hint') is True for e in extra]),
            permission=np.array([e.get('localization_permission_score') if e.get('localization_permission_score') is not None else np.nan for e in extra]))
        out_hashes[path.name]=sha(path)
    r=dict(status='COMPLETED',contract=contract,batch=batch,arm=arm,groups=groups,
        seconds=time.perf_counter()-started,baseline_description=description,
        research_head_overlay=(None if arm=='R373' else dict(model_sha256=spec['models'][arm],both_xy_routes=True,
            unchanged_normalization=True,unchanged_Z_and_permission_rules=True,original_metadata_is_baseline_provenance=True)),
        parity=parity,all_trials_baseline_stream_and_b0_parity=True,b0_windows=b0_log,
        controls=session.controls,gaps=session.gaps,output_hashes=out_hashes,
        implementation_sha256=base.native.implementation_sources(predictor),
        test_fit_or_selection=False,hardware_operated=False,production_changed=False)
    dump(completed,r);del predictor,session,output,fields;gc.collect()
    return r


def analyze(spec,entries):
    from training import real_simulation_paper_r395 as base
    from training.real_jump_audit_r402 import score,dwell_score,summarize
    rows=[];inputs={}
    for item in entries:
        batch=item['device'];trial=ROOT/item['path'];streams={}
        for arm in ARMS:
            p=OUT/'native'/batch/arm/(item['id']+'.npz')
            with np.load(p,allow_pickle=False) as f:streams[arm]={k:f[k].copy() for k in f.files}
            inputs[p.relative_to(ROOT).as_posix()]=sha(p)
        a=streams['R373'];t=a['t_ns']
        for arm in ARMS[1:]:
            np.testing.assert_array_equal(t,streams[arm]['t_ns'])
            np.testing.assert_array_equal(a['b0_epoch'],streams[arm]['b0_epoch'])
            np.testing.assert_allclose(a['b'],streams[arm]['b'],atol=0,rtol=0,equal_nan=True)
        seg=base.command_segments(base.read_rows(trial/'stage_commands.csv'))
        ref=base.command_reference(t,seg,base.reference_origin(read(trial/'meta.json')));q=ref['xyz_mm']
        phase=np.array([seg[j]['phase'] if j>=0 else 'before_stage' for j in ref['command_segment_id']])
        primary,action=base.mask_for(item['mode'],phase,ref)
        finite=np.isfinite(a['b']).all(1);primary&=finite
        action=action&(q[:,2]>=.4)&finite
        shared=np.logical_and.reduce([s['ok'] for s in streams.values()])
        dt=np.r_[1.,np.diff(t)/1e9];v=np.vstack((np.zeros(3),np.diff(q,axis=0)))/dt[:,None]
        dwell=action&(np.linalg.norm(v,axis=1)<.02)
        safe=(q[:,2]<=-.2)&finite
        for arm,s in streams.items():
            raw=base.metric(s['xyz'],q,s['ok'],primary&shared,t)
            rows.append(dict(id=item['id'],batch=batch,arm=arm,mode=item['mode'],group=item['group'],
                moving_shared=score(s['xyz'],q,s['ok'],primary&shared,t),
                moving_xyz_shared=raw,moving_own=score(s['xyz'],q,s['ok'],primary,t),
                dwell=dwell_score(s['xyz'],s['ok'],dwell,t,phase),
                safe_reference=dict(frames=int(safe.sum()),outputs=int((safe&s['ok']).sum()),
                    without_external_hint_frames=int((safe&~s['hint']).sum()),
                    without_external_hint_outputs=int((safe&~s['hint']&s['ok']).sum()),
                    with_external_hint_frames=int((safe&s['hint']).sum()),
                    with_external_hint_outputs=int((safe&s['hint']&s['ok']).sum()))))
    summaries=[]
    for batch in spec['journal_prefixes']:
        for arm in ARMS:
            rr=[r for r in rows if r['batch']==batch and r['arm']==arm and r['mode']=='waypoint_xyz']
            s=summarize([r['moving_shared'] for r in rr])
            xyz=[r['moving_xyz_shared'].get('xyz_mean_mm') for r in rr]
            s.update(batch=batch,arm=arm,raw_xyz_mean_mm=float(np.mean([v for v in xyz if v is not None])),
                     own_coverage=summarize([r['moving_own'] for r in rr])['coverage'])
            summaries.append(s)
    dump(OUT/'report.json',dict(status='COMPLETED_FROZEN_RETROSPECTIVE_REAL_TEST',spec_sha256=sha(R397/'frozen_real_replay_spec.json'),
        summaries=summaries,rows=rows,output_hashes=inputs,
        comparison='all four frozen heads; common validity accuracy and independent own coverage; no extra filters or alignment',
        diagnostics_not_deployable_corrections=True,previously_viewed_real_records=True,
        not_new_blind_test=True,no_fit_or_selection=True,production_changed=False,hardware_operated=False))


def main():
    if socket.gethostname()!='googo-Super-Server': raise RuntimeError('Authorized server only')
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('all','verify'),default='all');a=p.parse_args()
    spec,entries=frozen_inputs()
    if a.phase=='verify': print('R405 immutable inputs verified',len(entries),flush=True);return
    dump(OUT/'work_state.json',dict(status='RUNNING',pid=os.getpid(),spec_sha256=sha(R397/'frozen_real_replay_spec.json')))
    for batch in spec['journal_prefixes']:
        group=[e for e in entries if e['device']==batch]
        for arm in ARMS:
            replay(batch,arm,spec,group)
            dump(OUT/'work_state.json',dict(status='RUNNING',pid=os.getpid(),last_completed=[batch,arm],
                actual_timestamps_baselines=True,hardware_operated=False,production_changed=False))
    analyze(spec,entries)
    dump(OUT/'work_state.json',dict(status='COMPLETED_FROZEN_RETROSPECTIVE_REAL_TEST',report='report.json',
        no_fit_or_selection=True,hardware_operated=False,production_changed=False))


if __name__=='__main__': main()
