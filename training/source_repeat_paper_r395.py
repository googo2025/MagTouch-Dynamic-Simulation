"""Frozen supplemental SOURCE online batch; no hardware or production writes."""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from training import real_simulation_paper_r395 as base
from training.audit_real_translation_r395 import NEW_SOURCE

DEST=base.OUT / "source_repeat"


def inventory():
    DEST.mkdir(exist_ok=True)
    prog=base.read(NEW_SOURCE/'plan_progress.json')
    assert prog['status']=='COMPLETED' and prog['completed_trials']==prog['planned_trials']==16
    entries=[]
    for trial in sorted(NEW_SOURCE.glob('Trial_*')):
        base.comparison._assert_not_excluded(trial)
        meta=base.read(trial/'meta.json'); item=base.read(trial/'batch_plan_item.json')
        assert meta['status']=='completed'
        tp=meta['trial_parameters']
        entries.append(dict(id='SOURCE_REPEAT_'+tp['trial_id'], device='SOURCE_REPEAT',
            path=trial.relative_to(base.ROOT).as_posix(),mode=tp['command_mode'],group=item['group'],
            speed_mm_s=tp['speed_mm_s'],repeats=tp['repeat_count'],
            journal=str(base.comparison._journal_path(trial,meta)),
            raw_hashes={f:base.sha(trial/f) for f in base.FILES if (trial/f).is_file()}))
    protocol=dict(schema='MagTouch-R395-Supplemental-SOURCE-1', status='FROZEN_BEFORE_REPLAY',
        real_batches=1,complete_trials=16,new_complex_trials=12,
        models=['R373','VirtualXY1000'],recorded_online_model='VirtualXY1000',
        comparison='same native input timestamps, B0, controls and raw commanded reference',
        paired_main='shared valid moving frames; Trial-equal raw and fixed-translation shape error',
        bias_diagnosis_not_model_correction=True, no_fit_or_selection=True,
        raw_inputs=entries, original_protocol_sha256=base.sha(base.OUT/'protocol.json'))
    file=DEST/'protocol.json'
    if file.exists(): assert base.read(file)==protocol, 'Supplemental frozen data protocol changed'
    else: base.dump(file,protocol)
    return entries


def analyze(entries):
    (DEST/'streams').mkdir(exist_ok=True)
    rows=[]
    for item in entries:
        a=np.load(base.OUT/'native/SOURCE_REPEAT/R373'/(item['id']+'.npz'))
        b=np.load(base.OUT/'native/SOURCE_REPEAT/VirtualXY1000'/(item['id']+'.npz'))
        assert np.array_equal(a['t_ns'],b['t_ns'])
        np.testing.assert_allclose(a['b'],b['b'],atol=0,rtol=0,equal_nan=True)
        assert np.array_equal(a['b0_epoch'],b['b0_epoch'])
        trial=base.ROOT/item['path']; meta=base.read(trial/'meta.json')
        seg=base.command_segments(base.read_rows(trial/'stage_commands.csv'))
        ref=base.command_reference(a['t_ns'],seg,base.reference_origin(meta));q=ref['xyz_mm']
        phase=np.array([seg[j]['phase'] if j>=0 else 'before_stage' for j in ref['command_segment_id']])
        primary,action=base.mask_for(item['mode'],phase,ref)
        finite=np.isfinite(a['b']).all(1); primary&=finite
        whole=action&(q[:,2]>=.4)&finite; shared=primary&a['ok']&b['ok']
        for label,p in [('R373',a),('VirtualXY1000',b)]:
            for scope,mask in [('primary_moving',primary),('whole_action',whole),('shared_full_chain',shared)]:
                r=dict(id=item['id'],group=item['group'],mode=item['mode'],model=label,scope=scope,
                       **base.metric(p['xyz'],q,p['ok'],mask,a['t_ns']))
                rows.append(r)
        np.savez_compressed(DEST/'streams'/(item['id']+'.npz'),t_ns=a['t_ns'],q=q,
            primary=primary,whole_action=whole,phase=phase,hint=a['hint'],safe=(q[:,2]<=-.2),
            R373_xyz=a['xyz'],R373_ok=a['ok'],VirtualXY1000_xyz=b['xyz'],VirtualXY1000_ok=b['ok'])
    df=pd.DataFrame(rows);df.to_csv(DEST/'trial_metrics.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for model in ['R373','VirtualXY1000']:
        sub=df[(df['mode']=='waypoint_xyz')&(df.model==model)&(df.scope=='shared_full_chain')]
        summaries.append(dict(model=model,trials=len(sub),**{k:float(sub[k].mean()) for k in
            ['xy_mean_mm','xyz_mean_mm','z_mae_mm','bias_x_mm','bias_y_mm','bias_z_mm','translation_removed_xyz_diagnostic_mm']}))
    report=dict(status='COMPLETED',complete_trials=16,complex_trials=12,summary=summaries,
        parity=base.read(base.OUT/'native/SOURCE_REPEAT/VirtualXY1000/completed.json')['parity'],
        raw_files_unchanged=True,same_stream=True,new_source_raw_hashes=base.read(DEST/'protocol.json')['raw_inputs'])
    base.dump(DEST/'report.json',report)
    for item in entries:
        assert all(base.sha(base.ROOT/item['path']/f)==h for f,h in item['raw_hashes'].items())
    base.dump(DEST/'work_state.json',dict(status='COMPLETED',report='report.json',raw_files_unchanged=True,hardware_operated=False))
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['inventory','R373','VirtualXY1000','analyze'],required=True)
    phase=parser.parse_args().phase
    entries=inventory()
    if phase=='inventory':
        base.snapshot('SOURCE_REPEAT',entries);return
    if phase=='analyze': analyze(entries);return
    journal=base.OUT/'native/SOURCE_REPEAT/native_protocol_prefix.jsonl'
    assert journal.exists()
    header=next(base.native._read_journal(journal))
    base.dump(DEST/(phase+'_state.json'),dict(status='RUNNING',pid=os.getpid(),model=phase,no_hardware=True))
    base.replay('SOURCE_REPEAT',entries,phase,journal,header)
    base.dump(DEST/(phase+'_state.json'),dict(status='COMPLETED',model=phase))


if __name__=='__main__': main()
