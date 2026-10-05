"""Frozen two-head native replay, reusing audited R373 baseline, no fitting.

Shared R405 journal/reference/metric implementation is unchanged. This wrapper
only binds the two research heads, extended features, three-arm output location,
and explicit metadata. Both training arms must have stopped before test access.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/r460_source_real_history_refinement_20261004'
EVAL = OUT/'new_trajectory_native_replay'
ARMS = ('R373', 'history_position', 'history_motion_hold_anchor')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def prepare():
    """Only hash/copy/transport existing recordings on laptop, no scoring."""
    if socket.gethostname() == 'googo-Super-Server':raise RuntimeError('Prepare on laptop')
    if (EVAL/'input_manifest.json').exists():raise FileExistsError('Preserve test specification')
    inherited=read(ROOT/'analysis/r397_virtual_breakthrough_20261003/frozen_real_replay_spec.json')
    files={};entries=[];omitted=[]
    def add(p, expected=None):
        p=Path(p);h=sha(p)
        if expected is not None and h!=expected:raise ValueError('Original input changed: '+str(p))
        files[p.relative_to(ROOT).as_posix()]=dict(sha256=h,bytes=p.stat().st_size)
    add(Path(__file__))
    add(ROOT/'training/frozen_real_replay_r405.py')
    add(ROOT/'training/real_history_refinement_r460.py')
    add(ROOT/'analysis/r405_frozen_virtual_real_replay_20261003/recovery_v2/independent_audit.json')
    for rel,h in inherited['protocols'].items():
        add(ROOT/rel,h);entries.extend(read(ROOT/rel)['raw_inputs'])
    excluded=read(ROOT/'analysis/r387_soft_state_research_map_20261002/exclusions.json')['excluded_path_prefixes']
    train=read(ROOT/'analysis/r372_direct_source_plateau_20261001/data_manifest.json')['xy']['records']
    trained_paths={r['path'] for r in train}
    for item in entries:
        assert item['path'] not in trained_paths and not any(item['path'].startswith(p) for p in excluded)
        trial=ROOT/item['path'];assert read(trial/'meta.json')['status']=='completed'
        for name,h in item['raw_hashes'].items():
            path=trial/name
            if sha(path)!=h:raise ValueError('Recording modified')
            if name=='magnetic_raw.csv':omitted.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=h,reason='native journal is complete original ordered magnetic input'))
            else:add(path,h)
        add(ROOT/'analysis/r395_new_real_simulation_paper_20261003/native'/item['device']/'R373'/(item['id']+'.npz'))
    for batch,record in inherited['journal_prefixes'].items():
        add(ROOT/record['path'],record['sha256'])
        header=json.loads((ROOT/record['path']).open(encoding='utf8').readline())
        for row in header['models'].values():
            name=row['selection']['path'].replace('\\','/').rsplit('/',1)[-1]
            add(ROOT/'config'/name,row['checkpoint_or_profile_sha256'])
        add(ROOT/'analysis/r395_new_real_simulation_paper_20261003/native'/batch/'R373/completed.json')
    assert len(entries)==48 and sum(r['mode']=='waypoint_xyz' for r in entries)==36
    spec=dict(schema='R460-New-Real-Trajectories-1',entries=entries,journal_prefixes=inherited['journal_prefixes'],
        baseline_xy_sha256=inherited['models']['R373'],train_arms=list(ARMS[1:]),
        source_train_paths_disjoint=True,raw_inputs_all_checked_on_laptop=True,not_transferred=omitted,
        retained_primary='All36new complex paths;old12controls reported separately',
        same_actual_B0_dt_complete_prefix=True,baseline='Reuse audited immutable R373 full-native output; no repeated R373 training',
        new_arms_freeze_before_server_test_read=True,real_train_or_selection=False,
        previously_viewed_recordings=True,prospective_blind_test=False,
        shared_metric_source_sha256=sha(ROOT/'training/frozen_real_replay_r405.py'))
    dump(EVAL/'input_spec.json',spec);add(EVAL/'input_spec.json')
    manifest=dict(files=files,complete_trials=48,new_geometry_trials=36,
        baseline_independent_audit_sha256=sha(ROOT/'analysis/r405_frozen_virtual_real_replay_20261003/recovery_v2/independent_audit.json'))
    dump(EVAL/'input_manifest.json',manifest)
    with zipfile.ZipFile(EVAL/'test_inputs.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for rel in files:z.write(ROOT/rel,rel)
        z.write(EVAL/'input_manifest.json', (EVAL/'input_manifest.json').relative_to(ROOT).as_posix())
    dump(EVAL/'transport.json',dict(status='PACKAGED_NOT_EVALUATED',bytes=(EVAL/'test_inputs.zip').stat().st_size,sha256=sha(EVAL/'test_inputs.zip')))
    print(json.dumps(read(EVAL/'transport.json')))


def frozen_contract():
    if not (OUT/'training_complete.json').exists():raise RuntimeError('Both arms must freeze first')
    models={'R373':read(OUT/'protocol.json')['baseline_xy_sha256']}
    for arm in ARMS[1:]:
        f=read(OUT/arm/'freeze.json')
        assert f['status']=='COMPLETED_PLATEAU' and f['stop_reason']=='early_stop_min_lr_plateau'
        assert f['protocol_sha256']==sha(OUT/'protocol.json') and f['model_sha256']==sha(OUT/arm/'best_model.npz')
        models[arm]=f['model_sha256']
    return models


def run():
    assert socket.gethostname()=='googo-Super-Server'
    models=frozen_contract() # FIRST, before opening any new real test input
    manifest=read(EVAL/'input_manifest.json')
    for rel,r in manifest['files'].items():
        assert sha(ROOT/rel)==r['sha256'] and (ROOT/rel).stat().st_size==r['bytes'],rel
    spec=read(EVAL/'input_spec.json');spec['models']=models
    frozen_spec=OUT/'frozen_real_replay_spec.json'
    if frozen_spec.exists():assert read(frozen_spec)==spec
    else:dump(frozen_spec,spec)
    # Original process-only POSIX separator bootstrap, no scientific training.
    sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'training')]
    import training.domain_randomization_r369
    from training import frozen_real_replay_r405 as shared
    from real_history_refinement_r460 import install_research_head
    shared.OUT=EVAL;shared.R397=OUT;shared.ARMS=ARMS
    def artifact(path):return dict(path=Path(path))
    def install(predictor,value):
        file=value['path'];engine=predictor.engine
        with np.load(file,allow_pickle=False) as a:
            for decoder in (engine.memory_decoder,engine.candidate_decoder):
                np.testing.assert_array_equal(a['mean'][:53],decoder.xy_model.mean)
                np.testing.assert_array_equal(a['scale'][:53],decoder.xy_model.scale)
        install_research_head(engine,file)
        return engine.memory_decoder.xy_model,engine.candidate_decoder.xy_model
    shared.arrays=artifact;shared.install=install
    dump(EVAL/'execution_binding.json',dict(shared_source='training/frozen_real_replay_r405.py',
        shared_sha256=sha(ROOT/'training/frozen_real_replay_r405.py'),wrapper_sha256=sha(__file__),
        train_protocol_sha256=sha(OUT/'protocol.json'),models=models,
        changes='Only3arm binding and101D research XY feature/model overlay;original journal/B0/timestamps/reference/masks/metrics unchanged',
        baseline_reused=True,scientific_training=False,hardware_operated=False,production_changed=False))
    dump(EVAL/'work_state.json',dict(status='RUNNING',pid=os.getpid(),started=time.time(),all_models_frozen=True))
    for batch in spec['journal_prefixes']:
        group=[e for e in spec['entries'] if e['device']==batch]
        # The original audited baseline outputs are immutable; no fake model run.
        target=EVAL/'native'/batch/'R373';target.mkdir(parents=True,exist_ok=True)
        for entry in group:
            src=ROOT/'analysis/r395_new_real_simulation_paper_20261003/native'/batch/'R373'/(entry['id']+'.npz')
            dst=target/src.name
            if dst.exists():assert sha(src)==sha(dst)
            else:shutil.copyfile(src,dst)
        for arm in ARMS[1:]:
            completed=shared.replay(batch,arm,spec,group)
            # Correct inherited metadata; do not present the53D parser as101D.
            completed['research_head_overlay'].update(original53_normalization_preserved=True,
                added48_source_train_normalized=True,unchanged_normalization=False,
                corrected_phi_EMA_seconds=[.3,1.,4.],label='R460source-real refined research head, not R397virtual head')
            shared.dump(EVAL/'native'/batch/arm/'completed.json',completed)
            dump(EVAL/'work_state.json',dict(status='RUNNING',pid=os.getpid(),last_completed=[batch,arm],all_models_frozen=True))
    shared.analyze(spec,spec['entries'])
    report=read(EVAL/'report.json')
    report['comparison']='Frozen R373 + source-real multiscalehistory + matched motion/hold/anchor objectives;all36new geometries,raw/centered costs and all controls'
    report['baseline_reused']='Exact previous independently audited native R373 output, SHA checked before use'
    report['added_XYZ_output_smoothing']=False
    report['input_temporal_filtering']=True
    report['diagnostics_amplitude_reversal_sensitivity']='PENDING_SEPARATE_AUDIT_NOT_CLAIMED_COMPLETE'
    shared.dump(EVAL/'report.json',report)
    dump(EVAL/'work_state.json',dict(status='COMPLETED_FROZEN_RETROSPECTIVE_REAL_TEST',report_sha256=sha(EVAL/'report.json'),
        independently_recomputed=False,training_performed=False,hardware_operated=False,production_changed=False))


def launch():
    assert socket.gethostname()=='googo-Super-Server'
    frozen_contract()
    jobfile=EVAL/'job.json'
    if jobfile.exists():
        try:os.kill(read(jobfile)['pid'],0);print('EXISTING_REPLAY_NO_DUPLICATE');return
        except ProcessLookupError:raise RuntimeError('Previous replay ended; inspect or explicit resume, do not silently restart')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2',
        CUDA_VISIBLE_DEVICES='',MAGTOUCH_R460_SOURCE=str(ROOT))
    with (EVAL/'stdout.log').open('ab') as out,(EVAL/'stderr.log').open('ab') as err:
        job=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--run'],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    record=dict(pid=job.pid,cpu_threads=2,gpu=False,finite_jobs=6,status='LAUNCHED_NATIVE_NEW_TRAJECTORY_REPLAY',
                wrapper_sha256=sha(__file__),original_models_or_UI_changed=False,hardware_operated=False)
    dump(jobfile,record);print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--launch',action='store_true');a=parser.parse_args()
    if a.prepare:prepare()
    elif a.run:run()
    elif a.launch:launch()
    else:parser.error('Choose --prepare/--run/--launch')
