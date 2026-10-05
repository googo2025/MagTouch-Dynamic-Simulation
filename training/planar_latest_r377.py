"""Causal spatial-module controls on latest magnetic records at common 10 Hz.

F/M/G keep the historical decoder/gate contract; the last column is cached
R373 full-chain replay on the exact same input cadence. Shared accepted frames
avoid a coverage-driven error reduction. No weight or threshold is fitted.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from training.probe_r136_gate_fullstream_replay import make_predictor
from training.probe_had239_local_forward_r104 import load_profile
from training.evaluate_branch_inverse_r139 import FIVE_CONFIG,FIVE_KEY
from training.train_source_to_plateau_r372 import dump,sha
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'analysis/r374_matched_real_virtual_20261002'
OUT=ROOT/'analysis/r377_offline_continuation_20261002'
CASES=('source_000005','source_000006','target_000014','target_000015')

def case(id):
    with np.load(SRC/'streams'/f'{id}_matched.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    results={'N':a['real_new'].copy()};valids={'N':a['real_new_ok'].copy()}
    profile=load_profile(FIVE_CONFIG,FIVE_KEY)
    for name in ('F','M','G'):
        engine=make_predictor(profile,np.zeros(15),70.,name).engine
        xyz=np.full((len(a['t']),3),np.nan);ok=np.zeros(len(a['t']),bool)
        for i,t in enumerate(a['t']):
            if a['reset'][i] or not a['valid'][i]:engine.set_baseline(np.zeros(15))
            if not a['valid'][i] or not np.isfinite(a['q'][i]).all():continue
            r=engine.push(int(round(t*1e9))+10_000_000_000,a['b'][i]);ok[i]=bool(r['valid'])
            if ok[i]:xyz[i]=r['xyz']
        results[name]=xyz;valids[name]=ok
        print(id,name,'complete',flush=True)
    common=a['primary'].copy()
    for k in results:common&=valids[k]
    assert common.sum()>=.85*a['primary'].sum(),(id,common.sum(),a['primary'].sum())
    scores={};centered={}
    for k,xyz in results.items():
        d=xyz[common,:2]-a['q'][common,:2]
        scores[k]=float(np.linalg.norm(d,axis=1).mean())
        centered[k]=float(np.linalg.norm(d-d.mean(0),axis=1).mean())
    np.savez_compressed(OUT/f'{id}_planar.npz',t=a['t'],q=a['q'],common=common,**results)
    row={'id':id,'device':id.split('_')[0],'mode':'angle_scan' if id in (CASES[0],CASES[2]) else 'circle_scan',
        'eligible_frames':int(a['primary'].sum()),'common_frames':int(common.sum()),'xy_raw_mean_mm':scores,
        'xy_centered_mean_mm':centered,'matched_stream_sha256':sha(SRC/'streams'/f'{id}_matched.npz')}
    dump(OUT/f'{id}_planar.json',row);print(row,flush=True);return row

def main():
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(case,CASES))
    groups={}
    for d in ('source','target'):
        rr=[r for r in rows if r['device']==d]
        raw={k:float(np.mean([r['xy_raw_mean_mm'][k] for r in rr])) for k in ('F','M','G','N')}
        centered={k:float(np.mean([r['xy_centered_mean_mm'][k] for r in rr])) for k in raw}
        groups[d]={'raw':raw,'centered':centered,'memory_change_pct':100*(raw['M']/raw['F']-1),
            'memory_centered_change_pct':100*(centered['M']/centered['F']-1)}
    dump(OUT/'planar_ablation.json',{'status':'COMPLETE','controlled_hz':10,'rows':rows,'groups':groups,
        'fitting':False,'common_masks':True,'alignment':False,
        'scope':'F/M/G spatial decoder controls under common historical gate, N full R373 chain; final-column difference includes shared causal state/permission policy, not a neural-head-only ablation'})
    print(json.dumps(groups,indent=2),flush=True)
if __name__=='__main__':main()
