"""Paper evidence from R373 native online logs; no fitting or data mutation.

The paired Z control is the logged shared inverse before the *complete* depth
path. It is not an increment-only ablation. Primary masks are reused verbatim
from the completed R374 audit. All headline values remain unregistered.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from training.audit_r354_logged_depth_path import summary, steps
from training.train_source_to_plateau_r372 import dump, sha

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'analysis/r374_matched_real_virtual_20261002'
OUT=ROOT/'analysis/r377_offline_continuation_20261002'

def load(row):
    with np.load(SRC/'streams'/f"{row['id']}_online.npz",allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}

def main():
    report=json.loads((SRC/'report.json').read_text(encoding='utf8'))
    rows=[];biases={}
    for device in ('source','target'):
        grid=next(r for r in report['records'] if r['id']==f"{device}_{2 if device=='source' else 11:06d}")
        biases[device]=grid['online']['primary']['bias_xyz_mm']
    for row in report['records']:
        for name,h in row['hashes'].items():
            assert sha(ROOT/row['path']/name)==h,(row['id'],name)
        if row['mode']=='none':continue
        a=load(row);frame=pd.read_csv(ROOT/row['path']/'live_predictions.csv',
            usecols=['extra_outputs_json','t_prediction_ns'],low_memory=False)
        np.testing.assert_array_equal(frame.t_prediction_ns.to_numpy(np.int64),a['time'])
        before=np.array([float(json.loads(v).get('original_z_mm',np.nan)) for v in frame.extra_outputs_json])
        if not np.isfinite(before[a['valid']]).all():raise ValueError('Pre-depth missing on valid frame')
        common=a['primary']&a['valid'];pre=summary(before,a['q'],common);final=summary(a['xyz'][:,2],a['q'],common)
        delta=a['xyz'][common]-a['q'][common]-np.asarray(biases[row['device']])
        pairs_pre=pairs_final=None
        if row['mode'] in ('angle_scan','circle_scan'):
            pairs_pre=steps(before,a['time'],a['q'],a['action'],a['valid'])
            pairs_final=steps(a['xyz'][:,2],a['time'],a['q'],a['action'],a['valid'])
            assert pairs_pre['pairs']==pairs_final['pairs']
        rows.append({'id':row['id'],'device':row['device'],'mode':row['mode'],
            'pre_depth':pre,'final_online':final,'z_mae_change_pct':100*(final['z_mae_mm']/pre['z_mae_mm']-1),
            'steps_pre':pairs_pre,'steps_final':pairs_final,
            'grid_translation_diagnostic':{'bias_xyz_mm':biases[row['device']],
                'xy_mean_mm':float(np.linalg.norm(delta[:,:2],axis=1).mean()),'z_mae_mm':float(np.abs(delta[:,2]).mean())},
            'online_primary':row['online']['primary'],
            'stream_sha256':sha(SRC/'streams'/f"{row['id']}_online.npz")})
        np.savez_compressed(OUT/f"{row['id']}_depth.npz",pre_depth=before)
    result={'schema':'R377-Latest-Native-Online-Paper-1','status':'COMPLETE',
        'trials':len(report['records']),'prediction_frames':sum(r['online']['frames'] for r in report['records']),
        'safe_frames':sum(r['online']['safe_frames'] for r in report['records']),
        'safe_outputs':sum(r['online']['safe_outputs'] for r in report['records']),
        'safe_without_hint_frames':sum(r['online']['safe_without_hint_frames'] for r in report['records']),
        'safe_without_hint_outputs':sum(r['online']['safe_without_hint_outputs'] for r in report['records']),
        'grid_biases':biases,'rows':rows,
        'control':'logged original_z_mm vs final complete causal depth path on identical native timestamps, XY and validity',
        'primary_masks':'R374 predefined masks reused unchanged, nominal depth>=0.4 mm; coverage is not physical contact recall',
        'main_scores_unregistered':True,'raw_hashes_unchanged':True}
    dump(OUT/'native_paper_evidence.json',result)
    print(json.dumps({k:result[k] for k in ('trials','prediction_frames','safe_frames','safe_outputs')},indent=2))
    for r in rows:
        print(r['id'],r['mode'],f"Z {r['pre_depth']['z_mae_mm']:.3f}->{r['final_online']['z_mae_mm']:.3f} ({r['z_mae_change_pct']:+.1f}%)",r['steps_pre'],r['steps_final'],flush=True)

if __name__=='__main__':main()
