"""Portable saved-output reduction of the formal GRU and eight-sequence studies.

Run from an extracted supplement. Only Python and NumPy are required.
No model fitting, inference, network access, hardware or private directories.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from statistics_core import read_rows,reference_origin,command_segments,command_reference,mask_for,score,dwell_score,summarize

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def mean(values):
 v=[v for v in values if v is not None];return float(np.mean(v)) if v else None
def reference(folder,t,mode,finite):
 c=command_segments(read_rows(folder/'stage_commands.csv'));r=command_reference(t,c,reference_origin(read(folder/'meta.json')));q=r['xyz_mm']
 phase=np.array([c[i]['phase'] if i>=0 else 'before_stage' for i in r['command_segment_id']]);primary,action=mask_for(mode,phase,r)
 primary&=finite;action=action&(q[:,2]>=.4)&finite;dt=np.r_[1.,np.diff(t)/1e9];v=np.vstack((np.zeros(3),np.diff(q,axis=0)))/dt[:,None]
 return q,phase,primary,action&(np.linalg.norm(v,axis=1)<.02)
def reduce_record(a,q,phase,primary,dwell):
 return {'primary':score(a['xyz'],q,a['ok'],primary,a['t_ns']),'dwell':dwell_score(a['xyz'],a['ok'],dwell,a['t_ns'],phase)}
def numerical_equal(a,b,where='root'):
 if isinstance(a,dict):
  assert a.keys()==b.keys(),where
  for k in a:numerical_equal(a[k],b[k],where+'.'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),where
  for i,(x,y) in enumerate(zip(a,b)):numerical_equal(x,y,where+'.'+str(i))
 elif isinstance(a,(int,float)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,err_msg=where)
 else:assert a==b,(where,a,b)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 root=args.root.resolve();out=args.output.resolve();assert not out.exists(),'Use a new output directory'
 plan=read(root/'manifests/study.json');files=read(root/'FILE_MANIFEST.json')
 for row in files['files']:assert sha(root/row['path'])==row['sha256'],row['path']
 details=[];pairs=[]
 for entry in plan['formal_records']:
  domain=entry['device'];id=entry['id'];folder=root/'raw/formal'/id
  baseline=load(root/'outputs/formal'/domain/'R373'/(id+'.npz'));t=baseline['t_ns'];q,phase,primary,dwell=reference(folder,t,entry['mode'],np.isfinite(baseline['b']).all(1))
  for seed in [7,17,27]:
   controls=['R373']+[f'{c}_seed{seed}' for c in ['H0_matched_L1','multi_L1','multi_L2']];names=controls+[f'GRU_seed{seed}'];streams={}
   for arm in names:
    a=baseline if arm=='R373' else load(root/'outputs/formal'/domain/arm/(id+'.npz'));streams[arm]=a
    np.testing.assert_array_equal(a['t_ns'],t);np.testing.assert_array_equal(a['b0_epoch'],baseline['b0_epoch']);np.testing.assert_allclose(a['b'],baseline['b'],rtol=0,atol=0,equal_nan=True)
    r=reduce_record(a,q,phase,primary,dwell);details.append({'domain':domain,'seed':seed,'id':id,'mode':entry['mode'],'arm':arm,**r})
   b=streams[f'GRU_seed{seed}']
   for arm in controls:
    a=streams[arm];shared=a['ok']&b['ok']&np.isfinite(a['xyz']).all(1)&np.isfinite(b['xyz']).all(1)
    pairs.append({'domain':domain,'seed':seed,'id':id,'mode':entry['mode'],'baseline':arm,'enhanced':f'GRU_seed{seed}','baseline_common':score(a['xyz'],q,a['ok'],primary&shared,t),'enhanced_common':score(b['xyz'],q,b['ok'],primary&shared,t)})
 summaries=[];order=[];metrics=['raw_xy_mean_mm','centered_xy_diagnostic_mm','xy_residual_step_rms_mm','xy_residual_jump_rate_percent','dwell_xy_wobble_rms_mm','coverage']
 for domain in ['SOURCE','SOURCE_REPEAT','Yellow']:
  for seed in [7,17,27]:
   for arm in ['R373']+[f'{c}_seed{seed}' for c in ['H0_matched_L1','multi_L1','multi_L2','GRU']]:
    rr=[r for r in details if r['domain']==domain and r['seed']==seed and r['arm']==arm and r['mode']=='waypoint_xyz'];assert len(rr)==12
    s=summarize([r['primary'] for r in rr]);s.update(domain=domain,seed=seed,arm=arm,dwell_xy_wobble_rms_mm=mean(r['dwell']['mean_xy_wobble_rms_mm'] for r in rr),total_reference_frames=sum(r['primary']['reference_frames'] for r in rr),total_output_frames=sum(r['primary']['output_frames'] for r in rr),zero_valid_output_records=sum(r['primary']['output_frames']==0 for r in rr));summaries.append(s)
  for cfg in ['H0_matched_L1','multi_L1','multi_L2','GRU']:
   rr=[r for r in summaries if r['domain']==domain and r['arm'].startswith(cfg+'_seed')];assert len(rr)==3
   order.append({'domain':domain,'configuration':cfg,'order_repetitions':3,'metrics':{m:{'mean':mean(r[m] for r in rr),'sample_std':float(np.std([r[m] for r in rr],ddof=1)) if all(r[m] is not None for r in rr) else None} for m in metrics}})
 fixed={};same_details=[];same_pairs=[]
 for entry in plan['same_condition_records']:
  uid=entry['uid'];folder=root/'raw/same_condition'/uid;a=load(root/'outputs/same_condition/H0'/(uid+'.npz'));b=load(root/'outputs/same_condition/H2'/(uid+'.npz'));t=a['t_ns']
  np.testing.assert_array_equal(t,b['t_ns']);np.testing.assert_allclose(a['b'],b['b'],rtol=0,atol=0,equal_nan=True)
  q,phase,primary,dwell=reference(folder,t,'waypoint_xyz',np.isfinite(a['b']).all(1));shared=a['ok']&b['ok']
  for arm,x in [('H0',a),('H2',b)]:same_details.append({'uid':uid,'trial':entry['trial'],'arm':arm,**reduce_record(x,q,phase,primary,dwell)})
  same_pairs.append({'uid':uid,'trial':entry['trial'],'H0':score(a['xyz'],q,a['ok'],primary&shared,t),'H2':score(b['xyz'],q,b['ok'],primary&shared,t)})
 for arm in ['H0','H2']:
  rr=[r for r in same_details if r['arm']==arm];assert len(rr)==8
  fixed[arm]={**summarize([r['primary'] for r in rr]),'dwell_xy_wobble_rms_mm':mean(r['dwell']['mean_xy_wobble_rms_mm'] for r in rr)}
 result={'GRU_order_summaries':order,'GRU_summaries':summaries,'GRU_record_details':details,'GRU_common_output_pairs':pairs,'same_condition_summary':fixed,'same_condition_record_details':same_details,'same_condition_common_output_pairs':same_pairs}
 numerical_equal(result,read(root/'expected/scientific_results.json'))
 out.mkdir(parents=True);(out/'scientific_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 table=['| Domain | Configuration | Raw XY mm | Centered diagnostic mm | Step RMS mm | Jump % | Dwell RMS mm | Coverage % |','|---|---|---:|---:|---:|---:|---:|---:|']
 for row in order:
  cols=[]
  for metric in metrics:
   x=row['metrics'][metric];factor=100 if metric=='coverage' else 1;digits=2 if metric in ['coverage','xy_residual_jump_rate_percent'] else 3
   cols.append('NA' if x['mean'] is None else f"{factor*x['mean']:.{digits}f} ± {factor*x['sample_std']:.{digits}f}")
  table.append('| '+row['domain']+' | '+row['configuration']+' | '+' | '.join(cols)+' |')
 (out/'GRU_table.md').write_text('\n'.join(table)+'\n')
 table=['| Model | Raw XY mm | Centered diagnostic mm | Step RMS mm | Jump % | Dwell RMS mm | Coverage % |','|---|---:|---:|---:|---:|---:|---:|']
 for arm,s in fixed.items():
  vals=[f"{s[m]:.3f}" for m in metrics[:3]]+[f"{s[metrics[3]]:.2f}",f"{s[metrics[4]]:.3f}",f"{100*s['coverage']:.2f}"]
  table.append('| '+arm+' | '+' | '.join(vals)+' |')
 (out/'Same_condition_Table9.md').write_text('\n'.join(table)+'\n')
 receipt={'status':'ALL_FORMAL_GRU_AND_SAME_CONDITION_SAVED_OUTPUT_STATISTICS_PASS','formal_unique_sequences':48,'GRU_models':3,'GRU_model_sequence_replays':144,'same_condition_sequences':8,'same_condition_model_sequence_replays':16,'full_record_and_common_output_fields_match_atol':1e-12,'checked_files':len(files['files']),'training_updates':0,'inference_calls':0,'source_of_reference':'Recorded commands and EST timing, not independent physical metrology'}
 (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
