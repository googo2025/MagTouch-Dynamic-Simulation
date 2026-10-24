"""Portable principal-table, direction-hold and command-response reduction; Python/NumPy only."""
from pathlib import Path
import argparse,json,hashlib,csv,re,math
from collections import Counter
import numpy as np
from statistics_core import read_rows,reference_origin,command_segments,command_reference,mask_for,score,dwell_score
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def numerical_equal(a,b,where='root'):
 if isinstance(a,dict):
  assert set(a)==set(b),where
  for k in a:numerical_equal(a[k],b[k],where+'.'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),where
  for i,(x,y) in enumerate(zip(a,b)):numerical_equal(x,y,where+'.'+str(i))
 elif isinstance(a,(int,float)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=0,atol=1e-12,err_msg=where)
 else:assert a==b,(where,a,b)
def targeted(root,study):
 eligible=set(study['H0_comparable_uids']);cohort={'groups':study['groups']};records={e['uid']:e for e in study['records']};actions={r['uid']:r for r in read(root/'manifests/arrival_actions.json')};cache={};pins={}
 for uid in eligible:
  streams={}
  for arm in ['H0','H2']:
   f=root/'outputs'/arm/(uid+'.npz');r=read(f.with_suffix('.receipt.json'));assert sha(f)==r['output_sha256'];pins[str(f)]=r['output_sha256']
   with np.load(f,allow_pickle=False) as z:streams[arm]={k:z[k].copy() for k in ['t_ns','xyz','ok']}
  np.testing.assert_array_equal(streams['H0']['t_ns'],streams['H2']['t_ns']);t=streams['H0']['t_ns'];assert (np.diff(t)>0).all()
  valid={a:streams[a]['ok']&np.isfinite(streams[a]['xyz']).all(1) for a in streams};shared=valid['H0']&valid['H2']
  folder=root/'raw'/uid;meta=read(folder/'meta.json');ref=meta['acquisition']['printer_reference'];origin=np.array([ref[k] for k in ['sensor_center_x_mm','sensor_center_y_mm','contact_z_mm']])
  commands=list(csv.DictReader((folder/'stage_commands.csv').open(encoding='utf-8-sig')));position=[None]*3;moves=[]
  for i,c in enumerate(commands):
   if not c['command'].startswith('G1 '):continue
   old=position.copy()
   for n,k in enumerate(['x_mm','y_mm','z_mm']):
    if c.get(k) not in ['',None]:position[n]=float(c[k])
   duration=None
   for w in commands[i+1:]:
    if w['command'].startswith('G1 '):break
    if w['command']=='HOST_WAIT' and w['phase']==c['phase']+'_host_wait':
     m=re.search(r'EST=([\d.]+)s',w['response']);duration=float(m[1]) if m else None
   stamp=int(c.get('t_send_perf_ns') or c['t_perf_ns']);moves.append({'start':stamp,'arrival':stamp+round(duration*1e9) if duration is not None else None,'phase':c['phase'],'from':old,'to':position.copy()})
  cache[uid]=(streams,t,valid,shared,origin,moves)
 holds=[];responses=[];summaries=[]
 for gi,g in enumerate(cohort['groups']):
  required=g['incoming_projected_bearings_deg'];group_holds=[]
  for uid in g['record_uids']:
   if uid not in eligible:continue
   streams,t,valid,shared,origin,moves=cache[uid]
   windows=[w for w in actions[uid]['windows'] if all(abs(a-b)<1e-5 for a,b in zip(w['target_printer_xyz_mm'],g['target_printer_xyz_mm']))]
   for wi,w in enumerate(windows):
    age=(t-int(w['arrival_reference_ns']))/1e9;sample=(age>=.1)&(age<=.4)&(t<int(w['next_move_ns']));target=(np.array(w['target_printer_xyz_mm'])-origin)*[1,1,-1];direction=w['bearing_deg']
    for maskname in ['own','common']:
     for arm in streams:
      take=sample&(valid[arm] if maskname=='own' else shared);n=int(take.sum());r={'group':gi,'uid':uid,'window':wi,'direction_deg':direction,'arm':arm,'mask':maskname,'reference_frames':int(sample.sum()),'output_frames':n}
      if n>=2:
       x=streams[arm]['xyz'][take];r.update(mean_xyz_mm=x.mean(0).tolist(),raw_xy_mean_mm=float(np.linalg.norm((x-target)[:,:2],axis=1).mean()),xy_wobble_rms_mm=float(np.sqrt(np.mean(np.sum((x[:,:2]-x[:,:2].mean(0))**2,axis=1)))))
      else:r['status']='INSUFFICIENT_OUTPUTS_RETAINED'
      group_holds.append(r);holds.append(r)
    match=[k for k,m in enumerate(moves) if m['phase']==w['phase'] and m['arrival']==int(w['arrival_reference_ns'])];assert len(match)==1;k=match[0]
    if k==0 or any(v is None for v in moves[k-1]['from']+moves[k-1]['to']+moves[k]['from']+moves[k]['to']):continue
    prev,cur=moves[k-1],moves[k];a=np.array(prev['to'])-prev['from'];b=np.array(cur['to'])-cur['from']
    if np.linalg.norm(a[:2])<=1e-9 or np.linalg.norm(b[:2])<=1e-9 or abs(a[2])>1e-6 or abs(b[2])>1e-6:continue
    if np.dot(a[:2],b[:2])/(np.linalg.norm(a[:2])*np.linalg.norm(b[:2]))>-.95:continue
    t0=cur['start'];pre=(t>=t0-200_000_000)&(t<=t0-50_000_000)&shared;post=np.flatnonzero((t>=t0)&(t<=t0+1_500_000_000)&shared);unit=b[:2]/np.linalg.norm(b[:2]);nominal=.2/g['nominal_speed_mm_s']
    for arm in streams:
     r={'group':gi,'uid':uid,'window':wi,'arm':arm,'pre_common_samples':int(pre.sum()),'post_common_samples':len(post),'nominal_command_crossing_s':nominal,'crossing_observed':False}
     if pre.sum()<3:r['status']='INSUFFICIENT_PRE_SAMPLES_RETAINED'
     else:
      x=streams[arm]['xyz'][:,:2];base=np.median(x[pre],axis=0);progress=(x[post]-base)@unit;cross=None
      for n in range(len(post)-2):
       ids=post[n:n+3]
       if np.all(progress[n:n+3]>=.2) and np.all(np.diff(ids)==1) and np.all(np.diff(t[ids])<=250_000_000):cross=int(ids[0]);break
      if cross is None:r['status']='NO_SUSTAINED_CROSSING_WITHIN_FIXED_WINDOW'
      else:r.update(status='OBSERVED',crossing_observed=True,observed_crossing_s=(int(t[cross])-t0)/1e9,excess_vs_command_reference_s=(int(t[cross])-t0)/1e9-nominal)
     responses.append(r)
  for maskname in ['own','common']:
   for arm in ['H0','H2']:
    rows=[r for r in group_holds if r['arm']==arm and r['mask']==maskname];means=[];directions=[]
    for d in required:
     rr=[r for r in rows if r['direction_deg']==d and 'mean_xyz_mm' in r]
     if rr:means.append(np.mean([r['mean_xyz_mm'] for r in rr],axis=0));directions.append(d)
    good=[r for r in rows if 'mean_xyz_mm' in r];s={'group':gi,'depth_mm':g['depth_mm'],'nominal_speed_mm_s':g['nominal_speed_mm_s'],'arm':arm,'mask':maskname,'original_candidate_arrivals':g['arrivals'],'evaluated_arrivals':len(rows),'defined_arrivals':len(good),'required_directions':required,'directions_with_outputs':directions,'reference_frames':sum(r['reference_frames'] for r in rows),'output_frames':sum(r['output_frames'] for r in rows)}
    if good:s.update(raw_xy_mean_mm=float(np.mean([r['raw_xy_mean_mm'] for r in good])),xy_hold_wobble_rms_mm=float(np.mean([r['xy_wobble_rms_mm'] for r in good])))
    if len(directions)==len(required):
     m=np.array(means)[:,:2];s['between_direction_xy_spread_rms_mm']=float(np.sqrt(np.mean(np.sum((m-m.mean(0))**2,axis=1))))
    else:s['status']='INCOMPLETE_DIRECTION_OUTPUTS_RETAINED'
    summaries.append(s)
 response_summary=[]
 for arm in ['H0','H2']:
  rr=[r for r in responses if r['arm']==arm];obs=[r['excess_vs_command_reference_s'] for r in rr if r['crossing_observed']]
  response_summary.append({'arm':arm,'events':len(rr),'observed':len(obs),'censored_or_unresolved':len(rr)-len(obs),'median_excess_s':float(np.median(obs)) if obs else None,'statuses':dict(Counter(r['status'] for r in rr))})
 return {'hold_rows':holds,'hold_summaries':summaries,'response_rows':responses,'response_summary':response_summary}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();assert not out.exists(),'Use a new output directory'
 manifest=read(root/'FILE_MANIFEST.json')
 for r in manifest['files']:assert sha(root/r['path'])==r['sha256'],r['path']
 study=read(root/'manifests/study.json');expected=read(root/'expected/scientific_results.json');own=[];common=[];cache={}
 for e in study['records']:
  uid=e['uid'];folder=root/'raw'/uid
  for arm in ['H0','H2']:
   p=root/'outputs'/arm/(uid+'.npz')
   if not p.exists():assert arm=='H2' and uid not in study['H0_comparable_uids'];continue
   with np.load(p,allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
   seg=command_segments(read_rows(folder/'stage_commands.csv'));r=command_reference(v['t_ns'],seg,reference_origin(read(folder/'meta.json')));q=r['xyz_mm'];phase=np.array([seg[i]['phase'] if i>=0 else 'before_stage' for i in r['command_segment_id']]);primary,action=mask_for(e['mode'],phase,r);primary&=v['raw_input_valid'];action&=v['raw_input_valid']&(q[:,2]>=.4);dt=np.r_[1.,np.diff(v['t_ns'])/1e9];velocity=np.vstack((np.zeros(3),np.diff(q,axis=0)))/dt[:,None];dwell=action&(np.linalg.norm(velocity,axis=1)<.02)
   np.testing.assert_allclose(q,v['q'],rtol=0,atol=1e-12,equal_nan=True)
   for left,right in [(phase,v['phase']),(primary,v['primary']),(dwell,v['dwell'])]:np.testing.assert_array_equal(left,right)
   own.append({'uid':uid,'trial':e['trial'],'mode':e['mode'],'speed_mm_s':e['speed_mm_s'],'arm':arm,'primary':score(v['xyz'],q,v['ok'],primary,v['t_ns']),'dwell':dwell_score(v['xyz'],v['ok'],dwell,v['t_ns'],phase)})
   if uid in study['H0_comparable_uids']:cache[(uid,arm)]=v
 h0={r['uid']:r for r in own if r['arm']=='H0'};condition=sorted(u for u,r in h0.items() if r['primary'].get('raw_xy_mean_mm') is not None and r['primary']['raw_xy_mean_mm']<=study['H0_condition']['raw_XY_upper_mm'] and r['primary']['coverage']>=study['H0_condition']['minimum_coverage']);assert condition==sorted(study['H0_comparable_uids'])
 for uid in condition:
  x=cache[(uid,'H0')];y=cache[(uid,'H2')]
  for k in ['t_ns','q','primary','dwell','phase','raw_input_valid']:np.testing.assert_array_equal(x[k],y[k])
  shared=x['ok']&y['ok']&np.isfinite(x['xyz']).all(1)&np.isfinite(y['xyz']).all(1);e=next(e for e in study['records'] if e['uid']==uid);row={'uid':uid,'trial':e['trial'],'mode':e['mode'],'speed_mm_s':e['speed_mm_s'],'common_primary':{},'common_dwell':{}}
  for arm,v in [('H0',x),('H2',y)]:row['common_primary'][arm]=score(v['xyz'],v['q'],shared,v['primary'],v['t_ns']);row['common_dwell'][arm]=dwell_score(v['xyz'],shared,v['dwell'],v['t_ns'],v['phase'])
  common.append(row)
 aggregate={};keys=['raw_xy_mean_mm','centered_xy_diagnostic_mm','xy_residual_step_rms_mm','xy_residual_jump_rate_percent','coverage']
 for arm in ['H0','H2']:
  aggregate[arm]={}
  for k in keys:
   vals=[r['common_primary'][arm][k] for r in common if r['common_primary'][arm].get(k) is not None];aggregate[arm][k]={'record_count':len(vals),'mean':float(np.mean(vals)) if vals else None,'SD':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
 result={'own_record_statistics':own,'common_record_statistics':common,'common_equal_record_mean':aggregate,'targeted':targeted(root,study)}
 numerical_equal(result,expected)
 out.mkdir(parents=True);(out/'scientific_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 lines=['| Metric | H0 mean | H2 mean |','|---|---:|---:|']
 for k in keys:lines.append('| '+k+' | '+str(aggregate['H0'][k]['mean'])+' | '+str(aggregate['H2'][k]['mean'])+' |')
 (out/'Table_S5.md').write_text('\n'.join(lines)+'\n')
 from table_outputs import write_tables
 write_tables(out,result,study)
 receipt={'status':'ALL12_H0_6H2_AND_DIRECTION_RESPONSE_SAVED_STATISTICS_PASS','H0_records':12,'H2_records':6,'hold_arrivals':76,'joint_responses':75,'checked_files':len(manifest['files']),'maximum_absolute_numeric_tolerance':1e-12,'raw_command_references_reconstructed':True,'new_inference':0,'new_training':0,'scope':'The documented principal supplementary tables and complete scientific fields, not every historical training stage or independent physical metrology.'}
 (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
