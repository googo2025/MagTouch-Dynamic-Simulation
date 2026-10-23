"""Reconstruct only the inherited 32 TRAIN / 3 DEV contexts; no evaluation reads."""
from pathlib import Path
import json,os,sys,time,traceback
from mm_gru_common_20261024 import *

def main():
 setup(9);guard();plan=json.loads((OUT/'protocol.json').read_text())
 for rel,h in plan['immutable_pins'].items():assert sha(ROOT/rel)==h,rel
 for name,h in plan['sources'].items():assert sha(OUT/name)==h,name
 assert not (OUT/'inputs').exists(),'Preserve existing preparation'
 (OUT/'inputs').mkdir()
 import numpy as np
 from training import retrain_r133_coverage_xy as old
 from training.probe_causal_xy_neural_r164 import guarded_positions
 from training.probe_had239_local_forward_r104 import ForwardAtlas,load_profile
 from training.real_history_refinement_r460 import augment_existing
 man=json.loads((OLD/'data_manifest.json').read_text());a=load(OLD/man['train']['file'])
 mean=a['mean'][:53];scale=a['scale'][:53]
 profile=load_profile(old.FIVE_CONFIG,old.FIVE_KEY);samples=profile['samples']
 atlas=ForwardAtlas(np.array([[s['x'],s['y'],s['depth']] for s in samples]),np.array([s['delta'] for s in samples]),'unit_logamp')
 memory=json.loads(old.FIVE_REPORT.read_text());basis=np.asarray(memory['basis']);strength=float(memory['selection']['selected']['strength'])
 contexts=np.zeros((11415,STEPS,54));lengths=np.ones(11415,dtype=np.int64)
 prev_contexts=np.zeros_like(contexts);prev_lengths=np.ones_like(lengths);receipts=[]
 for rec in man['records']:
  n=rec['trial'];assert rec['role'] in ['train','validation']
  if n in old.BASE_TRAIN+old.BASE_VALID:
   row=old.load_old(n,atlas,basis,strength);row['guarded']=guarded_positions(row,atlas)
  else:row=old.load_new(ROOT/rec['path'],atlas,basis,strength)
  x=old.features(row,atlas);ext=augment_existing(x,row['time'],row['valid'])
  if rec['role']=='train':
   ids=np.flatnonzero(a['owners']==n);selected=a['source_index'][ids]
   np.testing.assert_allclose(ext[selected],a['x'][ids],atol=1e-10,rtol=0)
   np.testing.assert_array_equal(row['time'][selected],a['time'][ids])
   c,l=window_array(x,row['time'],row['valid'],selected,mean,scale);contexts[ids]=c;lengths[ids]=l
   pairids=ids[a['pair'][ids]];previous=a['source_index'][pairids]-1
   if len(previous):
    np.testing.assert_allclose(ext[previous],a['prev_x'][pairids],atol=1e-10,rtol=0)
    pc,pl=window_array(x,row['time'],row['valid'],previous,mean,scale);prev_contexts[pairids]=pc;prev_lengths[pairids]=pl
  else:
   entry=next(e for e in man['validation'] if e['trial']==n);v=load(OLD/entry['file']);selected=v['source_index']
   np.testing.assert_allclose(ext[selected],v['x'],atol=1e-10,rtol=0)
   c,l=window_array(x,row['time'],row['valid'],selected,mean,scale)
   np.savez_compressed(OUT/'inputs'/f'validation_{n}.npz',context=c,length=l)
  receipts.append({'trial':n,'role':rec['role'],'native_frames':len(x),'valid_context_frames':int(row['valid'].sum()),'supervised_rows':len(selected),'length_min':int(l.min()),'length_max':int(l.max()),'length_mean':float(l.mean())})
  print('PREPARED',n,rec['role'],len(selected),flush=True);guard()
 anchor=np.flatnonzero(a['group']==2);contexts[anchor,0,:53]=(a['x'][anchor,:53]-mean)/scale
 assert len(anchor)==565 and [int((a['group']==g).sum()) for g in range(3)]==[7217,3633,565]
 np.testing.assert_allclose(contexts[np.arange(11415),lengths-1,:53],(a['x'][:,:53]-mean)/scale,atol=1e-10,rtol=0)
 np.savez_compressed(OUT/'inputs/train_contexts.npz',context=contexts,length=lengths,prev_context=prev_contexts,prev_length=prev_lengths)
 inputs={p.name:sha(p) for p in (OUT/'inputs').glob('*.npz')}
 dump(OUT/'preparation_receipt.json',{'status':'PASS','records':receipts,'TRAIN_records':32,'DEV_records':3,'supervised_rows':11415,'native_prefix_first':True,'original_features_and_targets_unchanged':True,'context_parity_atol':1e-10,'evaluation_records_read':0,'input_hashes':inputs,'protocol_sha256':sha(OUT/'protocol.json'),'RSS_bytes':rss()})
 print('PREPARATION_PASS',json.dumps({'files':len(inputs),'rows':11415,'evaluation_records_read':0}),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump(OUT/'preparation_failure.json',{'status':'FAILED','traceback':traceback.format_exc()});raise
