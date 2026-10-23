from pathlib import Path
from dataclasses import asdict
import os,sys,json,time,random,argparse,resource,traceback,copy
from mm_gru_common_20261024 import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--cpu',type=int,required=True);args=ap.parse_args()
 setup(args.cpu);plan=json.loads((OUT/'protocol.json').read_text());psha=sha(OUT/'protocol.json')
 assert args.seed in plan['order_seeds'] and args.cpu in plan['training_cpus']
 for name,h in plan['sources'].items():assert sha(OUT/name)==h,name
 receipt=json.loads((OUT/'preparation_receipt.json').read_text());assert receipt['status']=='PASS' and receipt['protocol_sha256']==psha
 for name,h in receipt['input_hashes'].items():assert sha(OUT/'inputs'/name)==h,name
 for rel,h in plan['immutable_pins'].items():assert sha(ROOT/rel)==h,rel
 folder=OUT/f'seed{args.seed}';folder.mkdir(exist_ok=False)
 dump(folder/'registration.json',{'pid':os.getpid(),'seed':args.seed,'cpu':args.cpu,'protocol_sha256':psha,'worker_sha256':sha(__file__),'GPU_used':False})
 import numpy as np
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 torch.manual_seed(plan['common_gru_initialization_seed']);random.seed(args.seed)
 from training.real_history_refinement_r460 import Plateau,forward
 a=load(OLD/'inputs/train.npz');mean=a['mean'][:53];scale=a['scale'][:53]
 c=load(OUT/'inputs/train_contexts.npz');guard();model=make_model(a)
 assert not torch.cuda.is_initialized()
 def tensor(v):return torch.as_tensor(v,dtype=torch.int64 if v.dtype.kind in 'iu' else torch.float64)
 data={k:tensor(v) for k,v in dict(u=(a['x'][:,:53]-mean)/scale,context=c['context'],length=c['length'],prev_u=(a['prev_x'][:,:53]-mean)/scale,prev_context=c['prev_context'],prev_length=c['prev_length'],y=a['y'],guard=a['guard'],prev_guard=a['prev_guard'],dq=a['dq'],weight=a['weight']).items()}
 for name,mask in [('pair_weight',a['pair']),('hold_weight',a['hold_pair']),('anchor_weight',a['group']==2)]:data[name]=tensor(mask.astype(float)*len(mask)/max(int(mask.sum()),1))
 del c
 vals=[]
 for item in json.loads((OLD/'data_manifest.json').read_text())['validation']:
  v=load(OLD/item['file']);vc=load(OUT/'inputs'/f"validation_{item['trial']}.npz")
  vals.append((item,v,{'u':tensor((v['x'][:,:53]-mean)/scale),'context':tensor(vc['context']),'length':tensor(vc['length'])}))
 anchor_ids=np.flatnonzero(a['group']==2);anchors={k:v[anchor_ids] for k,v in data.items() if k in ['u','context','length']};anchors['context']=anchors['context'][:,:1]
 def predict(d):
  with torch.no_grad():
   return np.vstack([model(d['u'][s:s+256],d['context'][s:s+256],d['length'][s:s+256]).numpy() for s in range(0,len(d['u']),256)])
 def val_metrics():
  rows=[]
  for item,v,d in vals:
   pred=v['guard']+predict(d);err=pred-v['q'];use=v['core'];residual=err[use];core_ids=np.flatnonzero(use)
   near=((np.diff(v['time'][core_ids])>0)&(np.diff(v['time'][core_ids])<=250_000_000)&(np.diff(v['source_index'][core_ids])==1))
   steps=np.linalg.norm(np.diff(residual,axis=0)[near],axis=1);holdids=np.flatnonzero(v['hold'])
   groups=np.split(holdids,np.flatnonzero((np.diff(holdids)!=1)|(np.diff(v['time'][holdids])>250_000_000))+1) if len(holdids) else []
   hold_values=[]
   for g in groups:
    if len(g)>=3 and v['time'][g[-1]]-v['time'][g[0]]>=200_000_000:
     centered=pred[g]-pred[g].mean(0);hold_values.append(float(np.sqrt(np.mean(np.sum(centered**2,axis=1)))))
   rows.append(dict(trial=item['trial'],mode=item['mode'],frames=int(use.sum()),raw_xy_mm=float(np.linalg.norm(residual,axis=1).mean()),centered_xy_mm=float(np.linalg.norm(residual-residual.mean(0),axis=1).mean()),bias_xy_mm=residual.mean(0).tolist(),residual_step_rms_mm=float(np.sqrt(np.mean(steps**2))) if len(steps) else 0.,jump_over_0p2_percent=float(100*np.mean(steps>.2)) if len(steps) else 0.,hold_xy_rms_mm=float(np.mean(hold_values)) if hold_values else 0.,hold_blocks=len(hold_values)))
  anchor_rms=float(np.sqrt(np.mean(np.sum(predict(anchors)**2,axis=1))))
  objective=float(np.mean([r['raw_xy_mm']+.1*r['residual_step_rms_mm']+.1*r['hold_xy_rms_mm'] for r in rows])+.1*anchor_rms)
  return objective,rows,anchor_rms
 def loss_for(batch):
  pred=model(batch['u'],batch['context'],batch['length'])
  loss=((pred-batch['y']).square().sum(1)*batch['weight']).mean()+model.regularizer()
  previous=model(batch['prev_u'],batch['prev_context'],batch['prev_length'])
  increment=batch['guard']+pred-batch['prev_guard']-previous-batch['dq']
  for name,weightkey in [('increment','pair_weight'),('hold','hold_weight')]:loss=loss+plan['constraint_weights'][name]*(increment.square().sum(1)*batch[weightkey]).mean()
  return loss+plan['constraint_weights']['anchor']*(pred.square().sum(1)*batch['anchor_weight']).mean()
 # Qualification consumes no optimizer updates and restores every tensor/RNG.
 before=copy.deepcopy(model.state_dict());rng_before=torch.get_rng_state().clone()
 initial_pred=predict(data);base=forward([a[k][:53] if k=='w1' else a[k] for k in KEYS],(a['x'][:,:53]-mean)/scale)[2]
 np.testing.assert_allclose(initial_pred,base,rtol=0,atol=1e-11)
 with torch.no_grad():model.readout.weight.copy_(torch.linspace(-.02,.02,32,dtype=torch.float64).reshape(2,16))
 ix=np.r_[np.arange(8),anchor_ids[:2]];d={k:v[ix] for k,v in data.items() if k in ['u','context','length']}
 expected=predict(d);arrays={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
 actual=numpy_prediction(arrays,d['u'].numpy(),d['context'].numpy(),d['length'].numpy())
 np.testing.assert_allclose(actual,expected,rtol=1e-11,atol=1e-11)
 # Padding must have no influence on a causal finite-window prediction.
 pad=d['context'].clone()
 for i,n in enumerate(d['length']):pad[i,n:]=123.
 with torch.no_grad():np.testing.assert_allclose(model(d['u'],pad,d['length']).numpy(),expected,rtol=0,atol=1e-11)
 model.load_state_dict(before);torch.set_rng_state(rng_before)
 t0=time.monotonic();c0=time.process_time();batch={k:v[:256] for k,v in data.items()};loss_for(batch).backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
 model.zero_grad(set_to_none=True)
 for k,v in model.state_dict().items():assert torch.equal(v,before[k]),k
 assert torch.equal(torch.get_rng_state(),rng_before)
 profile_wall=time.monotonic()-t0;profile_cpu=time.process_time()-c0
 obj0,rows0,anchor0=val_metrics();guard()
 dump(folder/'qualification.json',{'status':'PASS','initial_training_rows':11415,'H0_initial_function_parity_atol':1e-11,'torch_numpy_recurrent_parity_atol':1e-11,'padding_invariance_atol':1e-11,'optimizer_updates':0,'parameters':sum(p.numel() for p in model.parameters()),'common_initialization_seed':plan['common_gru_initialization_seed'],'order_seed':args.seed,'profile_batch_rows':256,'profile_wall_seconds':profile_wall,'profile_CPU_seconds':profile_cpu,'RSS_bytes':rss(),'initial_DEV_objective':obj0,'initial_DEV_rows':rows0})
 save_model(folder/'initial_model.npz',model,mean,scale);save_model(folder/'best_model.npz',model,mean,scale)
 optimizer=torch.optim.Adam(model.parameters(),lr=plan['initial_lr'],eps=1e-8,foreach=False);ctrl=Plateau(lr=plan['initial_lr']);rng=np.random.default_rng(args.seed)
 best_obj=obj0;best_epoch=0;free_obj=obj0;free_epoch=0;updates=0;start=time.monotonic();cpu0=time.process_time();stop='max_epochs_budget_truncated'
 hist=(folder/'history.jsonl').open('x');history_sha=hashlib.sha256()
 try:
  for epoch in range(1,plan['max_epochs']+1):
   guard();lr=ctrl.lr
   for group in optimizer.param_groups:group['lr']=lr
   order=rng.permutation(11415);seen=0;loss_sum=0.;t0=time.monotonic()
   for ids in np.array_split(order,45):
    ix=torch.as_tensor(ids);batch={k:v[ix] for k,v in data.items()};optimizer.zero_grad(set_to_none=True);loss=loss_for(batch)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
    loss.backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
    optimizer.step();updates+=1;seen+=len(ids);loss_sum+=float(loss.detach())*len(ids)
   assert seen==11415 and updates==45*epoch
   objective,rows,anchor=val_metrics()
   qualified=(np.mean([r['raw_xy_mm'] for r in rows])<=np.mean([r['raw_xy_mm'] for r in rows0])+.005 and all(r['raw_xy_mm']<=b['raw_xy_mm']+.02 for r,b in zip(rows,rows0)) and anchor<=anchor0+.02)
   if objective<free_obj:free_obj=objective;free_epoch=epoch;save_model(folder/'unconstrained_best_model.npz',model,mean,scale)
   if qualified and objective<best_obj:best_obj=objective;best_epoch=epoch;save_model(folder/'best_model.npz',model,mean,scale)
   event=ctrl.observe(objective)
   rec={'epoch':epoch,'updates':updates,'lr_used':lr,'next_lr':ctrl.lr,'train_objective':loss_sum/seen,'validation_objective':objective,'validation':rows,'anchor_rms':anchor,'qualified':bool(qualified),'best_epoch':best_epoch,'best_objective':best_obj,'scheduler_event':event,'epoch_wall_seconds':time.monotonic()-t0,'elapsed_wall_seconds':time.monotonic()-start,'elapsed_cpu_seconds':time.process_time()-cpu0,'RSS_bytes':rss()}
   line=json.dumps(rec,allow_nan=False)+'\n';hist.write(line);hist.flush();os.fsync(hist.fileno());history_sha.update(line.encode())
   state={'epoch':epoch,'updates':updates,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'controller':asdict(ctrl),'numpy_order_rng':rng.bit_generator.state,'numpy_global_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'CUDA_rng':None,'best_obj':best_obj,'best_epoch':best_epoch,'free_obj':free_obj,'free_epoch':free_epoch,'protocol_sha256':psha,'history_sha256':history_sha.hexdigest()}
   tmp=folder/'last_state.writing.pt';torch.save(state,tmp)
   with tmp.open('rb') as fp:os.fsync(fp.fileno())
   tmp.replace(folder/'last_state.pt');dump(folder/'status.json',dict(rec,status='TRAINING',seed=args.seed,pid=os.getpid(),cpu=args.cpu,observed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
   if epoch%10==0:print(args.seed,epoch,objective,best_epoch,rec['epoch_wall_seconds'],flush=True)
   if event=='early_stop_min_lr_plateau':stop=event;break
 finally:hist.close()
 result={'status':'COMPLETED_PLATEAU' if stop=='early_stop_min_lr_plateau' else 'BUDGET_TRUNCATED','seed':args.seed,'epochs':epoch,'updates':updates,'best_epoch':best_epoch,'best_objective':best_obj,'baseline_objective':obj0,'stop_reason':stop,'best_model_sha256':sha(folder/'best_model.npz'),'protocol_sha256':psha,'worker_sha256':sha(__file__),'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu0,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'GPU_used':False,'evaluation_records_read':0,'native_evaluation_pending':True}
 dump(folder/'freeze.json',result);dump(folder/'status.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  seed=next((sys.argv[i+1] for i,x in enumerate(sys.argv[:-1]) if x=='--seed'),'unknown')
  p=OUT/f'seed{seed}'
  if p.exists():dump(p/'failure.json',{'status':'FAILED','traceback':traceback.format_exc(),'observed_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
  raise
