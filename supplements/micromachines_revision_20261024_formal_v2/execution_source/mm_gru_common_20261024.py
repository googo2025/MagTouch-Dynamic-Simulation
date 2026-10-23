"""Independent CPU-only finite-window residual GRU; no acquisition or UI imports."""
from pathlib import Path
from collections import deque
import os,sys,json,hashlib,shutil
ROOT=Path('/home/researcher/magtouch_real_stability_r460_20261004')
OUT=ROOT/'analysis/micromachines_revision_validation_20261023/gru_temporal_v2_20261024'
OLD=ROOT/'analysis/r460_source_real_history_refinement_20261004'
KEYS=('w1','b1','w2','b2','w3','b3')
STEPS=64
WIDTH=16
HORIZON_NS=4_000_000_000

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def dump(p,j):
 p=Path(p);t=p.with_suffix(p.suffix+'.writing')
 t.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False));t.replace(p)

def load(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def rss():
 return next(int(l.split()[1])*1024 for l in Path('/proc/self/status').read_text().splitlines() if l.startswith('VmRSS:'))

def guard():
 available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
 if available<8*1024**3 or rss()>2*1024**3 or shutil.disk_usage(OUT).free<20*1024**3:raise RuntimeError('Memory/disk guard; preserve failure evidence')

def setup(cpu):
 import socket
 assert socket.gethostname()=='RESEARCH_HOST'
 assert Path('/etc/machine-id').read_text().strip()=='RESEARCH_MACHINE_ID'
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 os.sched_setaffinity(0,{int(cpu)});os.nice(10)
 os.environ['MAGTOUCH_R460_SOURCE']=str(ROOT)
 sys.path[:0]=[str(ROOT),str(ROOT/'code'),str(ROOT/'tmp/r360_math_packages')]

def make_model(a):
 import torch
 class ResidualGRU(torch.nn.Module):
  def __init__(self):
   super().__init__()
   self.base=torch.nn.ParameterList([torch.nn.Parameter(torch.as_tensor(a[k][:53].copy() if k=='w1' else a[k].copy(),dtype=torch.float64)) for k in KEYS])
   self.gru=torch.nn.GRU(54,WIDTH,batch_first=True,dtype=torch.float64)
   self.readout=torch.nn.Linear(WIDTH,2,bias=False,dtype=torch.float64)
   torch.nn.init.zeros_(self.readout.weight)
  def forward(self,u,context,length):
   h=torch.tanh(torch.tanh(u@self.base[0]+self.base[1])@self.base[2]+self.base[3])@self.base[4]+self.base[5]
   seq,_=self.gru(context)
   return h+self.readout(seq[torch.arange(len(u)),length-1])
  def regularizer(self):
   weights=list(self.base[::2])+[self.gru.weight_ih_l0,self.gru.weight_hh_l0,self.readout.weight]
   return .0005*sum(w.square().sum() for w in weights)
 return ResidualGRU()

def save_model(path,model,mean,scale):
 import numpy as np
 arrays={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
 np.savez_compressed(path,**arrays,mean=mean,scale=scale,window_frames=STEPS,window_ns=HORIZON_NS)

def window_array(x,t,valid,indices,mean,scale):
 """Whole accepted-frame prefixes first; supervision never drives context/reset."""
 import numpy as np
 contexts=np.zeros((len(indices),STEPS,54),dtype=np.float64);lengths=np.ones(len(indices),dtype=np.int64)
 lookup={int(v):i for i,v in enumerate(indices)};history=deque();previous=None
 for i in range(len(t)):
  if not valid[i]:history.clear();previous=None;continue
  reset=previous is None or int(t[i])<=previous or int(t[i])-previous>250_000_000
  if reset:history.clear()
  dt=0. if reset else (int(t[i])-previous)/100_000_000.
  history.append((int(t[i]),np.r_[(x[i,:53]-mean)/scale,dt]))
  while len(history)>STEPS or int(t[i])-history[0][0]>HORIZON_NS:history.popleft()
  previous=int(t[i])
  if i in lookup:
   j=lookup[i];lengths[j]=len(history);contexts[j,:len(history)]=[v for _,v in history]
 if len(indices):
  assert np.isfinite(contexts).all()
  np.testing.assert_allclose(contexts[np.arange(len(indices)),lengths-1,:53],(x[indices,:53]-mean)/scale,atol=0,rtol=0)
 return contexts,lengths

def numpy_prediction(a,u,context,length):
 import numpy as np
 b=[a[f'base.{i}'] for i in range(6)]
 out=np.tanh(np.tanh(u@b[0]+b[1])@b[2]+b[3])@b[4]+b[5]
 for i in range(len(u)):
  h=np.zeros(WIDTH);inp=context[i,:length[i]]@a['gru.weight_ih_l0'].T+a['gru.bias_ih_l0']
  for v in inp:
   q=h@a['gru.weight_hh_l0'].T+a['gru.bias_hh_l0']
   r=1/(1+np.exp(-(v[:WIDTH]+q[:WIDTH])))
   z=1/(1+np.exp(-(v[WIDTH:2*WIDTH]+q[WIDTH:2*WIDTH])))
   n=np.tanh(v[2*WIDTH:]+r*q[2*WIDTH:]);h=(1-z)*n+z*h
  out[i]+=h@a['readout.weight'].T
 return out

def install_gru_head(engine,artifact):
 """Both native decoder routes retain their own reset-aware window state."""
 import numpy as np
 from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory
 a=load(artifact)
 class WindowHistory(CausalXYFeatureHistory):
  def reset(self):
   super().reset();self.window=deque()
  def push(self,**kw):
   previous=self.last_time_ns;t=int(kw['time_ns'])
   reset=previous is None or t<=previous or t-previous>self.max_gap_ns or int(kw['b0_id'])!=self.last_b0_id or int(kw['stream_id'])!=self.last_stream_id
   original=super().push(**kw)
   if reset:self.window.clear()
   dt=0. if reset else (t-previous)/100_000_000.
   self.window.append((t,np.r_[(original-a['mean'])/a['scale'],dt]))
   while len(self.window)>STEPS or t-self.window[0][0]>HORIZON_NS:self.window.popleft()
   window=np.zeros((STEPS,54));window[:len(self.window)]=[v for _,v in self.window]
   return np.r_[len(self.window),window.ravel()]
 class GRUHead:
  mean=a['mean'];scale=a['scale']
  def predict(self,feature):
   f=np.asarray(feature).reshape(-1,1+STEPS*54);length=f[:,0].astype(np.int64);context=f[:,1:].reshape(-1,STEPS,54)
   assert ((length>=1)&(length<=STEPS)).all()
   u=context[np.arange(len(f)),length-1,:53]
   return numpy_prediction(a,u,context,length)
 for decoder in (engine.memory_decoder,engine.candidate_decoder):
  np.testing.assert_array_equal(decoder.xy_model.mean[:53],a['mean']);np.testing.assert_array_equal(decoder.xy_model.scale[:53],a['scale'])
  decoder.xy_features=WindowHistory();decoder.xy_model=GRUHead()
 return engine.memory_decoder.xy_model,engine.candidate_decoder.xy_model
