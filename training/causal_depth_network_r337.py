"""Optional causal Z residual network; magnetic likelihood can veto its output.

The old XY/state/permission models are unchanged. No command, trajectory
template, nominal shallow label or Target-device fitting is used.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import numpy as np
from training.analyze_z_evidence_r334 import (ROOT, OUT as R334, prepare, profiles,
    apply_profiles, summary, source_validation, SOURCE_D, VALIDATION_IDS, dump,
    objective, sha, CoverageGuardedNeuralDecoder, DepthPolicy, records)
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory

OUT=ROOT/"analysis/r337_causal_depth_network_20260929"


def load_policy():
    f=json.loads((R334/"freeze.json").read_text(encoding="utf-8"))
    assert f["runtime_sha256"]==sha(ROOT/"training/z_evidence_runtime_r334.py")
    return DepthPolicy(**f["policy"])


def features(d,decoder,profile,base):
    z,cost,corrected=profile;out=np.zeros((len(d["time"]),56));hist=CausalXYFeatureHistory();j=0
    phi=np.zeros((len(out),16));phi[d["valid"]]=decoder.atlas.features(d["observed"][d["valid"]])
    for i,t in enumerate(d["time"]):
        if not d["valid"][i]:hist.reset();continue
        xyz=np.r_[d["final_xy"][i],d["live"][i,2]]
        x=hist.push(time_ns=t,b0_id=int(d["epoch"][i]),stream_id=0,
                    xyz=xyz,phi=phi[i],memory=d["memory"][i])
        out[i]=np.r_[x,base[i,2]/3.,np.log1p(cost[j].min()),
                     np.log1p(np.interp(d["live"][i,2],z,cost[j]))]
        j+=1
    assert j==len(cost)
    return out


def infer(model,d,x,profile,base):
    z,cost,phi=profile;out=base.copy();delta=np.zeros(len(out));veto=np.zeros(len(out),bool)
    ids=np.flatnonzero(d["valid"]);residual=np.clip(model.predict(x[ids]).reshape(-1),-1.,1.)
    candidate=np.clip(base[ids,2]+residual,z[0],z[-1])
    for j,i in enumerate(ids):
        candidate_cost=np.interp(candidate[j],z,cost[j]);best=max(float(cost[j].min()),1e-8)
        veto[i]=candidate_cost>2.*best
        if not veto[i]:out[i,2]=candidate[j];delta[i]=residual[j]
    np.testing.assert_array_equal(out[:,:2],d["live"][:,:2])
    return out,delta,veto


class DepthMLP:
    def __init__(self,width=32,seed=337):self.width=width;self.rng=np.random.default_rng(seed)
    def predict(self,x):
        u=np.clip((x-self.mean)/self.scale,-8.,8.)
        for i,(w,b) in enumerate(zip(self.params[::2],self.params[1::2])):
            u=u@w+b
            if i<2:u=np.tanh(u)
        return u
    def fit(self,x,y,weight,valid):
        self.mean=np.average(x,axis=0,weights=weight)
        self.scale=np.maximum(np.sqrt(np.average((x-self.mean)**2,axis=0,weights=weight)),.1)
        u=np.clip((x-self.mean)/self.scale,-8.,8.);width=self.width
        params=[self.rng.normal(0,np.sqrt(2/x.shape[1]),(x.shape[1],width)),np.zeros(width),
                self.rng.normal(0,np.sqrt(2/width),(width,width)),np.zeros(width),
                self.rng.normal(0,.02,(width,1)),np.zeros(1)]
        moments=[np.zeros_like(p) for p in params];squares=[np.zeros_like(p) for p in params]
        iteration=0;best=float('inf');best_params=None;self.history=[]
        original=[summary(d,d["live"]) for d,xx,p,base in valid]
        for epoch in range(1,101):
            for idx in np.array_split(self.rng.permutation(len(x)),max(1,(len(x)+255)//256)):
                w1,b1,w2,b2,w3,b3=params
                h1=np.tanh(u[idx]@w1+b1);h2=np.tanh(h1@w2+b2);pred=h2@w3+b3
                g3=2*(pred-y[idx])*weight[idx,None]/len(idx)
                g2=(g3@w3.T)*(1-h2*h2);g1=(g2@w2.T)*(1-h1*h1)
                gradients=[u[idx].T@g1+.001*w1,g1.sum(0),h1.T@g2+.001*w2,g2.sum(0),h2.T@g3+.001*w3,g3.sum(0)]
                iteration+=1
                for p,g,m,v in zip(params,gradients,moments,squares):
                    m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
                    p-=.001*(m/(1-.9**iteration))/(np.sqrt(v/(1-.999**iteration))+1e-8)
            if epoch%5:continue
            self.params=params
            current=[summary(d,infer(self,d,xx,p,base)[0]) for d,xx,p,base in valid]
            qualified=all(current[j][k]<=original[j][k]+.01 for j in (3,4,5)
                          for k in ('z_raw_mae_mm','z_centered_mae_mm'))
            score=float(np.mean([objective(r) for r in current]))
            self.history.append({'epoch':epoch,'objective':score,'qualified':qualified,'trials':current})
            if qualified and score<best:best=score;best_params=[p.copy() for p in params];self.best_epoch=epoch
            if epoch%20==0:print('Z MLP',width,epoch,round(score,4),qualified,flush=True)
        self.qualified=best_params is not None
        self.params=best_params if self.qualified else [p.copy() for p in params]
        self.best=best if self.qualified else None
        if not self.qualified:self.best_epoch=None
        return self
    def save(self,p):np.savez_compressed(p,mean=self.mean,scale=self.scale,**{f'p{i}':v for i,v in enumerate(self.params)})
    @classmethod
    def load(cls,p):
        model=cls()
        with np.load(p,allow_pickle=False) as a:
            model.mean=a['mean'].copy();model.scale=a['scale'].copy();model.params=[a[f'p{i}'].copy() for i in range(6)]
        return model


def selection(anchor_weight=1.):
    if (OUT/'freeze.json').exists():raise FileExistsError('Preserve frozen R337')
    declared={'source_train_ids':[i for i in range(86,110) if i not in VALIDATION_IDS],
        'source_validation_ids':list(VALIDATION_IDS),'widths':[32,64],'epochs':100,'seed':337,
        'features':'56: current XYZ, raw 16D field, 2D memory, past .1/.4s differences, R334 Z and two likelihood costs',
        'objective':'Source Trial-equal depth MAE+jump composite; static and true-Z validation worsening <=.01mm',
        'loss':'weighted residual MSE + .001/2 weight norm squared; no bias penalty',
        'static_anchors':'565 exact atlas states with zero Z residual',
        'static_anchor_trial_weight':float(anchor_weight),
        'negative_or_ambiguous_shallow_labels':False,'target_fit_rows':0,'target_selection_rows':0,
        'scope':'saved-stream Z-only diagnostic, not online/full R332 replay'}
    dump(OUT/'declared_protocol.json',declared)
    decoder=CoverageGuardedNeuralDecoder.from_report();policy=load_policy();tx=[];ty=[];tw=[];counts=[];valid=[]
    raw_manifest=[]
    for number in range(86,110):
        path=next(SOURCE_D.glob(f'Trial_{number:06d}_*'));d=prepare(path,decoder)
        p=profiles(d,decoder,1.,'final');base=apply_profiles(d,p,policy)[0];x=features(d,decoder,p,base)
        if number in VALIDATION_IDS:valid.append((d,x,p,base));continue
        ids=np.flatnonzero(d['valid'] & d['calibrated_action'])[::4]
        tx.append(x[ids]);ty.append(np.clip(d['q'][ids,2]-base[ids,2],-1.,1.)[:,None]);tw.append(np.full(len(ids),1/max(len(ids),1)))
        counts.append({'trial':number,'samples':len(ids),'path':path.relative_to(ROOT).as_posix()})
        raw_manifest.extend({'path':(path/name).relative_to(ROOT).as_posix(),'sha256':sha(path/name)}
                            for name in ('meta.json','magnetic_raw.csv','live_predictions.csv','events.csv','stage_commands.csv'))
        print('Z training Source',number,len(ids),flush=True)
    # validation ordering is the declared order, not filesystem/scan order.
    valid.sort(key=lambda row:list(VALIDATION_IDS).index(int(row[0]['meta']['trial_parameters']['trial_id'])))
    xx=np.stack(np.meshgrid(decoder.atlas.x,decoder.atlas.y,decoder.atlas.z,indexing='ij'),axis=-1).reshape(-1,3)[::3]
    field=decoder.atlas.features(decoder.atlas.forward(xx))
    anchor=np.column_stack((xx/np.array([10,10,3]),field,np.zeros((len(xx),34)),xx[:,2]/3.,np.zeros((len(xx),2))))
    assert anchor.shape[1]==56
    tx.append(anchor);ty.append(np.zeros((len(xx),1)));tw.append(np.full(len(xx),anchor_weight/len(xx)))
    x=np.vstack(tx);y=np.vstack(ty);weight=np.concatenate(tw);weight*=len(weight)/weight.sum()
    models=[]
    for width in declared['widths']:
        model=DepthMLP(width).fit(x,y,weight,valid);file=OUT/f'z_mlp_width{width}.npz';model.save(file)
        models.append({'width':width,'file':file.relative_to(ROOT).as_posix(),'best_epoch':model.best_epoch,
                       'objective':model.best,'qualified':model.qualified,'history':model.history})
    qualified=[r for r in models if r['qualified']]
    selected=min(qualified,key=lambda r:r['objective']) if qualified else None
    dump(OUT/'source_selection.json',{'declaration':declared,'models':models,'selected':selected,
         'sample_counts':counts,'train_samples':len(x),'static_anchor_samples':len(xx),'raw_manifest':raw_manifest})
    if selected is None:
        print('NOT QUALIFIED: no Target evaluation or deployment',flush=True)
        return
    dump(OUT/'freeze.json',{'selected_file':selected['file'],'selected_width':selected['width'],
         'selected_epoch':selected['best_epoch'],'model_sha256':sha(ROOT/selected['file']),
         'runtime_sha256':sha(Path(__file__)),'selection_sha256':sha(OUT/'source_selection.json'),
         'depth_policy':asdict(policy),'target_fit_rows':0,'target_selection_rows':0,'prospective_blind':False})
    print('FROZEN',selected['width'],selected['best_epoch'],selected['objective'],flush=True)


def evaluate(device):
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
    assert freeze['model_sha256']==sha(ROOT/freeze['selected_file'])
    assert freeze['runtime_sha256']==sha(Path(__file__))
    assert freeze['selection_sha256']==sha(OUT/'source_selection.json')
    model=DepthMLP.load(ROOT/freeze['selected_file']);policy=DepthPolicy(**freeze['depth_policy'])
    decoder=CoverageGuardedNeuralDecoder.from_report();rows=[]
    for path in records(device):
        d=prepare(path,decoder);p=profiles(d,decoder,1.,'final');base=apply_profiles(d,p,policy)[0]
        x=features(d,decoder,p,base);xyz,residual,veto=infer(model,d,x,p,base)
        old,new=summary(d,d['live']),summary(d,xyz);stem=sha(path/'meta.json')[:16];stream=OUT/'streams'/(stem+'.npz')
        np.savez_compressed(stream,time=d['time'],q=d['q'],valid=d['valid'],old=d['live'],new=xyz,base=base,
            calibrated=d['calibrated_action'],action=d['action'],residual=residual,veto=veto)
        rows.append({'path':path.relative_to(ROOT).as_posix(),'stream':stream.relative_to(ROOT).as_posix(),
            'trial':int(d['meta']['trial_parameters']['trial_id']),'mode':d['mode'],'old':old,'new':new,
            'xy_and_validity_unchanged':True,'network_veto_frames':int(veto.sum())})
        print(device,path.name[:12],d['mode'],round(old['z_centered_mae_mm'],3),'->',round(new['z_centered_mae_mm'],3),flush=True)
    macro={key:{k:float(np.mean([r[k][key] for r in rows])) for k in ('old','new')}
           for key in ('z_raw_mae_mm','z_centered_mae_mm','z_step_p95_mm','z_step_max_mm')}
    dump(OUT/f'report_{device}.json',{'schema':'R337-FrozenCausalDepthNetwork-1','freeze':freeze,
        'new_online_test':False,'full_r332_replayed':False,'rows':rows,'trial_macro':macro})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('select','freeze-saved','source','target'))
    p.add_argument('--anchor-weight',type=float,default=1.)
    p.add_argument('--anchor-ablation',action='store_true');args=p.parse_args()
    if args.anchor_ablation:OUT=ROOT/'analysis/r340_depth_static_anchor_20260929'
    if not np.isfinite(args.anchor_weight) or args.anchor_weight<=0:raise ValueError('Positive anchor weight required')
    OUT.mkdir(exist_ok=True);(OUT/'streams').mkdir(exist_ok=True)
    if args.task=='select':selection(args.anchor_weight)
    elif args.task=='freeze-saved':
        if (OUT/'freeze.json').exists():raise FileExistsError('Freeze already exists')
        report=json.loads((OUT/'source_selection.json').read_text(encoding='utf-8'));selected=report['selected']
        if selected is None or not selected['qualified']:raise ValueError('No Source-qualified model')
        dump(OUT/'freeze.json',{'selected_file':selected['file'],'selected_width':selected['width'],
            'selected_epoch':selected['best_epoch'],'model_sha256':sha(ROOT/selected['file']),
            'runtime_sha256':sha(Path(__file__)),'selection_sha256':sha(OUT/'source_selection.json'),
            'depth_policy':asdict(load_policy()),'target_fit_rows':0,'target_selection_rows':0,'prospective_blind':False})
        print('FROZEN saved Source-trained weights',selected['width'],selected['best_epoch'])
    else:evaluate(args.task)
