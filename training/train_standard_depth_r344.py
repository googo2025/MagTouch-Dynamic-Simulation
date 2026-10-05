"""Expanded Source standard-motion Z learning; no complex/Target selection."""
import argparse
import json
from dataclasses import asdict
import numpy as np
from training.standard_motion_protocol_r344 import OUT, load, assert_fit_allowed
from training.analyze_z_evidence_r334 import ROOT, sha, dump, summary, objective
from training.develop_unified_depth_r341 import models, cache, data_for_summary, apply
from training.causal_depth_network_r337 import DepthMLP
from training.unified_depth_runtime_r341 import SourceFeatureSupport, UnifiedDepthPolicy
from training.fast_depth_features_r344 import augment


def features(a, decoder, dimension):
    return a['x'] if dimension == 56 else augment(decoder.atlas,a['time'],a['x'],a['valid'],a['epoch'])


def anchors(decoder):
    q = np.stack(np.meshgrid(decoder.atlas.x,decoder.atlas.y,decoder.atlas.z,indexing='ij'),axis=-1).reshape(-1,3)[::3]
    phi = decoder.atlas.features(decoder.atlas.forward(q))
    return np.column_stack((q/np.array([10,10,3]),phi,np.zeros((len(q),34)),q[:,2]/3,np.zeros((len(q),2))))


def nn_proposal(a,x,model):
    v=a['valid']; proposed=dict(a)
    proposed['proposal']=np.clip(a['base'][v,2]+np.clip(model.predict(x[v]).reshape(-1),-1,1),a['z'][0],a['z'][-1])
    return proposed


def fit(x,y,weight,check,dimension):
    model=DepthMLP(32,344); model.mean=np.average(x,axis=0,weights=weight)
    model.scale=np.maximum(np.sqrt(np.average((x-model.mean)**2,axis=0,weights=weight)),.1)
    u=np.clip((x-model.mean)/model.scale,-8,8); rng=model.rng; width=32
    params=[rng.normal(0,np.sqrt(2/dimension),(dimension,width)),np.zeros(width),
            rng.normal(0,np.sqrt(2/width),(width,width)),np.zeros(width),
            rng.normal(0,.02,(width,1)),np.zeros(1)]
    moments=[np.zeros_like(p) for p in params]; squares=[p.copy() for p in moments]
    iteration=0; best=None; history=[]
    for epoch in range(1,81):
        for idx in np.array_split(rng.permutation(len(x)),max(1,(len(x)+255)//256)):
            w1,b1,w2,b2,w3,b3=params
            h1=np.tanh(u[idx]@w1+b1); h2=np.tanh(h1@w2+b2); pred=h2@w3+b3
            g3=2*(pred-y[idx])*weight[idx,None]/len(idx)
            g2=(g3@w3.T)*(1-h2*h2); g1=(g2@w2.T)*(1-h1*h1)
            grads=[u[idx].T@g1+.001*w1,g1.sum(0),h1.T@g2+.001*w2,g2.sum(0),h2.T@g3+.001*w3,g3.sum(0)]
            iteration+=1
            for p,g,m,v in zip(params,grads,moments,squares):
                m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
                p-=.001*(m/(1-.9**iteration))/(np.sqrt(v/(1-.999**iteration))+1e-8)
        if epoch%10: continue
        model.params=params; result=check(model); result['epoch']=epoch; history.append(result)
        print('Source NN',dimension,epoch,round(result['objective'],5),result['qualified'],flush=True)
        if result['qualified'] and (best is None or result['objective']<best['objective']):
            best=dict(result);best['params']=[p.copy() for p in params]
    if best is None: return model,history,None
    model.params=best.pop('params'); return model,history,best


def select():
    if (OUT/'freeze.json').exists(): raise FileExistsError('Preserve R344 freeze')
    train=load('train'); validation=load('selection'); decoder,baseline_model,dp=models()
    oldf=json.loads((ROOT/'analysis/r341_unified_depth_20260929/freeze.json').read_text(encoding='utf-8'))
    oldsupport=SourceFeatureSupport.load(ROOT/oldf['support_file']); oldpolicy=UnifiedDepthPolicy(**oldf['policy'])
    declared={'dimensions':[56,59],'width':32,'epochs':80,'seed':344,'batch':256,
       'learning_rate':.001,'weight_decay':.001,'static_anchor_trial_weight':4.,'stride':8,
       'target_fit_rows':0,'target_selection_rows':0,'complex_fit_rows':0,'complex_selection_rows':0,
       'policy':'frozen R341 neural strength 4 and continuity .01; Source-only support quantile .99',
       'qualification':'all Source selection grid/z_path/compound raw+centered <= old+.01; mode-equal objective < R341',
       'selection_roles_sha256':sha(OUT/'dataset_roles.json'),'comparison':'same real training data, 56D versus 59D causal depth tangent',
       'scope':'new localization layer; saved XY/history/permission unchanged; response has historical complex exposure'}
    dump(OUT/'declared_training.json',declared)
    training=[]; vals=[]; ref=[]; owners=[]
    for r in train:
        path=ROOT/r['path'];assert_fit_allowed(path,'train');a,s=cache(path,decoder,baseline_model,dp)
        ids=np.flatnonzero(a['valid'] & a['calibrated_action'])[::8]
        if not len(ids): raise ValueError('Empty declared training Trial '+r['path'])
        training.append((a,s,ids)); keep=ids[np.linspace(0,len(ids)-1,min(len(ids),64)).astype(int)]
        ref.append(a['x'][keep]);owners.extend([len(training)-1]*len(keep))
        print('Source standard train',s['trial'],s['mode'],len(ids),flush=True)
    anchor=anchors(decoder);ref.append(anchor);owners.extend([-1]*len(anchor));ref=np.vstack(ref);owners=np.asarray(owners)
    for r in validation:
        assert_fit_allowed(ROOT/r['path'],'selection');a,s=cache(ROOT/r['path'],decoder,baseline_model,dp)
        distance=oldsupport.distance(a['x'][a['valid']]); baseline=apply(a,oldpolicy,distance)[0]
        vals.append((a,s,summary(data_for_summary(a,s),a['old']),summary(data_for_summary(a,s),baseline)))
        print('Source selection',s['trial'],s['mode'],flush=True)
    def average(rows):
        groups={mode:[objective(r) for r,s in rows if s['mode']==mode] for mode in set(s['mode'] for r,s in rows)}
        return float(np.mean([np.mean(v) for v in groups.values()]))
    baseline_score=average([(r,s) for a,s,old,r in vals]); results=[]
    for dimension in declared['dimensions']:
        tx=[];ty=[];tw=[];counts=[]
        for a,s,ids in training:
            xx=features(a,decoder,dimension);tx.append(xx[ids]);ty.append(np.clip(a['q'][ids,2]-a['base'][ids,2],-1,1)[:,None])
            tw.append(np.full(len(ids),1/len(ids)));counts.append({'path':s['path'],'samples':len(ids)})
        tx.append(anchor if dimension==56 else np.column_stack((anchor,np.zeros((len(anchor),3)))))
        ty.append(np.zeros((len(anchor),1)));tw.append(np.full(len(anchor),4/len(anchor)))
        x=np.vstack(tx);y=np.vstack(ty);w=np.concatenate(tw);w*=len(w)/w.sum()
        # Source training normalization/support computed once, without validation.
        mean=np.average(x,axis=0,weights=w);scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=w)),.1)
        normalized=np.clip((ref-mean[:56])/scale[:56],-8,8)
        support=SourceFeatureSupport(mean[:56],scale[:56],normalized)
        loo=[]
        for owner in range(len(training)):
            probes=ref[owners==owner][::8]
            other=normalized[owners!=owner]
            loo.extend(SourceFeatureSupport(mean[:56],scale[:56],other).distance(probes))
        limit=max(float(np.quantile(loo,.99)),1e-6)
        policy=UnifiedDepthPolicy(neural_strength=4,continuity=.01,support_limit=limit)
        support_file=OUT/f'support_{dimension}.npz'
        np.savez_compressed(support_file,mean=mean[:56],scale=scale[:56],reference=normalized,owner=owners)
        probes=[(a,s,old,prior,features(a,decoder,dimension),support.distance(a['x'][a['valid']])) for a,s,old,prior in vals]
        def check(model):
            rows=[]; qualified=True
            for a,s,old,prior,xx,distance in probes:
                out=apply(nn_proposal(a,xx,model),policy,distance)[0];row=summary(data_for_summary(a,s),out)
                rows.append((row,s))
                if s['mode'] in ('grid_points','z_path','compound'):
                    qualified &= all(row[k]<=old[k]+.01 for k in ('z_raw_mae_mm','z_centered_mae_mm'))
            score=average(rows)
            return {'objective':score,'qualified':bool(qualified and score<baseline_score),
                    'trials':[{'path':s['path'],**r} for r,s in rows]}
        model,history,best=fit(x,y,w,check,dimension);file=OUT/f'z_mlp_{dimension}.npz';model.save(file)
        results.append({'dimension':dimension,'file':file.relative_to(ROOT).as_posix(),
            'support_file':support_file.relative_to(ROOT).as_posix(),'policy':asdict(policy),
            'selected':best,'history':history,'sample_counts':counts,'train_samples':len(x)})
    good=[r for r in results if r['selected'] is not None]
    selected=min(good,key=lambda r:r['selected']['objective']) if good else None
    report={'declared':declared,'baseline_R341_objective':baseline_score,'models':results,
      'selected':selected,'source_validation_baselines':[{'path':s['path'],'old':old,'R341':r} for a,s,old,r in vals]}
    dump(OUT/'source_selection.json',report)
    if selected:
        files=[OUT/'dataset_roles.json',OUT/'source_selection.json',ROOT/selected['file'],ROOT/selected['support_file'],
          ROOT/'training/standard_motion_protocol_r344.py',ROOT/'training/train_standard_depth_r344.py',ROOT/'training/fast_depth_features_r344.py',
          ROOT/'training/standard_depth_runtime_r344.py']
        files += [ROOT/v['path'] for v in oldf['dependencies']]
        files = list(dict.fromkeys(files))
        dump(OUT/'freeze.json',{'schema':'R344-SourceStandardDepth-1','model_file':selected['file'],
           'support_file':selected['support_file'],'dimension':selected['dimension'],'policy':selected['policy'],
           'selected_epoch':selected['selected']['epoch'],'target_fit_rows':0,'target_selection_rows':0,
           'complex_fit_rows':0,'complex_selection_rows':0,'prospective_blind':False,
           'complete_system_path_blind':False,'dependencies':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in files]})
    print('FROZEN' if selected else 'NOT QUALIFIED - heldout complex/Target not evaluated',flush=True)


if __name__=='__main__':
    select()
