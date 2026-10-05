"""Source-only incremental depth learning with profiled training nuisances.

Absolute depth teacher and magnetic adjudication stay frozen. A new residual
learns within-Trial changes without absorbing each training Trial's constant
reference offset. No nuisance is fitted on validation/Target or at inference.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import numpy as np

from training.analyze_z_evidence_r334 import ROOT, dump, sha, summary, objective
from training.all_motion_depth_r347 import baseline, declare, METRICS
from training.standard_motion_protocol_r344 import load, assert_fit_allowed
from training.develop_unified_depth_r341 import cache, apply, data_for_summary
from training.train_standard_depth_r344 import anchors
from training.fast_depth_features_r344 import augment
from training.causal_depth_network_r337 import DepthMLP

OUT=ROOT/'analysis/r348_fast_slow_residual_20260929'


def phase_masks(a):
    """Reference-based evaluation only, never an input to the model."""
    dt=np.diff(a['time'])/1e9
    ok=(dt>0)&(dt<=.25)&(a['epoch'][1:]==a['epoch'][:-1])
    vz=np.full(len(dt), np.nan); vxy=vz.copy()
    vz[ok]=np.diff(a['q'][:,2])[ok]/dt[ok]
    vxy[ok]=np.linalg.norm(np.diff(a['q'][:,:2],axis=0),axis=1)[ok]/dt[ok]
    vz=np.r_[np.nan,vz]; vxy=np.r_[np.nan,vxy]
    core=a['calibrated_action']; zmove=np.abs(vz)>.005; xymove=vxy>.005
    return {'fixed_hold':core&~zmove&~xymove&np.isfinite(vz),
            'pure_z_press':core&~xymove&(vz>.005),
            'pure_z_withdraw':core&~xymove&(vz<-.005),
            'constant_z_lateral':core&xymove&~zmove,
            'xyz_simultaneous':core&xymove&zmove}


def phases(a,s,xyz):
    # ONE translation for the entire calibrated Trial, not one per phase.
    good=a['valid']&a['calibrated_action']&np.isfinite(a['q']).all(1)
    bias=np.mean(xyz[good]-a['q'][good],axis=0) if good.any() else np.zeros(3)
    error=xyz[:,2]-a['q'][:,2]; result={}
    for name,mask in phase_masks(a).items():
        accepted=mask&a['valid']; n=int(accepted.sum())
        pairs=accepted[1:]&accepted[:-1]
        dt=np.diff(a['time'])/1e9
        pairs&=(dt>0)&(dt<=.25)&(a['epoch'][1:]==a['epoch'][:-1])
        dz=np.abs(np.diff(error))[pairs]
        result[name]={'reference_frames':int(mask.sum()),'output_frames':n,
            'z_raw_mae_mm':float(np.mean(np.abs(error[accepted]))) if n else None,
            'z_global_centered_mae_mm':float(np.mean(np.abs(error[accepted]-bias[2]))) if n else None,
            'z_step_p95_mm':float(np.quantile(dz,.95)) if len(dz) else None,
            'z_step_max_mm':float(dz.max()) if len(dz) else None,
            'trial_z_bias_mm':float(bias[2]) if n else None}
    return result


def residual_targets(a, ids, teacher_xyz):
    residual=a['q'][ids,2]-teacher_xyz[ids,2]
    nuisance=float(np.mean(residual))
    return np.clip(residual-nuisance,-.4,.4)[:,None], nuisance


def proposal(a,model,x,gain):
    p=dict(a); v=a['valid']
    delta=np.clip(model.predict(x[v]).reshape(-1),-.4,.4)*gain
    p['proposal']=np.clip(a['proposal']+delta,a['z'][0],a['z'][-1])
    return p


def macro(rows):
    modes=sorted(set(s['mode'] for r,s in rows))
    return float(np.mean([np.mean([objective(r) for r,s in rows if s['mode']==m]) for m in modes]))


def train():
    if (OUT/'freeze.json').exists(): raise FileExistsError('Preserve frozen experiment')
    protocol=declare(); decoder,teacher,dp,oldf,policy,support=baseline()
    declared={'schema':'R348-SourceOnlyIncrementalDepth-1','dimensions':[56,59],'epochs':60,
        'check_every':10,'gains':[.25,.5,1.],'width':32,'seed':348,'batch':256,'stride':8,
        'learning_rate':.0002,'weight_decay':.001,'static_anchor_trial_weight':4.,
        'teacher':'Frozen R340 proposal + R341 current-magnetic adjudication; neither refitted',
        'training_nuisance':'Mean (command Z - R341 Z) per TRAIN Trial only; subtracted from correction target; never supplied at inference.',
        'loss':'Trial-equal centered incremental MSE + 0.5 zero-increment MSE + 0.5 within-Trial increment-difference MSE + weight L2.',
        'correction_limit_mm':.4,'pair_max_gap_s':.35,'source_train_trials':45,'source_selection_trials':12,
        'criterion':'Every standard category raw/centered MAE <= R341+0.005 mm; P95 step <= R341+0.005; max-step mean <=R341+0.05; macro objective improves >=1%. Each phase with >=100 accepted selection frames also protects raw/global-centered MAE <=R341+0.005.',
        'target_fit_rows':0,'target_selection_rows':0,'complex_fit_rows':0,'complex_selection_rows':0,
        'scope':'Conditional depth experiment; unchanged saved XY, memory and permission; no new hardware run.',
        'policy':asdict(policy),'protocol_sha256':sha(ROOT/'analysis/r347_all_motion_20260929/protocol.json')}
    dump(OUT/'declared.json',declared)
    data=[]; validation=[]; nuisance_rows=[]
    for role in ('train','selection'):
        for row in load(role):
            assert_fit_allowed(ROOT/row['path'],role)
            a,s=cache(ROOT/row['path'],decoder,teacher,dp)
            distance=support.distance(a['x'][a['valid']]); prev=apply(a,policy,distance)[0]
            x=augment(decoder.atlas,a['time'],a['x'],a['valid'],a['epoch'])
            if role=='train':
                ids=np.flatnonzero(a['valid']&a['calibrated_action'])[::8]
                y,offset=residual_targets(a,ids,prev);data.append((a,s,ids,x,y))
                nuisance_rows.append({'path':s['path'],'mode':s['mode'],'training_nuisance_z_mm':offset,'samples':len(ids)})
            else:
                validation.append((a,s,distance,x,summary(data_for_summary(a,s),prev),phases(a,s,prev)))
    dump(OUT/'training_nuisances.json',nuisance_rows)
    base=macro([(r,s) for a,s,d,x,r,ph in validation]); all_results=[]; selected=None
    anchor=anchors(decoder)
    for dim in declared['dimensions']:
        xs=[];ys=[];ws=[];pairs=[];cursor=0
        for a,s,ids,x,y in data:
            xs.append(x[ids,:dim]);ys.append(y);ws.append(np.full(len(ids),1/len(ids)))
            dt=np.diff(a['time'][ids])/1e9
            ok=(dt>0)&(dt<=.35)&(a['epoch'][ids[1:]]==a['epoch'][ids[:-1]])
            pp=np.flatnonzero(ok);pairs.extend(zip((cursor+pp).tolist(),(cursor+pp+1).tolist()));cursor+=len(ids)
        xs.append(anchor if dim==56 else np.column_stack((anchor,np.zeros((len(anchor),3)))))
        ys.append(np.zeros((len(anchor),1)));ws.append(np.full(len(anchor),4/len(anchor)))
        x=np.vstack(xs);y=np.vstack(ys);w=np.concatenate(ws);w*=len(w)/w.sum();pair=np.asarray(pairs,int)
        model=DepthMLP(32,348);model.mean=teacher.mean.copy();model.scale=teacher.scale.copy()
        if dim==59:
            mu=np.average(x[:,56:],axis=0,weights=w)
            sd=np.maximum(np.sqrt(np.average((x[:,56:]-mu)**2,axis=0,weights=w)),.1)
            model.mean=np.r_[model.mean,mu];model.scale=np.r_[model.scale,sd]
        u=np.clip((x-model.mean)/model.scale,-8,8);p=[v.copy() for v in teacher.params]
        if dim==59:p[0]=np.vstack((p[0],np.zeros((3,32))))
        # Exactly zero additional correction before the first optimization step.
        p[4]*=0;p[5]*=0
        m=[np.zeros_like(v) for v in p];v=[a.copy() for a in m];rng=np.random.default_rng(348);iteration=0;history=[]
        def forward(idx):
            h1=np.tanh(u[idx]@p[0]+p[1]);h2=np.tanh(h1@p[2]+p[3]);return h1,h2,h2@p[4]+p[5]
        def grads(idx,gout):
            h1,h2,_=forward(idx);g2=(gout@p[4].T)*(1-h2*h2);g1=(g2@p[2].T)*(1-h1*h1)
            return [u[idx].T@g1,g1.sum(0),h1.T@g2,g2.sum(0),h2.T@gout,gout.sum(0)]
        for epoch in range(1,61):
            for idx in np.array_split(rng.permutation(len(x)),(len(x)+255)//256):
                pred=forward(idx)[2];gg=grads(idx,(2*(pred-y[idx])+pred)*w[idx,None]/len(idx))
                pp=pair[rng.integers(0,len(pair),size=len(idx))];ia,ib=pp[:,0],pp[:,1]
                ga=(forward(ib)[2]-forward(ia)[2]-(y[ib]-y[ia]))*((w[ia]+w[ib])/2)[:,None]/len(idx)
                aa,bb=grads(ia,-ga),grads(ib,ga);gg=[g+a+b for g,a,b in zip(gg,aa,bb)]
                for k in (0,2,4):gg[k]+=.001*p[k]
                iteration+=1
                for par,g,mm,vv in zip(p,gg,m,v):
                    mm*=.9;mm+=.1*g;vv*=.999;vv+=.001*g*g
                    par-=.0002*(mm/(1-.9**iteration))/(np.sqrt(vv/(1-.999**iteration))+1e-8)
            if epoch%10:continue
            model.params=p
            for gain in declared['gains']:
                evaluated=[];failures=[]
                for a,s,d,xx,prev,bph in validation:
                    xyz=apply(proposal(a,model,xx[:,:dim],gain),policy,d)[0]
                    evaluated.append((summary(data_for_summary(a,s),xyz),s,phases(a,s,xyz)))
                for mode in sorted(set(s['mode'] for r,s,ph in evaluated)):
                    for key,tol in (('z_raw_mae_mm',.005),('z_centered_mae_mm',.005),('z_step_p95_mm',.005),('z_step_max_mm',.05)):
                        delta=float(np.mean([r[key] for r,s,ph in evaluated if s['mode']==mode])-np.mean([r[key] for a,s,d,x,r,bph in validation if s['mode']==mode]))
                        if delta>tol:failures.append({'mode':mode,'metric':key,'delta_mm':delta})
                for phase in phase_masks(validation[0][0]):
                    count=sum(ph[phase]['output_frames'] for r,s,ph in evaluated)
                    if count<100:continue
                    for key in ('z_raw_mae_mm','z_global_centered_mae_mm'):
                        new=[ph[phase][key] for r,s,ph in evaluated if ph[phase][key] is not None]
                        old=[ph[phase][key] for a,s,d,x,r,ph in validation if ph[phase][key] is not None]
                        delta=float(np.mean(new)-np.mean(old))
                        if delta>.005:failures.append({'phase':phase,'metric':key,'delta_mm':delta})
                score=macro([(r,s) for r,s,ph in evaluated]);qualified=not failures and score<.99*base
                result={'dimension':dim,'epoch':epoch,'gain':gain,'objective':score,'qualified':qualified,'failures':failures,
                    'trials':[{'path':s['path'],'mode':s['mode'],'metrics':r,'phases':ph} for r,s,ph in evaluated]}
                history.append(result)
                print('R348 Source',dim,epoch,gain,round(score,6),qualified,'failures',len(failures),flush=True)
                if qualified and (selected is None or score<selected['objective']):
                    file=OUT/'qualified_increment.npz';model.save(file)
                    selected={k:result[k] for k in ('dimension','epoch','gain','objective')};selected['file']=file.relative_to(ROOT).as_posix()
        all_results.append({'dimension':dim,'history':history,'training_samples':len(x),'pairs':len(pair)})
    dump(OUT/'source_selection.json',{'baseline_objective':base,'models':all_results,'selected':selected})
    deps=[OUT/'declared.json',OUT/'training_nuisances.json',OUT/'source_selection.json',ROOT/'training/fast_slow_residual_r348.py',ROOT/'training/fast_depth_features_r344.py',ROOT/'analysis/r341_unified_depth_20260929/freeze.json',ROOT/'analysis/r347_all_motion_20260929/protocol.json']
    deps+=[ROOT/r['path'] for r in oldf['dependencies']]
    if selected:deps.append(ROOT/selected['file'])
    dump(OUT/'freeze.json',{'selected':selected,'fallback':'R341' if selected is None else None,
        'target_fit_rows':0,'target_selection_rows':0,'complex_fit_rows':0,'complex_selection_rows':0,
        'dependencies':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in dict.fromkeys(deps)]})
    print('R348 frozen',selected,flush=True)


def evaluate():
    f=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
    for dep in f['dependencies']:
        if sha(ROOT/dep['path'])!=dep['sha256']:raise ValueError('Freeze changed: '+dep['path'])
    decoder,teacher,dp,oldf,policy,support=baseline();selected=f['selected']
    model=DepthMLP.load(ROOT/selected['file']) if selected else None
    rows=[]
    for record in declare()['records']:
        for raw in record['raw_manifest']:
            if sha(ROOT/raw['path'])!=raw['sha256']:raise ValueError('Raw data changed')
        a,s=cache(ROOT/record['path'],decoder,teacher,dp);d=support.distance(a['x'][a['valid']]);prev=apply(a,policy,d)[0]
        if selected:
            x=a['x'] if selected['dimension']==56 else augment(decoder.atlas,a['time'],a['x'],a['valid'],a['epoch'])
            xyz=apply(proposal(a,model,x,selected['gain']),policy,d)[0]
        else:xyz=prev.copy()
        np.testing.assert_array_equal(xyz[:,:2],a['old'][:,:2])
        row={**record,'metrics':{name:summary(data_for_summary(a,s),z) for name,z in (('original',a['old']),('R341',prev),('selected',xyz))},
             'phases':{name:phases(a,s,z) for name,z in (('original',a['old']),('R341',prev),('selected',xyz))}}
        rows.append(row)
        print('R348 eval',record['role'],record['trial'],record['mode'],flush=True)
    groups={}
    for role,target,phase in sorted(set((r['role'],r['target'],ph) for r in rows for ph in r['phases']['R341'])):
        subset=[r for r in rows if (r['role'],r['target'])==(role,target)]
        stats={}
        for name in ('original','R341','selected'):
            valid=[r['phases'][name][phase] for r in subset if r['phases'][name][phase]['output_frames']]
            stats[name]={'trials':len(valid),'outputs':sum(r['output_frames'] for r in valid),
                **{k:float(np.mean(v)) if (v:=[r[k] for r in valid if r[k] is not None]) else None for k in ('z_raw_mae_mm','z_global_centered_mae_mm','z_step_p95_mm','z_step_max_mm','trial_z_bias_mm')}}
        groups[f'{role}/{"Target" if target else "Source"}/{phase}']=stats
    dump(OUT/'report.json',{'freeze_sha256':sha(OUT/'freeze.json'),'rows':rows,'phase_groups':groups,'selected':selected,
        'translation':'One constant XYZ per entire Trial diagnostics only, not independent phase registration.',
        'fresh_online_test':False,'target_fit_rows':0,'complex_fit_rows':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('train','evaluate'));a=p.parse_args()
    {'train':train,'evaluate':evaluate}[a.task]()
