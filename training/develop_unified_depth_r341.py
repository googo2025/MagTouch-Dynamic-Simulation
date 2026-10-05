"""Source-only final-depth selection, then frozen all-cohort paired audit."""
import argparse
from dataclasses import asdict
import json
import numpy as np
from training.analyze_z_evidence_r334 import (ROOT, SOURCE_D, VALIDATION_IDS, prepare, profiles,
    apply_profiles, summary, objective, sha, dump, records, CoverageGuardedNeuralDecoder)
from training.causal_depth_network_r337 import features, DepthMLP
from training.causal_depth_runtime_r340 import FREEZE as BASE_FREEZE
from training.z_evidence_runtime_r334 import DepthPolicy
from training.unified_depth_runtime_r341 import (UnifiedDepthPolicy, SourceFeatureSupport,
    UnifiedDepthAdjudicator)

OUT=ROOT/'analysis/r341_unified_depth_20260929'


def models():
    f=json.loads(BASE_FREEZE.read_text(encoding='utf-8'))
    assert f['model_sha256']==sha(ROOT/f['selected_file'])
    assert f['runtime_sha256']==sha(ROOT/'training/causal_depth_network_r337.py')
    return CoverageGuardedNeuralDecoder.from_report(), DepthMLP.load(ROOT/f['selected_file']), DepthPolicy(**f['depth_policy'])


def cache(path, decoder, model, depth_policy):
    file=OUT/'cache'/(sha(path/'meta.json')[:16]+'.npz')
    manifest_file=file.with_suffix('.json')
    raw=[{'path':(path/n).relative_to(ROOT).as_posix(),'sha256':sha(path/n)}
         for n in ('meta.json','magnetic_raw.csv','live_predictions.csv','events.csv','stage_commands.csv')]
    if file.exists():
        side=json.loads(manifest_file.read_text(encoding='utf-8'))
        assert side['raw_manifest']==raw
        with np.load(file,allow_pickle=False) as a: data={k:a[k].copy() for k in a.files}
        return data,side
    d=prepare(path,decoder);p=profiles(d,decoder,1.,'final');base=apply_profiles(d,p,depth_policy)[0]
    x=features(d,decoder,p,base);good=d['valid'];delta=np.clip(model.predict(x[good]).reshape(-1),-1.,1.)
    proposal=np.clip(base[good,2]+delta,p[0][0],p[0][-1])
    data={'time':d['time'],'valid':good,'old':d['live'],'q':d['q'],'action':d['action'],
          'calibrated_action':d['calibrated_action'],'constant':d['constant'],'epoch':d['epoch'],
          'base':base,'x':x,'z':p[0],'cost':p[1],'phi':p[2],'proposal':proposal}
    file.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(file,**data)
    side={'path':path.relative_to(ROOT).as_posix(),'mode':d['mode'],
          'trial':int(d['meta']['trial_parameters']['trial_id']),'raw_manifest':raw,
          'cache':file.relative_to(ROOT).as_posix(),'b0_amplitude_check':d['b0_amplitude_check']}
    dump(manifest_file,side);return data,side


def data_for_summary(a,side):
    return {'time':a['time'],'valid':a['valid'],'live':a['old'],'q':a['q'],
            'calibrated_action':a['calibrated_action'],'constant':a['constant'],'mode':side['mode']}


def apply(a,policy,distance):
    tracker=UnifiedDepthAdjudicator(policy);out=a['old'].copy();j=0
    restricted=np.zeros(len(out),bool);continuity=restricted.copy();weights=np.full(len(out),np.nan)
    for i,t in enumerate(a['time']):
        if not a['valid'][i]:tracker.reset();continue
        if not a['z'][0]<=a['old'][i,2]<=a['z'][-1]:tracker.reset();j+=1;continue
        result=tracker.step(t,a['z'],a['cost'][j],a['phi'][j],a['proposal'][j],distance[j],token=int(a['epoch'][i]))
        out[i,2]=result['z_mm'];restricted[i]=result['proposal_field_restricted']
        continuity[i]=result['used_continuity'];weights[i]=result['source_support_weight'];j+=1
    assert j==len(a['cost'])
    np.testing.assert_array_equal(out[:,:2],a['old'][:,:2])
    return out,restricted,continuity,weights


def select():
    if (OUT/'freeze.json').exists():raise FileExistsError('Preserve frozen R341')
    decoder,model,dp=models();reference=[];owners=[];source=[];validation=[]
    declared={'source_training_ids':[i for i in range(86,110) if i not in VALIDATION_IDS],
        'source_validation_ids':list(VALIDATION_IDS),'neural_strengths':[4.,16.,64.],
        'continuity':[0.,.002,.01],'support_quantiles':[.90,.99],'support_disabled_limit':1e6,
        'qualification':'Source real-Z/grid raw and centered MAE <= old+.01mm; static Z step P95 <=old+.01mm',
        'target_fit_rows':0,'target_selection_rows':0,'prospective_blind':False,
        'scope':'final Z adjudication with unchanged saved XY/permission; no hardware or UI'}
    dump(OUT/'declared_protocol.json',declared)
    for number in range(86,110):
        path=next(SOURCE_D.glob(f'Trial_{number:06d}_*'));a,side=cache(path,decoder,model,dp)
        source.append(side)
        if number in VALIDATION_IDS:validation.append((a,side));continue
        # Training-mask/stride identical to R340, then every fourth reference
        # example for bounded real-time nearest-support work.
        ids=np.flatnonzero(a['valid'] & a['calibrated_action'])[::4][::4]
        reference.append(np.clip((a['x'][ids]-model.mean)/model.scale,-8.,8.));owners.extend([number]*len(ids))
        print('Source support',number,len(ids),flush=True)
    xx=np.stack(np.meshgrid(decoder.atlas.x,decoder.atlas.y,decoder.atlas.z,indexing='ij'),axis=-1).reshape(-1,3)[::3]
    field=decoder.atlas.features(decoder.atlas.forward(xx))
    anchor=np.column_stack((xx/np.array([10,10,3]),field,np.zeros((len(xx),34)),xx[:,2]/3.,np.zeros((len(xx),2))))
    reference.append(np.clip((anchor-model.mean)/model.scale,-8.,8.));owners.extend([-1]*len(anchor))
    reference=np.vstack(reference);owners=np.asarray(owners)
    file=OUT/'source_support.npz';np.savez_compressed(file,mean=model.mean,scale=model.scale,reference=reference,owner=owners)
    support=SourceFeatureSupport(model.mean,model.scale,reference)
    loo=[]
    for number in declared['source_training_ids']:
        held=reference[owners==number][::4];other=reference[owners!=number]
        temporary=SourceFeatureSupport(model.mean,model.scale,other)
        # Convert normalized probes back to input coordinates for one exact
        # normalization path. No same-Trial neighbor can set its support limit.
        loo.extend(temporary.distance(held*model.scale+model.mean).tolist())
    limits=[max(float(np.quantile(loo,q)),1e-6) for q in declared['support_quantiles']]+[declared['support_disabled_limit']]
    validation.sort(key=lambda r:list(VALIDATION_IDS).index(r[1]['trial']))
    distances=[support.distance(a['x'][a['valid']]) for a,s in validation]
    old=[summary(data_for_summary(a,s),a['old']) for a,s in validation]
    candidates=[]
    for limit in limits:
        for neural in declared['neural_strengths']:
            for continuity in declared['continuity']:
                policy=UnifiedDepthPolicy(neural_strength=neural,continuity=continuity,support_limit=limit)
                rows=[summary(data_for_summary(a,s),apply(a,policy,d)[0]) for (a,s),d in zip(validation,distances)]
                qualified=all(rows[j][k]<=old[j][k]+.01 for j in (3,4,5) for k in ('z_raw_mae_mm','z_centered_mae_mm'))
                qualified &= rows[5]['z_step_p95_mm'] <=old[5]['z_step_p95_mm']+.01
                score=float(np.mean([objective(r) for r in rows]));candidates.append({'policy':asdict(policy),'qualified':bool(qualified),'objective':score,'trials':rows})
                print('Source policy',round(limit,4),neural,continuity,round(score,5),bool(qualified),flush=True)
    qualified=[c for c in candidates if c['qualified'] and c['objective']<np.mean([objective(r) for r in old])]
    selected=min(qualified,key=lambda c:c['objective']) if qualified else None
    report={'declared':declared,'selected':selected,'candidates':candidates,'baseline':old,
            'source_support_examples':len(reference),'leave_one_trial_quantiles':limits,
            'source_records':source,'target_fit_rows':0,'target_selection_rows':0}
    dump(OUT/'source_selection.json',report)
    if selected:
        dependencies=[ROOT/'training/unified_depth_runtime_r341.py',ROOT/'training/develop_unified_depth_r341.py',
            BASE_FREEZE,ROOT/json.loads(BASE_FREEZE.read_text(encoding='utf-8'))['selected_file'],
            ROOT/'training/causal_depth_network_r337.py',ROOT/'training/z_evidence_runtime_r334.py',
            file,OUT/'source_selection.json']
        dump(OUT/'freeze.json',{'schema':'R341-FrozenSourceFinalDepth-1','policy':selected['policy'],
            'support_file':file.relative_to(ROOT).as_posix(),'target_fit_rows':0,'target_selection_rows':0,
            'dependencies':[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for p in dependencies],
            'prospective_blind':False})
    print('FROZEN' if selected else 'NOT QUALIFIED',None if selected is None else selected['policy'],flush=True)


def evaluate(device):
    f=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'))
    assert f['target_fit_rows']==f['target_selection_rows']==0
    for item in f['dependencies']:assert sha(ROOT/item['path'])==item['sha256']
    policy=UnifiedDepthPolicy(**f['policy']);support=SourceFeatureSupport.load(ROOT/f['support_file'])
    decoder,model,dp=models();rows=[]
    for path in records(device):
        a,s=cache(path,decoder,model,dp);distance=support.distance(a['x'][a['valid']])
        xyz,restricted,continuity,weight=apply(a,policy,distance)
        d=data_for_summary(a,s);old,new=summary(d,a['old']),summary(d,xyz)
        stream=OUT/'streams'/(sha(path/'meta.json')[:16]+'.npz');stream.parent.mkdir(exist_ok=True)
        np.savez_compressed(stream,time=a['time'],q=a['q'],valid=a['valid'],old=a['old'],new=xyz,
            base=a['base'],calibrated=a['calibrated_action'],action=a['action'],weights=weight,
            restricted=restricted,continuity=continuity)
        rows.append({**s,'stream':stream.relative_to(ROOT).as_posix(),'old':old,'new':new,
            'xy_and_validity_unchanged':True,'restricted_frames':int(restricted.sum()),
            'support_weight_percentiles':np.percentile(weight[np.isfinite(weight)],[5,50,95]).tolist()})
        print(device,s['trial'],s['mode'],round(old['z_raw_mae_mm'],4),'->',round(new['z_raw_mae_mm'],4),flush=True)
    macro={key:{k:float(np.mean([r[k][key] for r in rows])) for k in ('old','new')}
        for key in ('z_raw_mae_mm','z_centered_mae_mm','z_step_p95_mm','z_step_max_mm')}
    dump(OUT/f'report_{device}.json',{'freeze':f,'rows':rows,'trial_macro':macro,
        'new_online_test':False,'full_r332_replayed':False,'parameter_selection':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('select','source','target'));args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    select() if args.task=='select' else evaluate(args.task)
