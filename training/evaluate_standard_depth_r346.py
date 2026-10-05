"""Source-only final choice, then no-selection complex-path evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from training.standard_motion_protocol_r344 import OUT as ROLES, load
from training.analyze_z_evidence_r334 import ROOT,dump,sha,summary
from training.develop_unified_depth_r341 import models,cache,apply,data_for_summary
from training.causal_depth_network_r337 import DepthMLP
from training.unified_depth_runtime_r341 import SourceFeatureSupport,UnifiedDepthPolicy
from training.fast_depth_features_r344 import augment

OUT=ROOT/'analysis/r346_complex_holdout_20260929'


def choose():
    file=OUT/'choice.json'
    if file.exists():raise FileExistsError('Preserve choice before heldout evaluation')
    candidates=[]
    for dim in (56,59):
        folder=ROOT/f'analysis/r346_centered_standard_depth_{dim}_20260929'
        selection=json.loads((folder/'source_selection.json').read_text(encoding='utf-8'))
        if selection['selected'] is not None:
            candidates.append({'freeze':(folder/'freeze.json').relative_to(ROOT).as_posix(),
               'freeze_sha256':sha(folder/'freeze.json'),'source_objective':selection['selected']['objective'],
               'dimension':dim,'source_selection_sha256':sha(folder/'source_selection.json')})
    if not candidates:raise ValueError('No Source-qualified candidate; heldout remains unopened')
    selected=min(candidates,key=lambda r:r['source_objective'])
    dump(file,{'schema':'SourceChoiceBeforeComplexScores-1','candidates':candidates,'selected':selected,
          'target_selection_rows':0,'complex_selection_rows':0,'roles_sha256':sha(ROLES/'dataset_roles.json'),
          'historically_seen_records':True,'complete_system_blind':False})
    print('Source-only choice',selected,flush=True)


def frozen():
    choice=json.loads((OUT/'choice.json').read_text(encoding='utf-8'));chosen=choice['selected'];path=ROOT/chosen['freeze']
    if sha(path)!=chosen['freeze_sha256']:raise ValueError('Frozen choice changed')
    f=json.loads(path.read_text(encoding='utf-8'))
    for dep in f['dependencies']:
        if sha(ROOT/dep['path'])!=dep['sha256']:raise ValueError('Frozen dependency changed: '+dep['path'])
    if any(f[k] for k in ('target_fit_rows','target_selection_rows','complex_fit_rows','complex_selection_rows')):
        raise ValueError('Source-standard-only fitting required')
    return f,DepthMLP.load(ROOT/f['model_file']),SourceFeatureSupport.load(ROOT/f['support_file'])


def proposal(a,decoder,model,f):
    b=dict(a);v=a['valid'];x=a['x'] if f['dimension']==56 else augment(decoder.atlas,a['time'],a['x'],v,a['epoch'])
    origin=a['old'][v,2] if f.get('proposal_origin')=='original' else a['base'][v,2]
    b['proposal']=np.clip(origin+np.clip(model.predict(x[v]).reshape(-1),-1,1),a['z'][0],a['z'][-1])
    return b


def evaluate():
    f,model,support=frozen();decoder,baseline_model,dp=models();policy=UnifiedDepthPolicy(**f['policy'])
    priorf=json.loads((ROOT/'analysis/r341_unified_depth_20260929/freeze.json').read_text(encoding='utf-8'))
    priorpolicy=UnifiedDepthPolicy(**priorf['policy']);priorsupport=SourceFeatureSupport.load(ROOT/priorf['support_file'])
    # The first access to complex labels happens here, after choice/freeze.
    rows=[]
    for role in ('selection','complex_final_evaluation'):
        for row in load(role):
            a,s=cache(ROOT/row['path'],decoder,baseline_model,dp)
            new,restricted,continuity,weights=apply(proposal(a,decoder,model,f),policy,support.distance(a['x'][a['valid']]))
            old=apply(a,priorpolicy,priorsupport.distance(a['x'][a['valid']]))[0]
            d=data_for_summary(a,s);destination=OUT/'streams'/f"{sha(ROOT/row['path']/'meta.json')[:16]}.npz"
            destination.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(destination,time=a['time'],q=a['q'],valid=a['valid'],action=a['action'],
                 calibrated=a['calibrated_action'],old=a['old'],prior=old,new=new,epoch=a['epoch'])
            record={**row,'stream':destination.relative_to(ROOT).as_posix(),
                 'original':summary(d,a['old']),'R341':summary(d,old),'new':summary(d,new),
                 'xy_and_permission_unchanged':True,'labels_not_in_inference':True,
                 'complex_state_layer_historical_exposure':role=='complex_final_evaluation',
                 'field_veto_rule':2,'accepted_frames':int(a['valid'].sum()),'restricted_frames':int(restricted.sum())}
            rows.append(record)
            print('frozen evaluation',role,'Target' if row['target'] else 'Source',row['trial'],
                 'Z raw',round(record['R341']['z_raw_mae_mm'],4),'->',round(record['new']['z_raw_mae_mm'],4),
                 'centered',round(record['R341']['z_centered_mae_mm'],4),'->',round(record['new']['z_centered_mae_mm'],4),flush=True)
    groups={}
    keys=('z_raw_mae_mm','z_centered_mae_mm','z_step_p95_mm','z_step_max_mm','xyz_raw_rmse_mm','xyz_centered_rmse_mm')
    for role,device in (('selection',False),('complex_final_evaluation',False),('complex_final_evaluation',True)):
        subset=[r for r in rows if r['role']==role and r['target']==device]
        groups[role+('_Target' if device else '_Source')]={'trials':len(subset),
           'macro':{key:{name:float(np.mean([r[name][key] for r in subset])) for name in ('original','R341','new')} for key in keys}}
    dump(OUT/'report.json',{'freeze':f,'choice_sha256':sha(OUT/'choice.json'),'rows':rows,'groups':groups,
        'target_fit_rows':0,'complex_fit_rows':0,'heldout_selection_rows':0,'fresh_online_test':False,
        'scope':'conditional saved XY/history/permission localization; not complete-system unseen evaluation'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('choose','evaluate'));a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);choose() if a.task=='choose' else evaluate()
