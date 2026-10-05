"""Frozen virtual-only inverse on a crossed virtual experimental protocol.

The scan is TEST data, not a hyperparameter search. Observation/reference/output
are separate. A software-state fault test is explicitly NOT a natural material
wrong-branch event. No real device or paper-model weights are changed.
"""
import argparse,csv,json,time
import numpy as np
from training.virtual_experiment_lab_r368 import ROOT,OUT,load_virtual_trial,dump,sha
from training.simulation_depth_r368 import frozen_context,decode_core,prepare_depth,depth_predictions,z_score
from training.branch_aware_inverse_r138 import BranchAwareInverse
from training.evidence_guarded_branch_r162 import EvidenceGuardedBranchInverse
from training.simulation_discovery_r367 import prepare,memory


def case(entry):
    data=load_virtual_trial(ROOT/entry['path'],allow_stress=True)
    o,r=data['observation'],data['reference']
    return {'id':entry['id'],'b':o['delta_b_uT'],'t':o['t_s'],
        'gate':o['sample_valid']&(np.linalg.norm(o['delta_b_uT'],axis=1)>=70),
        'q':r['q_mm'],'primary':r['primary']}


def scan():
    ctx=frozen_context();atlas,basis,pick,xy=ctx[:4]
    catalog=json.loads((OUT/'data_catalog.json').read_text(encoding='utf8'))
    folder=OUT/'scan_predictions';folder.mkdir(exist_ok=True)
    rows=[];start=time.perf_counter()
    for e in catalog['trials']:
        dest=folder/(e['id']+'.npz');meta=dest.with_suffix('.json')
        if meta.exists():rows.append(json.loads(meta.read_text(encoding='utf8')));continue
        a=case(e);decode_core(a,atlas,basis,pick,xy);prepare_depth(a,atlas,basis)
        pred,restrictions=depth_predictions(a,*ctx[4:])
        pred.update(F=a['base'],M=a['memory'],G=a['guarded'])
        np.savez_compressed(dest,accepted=a['gate'],**pred)
        row={**{k:e[k] for k in ('id','geometry_id','family','max_depth_mm','speed_mm_s','dwell_s','direction','history_strength','mechanism_stress')},
            'scores':{k:z_score(a,p) for k,p in pred.items()},
            'xy_local_count':int(a['local_count']),'xy_veto_count':int(a['veto_count']),
            'z_profile_veto_count':int(a['profile_veto_count']),
            'z_proposal_restricted_count':restrictions,'prediction_sha256':sha(dest)}
        dump(meta,row);rows.append(row)
        if len(rows)%24==0:print('virtual crossed scan',len(rows),'of',len(catalog['trials']),flush=True)
    report={'rows':rows,'wall_scan_s':time.perf_counter()-start,'model_selection_on_scan':False,
        'scope':'counterfactual history twins separated; empirical surrogate; fixed virtual-trained inverse; same observation stream and gate',
        'depth_freeze_sha256':sha(OUT/'depth_study/freeze.json')}
    dump(OUT/'scan_report.json',report)
    fields=['id','geometry_id','family','max_depth_mm','speed_mm_s','dwell_s','direction','history_strength','mechanism_stress','stage',
        'coverage','xy_mean_mm','z_mae_mm','xyz_mean_mm','xy_translation_removed_mm','z_translation_removed_mm','z_error_step_p95_mm','z_error_step_max_mm',
        'bias_x_mm','bias_y_mm','bias_z_mm','constant_z_pairs','constant_z_step_p95_mm','constant_z_jumps_over_0p2']
    with (OUT/'scan_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            for stage,s in r['scores'].items():
                v={k:r[k] for k in fields[:9]};v['stage']=stage
                for k in fields[10:18]:v[k]=s.get(k)
                for k,val in zip(('bias_x_mm','bias_y_mm','bias_z_mm'),s.get('bias_xyz_mm',[None]*3)):v[k]=val
                c=s.get('constant_reference_z_steps',{});v.update(constant_z_pairs=c.get('pairs'),constant_z_step_p95_mm=c.get('p95_mm'),constant_z_jumps_over_0p2=c.get('over_0p2_mm'))
                w.writerow(v)
    print('scan complete',len(rows),'wall s',report['wall_scan_s'],flush=True)


def branch_faults():
    """Predeclared state perturbations, not field manipulation or hidden truth.

    Shift the tracker's PREVIOUS ESTIMATE at a fixed path index. Both arms get
    the identical fault. Commands are read only for scoring, never for tracker
    initialization. All perturbations are reported, with a fault-free arm.
    """
    ctx=frozen_context();atlas,basis,pick,xy=ctx[:4]
    catalog=json.loads((OUT/'data_catalog.json').read_text(encoding='utf8'))
    chosen=[r for r in catalog['trials'] if r['history_strength']==1 and r['max_depth_mm']==1.4 and
        r['speed_mm_s']==1 and r['dwell_s']==.8 and r['direction']==1]
    offsets=[[0.,0.],[4.,0.],[-4.,0.],[0.,4.],[0.,-4.]]
    protocol={'records':[r['id'] for r in chosen],'offsets_of_previous_estimate_xy_mm':offsets,
        'fault_index':'floor(N/3), fixed before replay, regardless of prediction error',
        'no_command_in_fault_initializer':True,'arms':['continuity_without_veto','evidence_veto_ratio4'],
        'scope':'software-state fault recovery intervention; NOT evidence that this surrogate naturally produces the same erroneous basins as the real device'}
    dump(OUT/'branch_fault_protocol.json',protocol)
    folder=OUT/'branch_faults';folder.mkdir(exist_ok=True);rows=[]
    for e in chosen:
        a=case(e);decode_core(a,atlas,basis,pick,xy)
        fault=len(a['t'])//3
        for off in offsets:
            models=[BranchAwareInverse(atlas,spatial_penalty=.03,innovation_gain=2.,activation_distance_mm=1.),
                EvidenceGuardedBranchInverse(atlas,spatial_penalty=.03,innovation_gain=2.,activation_distance_mm=1.,max_local_cost_ratio=4.)]
            keys=protocol['arms'];pred={k:np.full_like(a['q'],np.nan) for k in keys};veto=np.zeros(len(a['t']),bool)
            localcost=np.full((len(a['t']),2),np.nan);globalcost=localcost.copy();applied=[False,False]
            for i in range(len(a['t'])):
                if not a['gate'][i]:
                    for m in models:m.reset()
                    continue
                for j,(key,m) in enumerate(zip(keys,models)):
                    if i==fault and m.previous is not None:
                        m.previous[:2]=np.clip(m.previous[:2]+off,[atlas.x[0],atlas.y[0]],[atlas.x[-1],atlas.y[-1]])
                        applied[j]=True
                    r=m.predict_frame(a['corrected'][i],time_ns=int(a['t'][i]*1e9)+10_000_000_000,
                        b0_id=0,stream_id=0,global_xyz=a['memory'][i])
                    pred[key][i]=r['xyz_mm'];localcost[i,j]=r['local_field_cost'];globalcost[i,j]=r['global_field_cost']
                    if j:veto[i]=r['local_branch_vetoed_by_field']
            scores={}
            window=(a['t']>=a['t'][fault])&(a['t']<=a['t'][fault]+2)&a['gate']&a['primary']
            for key,p in pred.items():
                error=np.linalg.norm(p[:,:2]-a['q'][:,:2],axis=1)
                good=np.flatnonzero(window&(error<=.5))
                scores[key]={'post_fault_xy_mean_mm':float(error[window].mean()) if window.any() else None,
                    'post_fault_over_1mm_frames':int((error[window]>1).sum()),
                    'first_within_0p5mm_s':float(a['t'][good[0]]-a['t'][fault]) if len(good) else None}
            ident=e['id']+'_fault_'+str(offsets.index(off))
            np.savez_compressed(folder/(ident+'.npz'),t=a['t'],q=a['q'],gate=a['gate'],veto=veto,
                local_cost=localcost,global_cost=globalcost,**pred)
            rows.append({'id':ident,'trial':e['id'],'family':e['family'],'offset_xy_mm':off,'fault_frame':fault,
                'fault_time_s':float(a['t'][fault]),'fault_applied':applied,'veto_count':int(veto.sum()),'scores':scores})
    dump(OUT/'branch_fault_report.json',{'protocol':protocol,'rows':rows})
    print('paired branch fault tests',len(rows),'vetoes',sum(r['veto_count'] for r in rows),flush=True)


def natural_branch_ablation():
    """Activation-conditioned diagnosis, NOT an unbiased global efficacy mean."""
    ctx=frozen_context();atlas,basis,pick,xy=ctx[:4]
    catalog=json.loads((OUT/'data_catalog.json').read_text(encoding='utf8'))
    scan_rows=json.loads((OUT/'scan_report.json').read_text(encoding='utf8'))['rows']
    # Include every rule activation, not only interventions that look improved.
    chosen={r['id'] for r in scan_rows if r['xy_veto_count']>0}
    chosen.update(r['id'] for r in catalog['trials'] if r['history_strength']==1 and r['max_depth_mm']==1.4 and
        r['speed_mm_s']==1 and r['dwell_s']==.8 and r['direction']==1)
    folder=OUT/'natural_branch_ablation';folder.mkdir(exist_ok=True);rows=[]
    for e in catalog['trials']:
        if e['id'] not in chosen:continue
        a=case(e);prepare(a,atlas,basis)
        p,h,field=memory(a,atlas,basis,pick['retention'],pick['strength'])
        models=[BranchAwareInverse(atlas,spatial_penalty=.03,innovation_gain=2.,activation_distance_mm=1.),
            EvidenceGuardedBranchInverse(atlas,spatial_penalty=.03,innovation_gain=2.,activation_distance_mm=1.,max_local_cost_ratio=4.)]
        keys=['without_field_veto','with_field_veto'];pred={k:np.full_like(p,np.nan) for k in keys}
        veto=np.zeros(len(p),bool);pre_veto=np.full_like(p,np.nan)
        for i in range(len(p)):
            if not a['gate'][i]:
                for m in models:m.reset()
                continue
            for j,(k,m) in enumerate(zip(keys,models)):
                r=m.predict_frame(field[i],time_ns=int(a['t'][i]*1e9)+10_000_000_000,b0_id=0,stream_id=0,global_xyz=p[i])
                pred[k][i]=r['xyz_mm'];pred[k][i,2]=a['base'][i,2]
                if j:
                    veto[i]=r['local_branch_vetoed_by_field'];pre_veto[i]=r['local_xyz_mm']
        # All causal estimates identical to previously saved guarded arm.
        recorded=np.load(OUT/'scan_predictions'/(e['id']+'.npz'))
        np.testing.assert_array_equal(pred['with_field_veto'],recorded['G'])
        active=np.flatnonzero(veto&a['primary']);actions=[]
        for i in active:
            actions.append({'time_s':float(a['t'][i]),
                'rejected_local_xy_error_mm':float(np.linalg.norm(pre_veto[i,:2]-a['q'][i,:2])),
                'accepted_global_xy_error_mm':float(np.linalg.norm(pred['with_field_veto'][i,:2]-a['q'][i,:2]))})
        np.savez_compressed(folder/(e['id']+'.npz'),t=a['t'],q=a['q'],gate=a['gate'],veto=veto,pre_veto=pre_veto,**pred)
        rows.append({'id':e['id'],'history_strength':e['history_strength'],'family':e['family'],
            'veto_count':int(veto.sum()),'scores':{k:z_score(a,pr) for k,pr in pred.items()},'working_domain_activations':actions})
    dump(OUT/'natural_branch_ablation.json',{'rows':rows,
        'selection':'all 25 scan records with any veto plus four predeclared nominal controls',
        'scope':'activation-conditioned diagnostic; not a scan-wide efficacy estimate; both arms differ only in the fourfold veto rule',
        'no_state_fault':True,'no_model_fitting':True})
    print('natural branch diagnostics',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['scan','branch','natural_branch']);a=p.parse_args()
    {'scan':scan,'branch':branch_faults,'natural_branch':natural_branch_ablation}[a.task]()
