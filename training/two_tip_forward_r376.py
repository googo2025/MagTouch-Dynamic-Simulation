"""First old-1mm/cone-domain forward fit using archived source-only records.

Same spatial/axis-memory model as R363, separately fitted geometry coefficients.
This tests a second known geometry; it does not predict a never-measured tip.
No neural network is retrained, no production coefficients are replaced.
"""
from pathlib import Path
import json, sys
import numpy as np
from training.run_forward_r361 import ROOT, metric, dump, sha
from training.magnetic_forward_r361 import spatial_features, fit_readout, MagneticForward
from training.forward_complex_r363 import axis_history, AxisForward
from training.build_trajectory_dataset_v1 import assemble,load_magnetic,read_rows,command_segments,reference_origin

OUT=ROOT/'analysis/r376_two_tip_forward_20261002'
PROFILE='magnetic_layer_v1_5*5-had-old1mmTouch'
STATIC=ROOT/'config/had239_old1mm_121pt_continuous_augmented_r65_20260924.json'
KEY='5x5_had239_old1mm_121pt_continuous_augmented_r65'
TAUS=(.3,1.,4.)

def manifest():
    rows=[]
    for session,numbers,role in [('S20260924_A',[5,6,7],'train'),('S20260924_A',[8],'validation'),('S20260926_A',[3,4,5,6,7,8],'test')]:
        for n in numbers:
            paths=list((ROOT/'data'/session).glob(f'Trial_{n:06d}_*'));assert len(paths)==1
            p=paths[0];m=json.loads((p/'meta.json').read_text(encoding='utf8'))
            assert m['acquisition']['magnetic_layer_profile_id']==PROFILE and m['status']=='completed'
            rows.append({'id':f'{session}_{n:06d}','path':p.relative_to(ROOT).as_posix(),'role':role,
                'mode':m['trial_parameters']['command_mode'],'profile':PROFILE,'contact_z_mm':float(reference_origin(m)[2]),
                'recorded_indenter_id':m['trial_parameters'].get('indenter_id'),
                'geometry':'user-identified old 1mm/cone; exact angle/apex geometry not recoverable from legacy flat_d6mm field',
                'hashes':{f:sha(p/f) for f in ('meta.json','magnetic_raw.csv','stage_commands.csv','events.csv')}})
    return rows

def load(row):
    p=ROOT/row['path'];m=json.loads((p/'meta.json').read_text(encoding='utf8'))
    full,_=assemble(load_magnetic(p/'magnetic_raw.csv'),command_segments(read_rows(p/'stage_commands.csv')),read_rows(p/'events.csv'),reference_origin(m))
    ix=np.arange(0,len(full['t_ns']),4);a={k:full[v][ix] for k,v in {'q':'xyz_mm','b':'delta_b_uT','valid':'input_valid','b0':'b0_id'}.items()}
    a['t']=(full['t_ns'][ix]-full['t_ns'][0])/1e9;a['reset']=np.r_[True,np.diff(a['b0'])!=0]
    known=np.isfinite(a['q']).all(1);a['primary']=a['valid']&known&(a['q'][:,2]>=.4)&(a['q'][:,2]<=1.6+1e-5)
    a['fit']=a['primary']|(a['valid']&known&(a['q'][:,2]<=-.2));a.update(row)
    return a

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'streams').mkdir(exist_ok=True)
    rows=manifest();mp=OUT/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text(encoding='utf8'))==rows
    else:dump(mp,rows)
    config=json.loads(STATIC.read_text(encoding='utf8'));ss=config['profiles'][KEY]['samples']
    q=np.array([[s['x'],s['y'],s['depth']] for s in ss]);b=np.array([s['delta'] for s in ss]);keep=(q[:,2]>=.4)&(q[:,2]<=1.6+1e-5)
    meta={'tip':PROFILE,'geometry_contract':'old 1mm/cone identity, no assumed cone angle','training_bounds_mm':[[-8,-8,.4],[8,8,1.6]],
          'static_source':str(STATIC.relative_to(ROOT)),'static_source_sha256':sha(STATIC),
          'static_source_role':'previously fitted calibration sample table, not independent validation or inverse predictions',
          'target_fit_rows':0,'heldout_test_used_for_selection':False,'physics_validated':False}
    static=MagneticForward(fit_readout(spatial_features(q[keep]),b[keep],np.ones(keep.sum()),.001),(),meta)
    data=[load(r) for r in rows];train=[r for r in data if r['role']=='train'];val=[r for r in data if r['role']=='validation']
    xx=[];yy=[];ww=[]
    for a in train:
        take=a['fit'];xx.append(axis_history(a['q'],a['t'],TAUS,a['reset'])[take]);yy.append((a['b']-static.rollout(a['q'],a['t'],a['reset']))[take]);ww.append(np.full(take.sum(),1/take.sum()))
    x=np.vstack(xx);y=np.vstack(yy);w=np.concatenate(ww);candidates=[];models={}
    for alpha in (.001,.01,.1):
        model=AxisForward(static,fit_readout(x,y,w,alpha),TAUS,True,dict(meta,alpha=alpha))
        obj=float(np.mean([metric(model.rollout(a['q'],a['t'],a['reset']),a)['nrmse'] for a in val]))
        candidates.append({'alpha':alpha,'validation_nrmse':obj});models[alpha]=model
    chosen=min(candidates,key=lambda a:a['validation_nrmse']);model=models[chosen['alpha']]
    model.save(OUT/'generator.json');dump(OUT/'freeze.json',{'selection':chosen,'candidates':candidates,'generator_sha256':sha(OUT/'generator.json'),
        'train_ids':[a['id'] for a in train],'validation_ids':[a['id'] for a in val],'test_used_to_fit_or_select':False,'target_fit_rows':0,
        'fit_method':'weighted ridge exact normal-equation solve; not an epoch-limited neural training run','static_calibration_samples':int(keep.sum())})
    result=[]
    for a in data:
        vb=model.rollout(a['q'],a['t'],a['reset']);sb=static.rollout(a['q'],a['t'],a['reset'])
        result.append({k:a[k] for k in ('id','path','role','mode')}|{'static':metric(sb,a),'dynamic':metric(vb,a),'frames':len(a['t'])})
        np.savez_compressed(OUT/'streams'/f"{a['id']}.npz",t=a['t'],q=a['q'],real_b=a['b'],virtual_b=vb,static_b=sb,primary=a['primary'])
    groups={role:{key:float(np.mean([r[key]['nrmse'] for r in result if r['role']==role])) for key in ('static','dynamic')} for role in ('train','validation','test')}
    dump(OUT/'report.json',{'status':'COMPLETE_EMPIRICAL_SECOND_GEOMETRY_FIT','records':result,'groups':groups,
        'neural_network_retrained':False,'cross_unseen_tip_prediction':False,'target_used':False,
        'scope':'next-day whole-record held-out field prediction for old1mm/cone, same architecture, fitted tip-specific coefficients',
        'geometry_identification':'user says cone; archived profile identifies old1mm, exact cone angle/radius still needs CAD/measurement',
        'legacy_record_reuse':'these recordings have historical analysis; retrospective holdout, not a new blind physical test'})
    dump(OUT/'two_tip_registry.json',{'schema':'MagTouch-KnownGeometryForwardRegistry-1','same_architecture':True,
        'models':{'final_5mm':{'path':'analysis/r363_complex_forward_20261001/models/generator.json','sha256':sha(ROOT/'analysis/r363_complex_forward_20261001/models/generator.json')},
                  'old_1mm_cone':{'path':'analysis/r376_two_tip_forward_20261002/generator.json','sha256':sha(OUT/'generator.json')}},
        'unknown_geometry':'unsupported, not interpolate arbitrary mesh shape from two examples'})
    print(json.dumps(groups,indent=2),flush=True)
if __name__=='__main__':main()
