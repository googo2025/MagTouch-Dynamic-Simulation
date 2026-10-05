"""Rank new actions using source-frozen disagreement, never target errors."""
from concurrent.futures import ProcessPoolExecutor
import argparse,json
import numpy as np
from training.domain_randomization_r369 import (ROOT,OUT,GEN,MANIFEST,geometry,replay,ResearchXY)
from training.forward_complex_r363 import AxisForward
from training.run_forward_r361 import dump,sha


def candidate(i):
    # Independent action geometry, not one of the augmentation Trial paths.
    rng=np.random.default_rng(369900+i);q,t,info=geometry(i,rng)
    g=AxisForward.load(GEN);b=g.rollout(q,t);contact=(q[:,2]>=.4)&g.support(q)
    a={'b':b,'t':t,'valid':np.ones(len(t),bool),'reset':np.r_[True,np.zeros(len(t)-1,bool)]}
    result=replay(a);m=contact&result['accepted']&np.isfinite(result['x']).all(1)
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    models=[ResearchXY()]+[ResearchXY.load(ROOT/r['file']) for r in freeze['selected'] if r['seed']==7]
    positions=np.array([result['x'][m,:2]*10+model.predict(result['x'][m]) for model in models])
    spread=float(np.sqrt(np.mean(np.sum(np.var(positions,axis=0),axis=1)))) if m.any() else 0.
    dist=json.loads((OUT/'source_parameter_distribution.json').read_text(encoding='utf8'))
    params=np.array(dist['clipped_parameters']);static=g.static(q);hist=b-static
    fields=[]
    for u in params[::4]:
        fields.append(static*np.repeat(1+.5*(u[:5]-1),3)+hist*np.repeat(1+.5*(u[5:10]-1),3)+.5*u[10:])
    field_spread=float(np.sqrt(np.mean(np.var(np.array(fields)[:,contact],axis=0)))/max(np.sqrt(np.mean(b[contact]**2)),1e-6))
    ident=f'PROSPECTIVE_{i:03d}'
    np.savez_compressed(OUT/'active_candidates'/f'{ident}.npz',q_mm=q,t_s=t,generated_field_uT=b,
                        original_output=result['pred'],accepted=result['accepted'])
    return {'id':ident,**info,'seed':369900+i,'frames':len(t),'duration_s':float(t[-1]),
            'xy_model_disagreement_mm':spread,'field_distribution_spread_nrmse':field_spread,
            'nominal_virtual_output_coverage':float(m.sum()/max(contact.sum(),1)),
            'path':(OUT/'active_candidates'/f'{ident}.npz').relative_to(ROOT).as_posix()}


def main(workers=1):
    f=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    assert f['target_fit_rows']==f['target_selection_rows']==0
    (OUT/'active_candidates').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(candidate,range(24)))
    # A relative priority only, not calibrated uncertainty or predicted error.
    matrix=np.array([[r['xy_model_disagreement_mm'],r['field_distribution_spread_nrmse'],1-r['nominal_virtual_output_coverage']] for r in rows])
    ranks=np.empty_like(matrix)
    for j in range(3):
        for i in range(len(rows)):ranks[i,j]=np.mean(matrix[:,j]<=matrix[i,j])
    for r,rank in zip(rows,ranks):r['priority_score']=float(rank.mean())
    ordered=sorted(rows,key=lambda r:r['priority_score'],reverse=True)
    # Three NEW geometries from distinct nonstandard families, plus controls.
    chosen=[];families=set()
    for r in ordered:
        if r['family'] not in ('depth_loop','stop_reverse','lissajous_xyz','piecewise_xyz'):continue
        if r['family'] in families:continue
        chosen.append(r);families.add(r['family'])
        if len(chosen)==3:break
    dump(OUT/'active_experiment_ranking.json',{'candidates':ordered,'proposed_new_paths':chosen,
         'source_only':True,'target_outcomes_used':False,'simulation_truth_errors_used_for_ranking':False,
         'freeze_sha256':sha(OUT/'freeze.json'),'priority':'mean within-pool percentile of model disagreement, source field variability, and nominal refusal',
         'interpretation':'hypothesis-generating relative priority; must be checked with real acquisition'})
    dump(OUT/'prospective_comparison_protocol.json',{
        'status':'proposal only, not executable printer commands; no hardware or UI operated',
        'new_paths':[r['id'] for r in chosen],
        'controls':['circle r4.25mm','eight-direction radial','fixed-XY pure Z press/withdraw','reserved SPACE_DEPTH 29-waypoint path'],
        'source_and_target':True,'independent_repeats_per_device':3,
        'paired_comparison':'same raw B stream and timestamps, frozen original and each source-selected candidate; do not compare different recording runs as algorithm-only evidence',
        'primary_metrics':['unregistered XY/XYZ error','Z error','coverage','safe-air false outputs','Z jump/error-step quantiles'],
        'diagnostics':['one fixed XYZ offset per complete Trial, never primary score or deployment calibration','no output-derived time shifting'],
        'experimental_unit':'complete Trial; frames and optimizer seeds are not independent devices',
        'comparators':['original','real-only fine-tuned control','nominal virtual','source-domain-randomized','curriculum plus hard mining'],
        'sim_value_test':'augmentation beats matched real-only training on predeclared new REAL actions, across repeats, without sacrificing static/pure-Z/safety controls',
        'practical_gain_threshold':'set before acquisition using measurement timing/repeatability; not fitted to target outcomes',
        'freeze_sha256':sha(OUT/'freeze.json')})
    print('active candidates ranked',len(rows),'new real-test proposals',len(chosen),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);a=p.parse_args();main(a.workers)
