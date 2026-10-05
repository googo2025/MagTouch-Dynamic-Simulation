"""Repeatable virtual Trials for mechanism studies, not device-model promotion.

Public loader deliberately keeps observations separate from action references.
Every counterfactual twin shares geometry, timestamps and the same noise.
No actual instrument, frozen inverse weights or paper file is changed.
"""
from pathlib import Path
import argparse, csv, json, time
import numpy as np
from training.run_forward_r361 import ROOT, dump, sha
from training.forward_complex_r363 import AxisForward
from training.augment_forward_r361 import append_segment
from training.simulation_discovery_r367 import OUT as CORE, GEN

OUT = ROOT / 'analysis/r368_virtual_experiment_lab_20261001'
FIELDS = [f'S{s}_B{axis}_uT' for s in range(1,6) for axis in 'xyz']


def motion(family, depth, speed, hold, reverse):
    points = [[0.,0.,-1.]] * 12
    def move(q, dwell=hold): append_segment(points,q,speed,dwell)
    move([0,0,depth])
    if family == 'circle':
        move([4.25,0,depth])
        for phase in np.linspace(0,reverse*2*np.pi,97)[1:]:
            move([4.25*np.cos(phase),4.25*np.sin(phase),depth],0)
        move([0,0,depth])
    elif family == 'radial':
        for phase in reverse*np.arange(8)*np.pi/4:
            move([4*np.cos(phase),4*np.sin(phase),depth]);move([0,0,depth])
    elif family == 'pure_z':
        for z in (max(.45,depth*.5),depth,max(.45,depth*.75),depth,max(.45,depth*.5),depth):
            move([0,0,z])
    elif family == 'compound':
        q = [[3,0,max(.45,depth*.6)],[0,3,depth],[-3,0,max(.45,depth*.75)],
             [0,-3,depth],[0,0,max(.45,depth*.6)]]
        if reverse < 0: q = q[::-1]
        for p in q: move(p)
    else: raise ValueError('Unknown standard mechanism motion')
    move([*points[-1][:2],-1],2.)
    q=np.array(points,float);return q,np.arange(len(q))*.1


def load_virtual_trial(path, *, allow_stress=False):
    """Same research role as a measured Trial; never mislabel as measured.

    Model inference should receive `observation`, not `reference` or meta.
    Training consumers must explicitly opt into counterfactual stress data.
    """
    path=Path(path);meta=json.loads((path/'meta.json').read_text(encoding='utf8'))
    if meta['data_origin'] != 'empirical_simulation':
        raise ValueError('Virtual origin required')
    if meta['mechanism_stress'] and not allow_stress:
        raise ValueError('Counterfactual stress Trials require explicit opt-in')
    with np.load(path/'observations.npz',allow_pickle=False) as a:
        obs={k:a[k].copy() for k in ('t_s','delta_b_uT','sample_valid','reset')}
    with np.load(path/'reference.npz',allow_pickle=False) as a:
        ref={k:a[k].copy() for k in ('q_mm','primary')}
    if len(obs['t_s']) != len(ref['q_mm']):raise ValueError('Trial length mismatch')
    return {'meta':meta,'observation':obs,'reference':ref}


def generate():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data_catalog.json').exists():
        print('Preserved existing virtual dataset');return
    started=time.perf_counter();g=AxisForward.load(GEN/'models/generator.json')
    sigma=np.array(json.loads((GEN/'synthetic_manifest.json').read_text(encoding='utf8'))['noise_sigma_uT'])
    protocol={'schema':'MagTouch-VirtualExperimentLab-1','purpose':'reproduce phenomena and algorithmic interventions; faster research experiments',
        'data_origin':'empirical_simulation','generator_sha256':sha(GEN/'models/generator.json'),
        'families':['circle','radial','pure_z','compound'],'depths_mm':[.6,1.4,2.2],
        'speeds_mm_s':[.5,1.,2.],'dwell_s':[0.,.8,2.],'direction_signs':[1,-1],
        'history_strengths':[0.,1.,2.],'history_2_scope':'counterfactual stress, not a calibrated physical material',
        'noise_seed_by_geometry':'368000 + geometry index; same realization for all history strengths',
        'sampling_hz':10,'tip_diameter_mm':5,'independent_devices':1,
        'current_role':'frozen inverse mechanism evaluation only; no model selection on this scan',
        'future_training':'can be assigned new TRAIN/VALIDATION roles only with a NEW held-out test set',
        'reference':'known virtual input coordinates; not independently measured real contact',
        'three_stage_contract':'observations, action references, inverse predictions saved separately',
        'hardware_operated':False,'formal_model_modified':False}
    dump(OUT/'data_protocol.json',protocol)
    entries=[];idx=0;frames=0
    for family in protocol['families']:
        for depth in protocol['depths_mm']:
            for speed in protocol['speeds_mm_s']:
                for hold in protocol['dwell_s']:
                    for direction in protocol['direction_signs']:
                        q,t=motion(family,depth,speed,hold,direction)
                        equilibrium=g.static(q);history=g.rollout(q,t)-equilibrium
                        noise=np.random.default_rng(368000+idx).normal(size=(len(q),15))*sigma
                        for strength in protocol['history_strengths']:
                            ident=f'VIRTUAL_{idx:04d}_H{int(strength)}';path=OUT/'trials'/ident
                            path.mkdir(parents=True,exist_ok=True)
                            b=equilibrium+strength*history+noise;reset=np.r_[True,np.zeros(len(t)-1,bool)]
                            valid=np.isfinite(b).all(1);primary=(q[:,2]>=.4)&(q[:,2]<=3)&(np.abs(q[:,:2])<=6).all(1)
                            np.savez_compressed(path/'observations.npz',t_s=t,delta_b_uT=b,sample_valid=valid,reset=reset)
                            np.savez_compressed(path/'reference.npz',q_mm=q,primary=primary)
                            with (path/'magnetic_virtual.csv').open('w',encoding='utf-8-sig',newline='') as f:
                                w=csv.writer(f);w.writerow(['t_s','sample_valid','reset']+FIELDS)
                                for i in range(len(t)):w.writerow([round(float(t[i]),8),int(valid[i]),int(reset[i])]+b[i].tolist())
                            with (path/'action_reference.csv').open('w',encoding='utf-8-sig',newline='') as f:
                                w=csv.writer(f);w.writerow(['t_s','generated_x_mm','generated_y_mm','generated_z_mm','evaluation_domain'])
                                for i in range(len(t)):w.writerow([round(float(t[i]),8)]+q[i].tolist()+[int(primary[i])])
                            entry={'id':ident,'geometry_id':idx,'family':family,'max_depth_mm':depth,
                                'speed_mm_s':speed,'dwell_s':hold,'direction':direction,'history_strength':strength,
                                'mechanism_stress':strength>1,'data_origin':'empirical_simulation',
                                'role':'mechanism_test','noise_seed':368000+idx,'frames':len(q),
                                'duration_s':float(t[-1]),'generator_sha256':protocol['generator_sha256'],
                                'path':path.relative_to(ROOT).as_posix(),
                                'hashes':{name:sha(path/name) for name in ('observations.npz','reference.npz','magnetic_virtual.csv','action_reference.csv')}}
                            dump(path/'meta.json',entry);entries.append(entry);frames+=len(q)
                        idx+=1
                        if idx%24==0:print('virtual geometries',idx,'Trials',len(entries),flush=True)
    elapsed=time.perf_counter()-started
    dump(OUT/'data_catalog.json',{'protocol_sha256':sha(OUT/'data_protocol.json'),'trials':entries,
        'geometry_count':idx,'trial_count':len(entries),'frame_count':frames,
        'generated_duration_s':float(sum(r['duration_s'] for r in entries)),
        'wall_generation_s':elapsed,'speedup_definition':'sum of serial virtual action durations / data generation wall time; excludes inverse replay and physical setup',
        'allow_normal_and_stress_pooling':False})
    print('virtual dataset complete',len(entries),frames,'generation s',elapsed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['generate']);args=p.parse_args()
    generate()
