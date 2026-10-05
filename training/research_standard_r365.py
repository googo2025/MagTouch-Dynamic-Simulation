"""Fixed standard-action counterfactuals for the frozen MagTouch module chain."""
import json
import numpy as np
from training.augment_forward_r361 import append_segment
from training.forward_complex_r363 import AxisForward
from training.research_formal_r365 import OUT, VIRTUAL, LABELS, replay_fields, metrics
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
from training.run_forward_r361 import dump


def standard_actions():
    actions = []
    for mode in ('grid', 'pure_z', 'radial', 'circle4p25', 'circle5p25', 'compound'):
        points = [[0., 0., -1.]] * 12
        def move(q, speed=1., hold=.4): append_segment(points, q, speed, hold)
        if mode == 'grid':
            for x, y in ((-2,-2), (0,-2), (2,-2), (2,0), (0,0), (-2,0), (-2,2), (0,2), (2,2)):
                move([x,y,-1],2,.2); move([x,y,.6],1,.6); move([x,y,-1],1,.5)
        elif mode == 'pure_z':
            for _ in range(2):
                for z in (.6,1.,1.4,2.2,1.4,1.,.6,-1): move([0,0,z],.7,.6)
        elif mode == 'radial':
            move([0,0,1.4])
            for theta in np.arange(8)*np.pi/4:
                move([4*np.cos(theta),4*np.sin(theta),1.4],2,.5); move([0,0,1.4],2,.5)
        elif mode.startswith('circle'):
            r=4.25 if mode=='circle4p25' else 5.25
            move([0,0,1.4]); move([r,0,1.4],2,.5)
            for theta in np.linspace(0,2*np.pi,129)[1:]: move([r*np.cos(theta),r*np.sin(theta),1.4],2,0)
            move([0,0,1.4],2,.5)
        else:
            move([0,0,.6]);
            for q in ([3,0,1.4],[0,3,2.2],[-3,0,1.4],[0,-3,.6],[0,0,1.4]): move(q,1.4,.6)
        move([*points[-1][:2],-1],1,2.)
        q=np.asarray(points,float); actions.append({'id':'STD_'+mode,'family':mode,'q':q,'t':np.arange(len(q))*.1})
    return actions


def main():
    folder=OUT/'standard_streams';folder.mkdir(parents=True,exist_ok=True)
    model=AxisForward.load(VIRTUAL/'models/generator.json')
    atlas=CoverageGuardedNeuralDecoder.from_report().atlas
    info=json.loads((VIRTUAL/'synthetic_manifest.json').read_text(encoding='utf8'))
    sigma=np.asarray(info['noise_sigma_uT']);rng=np.random.default_rng(365)
    rows=[]
    for a in standard_actions():
        q,t=a['q'],a['t'];noise=rng.normal(size=(len(t),15))*sigma
        destination=folder/(a['id']+'.npz')
        if destination.exists():
            rows.append(json.loads(destination.with_suffix('.json').read_text(encoding='utf8')));continue
        empirical=model.rollout(q,t);memory=empirical-model.static(q)
        contact=q[:,2]>=.4;equilibrium=np.zeros_like(empirical);equilibrium[contact]=atlas.forward(q[contact])
        primary=contact&model.support(q)
        # Same path/noise in all arms: independent empirical generator versus
        # exact-atlas equilibrium positive control and its historical residual.
        arms={'empirical_dynamic':empirical+noise,'atlas_static_control':equilibrium+noise,
              'atlas_plus_history_control':equilibrium+memory+noise}
        saved={'q':q,'t':t,'primary':primary};row={'id':a['id'],'family':a['family'],'scores':{}}
        for name,b in arms.items():
            result=replay_fields(b,t,components=True)
            row['scores'][name]={key:metrics(result[key],q,t,primary,result['valid']) for key in LABELS}
            row['scores'][name]['safe_output_frames']=int(((q[:,2]<=-.5)&result['valid']).sum())
            saved[name+'_field']=b;saved[name+'_valid']=result['valid']
            saved.update({name+'_'+key:result[key] for key in LABELS})
        np.savez_compressed(destination,**saved);dump(destination.with_suffix('.json'),row);rows.append(row)
        print('standard',a['id'],'completed',flush=True)
    dump(OUT/'standard_report.json',{'rows':rows,'component_labels':LABELS,'seed':365,
        'control_definition':'same path and same noise; exact frozen atlas is a consistency control, NOT independent physics validation',
        'component_mask':'unchanged final output permission shared by all decoder variants',
        'model_training':False,'hardware':False})


if __name__=='__main__':main()
