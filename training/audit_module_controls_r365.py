"""Verify incremental-Z removal using exact frozen engine and shared field."""
import json
import numpy as np
from training.research_formal_r365 import OUT,VIRTUAL,LABELS,replay_fields,metrics
from training.forward_complex_r363 import AxisForward
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
from training.run_forward_r361 import dump


def main():
    model=AxisForward.load(VIRTUAL/'models/generator.json')
    atlas=CoverageGuardedNeuralDecoder.from_report().atlas
    checked=[]
    for folder in ('standard_streams','virtual_streams'):
        for path in sorted((OUT/folder).glob('*.npz')):
            metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
            if not any('U' in s for s in metadata['scores'].values()):continue
            if metadata.get('increment_ablation')=='exact_frozen_engine_gain0':
                checked.append(metadata['id']);continue
            with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
            q,t=data['q'],data['t'];primary=data['primary'] if folder=='standard_streams' else data['core']
            if folder=='standard_streams':arms={name:data[name+'_field'] for name in metadata['scores']}
            else:
                empirical=data['empirical_field'];rollout=model.rollout(q,t)
                residual=rollout-model.static(q);noise=empirical-rollout
                mask=(q[:,2]>=.4)&(q[:,2]<=3);equilibrium=np.zeros_like(empirical);equilibrium[mask]=atlas.forward(q[mask])
                arms={'empirical_dynamic':empirical,'atlas_static_control':equilibrium+noise,
                      'atlas_plus_history_control':equilibrium+residual+noise}
            for arm,b in arms.items():
                result=replay_fields(b,t,disable_increment=True)
                np.testing.assert_array_equal(result['valid'],data[arm+'_valid'])
                np.testing.assert_array_equal(result['Final'][:,:2],data[arm+'_Final'][:,:2])
                data[arm+'_U']=result['Final']
                metadata['scores'][arm]['U']=metrics(result['Final'],q,t,primary,result['valid'])
            metadata['increment_ablation']='exact_frozen_engine_gain0'
            np.savez_compressed(path,**data);dump(path.with_suffix('.json'),metadata)
            checked.append(metadata['id']);print('audited exact-Z ablation',metadata['id'],flush=True)
    dump(OUT/'increment_control_verification.json',{'records':checked,'xy_bit_equal':True,
        'permission_bit_equal':True,'change':'Only in-memory incremental gain set to zero',
        'frozen_models_modified':False,'labels':LABELS})


if __name__=='__main__':main()
