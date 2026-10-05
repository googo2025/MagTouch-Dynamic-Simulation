"""Opt-in frozen Source Z refiner. No device, command or trajectory inputs.

Caller supplies an already permitted unaligned XYZ/memory observation. This
layer never turns a rejected/unknown state into an XYZ output. UI is unchanged.
"""
import json
import numpy as np
from training.analyze_z_evidence_r334 import ROOT,sha,DepthPolicy
from training.z_evidence_runtime_r334 import depth_profiles,MagneticDepthTracker
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory
from training.causal_depth_network_r337 import DepthMLP

FREEZE=ROOT/'analysis/r340_depth_static_anchor_20260929/freeze.json'


class FrozenCausalDepthRefiner:
    def __init__(self,decoder):
        self.freeze=json.loads(FREEZE.read_text(encoding='utf-8'))
        if self.freeze['target_fit_rows'] or self.freeze['target_selection_rows']:
            raise ValueError('Source-only depth artifact required')
        if self.freeze['model_sha256']!=sha(ROOT/self.freeze['selected_file']):
            raise ValueError('Frozen depth model changed')
        if self.freeze['runtime_sha256']!=sha(ROOT/'training/causal_depth_network_r337.py'):
            raise ValueError('Frozen feature/inference implementation changed')
        self.decoder=decoder;self.model=DepthMLP.load(ROOT/self.freeze['selected_file'])
        self.policy=DepthPolicy(**self.freeze['depth_policy'])
        self.depth=MagneticDepthTracker(self.policy);self.history=CausalXYFeatureHistory()

    def reset(self):
        self.depth.reset();self.history.reset()

    def step(self,time_ns,field,original_xyz,memory,*,baseline_epoch=0,stream_id=0,valid=True):
        if not valid:
            self.reset();return None
        original=np.asarray(original_xyz,float);state=np.asarray(memory,float)
        field=np.asarray(field,float)
        if original.shape!=(3,) or state.shape!=(2,) or field.shape!=(15,):
            raise ValueError('Current 15D field, unaligned XYZ and two-mode memory required')
        z,cost,phi=depth_profiles(self.decoder.atlas,field[None,:],original[None,:2],state[None,:],self.decoder.basis,self.policy)
        base=self.depth.step_profile(time_ns,z,cost[0],phi[0],token=(int(baseline_epoch),int(stream_id)))
        raw=self.decoder.atlas.features(field[None,:])[0]
        causal=self.history.push(time_ns=time_ns,b0_id=baseline_epoch,stream_id=stream_id,
                                 xyz=original,phi=raw,memory=state)
        x=np.r_[causal,base['z_mm']/3.,np.log1p(cost[0].min()),np.log1p(np.interp(original[2],z,cost[0]))]
        residual=float(np.clip(self.model.predict(x[None,:])[0,0],-1.,1.))
        proposed=float(np.clip(base['z_mm']+residual,z[0],z[-1]))
        veto=bool(np.interp(proposed,z,cost[0])>2.*max(float(cost[0].min()),1e-8))
        output=original.copy();output[2]=base['z_mm'] if veto else proposed
        return {'xyz_mm':output,'original_z_mm':float(original[2]),'field_depth_z_mm':base['z_mm'],
                'neural_z_residual_mm':0. if veto else residual,'neural_field_veto':veto,
                'source_only_frozen':True,'output_permission_changed':False}
