"""Frozen Source-only incremental proposal inside current-field adjudication."""
import json
import numpy as np
from training.analyze_z_evidence_r334 import ROOT,sha
from training.causal_depth_network_r337 import DepthMLP
from training.unified_depth_runtime_r341 import FrozenUnifiedDepthRefiner
from training.fast_depth_features_r344 import FastDepthContext
from training.z_evidence_runtime_r334 import depth_profiles

FREEZE=ROOT/'analysis/r348_fast_slow_residual_20260929/freeze.json'


class FastSlowIncrementalDepthRefiner:
    def __init__(self,decoder):
        f=json.loads(FREEZE.read_text(encoding='utf-8'))
        if f['selected'] is None:raise ValueError('No Source-qualified residual')
        if any(f[k] for k in ('target_fit_rows','target_selection_rows','complex_fit_rows','complex_selection_rows')):
            raise ValueError('Source-only freeze required')
        for dep in f['dependencies']:
            if sha(ROOT/dep['path'])!=dep['sha256']:raise ValueError('Frozen dependency changed: '+dep['path'])
        self.manifest=f;self.selection=f['selected'];self.decoder=decoder
        self.model=DepthMLP.load(ROOT/self.selection['file'])
        self.anchor=FrozenUnifiedDepthRefiner(decoder)
        self.fast=FastDepthContext(decoder.atlas)

    def reset(self):
        self.anchor.reset();self.fast.reset()

    def step(self,time_ns,field,original_xyz,memory,*,baseline_epoch=0,stream_id=0,valid=True):
        if not valid:self.reset();return None
        q,h,b=np.asarray(original_xyz,float),np.asarray(memory,float),np.asarray(field,float)
        if q.shape!=(3,) or h.shape!=(2,) or b.shape!=(15,) or not np.isfinite(np.r_[q,h,b]).all():
            raise ValueError('Finite permitted XYZ, memory and current magnetic field required')
        atlas=self.decoder.atlas
        if not atlas.z[0]<=q[2]<=atlas.z[-1]:
            self.reset();return {'xyz_mm':q.copy(),'depth_route':'outside_atlas_preserved','output_permission_changed':False}
        z,c,phi=depth_profiles(atlas,b[None,:],q[None,:2],h[None,:],self.decoder.basis,self.anchor.base.policy)
        token=(int(baseline_epoch),int(stream_id))
        base=self.anchor.base.depth.step_profile(time_ns,z,c[0],phi[0],token=token)
        x53=self.anchor.history.push(time_ns=time_ns,b0_id=baseline_epoch,stream_id=stream_id,
            xyz=q,phi=atlas.features(b[None,:])[0],memory=h)
        x56=np.r_[x53,base['z_mm']/3,np.log1p(c[0].min()),np.log1p(np.interp(q[2],z,c[0]))]
        x=x56 if self.selection['dimension']==56 else self.fast.push(time_ns,x56,token=token)
        teacher=float(np.clip(self.anchor.base.model.predict(x56[None,:])[0,0],-1,1))
        teacher_proposal=float(np.clip(base['z_mm']+teacher,z[0],z[-1]))
        increment=float(np.clip(self.model.predict(x[None,:])[0,0],-.4,.4))*self.selection['gain']
        proposed=float(np.clip(teacher_proposal+increment,z[0],z[-1]))
        distance=float(self.anchor.support.distance(x56[None,:])[0])
        evidence=self.anchor.final.step(time_ns,z,c[0],phi[0],proposed,distance,token=token)
        result=q.copy();result[2]=evidence['z_mm']
        return {'xyz_mm':result,'depth_route':'frozen_absolute_incremental_r348',
            'teacher_proposal_z_mm':teacher_proposal,'increment_z_mm':increment,
            'field_depth_z_mm':base['z_mm'],'output_permission_changed':False,**evidence}
