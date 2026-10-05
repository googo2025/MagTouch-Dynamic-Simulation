"""Independent experimental integration: R332 states + final-depth R341.

Import and construction never operate hardware. Both R332 coordinate routes
meet one final adjudicator and one output bookkeeping contract. State heads,
XY weights, permission and two private legacy decoder histories are preserved
in this FIRST migration stage, not claimed to be a jointly trained encoder.
"""
from dataclasses import dataclass
import numpy as np
from live_unified_tactile_r332 import EvidenceUnifiedTactileEngine
from training.unified_depth_runtime_r341 import FrozenUnifiedDepthRefiner


@dataclass(frozen=True)
class ReconstructionFrame:
    time_ns: int
    baseline_epoch: int
    delta_field: np.ndarray
    unaligned_xyz: np.ndarray
    memory: np.ndarray
    route: str


class UnifiedEvidenceReconstructionEngine(EvidenceUnifiedTactileEngine):
    """Same permission/XY as R332; one final layer for either permitted route."""
    def __init__(self,*args,final_refiner=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.final_refiner=final_refiner or FrozenUnifiedDepthRefiner(self.memory_decoder)
        self.final_anchor=None;self.final_was_valid=False
        self.latest_final_evidence=None

    def reset_unified(self):
        super().reset_unified()
        if hasattr(self,'final_refiner'):self.final_refiner.reset()
        self.final_anchor=None;self.final_was_valid=False
        self.latest_final_evidence=None

    def push(self,t,b,*,eligible=True):
        result=super().push(t,b,eligible=eligible)
        extra=result['extra'];old_valid=bool(result['valid'])
        extra.update(final_adjudication_version='R342-stage1-R341',
                     response_weights_unchanged=True,xy_weights_unchanged=True,
                     output_permission_unchanged=True,joint_encoder_implemented=False,
                     depth_output='current magnetic likelihood + frozen Source-only causal Z proposal; no XYZ EMA')
        self.latest_final_evidence=None
        if not old_valid:
            self.final_refiner.reset();self.final_anchor=None;self.final_was_valid=False
            result['family']='UnifiedEvidenceReconstructionR342'
            return result
        # This boundary is common to the primary and supplemental routes.
        current=ReconstructionFrame(int(t),int(self.baseline_epoch),
            np.asarray(b,float)-self.b0,np.asarray(extra['unaligned_xyz_mm'],float),
            np.asarray(extra['memory_state'],float),str(extra['localization_path']))
        evidence=self.final_refiner.step(current.time_ns,current.delta_field,current.unaligned_xyz,
            current.memory,baseline_epoch=current.baseline_epoch,stream_id=0,valid=True)
        xyz=np.asarray(result['xyz'],float).copy()
        np.testing.assert_array_equal(evidence['xyz_mm'][:2],current.unaligned_xyz[:2])
        xyz[2]=evidence['xyz_mm'][2]
        if self.final_anchor is None:self.final_anchor=xyz.copy()
        relative=xyz-self.final_anchor
        result['xyz']=xyz.tolist()
        result['raw']=dict(zip(('x','y','z','dX','dY','dZ'),[*xyz.tolist(),*relative.tolist()]))
        extra.update(xyz_mm=xyz.tolist(),unaligned_xyz_mm=evidence['xyz_mm'].tolist(),
            trace_break=bool(not self.final_was_valid),
            original_xyz_preserved=bool(np.array_equal(evidence['xyz_mm'],current.unaligned_xyz)),
            original_xy_preserved=True,original_z_mm=float(current.unaligned_xyz[2]),
            final_depth_evidence={k:v for k,v in evidence.items() if k!='xyz_mm'})
        self.latest_final_evidence=evidence;self.final_was_valid=True
        result['family']='UnifiedEvidenceReconstructionR342'
        return result
