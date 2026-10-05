"""Current-field constrained neural proposal, inside the final depth inverse.

First unification stage: preserve XY and output permission. The frozen R340
network supplies a proposal, not a final coordinate. Current magnetic cost,
Source training support and a field-innovation-aware branch prior adjudicate
it together. No command, nominal depth, device label or future-frame argument.
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np

from training.causal_depth_runtime_r340 import FrozenCausalDepthRefiner, FREEZE as R340_FREEZE
from training.z_evidence_runtime_r334 import depth_profiles, MagneticDepthTracker
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/'analysis/r341_unified_depth_20260929/freeze.json'


@dataclass(frozen=True)
class UnifiedDepthPolicy:
    neural_strength: float = 16.
    continuity: float = .002
    support_limit: float = 1.
    field_veto_ratio: float = 2.
    innovation_gain: float = 8.
    max_gap_s: float = .25

    def __post_init__(self):
        v = list(asdict(self).values())
        if not np.isfinite(v).all() or min(v) < 0 or self.support_limit <= 0:
            raise ValueError('Finite nonnegative policy and positive support limit required')
        if self.field_veto_ratio < 1 or self.max_gap_s <= 0:
            raise ValueError('Ordered field/gap policy required')


class SourceFeatureSupport:
    """Distances to frozen Source-only feature examples, not a contact score."""
    def __init__(self, mean, scale, reference):
        self.mean = np.asarray(mean, float); self.scale = np.asarray(scale, float)
        self.reference = np.asarray(reference, float)
        if (self.mean.shape != (56,) or self.scale.shape != (56,) or
                self.reference.ndim != 2 or self.reference.shape[1] != 56 or
                not len(self.reference) or np.any(self.scale <= 0) or
                not all(np.isfinite(a).all() for a in (self.mean, self.scale, self.reference))):
            raise ValueError('Finite Source 56D support reference required')
        self.norm = np.sum(self.reference*self.reference, axis=1)

    def normalize(self, x):
        return np.clip((np.asarray(x, float)-self.mean)/self.scale, -8., 8.)

    def distance(self, x):
        u = self.normalize(x).reshape(-1, 56); answer = np.empty(len(u))
        for start in range(0, len(u), 192):
            v = u[start:start+192]
            dist = np.sum(v*v, axis=1)[:, None]+self.norm[None, :]-2*v@self.reference.T
            answer[start:start+len(v)] = np.maximum(dist.min(axis=1), 0.)/56.
        return answer

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(a['mean'], a['scale'], a['reference'])


class UnifiedDepthAdjudicator:
    def __init__(self, policy):
        self.policy = policy; self.reset()

    def reset(self):
        self.previous_z = self.previous_phi = self.previous_t = self.previous_token = None

    def step(self, time_ns, z, cost, phi, proposal, support_distance, *, token=0):
        z, c, phi = np.asarray(z, float), np.asarray(cost, float), np.asarray(phi, float)
        if (z.ndim != 1 or c.shape != z.shape or len(z) < 3 or not np.all(np.diff(z)>0)
                or phi.shape != (16,) or not np.isfinite(np.r_[z,c,phi,proposal,support_distance]).all()
                or support_distance < 0):
            raise ValueError('Finite ordered current magnetic likelihood required')
        now = int(time_ns); dt = None if self.previous_t is None else (now-self.previous_t)/1e9
        same = self.previous_z is not None and token == self.previous_token and dt is not None and 0 < dt <= self.policy.max_gap_s
        global_id = int(np.argmin(c)); minimum = max(float(c[global_id]), 1e-8)
        allowed = c <= self.policy.field_veto_ratio*minimum
        # Curvature normalizes the proposal's influence to the CURRENT field.
        centre = z[global_id]; r = .05
        curvature = max((np.interp(min(centre+r,z[-1]),z,c)+np.interp(max(centre-r,z[0]),z,c)-2*c[global_id])/(2*r*r), 1e-5)
        support = (self.policy.support_limit/(self.policy.support_limit+support_distance))**2
        objective = c+self.policy.neural_strength*curvature*support*(z-proposal)**2
        unconstrained_id = int(np.argmin(objective))
        objective = np.where(allowed, objective, np.inf)
        instantaneous_id = int(np.argmin(objective)); selected = instantaneous_id
        if same and self.policy.continuity:
            innovation = float(np.linalg.norm(phi-self.previous_phi))
            scale = (max(dt,.025)/.1)**2*(1+self.policy.innovation_gain*innovation)**2
            selected = int(np.argmin(objective+self.policy.continuity*(z-self.previous_z)**2/scale))
        # Refine the current objective, then recheck the UNSHIFTED field cost.
        finite_objective = c+self.policy.neural_strength*curvature*support*(z-proposal)**2
        if same and self.policy.continuity:
            finite_objective = finite_objective+self.policy.continuity*(z-self.previous_z)**2/scale
        value = MagneticDepthTracker.refine(z, finite_objective, selected)
        if np.interp(value,z,c) > self.policy.field_veto_ratio*minimum:
            value = float(z[selected])
        self.previous_z, self.previous_phi = value, phi.copy()
        self.previous_t, self.previous_token = now, token
        return {'z_mm':value, 'source_support_weight':float(support),
                'source_support_distance':float(support_distance),
                'field_cost_ratio':float(np.interp(value,z,c)/minimum),
                'proposal_field_restricted':bool(not allowed[unconstrained_id]),
                'used_continuity':bool(selected != instantaneous_id),
                'proposal_z_mm':float(proposal)}


class FrozenUnifiedDepthRefiner:
    """Standalone causal final-depth layer. It cannot authorize a new XYZ."""
    def __init__(self, decoder, *, policy=None, support=None):
        self.base = FrozenCausalDepthRefiner(decoder)
        if policy is None:
            from training.analyze_z_evidence_r334 import sha
            f = json.loads(FREEZE.read_text(encoding='utf-8'))
            if f['target_fit_rows'] or f['target_selection_rows']:
                raise ValueError('Source-only final-depth policy required')
            for item in f['dependencies']:
                if sha(ROOT/item['path']) != item['sha256']:
                    raise ValueError('Frozen depth dependency changed: '+item['path'])
            policy = UnifiedDepthPolicy(**f['policy'])
            support = SourceFeatureSupport.load(ROOT/f['support_file'])
            self.manifest = f
        else:
            if support is None: raise ValueError('Support required for injected policy')
            self.manifest = None
        self.policy, self.support = policy, support
        self.final = UnifiedDepthAdjudicator(policy)
        self.history = CausalXYFeatureHistory()

    def reset(self):
        self.base.reset(); self.final.reset(); self.history.reset()

    def step(self, time_ns, field, original_xyz, memory, *, baseline_epoch=0, stream_id=0, valid=True):
        if not valid:
            self.reset(); return None
        original, h, b = np.asarray(original_xyz,float), np.asarray(memory,float), np.asarray(field,float)
        if original.shape != (3,) or h.shape != (2,) or b.shape != (15,) or not np.isfinite(np.r_[original,h,b]).all():
            raise ValueError('Finite current XYZ, two-mode memory and 15D field required')
        atlas, basis = self.base.decoder.atlas, self.base.decoder.basis
        if not atlas.z[0] <= original[2] <= atlas.z[-1]:
            self.reset()
            return {'xyz_mm':original.copy(),'depth_route':'outside_atlas_preserved','output_permission_changed':False}
        z,cost,phi = depth_profiles(atlas,b[None,:],original[None,:2],h[None,:],basis,self.base.policy)
        raw_phi = atlas.features(b[None,:])[0]
        base = self.base.depth.step_profile(time_ns,z,cost[0],phi[0],token=(baseline_epoch,stream_id))
        x53 = self.history.push(time_ns=time_ns,b0_id=baseline_epoch,stream_id=stream_id,xyz=original,phi=raw_phi,memory=h)
        x = np.r_[x53,base['z_mm']/3.,np.log1p(cost[0].min()),np.log1p(np.interp(original[2],z,cost[0]))]
        residual = float(np.clip(self.base.model.predict(x[None,:])[0,0],-1.,1.))
        proposal = float(np.clip(base['z_mm']+residual,z[0],z[-1]))
        distance = float(self.support.distance(x[None,:])[0])
        verdict = self.final.step(time_ns,z,cost[0],phi[0],proposal,distance,token=(baseline_epoch,stream_id))
        output = original.copy(); output[2] = verdict['z_mm']
        return {'xyz_mm':output,'depth_route':'unified_field_neural_r341',
                'original_z_mm':float(original[2]),'field_depth_z_mm':base['z_mm'],
                **verdict,'output_permission_changed':False}
