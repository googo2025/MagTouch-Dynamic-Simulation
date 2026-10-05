"""Experimental conditional depth inverse; no commands, devices or XYZ EMA.

The caller still owns XY, response recognition, B0, and output permission.
Only depth is re-estimated from a current magnetic likelihood curve. A soft
branch prior is vetoed when the current field strongly favours another depth.
No released/unlocalized point is held or manufactured by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class DepthPolicy:
    memory_strength: float = 0.
    position_source: str = "guard"
    continuity: float = 0.01
    field_veto_ratio: float = 2.
    innovation_gain: float = 8.
    max_gap_s: float = .25
    grid_step_mm: float = .025

    def __post_init__(self):
        if self.position_source not in ("guard", "final"):
            raise ValueError("Known position source required")
        values = (self.memory_strength, self.continuity, self.field_veto_ratio,
                  self.innovation_gain, self.max_gap_s, self.grid_step_mm)
        if not np.isfinite(values).all() or not 0 <= self.memory_strength <= 1:
            raise ValueError("Finite policy required")
        if (self.continuity < 0 or self.field_veto_ratio < 1 or
                self.innovation_gain < 0 or self.max_gap_s <= 0 or self.grid_step_mm <= 0):
            raise ValueError("Ordered nonnegative policy required")


def depth_profiles(atlas, field, xy, memory, basis, policy):
    """Batch current-frame likelihoods, numerically identical to frame calls.

    Memory is supplied by the already causal frozen field-memory decoder.
    XYZ labels and future fields are not accepted by this function.
    """
    field, xy = np.asarray(field, float), np.asarray(xy, float)
    memory = np.asarray(memory, float)
    if field.ndim != 2 or field.shape[1] != 15 or xy.shape != (len(field), 2):
        raise ValueError("N-by-15 field and N-by-2 XY required")
    if memory.shape != (len(field), 2) or np.asarray(basis).shape != (16, 2):
        raise ValueError("Frozen two-mode memory required")
    if not all(np.isfinite(a).all() for a in (field, xy, memory, basis)):
        raise ValueError("Finite likelihood inputs required")
    z = np.linspace(atlas.z[0], atlas.z[-1],
                    round((atlas.z[-1]-atlas.z[0])/policy.grid_step_mm)+1)
    phi = atlas.features(field)-policy.memory_strength*memory@np.asarray(basis).T
    costs = np.empty((len(field), len(z)))
    for start in range(0, len(field), 192):
        end = min(start+192, len(field))
        points = np.empty((end-start, len(z), 3))
        points[:, :, :2] = xy[start:end, None, :]
        points[:, :, 2] = z[None, :]
        predicted = atlas.features(atlas.forward(points.reshape(-1, 3))).reshape(end-start, len(z), 16)
        costs[start:end] = np.sum((phi[start:end, None, :]-predicted)**2, axis=2)
    return z, costs, phi


class MagneticDepthTracker:
    def __init__(self, policy=DepthPolicy()):
        self.policy = policy
        self.reset()

    def reset(self):
        self.previous_z = None
        self.previous_phi = None
        self.previous_t = None
        self.previous_token = None

    @staticmethod
    def refine(z, cost, index):
        """Parabolic refinement of the current likelihood, not temporal mixing."""
        if index <= 0 or index >= len(z)-1:
            return float(z[index])
        a, b, c = cost[index-1:index+2]
        curvature = a-2*b+c
        shift = np.clip(.5*(a-c)/max(curvature, 1e-15), -.5, .5) if curvature > 0 else 0.
        return float(z[index]+shift*(z[index+1]-z[index]))

    def step_profile(self, time_ns, z, cost, phi, *, token=0, valid=True):
        if not valid:
            self.reset()
            return None
        z, cost, phi = np.asarray(z, float), np.asarray(cost, float), np.asarray(phi, float)
        if (z.ndim != 1 or cost.shape != z.shape or len(z) < 3 or
                not all(np.isfinite(a).all() for a in (z, cost, phi)) or
                not np.all(np.diff(z) > 0)):
            raise ValueError("Finite ordered current-field profile required")
        now = int(time_ns)
        dt = None if self.previous_t is None else (now-self.previous_t)/1e9
        same = (self.previous_z is not None and self.previous_token == token and
                dt is not None and 0 < dt <= self.policy.max_gap_s)
        global_id = int(np.argmin(cost)); chosen = global_id
        veto = False
        if same and self.policy.continuity:
            innovation = np.linalg.norm(phi-self.previous_phi)
            # Field innovation weakens the prior during real fast changes.
            scale = (max(dt, .025)/.1)**2*(1+self.policy.innovation_gain*innovation)**2
            objective = cost+self.policy.continuity*(z-self.previous_z)**2/scale
            chosen = int(np.argmin(objective))
            veto = bool(cost[chosen] > self.policy.field_veto_ratio*max(cost[global_id], 1e-8))
            if veto:
                chosen = global_id
        value = self.refine(z, cost, chosen)
        self.previous_z, self.previous_phi = value, phi.copy()
        self.previous_t, self.previous_token = now, token
        return {"z_mm": value, "global_z_mm": self.refine(z, cost, global_id),
                "global_cost": float(cost[global_id]), "selected_cost": float(cost[chosen]),
                "used_continuity": bool(chosen != global_id), "field_veto": veto}

    def step(self, time_ns, field, xy, memory, atlas, basis, *, token=0, valid=True):
        if not valid:
            self.reset()
            return None
        z, costs, phi = depth_profiles(atlas, np.asarray(field).reshape(1, 15),
            np.asarray(xy).reshape(1, 2), np.asarray(memory).reshape(1, 2), basis, self.policy)
        return self.step_profile(time_ns, z, costs[0], phi[0], token=token)

    def metadata(self):
        return {"policy": asdict(self.policy), "command_inputs": False,
                "xyz_ema": False, "output_permission_changed": False,
                "shallow_depth_extrapolated": False, "experimental": True}


class DepthEvidenceDecoder:
    """Opt-in adapter. The original decoder is not patched or mutated."""
    def __init__(self, decoder, policy):
        self.decoder = decoder
        self.atlas, self.basis = decoder.atlas, decoder.basis
        self.strength = decoder.strength
        self.policy = policy
        self.depth = MagneticDepthTracker(policy)

    def reset(self):
        self.decoder.reset()
        self.depth.reset()

    def predict_frame(self, delta_b_uT, *, time_ns, b0_id, stream_id, contact_active):
        row = self.decoder.predict_frame(delta_b_uT, time_ns=time_ns, b0_id=b0_id,
            stream_id=stream_id, contact_active=contact_active)
        if row is None:
            self.depth.reset()
            return None
        old = np.asarray(row["xyz_mm"], float)
        xy = old[:2].copy()
        if self.policy.position_source == "guard":
            xy -= np.asarray(row.get("xy_neural_correction_mm", [0., 0.]))
        result = self.depth.step(time_ns, delta_b_uT, xy, row["memory_state"],
            self.atlas, self.basis, token=(int(b0_id), int(stream_id)))
        new = old.copy()
        new[2] = result["z_mm"]
        return {**row, "xyz_mm": new, "original_z_mm": float(old[2]),
                "depth_evidence_r334": result}
