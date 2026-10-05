"""Had-239 / 5 mm: R125 slow magnetic memory + R138 fast branch choice.

This runtime is causal and hardware-independent. A normal local trajectory
uses the original unfiltered inverse. Only a >1 mm candidate jump triggers
the two-branch magnetic-likelihood comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.branch_aware_inverse_r138 import BranchAwareInverse
from training.had239_field_memory_depth_guard_runtime_r126 import (
    DepthGuardFieldMemoryDecoder, R125_REPORT,
)
from training.had239_field_memory_runtime_r123 import (
    DEFAULT_REPORT, FieldMemoryDecoder,
)
from training.fit_had239_atlas_causal_residual_r116 import ROOT, sha
from training.probe_had239_field_memory_r122 import feature_to_field

AUDIT_REPORT = ROOT / "analysis/r142_branch_fullrate_live_20260926/report_v2.json"
SPATIAL_PENALTY = .03
INNOVATION_GAIN = 2.0
ACTIVATION_DISTANCE_MM = 1.0


class BranchDepthMemoryDecoder(DepthGuardFieldMemoryDecoder):
    def __init__(self, atlas, basis, retention_10hz, strength):
        super().__init__(atlas, basis, retention_10hz, strength)
        self.branch = BranchAwareInverse(
            atlas, spatial_penalty=SPATIAL_PENALTY,
            innovation_gain=INNOVATION_GAIN,
            activation_distance_mm=ACTIVATION_DISTANCE_MM)

    @classmethod
    def from_report(cls, path: Path = R125_REPORT) -> "BranchDepthMemoryDecoder":
        audit = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
        if (audit.get("schema") != "MagTouch-R142-R126LiveBranchReplay-1" or
                audit.get("penalty") != SPATIAL_PENALTY or
                audit.get("innovation_gain") != INNOVATION_GAIN or
                audit.get("activation_distance_mm") != ACTIVATION_DISTANCE_MM or
                audit.get("memory_model_sha256") != sha(DEFAULT_REPORT)):
            raise ValueError("R142 selected branch settings/source changed")
        return super().from_report(path)

    def reset(self):
        super().reset()
        if hasattr(self, "branch"):
            self.branch.reset()

    def predict_frame(self, delta_b_uT: np.ndarray, *, time_ns: int,
                      b0_id: int, stream_id: int,
                      contact_active: bool) -> dict | None:
        result = FieldMemoryDecoder.predict_frame(
            self, delta_b_uT, time_ns=time_ns, b0_id=b0_id,
            stream_id=stream_id, contact_active=contact_active)
        if result is None:
            return None
        observed = np.asarray(delta_b_uT, float)[None, :]
        feature = self.atlas.features(observed) - self.strength * (
            self.state[None, :] @ self.basis.T)
        field = feature_to_field(self.atlas, feature)[0]
        branch = self.branch.predict_frame(
            field, time_ns=time_ns, b0_id=b0_id, stream_id=stream_id,
            global_xyz=result["xyz_mm"])
        xyz = np.asarray(branch["xyz_mm"], float).copy()
        base = result["base_xyz_mm"]
        xyz[2] = base[2] + self.z_weight * (xyz[2] - base[2])
        result.update({
            "xyz_mm": xyz,
            "z_guard_weight": self.z_weight,
            "branch_used_local": branch["used_local_branch"],
            "branch_global_switch_accepted": branch["global_branch_switch_accepted"],
            "branch_global_field_cost": branch["global_field_cost"],
            "branch_local_field_cost": branch["local_field_cost"],
            "branch_unfiltered_xyz_mm": branch["global_xyz_mm"].tolist(),
        })
        return result
