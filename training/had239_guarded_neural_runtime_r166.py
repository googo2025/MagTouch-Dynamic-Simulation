"""Had-239/5 mm causal decoder: magnetic-evidence branch guard and optional XY MLP."""
from __future__ import annotations

import numpy as np

from training.evidence_guarded_branch_r162 import EvidenceGuardedBranchInverse
from training.had239_branch_memory_runtime_r144 import (
    ACTIVATION_DISTANCE_MM, INNOVATION_GAIN, SPATIAL_PENALTY,
    BranchDepthMemoryDecoder,
)
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory
from training.had239_causal_xy_mlp_runtime_r165 import CausalXYMLP
from training.probe_had239_field_memory_r122 import feature_to_field

MAX_LOCAL_COST_RATIO = 4.0


class GuardedBranchMemoryDecoder(BranchDepthMemoryDecoder):
    def __init__(self, atlas, basis, retention_10hz, strength):
        super().__init__(atlas, basis, retention_10hz, strength)
        self.branch = EvidenceGuardedBranchInverse(
            atlas, spatial_penalty=SPATIAL_PENALTY,
            innovation_gain=INNOVATION_GAIN,
            activation_distance_mm=ACTIVATION_DISTANCE_MM,
            max_local_cost_ratio=MAX_LOCAL_COST_RATIO)

    def predict_frame(self, *args, **kwargs):
        row = super().predict_frame(*args, **kwargs)
        if row is not None:
            row["branch_local_vetoed_by_field"] = self.branch.last_veto
            row["branch_max_local_cost_ratio"] = MAX_LOCAL_COST_RATIO
        return row


class GuardedNeuralMemoryDecoder(GuardedBranchMemoryDecoder):
    def __init__(self, atlas, basis, retention_10hz, strength):
        super().__init__(atlas, basis, retention_10hz, strength)
        self.xy_features = CausalXYFeatureHistory()
        self.xy_model = CausalXYMLP()

    def reset(self):
        super().reset()
        if hasattr(self, "xy_features"):
            self.xy_features.reset()

    def predict_frame(self, delta_b_uT, *, time_ns, b0_id, stream_id,
                      contact_active):
        row = super().predict_frame(
            delta_b_uT, time_ns=time_ns, b0_id=b0_id,
            stream_id=stream_id, contact_active=contact_active)
        if row is None:
            self.xy_features.reset()
            return None
        observed = np.asarray(delta_b_uT, float)[None, :]
        corrected_phi = self.atlas.features(observed)[0] - (
            self.strength * self.basis @ self.state)
        corrected_phi = self.atlas.features(feature_to_field(
            self.atlas, corrected_phi[None, :]))[0]
        causal = self.xy_features.push(
            time_ns=time_ns, b0_id=b0_id, stream_id=stream_id,
            xyz=row["xyz_mm"], phi=corrected_phi,
            memory=row["memory_state"])
        correction = self.xy_model.predict(causal[None, :])[0]
        estimate = np.asarray(row["xyz_mm"], float).copy()
        estimate[:2] += correction
        row["xyz_mm"] = estimate
        row["xy_neural_correction_mm"] = correction.tolist()
        return row
