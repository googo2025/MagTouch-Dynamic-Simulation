"""Offline-only R89: R86 with sensor-only motion weighting of XY residual."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from causal_old1mm_xyz_r86 import R86OfflinePredictor


class R89OfflinePredictor(R86OfflinePredictor):
    def __init__(self, path: Path, key: str):
        super().__init__(path, key)
        settings = self.profile.get("r89_motion_weight")
        if not isinstance(settings, dict):
            raise ValueError("not an R89 motion-guard profile")
        model = settings["model"]
        self.motion_mean = np.asarray(model["mean"], float)
        self.motion_scale = np.asarray(model["scale"], float)
        self.motion_coef = np.asarray(model["coef"], float)
        self.xy_divisor = float(settings["xy_score_divisor"])
        if (self.motion_mean.shape != (9,) or self.motion_scale.shape != (9,) or
                self.motion_coef.shape != (10,) or not 0. < self.xy_divisor <= 1.):
            raise ValueError("invalid R89 motion model")
        if (not all(np.isfinite(v).all() for v in (
                self.motion_mean, self.motion_scale, self.motion_coef)) or
                np.any(self.motion_scale <= 0)):
            raise ValueError("non-finite R89 motion model")

    def _xy_scale(self, delta, amplitude):
        features = []
        for lag in (4, 10, 20):
            change = delta - self.delta_history[-1 - lag]
            energy = float(np.linalg.norm(change))
            features.extend((energy, energy / max(amplitude, 20.),
                             float(np.max(np.abs(change)))))
        standard = (np.asarray(features) - self.motion_mean) / self.motion_scale
        score = float(np.r_[1., standard] @ self.motion_coef)
        return float(np.clip(score / self.xy_divisor, 0., 1.))
