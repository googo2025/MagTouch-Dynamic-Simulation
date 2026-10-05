"""Offline-only streaming R86: R75 contact gate + grid-constrained XYZ residual."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from causal_old1mm_gate_r73 import R73OfflinePredictor


class R86OfflinePredictor(R73OfflinePredictor):
    def __init__(self, path: Path, key: str):
        super().__init__(path, key)
        settings = self.profile.get("r86_joint_residual")
        if not isinstance(settings, dict):
            raise ValueError("not an R86 grid-constrained profile")
        self.mean = np.asarray(settings["model"]["mean"], float)
        self.scale = np.asarray(settings["model"]["scale"], float)
        self.coef = np.asarray(settings["model"]["coef"], float)
        self.bound = float(settings["correction_bound_mm"])
        self.fade_full = float(settings["fade_full_below_mm"])
        self.fade_zero = float(settings["fade_zero_above_mm"])
        if (self.mean.shape != (64,) or self.scale.shape != (64,) or
                self.coef.shape != (65, 3) or
                list(settings["history_lags_frames"]) != [4, 10, 20] or
                not 0. < self.bound <= 5. or not self.fade_full < self.fade_zero):
            raise ValueError("invalid R86 residual dimensions/settings")
        if (not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.coef)) or
                np.any(self.scale <= 0)):
            raise ValueError("non-finite R86 model")
        self.delta_history = deque(maxlen=21)
        self.corrected_anchor = None

    def set_baseline(self, baseline):
        super().set_baseline(baseline)
        self.delta_history.clear()
        self.corrected_anchor = None

    def _xy_scale(self, delta, amplitude):
        """Extension point for a later sensor-only XY motion guard."""
        return 1.

    def push(self, t_ns, fields):
        t = int(t_ns)
        previous_t = self.last_t
        value = np.asarray(fields, float).reshape(-1)
        reset = (previous_t is not None and
                 (t <= previous_t or t - previous_t > 250_000_000))
        if reset or value.shape != (15,) or not np.isfinite(value).all():
            self.delta_history.clear()
            self.corrected_anchor = None
        row = super().push(t, fields)
        if self.b0 is None or value.shape != (15,) or not np.isfinite(value).all():
            return row
        delta = np.median([item[1] for item in self.history], axis=0) - self.b0
        self.delta_history.append(delta)
        if not row["valid"]:
            self.corrected_anchor = None
            return row
        if len(self.delta_history) < 21:
            row["extra"]["r86_residual_ready"] = False
            self.corrected_anchor = None
            return row
        base = np.asarray(row["xyz"], float)
        # Training used the sensor-centred model output. Five-point live
        # alignment is an output transform; never feed aligned XY back into
        # the residual features. Rotate the learned XY correction only after
        # the residual is evaluated in its original coordinate frame.
        feature_xyz = np.asarray(row["extra"].get("unaligned_xyz_mm", base), float)
        amp = float(np.linalg.norm(delta))
        parts = [feature_xyz, delta / max(amp, 1e-6), np.array([np.log1p(amp)])]
        for lag in (4, 10, 20):
            parts.append((delta - self.delta_history[-1 - lag]) / 100.)
        feature = np.concatenate(parts)
        correction = np.clip(np.r_[1., (feature - self.mean) / self.scale] @ self.coef,
                             -self.bound, self.bound)
        depth_weight = float(np.clip((self.fade_zero - feature_xyz[2]) /
                                     (self.fade_zero - self.fade_full), 0., 1.))
        weighted = correction * depth_weight
        xy_scale = float(self._xy_scale(delta, amp))
        if not np.isfinite(xy_scale) or not 0. <= xy_scale <= 1.:
            raise ValueError("invalid causal XY correction weight")
        weighted[:2] *= xy_scale
        weighted[:2] = self.engine.xy_rotation @ weighted[:2]
        xyz = base + weighted
        if not np.isfinite(xyz).all() or xyz[2] < 0:
            self.corrected_anchor = None
            row["extra"]["r86_residual_ready"] = False
            return row
        if self.corrected_anchor is None:
            self.corrected_anchor = xyz.copy()
        relative = xyz - self.corrected_anchor
        row["xyz"] = xyz.tolist()
        row["raw"].update({key: float(value) for key, value in zip(
            ("x", "y", "z", "dX", "dY", "dZ"), (*xyz, *relative))})
        row["extra"].update({"r86_residual_ready": True,
                             "r86_depth_weight": depth_weight,
                             "r86_xy_scale": xy_scale,
                             "r86_base_xyz_mm": base.tolist(),
                             "r86_correction_xyz_mm": weighted.tolist(),
                             "deployment_qualified": False})
        return row
