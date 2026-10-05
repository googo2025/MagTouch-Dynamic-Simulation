"""R91 live adapter for the explicitly selected, experimental R89 profile.

It only consumes grouped magnetic frames supplied by the existing UI. No
acquisition, stage or camera control is added here.
"""
from __future__ import annotations

from pathlib import Path

from causal_old1mm_xyz_r89 import R89OfflinePredictor
from live_normalized_fit import LiveNormalizedFitPredictor
from replay_backtest import JUMP_FAST


class _CausalEngineProxy:
    """Preserve the existing fit UI API while routing push to R89's gate."""

    def __init__(self, predictor):
        object.__setattr__(self, "_predictor", predictor)

    def __getattr__(self, name):
        return getattr(self._predictor.engine, name)

    def __setattr__(self, name, value):
        if name == "_predictor":
            object.__setattr__(self, name, value)
        else:
            setattr(self._predictor.engine, name, value)

    def set_baseline(self, baseline):
        return self._predictor.set_baseline(baseline)

    def push(self, t_ns, fields):
        return self._predictor.push(t_ns, fields)


class LiveR89Predictor(LiveNormalizedFitPredictor):
    """Use the ordinary UI B0 lifecycle with R89's complete causal decoder."""

    def __init__(self, profile, path: Path, preset: str, *, key: str,
                 threshold, contact_threshold, min_cosine, smoothing_ms,
                 jump_policy=JUMP_FAST):
        if not isinstance(profile.get("r89_motion_weight"), dict):
            raise ValueError("R89 motion parameters missing")
        defaults = profile["defaults"]
        requested = (float(threshold), float(contact_threshold),
                     float(min_cosine), float(smoothing_ms))
        fixed = tuple(float(defaults[name]) for name in (
            "threshold", "contact_threshold", "min_cosine", "smoothing_ms"))
        if any(abs(a - b) > 1e-9 for a, b in zip(requested, fixed)) or jump_policy != JUMP_FAST:
            raise ValueError("R89 requires its frozen contact settings; reload the preset defaults")
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        self.causal = R89OfflinePredictor(path, key)
        self.engine = _CausalEngineProxy(self.causal)
        self.metadata.update({
            "algorithm": "normalized_fit_r89_experimental",
            "deployment_qualified": False,
            "contact_gate": "R75 causal RFF + two-frame onset/release",
            "dynamic_residual": "R89 grid-constrained causal XYZ with sensor-only XY weight",
            "validation_warning": "Development retrospective only; no fresh long-idle/circle/XYZ test",
        })
        self.model_summary = (f"{preset} | 因果接触门控＋动态XYZ（未独立验证） | "
                              "已评估压深1.2–2.0 mm；先观察空载误报")
        self.begin_baseline()

    def begin_baseline(self, seconds=1.0):
        if hasattr(self, "causal"):
            self.causal.b0 = None
            self.causal.history.clear()
            self.causal.last_t = None
            self.causal.active = False
            self.causal.up = self.causal.down = 0
            self.causal.delta_history.clear()
            self.causal.corrected_anchor = None
        super().begin_baseline(seconds)
