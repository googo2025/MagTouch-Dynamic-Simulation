"""Offline-only causal R73 gate wrapped around the R65 static XYZ decoder.

This is intentionally not registered in the live UI. Its JSON payload has
deployment_validation.passed=false until a fresh independent idle/motion run.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np

from replay_backtest import JUMP_FAST, NormalizedFit, empty_record, load_profile


class R73OfflinePredictor:
    def __init__(self, path: Path, key: str):
        self.path = Path(path)
        self.profile = load_profile(self.path, key)
        gate = self.profile.get("r73_gate")
        if not isinstance(gate, dict):
            raise ValueError("not an R73 gated profile")
        self.rff = gate["rff"]
        self.rff_mean = np.asarray(self.rff["mean"], float)
        self.rff_scale = np.asarray(self.rff["scale"], float)
        self.rff_weights = np.asarray(self.rff["weights"], float)
        self.rff_phase = np.asarray(self.rff["phase"], float)
        self.rff_coef = np.asarray(self.rff["coef"], float)
        width = int(self.rff["width"])
        if (self.rff_mean.shape != (32,) or self.rff_scale.shape != (32,) or
                self.rff_weights.shape != (32, width) or
                self.rff_phase.shape != (width,) or
                self.rff_coef.shape != (width + 1,)):
            raise ValueError("invalid R73 RFF dimensions")
        if not all(np.isfinite(v).all() for v in (
                self.rff_mean, self.rff_scale, self.rff_weights,
                self.rff_phase, self.rff_coef)):
            raise ValueError("R73 RFF contains non-finite parameters")
        self.onset_amp = float(gate["onset_amplitude_uT"])
        self.onset_frames = int(gate["onset_confirm_frames"])
        self.off_amp = float(gate["continuation_release_amp_uT"])
        self.off_score = float(gate["continuation_release_score_below"])
        self.off_frames = int(gate["release_confirm_frames"])
        self.min_onset_score = float(gate["minimum_onset_score"])
        self.hard_release_amp = float(gate["hard_release_amp_uT"])
        defaults = self.profile["defaults"]
        self.engine = NormalizedFit(
            self.profile, Path("__r73_offline__"),
            defaults["threshold"], defaults["min_cosine"],
            defaults["smoothing_ms"], defaults["contact_threshold"], JUMP_FAST)
        self.b0 = None
        self.history = deque()
        self.last_t = None
        self.active = False
        self.up = self.down = 0

    def set_baseline(self, baseline):
        value = np.asarray(baseline, float).reshape(-1)
        if value.shape != (15,) or not np.isfinite(value).all():
            raise ValueError("invalid R73 B0")
        self.b0 = value.copy()
        self.engine.set_baseline(value)
        self.history.clear()
        self.last_t = None
        self.active = False
        self.up = self.down = 0

    def _score(self, delta):
        amplitude = max(float(np.linalg.norm(delta)), 1e-6)
        unit = delta / amplitude
        loga = math.log(amplitude)
        feature = np.r_[unit, loga, loga * loga, unit * loga]
        standard = (feature - self.rff_mean) / self.rff_scale
        phi = np.cos(standard @ self.rff_weights + self.rff_phase) * math.sqrt(
            2.0 / int(self.rff["width"]))
        return float(np.r_[phi, 1.] @ self.rff_coef)

    def push(self, t_ns, fields):
        t = int(t_ns)
        if self.b0 is None:
            return empty_record(t, "R73 等待 B0")
        value = np.asarray(fields, float).reshape(-1)
        if value.shape != (15,) or not np.isfinite(value).all():
            self.active = False
            self.up = self.down = 0
            self.history.clear()
            self.engine.anchor = None
            return empty_record(t, "R73 无效磁场帧")
        if self.last_t is not None and (t <= self.last_t or t - self.last_t > 250_000_000):
            self.active = False
            self.up = self.down = 0
            self.history.clear()
            self.engine.buffer.clear()
            self.engine.anchor = None
        self.last_t = t
        self.history.append((t, value.copy()))
        while len(self.history) > 1 and self.history[0][0] < t - 100_000_000:
            self.history.popleft()
        smoothed = np.median([row[1] for row in self.history], axis=0) - self.b0
        amplitude = float(np.linalg.norm(smoothed))
        contact_score = self._score(smoothed)
        if self.active:
            release = ((amplitude < self.off_amp and contact_score < self.off_score)
                       or amplitude < self.hard_release_amp)
            self.down = self.down + 1 if release else 0
            if self.down >= self.off_frames:
                self.active = False
                self.up = self.down = 0
        else:
            trigger = amplitude >= self.onset_amp and contact_score >= self.min_onset_score
            self.up = self.up + 1 if trigger else 0
            if self.up >= self.onset_frames:
                self.active = True
                self.down = 0
        original = self.engine.push(t, value)
        detail = {"r73_gate_active": self.active, "r73_gate_amplitude_uT": amplitude,
                  "r73_gate_score": contact_score, "r73_gate_onset_count": self.up,
                  "r73_gate_release_count": self.down,
                  "deployment_qualified": False}
        if not self.active:
            self.engine.anchor = None
            self.engine.contact_active = False
            self.engine.peak_amp = 0.
            return empty_record(t, "R73 判为非接触", detail)
        original["extra"].update(detail)
        return original
