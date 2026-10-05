#!/usr/bin/env python3
"""Hardware-independent causal runtime for the R122 had-239/5 mm decoder.

The caller owns B0 subtraction and contact detection.  In particular, a
contact decision must not be inferred from the XYZ estimate.  This module
does not import the acquisition UI, stage, serial port, or camera.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.fit_had239_atlas_causal_residual_r116 import (
    CONFIG, PROFILE_KEY, ROOT, ForwardAtlas, load_profile, samples, sha,
)
from training.probe_had239_field_memory_r122 import (
    RANK, corrected, observed_memory,
)

DEFAULT_REPORT = ROOT / "analysis/r122_had239_field_memory_20260926/report.json"
TRAIN_STEP_S = 0.1
MAX_GAP_NS = 250_000_000


class FieldMemoryDecoder:
    """One observation at a time; no motion command or future frame needed."""

    def __init__(self, atlas: ForwardAtlas, basis: np.ndarray,
                 retention_10hz: float, strength: float):
        self.atlas = atlas
        self.basis = np.asarray(basis, float)
        if self.basis.shape != (16, RANK):
            raise ValueError("expected 16-by-2 magnetic memory basis")
        if not (0 <= retention_10hz < 1 and 0 <= strength <= 1):
            raise ValueError("invalid memory coefficients")
        self.retention_10hz = retention_10hz
        self.strength = strength
        self.reset()

    @classmethod
    def from_report(cls, path: Path = DEFAULT_REPORT) -> "FieldMemoryDecoder":
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != "MagTouch-R122-LowRankFieldMemory-1":
            raise ValueError("unexpected magnetic memory artifact")
        if sha(CONFIG) != report["sources"]["static_config_sha256"]:
            raise ValueError("static atlas file changed after R122 fitting")
        profile = load_profile(CONFIG, PROFILE_KEY)
        xyz, field = samples(profile)
        atlas = ForwardAtlas(xyz, field, "unit_logamp")
        selected = report["selection"]["selected"]
        return cls(atlas, np.asarray(report["basis"], float),
                   float(selected["retention"]), float(selected["strength"]))

    def reset(self) -> None:
        self.state = np.zeros(RANK)
        self.previous_time_ns: int | None = None
        self.previous_b0_id: int | None = None
        self.previous_stream_id: int | None = None

    def predict_frame(self, delta_b_uT: np.ndarray, *, time_ns: int,
                      b0_id: int, stream_id: int,
                      contact_active: bool) -> dict | None:
        """Predict one frame or return None while the external gate is open.

        Only the current 15D B0-subtracted field and preceding state are used.
        A stop in stage motion does not reset the state.  Contact release does.
        For a changed sample rate, exponential retention preserves the fitted
        ~0.45 s memory time constant rather than treating 40 Hz as 10 Hz.
        """
        if not contact_active:
            self.reset()
            return None
        observed = np.asarray(delta_b_uT, float)
        if observed.shape != (15,) or not np.isfinite(observed).all():
            raise ValueError("delta_b_uT must contain 15 finite channels")
        now = int(time_ns)
        gap = None if self.previous_time_ns is None else now - self.previous_time_ns
        same = (gap is not None and 0 < gap <= MAX_GAP_NS and
                b0_id == self.previous_b0_id and
                stream_id == self.previous_stream_id)
        base = self.atlas.predict(observed[None, :])
        phi = self.atlas.features(observed[None, :])
        feature = np.column_stack((base, phi))
        h_obs = observed_memory(self.atlas, self.basis, feature, base)[0]
        rho = (self.retention_10hz ** (gap / 1e9 / TRAIN_STEP_S)
               if same else self.retention_10hz)
        self.state = (rho * self.state if same else np.zeros(RANK)) + (
            1 - rho) * h_obs
        row = {"feature": feature}
        estimate = corrected(self.atlas, row, self.state[None, :],
                             self.basis, self.strength)[0]
        self.previous_time_ns = now
        self.previous_b0_id = int(b0_id)
        self.previous_stream_id = int(stream_id)
        return {"xyz_mm": estimate, "base_xyz_mm": base[0],
                "memory_state": self.state.copy()}
