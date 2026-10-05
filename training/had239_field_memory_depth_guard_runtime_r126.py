"""Frozen R125 Z guard on the R123 causal magnetic-memory runtime."""
from __future__ import annotations

import json
from pathlib import Path

from training.had239_field_memory_runtime_r123 import (
    DEFAULT_REPORT, FieldMemoryDecoder,
)
from training.fit_had239_atlas_causal_residual_r116 import ROOT, sha

R125_REPORT = ROOT / "analysis/r125_had239_field_memory_depth_guard_20260926/report.json"


class DepthGuardFieldMemoryDecoder(FieldMemoryDecoder):
    z_weight = 0.0

    @classmethod
    def from_report(cls, path: Path = R125_REPORT) -> "DepthGuardFieldMemoryDecoder":
        guard = json.loads(path.read_text(encoding="utf-8"))
        if guard.get("schema") != "MagTouch-R125-DepthGuard-1":
            raise ValueError("unexpected R125 depth-guard artifact")
        if guard["r122_sha256"] != sha(DEFAULT_REPORT):
            raise ValueError("R122 source model changed after R125 validation")
        decoder = super().from_report(DEFAULT_REPORT)
        decoder.z_weight = float(guard["selected_z_weight"])
        if not 0 <= decoder.z_weight <= 1:
            raise ValueError("invalid R125 Z weight")
        return decoder

    def predict_frame(self, *args, **kwargs) -> dict | None:
        result = super().predict_frame(*args, **kwargs)
        if result is None:
            return None
        estimate = result["xyz_mm"].copy()
        base = result["base_xyz_mm"]
        estimate[2] = base[2] + self.z_weight * (estimate[2] - base[2])
        result["xyz_mm"] = estimate
        result["z_guard_weight"] = self.z_weight
        return result
