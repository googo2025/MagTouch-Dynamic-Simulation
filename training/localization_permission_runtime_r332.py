"""Streaming frozen Source-only fallback localization evidence.

This is permission for a magnetic XYZ candidate, not physical-contact truth.
There is no command/depth/device-label argument. It never holds an old point.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from training.unified_response_runtime_r330 import (
    ROOT, FrozenMLP, MagneticFeatureHistory, CausalConfirmation, digest)
from training.probe_localization_evidence_r331 import geometric_features

FREEZE = ROOT/"analysis/r332_source_grid_permission_20260929/freeze.json"


def frozen_permission():
    value = json.loads(FREEZE.read_text(encoding="utf-8"))
    if not value["selected"]:
        raise ValueError("No Source-qualified localization permission")
    for key in ("target_fit_rows", "target_selection_rows", "source_test_selection_rows"):
        if value[key] != 0:
            raise ValueError("Source-only permission model required")
    for item in value["tracked"]:
        if digest(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Frozen permission dependency changed: "+item["path"])
    return value, {"path": FREEZE.relative_to(ROOT).as_posix(), "sha256": digest(FREEZE)}


class StreamingLocalizationPermission:
    def __init__(self, decoder, *, model=None, selected=None):
        self.decoder = decoder
        if model is None:
            value, self.manifest = frozen_permission()
            self.model = FrozenMLP(ROOT/value["model"])
            self.selected = value["selected"]
        else:
            # Dependency injection for deterministic, hardware-free tests only.
            if selected is None:
                raise ValueError("Explicit policy required with injected test model")
            self.model, self.selected, self.manifest = model, selected, None
        self.history = MagneticFeatureHistory()
        self.confirm = CausalConfirmation(self.selected["confirmation_s"])
        self.reset()

    def reset(self):
        self.history.reset(); self.confirm.reset()

    def step(self, time_ns, delta_b_uT, observation, candidate=None):
        b = np.asarray(delta_b_uT, float)
        finite = b.shape == (15,) and np.isfinite(b).all()
        eligible = bool(observation.get("eligible", False) and finite)
        features = self.history.push(time_ns, b if finite else np.zeros(15), eligible)
        xyz = None if candidate is None else np.asarray(candidate["xyz_mm"], float)
        finite_xyz = xyz is not None and xyz.shape == (3,) and np.isfinite(xyz).all()
        strength, phase = int(observation.get("strength", 0)), int(observation.get("phase", 0))
        if strength not in range(7) or phase not in range(6):
            raise ValueError("Known magnetic response states required")
        score, feature = None, None
        if finite_xyz:
            geometry = geometric_features(self.decoder.atlas, b.reshape(1, 15), xyz.reshape(1, 3),
                np.asarray(candidate["memory_state"]).reshape(1, 2),
                np.asarray(candidate["xy_neural_correction_mm"]).reshape(1, 2),
                self.decoder.basis, self.decoder.strength)[0]
            feature = np.r_[features["relative"], geometry, np.eye(7)[strength], np.eye(6)[phase]]
            score = float(self.model.predict(feature.reshape(1, 61))[0, 1])
        good = bool(eligible and observation.get("response", False) and strength not in (1, 5)
                    and finite_xyz and np.all(np.abs(xyz[:2]) <= 10.) and 0 <= xyz[2] <= 3.
                    and score is not None and score >= self.selected["score_limit"])
        permitted = good if self.selected["confirmation_s"] == 0 else self.confirm.step(int(time_ns), good)
        return {"permitted": bool(permitted), "score": score, "features": feature,
                "semantics": "Source-trained selective localization evidence, not contact truth or calibrated probability"}
