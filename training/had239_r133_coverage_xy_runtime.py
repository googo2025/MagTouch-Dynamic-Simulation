"""Frozen R133-D causal XY network; same architecture/features as R130."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory
from training.had239_guarded_neural_runtime_r166 import (
    GuardedBranchMemoryDecoder, GuardedNeuralMemoryDecoder,
)
from training.probe_had239_local_forward_r104 import CONFIG, ROOT, sha
from training.had239_field_memory_runtime_r123 import DEFAULT_REPORT


ARTIFACT = ROOT / "analysis/r133_coverage_retrain_20260927/causal_xy_model.npz"
METADATA = ROOT / "analysis/r133_coverage_retrain_20260927/causal_xy_model.json"


class CoverageCausalXYMLP:
    def __init__(self, artifact: Path = ARTIFACT, metadata: Path = METADATA):
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("schema") != "MagTouch-R133-CoverageRetrainMLP-1":
            raise ValueError("unexpected R133-D model schema")
        if sha(artifact) != info.get("artifact_sha256"):
            raise ValueError("R133-D model weights changed")
        if (sha(CONFIG) != info.get("static_config_sha256") or
                sha(DEFAULT_REPORT) != info.get("memory_report_sha256")):
            raise ValueError("R133-D static atlas or memory source changed")
        with np.load(artifact, allow_pickle=False) as data:
            for key in ("mean", "scale", "w1", "b1", "w2", "b2", "w3", "b3"):
                setattr(self, key, data[key].copy())
        width = info.get("selected_width")
        if (width != 64 or self.mean.shape != (53,) or self.scale.shape != (53,) or
                self.w1.shape != (53, 64) or self.b1.shape != (64,) or
                self.w2.shape != (64, 64) or self.b2.shape != (64,) or
                self.w3.shape != (64, 2) or self.b3.shape != (2,) or
                not all(np.isfinite(getattr(self, key)).all() for key in (
                    "mean", "scale", "w1", "b1", "w2", "b2", "w3", "b3")) or
                np.any(self.scale <= 0)):
            raise ValueError("invalid R133-D causal XY weights")
        self.metadata = info

    def predict(self, feature):
        x = np.asarray(feature, float).reshape(-1, 53)
        x = (x - self.mean) / self.scale
        return np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2) @ self.w3 + self.b3


class CoverageGuardedNeuralDecoder(GuardedNeuralMemoryDecoder):
    """Reuse the exact R130 online state machine, changing only the XY weights."""

    def __init__(self, atlas, basis, retention_10hz, strength):
        GuardedBranchMemoryDecoder.__init__(self, atlas, basis, retention_10hz, strength)
        self.xy_features = CausalXYFeatureHistory()
        self.xy_model = CoverageCausalXYMLP()
