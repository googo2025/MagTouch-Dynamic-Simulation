"""Pure-NumPy inference for the frozen R165 causal XY residual network."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.had239_field_memory_runtime_r123 import DEFAULT_REPORT
from training.probe_had239_local_forward_r104 import CONFIG, ROOT, sha

ARTIFACT = ROOT / "analysis/r165_causal_xy_mlp_20260926/model.npz"
METADATA = ROOT / "analysis/r165_causal_xy_mlp_20260926/model.json"


class CausalXYMLP:
    def __init__(self, artifact: Path = ARTIFACT, metadata: Path = METADATA):
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("schema") != "MagTouch-R165-CausalXYMLP-1":
            raise ValueError("unexpected R165 model schema")
        if sha(artifact) != info.get("artifact_sha256"):
            raise ValueError("R165 MLP weights changed")
        if (sha(CONFIG) != info.get("static_config_sha256") or
                sha(DEFAULT_REPORT) != info.get("memory_report_sha256")):
            raise ValueError("R165 static/memory source changed")
        with np.load(artifact, allow_pickle=False) as data:
            self.mean = data["mean"].copy()
            self.scale = data["scale"].copy()
            self.w1 = data["w1"].copy()
            self.b1 = data["b1"].copy()
            self.w2 = data["w2"].copy()
            self.b2 = data["b2"].copy()
            self.w3 = data["w3"].copy()
            self.b3 = data["b3"].copy()
        width = int(info.get("selected_width", 0))
        if (width not in (32, 64) or
                self.mean.shape != (53,) or self.scale.shape != (53,) or
                self.w1.shape != (53, width) or self.b1.shape != (width,) or
                self.w2.shape != (width, width) or self.b2.shape != (width,) or
                self.w3.shape != (width, 2) or self.b3.shape != (2,) or
                not all(np.isfinite(value).all() for value in (
                    self.mean, self.scale, self.w1, self.b1, self.w2,
                    self.b2, self.w3, self.b3)) or np.any(self.scale <= 0)):
            raise ValueError("invalid R165 model weights")
        self.metadata = info

    def predict(self, feature):
        x = np.asarray(feature, float).reshape(-1, 53)
        x = (x - self.mean) / self.scale
        return np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2) @ self.w3 + self.b3
