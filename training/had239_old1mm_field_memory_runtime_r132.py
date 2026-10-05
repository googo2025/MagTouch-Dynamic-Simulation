"""Frozen had-239 / old 1 mm tip magnetic-memory decoder; no device access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.fit_had239_old1mm_field_memory_r131 import (
    GATE_CONFIG, OUT, PHYSICAL_ID, ROOT, STATIC_CONFIG, old1mm_atlas,
)
from training.had239_field_memory_runtime_r123 import FieldMemoryDecoder
from training.probe_had239_local_forward_r104 import load_profile, sha

DEFAULT_REPORT = OUT


class Old1mmFieldMemoryDecoder(FieldMemoryDecoder):
    @classmethod
    def from_report(cls, path: Path = DEFAULT_REPORT) -> "Old1mmFieldMemoryDecoder":
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("schema") != "MagTouch-R131-Old1mmFieldMemory-3" or
                report.get("physical_profile_id") != PHYSICAL_ID or
                report["source"]["source_config_sha256"] != sha(STATIC_CONFIG) or
                report["source"]["gate_config_sha256"] != sha(GATE_CONFIG)):
            raise ValueError("old1mm report, physical ID or source hashes mismatch")
        gate = load_profile(GATE_CONFIG, "5x5_had239_old1mm_old_formal_r92")
        if gate.get("physical_profile_id") != PHYSICAL_ID:
            raise ValueError("wrong old1mm gate profile")
        atlas, _ = old1mm_atlas()
        selection = report["selection"]["selected"]
        return cls(atlas, np.asarray(report["basis"], float),
                   float(selection["retention"]), float(selection["strength"]))
