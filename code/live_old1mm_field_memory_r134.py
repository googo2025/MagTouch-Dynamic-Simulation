"""Had-239 / old 1 mm tip: R131 field-memory XYZ with the R89 causal gate.

No acquisition or stage I/O occurs when this module is imported.  The R89
profile supplies only the already-tested R73 contact decision and B0 UI
fingerprint; no R89 XYZ residual is used in this decoder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal_old1mm_gate_r73 import R73OfflinePredictor  # noqa: E402
from live_normalized_fit import LiveNormalizedFitPredictor  # noqa: E402
from replay_backtest import JUMP_FAST, empty_record  # noqa: E402
from training.fit_had239_old1mm_field_memory_r131 import PHYSICAL_ID  # noqa: E402
from training.had239_old1mm_field_memory_runtime_r132 import (  # noqa: E402
    DEFAULT_REPORT, Old1mmFieldMemoryDecoder,
)
from training.probe_had239_local_forward_r104 import sha  # noqa: E402

R89_CONFIG = ROOT / "config/had239_old1mm_grid_safe_causal_candidate_r89_20260925.json"
R89_KEY = "5x5_had239_old1mm_grid_safe_causal_candidate_r89"


class Old1mmMemoryGateProxy:
    """Preserve fit UI methods, but replace R73 XYZ with R131 memory XYZ."""

    def __init__(self, gate: R73OfflinePredictor,
                 decoder: Old1mmFieldMemoryDecoder):
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "decoder", decoder)
        object.__setattr__(self, "memory_anchor", None)
        object.__setattr__(self, "baseline_epoch", 0)
        object.__setattr__(self, "force_no_contact", False)
        object.__setattr__(self, "latest_model_output", None)

    def __getattr__(self, name):
        return getattr(self.gate.engine, name)

    def __setattr__(self, name, value):
        if name in {"gate", "decoder", "memory_anchor", "baseline_epoch",
                    "force_no_contact", "latest_model_output"}:
            object.__setattr__(self, name, value)
        elif name == "b0":
            self.gate.b0 = None if value is None else np.asarray(value, float)
            self.gate.engine.b0 = value
            self.decoder.reset()
            self.memory_anchor = None
        elif name == "anchor":
            self.memory_anchor = value
            self.gate.engine.anchor = value
        else:
            setattr(self.gate.engine, name, value)

    @property
    def anchor(self):
        return self.memory_anchor

    @property
    def b0(self):
        return self.gate.engine.b0

    def _reset_dynamic(self):
        self.gate.engine._reset_dynamic()
        self.gate.engine.buffer.clear()
        self.gate.history.clear()
        self.gate.last_t = None
        self.gate.active = False
        self.gate.up = self.gate.down = 0
        self.decoder.reset()
        self.memory_anchor = None
        self.latest_model_output = None

    def set_baseline(self, baseline):
        self.gate.set_baseline(baseline)
        self.decoder.reset()
        self.memory_anchor = None
        self.latest_model_output = None
        self.baseline_epoch += 1

    def set_xy_rigid_transform(self, *args, **kwargs):
        self.gate.engine.set_xy_rigid_transform(*args, **kwargs)
        self.memory_anchor = None

    def push(self, t_ns, fields):
        t = int(t_ns)
        if self.force_no_contact:
            self._reset_dynamic()
            return empty_record(t, "已确认安全抬离高度 / 禁止接触输出", {
                "external_no_contact_hint": True,
                "contact_gate_status": "known-safe-height suppression",
            })
        row = self.gate.push(t, fields)
        if not row["valid"]:
            self.decoder.reset()
            self.memory_anchor = None
            self.latest_model_output = None
            return row
        observed = np.asarray(fields, float).reshape(-1)
        delta = observed - np.asarray(self.gate.b0, float)
        estimate = self.decoder.predict_frame(
            delta, time_ns=t, b0_id=self.baseline_epoch, stream_id=0,
            contact_active=True)
        if estimate is None:
            raise ValueError("R134 gate accepted but field decoder rejected")
        self.latest_model_output = estimate
        raw_xyz = np.asarray(estimate["xyz_mm"], float)
        xyz = raw_xyz.copy()
        xyz[:2] = (self.gate.engine.xy_rotation @ raw_xyz[:2] +
                   np.asarray(self.gate.engine.xy_translation, float))
        if self.memory_anchor is None:
            self.memory_anchor = xyz.copy()
        relative = xyz - self.memory_anchor
        row["xyz"] = xyz.tolist()
        row["raw"].update({key: float(value) for key, value in zip(
            ("x", "y", "z", "dX", "dY", "dZ"), (*xyz, *relative))})
        row["extra"].update({
            "fit_model": "R131 old-1mm forward atlas + low-rank causal field memory",
            "fit_variant": "had239_old1mm_field_memory_r131",
            "continuous_xy_method": "irregular-depth trilinear 15D forward atlas + causal field correction",
            "calibration_depth_range_mm": [0.2, 1.6],
            "dynamic_model_enabled": True,
            "memory_state": estimate["memory_state"].tolist(),
            "uncorrected_static_xyz_mm": estimate["base_xyz_mm"].tolist(),
            "unaligned_xyz_mm": raw_xyz.tolist(),
            "field_memory_source": "raw 15D B0-subtracted delta-B",
            "contact_gate_status": "R89/R73 causal RFF gate; trained for 1.2-2.0 mm",
            "deployment_qualified": False,
        })
        return row


class LiveOld1mmFieldMemoryPredictor(LiveNormalizedFitPredictor):
    family = "Old1mmFieldMemoryR134"
    visual_scale = (10.0, 10.0, 2.0)

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=JUMP_FAST):
        if (profile.get("physical_configuration", {}).get("profile_id") != PHYSICAL_ID and
                profile.get("physical_profile_id") != PHYSICAL_ID):
            raise ValueError("R134 requires had-239 with the old 1 mm tip")
        defaults = profile["defaults"]
        requested = (float(threshold), float(contact_threshold),
                     float(min_cosine), float(smoothing_ms))
        fixed = tuple(float(defaults[key]) for key in (
            "threshold", "contact_threshold", "min_cosine", "smoothing_ms"))
        if any(abs(a - b) > 1e-9 for a, b in zip(requested, fixed)) or jump_policy != JUMP_FAST:
            raise ValueError("R134 requires the frozen R89 causal contact gate defaults")
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        gate = R73OfflinePredictor(R89_CONFIG, R89_KEY)
        decoder = Old1mmFieldMemoryDecoder.from_report()
        self.engine = Old1mmMemoryGateProxy(gate, decoder)
        self.metadata.update({
            "algorithm": "had239_old1mm_field_memory_r134_experimental",
            "fit_variant": "had239_old1mm_field_memory_r131",
            "physical_profile_id": PHYSICAL_ID,
            "training_contact_z_mm": 16.0,
            "dynamic_model": True,
            "dynamic_state_model": True,
            "deployment_qualified": False,
            "field_memory_report_sha256": sha(DEFAULT_REPORT),
            "r89_gate_config_sha256": sha(R89_CONFIG),
            "contact_gate": "R89/R73 causal RFF, calibrated for 1.2-2.0 mm",
            "validated_joint_depth_mm": [1.2, 1.6],
            "atlas_full_grid_depth_mm": [0.2, 1.6],
            "validation_status": "offline next-day trial audit only; physical live test pending",
        })
        self.model_summary = (
            f"{preset} | R131 磁场记忆连续 XYZ＋R89 因果接触门控 | "
            "联合验证压深 1.2–1.6 mm；先空载，再受控测试")
        self.begin_baseline()

    def begin_baseline(self, seconds=1.0):
        super().begin_baseline(seconds)
        if isinstance(self.engine, Old1mmMemoryGateProxy):
            self.engine.force_no_contact = False

    def set_no_contact_hint(self, value, reason=""):
        super().set_no_contact_hint(value, reason)
        self.engine.force_no_contact = value is True
        if value is True:
            self.engine._reset_dynamic()

    def request_confirmed_baseline(self, seconds=.4, reason="pre_action_b0"):
        super().request_confirmed_baseline(seconds, reason)
        self.engine.force_no_contact = True

    def push_group(self, data, t_ns):
        output = super().push_group(data, t_ns)
        if not any(row.interaction_valid for row in output):
            self.engine.decoder.reset()
        return output
