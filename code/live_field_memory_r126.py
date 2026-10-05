"""UI adapter for the frozen had-239/5 mm R125 magnetic-memory decoder.

The existing fit gate/B0 lifecycle stays in charge of contact.  Only
accepted frames enter the new field-state estimator; its input is the *raw*
current B0-subtracted 15D field, not the gate's 100 ms median.  No device is
opened by importing this module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_normalized_fit import LiveNormalizedFitPredictor  # noqa: E402
from replay_backtest import JUMP_FAST, NormalizedFit, empty_record  # noqa: E402
from training.fit_had239_atlas_causal_residual_r116 import sha  # noqa: E402
from training.had239_field_memory_depth_guard_runtime_r126 import (  # noqa: E402
    DepthGuardFieldMemoryDecoder, R125_REPORT,
)
from training.had239_field_memory_runtime_r123 import DEFAULT_REPORT  # noqa: E402


class FieldMemoryGateEngine(NormalizedFit):
    """Reuse the existing contact gate while replacing only XYZ inference."""

    def __init__(self, profile, threshold, min_cosine, smoothing_ms,
                 contact_threshold, jump_policy, decoder):
        self.memory_decoder = decoder
        self.baseline_epoch = 0
        self.latest_model_output = None
        self.raw_delta = None
        self.force_no_contact = False
        super().__init__(profile, Path("__live_r126__"), threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)

    def _reset_dynamic(self):
        super()._reset_dynamic()
        self.memory_decoder.reset()
        self.latest_model_output = None

    def set_baseline(self, b):
        super().set_baseline(b)
        self.baseline_epoch += 1
        self.memory_decoder.reset()
        self.latest_model_output = None

    def _dynamic_predict(self, t, delta, unit, amp, base_xyz):
        if self.raw_delta is None:
            raise ValueError("R126 missing current raw B0-subtracted field")
        self.latest_model_output = self.memory_decoder.predict_frame(
            self.raw_delta, time_ns=int(t), b0_id=self.baseline_epoch,
            stream_id=0, contact_active=True)
        if self.latest_model_output is None:
            raise ValueError("R126 gate accepted frame but decoder rejected it")
        return (np.asarray(self.latest_model_output["xyz_mm"], float),
                None, 1.0, True)

    def push(self, t, b):
        if self.force_no_contact:
            self.buffer.clear()
            self.contact_active = False
            self.peak_amp = 0.0
            self.memory_decoder.reset()
            self.anchor = None
            return empty_record(t, "已确认安全抬离高度 / 禁止接触输出", {
                "external_no_contact_hint": True,
                "contact_gate_status": "known-safe-height suppression",
            })
        observed = np.asarray(b, float).reshape(-1)
        self.raw_delta = (observed - self.b0 if self.b0 is not None and
                          observed.shape == (15,) and
                          np.isfinite(observed).all() else None)
        result = super().push(t, b)
        if not result["valid"]:
            self.memory_decoder.reset()
            self.latest_model_output = None
        elif self.latest_model_output is not None:
            extra = result["extra"]
            extra.update({
                "fit_model": "R125 low-rank causal field memory + R104 atlas Z",
                "fit_variant": "had239_field_memory_depth_guard_r125",
                "dynamic_model_enabled": True,
                "memory_state": self.latest_model_output["memory_state"].tolist(),
                "memory_z_weight": self.latest_model_output.get("z_guard_weight", 1.0),
                "uncorrected_static_xyz_mm": self.latest_model_output[
                    "base_xyz_mm"].tolist(),
                "field_memory_source": "raw current 15D deltaB; gate uses its own smoothing",
                "contact_gate_status": "R92 heuristic gate, not independently qualified",
            })
        return result


class LiveR126Predictor(LiveNormalizedFitPredictor):
    family = "FieldMemoryR126"
    visual_scale = (10.0, 10.0, 3.0)

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None,
                 jump_policy=JUMP_FAST):
        if profile.get("physical_profile_id") != "magnetic_layer_v1_5*5-had-239":
            raise ValueError("R126 only supports had-239 with the final 5 mm tip")
        if abs(float(profile.get("training_contact_z_mm")) - 23.70) > 1e-6:
            raise ValueError("R126 training Contact Z must be 23.70 mm")
        if jump_policy != JUMP_FAST:
            raise ValueError("R126 requires the validation-selected fast contact policy")
        defaults = profile["defaults"]
        requested = (float(threshold), float(contact_threshold),
                     float(min_cosine), float(smoothing_ms))
        fixed = (float(defaults["threshold"]), 70.0,
                 float(defaults["min_cosine"]),
                 float(defaults["smoothing_ms"]))
        if any(abs(a - b) > 1e-9 for a, b in zip(requested, fixed)):
            raise ValueError("R126 requires its frozen R127 gate settings")
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        decoder = DepthGuardFieldMemoryDecoder.from_report()
        self.engine = FieldMemoryGateEngine(
            profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, jump_policy, decoder)
        self.metadata.update({
            "algorithm": "had239_causal_field_memory_r126_experimental",
            "fit_variant": "had239_field_memory_depth_guard_r125",
            "dynamic_model": True,
            "dynamic_state_model": True,
            "deployment_qualified": False,
            "model_report_sha256": sha(R125_REPORT),
            "source_model_report_sha256": sha(DEFAULT_REPORT),
            "contact_gate": "R127 validation-selected fast 70 uT default; true long-idle test pending",
            "validation_status": "offline full-rate replay only; independent live test pending",
            "magnetic_input": "raw current B0-subtracted 15D field",
            "depth_output": "R104 continuous atlas Z; R125 validation-selected weight zero",
        })
        self.model_summary = (
            f"{preset} | R125 磁场记忆 XY＋连续磁图 Z | "
            "快速临时接触门控（空载未独立通过） | dXYZ=本次接触首点差值")
        self.begin_baseline()

    def begin_baseline(self, seconds=1.0):
        super().begin_baseline(seconds)
        if hasattr(self.engine, "force_no_contact"):
            self.engine.force_no_contact = False

    def set_no_contact_hint(self, value, reason=""):
        super().set_no_contact_hint(value, reason)
        self.engine.force_no_contact = value is True
        if value is True:
            self.engine.memory_decoder.reset()
            self.engine.anchor = None

    def request_confirmed_baseline(self, seconds=.4, reason="pre_action_b0"):
        super().request_confirmed_baseline(seconds, reason)
        self.engine.force_no_contact = True

    def push_group(self, data, t_ns):
        output = super().push_group(data, t_ns)
        if not any(row.interaction_valid for row in output):
            # The base adapter may reject a malformed/missing sensor group
            # before calling engine.push; never carry h across that gap.
            self.engine.memory_decoder.reset()
        return output
