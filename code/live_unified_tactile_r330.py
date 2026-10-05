"""Opt-in magnetic state + selective XYZ mode; no device access on import.

The legacy threshold/cosine/three-frame-release gate is NOT called. Source-only
R300 owns response states; unchanged R133 owns coordinates. Weak observations
update causal history without being made into shallow XYZ supervision.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from live_normalized_fit import LiveNormalizedFitPredictor
from live_field_memory_r133 import CoverageBranchGateEngine
from replay_backtest import JUMP_FAST, empty_record, record, vector
from training.had239_r133_coverage_xy_runtime import ARTIFACT, METADATA, CoverageGuardedNeuralDecoder
from training.unified_response_runtime_r330 import StreamingSourceResponse, digest
from training.unified_tactile_policy_r330 import UnifiedOutputPolicy

GATE_NAME = "统一状态门控（R330，独立模式）"
PRESET_NAME = "5×5｜源5 mm模型｜浅动作状态＋选择性XYZ R330（实验）"
STATE_LABELS = {
    "position": "磁响应支持定位 / XYZ有效",
    "loading_unlocalized": "压入磁响应 / 定位证据不足，不输出XYZ",
    "weak_hold_unlocalized": "弱响应停驻 / 不输出XYZ",
    "withdrawal_unlocalized": "撤出磁响应 / 不输出XYZ",
    "residual": "残余磁响应 / 不输出XYZ",
    "quiet": "无显著磁响应 / 不等于已证明无接触",
    "uncertain": "磁响应状态不确定 / 不输出XYZ",
    "invalid_input": "磁场无效或B0后预热 / 不输出XYZ",
}


class UnifiedTactileEngine(CoverageBranchGateEngine):
    """Reuse calibration/transform scaffolding, never legacy push/gates."""
    def __init__(self, profile, threshold, min_cosine, smoothing_ms,
                 contact_threshold, decoder, observer=None):
        super().__init__(profile, threshold, min_cosine, smoothing_ms,
                         contact_threshold, JUMP_FAST, decoder)
        self.response_observer = observer or StreamingSourceResponse()
        self.output_policy = UnifiedOutputPolicy()
        self.baseline_effective_ns = None
        self.reset_unified()

    def reset_unified(self):
        self.response_observer.reset(); self.output_policy.reset()
        self.memory_decoder.reset(); self.latest_model_output = None
        self.anchor = None; self.baseline_effective_ns = None

    def set_baseline(self, b):
        super().set_baseline(b)
        if hasattr(self, "response_observer"): self.reset_unified()

    def push(self, t, b, *, eligible=True):
        t = int(t); observed = np.asarray(b, float).reshape(-1)
        finite = observed.shape == (15,) and np.isfinite(observed).all()
        if self.b0 is None:
            return empty_record(t, "等待显式无载B0", {"contact_gate": GATE_NAME})
        if self.baseline_effective_ns is None: self.baseline_effective_ns = t
        allowed = bool(eligible and finite and t >= self.baseline_effective_ns+700_000_000)
        delta = observed-self.b0 if finite else np.zeros(15)
        response = self.response_observer.step(t, delta, eligible=allowed,
                                              reset_token=self.baseline_epoch)
        if response["history_reset"] or not allowed:
            self.memory_decoder.reset(); self.latest_model_output = None
        output = None
        # Safe-height hint is only an output veto, not a state feature/label.
        if not self.force_no_contact and allowed and response["response"] and response["strength"] == 6:
            output = self.memory_decoder.predict_frame(delta, time_ns=t,
                b0_id=self.baseline_epoch, stream_id=0, contact_active=True)
        elif response["strength"] in (1, 5):
            self.memory_decoder.reset()
        candidate = None if output is None else np.asarray(output["xyz_mm"], float)
        verdict = self.output_policy.decide(response, candidate, decoder_valid=output is not None)
        if self.force_no_contact:
            verdict.update(system_state="safe_height_suppressed", xyz_valid=False,
                           xyz_mm=None, trace_break=True)
            verdict["rejection_reasons"].append("external_known_safe_height")
            self.output_policy.reset(); self.memory_decoder.reset()
        extra = {**verdict, "response_observation": response,
                 "adapter.interaction_valid": verdict["xyz_valid"],
                 "contact_gate": GATE_NAME, "legacy_gate_applied": False,
                 "legacy_threshold_applied": False, "legacy_release_lock_applied": False,
                 "fit_variant": "source_response_selective_xyz_r330",
                 "depth_output": "frozen continuous magnetic atlas; XY-only network",
                 "baseline_epoch": self.baseline_epoch,
                 "external_no_contact_hint": bool(self.force_no_contact),
                 "relative_reference": "first emitted valid XYZ since output interruption"}
        self.latest_model_output = output
        if not verdict["xyz_valid"]:
            self.anchor = None
            return empty_record(t, STATE_LABELS.get(verdict["system_state"], "已确认安全抬离 / 不输出XYZ"), extra)
        unaligned = candidate.copy(); xyz = candidate.copy()
        xyz[:2] = self.xy_rotation@xyz[:2]+self.xy_translation
        extra["xyz_mm"] = xyz.tolist()
        extra["unaligned_xyz_mm"] = unaligned.tolist()
        extra["xy_translation_mm"] = self.xy_translation.tolist()
        extra["xy_rotation_deg"] = float(np.degrees(self.xy_rotation_rad))
        for key in ("branch_used_local", "branch_global_switch_accepted", "branch_local_vetoed_by_field",
                    "branch_global_field_cost", "branch_local_field_cost", "xy_neural_correction_mm"):
            if key in output:
                v = output[key]; extra[key] = v.tolist() if isinstance(v, np.ndarray) else v
        if "memory_state" in output: extra["memory_state"] = np.asarray(output["memory_state"]).tolist()
        if self.anchor is None: self.anchor = xyz.copy()
        relative = xyz-self.anchor
        raw = dict(zip(("x", "y", "z", "dX", "dY", "dZ"), [*xyz.tolist(), *relative.tolist()]))
        result = record(t, raw, xyz, "UnifiedSelectiveTactileR330", extra)
        result["status"] = STATE_LABELS["position"]
        return result


class LiveUnifiedTactilePredictor(LiveNormalizedFitPredictor):
    family = "UnifiedSelectiveTactileR330"
    visual_scale = (10., 10., 3.)

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=GATE_NAME):
        if jump_policy != GATE_NAME:
            raise ValueError("R330必须选择独立统一状态门控，不可叠加旧门控")
        if profile.get("physical_profile_id") != "magnetic_layer_v1_5*5-had-239":
            raise ValueError("载入冻结源模型参数；实际测试磁片不受此限制")
        # Base adapter supplies explicit B0/I/O/transforms only. Its legacy
        # engine is replaced before observations; its gate is never evaluated.
        super().__init__(profile, path, preset, threshold, min_cosine, smoothing_ms,
                         contact_threshold, JUMP_FAST)
        self.engine = UnifiedTactileEngine(profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, CoverageGuardedNeuralDecoder.from_report())
        self.metadata.update({
            "algorithm": "source_response_selective_xyz_r330", "fit_variant": "source_response_selective_xyz_r330",
            "contact_gate": GATE_NAME, "anti_jump_policy": GATE_NAME,
            "threshold": None, "contact_threshold": None, "min_cosine": None, "smoothing_ms": 0,
            "legacy_gate_applied": False, "auto_b0_after_release": False,
            "response_protocols": self.engine.response_observer.manifest,
            "xy_mlp_artifact_sha256": digest(ARTIFACT), "xy_mlp_metadata_sha256": digest(METADATA),
            "source_only_response_fitting_and_selection": True, "target_fit_rows": 0,
            "localization_domain_estimated_mm": {"x": [-10, 10], "y": [-10, 10], "z": [0, 3.]},
            "atlas_calibration_depth_mm": [.4, 3.], "commanded_depth_not_a_gate": True,
            "weak_action_is_contact_truth": False, "suppressed_xyz_is_zero_error": False,
            "baseline_semantics": "explicit confirmed unloaded B0 only; 0.7 s response warmup",
            "response_evidence_confirmation": "3 consecutive noise-whitened evidence frames; not legacy 3-frame release lock",
            "validation_status": "integrated offline/unit tests; new online validation pending",
            "deployment_qualified": False,
        })
        self.model_summary = (f"{preset} | 独立状态门控，无旧阈值/3帧释放锁；"
                              "弱动作识别，支持充分时输出XYZ；源侧训练，跨片冻结测试")
        self.begin_baseline()

    def begin_baseline(self, seconds=1.0):
        super().begin_baseline(seconds)
        if hasattr(self.engine, "reset_unified"):
            self.engine.reset_unified(); self.engine.force_no_contact = False

    def set_no_contact_hint(self, value, reason=""):
        super().set_no_contact_hint(value, reason)
        self.engine.force_no_contact = value is True

    def request_confirmed_baseline(self, seconds=.4, reason="pre_action_b0"):
        super().request_confirmed_baseline(seconds, reason)
        self.engine.force_no_contact = True

    def push_group(self, data, t_ns):
        # Selective suppression must not invoke R126's invalid-output reset.
        if vector(data) is None: self.engine.reset_unified()
        rows = LiveNormalizedFitPredictor.push_group(self, data, t_ns)
        if self._fault: self.engine.reset_unified()
        if self.ready and any(r.extra_outputs.get("controlled_pre_action_b0") for r in rows):
            self.engine.reset_unified()
        return rows
