"""Independent R332 mode: immutable R330 XYZ plus frozen fallback permission.

Import/construct is inert with respect to hardware. Provisional positions use
current and past magnetic fields only; rejected positions never appear as XYZ.
"""
from __future__ import annotations

import numpy as np
from live_unified_tactile_r330 import (
    UnifiedTactileEngine, LiveUnifiedTactilePredictor, GATE_NAME as R330_GATE, STATE_LABELS)
from replay_backtest import record
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
from training.localization_permission_runtime_r332 import StreamingLocalizationPermission

GATE_NAME = "统一状态＋定位证据（R332，独立模式）"
PRESET_NAME = "5×5｜源5 mm模型｜动作状态＋磁定位证据XYZ R332（实验）"


class EvidenceUnifiedTactileEngine(UnifiedTactileEngine):
    def __init__(self, profile, threshold, min_cosine, smoothing_ms, contact_threshold,
                 decoder, observer=None, candidate_decoder=None, permission=None):
        super().__init__(profile, threshold, min_cosine, smoothing_ms, contact_threshold, decoder, observer)
        self.candidate_decoder = candidate_decoder or CoverageGuardedNeuralDecoder.from_report()
        self.permission = permission or StreamingLocalizationPermission(self.candidate_decoder)
        self.combined_anchor = None; self.combined_was_valid = False
        self.latest_permission_features = None; self.latest_candidate_output = None

    def reset_unified(self):
        super().reset_unified()
        if hasattr(self, "candidate_decoder"): self.candidate_decoder.reset()
        if hasattr(self, "permission"): self.permission.reset()
        self.combined_anchor = None; self.combined_was_valid = False
        self.latest_permission_features = None; self.latest_candidate_output = None

    def push(self, t, b, *, eligible=True):
        result = super().push(t, b, eligible=eligible)
        observation = result["extra"].get("response_observation")
        if observation is None:
            self.candidate_decoder.reset(); self.permission.reset()
            self.combined_anchor = None; self.combined_was_valid = False
            result["family"] = "UnifiedEvidenceTactileR332"
            result["extra"].update(contact_gate=GATE_NAME, fit_variant="source_response_localization_evidence_r332")
            return result
        observed = np.asarray(b, float).reshape(-1)
        delta = observed-self.b0 if observed.shape == (15,) and np.isfinite(observed).all() else np.zeros(15)
        if (observation["history_reset"] or not observation["eligible"] or not observation["response"]
                or observation["strength"] in (1, 5) or self.force_no_contact):
            self.candidate_decoder.reset()
        candidate = None
        # Negative residual candidates are still evaluated after an explicit
        # reset, exactly as in the frozen Source feature extraction. They can
        # never arm XYZ, but their decoder lifecycle must match the next frame.
        if not self.force_no_contact and observation["eligible"] and observation["response"]:
            candidate = self.candidate_decoder.predict_frame(delta, time_ns=int(t),
                b0_id=self.baseline_epoch, stream_id=0, contact_active=True)
        evidence = self.permission.step(t, delta, observation, candidate)
        self.latest_permission_features = evidence["features"]
        self.latest_candidate_output = candidate
        base_valid = bool(result["valid"])
        added = bool(not base_valid and evidence["permitted"] and not self.force_no_contact)
        extra = result["extra"]
        extra.update(contact_gate=GATE_NAME, fit_variant="source_response_localization_evidence_r332",
                     localization_path="existing_supported" if base_valid else "provisional_permission" if added else "suppressed",
                     localization_permission_score=evidence["score"],
                     localization_permission_semantics=evidence["semantics"],
                     localization_permission_protocol=self.permission.manifest,
                     provisional_position_available=candidate is not None,
                     original_xyz_preserved=base_valid)
        if added:
            xyz = np.asarray(candidate["xyz_mm"], float).copy()
            unaligned = xyz.copy(); xyz[:2] = self.xy_rotation@xyz[:2]+self.xy_translation
            extra.update(system_state="position", xyz_valid=True, xyz_mm=xyz.tolist(),
                         unaligned_xyz_mm=unaligned.tolist(),
                         rejection_reasons=[], memory_state=np.asarray(candidate["memory_state"]).tolist(),
                         xy_neural_correction_mm=np.asarray(candidate["xy_neural_correction_mm"]).tolist())
            extra["adapter.interaction_valid"] = True
            for key in ("branch_used_local", "branch_global_switch_accepted", "branch_local_vetoed_by_field",
                        "branch_global_field_cost", "branch_local_field_cost"):
                if key in candidate:
                    value = candidate[key]
                    extra[key] = value.tolist() if isinstance(value, np.ndarray) else value
            result = record(t, {}, xyz.tolist(), "UnifiedEvidenceTactileR332", extra)
            result["status"] = "磁定位证据支持 / XYZ有效"
            self.latest_model_output = candidate
        result["family"] = "UnifiedEvidenceTactileR332"
        valid = bool(result["valid"])
        extra["trace_break"] = bool(not valid or not self.combined_was_valid)
        if valid:
            xyz = np.asarray(result["xyz"], float)
            if self.combined_anchor is None: self.combined_anchor = xyz.copy()
            relative = xyz-self.combined_anchor
            result["raw"] = dict(zip(("x", "y", "z", "dX", "dY", "dZ"), [*xyz.tolist(), *relative.tolist()]))
        else:
            self.combined_anchor = None
        self.combined_was_valid = valid
        return result


class LiveEvidenceTactilePredictor(LiveUnifiedTactilePredictor):
    family = "UnifiedEvidenceTactileR332"

    def __init__(self, profile, path, preset, threshold, min_cosine, smoothing_ms,
                 contact_threshold=None, jump_policy=GATE_NAME):
        if jump_policy != GATE_NAME:
            raise ValueError("R332必须选择独立状态＋定位证据模式，不能叠加旧门控")
        super().__init__(profile, path, preset, threshold, min_cosine, smoothing_ms,
                         contact_threshold, jump_policy=R330_GATE)
        self.engine = EvidenceUnifiedTactileEngine(profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, CoverageGuardedNeuralDecoder.from_report())
        self.metadata.update(algorithm="source_response_localization_evidence_r332",
            fit_variant="source_response_localization_evidence_r332", contact_gate=GATE_NAME,
            anti_jump_policy=GATE_NAME, localization_permission_protocol=self.engine.permission.manifest,
            localization_permission_score_limit=self.engine.permission.selected["score_limit"],
            localization_permission_confirmation_s=self.engine.permission.selected["confirmation_s"],
            provisional_decoder="same frozen R133; independent causal history, never stale displayed coordinates",
            shallow_xyz_accuracy_verified=False, independent_online_validation_pending=True)
        self.model_summary = (f"{preset} | 独立状态＋定位证据；保留原有效XYZ，按磁图和历史证据恢复过渡点；"
                              "不加旧阈值/3帧释放锁；源侧训练，跨片权重冻结")
        self.begin_baseline()
