"""Optional R130 had-239/5 mm guard and causal-MLP UI adapters.

Importing this module never opens a sensor, stage, or serial device.
"""
from __future__ import annotations

from live_field_memory_branch_r145 import BranchMemoryGateEngine, LiveR128Predictor
from training.had239_causal_xy_mlp_runtime_r165 import ARTIFACT, METADATA
from training.had239_guarded_neural_runtime_r166 import (
    MAX_LOCAL_COST_RATIO, GuardedBranchMemoryDecoder,
    GuardedNeuralMemoryDecoder,
)
from training.probe_had239_local_forward_r104 import sha


class GuardedBranchGateEngine(BranchMemoryGateEngine):
    def push(self, t, b):
        result = super().push(t, b)
        if result["valid"] and self.latest_model_output is not None:
            output = self.latest_model_output
            result["extra"].update({
                "fit_model": "R125 slow field memory + R162 evidence-guarded branch",
                "fit_variant": "had239_5mm_guarded_branch_r130",
                "branch_local_vetoed_by_field": output[
                    "branch_local_vetoed_by_field"],
                "branch_max_local_cost_ratio": MAX_LOCAL_COST_RATIO,
            })
            if "xy_neural_correction_mm" in output:
                result["extra"].update({
                    "fit_model": "R125 field memory + R162 guard + R165 causal XY MLP",
                    "fit_variant": "had239_5mm_guarded_causal_mlp_r130_experimental",
                    "xy_neural_correction_mm": output[
                        "xy_neural_correction_mm"],
                })
        return result


class LiveR130GuardPredictor(LiveR128Predictor):
    family = "GuardedFieldMemoryR130"

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=None):
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        decoder = GuardedBranchMemoryDecoder.from_report()
        self.engine = GuardedBranchGateEngine(
            profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, jump_policy, decoder)
        self.metadata.update({
            "algorithm": "had239_5mm_guarded_branch_r130",
            "fit_variant": "had239_5mm_guarded_branch_r130",
            "branch_max_local_cost_ratio": MAX_LOCAL_COST_RATIO,
            "branch_guard_logic": "reject local branch when its field cost exceeds four times global cost",
            "validation_status": "already-viewed full-rate live replay; new real-time test pending",
            "deployment_qualified": False,
        })
        self.model_summary = (
            f"{preset} | R125 慢磁记忆＋R162 磁场证据分支保护；"
            "不拟圆、不做 XYZ 平滑；需新实测")
        self.begin_baseline()


class LiveR130NeuralPredictor(LiveR130GuardPredictor):
    family = "GuardedNeuralFieldMemoryR130"

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=None):
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        decoder = GuardedNeuralMemoryDecoder.from_report()
        self.engine = GuardedBranchGateEngine(
            profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, jump_policy, decoder)
        self.metadata.update({
            "algorithm": "had239_5mm_guarded_causal_xy_mlp_r130_experimental",
            "fit_variant": "had239_5mm_guarded_causal_mlp_r130_experimental",
            "xy_mlp_artifact_sha256": sha(ARTIFACT),
            "xy_mlp_metadata_sha256": sha(METADATA),
            "xy_mlp_history_s": [0.1, 0.4],
            "xy_mlp_output": "additive XY residual after guarded branch; Z unchanged",
            "xy_mlp_training_session_contact_z_mm": 23.360001,
            "static_atlas_training_contact_z_mm": 23.70,
            "validation_status": "complete-Trial split on same-day had-239/5mm data; future live test pending",
            "deployment_qualified": False,
        })
        self.model_summary = (
            f"{preset} | R125 慢磁记忆＋R162 分支保护＋R165 因果 XY 网络；"
            "不拟圆、不用指令；需新实测")
        self.begin_baseline()
