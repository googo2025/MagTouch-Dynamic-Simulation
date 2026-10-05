"""Optional R133-D retrained causal-XY predictor; no hardware on import."""
from __future__ import annotations

from live_field_memory_r130 import GuardedBranchGateEngine, LiveR130GuardPredictor
from training.had239_r133_coverage_xy_runtime import (
    ARTIFACT, METADATA, CoverageGuardedNeuralDecoder,
)
from training.probe_had239_local_forward_r104 import sha


class CoverageBranchGateEngine(GuardedBranchGateEngine):
    def push(self, t, b):
        result = super().push(t, b)
        if result["valid"] and self.latest_model_output is not None:
            result["extra"].update({
                "fit_model": "R125 field memory + R162 guard + R133-D retrained causal XY MLP",
                "fit_variant": "had239_5mm_guarded_causal_xy_r133_D_retrain_experimental",
            })
        return result


class LiveR133CoveragePredictor(LiveR130GuardPredictor):
    family = "GuardedNeuralFieldMemoryR133"

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=None):
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        decoder = CoverageGuardedNeuralDecoder.from_report()
        self.engine = CoverageBranchGateEngine(
            profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, jump_policy, decoder)
        self.metadata.update({
            "algorithm": "had239_5mm_guarded_causal_xy_r133_D_retrain_experimental",
            "fit_variant": "had239_5mm_guarded_causal_xy_r133_D_retrain_experimental",
            "xy_mlp_artifact_sha256": sha(ARTIFACT),
            "xy_mlp_metadata_sha256": sha(METADATA),
            "xy_mlp_history_s": [0.1, 0.4],
            "xy_mlp_output": "additive XY residual after guarded branch; Z unchanged",
            "xy_mlp_training_trials": "S20260926_A:11-15,24-26; S20260927_A:R133-D:86-109",
            "xy_mlp_validation_trials": "S20260926_A:27-29",
            "xy_mlp_training_session_contact_z_mm": 23.360001,
            "static_atlas_training_contact_z_mm": 23.70,
            "validation_status": "R133 E/F same-stream offline replay only; new online test pending",
            "deployment_qualified": False,
        })
        self.model_summary = (
            f"{preset} | R125 慢磁记忆＋R162 分支保护＋R133-D 重训因果 XY 网络；"
            "Z/接触门控未重训；在线实测待完成")
        self.begin_baseline()
