"""Optional R223 shallow-support adaptive branch rule; no hardware on import."""
from __future__ import annotations

from live_field_memory_r133 import (CoverageBranchGateEngine,
                                     LiveR133CoveragePredictor)
from training.had239_branch_memory_runtime_r144 import (ACTIVATION_DISTANCE_MM,
                                                         INNOVATION_GAIN,
                                                         SPATIAL_PENALTY)
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
from training.shallow_adaptive_branch_r223 import ShallowAdaptiveEvidenceBranch


class AdaptiveCoverageDecoder(CoverageGuardedNeuralDecoder):
    def __init__(self, atlas, basis, retention_10hz, strength):
        super().__init__(atlas, basis, retention_10hz, strength)
        self.branch = ShallowAdaptiveEvidenceBranch(
            atlas, spatial_penalty=SPATIAL_PENALTY,
            innovation_gain=INNOVATION_GAIN,
            activation_distance_mm=ACTIVATION_DISTANCE_MM,
            max_local_cost_ratio=4.)

    def predict_frame(self, *args, **kwargs):
        row = super().predict_frame(*args, **kwargs)
        if row is not None:
            row["branch_max_local_cost_ratio"] = self.branch.max_local_cost_ratio
        return row


class AdaptiveBranchGateEngine(CoverageBranchGateEngine):
    def push(self, t, b):
        result = super().push(t, b)
        if result["valid"] and self.latest_model_output is not None:
            result["extra"].update({
                "fit_model": "R133-D causal XY + depth-of-atlas-support branch evidence",
                "fit_variant": "r223_r133_xy_shallow_adaptive_branch_experimental",
                "branch_max_local_cost_ratio": self.latest_model_output[
                    "branch_max_local_cost_ratio"],
            })
        return result


class LiveR223AdaptiveBranchPredictor(LiveR133CoveragePredictor):
    family = "GuardedNeuralFieldMemoryR223"

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=None):
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        decoder = AdaptiveCoverageDecoder.from_report()
        self.engine = AdaptiveBranchGateEngine(
            profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, jump_policy, decoder)
        self.metadata.update({
            "algorithm": "r223_r133_xy_shallow_adaptive_branch_experimental",
            "fit_variant": "r223_r133_xy_shallow_adaptive_branch_experimental",
            "branch_guard_logic": "ratio=1.25+2.75*clip((current_global_Z-0.6)/0.4,0,1)",
            "branch_max_local_cost_ratio_range": [1.25, 4.0],
            "unchanged_from_r133": "atlas, memory, causal XY weights, Z, contact gate, B0 lifecycle",
            "selection_status": "hypothesis developed on viewed R208 and screened on earlier online records",
            "validation_status": "new independent online testing pending",
            "deployment_qualified": False,
        })
        self.model_summary = (
            f"{preset} | R133 冻结磁图/记忆/XY 网络＋R223 浅层自适应分支证据；"
            "Z 与接触门控不变；仅供新 Trial 实验")
        self.begin_baseline()
