"""Optional had-239/5 mm R128 UI adapter; no hardware on import."""
from __future__ import annotations

from live_field_memory_r126 import (
    FieldMemoryGateEngine, LiveR126Predictor,
)
from training.had239_branch_memory_runtime_r144 import (
    AUDIT_REPORT, BranchDepthMemoryDecoder,
)
from training.fit_had239_atlas_causal_residual_r116 import sha


class BranchMemoryGateEngine(FieldMemoryGateEngine):
    def push(self, t, b):
        result = super().push(t, b)
        if result["valid"] and self.latest_model_output is not None:
            output = self.latest_model_output
            result["extra"].update({
                "fit_model": "R125 slow magnetic memory + R138 fast inverse branch",
                "fit_variant": "had239_branch_memory_r128_experimental",
                "branch_used_local": output["branch_used_local"],
                "branch_global_switch_accepted": output[
                    "branch_global_switch_accepted"],
                "branch_global_field_cost": output["branch_global_field_cost"],
                "branch_local_field_cost": output["branch_local_field_cost"],
                "branch_unfiltered_xyz_mm": output["branch_unfiltered_xyz_mm"],
                "branch_activation_distance_mm": 1.0,
            })
        return result


class LiveR128Predictor(LiveR126Predictor):
    family = "BranchFieldMemoryR128"

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=None):
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        decoder = BranchDepthMemoryDecoder.from_report()
        self.engine = BranchMemoryGateEngine(
            profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, jump_policy, decoder)
        self.metadata.update({
            "algorithm": "had239_branch_memory_r128_experimental",
            "fit_variant": "had239_branch_memory_r128",
            "branch_audit_report_sha256": sha(AUDIT_REPORT),
            "branch_tracking": "current-field global/local inverse; no XYZ EMA or shape fit",
            "branch_activation_distance_mm": 1.0,
            "validation_status": "already-viewed full-rate live replay only; prospective live test pending",
            "deployment_qualified": False,
        })
        self.model_summary = (
            f"{preset} | R125 慢磁场记忆＋R138 快位置分支 | "
            "原接触门控未独立通过长空载验证；先确认实物和 B0")
        self.begin_baseline()
