"""Opt-in R219 magnetic-evidence branch probe; no hardware on import.

The R133 static atlas, two-dimensional magnetic memory, 53-D causal XY MLP,
contact gate and Z output remain frozen. Only the local/global magnetic-cost
veto changes from 4.0 to 1.25.  This candidate was chosen retrospectively
from R208; it must not be described as prospectively validated.
"""
from __future__ import annotations

from live_field_memory_r133 import (CoverageBranchGateEngine,
                                     LiveR133CoveragePredictor)
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder


MAX_LOCAL_COST_RATIO = 1.25


class TightCoverageDecoder(CoverageGuardedNeuralDecoder):
    def __init__(self, atlas, basis, retention_10hz, strength):
        super().__init__(atlas, basis, retention_10hz, strength)
        self.branch.max_local_cost_ratio = MAX_LOCAL_COST_RATIO

    def predict_frame(self, *args, **kwargs):
        row = super().predict_frame(*args, **kwargs)
        if row is not None:
            row["branch_max_local_cost_ratio"] = MAX_LOCAL_COST_RATIO
        return row


class R219BranchGateEngine(CoverageBranchGateEngine):
    def push(self, t, b):
        result = super().push(t, b)
        if result["valid"] and self.latest_model_output is not None:
            result["extra"].update({
                "fit_model": "R133-D causal XY + stricter magnetic-evidence branch veto",
                "fit_variant": "r219_r133_xy_branch_ratio_1p25_experimental",
                "branch_max_local_cost_ratio": MAX_LOCAL_COST_RATIO,
            })
        return result


class LiveR219TightBranchPredictor(LiveR133CoveragePredictor):
    family = "GuardedNeuralFieldMemoryR219"

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=None):
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy)
        decoder = TightCoverageDecoder.from_report()
        self.engine = R219BranchGateEngine(
            profile, threshold, min_cosine, smoothing_ms,
            contact_threshold, jump_policy, decoder)
        self.metadata.update({
            "algorithm": "r219_r133_xy_branch_ratio_1p25_experimental",
            "fit_variant": "r219_r133_xy_branch_ratio_1p25_experimental",
            "branch_max_local_cost_ratio": MAX_LOCAL_COST_RATIO,
            "branch_guard_logic": "reject local continuation when magnetic cost > 1.25 times global",
            "unchanged_from_r133": "atlas, memory, causal XY weights, Z, contact gate, B0 lifecycle",
            "selection_status": "chosen on already-viewed R208; prior R136 regression screen only",
            "validation_status": "new independent online testing pending",
            "deployment_qualified": False,
        })
        self.model_summary = (
            f"{preset} | R133 冻结磁图/记忆/XY 网络＋R219 强磁证据分支否决；"
            "Z 与接触门控不变；仅供新 Trial 实验")
        self.begin_baseline()
