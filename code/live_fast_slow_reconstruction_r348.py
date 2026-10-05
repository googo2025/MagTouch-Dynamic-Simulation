"""Opt-in R348 reconstruction with the frozen Source-only incremental Z head.

Construction and import never connect to acquisition or motion hardware.  The
R332 action/position permission and XY path are unchanged; only an already
permitted frame's final depth is adjudicated by the frozen R348 refiner.
"""
from __future__ import annotations

from live_shared_reconstruction_r343 import SharedCausalReconstructionEngine
from live_unified_tactile_r330 import LiveUnifiedTactilePredictor, GATE_NAME as R330_GATE
from training.analyze_z_evidence_r334 import sha
from training.fast_slow_runtime_r348 import FREEZE, FastSlowIncrementalDepthRefiner
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder


GATE_NAME = "统一状态＋定位证据＋增量Z（R348，独立模式）"
PRESET_NAME = "5×5｜源5 mm模型｜共享因果特征＋快慢深度R348（实验）"


class FastSlowReconstructionEngineR348(SharedCausalReconstructionEngine):
    """Exact native-field replay stack with one shared causal history."""

    def __init__(self, profile, threshold, min_cosine, smoothing_ms,
                 contact_threshold, decoder=None):
        super().__init__(profile, threshold, min_cosine, smoothing_ms,
                         contact_threshold,
                         decoder or CoverageGuardedNeuralDecoder.from_report())
        self.final_refiner = FastSlowIncrementalDepthRefiner(self.memory_decoder)
        self.reset_unified()

    def push(self, t, b, *, eligible=True):
        result = super().push(t, b, eligible=eligible)
        result["family"] = "FastSlowReconstructionR348"
        result["extra"].update(
            fit_variant="source_only_fast_slow_incremental_depth_r348",
            final_adjudication_version="R348-frozen-incremental-R341-evidence",
            output_permission_unchanged=True,
            original_xy_preserved=True,
            no_future_or_command_input=True,
        )
        return result


class LiveFastSlowReconstructionPredictorR348(LiveUnifiedTactilePredictor):
    """UI adapter; never passes through the legacy contact/display gate."""

    family = "FastSlowReconstructionR348"

    def __init__(self, profile, path, preset, threshold, min_cosine,
                 smoothing_ms, contact_threshold=None, jump_policy=GATE_NAME):
        if jump_policy != GATE_NAME:
            raise ValueError("R348必须选择独立动作状态＋磁定位证据门，不能叠加旧门控")
        super().__init__(profile, path, preset, threshold, min_cosine,
                         smoothing_ms, contact_threshold, jump_policy=R330_GATE)
        self.engine = FastSlowReconstructionEngineR348(
            profile, threshold, min_cosine, smoothing_ms, contact_threshold)
        self.metadata.update(
            algorithm="source_only_fast_slow_incremental_depth_r348",
            fit_variant="source_only_fast_slow_incremental_depth_r348",
            contact_gate=GATE_NAME,
            anti_jump_policy=GATE_NAME,
            legacy_gate_applied=False,
            localization_permission_protocol=self.engine.permission.manifest,
            depth_increment_freeze_sha256=sha(FREEZE),
            depth_increment_artifact=self.engine.final_refiner.selection["file"],
            depth_increment_dimension=self.engine.final_refiner.selection["dimension"],
            depth_increment_gain=self.engine.final_refiner.selection["gain"],
            target_fit_rows=0,
            complex_depth_fit_rows=0,
            cross_device_output_registration=False,
            validation_status="frozen full-field offline replay; R348-specific online test pending",
            deployment_qualified=False,
        )
        self.model_summary = (
            f"{preset} | R332定位许可＋R343共享因果磁历史＋R348增量Z；"
            "XY/许可不变，源侧冻结跨片测试，不读取动作指令或未来帧")
        self.begin_baseline()
