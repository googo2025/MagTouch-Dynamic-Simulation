"""R354 final-model online test entry with two frozen importable Full Plans.

Importing this module only registers plan identities and safety preflight;
hardware stays disconnected until the operator runs an experiment in the UI.
"""
from __future__ import annotations

import magtouch_acquisition_ui_v1_3_4_R348 as previous
from live_fast_slow_reconstruction_r348 import (
    GATE_NAME, LiveFastSlowReconstructionPredictorR348,
)
from training.waypoint_xyz_protocol_r136 import FROZEN_SOURCE_PLANS


core = previous.core
core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R354"
core.BUILD_ID = "20260930-V1.3.4-R354-FROZEN-R348-ONLINE-TEST"
core.__file__ = __file__

# R136 requires byte-for-byte registered waypoint plans before connecting to
# hardware. These files also contain matched conventional test actions.
FROZEN_SOURCE_PLANS.update({
    "R354_SOURCE_R348_FINAL_ONLINE": (
        "R354_SOURCE_R348_FINAL_ONLINE.json",
        "b5646e9dfef9485ee2f471fc90e662406de9a5dd0b67073b1b2e6e49775d6dae"),
    "R354_TARGET_R348_FINAL_ONLINE": (
        "R354_TARGET_R348_FINAL_ONLINE.json",
        "e33d59c4368c1f344a03b81105855f7940c769a21083a8db050f3d5ac4bb1a5a"),
})


def validate_r354_plan_row(host, item):
    """Bind this paper's new Trial to the exact frozen online predictor."""
    if item.get("paper_protocol_r348") != "frozen_final_online_v1":
        return
    if item.get("system_protocol_r332"):
        raise ValueError("R354计划不能伪装成旧R332验收计划")
    with host.live_predict_lock:
        predictor = host.live_predictor
        enabled = bool(host.live_predict_enabled)
    if not enabled or not isinstance(predictor, LiveFastSlowReconstructionPredictorR348):
        raise ValueError("R354计划要求已启用R348最终模型，而非旧R332/R133模型")
    metadata = predictor.metadata
    if metadata.get("contact_gate") != GATE_NAME:
        raise ValueError("R348最终模型不能叠加旧接触门控")
    if (item.get("required_depth_increment_freeze_sha256") !=
            metadata.get("depth_increment_freeze_sha256")):
        raise ValueError("R348快慢深度增量冻结SHA不一致")
    if (item.get("required_localization_permission_sha256") !=
            metadata.get("localization_permission_protocol", {}).get("sha256")):
        raise ValueError("R348磁定位许可冻结SHA不一致")
    if not item.get("record_live_predictions") or not item.get("requires_dynamic_b0"):
        raise ValueError("R354计划必须实时保存预测并逐Trial采B0")


_old_preflight = core.MagTouchApp._validate_full_plan_required_hardware
_old_dynamic_b0 = core.MagTouchApp._capture_full_plan_dynamic_b0
_old_execute = core.MagTouchApp._execute_stage_experiment


def _r354_preflight(host, item):
    validate_r354_plan_row(host, item)
    return _old_preflight(host, item)


def _r354_dynamic_b0(host, item, queue_index, total):
    validate_r354_plan_row(host, item)
    return _old_dynamic_b0(host, item, queue_index, total)


def _r354_execute(host, cfg):
    validate_r354_plan_row(host, cfg)
    return _old_execute(host, cfg)


core.MagTouchApp._validate_full_plan_required_hardware = _r354_preflight
core.MagTouchApp._capture_full_plan_dynamic_b0 = _r354_dynamic_b0
core.MagTouchApp._execute_stage_experiment = _r354_execute


def main():
    return previous.main()


if __name__ == "__main__":
    main()
