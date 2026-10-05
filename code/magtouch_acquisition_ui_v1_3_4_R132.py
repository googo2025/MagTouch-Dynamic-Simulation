"""R132 paper-plan preflight over the unchanged R131 acquisition UI.

Only plan rows carrying ``paper_protocol_r132`` receive the additional frozen
model/geometry checks. Importing this module never opens or moves hardware.
"""
from __future__ import annotations

import math

import magtouch_acquisition_ui_v1_3_4_R68 as core
from live_algorithm_ui_r131 import LiveAlgorithmPanel


core.LiveAlgorithmPanel = LiveAlgorithmPanel
core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R132"
core.BUILD_ID = "20260927-V1.3.4-R132-PAPER-PLAN-PREFLIGHT"
core.__file__ = __file__

_original_hardware_preflight = core.MagTouchApp._validate_full_plan_required_hardware
_original_dynamic_b0 = core.MagTouchApp._capture_full_plan_dynamic_b0
_original_execute = core.MagTouchApp._execute_stage_experiment


def validate_paper_plan_row(host, item):
    """Fail closed before a paper Trial if the loaded model or Z changed."""
    if item.get("paper_protocol_r132") != "frozen_prospective_v1":
        return
    active_id = str(getattr(host, "active_mag_profile_id", "") or "")
    profile = (getattr(host, "maglayer_profiles", {}) or {}).get(active_id, {})
    actual_z = profile.get("calibration", {}).get("contact_z_mm")
    expected_z = float(item["contact_z_mm"])
    tolerance = float(item.get("paper_contact_z_tolerance_mm", 0.02))
    if (actual_z is None or not math.isfinite(float(actual_z)) or
            abs(float(actual_z) - expected_z) > tolerance):
        raise ValueError(
            f"论文计划 Contact Z={expected_z:.6f} mm，当前磁片 Profile="
            f"{actual_z} mm（容差 {tolerance:g} mm）。重新五点校准后须生成新计划，"
            "不能用旧计划的冻结坐标直接运动。")
    with host.live_predict_lock:
        predictor = host.live_predictor
        enabled = bool(host.live_predict_enabled)
    if predictor is None or not enabled:
        raise ValueError("论文计划要求已载入并启用指定实时模型。")
    meta = getattr(predictor, "metadata", {}) or {}
    checks = {
        "required_model_preset": ("模型选项", meta.get("preset")),
        "required_model_family": ("模型族", getattr(predictor, "family", None)),
        "required_calibration_sha256": ("静态磁图 SHA-256", meta.get("calibration_sha256")),
        "required_xy_mlp_artifact_sha256": ("因果网络 SHA-256", meta.get("xy_mlp_artifact_sha256")),
        "required_training_profile_id": ("训练磁片", meta.get("model_training_profile_id")),
    }
    for key, (label, actual) in checks.items():
        required = item.get(key)
        if required and str(required) != str(actual):
            raise ValueError(f"论文计划{label}不一致：要求 {required}；当前 {actual}。")
    if item.get("require_zero_output_alignment"):
        rotation = float(getattr(predictor, "xy_rotation_deg", float("nan")))
        translation = getattr(predictor, "xy_translation", ())
        if (not math.isfinite(rotation) or abs(rotation) > 1e-8 or
                len(translation) != 2 or
                any(not math.isfinite(float(v)) or abs(float(v)) > 1e-8
                    for v in translation)):
            raise ValueError("零样本计划禁止模型输出的 XY 旋转/平移五点配准。")


def _paper_hardware_preflight(self, item):
    _original_hardware_preflight(self, item)
    validate_paper_plan_row(self, item)


def _paper_dynamic_b0(self, item, queue_index, total):
    # Re-check before the first safe-Z movement of *each* Plan row.
    _paper_hardware_preflight(self, item)
    return _original_dynamic_b0(self, item, queue_index, total)


def _paper_execute(self, cfg):
    # Final plan-row check after recorder creation, before the first G90/move.
    _paper_hardware_preflight(self, cfg)
    return _original_execute(self, cfg)


core.MagTouchApp._validate_full_plan_required_hardware = _paper_hardware_preflight
core.MagTouchApp._capture_full_plan_dynamic_b0 = _paper_dynamic_b0
core.MagTouchApp._execute_stage_experiment = _paper_execute


def main():
    return core.main()


if __name__ == "__main__":
    main()
