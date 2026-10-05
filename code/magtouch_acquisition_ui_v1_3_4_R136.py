"""R136: bounded non-planar XYZ waypoint Trials, preserving R133 models.

The new motion action is intentionally Full-Plan-only. Importing this module
does not connect to a printer or start acquisition. The implementation uses
the existing R132 model/Profile/Contact-Z preflight and the R68 motion guard.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import magtouch_acquisition_ui_v1_3_4_R133 as previous
from training.waypoint_xyz_protocol_r136 import (
    FROZEN_SOURCE_PLANS, MAX_WAYPOINTS, SCHEMA, compile_waypoint_path,
    motion_sha256, validate_frozen_source_plan,
)


core = previous.previous.core
core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R136"
core.BUILD_ID = "20260928-V1.3.4-R136-BOUNDED-XYZ-WAYPOINT"
core.__file__ = __file__

ACTION = "waypoint_xyz"
LABEL = "waypoint_xyz / 受限非平面XYZ航点（仅完整计划）"
core.EXPERIMENT_ACTION_LABELS[ACTION] = LABEL
core.TRIAL_MODE_LABELS[ACTION] = LABEL
core.TRIAL_MODE_CODES[LABEL] = ACTION
core.AUTO_Z_ALLOWED_PHASE_PREFIXES += ("waypoint_",)

_R133LiveAlgorithmPanel = core.LiveAlgorithmPanel


class R136LiveAlgorithmPanel(_R133LiveAlgorithmPanel):
    """Keep the model guidance aligned with the later real online evidence."""

    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        for widget in self.grid_slaves(row=11, column=0):
            if isinstance(widget, core.ttk.Label):
                widget.configure(text=(
                    "R133-D 已有 9/28 had-239 与 Yellow 真实在线复刻；"
                    "XY 圆形改善，但 Yellow 的 Z、静态网格及浅压深覆盖仍有限。"
                    "原 R136/R208 冻结计划仍使用 R133；R219 固定强否决仅作消融，"
                    "R223 浅层自适应分支待新 Trial 验证。不能用旧数据宣称盲测。"))


core.LiveAlgorithmPanel = R136LiveAlgorithmPanel

_original_validate = core.MagTouchApp._validate_action_cfg
_original_execute = core.MagTouchApp._execute_stage_experiment
_original_refresh = core.MagTouchApp.refresh_full_plan_tree
_original_edit = core.MagTouchApp.edit_selected_plan_item
_original_apply = core.MagTouchApp._apply_full_plan_item_ui
_original_summary = core.MagTouchApp._build_trial_summary
_original_states = core.MagTouchApp._update_trial_param_states
_original_plan_events = core.MagTouchApp._plan_action_events
_original_record_stage = core.MagTouchApp._record_stage
_original_guard_target = core.PrinterController._guard_absolute_target
_original_start_full_plan = core.MagTouchApp.start_full_plan
_original_create_plan_run_folder = core.MagTouchApp._create_plan_run_folder
_original_dynamic_b0 = core.MagTouchApp._capture_full_plan_dynamic_b0
WAYPOINT_CONFIRMED_SAFE_Z_PHASES = frozenset({
    "waypoint_safe_z", "waypoint_retract_safe", "waypoint_final_safe_z",
})
WAYPOINT_ALLOWED_Z_PHASES = WAYPOINT_CONFIRMED_SAFE_Z_PHASES | frozenset({
    "waypoint_approach", "waypoint_contact_zero",
    # The inherited per-Trial B0 calibration safely lifts before the Trial
    # recorder or first waypoint G90.  It uses the current action as mode.
    "dynamic_b0_safe_z",
})


def _guard_absolute_target(self, x=None, y=None, z=None, mode="manual", phase="move"):
    """Tighten R136's Z firewall beyond the core's necessary prefix hook."""
    if z is not None and str(phase).startswith("waypoint_") and mode != ACTION:
        raise RuntimeError(f"SAFETY_Z_PHASE_BLOCK: waypoint Z phase requires {ACTION}, got {mode}")
    if mode == ACTION and z is not None:
        index = re.fullmatch(r"waypoint_(\d{3})", str(phase))
        legal_numbered = index is not None and int(index.group(1)) < MAX_WAYPOINTS
        if phase not in WAYPOINT_ALLOWED_Z_PHASES and not legal_numbered:
            raise RuntimeError(f"SAFETY_Z_PHASE_BLOCK: forbidden R136 Z phase={phase}")
    return _original_guard_target(self, x=x, y=y, z=z, mode=mode, phase=phase)


def _validated_waypoint(self, cfg):
    if cfg.get("action") != ACTION:
        return None
    if (cfg.get("paper_protocol_r132") != "frozen_prospective_v1" or
            not cfg.get("record_live_predictions") or
            not cfg.get("requires_dynamic_b0") or
            not cfg.get("require_zero_output_alignment") or
            not cfg.get("required_waypoint_sha256") or
            not cfg.get("required_motion_sha256") or
            not cfg.get("required_model_preset") or
            not cfg.get("required_xy_mlp_artifact_sha256")):
        raise ValueError("R136 waypoint_xyz 只允许带冻结路径 SHA、模型 SHA、动态 B0 和零输出对齐的 Full Plan；不能直接单次点动。")
    plan = compile_waypoint_path(cfg, machine_bounds=core.A5S_MACHINE_BOUNDS_MM)
    metadata_depth = float(cfg.get("indent_mm", -1))
    if (not math.isfinite(metadata_depth) or
            abs(metadata_depth - plan["envelope"]["max_depth"]) > 1e-7):
        raise ValueError("waypoint indent_mm must equal maximum waypoint depth for accurate Trial metadata")
    # Reuse all legacy general limits, but never let its single-target
    # compound envelope stand in for validation of the actual path.
    base = dict(cfg)
    base.update(action="press", indent_mm=plan["envelope"]["max_depth"],
                dx_mm=0.0, dy_mm=0.0)
    _original_validate(self, base)
    return plan


def _validate_action_cfg(self, cfg):
    if cfg.get("action") != ACTION:
        return _original_validate(self, cfg)
    return _validated_waypoint(self, cfg)["envelope"]


def _execute_waypoint(self, cfg):
    # The full queue was checked before connecting; check again immediately
    # before any G90 or move, including the source-plan identity. The legacy
    # Full Plan editor can remain accessible while a run is active; a later
    # row edit must never turn the initial preflight into a one-time check.
    if hasattr(self, "full_plan_items"):
        validate_r136_running_plan_host(self)
    self._validate_full_plan_required_hardware(cfg)
    plan = _validated_waypoint(self, cfg)
    action = ACTION
    x0, y0 = float(cfg["center_x_mm"]), float(cfg["center_y_mm"])
    contact = float(cfg["contact_z_mm"])
    safe = contact + float(cfg["safe_lift_mm"])
    approach = contact + float(cfg["approach_clearance_mm"])
    xy_speed = float(cfg["speed_mm_s"])
    z_speed = float(cfg["z_speed_mm_s"])
    reps = int(cfg["repeats"])
    interval = float(cfg.get("repeat_interval_s", 0.0))

    def check_abort():
        if self.motion_abort_event.is_set() or self.batch_abort_event.is_set():
            raise RuntimeError("MOTION_ABORTED")

    self._motion_event("SAFETY_PREFLIGHT_PASS", json.dumps(
        plan["envelope"], ensure_ascii=False))
    self._motion_event("WAYPOINT_PLAN", json.dumps({
        "schema": SCHEMA, "path_sha256": plan["path_sha256"],
        "motion_sha256": motion_sha256(cfg),
        "segments": plan["segments"], "estimated_seconds_per_repeat":
        plan["estimated_seconds_per_repeat"],
        "coordinate_convention": "XY offsets from center; depth below contact plane",
    }, ensure_ascii=False))
    self._motion_event("Z_ENVELOPE", json.dumps({
        "contact_z_mm": contact, "z_min_planned_mm": plan["envelope"]["z_min"],
        "z_max_planned_mm": safe, "action": action,
    }, ensure_ascii=False))
    self.log(f"R136 waypoint preflight PASS: {len(plan['segments'])} points, "
             f"path SHA={plan['path_sha256'][:12]}, "
             f"motion SHA={motion_sha256(cfg)[:12]}, "
             f"XYZ envelope={plan['envelope']}")
    check_abort()
    self.printer.send_line("G90", mode=action, phase="experiment_absolute_mode", strict_ack=True)
    check_abort()
    self.printer.move_abs(z=safe, speed_mm_s=z_speed, mode=action, phase="waypoint_safe_z")
    check_abort()
    self.printer.move_abs(x=x0, y=y0, speed_mm_s=xy_speed, mode=action, phase="waypoint_center_safe")

    for repeat in range(reps):
        check_abort()
        self._motion_event("ACTION_REPEAT_BEGIN", json.dumps(
            {"action": action, "repeat": repeat+1}, ensure_ascii=False))
        self.printer.move_abs(z=approach, speed_mm_s=z_speed, mode=action,
                              phase="waypoint_approach")
        # R132 already captured this Trial's B0 at the verified safe lift.
        # The approach pose may be magnetically loaded; never rebaseline here.
        self._motion_event("WAYPOINT_APPROACH_SETTLE", "no B0 update; safe-lift B0 retained")
        self.printer.dwell(0.25, mode=action, phase="waypoint_approach_settle")
        check_abort()
        self.printer.move_abs(z=contact, speed_mm_s=min(z_speed, 0.8),
                              mode=action, phase="waypoint_contact_zero")
        previous_depth = 0.0
        for point in plan["segments"]:
            check_abort()
            self._motion_event("WAYPOINT_BEGIN", json.dumps({
                "repeat": repeat+1, **point,
            }, ensure_ascii=False))
            z_target = point["z_mm"] if abs(point["depth_mm"]-previous_depth) > 1e-9 else None
            self.printer.move_abs(
                x=point["x_mm"], y=point["y_mm"], z=z_target,
                speed_mm_s=point["speed_mm_s"], mode=action,
                phase=f"waypoint_{point['index']:03d}")
            if point["hold_s"]:
                check_abort()
                self.printer.dwell(point["hold_s"], mode=action,
                                   phase=f"waypoint_hold_{point['index']:03d}")
            self._motion_event("WAYPOINT_END", json.dumps({
                "repeat": repeat+1, "index": point["index"],
            }, ensure_ascii=False))
            previous_depth = point["depth_mm"]
        # Never drag the indenter back across the specimen. Fully lift before
        # recentering or repeating; no direct XY return while contact remains.
        check_abort()
        self.printer.move_abs(z=safe, speed_mm_s=z_speed, mode=action,
                              phase="waypoint_retract_safe")
        check_abort()
        self.printer.move_abs(x=x0, y=y0, speed_mm_s=xy_speed, mode=action,
                              phase="waypoint_recenter_safe")
        self._motion_event("ACTION_REPEAT_END", str(repeat+1))
        if repeat < reps-1 and interval:
            check_abort()
            self.printer.dwell(interval, mode=action, phase="waypoint_repeat_interval")
    check_abort()
    self.printer.move_abs(z=safe, speed_mm_s=z_speed, mode=action,
                          phase="waypoint_final_safe_z")
    self._set_printer_position_async(self.printer.query_position())


def _execute_stage_experiment(self, cfg):
    if (getattr(self, "_r136_plan_protected", False) and
            getattr(self, "full_plan_running", False)):
        validate_r136_running_plan_host(self)
    if cfg.get("action") == ACTION:
        return _execute_waypoint(self, cfg)
    return _original_execute(self, cfg)


def _refresh_full_plan_tree(self):
    _original_refresh(self)
    if not hasattr(self, "plan_tree"):
        return
    for i, item in enumerate(self.full_plan_items):
        if item.get("action") != ACTION:
            continue
        values = list(self.plan_tree.item(str(i), "values"))
        if len(values) >= 6:
            values[4] = f"XYZ {len(item.get('waypoints_xyz_mm', []))}点"
            values[5] = str(item.get("required_waypoint_sha256", ""))[:10]
            self.plan_tree.item(str(i), values=values)


def _edit_selected_plan_item(self):
    idx = self._selected_plan_index()
    if idx is None or self.full_plan_items[idx].get("action") != ACTION:
        return _original_edit(self)
    item = self.full_plan_items[idx]
    win = core.tk.Toplevel(self)
    win.title(f"R136 冻结 XYZ 路径预览 #{idx+1}")
    win.transient(self)
    body = core.ScrolledText(win, width=100, height=26)
    body.pack(fill="both", expand=True, padx=10, pady=10)
    body.insert("1.0", json.dumps({
        "group": item.get("group"),
        "waypoint_schema": item.get("waypoint_schema"),
        "required_waypoint_sha256": item.get("required_waypoint_sha256"),
        "required_motion_sha256": item.get("required_motion_sha256"),
        "waypoints_xyz_mm": item.get("waypoints_xyz_mm"),
    }, ensure_ascii=False, indent=2))
    body.configure(state="disabled")

    def dry_run():
        try:
            source_sha = (validate_r136_running_plan_host(self)
                          if getattr(self, "full_plan_running", False)
                          else validate_r136_full_plan_host(self))
            plan = _validated_waypoint(self, item)
            lines = [
                "R136 冻结源计划与XYZ路径离线预检: PASS（当前Profile/模型另见下一行）",
                f"冻结四条源计划 SHA-256: {source_sha}",
                f"SHA-256: {plan['path_sha256']}",
                f"完整动作 SHA-256: {motion_sha256(item)}",
                f"XYZ 包络: {json.dumps(plan['envelope'], ensure_ascii=False)}",
                f"每轮路径长度: {plan['total_length_mm']:.3f} mm",
                f"每轮分段运动: {plan['motion_seconds_per_repeat']:.1f} s，"
                f"显式停驻: {plan['total_hold_s']:.1f} s，"
                f"逐段主机等待余量: {plan['host_wait_margin_seconds_per_repeat']:.1f} s",
                f"每轮路径预计时长: {plan['estimated_seconds_per_repeat']:.1f} s"
                "（不含接近/撤离；每个航点完成等待，非无停顿流式扫描）",
                f"全部重复与间隔预计: {plan['estimated_seconds_all_repeats_excluding_approach']:.1f} s"
                "（仍不含接近/撤离、B0、文件收尾）",
                f"混合 XY+Z 段: {plan['mixed_xyz_segments']}",
                "逐段绝对指令目标（仅显示，不发送）：",
            ]
            for point in plan["segments"]:
                lines.append(
                    f"{point['index']:03d}  X={point['x_mm']:.3f}  "
                    f"Y={point['y_mm']:.3f}  Z={point['z_mm']:.3f}  "
                    f"F={point['speed_mm_s']:.3f} mm/s  "
                    f"hold={point['hold_s']:.2f} s")
            try:
                self._validate_full_plan_required_hardware(item)
                lines.insert(1, "当前 Profile/Contact Z/模型配置预检: PASS；动态 B0 仅在正式运行时采集")
            except Exception as exc:
                lines.insert(1, f"当前 Profile/Contact Z/模型配置预检: FAIL — {exc}")
            report = "\n".join(lines)
        except Exception as exc:
            report = f"R136 离线全路径预检: FAIL\n{exc}"
        body.configure(state="normal")
        body.delete("1.0", "end")
        body.insert("1.0", report)
        body.configure(state="disabled")

    core.ttk.Button(win, text="离线 Dry-run：校验完整 XYZ 包络与逐段指令（不连接设备）",
                    command=dry_run).pack(padx=10, pady=(0, 8))
    core.ttk.Label(win, text="冻结路径只读。离线 Dry-run 只检查软件包络与指令，不证明实物无碰撞、触头已正确安装或 Contact Z 正确；首次实机运行仍须操作者现场核验 Home、急停及机械间隙。若要改路径，请另立计划版本并重新预检，不要覆盖原始预注册计划。",
                   wraplength=720).pack(padx=10, pady=(0, 10))
    core.ttk.Button(win, text="关闭 / Close", command=win.destroy).pack(pady=(0, 10))


def _apply_full_plan_item_ui(self, item):
    self._r136_current_waypoint_item = item if item.get("action") == ACTION else None
    _original_apply(self, item)
    if item.get("action") == ACTION and hasattr(self, "trial_param_hint_var"):
        self.trial_param_hint_var.set(
            f"受限 XYZ 航点：{len(item.get('waypoints_xyz_mm', []))} 点；"
            f"路径 SHA {str(item.get('required_waypoint_sha256', ''))[:12]}。"
            "仅完整计划可运行；查看计划条目可预览冻结路径。")


def _build_trial_summary(self):
    if self._trial_mode_code() == ACTION:
        item = getattr(self, "_r136_current_waypoint_item", None)
        if item is not None:
            return f"waypoint_xyz_n{len(item.get('waypoints_xyz_mm', []))}_sha{str(item.get('required_waypoint_sha256', ''))[:10]}"
        return "waypoint_xyz_plan_only"
    return _original_summary(self)


def _update_trial_param_states(self):
    _original_states(self)
    if self._trial_mode_code() == ACTION:
        for key, pair in getattr(self, "trial_param_widgets", {}).items():
            if key not in {"mode", "indenter", "material", "maglayer"}:
                self._set_param_enabled(pair, key in {"x", "y", "repeat", "repeat_interval"})
        if hasattr(self, "trial_param_hint_var"):
            self.trial_param_hint_var.set("R136 非平面 XYZ 航点仅从冻结 Full Plan JSON 载入；实验页不可直接创建或执行。")


def _plan_action_events(self, item):
    if item.get("action") == ACTION:
        return len(item.get("waypoints_xyz_mm", [])) * int(item.get("repeats", 1))
    return _original_plan_events(self, item)


def _record_stage(self, row):
    """Preserve the existing live no-contact hint after a completed safe-Z move.

    R68 recognizes its old safe phase names, but not the new R136 names.
    Mark a waypoint safe-Z command as pending *after* its G1 is logged; the
    subsequent HOST_WAIT completion then sets the existing confirmed-safe
    hint. No hint is asserted during motion or at the approach/contact pose.
    """
    _original_record_stage(self, row)
    if (str(row.get("command", "")).startswith("G1 ") and
            str(row.get("phase", "")) in WAYPOINT_CONFIRMED_SAFE_Z_PHASES and
            row.get("z_mm") is not None):
        self._live_stage_pending_safe_height = True


def validate_r136_full_plan_host(self):
    """Fail before hardware connection if the loaded four-row plan was edited."""
    items = self.full_plan_items
    if (self.plan_name_var.get() not in FROZEN_SOURCE_PLANS and
            not any(item.get("action") == ACTION for item in items)):
        return None
    if any(item.get("status", "PENDING") != "PENDING" for item in items):
        raise ValueError("R136 prospective Full Plan must start with all four items PENDING")
    center, contact = self._plan_reference_geometry()
    definition = {
        "plan_name": self.plan_name_var.get(),
        "magnetic_layer_profile_id": self.active_mag_profile_id,
        "reference_center_printer_mm": list(center),
        "reference_contact_z_mm": contact,
        "items": items,
    }
    return validate_frozen_source_plan(definition)


def validate_r136_running_plan_host(self):
    """Recheck every frozen source row while allowing normal status changes."""
    if self.plan_name_var.get() not in FROZEN_SOURCE_PLANS:
        raise ValueError("R136 running plan identity changed after startup")
    center, contact = self._plan_reference_geometry()
    return validate_frozen_source_plan({
        "plan_name": self.plan_name_var.get(),
        "magnetic_layer_profile_id": self.active_mag_profile_id,
        "reference_center_printer_mm": list(center),
        "reference_contact_z_mm": contact,
        "items": self.full_plan_items,
    })


def _capture_full_plan_dynamic_b0(self, item, queue_index, total):
    # R68 captures dynamic B0 only after moving to its safe pose. Recheck the
    # source contract *before* that first Trial motion, not merely before the
    # later waypoint G90. This also protects the leading/trailing no-load row.
    if (getattr(self, "_r136_plan_protected", False) or
            self.plan_name_var.get() in FROZEN_SOURCE_PLANS or
            any(row.get("action") == ACTION for row in self.full_plan_items)):
        validate_r136_running_plan_host(self)
        self._validate_full_plan_required_hardware(item)
    return _original_dynamic_b0(self, item, queue_index, total)


def _start_full_plan(self):
    try:
        source_sha = validate_r136_full_plan_host(self)
    except Exception as exc:
        core.messagebox.showerror("R136 冻结路径预检失败", str(exc))
        return
    self._r136_plan_protected = source_sha is not None
    try:
        return _original_start_full_plan(self)
    finally:
        if not getattr(self, "full_plan_running", False):
            self._r136_plan_protected = False


def _create_plan_run_folder(self, pending_indices):
    source_sha = validate_r136_full_plan_host(self)
    if source_sha is not None:
        self._r136_plan_protected = True
    folder = _original_create_plan_run_folder(self, pending_indices)
    if source_sha is not None:
        filename, _ = FROZEN_SOURCE_PLANS[self.plan_name_var.get()]
        source = Path(__file__).resolve().parents[1] / "config/paper_plans_r136" / filename
        # R68 snapshots full_plan_path, which can still be its current-plan
        # cache after importing another JSON. Replace only this run's copy.
        shutil.copy2(source, folder / "plan_source_snapshot.json")
    return folder


core.MagTouchApp._validate_action_cfg = _validate_action_cfg
core.MagTouchApp._execute_stage_experiment = _execute_stage_experiment
core.MagTouchApp.refresh_full_plan_tree = _refresh_full_plan_tree
core.MagTouchApp.edit_selected_plan_item = _edit_selected_plan_item
core.MagTouchApp._apply_full_plan_item_ui = _apply_full_plan_item_ui
core.MagTouchApp._build_trial_summary = _build_trial_summary
core.MagTouchApp._update_trial_param_states = _update_trial_param_states
core.MagTouchApp._plan_action_events = _plan_action_events
core.MagTouchApp._record_stage = _record_stage
core.PrinterController._guard_absolute_target = _guard_absolute_target
core.MagTouchApp.start_full_plan = _start_full_plan
core.MagTouchApp._create_plan_run_folder = _create_plan_run_folder
core.MagTouchApp._capture_full_plan_dynamic_b0 = _capture_full_plan_dynamic_b0


def main():
    return previous.main()


if __name__ == "__main__":
    main()
