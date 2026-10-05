#!/usr/bin/env python3
"""Read-only same-stream audit for future R136 nonplanar XYZ plan runs.

Run only after a human has collected a completed plan. XY/Z references are
time-interpolated machine commands; no contact-position camera claim is made.
Nothing in this file sends a device command, changes model weights, or writes
to a Trial. All new path data are held out from training by this auditor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from training.audit_r133_online_20260928 import EXPECTED_SHA, make_atlas, score
from training.evaluate_branch_inverse_r139 import FIVE_CONFIG
from training.had239_causal_xy_mlp_runtime_r165 import CausalXYMLP
from training.probe_had239_local_forward_r104 import ROOT
from training.retrain_r133_coverage_xy import features, load_new
from training.build_trajectory_dataset_v1 import read_rows
from training.waypoint_xyz_protocol_r136 import (FROZEN_SOURCE_PLANS, SCHEMA,
                                                  compile_waypoint_path,
                                                  motion_sha256, path_sha256,
                                                  validate_frozen_source_plan)


R136_MACHINE_BOUNDS_MM = {"x": (0.0, 305.0), "y": (0.0, 305.0),
                           "z": (0.0, 320.0)}


def mean(values):
    return float(np.mean(np.asarray(values, float)))


def interaction_valid_mask(frames: pd.DataFrame):
    """Decode recorded booleans without treating the string 'False' as true."""
    if "interaction_valid" not in frames:
        raise ValueError("R136 prediction stream lacks interaction_valid")
    text = frames["interaction_valid"].astype(str).str.strip().str.lower()
    valid = text.isin(("true", "false", "1", "0"))
    if not bool(valid.all()):
        raise ValueError("R136 prediction stream has invalid interaction_valid values")
    return text.isin(("true", "1")).to_numpy(bool)


def longest_true_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def validate_contact_prediction_timestamps(times):
    gaps = np.diff(np.asarray(times, dtype=np.int64))
    if len(gaps) == 0 or np.any(gaps <= 0):
        raise ValueError("R136 contact prediction timestamps are not strictly increasing")
    max_gap_s = float(gaps.max()/1e9)
    if max_gap_s > 1.0:
        raise ValueError("R136 contact prediction stream has a gap longer than 1 s")
    return max_gap_s


def validate_no_load_prediction_window(frames: pd.DataFrame, cfg: dict):
    """Require enough observed time to make the planned empty-load check real."""
    if "t_prediction_ns" not in frames or "interaction_valid" not in frames:
        raise ValueError("R136 no-load prediction columns are missing")
    interaction_valid_mask(frames)
    t = frames["t_prediction_ns"].to_numpy(np.int64)
    if len(t) < 500 or np.any(np.diff(t) <= 0):
        raise ValueError("R136 no-load prediction stream is too short or not monotonic")
    if np.max(np.diff(t)) > 2_000_000_000:
        raise ValueError("R136 no-load prediction stream has a gap longer than 2 s")
    span_s = float((t[-1]-t[0])/1e9)
    if span_s < float(cfg["hold_s"])-5.0:
        raise ValueError(f"R136 no-load prediction span {span_s:.1f}s is shorter than planned hold")
    return span_s


def parse_r136_g1(command: str):
    """Parse the actual logged G1 text, not just its companion CSV columns."""
    parts = str(command).strip().split()
    if not parts or parts[0] != "G1":
        raise ValueError("R136 stage row is not a G1 command")
    axes = {}
    for token in parts[1:]:
        match = re.fullmatch(r"([XYZF])([+-]?(?:\d+(?:\.\d*)?|\.\d+))", token)
        if match is None or match.group(1) in axes:
            raise ValueError(f"R136 malformed or duplicate G1 axis token: {token}")
        axes[match.group(1)] = float(match.group(2))
    if "F" not in axes:
        raise ValueError("R136 G1 command lacks feed")
    return axes


def validate_stage_waypoint_commands(stage_rows, cfg):
    """Verify the saved G1 targets and completions match the frozen path.

    A matching JSON plan alone cannot establish what the UI actually sent.
    This read-only check guards against silently scoring a wrong speed,
    position, depth, missing segment, or incomplete host wait as the plan.
    """
    compiled = compile_waypoint_path(cfg, machine_bounds=R136_MACHINE_BOUNDS_MM)
    center_x, center_y = float(cfg["center_x_mm"]), float(cfg["center_y_mm"])
    contact = float(cfg["contact_z_mm"])
    safe = contact + float(cfg["safe_lift_mm"])
    approach = contact + float(cfg["approach_clearance_mm"])
    speed = float(cfg["speed_mm_s"])
    z_speed = float(cfg["z_speed_mm_s"])

    def move(phase, *, x=None, y=None, z=None, speed_mm_s):
        return {"phase": phase, "x_mm": x, "y_mm": y, "z_mm": z,
                "feed_mm_min": 60*speed_mm_s}

    expected = []
    expected_dwells = []
    expected_event_order = []

    def add_move(point):
        expected.append(point)
        expected_event_order.extend((("G1", point["phase"]),
                                     ("HOST_WAIT", point["phase"] + "_host_wait")))

    def add_dwell(phase, seconds):
        expected_dwells.append((phase, seconds))
        expected_event_order.append(("HOST_DWELL", phase))

    add_move(move("waypoint_safe_z", z=safe, speed_mm_s=z_speed))
    add_move(move("waypoint_center_safe", x=center_x, y=center_y,
                  speed_mm_s=speed))
    for repeat in range(int(cfg["repeats"])):
        add_move(move("waypoint_approach", z=approach, speed_mm_s=z_speed))
        add_dwell("waypoint_approach_settle", 0.25)
        add_move(move("waypoint_contact_zero", z=contact,
                      speed_mm_s=min(z_speed, 0.8)))
        previous_depth = 0.0
        for point in compiled["segments"]:
            z_target = (point["z_mm"] if
                        abs(point["depth_mm"]-previous_depth) > 1e-9 else None)
            add_move(move(f"waypoint_{point['index']:03d}",
                          x=point["x_mm"], y=point["y_mm"], z=z_target,
                          speed_mm_s=point["speed_mm_s"]))
            if point["hold_s"]:
                add_dwell(f"waypoint_hold_{point['index']:03d}",
                          point["hold_s"])
            previous_depth = point["depth_mm"]
        add_move(move("waypoint_retract_safe", z=safe, speed_mm_s=z_speed))
        add_move(move("waypoint_recenter_safe", x=center_x, y=center_y,
                      speed_mm_s=speed))
        if repeat < int(cfg["repeats"])-1 and cfg.get("repeat_interval_s", 0):
            add_dwell("waypoint_repeat_interval",
                      float(cfg["repeat_interval_s"]))
    add_move(move("waypoint_final_safe_z", z=safe,
                  speed_mm_s=z_speed))
    actual_event_order = [(("G1" if str(row.get("command", "")).startswith("G1 ")
                            else row.get("command")), row.get("phase"))
                          for row in stage_rows
                          if str(row.get("phase", "")).startswith("waypoint_") and
                          (str(row.get("command", "")).startswith("G1 ") or
                           row.get("command") in {"HOST_WAIT", "HOST_DWELL"})]
    if actual_event_order != expected_event_order:
        mismatch = next((i for i, (actual, target) in enumerate(
            zip(actual_event_order, expected_event_order)) if actual != target),
            min(len(actual_event_order), len(expected_event_order)))
        raise ValueError(f"R136 waypoint action event order mismatch at item {mismatch}")
    commands = [(index, row) for index, row in enumerate(stage_rows)
                if str(row.get("command", "")).startswith("G1 ") and
                str(row.get("phase", "")).startswith("waypoint_")]
    all_g1 = [row for row in stage_rows
              if str(row.get("command", "")).startswith("G1 ")]
    if len(all_g1) != len(commands):
        raise ValueError("R136 Trial contains an unplanned non-waypoint G1 move")
    all_waits = [row for row in stage_rows if row.get("command") == "HOST_WAIT"]
    all_dwells = [row for row in stage_rows if row.get("command") == "HOST_DWELL"]
    if len(all_waits) != len(expected) or len(all_dwells) != len(expected_dwells):
        raise ValueError("R136 Trial contains an unplanned or missing host wait/dwell")
    if len(commands) != len(expected):
        raise ValueError(f"R136 action G1 count {len(commands)} != planned {len(expected)}")
    for ordinal, ((stage_index, row), point) in enumerate(zip(commands, expected)):
        phase = point["phase"]
        if row.get("phase") != phase or row.get("mode") != "waypoint_xyz":
            raise ValueError(f"R136 waypoint phase/mode mismatch at move {ordinal}")
        if str(row.get("dry_run", "")).lower() not in {"false", "0"}:
            raise ValueError(f"R136 waypoint move {ordinal} was dry-run or unlabelled")
        command_axes = parse_r136_g1(row.get("command", ""))
        for axis in ("x_mm", "y_mm", "z_mm"):
            target = point[axis]
            actual = row.get(axis)
            text_axis = axis[0].upper()
            if target is None:
                if actual not in (None, "") or text_axis in command_axes:
                    raise ValueError(f"R136 waypoint {axis} mismatch at move {ordinal}")
            elif (actual in (None, "") or text_axis not in command_axes or
                  abs(float(actual)-target) > 5e-4 or
                  abs(command_axes[text_axis]-target) > 5e-4):
                raise ValueError(f"R136 waypoint {axis} mismatch at move {ordinal}")
        if (row.get("feed_mm_min") in (None, "") or
                abs(float(row["feed_mm_min"])-point["feed_mm_min"]) > 0.051 or
                abs(command_axes["F"]-point["feed_mm_min"]) > 0.051):
            raise ValueError(f"R136 waypoint feed mismatch at move {ordinal}")
        next_g1 = (commands[ordinal+1][0] if ordinal+1 < len(commands)
                   else len(stage_rows))
        completed = any(
            stage_rows[k].get("command") == "HOST_WAIT" and
            stage_rows[k].get("phase") == phase + "_host_wait"
            for k in range(stage_index+1, next_g1))
        if not completed:
            raise ValueError(f"R136 waypoint host completion missing at move {ordinal}")
    dwells = [row for row in stage_rows
              if row.get("command") == "HOST_DWELL" and
              str(row.get("phase", "")).startswith("waypoint_")]
    if len(dwells) != len(expected_dwells):
        raise ValueError(f"R136 waypoint dwell count {len(dwells)} != planned {len(expected_dwells)}")
    for ordinal, (row, (phase, seconds)) in enumerate(zip(dwells, expected_dwells)):
        try:
            response = str(row.get("response", ""))
            actual_seconds = float(response[:-1]) if response.endswith("s") else float("nan")
        except ValueError:
            actual_seconds = float("nan")
        if row.get("phase") != phase or not np.isfinite(actual_seconds) or abs(actual_seconds-seconds) > 5e-4:
            raise ValueError(f"R136 waypoint dwell mismatch at item {ordinal}")
    return {"verified_waypoint_g1": len(compiled["segments"])*int(cfg["repeats"]),
            "verified_action_g1": len(commands),
            "verified_repeats": int(cfg["repeats"]),
            "verified_host_waits": len(commands),
            "verified_host_dwells": len(dwells)}


def validate_completed_plan(folder: Path, paths: list[Path]):
    """Fail closed on partial, reordered, or mixed-assembly plan runs."""
    definition = json.loads((folder / "plan_definition.json").read_text(encoding="utf-8"))
    run = json.loads((folder / "plan_run_meta.json").read_text(encoding="utf-8"))
    items = definition.get("items")
    if not isinstance(items, list) or len(items) != 4:
        raise ValueError("R136 prospective plan must have exactly four ordered items")
    if [item.get("action") for item in items] != ["none", "waypoint_xyz", "waypoint_xyz", "none"]:
        raise ValueError("R136 prospective plan must bracket two waypoint paths with no-load Trials")
    if (run.get("status") != "COMPLETED" or run.get("aborted_trials") != 0 or
            run.get("planned_trials") != 4 or run.get("completed_trials") != 4 or
            len(paths) != 4):
        raise ValueError("R136 plan is incomplete or has an aborted Trial")
    if (definition.get("plan_run_id") != folder.name or
            run.get("plan_run_id") != folder.name):
        raise ValueError("R136 plan-run ID does not match its directory")
    if definition.get("magnetic_layer_profile_id") != run.get("magnetic_layer_profile_id"):
        raise ValueError("R136 plan physical Profile changed between definition and completion")
    expected_hashes = [item.get("required_waypoint_sha256") for item in items[1:3]]
    if not all(expected_hashes) or len(set(expected_hashes)) != 2:
        raise ValueError("R136 plan must contain two different frozen waypoint paths")
    for index, (item, path) in enumerate(zip(items, paths)):
        actual = json.loads((path / "batch_plan_item.json").read_text(encoding="utf-8"))
        if actual.get("group") != item.get("group") or actual.get("action") != item.get("action"):
            raise ValueError(f"R136 Trial order/action mismatch at item {index}: {path.name}")
        if actual.get("required_magnetic_profile_id") != run["magnetic_layer_profile_id"]:
            raise ValueError(f"R136 physical Profile mismatch: {path.name}")
        if item["action"] == "waypoint_xyz":
            for key in ("required_waypoint_sha256", "required_motion_sha256"):
                if actual.get(key) != item.get(key):
                    raise ValueError(f"R136 frozen {key} mismatch: {path.name}")
        # The run definition is SHA-matched to the preregistered source below.
        # A Trial snapshot must also match its own planned row, including
        # motion speeds, dwell, contact geometry, and no-load durations; the
        # live status is the only legitimately mutable field.
        if (set(actual) - {"status"} != set(item) - {"status"} or
                any(actual.get(key) != value for key, value in item.items()
                    if key != "status")):
            raise ValueError(f"R136 Trial item differs from frozen run definition: {path.name}")
    return definition, run


def audit_run(folder: Path):
    folder = folder.resolve()
    paths = sorted(p for p in folder.glob("Trial_*") if p.is_dir())
    if not paths:
        raise ValueError(f"no Trial directories: {folder}")
    definition, run = validate_completed_plan(folder, paths)
    source_plan_sha = validate_frozen_source_plan(definition)
    calibration_sha = hashlib.sha256(FIVE_CONFIG.read_bytes()).hexdigest()
    if calibration_sha != definition["items"][1]["required_calibration_sha256"]:
        raise ValueError("R136 offline static calibration differs from frozen UI plan")
    atlas, basis, strength = make_atlas()
    old_model = CausalXYMLP()
    rows = []
    hashes = []
    motion_hashes = []
    for path in paths:
        cfg = json.loads((path / "batch_plan_item.json").read_text(encoding="utf-8"))
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        mode = meta["trial_parameters"]["command_mode"]
        if EXPECTED_SHA not in json.dumps(meta.get("live_prediction", meta), ensure_ascii=False):
            raise ValueError(f"wrong frozen model artifact: {path}")
        if mode == "none":
            frames = pd.read_csv(path / "live_predictions.csv", low_memory=False)
            span_s = validate_no_load_prediction_window(frames, cfg)
            active = interaction_valid_mask(frames)
            rows.append({
                "trial": path.name, "mode": mode,
                "prediction_frames": int(len(frames)),
                "prediction_span_s": span_s,
                "prediction_max_gap_s": float(np.diff(
                    frames["t_prediction_ns"].to_numpy(np.int64)).max()/1e9),
                "interaction_valid_frames": int(active.sum()),
                "interaction_valid_rate": float(active.mean()),
                "longest_false_contact_run_frames": longest_true_run(active),
            })
            continue
        if mode != "waypoint_xyz" or cfg.get("waypoint_schema") != SCHEMA:
            raise ValueError(f"unexpected action/schema: {path}")
        actual_sha = path_sha256(cfg["waypoints_xyz_mm"])
        if actual_sha != cfg.get("required_waypoint_sha256"):
            raise ValueError(f"modified waypoint list: {path}")
        actual_motion_sha = motion_sha256(cfg)
        if actual_motion_sha != cfg.get("required_motion_sha256"):
            raise ValueError(f"modified waypoint motion configuration: {path}")
        stage_conformance = validate_stage_waypoint_commands(
            read_rows(path / "stage_commands.csv"), cfg)
        hashes.append(actual_sha)
        motion_hashes.append(actual_motion_sha)
        data = load_new(path, atlas, basis, strength, min_core_frames=0)
        max_prediction_gap_s = validate_contact_prediction_timestamps(data["time"])
        core = data["core"]
        row = {
            "trial": path.name, "mode": mode, "path_sha256": actual_sha,
            "motion_sha256": actual_motion_sha,
            "stage_command_conformance": stage_conformance,
            "core_reference_frames": int(data["core_ref"].sum()),
            "core_valid_frames": int(core.sum()),
            "prediction_max_gap_s": max_prediction_gap_s,
            "coverage": float(core.sum()/max(int(data["core_ref"].sum()), 1)),
            "commanded_depth_range_mm": [
                float(np.min(data["truth"][data["core_ref"], 2])),
                float(np.max(data["truth"][data["core_ref"], 2]))]
            if data["core_ref"].any() else None,
        }
        if core.sum() >= 3:
            truth = data["truth"][core]
            static = atlas.predict(data["observed"][core])
            memory = atlas.predict(data["corrected"][core])
            guarded = data["guarded"][core]
            old = guarded.copy()
            old[:, :2] += old_model.predict(features(data, atlas)[core])
            new = data["live"][core]
            times = data["time"][core]
            close = (np.diff(times) <= 200_000_000)
            jump = np.linalg.norm(np.diff(new[:, :2], axis=0), axis=1)
            row.update({
                "static_atlas_same_stream_probe": score(static, truth),
                "memory_atlas_same_stream_probe": score(memory, truth),
                "guard_same_stream_probe": score(guarded, truth),
                "r130_same_stream_replay": score(old, truth),
                "r133_saved_online": score(new, truth),
                "r133_max_consecutive_xy_jump_mm":
                float(np.max(jump[close])) if close.any() else None,
            })
        else:
            row.update({
                "localization_unavailable_reason":
                    "fewer than three accepted core frames; zero/low coverage retained",
                "static_atlas_same_stream_probe": None,
                "memory_atlas_same_stream_probe": None,
                "guard_same_stream_probe": None,
                "r130_same_stream_replay": None,
                "r133_saved_online": None,
                "r133_max_consecutive_xy_jump_mm": None,
            })
        rows.append(row)
    if not hashes:
        raise ValueError("no waypoint_xyz Trial in plan run")
    contact = [r for r in rows if r["mode"] == "waypoint_xyz"]
    empty = [r for r in rows if r["mode"] == "none"]
    trial_macro = {
        "coverage": mean([r["coverage"] for r in contact]),
        "unscorable_contact_trials": [r["trial"] for r in contact
                                       if r["r133_saved_online"] is None],
    }
    for key in ("static_atlas_same_stream_probe", "memory_atlas_same_stream_probe",
                "guard_same_stream_probe", "r130_same_stream_replay",
                "r133_saved_online"):
        trial_macro[key] = (
            {metric: mean([r[key][metric] for r in contact]) for metric in (
                "xy_raw_mean_mm", "xy_translation_removed_mean_mm",
                "z_raw_mae_mm", "z_translation_removed_mae_mm")}
            if all(r[key] is not None for r in contact) else None)
    report = {
        "schema": "MagTouch-R136-Waypoint-ReadOnlyAudit-1",
        "plan_run": str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
        "plan_name": definition["plan_name"],
        "frozen_source_plan_sha256": source_plan_sha,
        "physical_profile_id": run["magnetic_layer_profile_id"],
        "completed_plan_integrity": True,
        "reference": "time-interpolated machine command, not independent physical contact-point XYZ",
        "translation_removed_policy": "subtract each Trial's mean signed residual only for diagnostic shape scores; headline remains raw",
        "comparison_policy": "R133 saved online; R130 frozen same-stream replay; F/M/G are same-stream probes, not separately collected online Trials",
        "expected_model_sha256": EXPECTED_SHA,
        "offline_static_calibration_sha256": calibration_sha,
        "path_sha256": sorted(set(hashes)),
        "motion_sha256": sorted(set(motion_hashes)),
        "contact_trials": len(contact), "no_load_trials": len(empty),
        "no_load_valid_frames": sum(r["interaction_valid_frames"] for r in empty),
        "no_load_total_frames": sum(r["prediction_frames"] for r in empty),
        "no_load_valid_rate": (sum(r["interaction_valid_frames"] for r in empty) /
                               max(sum(r["prediction_frames"] for r in empty), 1)),
        "trial_macro": trial_macro,
        "trials": rows,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_run", type=Path,
                        help="completed R136 Full Plan run directory; read-only")
    args = parser.parse_args()
    print(json.dumps(audit_run(args.plan_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
