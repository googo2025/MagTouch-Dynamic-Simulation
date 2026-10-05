"""Pure, hardware-free validation for bounded 3-D MagTouch waypoint paths.

Waypoints use XY offsets from the calibrated center and *indentation depth*
from the calibrated contact plane; no user-supplied absolute printer Z is
accepted. The compiled segments are piecewise-linear commanded paths, not
measured contact-point trajectories.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "MagTouch-WaypointXYZ-R136-1"
MAX_WAYPOINTS = 128
MAX_XY_OFFSET_MM = 8.0
MAX_DEPTH_MM = 3.0
MAX_SEGMENT_MM = 2.5
MAX_PATH_LENGTH_MM = 120.0
MAX_ESTIMATED_SECONDS = 360.0
MAX_HOLD_SECONDS = 20.0
MAX_TOTAL_HOLD_SECONDS = 120.0
MAX_REPEATS = 3
MOTION_FREEZE_FIELDS = (
    "waypoint_schema", "waypoints_xyz_mm", "center_x_mm", "center_y_mm",
    "contact_z_mm", "safe_lift_mm", "approach_clearance_mm",
    "speed_mm_s", "z_speed_mm_s", "repeats", "repeat_interval_s",
)

FROZEN_SOURCE_PLANS = {
    "R136_HAD239_R133_WAYPOINT_UNSEEN_3D": (
        "R136_HAD239_R133_WAYPOINT_UNSEEN_3D.json",
        "10cb2e45ad36e18cb274667a37efaaa4b9009708d1b50a9a50b8aa24381dd9ad"),
    "R136_YELLOW_R133_WAYPOINT_UNSEEN_3D": (
        "R136_YELLOW_R133_WAYPOINT_UNSEEN_3D.json",
        "8c7fbadf14d5fed122e13a76ebb04a5fa92d8a073d6fdbfab35c85cd0e7bd76c"),
    "R208_SOURCE_SPACE_DEPTH_SPEED_PAIR": (
        "R208_SOURCE_SPACE_DEPTH_SPEED_PAIR.json",
        "5ff74112d75a0448eef3ab903059d0c5343bbf319cdbf2ef0817f99a7f77a205"),
    "R208_TARGET_SPACE_DEPTH_SPEED_PAIR": (
        "R208_TARGET_SPACE_DEPTH_SPEED_PAIR.json",
        "3c028068bec2241133bbbaaababba5db0fe4958d362ed07154709849f9c79715"),
    "R208_SOURCE_HISTORY_RETURN_SPEED_PAIR": (
        "R208_SOURCE_HISTORY_RETURN_SPEED_PAIR.json",
        "3787542433b6fe009915f6126dd3db19774874d71bfb7100dca93b63b82a80d5"),
    "R208_TARGET_HISTORY_RETURN_SPEED_PAIR": (
        "R208_TARGET_HISTORY_RETURN_SPEED_PAIR.json",
        "cf120a4d347b4fcdcf00946ff233f47402152531fa62e6461577f99b7b983a35"),
    "R221_SOURCE_R219_FAST_RECONTACT": (
        "R221_SOURCE_R219_FAST_RECONTACT.json",
        "399d3bc9ac4ce7114f192cb91cdd21e9698c894eb214370085b4a7f29f81fa2d"),
    "R221_TARGET_R219_FAST_RECONTACT": (
        "R221_TARGET_R219_FAST_RECONTACT.json",
        "c744af4aef893ac4b6a30716f198474840fa809c1290778c8259077925a14431"),
    "R224_SOURCE_R223_FAST_RECONTACT": (
        "R224_SOURCE_R223_FAST_RECONTACT.json",
        "87ef8ad93c32a2b9a936408cc1e1aa4c3c9c468ade7057375f4aeef54d1bcd66"),
    "R224_TARGET_R223_FAST_RECONTACT": (
        "R224_TARGET_R223_FAST_RECONTACT.json",
        "b41fed044aca1dc061f8eb4be491b112e03dca5a30c052e955f0684f553ac0b7"),
}


def validate_frozen_source_plan(definition):
    """Match a UI queue or run definition to a preregistered source file.

    Status is mutable while a plan runs; every other source item field is
    frozen. This pure check is shared by the UI preflight and read-only audit.
    """
    name = definition.get("plan_name")
    if name not in FROZEN_SOURCE_PLANS:
        raise ValueError(f"not a frozen R136 source plan: {name}")
    filename, expected_sha = FROZEN_SOURCE_PLANS[name]
    source_path = Path(__file__).resolve().parents[1] / "config/paper_plans_r136" / filename
    actual_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"R136 source plan SHA changed: {filename}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (definition.get("magnetic_layer_profile_id") != source.get("paper_target_profile_id") or
            definition.get("reference_center_printer_mm") != source.get("reference_center_printer_mm") or
            definition.get("reference_contact_z_mm") != source.get("reference_contact_z_mm")):
        raise ValueError("R136 run physical Profile or coordinate reference differs from frozen source")
    actual_items, expected_items = definition.get("items", []), source["items"]
    if len(actual_items) != len(expected_items):
        raise ValueError("R136 run item count differs from frozen source")
    for index, (actual, expected) in enumerate(zip(actual_items, expected_items)):
        if set(actual)-{"status"} != set(expected)-{"status"}:
            raise ValueError(f"R136 run item {index} field set differs from frozen source")
        for key, value in expected.items():
            if key != "status" and actual.get(key) != value:
                raise ValueError(f"R136 run item {index} field {key} differs from frozen source")
    return expected_sha


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def path_sha256(waypoints):
    data = json.dumps(waypoints, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def motion_sha256(cfg):
    """Freeze the coordinate frame and execution speeds, not just point offsets."""
    payload = {name: cfg[name] for name in MOTION_FREEZE_FIELDS}
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compile_waypoint_path(cfg, *, machine_bounds):
    """Return validated absolute targets, envelope and conservative duration.

    This function has no UI, printer, camera or acquisition dependency.
    Runtime callers must invoke it again immediately before motion.
    """
    if cfg.get("waypoint_schema") != SCHEMA:
        raise ValueError(f"waypoint_schema must be {SCHEMA}")
    rows = cfg.get("waypoints_xyz_mm")
    if not isinstance(rows, list) or not (4 <= len(rows) <= MAX_WAYPOINTS):
        raise ValueError(f"waypoints_xyz_mm must contain 4~{MAX_WAYPOINTS} JSON objects")
    if cfg.get("required_waypoint_sha256"):
        if str(cfg["required_waypoint_sha256"]).lower() != path_sha256(rows):
            raise ValueError("waypoint path SHA-256 mismatch; create a new frozen plan after edits")
    if cfg.get("required_motion_sha256"):
        if str(cfg["required_motion_sha256"]).lower() != motion_sha256(cfg):
            raise ValueError("waypoint motion SHA-256 mismatch; create a new frozen plan after edits")

    center_x = _number(cfg.get("center_x_mm"), "center_x_mm")
    center_y = _number(cfg.get("center_y_mm"), "center_y_mm")
    contact_z = _number(cfg.get("contact_z_mm"), "contact_z_mm")
    safe_lift = _number(cfg.get("safe_lift_mm"), "safe_lift_mm")
    clearance = _number(cfg.get("approach_clearance_mm"), "approach_clearance_mm")
    z_speed = _number(cfg.get("z_speed_mm_s"), "z_speed_mm_s")
    interval = _number(cfg.get("repeat_interval_s", 0.0), "repeat_interval_s")
    repeats_raw = cfg.get("repeats", 1)
    if isinstance(repeats_raw, bool) or not isinstance(repeats_raw, int) or not (1 <= repeats_raw <= MAX_REPEATS):
        raise ValueError(f"waypoint repeats must be integer 1~{MAX_REPEATS}")
    if not (1.0 <= safe_lift <= 10.0 and 0.5 <= clearance <= safe_lift):
        raise ValueError("waypoint safe lift must be 1~10 mm and approach clearance 0.5~safe lift")
    if not (0.05 <= z_speed <= 2.0):
        raise ValueError("waypoint Z speed cap must be 0.05~2 mm/s")
    if not (0.0 <= interval <= 30.0):
        raise ValueError("waypoint repeat interval must be 0~30 s")

    segments = []
    last = (0.0, 0.0, 0.0)
    total_length = total_hold = motion_seconds = host_wait_margin_seconds = 0.0
    mixed = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
                "x_offset_mm", "y_offset_mm", "depth_mm", "speed_mm_s", "hold_s"}:
            raise ValueError(f"waypoint {i} must contain exactly x_offset_mm, y_offset_mm, depth_mm, speed_mm_s, hold_s")
        x = _number(row["x_offset_mm"], f"waypoint {i} x_offset_mm")
        y = _number(row["y_offset_mm"], f"waypoint {i} y_offset_mm")
        d = _number(row["depth_mm"], f"waypoint {i} depth_mm")
        speed = _number(row["speed_mm_s"], f"waypoint {i} speed_mm_s")
        hold = _number(row["hold_s"], f"waypoint {i} hold_s")
        if abs(x) > MAX_XY_OFFSET_MM or abs(y) > MAX_XY_OFFSET_MM:
            raise ValueError(f"waypoint {i} XY offset exceeds ±{MAX_XY_OFFSET_MM:g} mm")
        if not (0.0 <= d <= MAX_DEPTH_MM):
            raise ValueError(f"waypoint {i} depth must be 0~{MAX_DEPTH_MM:g} mm")
        if not (0.2 <= speed <= 4.0):
            raise ValueError(f"waypoint {i} speed must be 0.2~4 mm/s")
        if not (0.0 <= hold <= MAX_HOLD_SECONDS):
            raise ValueError(f"waypoint {i} hold must be 0~{MAX_HOLD_SECONDS:g} s")
        dx, dy, dz = x-last[0], y-last[1], d-last[2]
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        if not (0.05 <= distance <= MAX_SEGMENT_MM):
            raise ValueError(f"waypoint {i} segment length must be 0.05~{MAX_SEGMENT_MM:g} mm")
        if math.hypot(dx, dy) > 0.05 and abs(dz) > 0.05:
            mixed += 1
        # Marlin G1 has one vector feed; cap the entire segment by the more
        # conservative Z speed whenever Z changes.
        actual_speed = min(speed, z_speed) if abs(dz) > 1e-9 else speed
        total_length += distance
        total_hold += hold
        move_seconds = distance / actual_speed
        # PrinterController.move_abs waits for every segment to complete and
        # adds this exact interruptible host margin.  Include it in the
        # conservative dry-run duration instead of depicting a queued,
        # uninterrupted smooth path that the UI does not actually execute.
        wait_margin = max(.20, min(1.50, .15*move_seconds + .15))
        motion_seconds += move_seconds
        host_wait_margin_seconds += wait_margin
        point = {
            "index": i, "x_mm": center_x+x, "y_mm": center_y+y,
            "z_mm": contact_z-d, "depth_mm": d,
            "x_offset_mm": x, "y_offset_mm": y,
            "speed_mm_s": actual_speed, "requested_speed_mm_s": speed,
            "hold_s": hold, "segment_length_mm": distance,
        }
        segments.append(point)
        last = (x, y, d)
    if mixed < 2:
        raise ValueError("waypoint_xyz requires at least two simultaneous XY+Z segments")
    if total_length > MAX_PATH_LENGTH_MM or total_hold > MAX_TOTAL_HOLD_SECONDS:
        raise ValueError("waypoint path length or cumulative hold exceeds safety cap")
    estimated_seconds = motion_seconds + host_wait_margin_seconds + total_hold
    total_repeated_seconds = repeats_raw*estimated_seconds + (repeats_raw-1)*interval
    if estimated_seconds > MAX_ESTIMATED_SECONDS or total_repeated_seconds > 600.0:
        raise ValueError("waypoint path estimated duration exceeds safety cap")

    xs = [center_x] + [p["x_mm"] for p in segments]
    ys = [center_y] + [p["y_mm"] for p in segments]
    zs = [contact_z, contact_z+clearance, contact_z+safe_lift] + [p["z_mm"] for p in segments]
    envelope = {
        "x_min": min(xs), "x_max": max(xs),
        "y_min": min(ys), "y_max": max(ys),
        "z_min": min(zs), "z_max": max(zs),
        "max_depth": max(p["depth_mm"] for p in segments),
        "points_per_repeat": len(segments),
        "planned_contacts": len(segments)*repeats_raw,
    }
    for axis in ("x", "y", "z"):
        low, high = machine_bounds[axis]
        if envelope[f"{axis}_min"] < low or envelope[f"{axis}_max"] > high:
            raise ValueError(f"waypoint {axis.upper()} envelope exceeds machine bounds")
    return {
        "segments": segments, "envelope": envelope,
        "path_sha256": path_sha256(rows),
        "total_length_mm": total_length,
        "total_hold_s": total_hold,
        "motion_seconds_per_repeat": motion_seconds,
        "host_wait_margin_seconds_per_repeat": host_wait_margin_seconds,
        "estimated_seconds_per_repeat": estimated_seconds,
        "estimated_seconds_all_repeats_excluding_approach": total_repeated_seconds,
        "mixed_xyz_segments": mixed,
    }
