#!/usr/bin/env python3
"""Prospective read-only within-Trial repeat audit for R136 waypoint paths.

Requires a complete frozen R136 plan, then separates the two repeats using
ACTION_REPEAT_BEGIN/END timestamps. Saved R133 online predictions only; no
model training, UI connection, printer command, or redefinition of reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.audit_r133_online_20260928 import make_atlas, score
from training.audit_r136_waypoint_paths import audit_run
from training.build_trajectory_dataset_v1 import (command_reference, command_segments,
                                                  read_rows, reference_origin)
from training.probe_had239_local_forward_r104 import sha
from training.retrain_r133_coverage_xy import load_new


def repeat_windows(events, expected=2):
    starts, ends = {}, {}
    for row in events:
        kind = row.get("event")
        if kind == "ACTION_REPEAT_BEGIN":
            payload = json.loads(row["value"])
            if payload.get("action") != "waypoint_xyz":
                raise ValueError("repeat event is not waypoint_xyz")
            number = int(payload["repeat"])
            if number in starts:
                raise ValueError("duplicate repeat start")
            starts[number] = int(row["t_perf_ns"])
        elif kind == "ACTION_REPEAT_END":
            number = int(row["value"])
            if number in ends:
                raise ValueError("duplicate repeat end")
            ends[number] = int(row["t_perf_ns"])
    if set(starts) != set(range(1, expected+1)) or set(ends) != set(starts):
        raise ValueError("missing or extra waypoint repeat events")
    windows = [(number, starts[number], ends[number]) for number in sorted(starts)]
    if any(a >= b for _, a, b in windows) or any(
            windows[i][2] >= windows[i+1][1] for i in range(len(windows)-1)):
        raise ValueError("overlapping or reversed waypoint repeat windows")
    return windows


def repeat_masks(time, core_reference, windows):
    """No scored waypoint reference frame may fall outside both repeat events."""
    masks = [(time >= start) & (time <= end) for _, start, end in windows]
    memberships = sum(mask.astype(np.int8) for mask in masks)
    if np.any(core_reference & (memberships != 1)):
        raise ValueError("waypoint command reference is outside or duplicated across repeat windows")
    return masks


def score_slice(data, planned, accepted):
    row = {"reference_frames": int(planned.sum()),
           "accepted_frames": int(accepted.sum()),
           "coverage": float(accepted.sum()/max(planned.sum(), 1))}
    if accepted.sum() >= 3:
        row["r133_saved_online"] = score(data["live"][accepted], data["truth"][accepted])
    elif planned.sum() == 0:
        row["localization_unavailable_reason"] = (
            "no prediction-timestamp reference frames in this commanded segment")
    else:
        row["localization_unavailable_reason"] = (
            "fewer than three accepted frames; coverage remains reported")
    return row


def audit_repeats(folder: Path):
    folder = folder.resolve()
    parent = audit_run(folder)
    atlas, basis, strength = make_atlas()
    rows = []
    for path in sorted(p for p in folder.glob("Trial_*") if p.is_dir()):
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["trial_parameters"]["command_mode"] != "waypoint_xyz":
            continue
        cfg = json.loads((path / "batch_plan_item.json").read_text(encoding="utf-8"))
        expected_phases = [f"waypoint_{index:03d}"
                           for index in range(len(cfg["waypoints_xyz_mm"]))]
        data = load_new(path, atlas, basis, strength, min_core_frames=0)
        events = read_rows(path / "events.csv")
        windows = repeat_windows(events, expected=2)
        masks = repeat_masks(data["time"], data["core_ref"], windows)
        commands = command_segments(read_rows(path / "stage_commands.csv"))
        ref = command_reference(data["time"], commands, reference_origin(meta))
        phases = np.asarray([commands[i]["phase"] if i >= 0 else "before_stage"
                             for i in ref["command_segment_id"]])
        reps = []
        for (number, start, end), in_repeat in zip(windows, masks):
            planned = data["core_ref"] & in_repeat
            accepted = data["core"] & in_repeat
            if planned.sum() == 0:
                raise ValueError(f"repeat {number} has no waypoint motion reference frames: {path}")
            per_phase = []
            observed_phases = set(phases[planned])
            if not observed_phases.issubset(expected_phases):
                raise ValueError(f"unexpected scored phase in {path}: {observed_phases-set(expected_phases)}")
            for phase in expected_phases:
                p = planned & (phases == phase)
                a = accepted & (phases == phase)
                per_phase.append({"phase": phase, **score_slice(data, p, a)})
            reps.append({"repeat": number, "event_window_ns": [start, end],
                         **score_slice(data, planned, accepted),
                         "waypoint_segments": per_phase})
        if len(reps) != 2:
            raise ValueError("each R136 waypoint Trial must have two repeats")
        delta = {"coverage": reps[1]["coverage"]-reps[0]["coverage"]}
        if all("r133_saved_online" in r for r in reps):
            delta.update({key: reps[1]["r133_saved_online"][key]-
                          reps[0]["r133_saved_online"][key]
                          for key in ("xy_raw_mean_mm", "xy_translation_removed_mean_mm",
                                      "z_raw_mae_mm", "z_translation_removed_mae_mm")})
        else:
            delta["localization_unavailable_reason"] = (
                "at least one complete repeat has fewer than three accepted frames; "
                "coverage and per-segment failures remain reported")
        rows.append({"trial": path.name, "raw_magnetic_sha256": sha(path / "magnetic_raw.csv"),
                     "repeat2_minus_repeat1": delta, "repeats": reps})
    if len(rows) != 2:
        raise ValueError("complete R136 plan must have two scored waypoint Trials")
    return {"schema": "MagTouch-R190-WaypointRepeatStrata-1",
            "status": "read-only completed R136 plan repeat analysis; no training or hardware operation",
            "reference": "time-interpolated command from stage log, not independently measured contact point",
            "plan_name": parent["plan_name"],
            "frozen_source_plan_sha256": parent["frozen_source_plan_sha256"],
            "path_sha256": parent["path_sha256"],
            "motion_sha256": parent["motion_sha256"],
            "no_load_valid_frames": parent["no_load_valid_frames"],
            "trials": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan_run", type=Path,
                        help="completed R136 Full Plan folder; read-only")
    args = parser.parse_args()
    print(json.dumps(audit_repeats(args.plan_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
