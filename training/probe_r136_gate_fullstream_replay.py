#!/usr/bin/env python3
"""Hardware-free R133 full-stream gate replay, with strict 70-uT parity.

Only already-seen Trials are used. This is a diagnostic of possible shallow
coverage recovery, not threshold selection for deployment. It replays every
saved prediction timestamp causally, changing only the contact threshold.
No output model or UI setting is changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from training.audit_r133_online_20260928 import PLANS, make_atlas, score
from training.build_trajectory_dataset_v1 import (
    baseline_schedule, command_reference, command_segments, load_magnetic, read_rows,
    reference_origin,
)
from training.evaluate_branch_inverse_r139 import FIVE_CONFIG, FIVE_KEY
from training.probe_had239_local_forward_r104 import ROOT, load_profile
from training.retrain_r133_coverage_xy import load_new


sys.path.insert(0, str(ROOT / "code"))
from live_field_memory_r133 import LiveR133CoveragePredictor  # noqa: E402
from live_field_memory_r126 import LiveR126Predictor  # noqa: E402
from live_field_memory_r130 import (  # noqa: E402
    LiveR130GuardPredictor, LiveR130NeuralPredictor,
)
from live_field_memory_r219 import LiveR219TightBranchPredictor  # noqa: E402
from live_field_memory_r223 import LiveR223AdaptiveBranchPredictor  # noqa: E402
from replay_backtest import JUMP_FAST  # noqa: E402


OUT = ROOT / "analysis/r136_gate_fullstream_replay_20260928/report.json"
OUT_CRITICAL = ROOT / "analysis/r136_gate_fullstream_replay_20260928/report_critical_v2.json"
PRESET = "5×5｜had-239＋5 mm｜D覆盖重训因果XY网络R133（实验）"


def make_predictor(profile, baseline, threshold, variant="N133"):
    defaults = profile["defaults"]
    cls = {"F": LiveR126Predictor, "M": LiveR126Predictor,
           "G": LiveR130GuardPredictor, "N130": LiveR130NeuralPredictor,
           "N133": LiveR133CoveragePredictor,
           "N219": LiveR219TightBranchPredictor,
           "N223": LiveR223AdaptiveBranchPredictor}.get(variant)
    if cls is None:
        raise ValueError(f"unknown ablation variant: {variant}")
    predictor = cls(
        profile, FIVE_CONFIG, PRESET,
        threshold=float(defaults["threshold"]),
        min_cosine=float(defaults["min_cosine"]),
        smoothing_ms=float(defaults["smoothing_ms"]),
        contact_threshold=70., jump_policy=JUMP_FAST)
    predictor.b0 = np.asarray(baseline, float).reshape(5, 3)
    predictor.engine.set_baseline(np.asarray(baseline, float).reshape(15))
    predictor.engine.contact_threshold = float(threshold)
    predictor.metadata["contact_threshold"] = float(threshold)
    if variant == "F":
        # Disables field-memory *correction* while retaining identical R126
        # gate/B0 lifecycle. At strength zero, XYZ = frozen static atlas.
        predictor.engine.memory_decoder.strength = 0.0
    return predictor


WAYPOINT_CONFIRMED_SAFE_PHASES = (
    "waypoint_safe_z", "waypoint_retract_safe", "waypoint_final_safe_z",
)


def replay(path, profile, threshold, variant="N133", *, gate_policy="fixed",
           waypoint_hint_policy="none"):
    if gate_policy not in {"fixed", "onset70_continue50"}:
        raise ValueError(f"unknown gate policy: {gate_policy}")
    if waypoint_hint_policy not in {"none", "r136", "r136_plus_approach"}:
        raise ValueError(f"unknown waypoint hint policy: {waypoint_hint_policy}")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    b0 = meta.get("dynamic_b0_calibration", {}).get("b0_5x3_uT")
    if b0 is None:
        raise ValueError(f"missing logged dynamic B0: {path}")
    frame = pd.read_csv(path / "live_predictions.csv", low_memory=False)
    t = frame.t_prediction_ns.to_numpy(np.int64)
    raw = load_magnetic(path / "magnetic_raw.csv")
    order = np.argsort(raw["t_group_ns"])
    raw_t = raw["t_group_ns"][order]
    index = np.searchsorted(raw_t, t)
    matched = index < len(raw_t)
    matched[matched] &= raw_t[index[matched]] == t[matched]
    if matched.mean() < .98:
        raise ValueError(f"timestamp match {path}: {matched.mean():.3f}")
    values = raw["b_uT"][order]
    valid_sensor = raw["sensor_valid"][order].all(axis=1)
    predictor = make_predictor(profile, b0, threshold, variant)
    stages = sorted(read_rows(path / "stage_commands.csv"),
                    key=lambda row: int(row["t_perf_ns"]))
    baselines = baseline_schedule(read_rows(path / "events.csv"), raw,
                                  command_segments(stages), reference_origin(meta))
    # The first entry is the exact logged dynamic B0 used at Trial start.
    updates = [entry for entry in baselines if entry["source"] == "logged_pre_action_window"]
    baseline_index = 0
    stage_index = 0
    safe_height = False
    pending_safe_height = False
    safe_prefixes = (
        "safe_z", "approach", "press_approach", "preload_approach", "retract",
        "grid_approach", "grid_retract", "grid_reference_safe",
        "grid_reference_approach", "final_safe_z", "path_approach",
        "path_release", "dynamic_b0_safe_z", "recenter")
    if waypoint_hint_policy != "none":
        safe_prefixes += WAYPOINT_CONFIRMED_SAFE_PHASES
    if waypoint_hint_policy == "r136_plus_approach":
        safe_prefixes += ("waypoint_approach",)
    xyz = np.full((len(t), 3), np.nan)
    observed = np.full((len(t), 15), np.nan)
    b0_epoch = np.zeros(len(t), np.int32)
    accepted = np.zeros(len(t), bool)
    statuses = {}
    extras = []
    for i, when in enumerate(t):
        while baseline_index < len(updates) and int(updates[baseline_index]["effective_ns"]) <= int(when):
            new_b0 = np.asarray(updates[baseline_index]["b0_uT"], float).reshape(15)
            predictor.b0 = new_b0.reshape(5, 3)
            predictor.engine.set_baseline(new_b0)
            predictor._previous = None
            baseline_index += 1
        # Match the UI's state machine at the *processing* time of this saved
        # prediction, not at the earlier sensor-group timestamp.
        processing_ns = int(frame.t_recorded_perf_ns.iloc[i])
        while stage_index < len(stages) and int(stages[stage_index]["t_perf_ns"]) <= processing_ns:
            stage = stages[stage_index]
            command, phase = stage.get("command", ""), stage.get("phase", "")
            hint = None
            if command.startswith("G1 "):
                if stage.get("z_mm") not in (None, ""):
                    pending_safe_height = any(phase.startswith(prefix)
                                              for prefix in safe_prefixes)
                    if not pending_safe_height:
                        safe_height = False
                else:
                    pending_safe_height = safe_height
                hint = False
            elif command == "HOST_WAIT":
                safe_height = bool(pending_safe_height)
                hint = safe_height
            if hint is not None:
                predictor.set_no_contact_hint(hint, phase)
            stage_index += 1
        if not matched[i] or not valid_sensor[index[i]]:
            group = [{"valid": False} for _ in range(5)]
        else:
            vector = values[index[i]].reshape(5, 3)
            observed[i] = vector.reshape(15) - np.asarray(predictor.b0, float).reshape(15)
            group = [{"x_ut": float(v[0]), "y_ut": float(v[1]), "z_ut": float(v[2]),
                      "valid": True} for v in vector]
        b0_epoch[i] = baseline_index
        if gate_policy == "onset70_continue50":
            # Pure retrospective probe: demand the frozen 70 uT gate before
            # the first accepted contact; only an already-active contact may
            # continue down to 50 uT.  A rejected frame re-arms at 70 uT.
            predictor.engine.contact_threshold = (
                50.0 if predictor.engine.contact_active else 70.0)
        outputs = predictor.push_group(group, int(when))
        if len(outputs) != 1:
            raise ValueError(f"replay output count {path}: {i}, {len(outputs)}")
        output = outputs[0]
        extras.append(dict(output.extra_outputs))
        accepted[i] = bool(output.interaction_valid)
        if accepted[i]:
            xyz[i] = output.visual_xyz_mm
        else:
            status = str(output.extra_outputs.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
    return {"frame": frame, "time": t, "xyz": xyz, "accepted": accepted,
            "observed": observed, "b0_epoch": b0_epoch,
            "extras": extras,
            "matched": matched, "invalid_status_counts": statuses}


def parity(online, replayed):
    current = online.interaction_valid.to_numpy(bool)
    saved = online[["visual_x_mm", "visual_y_mm", "visual_z_mm"]].to_numpy(float)
    replay_valid = replayed["accepted"]
    shared = current & replay_valid & np.isfinite(saved).all(axis=1)
    distance = np.linalg.norm(saved[shared]-replayed["xyz"][shared], axis=1)
    axis_diff = np.abs(saved[shared]-replayed["xyz"][shared])
    stored_extra = [json.loads(v) for v in online.extra_outputs_json]
    component_difference = {}
    amp_diff = []
    for i in np.flatnonzero(shared):
        a = stored_extra[i].get("amplitude_uT")
        b = replayed["extras"][i].get("amplitude_uT")
        if a is not None and b is not None:
            amp_diff.append(float(a)-float(b))
    for key in ("memory_state", "xy_neural_correction_mm",
                "branch_unfiltered_xyz_mm", "uncorrected_static_xyz_mm"):
        values = []
        for i in np.flatnonzero(shared):
            a = stored_extra[i].get(key)
            b = replayed["extras"][i].get(key)
            if isinstance(a, list) and isinstance(b, (list, tuple)) and len(a) == len(b):
                values.append(float(np.linalg.norm(np.asarray(a, float)-np.asarray(b, float))))
        if values:
            component_difference[key] = float(np.median(values))
    return {
        "frames": int(len(current)),
        "valid_mask_agreement": float(np.mean(current == replay_valid)),
        "saved_valid_frames": int(current.sum()),
        "replay_valid_frames": int(replay_valid.sum()),
        "shared_valid_frames": int(shared.sum()),
        "median_xyz_difference_mm": float(np.median(distance)) if len(distance) else None,
        "p95_xyz_difference_mm": float(np.percentile(distance, 95)) if len(distance) else None,
        "median_abs_axis_difference_mm": np.median(axis_diff, axis=0).tolist() if len(axis_diff) else None,
        "median_component_difference": component_difference,
        "median_signed_amplitude_difference_uT": float(np.median(amp_diff)) if amp_diff else None,
        "p95_abs_amplitude_difference_uT": float(np.percentile(np.abs(amp_diff), 95)) if amp_diff else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="only first no-load and first compound per device; do not save report")
    parser.add_argument("--critical", action="store_true",
                        help="pre/post no-load, both compounds, Z reversal; save separate report")
    args = parser.parse_args()
    if args.pilot and args.critical:
        parser.error("select at most one of --pilot and --critical")
    output_path = OUT_CRITICAL if args.critical else OUT
    if not args.pilot and output_path.exists():
        raise FileExistsError(f"preserve versioned report: {output_path}")
    profile = load_profile(FIVE_CONFIG, FIVE_KEY)
    atlas, basis, strength = make_atlas()
    report = {
        "schema": "MagTouch-R136-FullstreamGateReplay-Diagnostic-1",
        "status": "already-seen 9/28 data, offline counterfactual only; not approved threshold",
        "model": PRESET, "thresholds_uT": [70, 50], "groups": {},
    }
    for device, plan in PLANS.items():
        paths = sorted(p for p in (ROOT / "data/S20260928_A" / plan).glob("Trial_*") if p.is_dir())
        if args.pilot:
            none = [p for p in paths if "_none_" in p.name]
            compound = [p for p in paths if "_compound_" in p.name]
            paths = none[:1] + compound[:1]
        if args.critical:
            paths = [p for p in paths if any(tag in p.name for tag in
                     ("_none_", "_compound_", "_z_path_"))]
        rows = []
        for path in paths:
            mode = json.loads((path / "meta.json").read_text(encoding="utf-8"))[
                "trial_parameters"]["command_mode"]
            before = replay(path, profile, 70.)
            parity_report = parity(before["frame"], before)
            if (parity_report["valid_mask_agreement"] < .995 or
                    (parity_report["median_xyz_difference_mm"] is not None and
                     parity_report["median_xyz_difference_mm"] > .05)):
                raise ValueError(f"70-uT replay parity failed, no lower-threshold inference: {path} {parity_report}")
            after = replay(path, profile, 50.)
            row = {"trial": path.name, "mode": mode,
                   "parity_70_uT": parity_report,
                   "valid_50_uT": int(after["accepted"].sum()),
                   "invalid_status_50_uT": after["invalid_status_counts"]}
            if mode != "none":
                data = load_new(path, atlas, basis, strength)
                ref = data["core_ref"]
                truth = data["truth"]
                old_valid, new_valid = before["accepted"], after["accepted"]
                common = ref & old_valid & new_valid & np.isfinite(truth).all(axis=1)
                recovered = ref & ~old_valid & new_valid & np.isfinite(truth).all(axis=1)
                lost = ref & old_valid & ~new_valid
                segments = command_segments(read_rows(path / "stage_commands.csv"))
                command = command_reference(before["time"], segments,
                                            reference_origin(json.loads((path / "meta.json").read_text(encoding="utf-8"))))
                known_nocontact = command["contact_command"] == 0
                row["shared_valid_motion_frames"] = int(common.sum())
                row["newly_recovered_motion_frames"] = int(recovered.sum())
                row["lost_motion_frames"] = int(lost.sum())
                row["newly_recovered_motion_score"] = (
                    score(after["xyz"][recovered], truth[recovered]) if recovered.any() else None)
                row["paired_shared_motion_score"] = {
                    "frozen_70_uT": score(before["xyz"][common], truth[common]) if common.any() else None,
                    "counterfactual_50_uT": score(after["xyz"][common], truth[common]) if common.any() else None,
                }
                row["known_command_no_contact_frames"] = int(known_nocontact.sum())
                row["accepted_at_known_no_contact_70_uT"] = int((known_nocontact & old_valid).sum())
                row["accepted_at_known_no_contact_50_uT"] = int((known_nocontact & new_valid).sum())
                for key, replayed in (("frozen_70_uT", before), ("counterfactual_50_uT", after)):
                    use = ref & replayed["accepted"] & np.isfinite(truth).all(axis=1)
                    row[key] = {
                        "reference_frames": int(ref.sum()), "accepted_frames": int(use.sum()),
                        "coverage": float(use.sum()/max(ref.sum(), 1)),
                        "score_valid_only": score(replayed["xyz"][use], truth[use])
                        if use.any() else None,
                    }
            rows.append(row)
            print(device, mode, path.name.split("_")[1],
                  "parity", round(parity_report["valid_mask_agreement"], 5),
                  "median_xyz_diff", parity_report["median_xyz_difference_mm"],
                  "valid70", parity_report["replay_valid_frames"],
                  "valid50", row["valid_50_uT"],
                  "core_coverage70", row.get("frozen_70_uT", {}).get("coverage"),
                  "core_coverage50", row.get("counterfactual_50_uT", {}).get("coverage"),
                  "core_xy50", row.get("counterfactual_50_uT", {}).get("score_valid_only", {}).get("xy_raw_mean_mm")
                  if row.get("counterfactual_50_uT", {}).get("score_valid_only") else None,
                  flush=True)
        report["groups"][device] = rows
    if not args.pilot:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print("REPORT", output_path)


if __name__ == "__main__":
    main()
