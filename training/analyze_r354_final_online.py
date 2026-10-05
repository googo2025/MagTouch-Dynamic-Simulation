"""Audit the 20 newly saved R354 online Trials without changing raw records.

Command references use the newly logged G1 serial-dispatch timestamp and the
logged host travel estimate. Translation is diagnosed both per Trial and using
one independent 1.4-mm grid-derived offset per physical device; neither is
fed back to inference or used to select a model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from training.build_trajectory_dataset_v1 import (
    command_reference, command_segments, read_rows, reference_origin, sha,
)
from training.retrain_r133_coverage_xy import recorded_bool_mask


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "r354_final_online_20260930"
DATA = ROOT / "data" / "S20260930_A"
FAMILY = "FastSlowReconstructionR348"
MAIN_PLAN_NAMES = (
    "Plan_20260930_044014_R354_SOURCE_R348_FINAL_ONLINE__mag_5x5-had-239",
    "Plan_20260930_045900_R354_TARGET_R348_FINAL_ONLINE__mag_5x5-had-Yelow",
)


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def nanfloat(value):
    return float(value) if np.isfinite(value) else None


def mask_for(mode, phase, ref):
    known = np.isfinite(ref["xyz_mm"]).all(1)
    depth = ref["xyz_mm"][:, 2]
    moving = ref["command_moving"]
    if mode == "grid_points":
        primary = (phase == "grid_indent") & ~moving & (
            ref["hold_age_s"] >= .15) & (ref["hold_age_s"] <= .55)
        action = phase == "grid_indent"
    elif mode == "angle_scan":
        action = np.array([bool(re.fullmatch(r"angle_(?:return_)?\d+", p)) for p in phase])
        primary = action & moving
    elif mode == "circle_scan":
        action = np.array([bool(re.fullmatch(r"circle_\d+", p)) for p in phase])
        primary = action & moving
    elif mode == "z_path":
        action = np.array([bool(re.fullmatch(r"path_\d+", p)) for p in phase])
        primary = action & moving
    elif mode == "compound":
        action = phase == "xyz_compound"
        primary = action & moving
    elif mode == "waypoint_xyz":
        action = np.array([bool(re.fullmatch(r"waypoint_\d{3}", p)) for p in phase])
        primary = action & moving
    elif mode == "none":
        primary = action = np.zeros(len(phase), bool)
    else:
        raise ValueError(mode)
    return primary & known & (depth >= .4), action & known


def score(a, mask, offset=None):
    eligible = mask & np.isfinite(a["q"]).all(1)
    good = eligible & a["valid"]
    result = {"reference_frames": int(eligible.sum()), "output_frames": int(good.sum()),
              "coverage": float(good.sum() / eligible.sum()) if eligible.any() else None}
    if not good.any():
        return result
    error = a["xyz"][good] - a["q"][good]
    bias = error.mean(0)
    centered = error - bias
    result.update(
        bias_xyz_mm=bias.tolist(),
        xy_raw_mean_mm=float(np.linalg.norm(error[:, :2], axis=1).mean()),
        z_raw_mae_mm=float(np.abs(error[:, 2]).mean()),
        xy_raw_p95_mm=float(np.quantile(np.linalg.norm(error[:, :2], axis=1), .95)),
        z_raw_p95_mm=float(np.quantile(np.abs(error[:, 2]), .95)),
        xy_trial_centered_mean_mm=float(np.linalg.norm(centered[:, :2], axis=1).mean()),
        z_trial_centered_mae_mm=float(np.abs(centered[:, 2]).mean()),
    )
    if offset is not None:
        e = error - offset
        result["xy_grid_offset_removed_mean_mm"] = float(np.linalg.norm(e[:, :2], axis=1).mean())
        result["z_grid_offset_removed_mae_mm"] = float(np.abs(e[:, 2]).mean())
        result["residual_bias_after_grid_offset_mm"] = e.mean(0).tolist()
    return result


def steps(a, mask, depth_only=False):
    dt = np.diff(a["time"]) / 1e9
    pairs = mask[1:] & mask[:-1] & a["valid"][1:] & a["valid"][:-1] & (dt > 0) & (dt <= .25)
    if depth_only:
        pairs &= np.abs(np.diff(a["q"][:, 2])) < .01
    dz = np.abs(np.diff(a["xyz"][:, 2]))[pairs]
    de = np.abs(np.diff(a["xyz"][:, 2] - a["q"][:, 2]))[pairs]
    return {"pairs": int(pairs.sum()),
            "z_step_p95_mm": float(np.quantile(dz, .95)) if len(dz) else None,
            "z_step_max_mm": float(dz.max()) if len(dz) else None,
            "z_steps_over_0p2_mm": int((dz > .2).sum()),
            "z_error_step_p95_mm": float(np.quantile(de, .95)) if len(de) else None}


def load_trial(path):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    pr = meta["live_prediction_recording"]
    alg = pr["algorithm_metadata"]
    if pr["model_family"] != FAMILY or alg["fit_variant"] != "source_only_fast_slow_incremental_depth_r348":
        raise ValueError("Unexpected live model " + str(path))
    if alg.get("target_fit_rows") != 0 or alg.get("cross_magnet_output_registration"):
        raise ValueError("Target adaptation/registration in " + str(path))
    frame = pd.read_csv(path / "live_predictions.csv", low_memory=False)
    extra = [json.loads(value) for value in frame.extra_outputs_json]
    t = frame.t_prediction_ns.to_numpy(np.int64)
    if (np.diff(t) <= 0).any():
        raise ValueError("Non-increasing prediction timestamps: " + str(path))
    xyz = frame[["visual_x_mm", "visual_y_mm", "visual_z_mm"]].to_numpy(float)
    valid = recorded_bool_mask(frame.interaction_valid) & np.isfinite(xyz).all(1)
    if not np.array_equal(valid, np.array([bool(value.get("xyz_valid")) for value in extra])):
        raise ValueError("Recorded validity differs from online model output: " + str(path))
    safe_hint = np.array([bool(value.get("external_no_contact_hint")) for value in extra])
    stages = read_rows(path / "stage_commands.csv")
    moves = [r for r in stages if re.match(r"^G0?1(?:\s|$)", r["command"].strip().upper())]
    if not moves or not all(r.get("t_send_perf_ns") and r.get("t_ack_perf_ns") for r in moves):
        raise ValueError("Missing R354 serial dispatch/ACK stamps: " + str(path))
    if not all(int(r["t_send_perf_ns"]) <= int(r["t_ack_perf_ns"]) for r in moves):
        raise ValueError("Invalid send/ACK order: " + str(path))
    segments = command_segments(stages)
    ref = command_reference(t, segments, reference_origin(meta))
    phase = np.array([segments[j]["phase"] if j >= 0 else "before_stage" for j in ref["command_segment_id"]])
    mode = meta["trial_parameters"]["command_mode"]
    primary, action = mask_for(mode, phase, ref)
    safe = np.isfinite(ref["xyz_mm"][:, 2]) & (ref["xyz_mm"][:, 2] <= -.2)
    if mode == "none":
        safe[:] = True
    raw_time = pd.read_csv(path / "magnetic_raw.csv", usecols=["t_group_ns"]).t_group_ns.to_numpy(np.int64)
    matched = np.isin(t, raw_time)
    if matched.mean() < .99:
        raise ValueError("Prediction/raw timestamp mismatch: " + str(path))
    return meta, {"time": t, "xyz": xyz, "valid": valid, "q": ref["xyz_mm"],
                  "phase": phase, "moving": ref["command_moving"], "primary": primary,
                  "action": action, "safe": safe, "safe_hint": safe_hint, "matched": matched,
                  "raw_rows": len(raw_time), "algorithm": alg,
                  "send_ack_ms": np.array([(int(r["t_ack_perf_ns"]) - int(r["t_send_perf_ns"])) / 1e6 for r in moves])}


def summarize_trial(path, device, a, meta, grid_offset):
    mode = meta["trial_parameters"]["command_mode"]
    result = {"device": device, "trial_id": meta["trial_parameters"]["trial_id"],
              "mode": mode, "path": path.relative_to(ROOT).as_posix(),
              "prediction_frames": len(a["time"]), "raw_magnetic_frames": a["raw_rows"],
              "valid_xyz_frames": int(a["valid"].sum()),
              "matched_raw_timestamp_fraction": float(a["matched"].mean()),
              "input_queue_drops": meta["live_prediction_recording"].get("input_groups_dropped_by_live_queue"),
              "send_ack_ms_p95": float(np.quantile(a["send_ack_ms"], .95)),
              "safe_frames": int(a["safe"].sum()),
              "safe_output_frames": int((a["safe"] & a["valid"]).sum()),
              "safe_without_external_hint_frames": int((a["safe"] & ~a["safe_hint"]).sum()),
              "safe_without_external_hint_output_frames": int((a["safe"] & ~a["safe_hint"] & a["valid"]).sum()),
              "primary": score(a, a["primary"], grid_offset),
              "full_calibrated_action": score(a, a["action"] & (a["q"][:, 2] >= .4), grid_offset)}
    if mode in ("angle_scan", "circle_scan"):
        const = a["action"] & np.isclose(a["q"][:, 2], float(meta["trial_parameters"]["z_indent_mm"]), atol=.01)
        result["constant_depth"] = score(a, const, grid_offset)
        result["constant_depth_steps"] = steps(a, const, depth_only=True)
    if mode == "z_path":
        result["shallow_command_bins"] = {}
        for lo, hi in ((0., .2), (.2, .4), (.4, .6), (.6, 1.), (1., 2.)):
            mask = a["action"] & (a["q"][:, 2] >= lo) & (a["q"][:, 2] < hi)
            result["shallow_command_bins"][f"{lo:g}-{hi:g} mm"] = score(a, mask, grid_offset)
    return result


def main():
    # The two subsequently acquired target repetitions are a separate
    # robustness analysis; they must not silently replace either main plan.
    plans = [DATA / name for name in MAIN_PLAN_NAMES]
    if not all(plan.is_dir() for plan in plans):
        raise FileNotFoundError("Missing one of the two frozen main R354 plans")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "streams").mkdir(exist_ok=True)
    report = {"schema": "R354-Frozen-Final-Online-1",
              "reference": "G1 serial-dispatch timestamp plus logged EST duration; not motor encoder feedback",
              "offset_diagnostic": "One signed XYZ offset per physical device from 1.4-mm grid holds; applied only to analysis, never to inference",
              "model_refit_or_selection": False, "raw_data_modified": False,
              "plans": {}}
    all_hashes = set()
    for plan in plans:
        pm = json.loads((plan / "plan_run_meta.json").read_text(encoding="utf-8"))
        if pm["status"] != "COMPLETED" or pm["completed_trials"] != 10 or pm["aborted_trials"]:
            raise ValueError("Incomplete plan: " + str(plan))
        device = "target" if "TARGET" in plan.name else "source"
        paths = sorted(plan.glob("Trial_*"))
        if len(paths) != 10:
            raise ValueError("Expected ten Trial folders: " + str(plan))
        loaded = []
        for path in paths:
            meta, a = load_trial(path)
            alg = a["algorithm"]
            all_hashes.add(alg.get("depth_increment_freeze_sha256"))
            loaded.append((path, meta, a))
        grid = next((a for path, meta, a in loaded if meta["trial_parameters"]["command_mode"] == "grid_points"
                     and float(meta["trial_parameters"]["z_indent_mm"]) == 1.4), None)
        if grid is None:
            raise ValueError("No independent grid offset source")
        offset_mask = grid["primary"] & grid["valid"]
        if offset_mask.sum() < 20:
            raise ValueError("Insufficient accepted grid frames for translation diagnostic")
        offset = (grid["xyz"][offset_mask] - grid["q"][offset_mask]).mean(0)
        trials = []
        for path, meta, a in loaded:
            trials.append(summarize_trial(path, device, a, meta, offset))
            np.savez_compressed(OUT / "streams" / f"{device}_{meta['trial_parameters']['trial_id']}.npz",
                                time=a["time"], xyz=a["xyz"], valid=a["valid"], q=a["q"],
                                primary=a["primary"], action=a["action"], safe=a["safe"], phase=a["phase"])
        report["plans"][device] = {"path": plan.relative_to(ROOT).as_posix(),
                                   "plan_meta_sha256": sha(plan / "plan_run_meta.json"),
                                   "grid_offset_xyz_mm": offset.tolist(),
                                   "grid_offset_frames": int(offset_mask.sum()),
                                   "trials": trials,
                                   "prediction_frames": sum(t["prediction_frames"] for t in trials),
                                   "valid_xyz_frames": sum(t["valid_xyz_frames"] for t in trials),
                                   "safe_frames": sum(t["safe_frames"] for t in trials),
                                   "safe_output_frames": sum(t["safe_output_frames"] for t in trials),
                                   "safe_without_external_hint_frames": sum(t["safe_without_external_hint_frames"] for t in trials),
                                   "safe_without_external_hint_output_frames": sum(t["safe_without_external_hint_output_frames"] for t in trials)}
        print(device, "grid offset", offset, "prediction frames", report["plans"][device]["prediction_frames"], flush=True)
        for t in trials:
            s = t["primary"]
            print(t["trial_id"], t["mode"], "coverage", s["coverage"],
                  "xy", s.get("xy_raw_mean_mm"), "xy-grid-centered", s.get("xy_grid_offset_removed_mean_mm"),
                  "z", s.get("z_raw_mae_mm"), flush=True)
    if len(all_hashes) != 1 or None in all_hashes:
        raise ValueError("Final depth freeze differs between online plans")
    report["depth_increment_freeze_sha256"] = next(iter(all_hashes))
    dump(OUT / "report.json", report)
    print("Report", (OUT / "report.json").relative_to(ROOT))


if __name__ == "__main__":
    main()
