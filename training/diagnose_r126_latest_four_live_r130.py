#!/usr/bin/env python3
"""R130 read-only diagnosis of S20260926_A live Trials 12--15.

Uses recorded *live* predictions, magnetic/stage/event source files, and the
same command interpolation as earlier R52 analyses. It never re-runs the UI,
changes model coefficients, or treats printer commands as measured tip pose.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from training.build_trajectory_dataset_v1 import (
    command_reference, command_segments, read_rows, reference_origin,
)
from training.fit_had239_atlas_causal_residual_r116 import ROOT, sha

SESSION = ROOT / "data/S20260926_A"
OUT = ROOT / "analysis/r130_r126_latest_four_live_20260926/report_v4.json"
TRIALS = (12, 13, 14, 15)
TRAIN_CONTACT_Z = 23.70


def xy_summary(pred: np.ndarray, truth: np.ndarray) -> dict:
    diff = pred[:, :2] - truth[:, :2]
    distance = np.linalg.norm(diff, axis=1)
    bias = diff.mean(axis=0)
    centered = np.linalg.norm(diff - bias, axis=1)
    return {"n": int(len(diff)), "xy_mean_mm": float(distance.mean()),
            "xy_p95_mm": float(np.percentile(distance, 95)),
            "xy_bias_mm": bias.tolist(),
            "xy_after_mean_bias_removed_mm": float(centered.mean()),
            "z_mae_mm": float(np.abs(pred[:, 2] - truth[:, 2]).mean())}


def circle_shape(pred: np.ndarray, cmd: np.ndarray) -> dict:
    center = np.mean(pred[:, :2], axis=0)
    radius = np.linalg.norm(pred[:, :2] - center, axis=1)
    angle = np.arctan2(cmd[:, 1], cmd[:, 0])
    design = np.column_stack((np.ones(len(angle)),
                              np.cos(angle), np.sin(angle),
                              np.cos(2 * angle), np.sin(2 * angle),
                              np.cos(4 * angle), np.sin(4 * angle)))
    coef = np.linalg.lstsq(design, radius, rcond=None)[0]
    harmonic4 = float(np.linalg.norm(coef[-2:]))
    return {"center_xy_mm": center.tolist(),
            "radius_mean_mm": float(radius.mean()),
            "radius_std_mm": float(radius.std()),
            "radius_p05_p95_mm": np.percentile(radius, [5, 95]).tolist(),
            "fourfold_amplitude_mm": harmonic4,
            "fourfold_over_mean_radius_pct": float(100 * harmonic4 / radius.mean()),
            "twofold_amplitude_mm": float(np.linalg.norm(coef[3:5]))}


def best_lag(t_ns: np.ndarray, command: np.ndarray,
             output: np.ndarray, mask: np.ndarray) -> dict:
    t = (t_ns - t_ns[0]) / 1e9
    candidates = []
    for delay in np.arange(-.6, .601, .025):
        query = t[mask] - delay
        if query.min() < t[0] or query.max() > t[-1]:
            continue
        reference = np.column_stack((np.interp(query, t, command[:, 0]),
                                     np.interp(query, t, command[:, 1])))
        error = output[mask, :2] - reference
        bias = error.mean(axis=0)
        rmse = float(np.sqrt(np.mean(np.sum((error - bias) ** 2, axis=1))))
        candidates.append((rmse, delay, bias))
    if not candidates:
        return {}
    best = min(candidates, key=lambda item: item[0])
    return {"lag_s_positive_means_prediction_trails": float(best[1]),
            "bias_corrected_xy_rmse_mm": best[0],
            "mean_xy_bias_mm": best[2].tolist()}


def jumps(pred: np.ndarray, base: np.ndarray, cmd: np.ndarray,
          magnetic: np.ndarray,
          t_ns: np.ndarray, eligible: np.ndarray,
          phases: np.ndarray, extra: list[dict]) -> dict:
    pair = eligible[1:] & eligible[:-1]
    dt = np.diff(t_ns) / 1e9
    pair &= (dt > 0) & (dt < .12)
    delta_pred = np.linalg.norm(np.diff(pred[:, :2], axis=0), axis=1)
    delta_base = np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1)
    delta_cmd = np.linalg.norm(np.diff(cmd[:, :2], axis=0), axis=1)
    delta_magnetic = np.linalg.norm(np.diff(magnetic, axis=0), axis=1)
    selected = np.flatnonzero(pair)
    top = selected[np.argsort(delta_pred[selected])[-8:][::-1]]
    return {"pairs": int(len(selected)),
            "over_0p5mm": int((delta_pred[selected] > .5).sum()),
            "over_1mm": int((delta_pred[selected] > 1).sum()),
            "base_over_1mm": int((delta_base[selected] > 1).sum()),
            "memory_only_over_1mm": int((
                (delta_pred[selected] > 1) & (delta_base[selected] <= 1)).sum()),
            "base_only_over_1mm": int((
                (delta_base[selected] > 1) & (delta_pred[selected] <= 1)).sum()),
            "prediction_max_step_mm": float(np.max(delta_pred[selected])),
            "magnetic_step_median_uT": float(np.median(delta_magnetic[selected])),
            "magnetic_step_p95_uT": float(np.percentile(delta_magnetic[selected], 95)),
            "top": [{"index": int(i + 1), "time_s": float((t_ns[i + 1] - t_ns[0]) / 1e9),
                     "phase": str(phases[i + 1]), "dt_s": float(dt[i]),
                     "prediction_step_mm": float(delta_pred[i]),
                     "static_map_step_mm": float(delta_base[i]),
                     "command_step_mm": float(delta_cmd[i]),
                     "magnetic_15d_step_uT": float(delta_magnetic[i]),
                     "amplitude_uT": extra[i + 1].get("amplitude_uT"),
                     "memory_state": extra[i + 1].get("memory_state")}
                    for i in top]}


def return_center_holds(pred: np.ndarray, base: np.ndarray,
                        ref: dict, phases: np.ndarray,
                        valid: np.ndarray) -> dict:
    rows = []
    for segment in np.unique(ref["command_segment_id"]):
        if segment < 0:
            continue
        phase = phases[np.flatnonzero(ref["command_segment_id"] == segment)[0]]
        if not phase.startswith("angle_return_"):
            continue
        same = (ref["command_segment_id"] == segment) & valid & ~ref["command_moving"]
        early = same & (ref["hold_age_s"] >= .05) & (ref["hold_age_s"] <= .20)
        late = same & (ref["hold_age_s"] >= .35) & (ref["hold_age_s"] <= .60)
        if early.sum() < 3 or late.sum() < 3:
            continue
        rows.append({"phase": phase, "early_n": int(early.sum()),
                     "late_n": int(late.sum()),
                     "memory_early_to_late_xy_mm":
                         (pred[late, :2].mean(axis=0) -
                          pred[early, :2].mean(axis=0)).tolist(),
                     "base_early_to_late_xy_mm":
                         (base[late, :2].mean(axis=0) -
                          base[early, :2].mean(axis=0)).tolist()})
    memory_norm = [np.linalg.norm(row["memory_early_to_late_xy_mm"]) for row in rows]
    base_norm = [np.linalg.norm(row["base_early_to_late_xy_mm"]) for row in rows]
    return {"holds": rows,
            "memory_median_shift_mm": float(np.median(memory_norm)) if rows else None,
            "base_median_shift_mm": float(np.median(base_norm)) if rows else None}


def trial(number: int) -> dict:
    paths = list(SESSION.glob(f"Trial_{number:06d}_*"))
    if len(paths) != 1:
        raise ValueError(paths)
    path = paths[0]
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if (meta["status"] != "completed" or
            meta["acquisition"]["magnetic_layer_profile_id"] !=
            "magnetic_layer_v1_5*5-had-239"):
        raise ValueError("different/incomplete physical profile")
    segments = command_segments(read_rows(path / "stage_commands.csv"))
    frame = pd.read_csv(path / "live_predictions.csv", low_memory=False)
    t_ns = frame["t_prediction_ns"].to_numpy(np.int64)
    magnetic_frame = pd.read_csv(path / "magnetic_raw.csv", low_memory=False)
    magnetic_t = magnetic_frame["t_group_ns"].to_numpy(np.int64)
    magnetic_columns = [f"S{sensor}_{axis}_uT" for sensor in range(1, 6)
                        for axis in ("Bx", "By", "Bz")]
    magnetic_raw = magnetic_frame[magnetic_columns].to_numpy(float)
    magnetic_lookup = np.searchsorted(magnetic_t, t_ns)
    magnetic_match = magnetic_lookup < len(magnetic_t)
    magnetic_match[magnetic_match] &= (
        magnetic_t[magnetic_lookup[magnetic_match]] == t_ns[magnetic_match])
    magnetic = np.full((len(t_ns), 15), np.nan)
    magnetic[magnetic_match] = magnetic_raw[magnetic_lookup[magnetic_match]]
    ref = command_reference(t_ns, segments, reference_origin(meta))
    truth = ref["xyz_mm"].astype(float)
    truth[:, 2] += TRAIN_CONTACT_Z - float(reference_origin(meta)[2])
    phases = np.asarray([segments[i]["phase"] if i >= 0 else "before_stage"
                         for i in ref["command_segment_id"]])
    pred = frame[["visual_x_mm", "visual_y_mm", "visual_z_mm"]].to_numpy(float)
    valid = frame["interaction_valid"].to_numpy(bool) & np.isfinite(pred).all(axis=1)
    extra = [json.loads(value) for value in frame["extra_outputs_json"]]
    base = np.full_like(pred, np.nan)
    for i, row in enumerate(extra):
        raw = row.get("uncorrected_static_xyz_mm")
        if not valid[i] or not isinstance(raw, list) or len(raw) != 3:
            continue
        base[i] = np.asarray(raw, float)
        angle = np.deg2rad(float(row.get("xy_rotation_deg", 0)))
        c, s = np.cos(angle), np.sin(angle)
        base[i, :2] = np.array([[c, -s], [s, c]]) @ base[i, :2] + np.asarray(
            row.get("xy_translation_mm", [0., 0.]), float)
    valid &= np.isfinite(base).all(axis=1)
    core = valid & (ref["contact_command"] == 1) & ref["command_moving"]
    if number in (12, 13):
        core &= np.asarray([phase.startswith("angle_") for phase in phases])
    else:
        core &= np.asarray([phase.startswith("circle_") and
                            phase != "circle_start" for phase in phases])
    if core.sum() < 50:
        raise ValueError(f"too few core predictions in Trial {number}")
    result = {"trial": number, "source": path.relative_to(ROOT).as_posix(),
              "meta_sha256": sha(path / "meta.json"),
              "prediction_sha256": sha(path / "live_predictions.csv"),
              "magnetic_sha256": sha(path / "magnetic_raw.csv"),
              "stage_sha256": sha(path / "stage_commands.csv"),
              "command_mode": meta["trial_parameters"]["command_mode"],
              "contact_z_mm": float(reference_origin(meta)[2]),
              "training_equivalent_depth_mm": float(
                  meta["trial_parameters"]["z_indent_mm"] +
                  TRAIN_CONTACT_Z - reference_origin(meta)[2]),
              "prediction_rows": int(len(frame)),
              "raw_magnetic_timestamp_match_rows": int(magnetic_match.sum()),
              "valid_rows": int(valid.sum()),
              "status_counts": dict(Counter(row.get("status") for row in extra)),
              "core_moving_rows": int(core.sum()),
              "static_map": xy_summary(base[core], truth[core]),
              "memory_model": xy_summary(pred[core], truth[core]),
              "memory_correction_xy_median_mm": float(np.median(
                  np.linalg.norm((pred - base)[core, :2], axis=1))),
              "memory_correction_xy_p95_mm": float(np.percentile(
                  np.linalg.norm((pred - base)[core, :2], axis=1), 95)),
              "jumps": jumps(pred, base, truth, magnetic, t_ns, core,
                             phases, extra)}
    if number in (12, 13):
        result["return_holds"] = return_center_holds(pred, base, ref, phases, valid)
    else:
        circle_path = (ref["contact_command"] == 1) & np.asarray(
            [bool(re.fullmatch(r"circle_\d+", phase)) for phase in phases])
        all_path = valid & circle_path
        count = max(int(match.group(1)) for segment in segments
                    if (match := re.fullmatch(r"circle_(\d+)", segment["phase"])))
        start = next(segment for segment in segments
                     if segment["phase"] == "circle_start")
        radius = float(np.linalg.norm(np.asarray(start["to_mm"][:2]) -
                                      np.asarray(reference_origin(meta)[:2])))
        result["circle_shape"] = {
            "command": circle_shape(truth[core], truth[core]),
            "static_map": circle_shape(base[core], truth[core]),
            "memory_model": circle_shape(pred[core], truth[core]),
            "circle_waypoint_count": count,
            "command_radius_mm": radius,
            "polyline_sag_mm": float(radius * (1 - np.cos(np.pi / count)))}
        result["circle_all_contact"] = {
            "rows": int(all_path.sum()),
            "path_frames": int(circle_path.sum()),
            "invalid_path_frames": int((circle_path & ~valid).sum()),
            "static_map": xy_summary(base[all_path], truth[all_path]),
            "memory_model": xy_summary(pred[all_path], truth[all_path]),
            "static_shape": circle_shape(base[all_path], truth[all_path]),
            "memory_shape": circle_shape(pred[all_path], truth[all_path])}
        result["lag"] = {
            "static_map": best_lag(t_ns, truth, base, core),
            "memory_model": best_lag(t_ns, truth, pred, core)}
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    results = [trial(number) for number in TRIALS]
    report = {"schema": "MagTouch-R130-LiveFourTrialDiagnostic-4",
              "supersedes": "report_v3.json: adds decoder-only jump counts and circle gate coverage; report.json has incorrect Trial 14 polygon sag",
              "warning": "stage command interpolation is not independently measured tip path; XY/Z model versus command only; no model tuning",
              "trials": results}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    for row in results:
        print("[R130]", row["trial"], "core", row["core_moving_rows"],
              "XY", round(row["static_map"]["xy_mean_mm"], 3), "->",
              round(row["memory_model"]["xy_mean_mm"], 3),
              "jumps>1", row["jumps"]["over_1mm"])
    print(OUT)


if __name__ == "__main__":
    main()
