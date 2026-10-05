#!/usr/bin/env python3
"""Retrain the frozen R130 XY architecture on R133-D, evaluate E/F once.

The atlas, memory, branch guard, gate, causal features, MLP architecture and
training objective are unchanged. This script never imports the acquisition UI.
E/F are replay-only evaluation, not online results or a fresh blind test.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from training.build_trajectory_dataset_v1 import (
    baseline_schedule, command_reference, command_segments, load_magnetic,
    read_rows, reference_origin,
)
from training.diagnose_r126_latest_four_live_r130 import circle_shape, xy_summary
from training.evaluate_branch_inverse_r139 import FIVE_CONFIG, FIVE_KEY, FIVE_REPORT
from training.evaluate_guarded_branch_r163 import load as load_old
from training.had239_causal_xy_features_r165 import offline_features
from training.had239_causal_xy_mlp_runtime_r165 import CausalXYMLP
from training.probe_causal_xy_neural_r164 import MLP, guarded_positions
from training.probe_had239_field_memory_r122 import feature_to_field
from training.probe_had239_local_forward_r104 import ForwardAtlas, ROOT, load_profile, sha


OUT = ROOT / "analysis/r133_coverage_retrain_20260927"
BASE_TRAIN = (11, 12, 13, 14, 15, 24, 25, 26)
BASE_VALID = (27, 28, 29)
D = tuple(range(86, 110))
E = tuple(range(112, 122))
F = tuple(range(124, 134))
GROUP = {
    "D": "Plan_20260927_213250_R133_D_HAD239_COVERAGE_TRAIN__mag_5x5-had-239",
    "E": "Plan_20260927_222827_R133_E_HAD239_UNSEEN_COMBINATIONS__mag_5x5-had-239",
    "F": "Plan_20260927_230418_R133_F_YELLOW_UNSEEN_COMBINATIONS__mag_5x5-had-Yelow",
}


def new_path(group: str, number: int) -> Path:
    match = list((ROOT / "data/S20260927_A" / GROUP[group]).glob(f"Trial_{number:06d}_*"))
    if len(match) != 1:
        raise ValueError((group, number, len(match)))
    return match[0]


def recorded_bool_mask(values):
    """Parse a saved CSV boolean without Python's bool('False') pitfall."""
    text = pd.Series(values).astype(str).str.strip().str.lower()
    if not bool(text.isin(("true", "false", "1", "0")).all()):
        raise ValueError("invalid recorded interaction_valid value")
    return text.isin(("true", "1")).to_numpy(bool)


def load_new(path: Path, atlas: ForwardAtlas, basis: np.ndarray, strength: float,
             *, min_core_frames: int = 20):
    # Training/evaluation keeps the historical >=20-frame guard by default.
    # Prospective R136 completeness audits explicitly pass zero so a recorded
    # but wholly rejected Trial remains visible as zero coverage.
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(path / "live_predictions.csv", low_memory=False)
    if frame.empty:
        raise ValueError(f"no saved prediction frames: {path.name}")
    t = frame.t_prediction_ns.to_numpy(np.int64)
    raw = load_magnetic(path / "magnetic_raw.csv")
    segments = command_segments(read_rows(path / "stage_commands.csv"))
    events = read_rows(path / "events.csv")
    ordered = np.argsort(raw["t_group_ns"])
    raw_t = raw["t_group_ns"][ordered]
    index = np.searchsorted(raw_t, t)
    matched = index < len(raw_t)
    matched[matched] &= raw_t[index[matched]] == t[matched]
    if matched.mean() < .98:
        raise ValueError(f"timestamp match {path.name}: {matched.mean():.3f}")
    measured = raw["b_uT"][ordered]
    observed = np.full((len(t), 15), np.nan)
    baselines = baseline_schedule(events, raw, segments, reference_origin(meta))
    if not baselines:
        raise ValueError(f"no B0: {path.name}")
    for baseline in baselines:
        use = matched & (t >= baseline["effective_ns"])
        observed[use] = measured[index[use]] - baseline["b0_uT"]
    extra = [json.loads(v) for v in frame.extra_outputs_json]
    live = frame[["visual_x_mm", "visual_y_mm", "visual_z_mm"]].to_numpy(float)
    valid = recorded_bool_mask(frame.interaction_valid) & matched & np.isfinite(live).all(axis=1)
    valid &= np.isfinite(observed).all(axis=1)
    memory = np.full((len(t), 2), np.nan)
    guard = np.full_like(live, np.nan)
    logged_correction = np.full((len(t), 2), np.nan)
    for i, value in enumerate(extra):
        if not valid[i]:
            continue
        state = value.get("memory_state")
        correction = value.get("xy_neural_correction_mm")
        if isinstance(state, list) and len(state) == 2 and isinstance(correction, list) and len(correction) == 2:
            memory[i] = state
            logged_correction[i] = correction
            guard[i] = live[i]
            guard[i, :2] -= correction
    valid &= np.isfinite(memory).all(axis=1) & np.isfinite(guard).all(axis=1)
    corrected = np.full_like(observed, np.nan)
    corrected_phi = atlas.features(observed[valid]) - strength * memory[valid] @ basis.T
    corrected[valid] = feature_to_field(atlas, corrected_phi)
    ref = command_reference(t, segments, reference_origin(meta))
    truth = ref["xyz_mm"].astype(float)
    phases = np.asarray([segments[j]["phase"] if j >= 0 else "before_stage"
                         for j in ref["command_segment_id"]])
    mode = meta["trial_parameters"]["command_mode"]
    if mode == "angle_scan":
        moving_phase = np.asarray([p.startswith("angle_") for p in phases])
    elif mode == "circle_scan":
        moving_phase = np.asarray([bool(re.fullmatch(r"circle_\d+", p)) for p in phases])
    elif mode == "grid_points":
        moving_phase = (phases == "grid_indent") & ~ref["command_moving"] & (
            ref["hold_age_s"] >= .15) & (ref["hold_age_s"] <= .55)
    elif mode == "compound":
        moving_phase = phases == "xyz_compound"
    elif mode == "z_path":
        moving_phase = np.asarray([bool(re.fullmatch(r"path_\d+", p)) for p in phases])
    elif mode == "waypoint_xyz":
        # R136 records each genuine nonplanar G1 segment separately. Select
        # only within-contact motion; approach, dwell, lift and recenter are
        # never scored as localization reference frames.
        moving_phase = np.asarray([bool(re.fullmatch(r"waypoint_\d{3}", p))
                                   for p in phases])
    else:
        raise ValueError((mode, path))
    core_ref = moving_phase & (ref["contact_command"] == 1) & np.isfinite(truth).all(axis=1)
    if mode != "grid_points":
        core_ref &= ref["command_moving"]
    core = valid & core_ref
    if core.sum() < min_core_frames:
        raise ValueError(f"too few core frames {path.name}: {core.sum()}")
    data = {"path": path, "meta": meta, "time": t, "valid": valid, "core": core,
            "core_ref": core_ref, "corrected": corrected, "guarded": guard,
            "live": live, "truth": truth, "memory": memory, "mode": mode,
            "logged_correction": logged_correction, "observed": observed}
    return data


def features(data, atlas):
    if "features" not in data:
        data["features"] = offline_features(data, atlas)
    return data["features"]


def evaluate(model, data, atlas):
    x = features(data, atlas)
    core = data["core"]
    guard = data["guarded"]
    pred = guard.copy()
    pred[core, :2] += model.predict(x[core])
    truth = data["truth"]
    old = data["live"]
    valid_ref = data.get("core_ref", core)
    old_xy = np.linalg.norm(old[core, :2] - truth[core, :2], axis=1)
    new_xy = np.linalg.norm(pred[core, :2] - truth[core, :2], axis=1)
    z = np.abs(old[core, 2] - truth[core, 2])
    out = {"trial": int(data["path"].name.split("_")[1]), "mode": data["mode"],
           "core_reference_frames": int(valid_ref.sum()), "core_valid_frames": int(core.sum()),
           "coverage": float(core.sum()/max(valid_ref.sum(), 1)),
           "old_xy_mean_mm": float(old_xy.mean()), "new_xy_mean_mm": float(new_xy.mean()),
           "guard_xy_mean_mm": float(np.linalg.norm(guard[core, :2] - truth[core, :2], axis=1).mean()),
           "old_xy_p95_mm": float(np.percentile(old_xy, 95)),
           "new_xy_p95_mm": float(np.percentile(new_xy, 95)),
           "z_mean_mm": float(z.mean()),
           "xy_bias_old_mm": np.mean(old[core, :2] - truth[core, :2], axis=0).tolist(),
           "xy_bias_new_mm": np.mean(pred[core, :2] - truth[core, :2], axis=0).tolist(),
           "command": {k: data["meta"]["trial_parameters"].get(k) for k in (
               "depth_mm", "indent_z_mm", "speed_mm_s", "circle_radius_mm", "dx_mm")},
           "path": data["path"].relative_to(ROOT).as_posix()}
    if data["mode"] == "circle_scan":
        out["old_circle"] = circle_shape(old[core], truth[core])
        out["new_circle"] = circle_shape(pred[core], truth[core])
    return out


def main():
    report_path, artifact_path, metadata_path = (
        OUT / "report.json", OUT / "causal_xy_model.npz", OUT / "causal_xy_model.json")
    if any(p.exists() for p in (report_path, artifact_path, metadata_path)):
        raise FileExistsError("R133 retraining outputs already exist; preserve frozen results")
    profile = load_profile(FIVE_CONFIG, FIVE_KEY)
    samples = profile["samples"]
    xyz = np.asarray([[s["x"], s["y"], s["depth"]] for s in samples])
    delta = np.asarray([s["delta"] for s in samples])
    atlas = ForwardAtlas(xyz, delta, "unit_logamp")
    memory_report = json.loads(FIVE_REPORT.read_text(encoding="utf-8"))
    basis = np.asarray(memory_report["basis"], float)
    strength = float(memory_report["selection"]["selected"]["strength"])
    old_model = CausalXYMLP()
    base = {}
    for n in BASE_TRAIN + BASE_VALID:
        row = load_old(n, atlas, basis, strength)
        row["guarded"] = guarded_positions(row, atlas)
        base[n] = row
        print("loaded original", n, int(row["core"].sum()), flush=True)
    new = {}
    for group, numbers in (("D", D),):
        for n in numbers:
            row = load_new(new_path(group, n), atlas, basis, strength)
            new[n] = row
            x = features(row, atlas)
            recovered = old_model.predict(x[row["core"]])
            parity = np.linalg.norm(recovered - row["logged_correction"][row["core"]], axis=1)
            row["feature_parity_median_mm"] = float(np.median(parity))
            row["feature_parity_p95_mm"] = float(np.percentile(parity, 95))
            if row["feature_parity_median_mm"] > .03 or row["feature_parity_p95_mm"] > .12:
                raise ValueError(f"offline feature parity failed Trial {n}: {row['feature_parity_median_mm']:.3f}/{row['feature_parity_p95_mm']:.3f}")
            print("loaded", group, n, int(row["core"].sum()), "parity", round(row["feature_parity_median_mm"], 4), flush=True)
    train_x, train_y, sample_count = [], [], {}
    for n in BASE_TRAIN + D:
        row = base[n] if n in base else new[n]
        choose = np.flatnonzero(row["core"])[::4]
        x = features(row, atlas)[choose]
        y = np.clip(row["truth"][choose, :2] - row["guarded"][choose, :2], -3., 3.)
        train_x.append(x)
        train_y.append(y)
        sample_count[str(n)] = len(choose)
    static = np.column_stack((xyz[::3] / np.array([10., 10., 3.]),
                              atlas.features(delta[::3]),
                              np.zeros((len(xyz[::3]), 34))))
    train_x.append(static)
    train_y.append(np.zeros((len(static), 2)))
    x = np.vstack(train_x)
    y = np.vstack(train_y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite training sample")
    print("training", len(x), "real", len(x)-len(static), "static", len(static), flush=True)
    model = MLP(64, seed=7).fit(x, y, [base[n] for n in BASE_VALID], atlas)
    print("selected epoch", model.best_epoch, "original validation", model.best_validation, flush=True)
    # Freeze the exact selected weights before reading E/F performance.
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(artifact_path, mean=model.mean, scale=model.scale,
                        w1=model.w1, b1=model.b1, w2=model.w2, b2=model.b2,
                        w3=model.w3, b3=model.b3)
    metadata = {"schema": "MagTouch-R133-CoverageRetrainMLP-1",
                "architecture": "53-64-64-2 tanh causal XY residual; same R130 feature and MSE+0.001 weight regularization",
                "artifact_sha256": sha(artifact_path), "selected_width": 64,
                "selected_epoch": model.best_epoch, "seed": 7,
                "train_original_trials": BASE_TRAIN, "train_D_trials": D,
                "validation_original_trials": BASE_VALID, "evaluation_E_trials": E,
                "evaluation_F_trials": F, "static_anchor_samples": len(static),
                "sample_count_by_trial": sample_count,
                "static_config_sha256": sha(FIVE_CONFIG),
                "memory_report_sha256": sha(FIVE_REPORT),
                "warning": "Offline replay only. E/F recordings were inspected before this retraining and are not blind; no E/F label used in optimization or epoch selection. Not deployed in UI."}
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    for group, numbers in (("E", E), ("F", F)):
        for n in numbers:
            row = load_new(new_path(group, n), atlas, basis, strength)
            new[n] = row
            x_eval = features(row, atlas)
            recovered = old_model.predict(x_eval[row["core"]])
            parity = np.linalg.norm(recovered - row["logged_correction"][row["core"]], axis=1)
            row["feature_parity_median_mm"] = float(np.median(parity))
            row["feature_parity_p95_mm"] = float(np.percentile(parity, 95))
            if row["feature_parity_median_mm"] > .03 or row["feature_parity_p95_mm"] > .12:
                raise ValueError(f"offline feature parity failed Trial {n}: {row['feature_parity_median_mm']:.3f}/{row['feature_parity_p95_mm']:.3f}")
            print("loaded", group, n, int(row["core"].sum()), "parity", round(row["feature_parity_median_mm"], 4), flush=True)
    results = {}
    for group, numbers in (("D", D), ("E", E), ("F", F)):
        rows = [evaluate(model, new[n], atlas) for n in numbers]
        results[group] = rows
        for row in rows:
            print(group, row["trial"], row["mode"], "old", round(row["old_xy_mean_mm"], 3),
                  "new", round(row["new_xy_mean_mm"], 3), "coverage", round(row["coverage"], 3), flush=True)
    report = {"schema": "MagTouch-R133-CoverageRetrainReplay-1", "metadata": metadata,
              "validation_history": model.history,
              "input_sha256": {str(n): sha(new[n]["path"] / "live_predictions.csv") for n in new},
              "feature_parity": {str(n): {"median_mm": new[n]["feature_parity_median_mm"],
                                         "p95_mm": new[n]["feature_parity_p95_mm"]} for n in new},
              "results": results}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
