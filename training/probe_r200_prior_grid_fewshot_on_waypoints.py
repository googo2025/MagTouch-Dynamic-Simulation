#!/usr/bin/env python3
"""Retrospective, read-only 1/9-point target calibration on frozen 3D paths.

The only labelled calibration source is the target-device stationary grid from
2026-09-27.  The evaluation paths were recorded on 2026-09-28.  The frozen
source-device atlas, memory, gate and R133 XY weights are not refitted.  No
evaluation labels are used to fit a correction.  Since these paths have
already been inspected, the result is exploratory, not prospective evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.audit_r133_online_20260928 import make_atlas, score
from training.had239_r133_coverage_xy_runtime import ARTIFACT, CoverageCausalXYMLP
from training.probe_had239_local_forward_r104 import ROOT, sha
from training.retrain_r133_coverage_xy import features, load_new


OUT = ROOT / "analysis/r200_prior_grid_fewshot_on_waypoints_20260928/report.json"
CALIBRATION_DIR = (ROOT / "data/S20260927_A" /
    "Plan_20260927_230418_R133_F_YELLOW_UNSEEN_COMBINATIONS__mag_5x5-had-Yelow")
EVALUATION_DIR = (ROOT / "data/S20260928_A" /
    "Plan_20260928_130827_R136_YELLOW_R133_WAYPOINT_UNSEEN_3D__mag_5x5-had-Yelow")
AXES = {1: (0.0,), 9: (-4.0, 0.0, 4.0)}


def frozen_r133(data, atlas, model):
    pred = data["guarded"].copy()
    valid = data["valid"]
    pred[valid, :2] += model.predict(features(data, atlas)[valid])
    return pred


def fit_grid(data, prediction, axes):
    """Median command-minus-model XYZ at fixed, nested grid locations."""
    grid = np.empty((len(axes), len(axes), 3), float)
    n = np.empty((len(axes), len(axes)), int)
    for i, x in enumerate(axes):
        for j, y in enumerate(axes):
            use = (data["core"] & np.isclose(data["truth"][:, 0], x, atol=1e-6)
                   & np.isclose(data["truth"][:, 1], y, atol=1e-6))
            if use.sum() < 5:
                raise ValueError(f"calibration point {(x, y)} has only {use.sum()} valid frames")
            grid[i, j] = np.median(data["truth"][use] - prediction[use], axis=0)
            n[i, j] = int(use.sum())
    return grid, n


def interpolate_at_predicted_xy(xy, axes, grid):
    """A deployable prediction-coordinate lookup; clamp outside support."""
    xy = np.asarray(xy, float)
    if len(axes) == 1:
        return np.repeat(grid[0, 0][None, :], len(xy), axis=0), np.ones(len(xy), bool)
    axis = np.asarray(axes, float)
    inside = np.all((xy >= axis[0]) & (xy <= axis[-1]), axis=1)
    clipped = np.clip(xy, axis[0], axis[-1])
    i = np.clip(np.searchsorted(axis, clipped[:, 0], side="right") - 1, 0, len(axis) - 2)
    j = np.clip(np.searchsorted(axis, clipped[:, 1], side="right") - 1, 0, len(axis) - 2)
    tx = ((clipped[:, 0] - axis[i]) / (axis[i + 1] - axis[i]))[:, None]
    ty = ((clipped[:, 1] - axis[j]) / (axis[j + 1] - axis[j]))[:, None]
    delta = ((1 - tx) * (1 - ty) * grid[i, j] + tx * (1 - ty) * grid[i + 1, j]
             + (1 - tx) * ty * grid[i, j + 1] + tx * ty * grid[i + 1, j + 1])
    return delta, inside


def main():
    if OUT.exists():
        raise FileExistsError(f"Preserve versioned report: {OUT}")
    grid_path = next(CALIBRATION_DIR.glob("Trial_000133_*"))
    paths = [next(EVALUATION_DIR.glob(f"Trial_{number:06d}_*")) for number in (26, 27)]
    atlas, basis, strength = make_atlas()
    model = CoverageCausalXYMLP()
    calibration = load_new(grid_path, atlas, basis, strength)
    grid_pred = frozen_r133(calibration, atlas, model)
    maps = {}
    for budget, axes in AXES.items():
        residual, counts = fit_grid(calibration, grid_pred, axes)
        maps[budget] = residual
        print("calibration", budget, "point-frame counts", counts.tolist(), flush=True)
    trial_rows = []
    for path in paths:
        data = load_new(path, atlas, basis, strength, min_core_frames=0)
        core = data["core"]
        original = data["live"][core]
        truth = data["truth"][core]
        row = {"trial": path.name, "reference_frames": int(data["core_ref"].sum()),
               "valid_frames": int(core.sum()), "coverage": float(core.sum()/data["core_ref"].sum()),
               "saved_online_zero_shot": score(original, truth), "adaptations": {}}
        for budget, axes in AXES.items():
            delta, inside = interpolate_at_predicted_xy(original[:, :2], axes, maps[budget])
            row["adaptations"][str(budget)] = {
                "score": score(original + delta, truth),
                "prediction_inside_calibration_xy_fraction": float(inside.mean()),
            }
        trial_rows.append(row)
        print(path.name[:12], "zero", round(row["saved_online_zero_shot"]["xy_raw_mean_mm"], 3),
              "one", round(row["adaptations"]["1"]["score"]["xy_raw_mean_mm"], 3),
              "nine", round(row["adaptations"]["9"]["score"]["xy_raw_mean_mm"], 3), flush=True)
    keys = ("xy_raw_mean_mm", "xy_translation_removed_mean_mm", "z_raw_mae_mm",
            "z_translation_removed_mae_mm")
    aggregate = {}
    for variant in ("zero", "1", "9"):
        rows = [r["saved_online_zero_shot"] if variant == "zero"
                else r["adaptations"][variant]["score"] for r in trial_rows]
        aggregate[variant] = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    report = {
        "schema": "MagTouch-R200-PriorGridFewShotFrozenWaypoints-1",
        "status": "retrospective exploratory; previously viewed test paths; no UI or source weights changed",
        "calibration": {"trial": grid_path.name, "date": "2026-09-27", "depth_mm": 1.45,
                        "raw_sha256": sha(grid_path / "magnetic_raw.csv"),
                        "frozen_xy_weights_sha256": sha(ARTIFACT),
                        "labels": "platform-command XYZ at stationary grid holds",
                        "model": "frozen source-device R133 applied offline to target calibration grid",
                        "budgets": {str(k): {"axes_mm": list(AXES[k]), "xyz_correction_mm": maps[k].tolist()}
                                    for k in AXES}},
        "evaluation": {"date": "2026-09-28", "paths": [p.name for p in paths],
                       "raw_sha256": [sha(p / "magnetic_raw.csv") for p in paths],
                       "model": "saved online source-device R133 zero-shot outputs, plus read-only correction",
                       "rule": "1 center constant or 9-position bilinear XY-conditioned XYZ residual; fixed points, no test fitting; gate unchanged",
                       "reference": "time-interpolated platform commands, not independently observed contact XYZ",
                       "trial_equal": aggregate, "trials": trial_rows},
        "caution": "Not a new blind test. No 9/28 test labels were used for calibration, but model choice was motivated after observing the test. A new held-out run is needed to validate the selected budget.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
