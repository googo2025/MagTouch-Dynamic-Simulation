#!/usr/bin/env python3
"""Device-specific shallow inverse probe, trained only on the earlier R133 grid.

This probes *whether* the weak-field frames rescued by causal history contain
enough spatial information. Commanded grid XY/Z are provisional supervision;
no fitted model here is deployed or described as an independent ground truth.
R208 was seen during development and is a retrospective transfer check.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from training.audit_r226_shallow_history_gate import (
    ROOT, _fit_ridge, _score, _sequential_gate, _threshold,
)
from training.audit_r227_assisted_contact_gate import load_rows, vote
from training.build_trajectory_dataset_v1 import (
    command_reference, command_segments, read_rows, reference_origin,
)
from training.probe_r200_prior_grid_fewshot_on_waypoints import interpolate_at_predicted_xy

OUT = ROOT / "analysis/r228_shallow_inverse_20260928/report_v6.json"
PRIOR_GRID = ROOT / "analysis/r200_prior_grid_fewshot_on_waypoints_20260928/report.json"
FEATURE = "history_vector"
LAMBDAS = (.01, .1, 1.0)
KERNEL_OPTIONS = ((.5, .02), (1.0, .02), (.5, .2), (1.0, .2))


def reference(row):
    meta = json.loads((row["path"] / "meta.json").read_text(encoding="utf-8"))
    commands = command_segments(read_rows(row["path"] / "stage_commands.csv"))
    return command_reference(row["time"], commands, reference_origin(meta))["xyz_mm"]


def fit(x, y, lam):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .05)
    v = (x-mean)/scale
    design = np.column_stack((np.ones(len(v)), v))
    reg = np.eye(design.shape[1])*lam
    reg[0, 0] = 0.
    beta = np.linalg.solve(design.T@design/len(x)+reg,
                           design.T@y/len(x))
    return mean, scale, beta


def infer(model, x):
    mean, scale, beta = model
    return np.column_stack((np.ones(len(x)), (x-mean)/scale))@beta


def squared_distances(a, b):
    return np.maximum(np.sum(a*a, axis=1)[:, None] +
                      np.sum(b*b, axis=1)[None, :] - 2*a@b.T, 0.)


def kernel_fit(x, y, gamma_factor, lam, sample_weight=None):
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
    v = (x-mean)/scale
    d = squared_distances(v, v)
    upper = d[np.triu_indices(len(d), 1)]
    median = float(np.median(upper[upper > 0]))
    gamma = gamma_factor/max(median, 1e-9)
    gram = np.exp(-gamma*d)
    weight = (np.ones(len(x)) if sample_weight is None else
              np.asarray(sample_weight, float))
    if weight.shape != (len(x),) or np.any(weight <= 0):
        raise ValueError("kernel sample weights must be positive and match rows")
    center = np.average(y, axis=0, weights=weight)
    alpha = np.linalg.solve(gram + np.diag(lam/weight), y-center)
    return mean, scale, v, gamma, center, alpha


def kernel_infer(model, x):
    mean, scale, train, gamma, center, alpha = model
    v = (x-mean)/scale
    return center + np.exp(-gamma*squared_distances(v, train))@alpha


def score(pred, truth):
    delta = pred-truth
    err = np.linalg.norm(delta[:,:2], axis=1)
    bias = delta.mean(axis=0)
    debiased = np.linalg.norm(delta[:,:2]-bias[:2], axis=1)
    zerr = np.abs(pred[:,2]-truth[:,2])
    return {"n": len(err), "xy_median_mm": float(np.median(err)),
            "xy_p90_mm": float(np.percentile(err, 90)),
            "mean_xyz_bias_mm": bias.tolist(),
            "diagnostic_xy_debiased_median_mm": float(np.median(debiased)),
            "diagnostic_xy_debiased_p90_mm": float(np.percentile(debiased, 90)),
            "z_mae_mm": float(np.mean(zerr)),
            "z_p90_mm": float(np.percentile(zerr, 90))}


def grid_data(row):
    truth = reference(row)
    take = row["eligible"] & (row["labels"] == "shallow_descent")
    x = row["features"][FEATURE][take]
    y = truth[take]
    finite = np.isfinite(y).all(axis=1)
    x, y = x[finite], y[finite]
    if len(y) < 100:
        raise ValueError(f"too few shallow grid frames: {row['path']}: {len(y)}")
    # A complete commanded XY grid point stays in one fold.
    xy_node = np.round(y[:,:2]).astype(int)
    groups = np.mod(3*xy_node[:,0] + 7*xy_node[:,1], 5)
    return x, y, groups


def select_lambda(x, y, groups):
    candidates = []
    for lam in LAMBDAS:
        fold = []
        for group in np.unique(groups):
            train, test = groups != group, groups == group
            if not train.any() or not test.any():
                continue
            pred = infer(fit(x[train], y[train], lam), x[test])
            fold.append(score(pred, y[test]))
        candidates.append({"lambda": lam,
                           "xy_macro_median_mm": float(np.mean([f["xy_median_mm"] for f in fold])),
                           "z_macro_mae_mm": float(np.mean([f["z_mae_mm"] for f in fold])),
                           "folds": fold})
    chosen = min(candidates, key=lambda item: item["xy_macro_median_mm"])
    return chosen["lambda"], candidates


def select_kernel(x, y, groups):
    candidates = []
    for gamma, lam in KERNEL_OPTIONS:
        fold = []
        for group in np.unique(groups):
            train, test = groups != group, groups == group
            pred = kernel_infer(kernel_fit(x[train], y[train], gamma, lam), x[test])
            fold.append(score(pred, y[test]))
        candidates.append({"gamma_factor": gamma, "lambda": lam,
                           "xy_macro_median_mm": float(np.mean([f["xy_median_mm"] for f in fold])),
                           "z_macro_mae_mm": float(np.mean([f["z_mae_mm"] for f in fold]))})
    return min(candidates, key=lambda item: item["xy_macro_median_mm"]), candidates


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = load_rows()
    onset_model = _fit_ridge(rows, FEATURE)
    threshold = _threshold(rows, onset_model)
    report = {"schema": "MagTouch-R228-ShallowInverseProbe-1",
              "status": "retrospective development; no UI or hardware changes",
              "training": "R133 grid nominal 0.02-0.2 mm descending frames, per device",
              "model": "40D causal-history ridge XYZ; regularization chosen by commanded XY-node grouped CV",
              "warning": "commanded depth is not independent physical contact; R208 was previously inspected",
              "devices": {}}
    for device in ("source", "target"):
        grids = [row for row in rows if row["device"] == device and
                 row["partition"] == "development" and row["mode"] == "grid_points"]
        if len(grids) != 1:
            raise ValueError(f"expected one grid for {device}, got {len(grids)}")
        x, y, groups = grid_data(grids[0])
        chosen, candidates = select_lambda(x, y, groups)
        shallow_model = fit(x, y, chosen)
        kernel_choice, kernel_candidates = select_kernel(x, y, groups)
        kernel_model = kernel_fit(x, y, kernel_choice["gamma_factor"],
                                  kernel_choice["lambda"])
        center = np.max(np.abs(y[:,:2]), axis=1) < .1
        if not center.any():
            raise ValueError("no prior-grid center shallow observations")
        prior_center_bias = (kernel_infer(kernel_model, x[center])[:,:2] -
                             y[center,:2]).mean(axis=0)
        eval_x, eval_y, onset_x, onset_y = [], [], [], []
        ui_x, ui_y, old_ui_pred = [], [], []
        selected_x, selected_y = [], []
        for row in rows:
            if row["device"] != device or row["partition"] != "evaluation":
                continue
            truth = reference(row) if row["mode"] != "none" else None
            if truth is None:
                continue
            shallow = row["eligible"] & (row["labels"] == "shallow_descent")
            if not shallow.any():
                continue
            score_onset = _score(row, onset_model)
            accepted = vote(score_onset, threshold, row["eligible"], required=2,
                            time=row["time"])
            state = _sequential_gate(row, score_onset, threshold)
            stage_gate = state | (row["ui_valid"] &
                                  (row["amplitude_uT"] >= 140.))
            rescued = shallow & accepted & ~row["ui_valid"]
            eval_x.append(row["features"][FEATURE][shallow])
            eval_y.append(truth[shallow])
            if rescued.any():
                onset_x.append(row["features"][FEATURE][rescued])
                onset_y.append(truth[rescued])
            saved = pd.read_csv(row["path"] / "live_predictions.csv",
                                usecols=["x", "y", "z"], low_memory=False)
            saved_xyz = saved[["x", "y", "z"]].to_numpy(float)
            valid_old = shallow & row["ui_valid"] & np.isfinite(saved_xyz).all(axis=1)
            if valid_old.any():
                ui_x.append(row["features"][FEATURE][valid_old])
                ui_y.append(truth[valid_old])
                old_ui_pred.append(saved_xyz[valid_old])
            selected = shallow & stage_gate
            if selected.any():
                selected_x.append(row["features"][FEATURE][selected])
                selected_y.append(truth[selected])
        eval_x, eval_y = np.vstack(eval_x), np.vstack(eval_y)
        onset_x, onset_y = np.vstack(onset_x), np.vstack(onset_y)
        ui_x, ui_y, old_ui_pred = np.vstack(ui_x), np.vstack(ui_y), np.vstack(old_ui_pred)
        selected_x, selected_y = np.vstack(selected_x), np.vstack(selected_y)
        prior_corrected = kernel_infer(kernel_model, onset_x)
        prior_corrected[:,:2] -= prior_center_bias
        old_grid_corrected = None
        if device == "target":
            prior = json.loads(PRIOR_GRID.read_text(encoding="utf-8"))
            calibration = prior["calibration"]["budgets"]["9"]
            delta, inside = interpolate_at_predicted_xy(
                kernel_infer(kernel_model, onset_x)[:,:2],
                tuple(float(a) for a in calibration["axes_mm"]),
                np.asarray(calibration["xyz_correction_mm"], float))
            if bool(np.all(inside)):
                old_grid_corrected = kernel_infer(kernel_model, onset_x)
                old_grid_corrected[:,:2] += delta[:,:2]
        report["devices"][device] = {
            "grid_trial": grids[0]["path"].name,
            "grid_training_frames": len(x),
            "grid_xy_range_mm": [np.min(y[:,:2],axis=0).tolist(),
                                 np.max(y[:,:2],axis=0).tolist()],
            "lambda_selection": candidates,
            "chosen_lambda": chosen,
            "kernel_selection": kernel_candidates,
            "chosen_kernel": kernel_choice,
            "R208_all_nominal_shallow": score(infer(shallow_model, eval_x), eval_y),
            "R208_newly_rescued": score(infer(shallow_model, onset_x), onset_y),
            "R208_all_nominal_shallow_kernel": score(kernel_infer(kernel_model, eval_x), eval_y),
            "R208_newly_rescued_kernel": score(kernel_infer(kernel_model, onset_x), onset_y),
            "R208_same_existing_UI_shallow_old": score(old_ui_pred, ui_y),
            "R208_same_existing_UI_shallow_kernel": score(kernel_infer(kernel_model, ui_x), ui_y),
            "R208_R231_gate_shallow_kernel": score(kernel_infer(kernel_model, selected_x), selected_y),
            "prior_grid_center_XY_bias_mm": prior_center_bias.tolist(),
            "R208_newly_rescued_kernel_prior_center_XY": score(prior_corrected, onset_y),
            "R208_newly_rescued_kernel_prior_day_9point_XY": (
                score(old_grid_corrected, onset_y) if old_grid_corrected is not None else None),
            "R208_rescued_xy_range_mm": [np.min(onset_y[:,:2],axis=0).tolist(),
                                           np.max(onset_y[:,:2],axis=0).tolist()],
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                   encoding="utf-8")
    print(json.dumps(report["devices"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
