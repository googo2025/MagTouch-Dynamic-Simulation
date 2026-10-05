#!/usr/bin/env python3
"""Causal shallow-contact separability audit on saved full magnetic streams.

This is an offline diagnostic, not a deployed contact gate.  Training uses
R133/R136 trials; the later R208 trials are evaluation only.  Stage commands
define *nominal* labels for scoring, but are never predictor inputs.  No future
magnetic samples, stage commands, trajectory templates, or saved UI predictions
enter the candidate scores.  The saved UI validity is a comparator only.
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from training.build_trajectory_dataset_v1 import (
    command_reference, command_segments, load_magnetic, read_rows,
    reference_origin,
)
from training.probe_had239_local_forward_r104 import ROOT, sha


OUT = ROOT / "analysis/r226_shallow_history_gate_20260928/report_v7.json"
DATA = ROOT / "data/S20260928_A"
PLAN_SETS = {
    "development": ("R133_RETRAIN", "R136_"),
    "evaluation": ("R208_",),
}
LAGS_NS = (100_000_000, 300_000_000, 700_000_000)
MIN_BASELINE_AGE_NS = 700_000_000
LABELS = ("shallow_descent", "shallow_ascent", "deep_contact", "deep_still",
          "lift_residual", "postcontact_lift", "precontact_lift",
          "unloaded_trial", "other")
TRAIN_MODES = ("none", "compound", "z_path", "waypoint_xyz")


def _bool_column(values):
    value = values.astype(str).str.lower().str.strip()
    if not value.isin(("true", "false", "1", "0")).all():
        raise ValueError("unexpected live validity encoding")
    return value.isin(("true", "1")).to_numpy(bool)


def _logged_b0(path):
    events = read_rows(path / "events.csv")
    rows = [row for row in events if row["event"] == "DYNAMIC_B0_APPLIED"]
    if len(rows) != 1:
        raise ValueError(f"expected one logged dynamic B0: {path}")
    payload = json.loads(rows[0]["value"])
    baseline = np.asarray(payload["b0_5x3_uT"], float).reshape(15)
    effective = max(int(rows[0]["t_perf_ns"]),
                    int(payload.get("completed_perf_ns", rows[0]["t_perf_ns"])))
    if not np.isfinite(baseline).all():
        raise ValueError(f"nonfinite B0: {path}")
    return baseline, effective


def _stage_labels(path, time, mode, meta):
    labels = np.full(len(time), "other", dtype="U20")
    if mode == "none":
        labels[:] = "unloaded_trial"
        return labels
    segments = command_segments(read_rows(path / "stage_commands.csv"))
    origin = reference_origin(meta)
    ref = command_reference(time, segments, origin)
    depth = ref["xyz_mm"][:, 2]
    seg_id = ref["command_segment_id"]
    phases = np.asarray([segments[i]["phase"] if i >= 0 else "before_stage"
                         for i in seg_id])
    within_contact = np.asarray([
        bool(re.fullmatch(r"waypoint_\d{3}|path_\d+", phase))
        or phase in ("xyz_compound", "grid_indent") for phase in phases])
    deep = np.isfinite(depth) & (depth >= .4) & ref["command_moving"] & within_contact
    labels[deep] = "deep_contact"
    deep_still = (np.isfinite(depth) & (depth >= .4) & ~ref["command_moving"]
                  & within_contact & (ref["hold_age_s"] >= .1))
    labels[deep_still] = "deep_still"
    direction = np.zeros(len(time), float)
    for i, seg in enumerate(segments):
        start, end = seg["from_mm"][2], seg["to_mm"][2]
        if np.isfinite(start) and np.isfinite(end):
            direction[seg_id == i] = start - end  # positive: deeper indentation
    shallow = (np.isfinite(depth) & (depth >= .02) & (depth < .2)
               & ref["command_moving"])
    labels[shallow & (direction > .02)] = "shallow_descent"
    labels[shallow & (direction < -.02)] = "shallow_ascent"
    deep_t = np.where(np.isfinite(depth) & (depth >= .5), time, -1)
    last_deep = np.maximum.accumulate(deep_t)
    safe = np.isfinite(depth) & (depth <= -.5)
    labels[safe & (last_deep < 0)] = "precontact_lift"
    labels[safe & (last_deep >= 0) & (time - last_deep <= 3_000_000_000)] = "lift_residual"
    labels[safe & (last_deep >= 0) & (time - last_deep > 3_000_000_000)] = "postcontact_lift"
    return labels


def _features(time, field):
    amplitude = np.linalg.norm(field, axis=1)
    lagged = []
    for lag in LAGS_NS:
        idx = np.searchsorted(time, time - lag, side="right") - 1
        idx = np.maximum(idx, 0)
        lagged.append(field[idx])
    scalar = [amplitude / 100.]
    for previous in lagged:
        previous_amp = np.linalg.norm(previous, axis=1)
        difference = field - previous
        scalar.extend(((amplitude - previous_amp) / 100.,
                       np.linalg.norm(difference, axis=1) / 100.,
                       np.sum(field * difference, axis=1) / (100. * np.maximum(amplitude, 1.))))
    scalar = np.column_stack(scalar)
    # Direction and signed field change are explicit causal vector-history probes.
    vector = np.column_stack((scalar, field / np.maximum(amplitude[:, None], 1.),
                              (field - lagged[1]) / 100.))
    return {"amplitude": scalar[:, :1], "history_scalar": scalar,
            "history_vector": vector}, amplitude


def load_trial(path, partition):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    mode = meta["trial_parameters"]["command_mode"]
    raw = load_magnetic(path / "magnetic_raw.csv")
    saved = pd.read_csv(path / "live_predictions.csv",
                        usecols=["t_prediction_ns", "interaction_valid"], low_memory=False)
    time = saved.t_prediction_ns.to_numpy(np.int64)
    raw_order = np.argsort(raw["t_group_ns"])
    raw_time = raw["t_group_ns"][raw_order]
    index = np.searchsorted(raw_time, time)
    matched = index < len(raw_time)
    matched[matched] &= raw_time[index[matched]] == time[matched]
    if np.mean(matched) < .98:
        raise ValueError(f"raw/live timestamp mismatch: {path}")
    safe_index = np.minimum(index, len(raw_time) - 1)
    field = raw["b_uT"][raw_order][safe_index].astype(float)
    sensors_ok = raw["sensor_valid"][raw_order][safe_index].all(axis=1)
    baseline, effective = _logged_b0(path)
    eligible = matched & sensors_ok & (time >= effective + MIN_BASELINE_AGE_NS)
    if not eligible.any():
        raise ValueError(f"no post-B0 full-history frames: {path}")
    field -= baseline
    features, amplitude = _features(time, field)
    labels = _stage_labels(path, time, mode, meta)
    valid_ui = _bool_column(saved.interaction_valid)
    if "had-Yelow" in path.name:
        device = "target"
    elif "had-239" in path.name:
        device = "source"
    else:
        raise ValueError(f"device not identified: {path}")
    return {"path": path, "partition": partition, "device": device,
            "mode": mode, "time": time, "eligible": eligible,
            "labels": labels, "features": features, "amplitude_uT": amplitude,
            "field_uT": field,
            "ui_valid": valid_ui, "raw_sha256": sha(path / "magnetic_raw.csv")}


def _fit_ridge(rows, feature_name, *, include_deep=False, device="both",
               train_modes=TRAIN_MODES):
    selected = [row for row in rows if row["partition"] == "development"
                and row["mode"] in train_modes
                and (device == "both" or row["device"] == device)]
    xs, ys, weights = [], [], []
    for row in selected:
        labels = row["labels"]
        label_targets = [("shallow_descent", 1.)]
        if include_deep:
            label_targets.append(("deep_contact", 1.))
        label_targets.extend((("unloaded_trial", 0.),
                              ("precontact_lift", 0.), ("lift_residual", 0.)))
        for label, target in label_targets:
            ids = np.flatnonzero(row["eligible"] & (labels == label))
            if not len(ids):
                continue
            # Keep class and negative-context weights comparable despite long
            # no-load trials and densely sampled adjacent frames.
            stride = max(1, int(np.ceil(len(ids) / 1200)))
            ids = ids[::stride]
            xs.append(row["features"][feature_name][ids])
            ys.append(np.full(len(ids), target))
            weights.append(np.full(len(ids), (1. if target else .5) / len(ids)))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)
    if y.sum() < 10 or len(y) - y.sum() < 10:
        raise ValueError(f"insufficient supervised classes for {feature_name}/{device}")
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .05)
    transformed = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.eye(transformed.shape[1]) * (.12 if feature_name in
                     ("history_vector", "long_vector") else .03)
    penalty[0, 0] = 0.
    beta = np.linalg.solve((transformed * w[:, None]).T @ transformed + penalty,
                           (transformed * w[:, None]).T @ y)
    return {"name": feature_name, "device": device, "include_deep": include_deep,
            "mean": mean,
            "scale": scale, "beta": beta,
            "training_frames": int(len(y)), "training_shallow_frames": int(y.sum())}


def _score(row, model):
    x = row["features"][model["name"]]
    return model["beta"][0] + ((x - model["mean"]) / model["scale"]) @ model["beta"][1:]


def _auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    # Pairwise Mann-Whitney statistic with exact tie handling.
    negative = np.sort(negative)
    below = np.searchsorted(negative, positive, side="left")
    equal = np.searchsorted(negative, positive, side="right") - below
    return float(np.mean((below + .5 * equal) / len(negative)))


def _threshold(rows, model, train_modes=TRAIN_MODES):
    negatives = {}
    for label in ("unloaded_trial", "precontact_lift", "lift_residual"):
        parts = [score[row["eligible"] & (row["labels"] == label)]
                 for row in rows if row["partition"] == "development"
                 and row["mode"] in train_modes
                 and (model["device"] == "both" or row["device"] == model["device"])
                 if (score := _score(row, model)) is not None]
        negatives[label] = np.concatenate([part for part in parts if len(part)])
    # Deliberately predeclared: <=1% of trained lifted-residual and <=0.1%
    # of trained long no-load scores exceed the threshold.
    threshold = max(float(np.percentile(negatives["unloaded_trial"], 99.9)),
                    float(np.percentile(negatives["lift_residual"], 99.)))
    return threshold


def _rate(mask, predicted):
    return {"n": int(mask.sum()), "positive": int(np.sum(mask & predicted)),
            "rate": float(np.mean(predicted[mask])) if mask.any() else None}


def _sequential_gate(row, onset_score, onset_threshold):
    """A frozen, causal 3-state probe; not a recommended UI implementation.

    A two-in-three onset vote can arm contact.  A large, rising field can also
    arm it if shallow evidence was missed.  Once active, a confirmed drop from
    the contact peak releases it.  Re-arming *never* uses the strong-field arm
    alone after a release, so magnetic residue cannot re-arm by amplitude.
    """
    time, amp, eligible = row["time"], row["amplitude_uT"], row["eligible"]
    onset = onset_score >= onset_threshold
    output = np.zeros(len(time), bool)
    active, ever_active = False, False
    peak, drop_count = 0., 0
    recent_peak = deque()
    onset_recent = deque(maxlen=3)
    start_index = np.flatnonzero(eligible)
    if not len(start_index):
        return output
    for i in range(int(start_index[0]), len(time)):
        if not eligible[i] or (i and time[i] - time[i-1] > 1_000_000_000):
            active, peak, drop_count = False, 0., 0
            recent_peak.clear()
            onset_recent.clear()
            continue
        onset_recent.append(bool(onset[i]))
        previous = max(0, np.searchsorted(time, time[i]-300_000_000, side="right")-1)
        strong_rise = amp[i] >= 120. and amp[i]-amp[previous] >= 20.
        if not active:
            if sum(onset_recent) >= 2 or (not ever_active and strong_rise):
                active, ever_active, peak = True, True, float(amp[i])
                drop_count = 0
                recent_peak.clear()
                recent_peak.append((int(time[i]), float(amp[i])))
        else:
            recent_peak.append((int(time[i]), float(amp[i])))
            while recent_peak and recent_peak[0][0] < time[i] - 800_000_000:
                recent_peak.popleft()
            peak = max(value for _, value in recent_peak)
            falling = (amp[i] <= .65*peak and peak-amp[i] >= 50.) or amp[i] < 40.
            drop_count = drop_count + 1 if falling else 0
            if drop_count >= 3:
                active, peak, drop_count = False, 0., 0
                onset_recent.clear()
                recent_peak.clear()
        output[i] = active
    return output


def _by_trial(row, scores, sequence):
    selected = row["eligible"]
    out = {"trial": row["path"].name, "device": row["device"],
           "mode": row["mode"], "partition": row["partition"],
           "raw_sha256": row["raw_sha256"],
           "labels": {label: {"frames": int(np.sum(selected & (row["labels"] == label))),
                               "ui_positive": int(np.sum(selected & (row["labels"] == label)
                                                         & row["ui_valid"])),
                               "amplitude_p50_uT": float(np.median(row["amplitude_uT"][
                                   selected & (row["labels"] == label)]))
                               if np.any(selected & (row["labels"] == label)) else None}
                      for label in LABELS}}
    out["candidates"] = {}
    for name, (score, threshold) in scores.items():
        predicted = score >= threshold
        out["candidates"][name] = {label: _rate(selected & (row["labels"] == label), predicted)
                                    for label in LABELS if np.any(selected & (row["labels"] == label))}
    out["candidates"]["sequential_probe"] = {
        label: _rate(selected & (row["labels"] == label), sequence)
        for label in LABELS if np.any(selected & (row["labels"] == label))}
    shallow_ids = np.flatnonzero(selected & (row["labels"] == "shallow_descent"))
    blocks = np.split(shallow_ids, np.flatnonzero(np.diff(shallow_ids) > 1) + 1)
    out["shallow_descent_episodes"] = []
    for block in blocks:
        if not len(block):
            continue
        episode = {"frames": int(len(block))}
        for label, mask in (("saved_ui", row["ui_valid"]),
                            ("sequential_probe", sequence)):
            positive = block[mask[block]]
            episode[f"{label}_onset_delay_ms"] = (
                float((row["time"][positive[0]] - row["time"][block[0]]) / 1e6)
                if len(positive) else None)
        out["shallow_descent_episodes"].append(episode)
    return out


def _pooled(rows, scored, device, partition, name, sequences):
    selected = [row for row in rows if row["device"] == device and row["partition"] == partition]
    output = {}
    for label in LABELS:
        masks = [row["eligible"] & (row["labels"] == label) for row in selected]
        n = sum(int(mask.sum()) for mask in masks)
        if not n:
            continue
        ui = sum(int(np.sum(mask & row["ui_valid"])) for row, mask in zip(selected, masks))
        result = {"frames": n, "saved_ui_positive": ui, "saved_ui_rate": ui/n}
        amplitude = np.concatenate([row["amplitude_uT"][mask]
                                    for row, mask in zip(selected, masks) if mask.any()])
        result["amplitude_p50_uT"] = float(np.median(amplitude))
        result["amplitude_below_50_fraction"] = float(np.mean(amplitude < 50.))
        if name == "sequential_probe":
            predicted = sum(int(np.sum(mask & sequences[row["path"]]))
                            for row, mask in zip(selected, masks))
            scores = []
        else:
            scores = [scored[row["path"]][name][0][mask]
                      for row, mask in zip(selected, masks) if mask.any()]
            predicted = sum(int(np.sum(part >= scored[row["path"]][name][1]))
                            for row, mask in zip(selected, masks) if mask.any()
                            for part in [scored[row["path"]][name][0][mask]])
        result["candidate_positive"] = predicted
        result["candidate_rate"] = predicted/n
        if scores:
            result["score_p50"] = float(np.median(np.concatenate(scores)))
        output[label] = result
    if name == "sequential_probe":
        return output
    positive = np.concatenate([scored[row["path"]][name][0][row["eligible"] &
                           (row["labels"] == "shallow_descent")]
                           for row in selected if np.any(row["eligible"] &
                           (row["labels"] == "shallow_descent"))])
    for label in ("unloaded_trial", "lift_residual", "precontact_lift"):
        parts = [scored[row["path"]][name][0][row["eligible"] & (row["labels"] == label)]
                 for row in selected if np.any(row["eligible"] & (row["labels"] == label))]
        if parts:
            output[f"auc_shallow_vs_{label}"] = _auc(positive, np.concatenate(parts))
    return output


def main():
    if OUT.exists():
        raise FileExistsError(f"preserve versioned result: {OUT}")
    rows = []
    for partition, prefixes in PLAN_SETS.items():
        for plan in sorted(DATA.iterdir()):
            if not plan.is_dir() or not any(part in plan.name for part in prefixes):
                continue
            for path in sorted(plan.glob("Trial_*")):
                if not path.is_dir():
                    continue
                meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
                mode = meta["trial_parameters"]["command_mode"]
                if mode not in ("none", "compound", "z_path", "waypoint_xyz", "grid_points"):
                    continue
                row = load_trial(path, partition)
                rows.append(row)
                counts = {label: int(np.sum(row["eligible"] & (row["labels"] == label)))
                          for label in LABELS}
                print(partition, row["device"], path.name.split("_")[1], counts, flush=True)
    models = {f"onset_{name}": _fit_ridge(rows, name) for name in
              ("amplitude", "history_scalar", "history_vector")}
    models.update({f"contact_{name}": _fit_ridge(rows, name, include_deep=True)
                   for name in ("amplitude", "history_scalar", "history_vector")})
    thresholds = {name: _threshold(rows, model) for name, model in models.items()}
    scored = {row["path"]: {name: (_score(row, model), thresholds[name])
                            for name, model in models.items()} for row in rows}
    sequences = {row["path"]: _sequential_gate(
        row, *scored[row["path"]]["onset_history_vector"]) for row in rows}
    report = {
        "schema": "MagTouch-R226-CausalShallowHistoryAudit-1",
        "status": "offline development audit; R208 already viewed in prior model research; no UI or hardware changes",
        "label_warning": "commanded 0.02-0.2 mm descent is not independent physical contact; lifted residual is commanded <=-0.5 mm within 3 s after >=0.5 mm depth",
        "partition": "R133/R136 none/compound/z_path/waypoint train and threshold selection; R133 grids are transfer checks, never fitted; R208 initially evaluated later, but the 800-ms state peak was chosen after seeing its failure, so final sequential probe on R208 is development evidence, NOT an independent blind test; each complete Trial remains within one set",
        "causality": "candidate features use current and preceding 0.1/0.3/0.7 s raw 15D field minus logged pre-action B0; no stage/command/future/UI features",
        "threshold_policy": "maximum of development no-load score p99.9 and lift-residual score p99; no evaluation threshold retuning",
        "sequential_probe": "offline state-machine probe: 2-in-3 shallow onset vote or initial >120 uT rising field, 3-frame 0.65-of-800ms-peak/50-uT release; after release only onset vote may re-arm",
        "models": {name: {"feature_count": int(len(model["mean"])),
                          "training_frames": model["training_frames"],
                          "training_shallow_frames": model["training_shallow_frames"],
                          "deep_contact_in_training": model["include_deep"],
                          "threshold": thresholds[name]}
                   for name, model in models.items()},
        "pooled": {partition: {device: {name: _pooled(rows, scored, device, partition, name, sequences)
                                           for name in (*models, "sequential_probe")}
                               for device in ("source", "target")}
                   for partition in PLAN_SETS},
        "trials": [_by_trial(row, scored[row["path"]], sequences[row["path"]]) for row in rows],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({device: {name: report["pooled"]["evaluation"][device][name]
                                      for name in models}
                      for device in ("source", "target")},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
