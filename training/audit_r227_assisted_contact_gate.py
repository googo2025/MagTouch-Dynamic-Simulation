#!/usr/bin/env python3
"""Evaluate a history-assisted *supplement* to the existing contact gate.

This never suppresses a frame accepted by the saved online gate, so deeper
motion cannot be lost as in the R226 replacement state machine. Labels and
saved validity are used only for retrospective scoring, never as classifier
features. R208 has already been inspected, so these are development results.
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np

from training.build_trajectory_dataset_v1 import (
    command_reference, command_segments, read_rows, reference_origin,
)
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
from training.audit_r226_shallow_history_gate import (
    DATA, PLAN_SETS, ROOT, _fit_ridge, _score, _threshold, load_trial,
)


OUT = ROOT / "analysis/r227_assisted_contact_gate_20260928/report_v2.json"
INTEREST = ("shallow_descent", "shallow_ascent", "deep_contact", "deep_still",
            "lift_residual", "postcontact_lift", "precontact_lift", "unloaded_trial")


def load_rows():
    rows = []
    for partition, prefixes in PLAN_SETS.items():
        for plan in sorted(DATA.iterdir()):
            if not plan.is_dir() or not any(part in plan.name for part in prefixes):
                continue
            for path in sorted(plan.glob("Trial_*")):
                if not path.is_dir():
                    continue
                meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
                if meta["trial_parameters"]["command_mode"] not in (
                        "none", "compound", "z_path", "waypoint_xyz", "grid_points"):
                    continue
                rows.append(load_trial(path, partition))
    return rows


def vote(scores, threshold, eligible, width=3, required=2, time=None):
    result = np.zeros(len(scores), bool)
    window = deque(maxlen=1 if required == 1 else width)
    for i in range(len(scores)):
        if (not eligible[i] or (time is not None and i and
                time[i]-time[i-1] > 1_000_000_000)):
            window.clear()
        if not eligible[i]:
            continue
        window.append(bool(scores[i] >= threshold))
        result[i] = len(window) >= required and sum(window) >= required
    return result


def summarize(rows, model, threshold, vote_required):
    grouped = {}
    for row in rows:
        score = _score(row, model)
        assisted = vote(score, threshold, row["eligible"],
                        required=vote_required, time=row["time"])
        ui = row["ui_valid"] & row["eligible"]
        rescued = assisted & ~ui
        combined = ui | rescued
        key = (row["partition"], row["device"])
        acc = grouped.setdefault(key, {})
        for label in INTEREST:
            mask = row["eligible"] & (row["labels"] == label)
            if not mask.any():
                continue
            slot = acc.setdefault(label, {"frames": 0, "ui": 0, "rescued": 0,
                                          "combined": 0})
            slot["frames"] += int(mask.sum())
            slot["ui"] += int(np.sum(mask & ui))
            slot["rescued"] += int(np.sum(mask & rescued))
            slot["combined"] += int(np.sum(mask & combined))
        if row["mode"] == "grid_points":
            for label, mask in (("grid_shallow", row["eligible"] &
                         (row["labels"] == "other") &
                         (row["amplitude_uT"] < 100.)),):
                if not mask.any():
                    continue
                slot = acc.setdefault(label, {"frames": 0, "ui": 0,
                                              "rescued": 0, "combined": 0})
                slot["frames"] += int(mask.sum())
                slot["ui"] += int(np.sum(mask & ui))
                slot["rescued"] += int(np.sum(mask & rescued))
                slot["combined"] += int(np.sum(mask & combined))
    return {partition: {device: grouped.get((partition, device), {})
                        for device in ("source", "target")}
            for partition in PLAN_SETS}


def decode_rescued_shallow(rows, model, threshold):
    """Probe XYZ for newly accepted shallow frames on isolated descent episodes.

    A gap resets the decoder, matching online no-contact semantics. This is a
    diagnostic, not a full-stream same-state replay of the hypothetical gate.
    """
    findings = {device: {"frames": 0, "xyz_valid": 0, "xy_errors_mm": [],
                         "z_estimates_mm": [], "z_references_mm": []}
                for device in ("source", "target")}
    decoder = CoverageGuardedNeuralDecoder.from_report()
    for row in rows:
        if row["partition"] != "evaluation" or row["mode"] == "none":
            continue
        score = _score(row, model)
        onset = vote(score, threshold, row["eligible"], required=2,
                     time=row["time"])
        shallow = row["eligible"] & (row["labels"] == "shallow_descent")
        rescued = shallow & onset & ~row["ui_valid"]
        if not rescued.any():
            continue
        meta = json.loads((row["path"] / "meta.json").read_text(encoding="utf-8"))
        segments = command_segments(read_rows(row["path"] / "stage_commands.csv"))
        ref = command_reference(row["time"], segments, reference_origin(meta))["xyz_mm"]
        result = findings[row["device"]]
        selected = np.flatnonzero(shallow & (onset | row["ui_valid"]))
        decoder.reset()
        previous = -1
        for i in selected:
            if i != previous + 1 or (previous >= 0 and
                    row["time"][i] - row["time"][previous] > 250_000_000):
                decoder.reset()
            previous = i
            prediction = decoder.predict_frame(
                row["field_uT"][i], time_ns=int(row["time"][i]),
                b0_id=1, stream_id=0, contact_active=True)
            if not rescued[i]:
                continue
            result["frames"] += 1
            if prediction is None:
                continue
            xyz = np.asarray(prediction["xyz_mm"], float)
            if not np.isfinite(xyz).all():
                continue
            result["xyz_valid"] += 1
            result["xy_errors_mm"].append(float(np.linalg.norm(xyz[:2]-ref[i, :2])))
            result["z_estimates_mm"].append(float(xyz[2]))
            result["z_references_mm"].append(float(ref[i, 2]))
    for result in findings.values():
        for name in ("xy_errors_mm", "z_estimates_mm", "z_references_mm"):
            values = result.pop(name)
            result[name.replace("_mm", "_median_mm")] = (
                float(np.median(values)) if values else None)
            result[name.replace("_mm", "_p90_mm")] = (
                float(np.percentile(values, 90)) if values else None)
    return findings


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = load_rows()
    model = _fit_ridge(rows, "history_vector")
    threshold = _threshold(rows, model)
    report = {
        "schema": "MagTouch-R227-HistoryAssistedGateAudit-1",
        "status": "offline retrospective development; no UI or hardware change; R208 not blind",
        "candidate": "existing online gate OR 40D causal historical onset score vote; no deep-frame veto",
        "model": {"feature_count": len(model["mean"]), "threshold": threshold,
                  "training_frames": model["training_frames"],
                  "training_shallow_frames": model["training_shallow_frames"]},
        "variants": {"one_of_one": summarize(rows, model, threshold, 1),
                     "two_of_three": summarize(rows, model, threshold, 2)},
        "isolated_shallow_xyz_diagnostic": decode_rescued_shallow(rows, model, threshold),
        "limitations": [
            "A rescued gate frame is not automatically a valid XYZ position.",
            "Static atlas starts at 0.4 mm and may not identify nominal 0.02-0.2 mm depth.",
            "Grid shallow command may not physically contact at every location.",
            "R208 previously inspected; the final gate requires independent validation."],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    for variant, parts in report["variants"].items():
        print(variant)
        for partition, devices in parts.items():
            for device, labels in devices.items():
                print(partition, device, {key: value for key, value in labels.items()
                                         if key in ("shallow_descent", "deep_contact",
                                                    "lift_residual", "unloaded_trial",
                                                    "grid_shallow")}, flush=True)


if __name__ == "__main__":
    main()
