#!/usr/bin/env python3
"""Read-only audit of the two 2026-09-28 R133 online replication plans.

The new prediction is the saved online output. The R130 comparator is an
offline, same-stream replay on the saved guard/memory state and contact mask;
it must not be described as another online experiment. This script never
modifies a Trial or fits a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.diagnose_r126_latest_four_live_r130 import circle_shape
from training.evaluate_branch_inverse_r139 import FIVE_CONFIG, FIVE_KEY, FIVE_REPORT
from training.had239_causal_xy_mlp_runtime_r165 import CausalXYMLP
from training.probe_had239_local_forward_r104 import ForwardAtlas, ROOT, load_profile
from training.retrain_r133_coverage_xy import features, load_new


PLANS = {
    "had-239": "Plan_20260928_001547_R133_RETRAIN_HAD239_ONLINE_REPLICATION__mag_5x5-had-239",
    "had-Yelow": "Plan_20260928_004033_R133_RETRAIN_YELLOW_ONLINE_REPLICATION__mag_5x5-had-Yelow",
}
EXPECTED_SHA = "8a7e48aa37c59a6f0975155e8bedf7135e07d84bf29a8444a8dd9b2808e7c112"
MODES = ("circle_scan", "angle_scan", "compound", "z_path", "grid_points")


def mean(values):
    return float(np.mean(np.asarray(values, float)))


def make_atlas():
    profile = load_profile(FIVE_CONFIG, FIVE_KEY)
    samples = profile["samples"]
    atlas = ForwardAtlas(
        np.asarray([[s["x"], s["y"], s["depth"]] for s in samples], float),
        np.asarray([s["delta"] for s in samples], float),
        "unit_logamp",
    )
    report = json.loads(FIVE_REPORT.read_text(encoding="utf-8"))
    basis = np.asarray(report["basis"], float)
    strength = float(report["selection"]["selected"]["strength"])
    return atlas, basis, strength


def score(prediction, truth):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    xy = error[:, :2]
    bias = xy.mean(axis=0)
    zbias = float(error[:, 2].mean())
    return {
        "xy_raw_mean_mm": mean(np.linalg.norm(xy, axis=1)),
        "xy_translation_removed_mean_mm": mean(np.linalg.norm(xy - bias, axis=1)),
        "xy_bias_mm": bias.tolist(),
        "z_raw_mae_mm": mean(np.abs(error[:, 2])),
        "z_translation_removed_mae_mm": mean(np.abs(error[:, 2] - zbias)),
        "z_bias_mm": zbias,
    }


def audit_trial(path: Path, atlas, basis, strength, old_model):
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    mode = meta["trial_parameters"]["command_mode"]
    if mode == "none":
        frame = pd.read_csv(path / "live_predictions.csv", low_memory=False)
        return {
            "trial": int(meta["trial_parameters"]["trial_id"]),
            "mode": mode,
            "prediction_frames": int(len(frame)),
            "interaction_valid_frames": int(frame.interaction_valid.astype(bool).sum()),
        }
    if mode not in MODES:
        raise ValueError((path, mode))
    if EXPECTED_SHA not in json.dumps(meta.get("live_prediction", meta), ensure_ascii=False):
        raise ValueError(f"unexpected model artifact in {path}")
    data = load_new(path, atlas, basis, strength)
    core = data["core"]
    static = atlas.predict(data["observed"][core])
    memory_only = atlas.predict(data["corrected"][core])
    guard_only = data["guarded"][core]
    old = guard_only.copy()
    old[:, :2] += old_model.predict(features(data, atlas)[core])
    new = data["live"][core]
    truth = data["truth"][core]
    result = {
        "trial": int(meta["trial_parameters"]["trial_id"]),
        "mode": mode,
        "core_reference_frames": int(data["core_ref"].sum()),
        "core_valid_frames": int(core.sum()),
        "coverage": float(core.sum() / data["core_ref"].sum()),
        "static_atlas_same_stream_probe": score(static, truth),
        "memory_corrected_atlas_same_stream_probe": score(memory_only, truth),
        "memory_plus_branch_guard_same_stream_probe": score(guard_only, truth),
        "old_r130_same_stream_replay": score(old, truth),
        "new_r133_saved_online": score(new, truth),
    }
    if mode == "circle_scan":
        result["old_circle"] = circle_shape(old, truth)
        result["new_circle"] = circle_shape(new, truth)
    return result


def aggregate(rows):
    contact = [r for r in rows if r["mode"] != "none"]
    empty = [r for r in rows if r["mode"] == "none"]
    out = {
        "contact_trials": len(contact),
        "no_load_trials": len(empty),
        "no_load_prediction_frames": sum(r["prediction_frames"] for r in empty),
        "no_load_interaction_valid_frames": sum(r["interaction_valid_frames"] for r in empty),
        "by_mode": {},
    }
    for mode in ("all",) + MODES:
        subset = contact if mode == "all" else [r for r in contact if r["mode"] == mode]
        if not subset:
            continue
        summary = {"trial_count": len(subset), "coverage_trial_macro": mean([r["coverage"] for r in subset])}
        for version in ("static_atlas_same_stream_probe",
                        "memory_corrected_atlas_same_stream_probe",
                        "memory_plus_branch_guard_same_stream_probe",
                        "old_r130_same_stream_replay", "new_r133_saved_online"):
            summary[version] = {
                metric: mean([r[version][metric] for r in subset])
                for metric in ("xy_raw_mean_mm", "xy_translation_removed_mean_mm",
                               "z_raw_mae_mm", "z_translation_removed_mae_mm")
            }
        if mode == "circle_scan":
            summary["old_fourfold_amplitude_mm"] = mean([r["old_circle"]["fourfold_amplitude_mm"] for r in subset])
            summary["new_fourfold_amplitude_mm"] = mean([r["new_circle"]["fourfold_amplitude_mm"] for r in subset])
        out["by_mode"][mode] = summary
    return out


def main():
    atlas, basis, strength = make_atlas()
    old_model = CausalXYMLP()
    out = {
        "schema": "MagTouch-R133-NewOnline-ReadOnlyAudit-1",
        "reference": "time-interpolated platform command",
        "bias_method": "subtract each Trial's mean signed XYZ residual only for diagnostic shape scores",
        "comparison": "saved R133 online; frozen R130 same-stream replay; F/M/G same-stream probes using R133's logged causal state, not independently re-run pipelines",
        "expected_r133_model_sha256": EXPECTED_SHA,
        "groups": {},
    }
    for name, folder in PLANS.items():
        group = ROOT / "data/S20260928_A" / folder
        paths = sorted(p for p in group.glob("Trial_*") if p.is_dir())
        if len(paths) != 12:
            raise ValueError(f"expected 12 trials for {name}, found {len(paths)}")
        rows = [audit_trial(path, atlas, basis, strength, old_model) for path in paths]
        out["groups"][name] = {"plan": str(group.relative_to(ROOT)), "summary": aggregate(rows), "trials": rows}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
