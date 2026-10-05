#!/usr/bin/env python3
"""Read-only full-rate replay of magnetic-evidence branch veto.

Uses the live contact mask and stored memory state; stage commands are only
an alignment reference. These already-viewed records are development data,
not a blind prospective test. No UI/acquisition/hardware import is made.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from training.build_trajectory_dataset_v1 import (
    baseline_schedule, command_reference, command_segments, load_magnetic,
    read_rows, reference_origin,
)
from training.diagnose_r126_latest_four_live_r130 import circle_shape, xy_summary
from training.evidence_guarded_branch_r162 import EvidenceGuardedBranchInverse
from training.evaluate_branch_inverse_r139 import FIVE_CONFIG, FIVE_ID, FIVE_KEY, FIVE_REPORT
from training.probe_had239_field_memory_r122 import feature_to_field
from training.probe_had239_local_forward_r104 import ForwardAtlas, ROOT, load_profile, sha

OUT = ROOT / "analysis/r163_guarded_branch_live_20260926/report.json"
NUMBERS = (12, 13, 14, 15, 30, 31, 32)
RATIOS = (2.0, 2.5, 3.0, 4.0, 1e6)


def load(number, atlas, basis, strength):
    path, = (ROOT / "data/S20260926_A").glob(f"Trial_{number:06d}_*")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta["acquisition"]["magnetic_layer_profile_id"] != FIVE_ID:
        raise ValueError("wrong magnet")
    frame = pd.read_csv(path / "live_predictions.csv", low_memory=False)
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
        raise ValueError("magnetic timestamps did not match")
    measured = raw["b_uT"][ordered]
    observed = np.full((len(t), 15), np.nan)
    baselines = baseline_schedule(events, raw, segments, reference_origin(meta))
    if not baselines:
        raise ValueError("no B0")
    for baseline in baselines:
        use = matched & (t >= baseline["effective_ns"])
        observed[use] = measured[index[use]] - baseline["b0_uT"]
    extra = [json.loads(value) for value in frame.extra_outputs_json]
    live = frame[["visual_x_mm", "visual_y_mm", "visual_z_mm"]].to_numpy(float)
    valid = frame.interaction_valid.to_numpy(bool) & matched & np.isfinite(live).all(axis=1)
    memory = np.full((len(t), 2), np.nan)
    for i, value in enumerate(extra):
        if valid[i] and isinstance(value.get("memory_state"), list):
            memory[i] = np.asarray(value["memory_state"], float)
    valid &= np.isfinite(memory).all(axis=1)
    corrected = np.full_like(observed, np.nan)
    corrected_phi = atlas.features(observed[valid]) - strength * memory[valid] @ basis.T
    corrected[valid] = feature_to_field(atlas, corrected_phi)
    global_ = np.full((len(t), 3), np.nan)
    global_[valid] = atlas.predict(corrected[valid])
    model_family = meta.get("live_prediction_recording", {}).get("model_family", "")
    if model_family != "BranchFieldMemoryR128":
        comparison = live
    else:
        comparison = np.full_like(global_, np.nan)
        for i, value in enumerate(extra):
            row = value.get("branch_unfiltered_xyz_mm")
            if isinstance(row, list) and len(row) == 3:
                comparison[i] = row
    same = valid & np.isfinite(comparison).all(axis=1)
    mismatch = float(np.median(np.linalg.norm(global_[same, :2] - comparison[same, :2], axis=1)))
    if mismatch > .35:
        raise ValueError(f"reconstructed global diverged {number}: {mismatch:.3f}mm")
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
        moving_phase = (phases == "xyz_compound")
    else:
        raise ValueError(mode)
    core = valid & moving_phase & (ref["contact_command"] == 1)
    if mode != "grid_points":
        core &= ref["command_moving"]
    if core.sum() < 100:
        raise ValueError("too few core rows")
    return {
        "path": path, "meta": meta, "time": t, "valid": valid, "core": core,
        "corrected": corrected, "global": global_, "live": live, "truth": truth,
        "memory": memory, "mode": mode, "observed": observed,
        "reconstruction_mismatch_mm": mismatch,
    }


def evaluate(data, atlas, ratio):
    t = data["time"]
    valid = data["valid"]
    core = data["core"]
    global_ = data["global"]
    truth = data["truth"]
    result = np.full_like(global_, np.nan)
    tracker = EvidenceGuardedBranchInverse(
        atlas, spatial_penalty=.03, innovation_gain=2.0,
        activation_distance_mm=1.0, max_local_cost_ratio=ratio)
    local, veto = np.zeros(len(t), bool), np.zeros(len(t), bool)
    for i in range(len(t)):
        if not valid[i]:
            tracker.reset()
            continue
        row = tracker.predict_frame(data["corrected"][i], time_ns=int(t[i]),
                                    b0_id=0, stream_id=0,
                                    global_xyz=global_[i])
        result[i] = row["xyz_mm"]
        local[i] = row["used_local_branch"]
        veto[i] = row["local_branch_vetoed_by_field"]
    pair = core[1:] & core[:-1] & (np.diff(t) > 0) & (np.diff(t) < 120_000_000)
    step = np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1)[pair]
    out = {"ratio": ratio, "core_rows": int(core.sum()),
           "local_core_rows": int((local & core).sum()),
           "vetoed_core_rows": int((veto & core).sum()),
           "position": xy_summary(result[core], truth[core]),
           "jumps_over_1mm": int((step > 1).sum()),
           "max_jump_mm": float(step.max()) if len(step) else None}
    if data["meta"]["trial_parameters"]["command_mode"] == "circle_scan":
        out["circle"] = circle_shape(result[core], truth[core])
    return out


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    profile = load_profile(FIVE_CONFIG, FIVE_KEY)
    samples = profile["samples"]
    xyz = np.asarray([[s["x"], s["y"], s["depth"]] for s in samples])
    delta = np.asarray([s["delta"] for s in samples])
    atlas = ForwardAtlas(xyz, delta, "unit_logamp")
    model = json.loads(FIVE_REPORT.read_text(encoding="utf-8"))
    basis = np.asarray(model["basis"], float)
    strength = float(model["selection"]["selected"]["strength"])
    rows = []
    for number in NUMBERS:
        data = load(number, atlas, basis, strength)
        core = data["core"]
        truth = data["truth"]
        row = {"trial": number,
               "source": data["path"].relative_to(ROOT).as_posix(),
               "source_sha256": {name: sha(data["path"] / name) for name in (
                   "meta.json", "live_predictions.csv", "magnetic_raw.csv",
                   "stage_commands.csv")},
               "reconstruction_mismatch_mm": data["reconstruction_mismatch_mm"],
               "live": xy_summary(data["live"][core], truth[core]),
               "global": xy_summary(data["global"][core], truth[core]),
               "variants": [evaluate(data, atlas, ratio) for ratio in RATIOS]}
        if number in (14, 15, 32):
            row["live_circle"] = circle_shape(data["live"][core], truth[core])
            row["global_circle"] = circle_shape(data["global"][core], truth[core])
        rows.append(row)
        print(number, "live", round(row["live"]["xy_mean_mm"], 3),
              "global", round(row["global"]["xy_mean_mm"], 3),
              "variants", [(v["ratio"], round(v["position"]["xy_mean_mm"], 3),
                            v["jumps_over_1mm"], v["local_core_rows"])
                           for v in row["variants"]], flush=True)
    report = {"schema": "MagTouch-R163-GuardedBranchFullRateReplay-1",
              "warning": "Previously viewed live trials; command positions are not independent contact measurement; stored live gate and memory state used; no circle fitting or future frames.",
              "ratios": RATIOS, "static_config_sha256": sha(FIVE_CONFIG),
              "memory_report_sha256": sha(FIVE_REPORT), "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
