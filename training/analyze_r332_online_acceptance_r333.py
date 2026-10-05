"""Saved-online R332 acceptance audit. No fitting, UI/hardware import or raw edit.

All 16 Trials are selected by completed plan metadata, never by outcome. Raw
and mean-translation-removed scores use the identical accepted-frame mask.
Extreme shallow commands describe actions, not independently confirmed contact.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from training.build_trajectory_dataset_v1 import (
    command_reference, command_segments, read_rows, reference_origin, sha, B_COLS,
)
from training.retrain_r133_coverage_xy import recorded_bool_mask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/r333_r332_online_acceptance_20260929"
EXPECTED = "bae01823ff8905370ad163d6b34fbdd7f7384378e0f9aba5dff747cd4ca51f4d"
RAW_FILES = ("meta.json", "batch_plan_item.json", "events.csv", "stage_commands.csv",
             "magnetic_raw.csv", "live_predictions.csv")


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def stats(values):
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v): return {"n": 0}
    return {"n": len(v), "mean": float(v.mean()), "std": float(v.std()),
            "p05": float(np.quantile(v, .05)), "p50": float(np.median(v)),
            "p95": float(np.quantile(v, .95)), "max": float(v.max()),
            "min": float(v.min()), "p95_minus_p05": float(np.quantile(v, .95)-np.quantile(v, .05))}


def scoring(a, mask):
    ref = mask & np.isfinite(a["q"]).all(1)
    good = ref & a["valid"]
    result = {"reference_frames": int(ref.sum()), "output_frames": int(good.sum()),
              "coverage": float(good.sum()/ref.sum()) if ref.any() else None}
    if not good.any(): return result
    e = a["xyz"][good]-a["q"][good]; bias = e.mean(0); c = e-bias
    result.update(bias_xyz_mm=bias.tolist(),
        xy_raw_mean_mm=float(np.linalg.norm(e[:, :2], axis=1).mean()),
        xy_centered_mean_mm=float(np.linalg.norm(c[:, :2], axis=1).mean()),
        z_raw_mae_mm=float(np.abs(e[:, 2]).mean()),
        z_centered_mae_mm=float(np.abs(c[:, 2]).mean()),
        xyz_raw_rmse_mm=float(np.sqrt(np.mean(np.sum(e*e, axis=1)))),
        xyz_centered_rmse_mm=float(np.sqrt(np.mean(np.sum(c*c, axis=1)))),
        xy_raw_p95_mm=float(np.quantile(np.linalg.norm(e[:, :2], axis=1), .95)),
        z_raw_p95_mm=float(np.quantile(np.abs(e[:, 2]), .95)))
    return result


def runs(mask, time, max_gap=.25):
    ids = np.flatnonzero(mask)
    if not len(ids): return []
    boundaries = (np.diff(ids) != 1) | (np.diff(time[ids])/1e9 > max_gap)
    return [v for v in np.split(ids, np.flatnonzero(boundaries)+1) if len(v)]


def primary_mask(mode, phase, ref):
    if mode == "angle_scan": selected = np.array([bool(re.fullmatch(r"angle_(?:return_)?\d+", p)) for p in phase])
    elif mode == "circle_scan": selected = np.array([bool(re.fullmatch(r"circle_\d+", p)) for p in phase])
    elif mode == "grid_points": selected = (phase == "grid_indent") & ~ref["command_moving"] & (ref["hold_age_s"] >= .15) & (ref["hold_age_s"] <= .55)
    elif mode == "compound": selected = phase == "xyz_compound"
    elif mode == "z_path": selected = np.array([bool(re.fullmatch(r"path_\d+", p)) for p in phase])
    elif mode == "none": return np.zeros(len(phase), bool)
    else: raise ValueError(mode)
    if mode != "grid_points": selected &= ref["command_moving"]
    return selected & (ref["xyz_mm"][:, 2] >= .4)


def movement_mask(mode, phase):
    if mode == "grid_points": return phase == "grid_indent"
    if mode == "angle_scan": return np.array([p.startswith("angle_") and not p.endswith("guard") for p in phase])
    if mode == "circle_scan": return np.array([bool(re.fullmatch(r"circle_\d+", p)) or p in ("circle_start", "indent") for p in phase])
    if mode == "compound": return phase == "xyz_compound"
    if mode == "z_path": return np.array([bool(re.fullmatch(r"path_\d+", p)) for p in phase])
    return np.zeros(len(phase), bool)


def jumping(a, mask):
    dt = np.diff(a["time"])/1e9
    pairs = mask[1:] & mask[:-1] & a["valid"][1:] & a["valid"][:-1] & (dt > 0) & (dt <= .25)
    de = np.diff(a["xyz"]-a["q"], axis=0)
    dz = np.abs(np.diff(a["xyz"][:, 2])); xy = np.linalg.norm(np.diff(a["xyz"][:, :2], axis=0), axis=1)
    switch = a["route"][1:] != a["route"][:-1]
    db = np.linalg.norm(np.diff(a["field"], axis=0), axis=1)
    out = {"pair_count": int(pairs.sum()), "dt_s": stats(dt[pairs]),
           "adjacent_z_step_mm": stats(dz[pairs]), "adjacent_xy_step_mm": stats(xy[pairs]),
           "adjacent_z_reference_residual_step_mm": stats(np.abs(de[pairs, 2])),
           "z_diff_sigma_proxy_mm": float(np.sqrt(np.mean(de[pairs, 2]**2)/2)) if pairs.any() else None,
           "by_route_pair": {}, "top_z_steps": []}
    for name, take in (("route_switch", pairs & switch), ("same_route", pairs & ~switch)):
        out["by_route_pair"][name] = {"pairs": int(take.sum()), "z_over_0p2_mm": int((take & (dz > .2)).sum()),
            "z_steps": stats(dz[take]), "field_step_uT": stats(db[take])}
    ids = np.flatnonzero(pairs)
    top = ids[np.argsort(dz[ids])[-10:][::-1]] if len(ids) else []
    for j in top:
        i = j+1
        out["top_z_steps"].append({"index": int(i), "time_s": float((a["time"][i]-a["time"][0])/1e9),
            "from_route": str(a["route"][j]), "to_route": str(a["route"][i]),
            "from_xyz_mm": a["xyz"][j].tolist(), "to_xyz_mm": a["xyz"][i].tolist(),
            "command_xyz_mm": a["q"][i].tolist(), "z_step_mm": float(dz[j]), "xy_step_mm": float(xy[j]),
            "field_step_uT": float(db[j]) if np.isfinite(db[j]) else None,
            "strength": int(a["strength"][i]), "score": float(a["permission"][i]) if np.isfinite(a["permission"][i]) else None})
    return out


def repeat_masks(path, a):
    events = read_rows(path/"events.csv"); begins=[]; windows=[]
    for ev in events:
        if ev["event"] == "ACTION_REPEAT_BEGIN": begins.append(int(ev["t_perf_ns"]))
        if ev["event"] == "ACTION_REPEAT_END" and begins:
            start = begins.pop(0); end = int(ev["t_perf_ns"])
            windows.append((a["time"] >= start) & (a["time"] <= end))
    return windows


def load(path):
    meta = json.loads((path/"meta.json").read_text(encoding="utf-8")); pr = meta["live_prediction_recording"]
    alg = pr["algorithm_metadata"]
    if pr["model_family"] != "UnifiedEvidenceTactileR332" or alg["localization_permission_protocol"]["sha256"] != EXPECTED:
        raise ValueError("Different model: "+str(path))
    if alg.get("target_fit_rows") != 0: raise ValueError("Target fitting reported")
    f = pd.read_csv(path/"live_predictions.csv", low_memory=False)
    extra = [json.loads(v) for v in f.extra_outputs_json]
    t = f.t_prediction_ns.to_numpy(np.int64)
    if (np.diff(t) <= 0).any(): raise ValueError("non-increasing saved predictions")
    xyz = f[["visual_x_mm", "visual_y_mm", "visual_z_mm"]].to_numpy(float)
    valid = recorded_bool_mask(f.interaction_valid) & np.isfinite(xyz).all(1)
    extra_valid = np.array([bool(e.get("xyz_valid")) for e in extra])
    if not np.array_equal(valid, extra_valid): raise ValueError("visual / XYZ validity disagreement")
    for i in np.flatnonzero(valid):
        np.testing.assert_allclose(xyz[i], extra[i]["xyz_mm"], atol=1e-10)
    seg = command_segments(read_rows(path/"stage_commands.csv"))
    ref = command_reference(t, seg, reference_origin(meta))
    phase = np.array([seg[j]["phase"] if j >= 0 else "before_stage" for j in ref["command_segment_id"]])
    # Exact timestamp magnetic matches, never reconstructed/smoothed coordinates.
    raw = pd.read_csv(path/"magnetic_raw.csv", usecols=["t_group_ns", "valid_sensor_count", *B_COLS])
    rtime = raw.t_group_ns.to_numpy(np.int64); order = np.argsort(rtime); rt = rtime[order]
    index = np.searchsorted(rt, t); matched = index < len(rt)
    matched[matched] &= rt[index[matched]] == t[matched]
    field = np.full((len(t), 15), np.nan); field[matched] = raw[B_COLS].to_numpy(float)[order][index[matched]]
    known = np.isfinite(ref["xyz_mm"]).all(1)
    a = {"time": t, "xyz": xyz, "valid": valid, "q": ref["xyz_mm"].astype(float),
         "phase": phase, "segment": ref["command_segment_id"], "moving": ref["command_moving"],
         "hold_age": ref["hold_age_s"], "known": known, "field": field, "matched": matched,
         "route": np.array([e.get("localization_path", "suppressed") for e in extra]),
         "permission": np.array([e.get("localization_permission_score") if e.get("localization_permission_score") is not None else np.nan for e in extra]),
         "amp": np.array([e.get("response_observation", {}).get("amplitude_uT", np.nan) for e in extra]),
         "strength": np.array([e.get("magnetic_strength", 0) for e in extra]),
         "action_phase": np.array([e.get("action_phase", 0) for e in extra]),
         "state": np.array([e.get("system_state", "") for e in extra]),
         "memory": np.array([e.get("memory_state", [np.nan, np.nan]) for e in extra]),
         "safe_hint": np.array([e.get("external_no_contact_hint", False) for e in extra]),
         "baseline_epoch": np.array([e.get("baseline_epoch", -1) for e in extra]),
         "history_reset": np.array([e.get("response_observation", {}).get("history_reset", False) for e in extra]),
         "branch_local": np.array([e.get("branch_used_local", False) for e in extra]),
         "branch_veto": np.array([e.get("branch_local_vetoed_by_field", False) for e in extra])}
    mode = meta["trial_parameters"]["command_mode"]
    a["primary"] = primary_mask(mode, phase, ref)
    a["action"] = movement_mask(mode, phase) & known
    a["calibrated_action"] = a["action"] & (a["q"][:, 2] >= .4)
    a["safe"] = known & (a["q"][:, 2] <= -.2)
    if mode == "none": a["safe"] |= np.ones(len(t), bool)
    return meta, a


def audit(path, device):
    hashes = [{"path": (path/f).relative_to(ROOT).as_posix(), "sha256": sha(path/f)} for f in RAW_FILES]
    meta, a = load(path); mode = meta["trial_parameters"]["command_mode"]; trial = int(meta["trial_parameters"]["trial_id"])
    result = {"path": path.relative_to(ROOT).as_posix(), "device": device, "trial": trial, "mode": mode,
              "nominal_depth_mm": meta["trial_parameters"].get("z_indent_mm"), "raw_manifest": hashes,
              "prediction_frames": len(a["time"]), "valid_xyz_frames": int(a["valid"].sum()),
              "exact_magnetic_timestamp_match_fraction": float(a["matched"].mean()),
              "input_groups_dropped": meta["live_prediction_recording"].get("input_groups_dropped_by_live_queue"),
              "primary": scoring(a, a["primary"]), "calibrated_full_action": scoring(a, a["calibrated_action"]),
              "safe_or_protocol_frames": int(a["safe"].sum()),
              "outputs_in_safe_or_protocol": int((a["safe"] & a["valid"]).sum()),
              "safe_or_protocol_external_hint_frames": int((a["safe"] & a["safe_hint"]).sum()),
              "safe_or_protocol_without_hint_frames": int((a["safe"] & ~a["safe_hint"]).sum()),
              "outputs_in_safe_or_protocol_without_hint": int((a["safe"] & ~a["safe_hint"] & a["valid"]).sum()),
              "route_counts": dict(Counter(a["route"][a["valid"]].tolist())),
              "states": dict(Counter(a["state"].tolist())), "depth_bins": {}, "repeats": [],
              "jumps_full_action": jumping(a, a["action"]), "jumps_primary": jumping(a, a["primary"])}
    for lo, hi in ((0., .1), (.1, .2), (.2, .4), (.4, .6), (.6, 1.), (1., 2.), (2., 3.01)):
        m = a["action"] & (a["q"][:, 2] >= lo-1e-7) & (a["q"][:, 2] < hi-1e-7)
        if m.any():
            d = scoring(a, m); d["depth_reference_is_contact_truth"] = False
            d["output_routes"] = dict(Counter(a["route"][m & a["valid"]].tolist()))
            d["fragment_count"] = len(runs(m & a["valid"], a["time"]))
            result["depth_bins"][f"{lo:g}to{hi:g}"] = d
    if mode in ("circle_scan", "angle_scan"):
        m = a["primary"] & a["valid"]; z = a["xyz"][m, 2]
        result["constant_depth_motion_z"] = stats(z)
        # Keep the circle's vertical entry separate from genuinely constant-Z
        # motion/holds; a commanded depth change is not a jitter event.
        constant = a["action"] & np.isclose(
            a["q"][:, 2], float(meta["trial_parameters"]["z_indent_mm"]), atol=1e-5)
        result["constant_depth_full_action"] = scoring(a, constant)
        result["jumps_constant_depth"] = jumping(a, constant)
    for repeat, rm in enumerate(repeat_masks(path, a), 1):
        result["repeats"].append({"repeat": repeat, "primary": scoring(a, a["primary"] & rm),
                                  "full_action": scoring(a, a["calibrated_action"] & rm)})
    if mode == "circle_scan":
        # Per-command-segment means, fixed geometry ordering, not fitted circles.
        groups=[]
        for rm in repeat_masks(path, a):
            values={}
            for si in np.unique(a["segment"][rm & a["action"]]):
                m=rm & a["action"] & (a["segment"]==si) & a["valid"]
                ids=np.flatnonzero(m)
                if len(ids)<3: continue
                p=str(a["phase"][ids[0]])
                if not re.fullmatch(r"circle_\d+",p): continue
                values[p]={"z_mean_mm":float(a["xyz"][m,2].mean()),
                    "z_std_mm":float(a["xyz"][m,2].std()), "frames":int(m.sum())}
            groups.append(values)
        common=sorted(set(groups[0]) & set(groups[1])) if len(groups)==2 else []
        if common:
            z1=np.array([groups[0][p]["z_mean_mm"] for p in common]);z2=np.array([groups[1][p]["z_mean_mm"] for p in common])
            result["circle_segment_repeatability"]={"paired_segments":len(common),
                "z_segment_mean_correlation":float(np.corrcoef(z1,z2)[0,1]),
                "z_segment_mean_difference_mm":stats(z2-z1),
                "repeat1_segment_z_means_mm":stats(z1), "repeat2_segment_z_means_mm":stats(z2),
                "by_repeat":groups}
    holds=[]
    for si in np.unique(a["segment"][a["action"]]):
        m = a["action"] & (a["segment"] == si) & ~a["moving"] & (a["hold_age"] >= .15) & (a["hold_age"] <= .65)
        if m.sum() < 3: continue
        ids = np.flatnonzero(m); good = m & a["valid"]
        h = {"segment": int(si), "phase": str(a["phase"][ids[0]]), "command_xyz_mm": a["q"][ids[0]].tolist(),
             "score": scoring(a, m), "xyz_valid_runs": len(runs(good, a["time"])),
             "magnetic_amplitude_uT": stats(a["amp"][m]),
             "strength_counts": dict(Counter(a["strength"][m].tolist())),
             "state_counts": dict(Counter(a["state"][m].tolist())),
             "output_route_counts": dict(Counter(a["route"][good].tolist()))}
        if good.sum() >= 3:
            h["z_scatter"] = stats(a["xyz"][good, 2]); h["xy_center_scatter_mm"] = stats(np.linalg.norm(a["xyz"][good, :2]-a["xyz"][good, :2].mean(0), axis=1))
        holds.append(h)
    result["steady_holds"] = holds
    result["steady_holds_summary"] = {"holds": len(holds), "all_rejected_holds": sum(h["score"]["output_frames"] == 0 for h in holds),
        "z_std_across_holds_mm": stats([h["z_scatter"]["std"] for h in holds if "z_scatter" in h]),
        "z_p95p05_across_holds_mm": stats([h["z_scatter"]["p95_minus_p05"] for h in holds if "z_scatter" in h])}
    # Negative reference outputs, keep lift transitions distinct from safe holds.
    negative = a["known"] & (a["q"][:, 2] < 0) & a["valid"]
    result["negative_reference_outputs"] = int(negative.sum())
    result["negative_reference_z"] = stats(a["q"][negative, 2])
    f = OUT/"streams"/f"{device}_{trial:06d}.npz"; np.savez_compressed(f, **a)
    result["stream"] = f.relative_to(ROOT).as_posix()
    for entry in hashes:
        if sha(ROOT/entry["path"]) != entry["sha256"]: raise ValueError("Raw file changed during analysis")
    return result


def aggregate(rows):
    summary = {"trials": len(rows), "prediction_frames": sum(r["prediction_frames"] for r in rows),
        "valid_xyz_frames": sum(r["valid_xyz_frames"] for r in rows),
        "safe_or_protocol_frames": sum(r["safe_or_protocol_frames"] for r in rows),
        "outputs_in_safe_or_protocol": sum(r["outputs_in_safe_or_protocol"] for r in rows),
        "safe_or_protocol_external_hint_frames": sum(r["safe_or_protocol_external_hint_frames"] for r in rows),
        "safe_or_protocol_without_hint_frames": sum(r["safe_or_protocol_without_hint_frames"] for r in rows),
        "outputs_in_safe_or_protocol_without_hint": sum(r["outputs_in_safe_or_protocol_without_hint"] for r in rows)}
    for name in ("primary", "calibrated_full_action"):
        vals=[r[name] for r in rows if r[name]["reference_frames"]]
        good=[r for r in vals if r["output_frames"]]
        summary[name] = {"reference_frames": sum(r["reference_frames"] for r in vals),
                        "output_frames": sum(r["output_frames"] for r in vals),
                        "coverage_trial_macro": float(np.mean([r["coverage"] for r in vals])) if vals else None}
        for k in ("xy_raw_mean_mm", "xy_centered_mean_mm", "z_raw_mae_mm", "z_centered_mae_mm"):
            summary[name][k+"_trial_macro"] = float(np.mean([r[k] for r in good])) if good else None
    return summary


def main():
    OUT.mkdir(exist_ok=True); (OUT/"streams").mkdir(exist_ok=True)
    folders = sorted((ROOT/"data/S20260929_A").glob("Plan_*R332_*"))
    if len(folders) != 2: raise ValueError("Expected exactly both prospective R332 plans")
    report = {"schema": "R333-R332SavedOnlineAcceptance-1", "new_saved_online_test": True,
        "reference": "logged EST-time platform action interpolation; shallow command not contact truth",
        "fitting_or_selection": False, "raw_inputs_edited": False, "hardware_operated": False,
        "translation": "per-Trial mean signed XYZ residual; diagnostic only, no scale/rotation/time/curve fitting",
        "jitter": "adjacent accepted observations; command residual differenced; gaps>250ms never bridged",
        "model_permission_sha256": EXPECTED, "runs": {}, "plans": []}
    for folder in folders:
        pm = json.loads((folder/"plan_run_meta.json").read_text(encoding="utf-8"))
        if pm["status"] != "COMPLETED" or pm["completed_trials"] != 8: raise ValueError("Plan incomplete")
        device = "target" if "YELLOW" in folder.name else "source"
        report["plans"].append({"device": device, "path": folder.relative_to(ROOT).as_posix(),
            "metadata_sha256": sha(folder/"plan_run_meta.json"), "metadata": pm})
        paths = sorted(p for p in folder.glob("Trial_*") if p.is_dir())
        if len(paths) != 8: raise ValueError("Wrong Trial count")
        rows=[]
        for path in paths:
            row=audit(path, device); rows.append(row)
            print(device, row["trial"], row["mode"], json.dumps(row["primary"], ensure_ascii=False), flush=True)
        report["runs"][device]={"trials": rows, "summary": aggregate(rows)}
    save(OUT/"report.json", report)
    print("Summary", json.dumps({d: v["summary"] for d, v in report["runs"].items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
