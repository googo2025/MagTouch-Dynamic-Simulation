"""Bounded integrated replay, fixed before target inspection; no training/device IO.

Two complete 9/28 replication plans and all R208 shallow/history protocols.
Commands only group scores AFTER inference, never affect runtime validity.
Both decoders run the same raw observations; this is not a new online test.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code"))
from live_unified_tactile_r330 import UnifiedTactileEngine, GATE_NAME
from live_field_memory_r133 import CoverageBranchGateEngine
from replay_backtest import load_profile
from live_algorithm_ui_r126 import PROFILE_KEY
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder, ARTIFACT
from training.unified_response_runtime_r330 import StreamingSourceResponse, digest
from training.audit_r274_frozen_phase_expansion import load_record, post_group, save_json, RAW_FILES
from training.audit_r133_online_20260928 import score
from training.test_unified_tactile_r330 import offline_response
from training.audit_r276_condition_adaptation import cohort

OUT = ROOT/"analysis/r330_unified_system_20260929"
INVENTORY = ROOT/"analysis/r274_frozen_phase_expansion_20260929/inventory.json"


def selected_records():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    return sorted([r for r in inventory if r["device"] in ("source", "target") and r["initially_readable"]
        and ("R133_RETRAIN_" in r["plan"] or "R208_SOURCE" in r["plan"] or "R208_TARGET" in r["plan"])],
        key=lambda r: (r["device"] != "source", r["path"]))


def create_freeze(records):
    paths = [Path(__file__), ROOT/"training/unified_response_runtime_r330.py",
             ROOT/"training/unified_tactile_policy_r330.py", ROOT/"code/live_unified_tactile_r330.py",
             ROOT/"code/magtouch_acquisition_ui_v1_3_4_R330.py", R92_PROFILE_PATH, ARTIFACT]
    protocol = {"schema": "R330-IntegratedSelectiveXYZ-1", "gate": GATE_NAME,
        "old_gate_applied": False, "response_selected": "Source validation-selected R300, not target-best R304",
        "fitting_or_parameter_selection_in_this_audit": False,
        "target_fit_and_selection_rows": 0, "independent_online_test": False,
        "cohorts": "whole R133_RETRAIN and R208 Source/Target plans; no score-based trial selection",
        "command_reference": "post-inference diagnostics only; extreme shallow command depth not contact truth",
        "comparison": "same raw availability-time stream; frozen legacy R133 fast gate vs standalone R330",
        "translation": "per-Trial signed mean residual removed for shape diagnostic only; not runtime calibration",
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
        "records": [r["path"] for r in records]}
    file = OUT/"freeze.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8")) != protocol:
        raise ValueError("R330 frozen audit changed; create a new version instead of silently overwriting")
    if not file.exists(): save_json(file, protocol)
    return protocol


def scored(xyz, valid, q, mask):
    denominator = mask & np.isfinite(q).all(1)
    accepted = denominator & valid
    return {"reference_frames": int(denominator.sum()), "output_frames": int(accepted.sum()),
        "coverage": float(accepted.sum()/denominator.sum()) if denominator.any() else None,
        "error_on_output_only": score(xyz[accepted], q[accepted]) if accepted.any() else None}


def audit(rec, profile):
    row = load_record(rec)
    observer = StreamingSourceResponse()
    engine = UnifiedTactileEngine(profile, profile["defaults"]["threshold"],
        profile["defaults"]["min_cosine"], profile["defaults"]["smoothing_ms"], 70.,
        CoverageGuardedNeuralDecoder.from_report(), observer=observer)
    legacy = CoverageBranchGateEngine(profile, profile["defaults"]["threshold"],
        profile["defaults"]["min_cosine"], profile["defaults"]["smoothing_ms"], 70.,
        "快速门控（阈值/形状）", CoverageGuardedNeuralDecoder.from_report())
    engine.set_baseline(np.zeros(15)); engine.baseline_effective_ns = row["B0_effective_ns"]
    legacy.set_baseline(np.zeros(15))
    n = len(row["time"]); new_xyz, old_xyz = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    new_valid, old_valid = np.zeros(n, bool), np.zeros(n, bool)
    state, strength, phase, response = np.zeros(n, dtype="U32"), np.zeros(n, int), np.zeros(n, int), np.zeros(n, bool)
    reasons = Counter(); timings = []; ready_time = None
    start = time.monotonic()
    for i, (t, b, ok) in enumerate(zip(row["time"], row["field_uT"], row["eligible"])):
        t0 = time.perf_counter()
        r = engine.push(t, b, eligible=ok)
        timings.append((time.perf_counter()-t0)*1000.)
        obs = r["extra"]["response_observation"]
        strength[i], phase[i], response[i] = obs["strength"], obs["phase"], obs["response"]
        state[i] = r["extra"]["system_state"]
        reasons.update(r["extra"]["rejection_reasons"])
        if r["valid"]: new_valid[i] = True; new_xyz[i] = r["xyz"]
        if ok:
            old = legacy.push(t, b)
            if old["valid"]: old_valid[i] = True; old_xyz[i] = old["xyz"]
        else:
            legacy._reset_dynamic(); legacy.buffer.clear(); legacy.anchor = None
    reference = offline_response(row["time"], row["field_uT"], row["eligible"], observer)
    for key, actual in (("strength", strength), ("phase", phase), ("response", response)):
        np.testing.assert_array_equal(actual, reference[key], err_msg=rec["path"]+":"+key)
    # Only here are platform annotations introduced.
    labels, q, annotation_reason = post_group(row, rec)
    core = np.isin(labels, ("supported_motion", "supported_hold"))
    lift = np.isin(labels, ("pre_action_lift", "lift_residue_0to3s", "late_lift", "protocol_no_motion"))
    metrics = {"core": {"legacy": scored(old_xyz, old_valid, q, core), "unified": scored(new_xyz, new_valid, q, core)},
        "paired_core": {}, "depth_bins": {}, "safe_lift_or_protocol_control_frames": int(lift.sum()),
        "legacy_outputs_in_lift_or_control": int((old_valid & lift).sum()),
        "unified_outputs_in_lift_or_control": int((new_valid & lift).sum()),
        "old_valid_core_new_rejected": int((core & old_valid & ~new_valid).sum()),
        "new_valid_core_old_rejected": int((core & new_valid & ~old_valid).sum()),
        "system_states": dict(Counter(state.tolist())), "rejection_reasons": dict(reasons)}
    paired = core & old_valid & new_valid
    for name, xyz in (("legacy", old_xyz), ("unified", new_xyz)):
        metrics["paired_core"][name] = scored(xyz, paired, q, paired)
    for lo, hi in ((0., .1), (.1, .2), (.2, .3), (.3, .4), (.4, .6), (.6, 1.), (1., 2.), (2., 3.01)):
        mask = row["eligible"] & np.isfinite(q).all(1) & (q[:, 2] >= lo) & (q[:, 2] < hi)
        if mask.any(): metrics["depth_bins"][f"{lo:g}to{hi:g}"] = {
            "legacy": scored(old_xyz, old_valid, q, mask), "unified": scored(new_xyz, new_valid, q, mask)}
    stem = digest(row["path"]/"meta.json")[:16]
    np.savez_compressed(OUT/"streams"/(stem+".npz"), time_ns=row["time"], eligible=row["eligible"],
        q=q, diagnostic_group=labels, system_state=state, strength=strength, phase=phase,
        response=response, xyz_valid=new_valid, xyz_mm=new_xyz, legacy_valid=old_valid, legacy_xyz_mm=old_xyz)
    return {"path": rec["path"], "device": rec["device"], "mode": rec["mode"],
        "response_training_cohort": cohort(rec), "frames": n, "annotation_reason": annotation_reason,
        "stream": "streams/"+stem+".npz", "metrics": metrics, "exact_response_equivalence": True,
        "runtime_ms_median_p95_p99": np.percentile(timings, [50, 95, 99]).tolist(),
        "runtime_ms_max": float(max(timings)), "wall_seconds": time.monotonic()-start}


def aggregate(rows):
    result = {}
    for device in ("source", "target"):
        group = [r for r in rows if r["device"] == device]
        if not group: continue
        s = {"trials": len(group), "raw_frames": sum(r["frames"] for r in group)}
        for key in ("safe_lift_or_protocol_control_frames", "legacy_outputs_in_lift_or_control", "unified_outputs_in_lift_or_control",
                    "old_valid_core_new_rejected", "new_valid_core_old_rejected"):
            s[key] = sum(r["metrics"][key] for r in group)
        for name in ("legacy", "unified"):
            contacts = [r["metrics"]["core"][name] for r in group if r["metrics"]["core"][name]["reference_frames"]]
            s[name] = {"core_reference_frames": sum(r["reference_frames"] for r in contacts),
                "core_output_frames": sum(r["output_frames"] for r in contacts),
                "core_coverage_trial_macro": float(np.mean([r["coverage"] for r in contacts])) if contacts else None}
            errors = [r["error_on_output_only"] for r in contacts if r["error_on_output_only"]]
            s[name]["output_error_trial_macro"] = {key: float(np.mean([r[key] for r in errors])) for key in (
                "xy_raw_mean_mm", "xy_translation_removed_mean_mm", "z_raw_mae_mm", "z_translation_removed_mae_mm")} if errors else None
        result[device] = s
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--device", choices=("source", "target", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0); args = parser.parse_args()
    OUT.mkdir(exist_ok=True); (OUT/"streams").mkdir(exist_ok=True)
    records = selected_records(); create_freeze(records)
    records = [r for r in records if args.device == "all" or r["device"] == args.device]
    if args.limit: records = records[:args.limit]
    file = OUT/("report_"+args.device+("_smoke" if args.limit else "")+".json")
    if file.exists(): raise FileExistsError(file)
    inputs = [(ROOT/r["path"]/name, digest(ROOT/r["path"]/name)) for r in records for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY); results = []
    for i, rec in enumerate(records):
        result = audit(rec, profile); results.append(result)
        print(f'{i+1}/{len(records)} {rec["device"]} {Path(rec["path"]).name} frames={result["frames"]} sec={result["wall_seconds"]:.1f}', flush=True)
        save_json(OUT/("progress_"+args.device+".json"), {"completed": len(results), "total": len(records), "rows": results})
    for path, expected in inputs:
        if digest(path) != expected: raise ValueError("Raw input changed during read-only replay: "+str(path))
    save_json(file, {"schema": "R330-IntegratedReplayResults-1", "new_online_test": False,
        "target_parameter_fit_or_selection": False, "translation_removed_is_diagnostic_only": True,
        "rejects_are_not_zero_error": True, "source_test_status": "diagnostic audit, not new blind trial",
        "raw_files_unchanged": True, "summary": aggregate(results), "rows": results,
        "inputs": [{"path": p.relative_to(ROOT).as_posix(), "sha256": h} for p, h in inputs]})
    print(json.dumps(aggregate(results), ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
