"""Frozen source-only replay, command diagnostics strictly after inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from training.train_r275_source_only_response import ROOT, OUT, INVENTORY
from training.audit_r274_frozen_phase_expansion import (load_record, post_group, metrics, pooled, scalar_rates,
    replay, domain, save_json, prefix_check, perturb)
from training.verify_r273_response_runtime import load_model, policy
from training.response_phase_state import PhasePolicy
from training.factorized_response_phase import DirectionPolicy
from training.diagnose_r274_whole_action import whole_actions, summary as whole_summary
from training.build_trajectory_dataset_v1 import sha


def load_frozen(folder=OUT):
    protocol = json.loads((folder / "freeze.json").read_text(encoding="utf-8"))
    assert protocol["frozen_before_target_replay"] and protocol["target_fit_rows"] == 0
    for item in protocol["selected_files"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("source-only frozen input changed: " + item["path"])
    with np.load(ROOT / protocol["null"], allow_pickle=False) as a:
        null = {k: a[k].copy() for k in a.files}
    # Same frozen source covariance and limits, regardless of domain.
    frozen = {"models": [load_model(ROOT / p) for p in protocol["models"]],
        "null": {d: null for d in ("source", "target")},
        "fresh": {d: protocol["fresh_rise_limit_uT"] for d in ("source", "target")},
        "change": {d: protocol["activity_change_limit_uT"] for d in ("source", "target")},
        "strength_policy": policy(PhasePolicy, protocol["strength_policy"]),
        "phase_policy": policy(DirectionPolicy, protocol["phase_policy"])}
    return frozen, protocol


def acceptance(by_domain, whole, criteria):
    result = {}
    for d in ("source_5mm", "target_5mm"):
        rates = by_domain[d]["pooled_frame_rates"]
        checks = {}
        for name in ("loading", "withdrawal"):
            g = whole[d][name]
            r = g["responsive_whole_action_hits"] / max(g["responsive_runs"], 1)
            checks[name] = {"rate": r, "pass": bool(r >= criteria[f"whole_responsive_{name}_fraction_min"])}
        for key, bound, lower in (("supported_state_preservation", "supported_preservation_min", True),
                                 ("safe_lift_loading_fraction", "safe_lift_loading_fraction_max", False),
                                 ("protocol_no_motion_any_action_fraction", "protocol_no_motion_any_action_fraction_max", False)):
            v = rates.get(key)
            checks[key] = {"rate": v, "pass": bool(v is not None and (v >= criteria[bound] if lower else v <= criteria[bound]))}
        result[d] = {"checks": checks, "pass": all(v["pass"] for v in checks.values())}
    return {"criteria_frozen_before_replay": criteria, "by_domain": result,
            "may_start_source_only_dropout_reentry_optimization": all(v["pass"] for v in result.values())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--robustness", action="store_true")
    args = parser.parse_args()
    frozen, protocol = load_frozen()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    if args.robustness:
        return robustness(inventory, frozen, protocol)
    if (OUT / "report.json").exists():
        raise FileExistsError(OUT / "report.json")
    streams = OUT / "streams"; streams.mkdir(exist_ok=True)
    checks = OUT / "trial_reports"; checks.mkdir(exist_ok=True)
    prior = json.loads((ROOT / "analysis/r274_frozen_phase_expansion_20260929/report.json").read_text(encoding="utf-8"))
    prior_hash = {r["path"]: r["input_hashes"] for r in prior["trials"]}
    results, failures, excluded = [], [], []
    start = time.monotonic()
    for i, rec in enumerate(inventory):
        if not rec["initially_readable"]:
            excluded.append({"path": rec["path"], "reason": "missing_raw_events_or_explicit_B0"}); continue
        key = hashlib.sha256(rec["path"].encode()).hexdigest()[:16]
        file = checks / f"{key}.json"
        hashes = prior_hash[rec["path"]]
        for h in hashes:
            if sha(ROOT / h["path"]) != h["sha256"]:
                raise ValueError("raw input changed since preceding audit")
        if file.exists():
            item = json.loads(file.read_text(encoding="utf-8"))
            assert item["input_hashes"] == hashes
            results.append(item); continue
        try:
            row = load_record(rec); prediction = replay(row, frozen)
            labels, q, ref_error = post_group(row, rec)
            groups, events = metrics(row, prediction, labels, q)
            a = {"time_ns": row["time"], "eligible": row["eligible"], "diagnostic_command_xyz_mm": q,
                 "diagnostic_labels": labels, **prediction}
            action_events = whole_actions(a)
            item = {"path": rec["path"], "key": key, "domain": domain(rec), "device": rec["device"], "mode": rec["mode"],
                    "date": rec["date"], "plan": rec["plan"], "original_36_record_scope": rec["original_36_record_scope"],
                    "frames": len(row["time"]), "raw_rows": row["raw_rows"], "eligible_frames": int(row["eligible"].sum()),
                    "invalid_time_rows": row["invalid_time_rows"], "groups": groups, "events": events,
                    "whole_actions": action_events, "reference_error": ref_error, "input_hashes": hashes,
                    "prefix_checked": prefix_check(row, frozen, prediction) if len(results) % 25 == 0 else 0}
            np.savez_compressed(streams / f"{key}.npz", **a, physical_contact_truth=np.full(len(labels), -1, np.int8))
            save_json(file, item); results.append(item)
        except (OSError, ValueError, KeyError) as e:
            failures.append({"path": rec["path"], "reason": str(e)})
        if len(results) % 10 == 0:
            print(f"source-only replay {len(results)}/{len(prior_hash)}, failure={len(failures)}, {time.monotonic()-start:.1f}s", flush=True)
    new = [r for r in results if not r["original_36_record_scope"]]
    by_domain = {d: pooled([r for r in new if r["domain"] == d]) for d in sorted({r["domain"] for r in new})}
    whole = {d: whole_summary([{"events": r["whole_actions"]} for r in new if r["domain"] == d]) for d in by_domain}
    by_mode = {d+":"+m: pooled([r for r in new if r["domain"] == d and r["mode"] == m])
               for d, m in sorted({(r["domain"], r["mode"]) for r in new})}
    result = {"schema": "R275-FrozenSourceOnlyAudit-1", "successful_trials": len(results), "new_scope_successful_trials": len(new),
              "failures": failures, "exclusions": excluded, "by_domain": by_domain, "by_domain_mode": by_mode,
              "whole_actions_by_domain": whole, "prefix_checks": sum(r["prefix_checked"] for r in results),
              "acceptance": acceptance(by_domain, whole, protocol["conditional_next_step_acceptance"]),
              "raw_input_files_unchanged": True, "target_fit_rows": 0, "target_threshold_or_normalization_fit_rows": 0,
              "semantics": "response-phase observer only; zero target labels/statistics for fitting or selection; recorded B0 input allowed; no new XYZ or contact truth, UI/hardware/paper unchanged; retrospective target transfer, not prospective blind",
              "whole_action_vs_shallow_window": "whole action hit can occur later than nominal shallow segment; .02-.2 and .02-.4 diagnostics separately retained; unsupported events never treated as zero-error",
              "trials": results}
    save_json(OUT / "report.json", result)
    with (OUT / "per_trial_rates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        rr = [{"path": r["path"], "domain": r["domain"], "date": r["date"], "mode": r["mode"],
               "old_scope": r["original_36_record_scope"], **scalar_rates(r["groups"])} for r in results]
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rr for k in r})); w.writeheader(); w.writerows(rr)
    load_frozen()
    for d, v in by_domain.items():
        print(d, json.dumps({"trials": v["trials"], "rates": v["pooled_frame_rates"], "shallow_events": v["weak_events"], "whole": whole[d]}), flush=True)
    print("acceptance", json.dumps(result["acceptance"]), flush=True)


def robustness(inventory, frozen, protocol):
    dest = OUT / "robustness_report.json"
    if dest.exists():
        raise FileExistsError(dest)
    base = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    modes, selected = set(), []
    good = {r["path"] for r in base["trials"]}
    for rec in inventory:
        pair = domain(rec), rec["mode"]
        if rec["path"] in good and not rec["original_36_record_scope"] and pair[0] in ("source_5mm", "target_5mm") and pair not in modes:
            modes.add(pair); selected.append(rec)
    conditions = ("baseline", "noise_1x", "noise_2x", "gain_0p9", "gain_1p1", "B0_offset_1x", "drop_1pct", "drop_5pct", "burst_100ms", "burst_300ms", "downsample_2", "downsample_4")
    results = []
    for i, rec in enumerate(selected):
        row = load_record(rec)
        for j, c in enumerate(conditions):
            altered = row if c == "baseline" else perturb(row, c, frozen["null"]["source"], 275000+i*31+j)
            pred = replay(altered, frozen); labels, q, _ = post_group(altered, rec)
            groups, events = metrics(altered, pred, labels, q)
            results.append({"path": rec["path"], "domain": domain(rec), "mode": rec["mode"], "condition": c,
                "raw_rows": altered["raw_rows"], "frames": len(altered["time"]), "eligible_frames": int(altered["eligible"].sum()),
                "groups": groups, "events": events, "unknown_eligible_frames": int(np.sum(altered["eligible"] & (pred["phase"] == 0)))})
        print("source-only robustness", i+1, "/", len(selected), flush=True)
    save_json(dest, {"schema": "R275-SourceOnlyRobustness-1", "selected_records": [r["path"] for r in selected],
        "selection": "first successfully replayed new record per source/target x mode, same rule as R274; no best-example selection",
        "noise": "both domains perturbed using SOURCE-only null covariance; seeds predeclared, not target-fitted",
        "by_condition": {c: pooled([r for r in results if r["condition"] == c]) for c in conditions}, "trials": results})


if __name__ == "__main__":
    main()
