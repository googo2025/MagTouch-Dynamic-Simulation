"""Large, frozen, offline response-phase audit. No gate, XYZ or hardware writes.

The command reference is used only AFTER inference to group diagnostics; it is
not shallow-contact truth. Missing baseline/reference/schema failures stay in
the accounting. Independent trial counts, not frames, determine uncertainty.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time as walltime

import numpy as np

from training.build_trajectory_dataset_v1 import load_magnetic, read_rows, reference_origin, command_segments, command_reference, sha
from training.audit_r226_shallow_history_gate import _logged_b0
from training.audit_r263_magnetic_evidence_uncertain_depth import evidence_scores, confirm_evidence
from training.probe_r245_joint_contact_state import causal_state_features
from training.probe_r269_hierarchical_response_state import SCALAR_IDS
from training.verify_r273_response_runtime import ROOT, HEADS, STRENGTH, CONDITIONAL, PHASE, load_model, policy
from training.response_phase_state import PhasePolicy
from training.response_phase_state_v3 import replay_rearm_phase
from training.factorized_response_phase import DirectionPolicy, replay_factorized

OUT = ROOT / "analysis/r274_frozen_phase_expansion_20260929"
GROUPS = ("weak_loading", "weak_withdrawal", "weak_hold", "supported_motion", "supported_hold",
          "pre_action_lift", "lift_residue_0to3s", "late_lift", "protocol_no_motion", "unannotated")
RAW_FILES = ("magnetic_raw.csv", "meta.json", "events.csv", "stage_commands.csv")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def freeze():
    h, s, c, p = [json.loads((d / "report.json").read_text(encoding="utf-8")) for d in (HEADS, STRENGTH, CONDITIONAL, PHASE)]
    paths = [HEADS / f"{name}_{h['head_training'][name]['selected']}.npz" for name in ("context", "direction")]
    paths.append(CONDITIONAL / f"conditional_direction_width{c['selected_width']}.npz")
    evidence_paths = {d: ROOT / f"analysis/r263_magnetic_evidence_uncertain_depth_20260929/{d}_null_model.npz" for d in ("source", "target")}
    null = {}
    for device, file in evidence_paths.items():
        with np.load(file, allow_pickle=False) as a:
            null[device] = {k: a[k].copy() for k in a.files}
    tracked = paths + list(evidence_paths.values()) + [d / "report.json" for d in (HEADS, STRENGTH, CONDITIONAL, PHASE)]
    tracked += [Path(__file__), ROOT / "training/response_phase_state.py", ROOT / "training/response_phase_state_v3.py",
                ROOT / "training/factorized_response_phase.py", ROOT / "training/probe_r245_joint_contact_state.py",
                ROOT / "training/audit_r263_magnetic_evidence_uncertain_depth.py"]
    protocol = {"schema": "R274-FrozenEvaluationProtocol-1", "frozen_before_expanded_replay": True,
        "selected_files": [{"path": v.relative_to(ROOT).as_posix(), "sha256": sha(v)} for v in tracked],
        "strength_policy": asdict(policy(PhasePolicy, s["policy"])),
        "phase_policy": asdict(policy(DirectionPolicy, p["policy"])),
        "fresh_rise_limits_uT": s["fresh_rise_limits_uT"], "activity_change_limits_uT": p["activity_change_limits_uT"],
        "sampling": "all native magnetic frames in availability order, including invalid/reset frames; no UI-valid-only selection",
        "eligibility": "finite all-5 sensor quality and >=0.7 s after explicit recorded B0; bad timestamps separately counted",
        "groups": "command reference for AFTER-inference grouping only; weak 0.02<=z<0.2 mm; supported z>=0.4 mm; safe z<=-0.5 mm",
        "domains": "Sep26+ source/target 5mm are main state-layer holdout; Sep23/24 source earlier-record stress; old 1mm tip separate stress",
        "event_rule": "contiguous weak loading/withdrawal command-group runs. All runs reported; evaluable needs >=2 eligible responsive frames spanning >=0.03 s",
        "robustness_subset": "first successfully replayed new record per main source/target x mode, selected before inspecting score, at most 24",
        "robustness_conditions": ["noise_1x", "noise_2x", "gain_0p9", "gain_1p1", "B0_offset_1x", "drop_1pct", "drop_5pct", "burst_100ms", "burst_300ms", "downsample_2", "downsample_4"],
        "uncertainty": "descriptive trial-macro metrics and 2000 trial-bootstrap intervals, not frame-independent CIs or independent-device claims",
        "fitting": "none: no expanded labels used for weights, feature scaling, whitening or threshold selection",
        "contact_truth": "unknown throughout; no XYZ generated, altered, aligned or suppressed",
        "not_zero_shot": "earlier source AND target data trained response heads and device noise calibration"}
    file = OUT / "protocol.json"
    if file.exists():
        if json.loads(file.read_text(encoding="utf-8")) != protocol:
            raise ValueError("frozen protocol mismatch; create a new audit, do not overwrite")
    else:
        save_json(file, protocol)
    return {"models": [load_model(v) for v in paths], "null": null,
            "strength_policy": policy(PhasePolicy, s["policy"]), "phase_policy": policy(DirectionPolicy, p["policy"]),
            "fresh": s["fresh_rise_limits_uT"], "change": p["activity_change_limits_uT"]}, protocol


def domain(record):
    if record["domain"] == "unseen_1mm_tip_stress":
        return "unseen_1mm_tip_stress"
    if record["domain"] == "source_5mm" and record["date"] < "S20260926_A":
        return "earlier_source_assembly_stress"
    return record["domain"]


def load_record(record):
    path = ROOT / record["path"]
    raw = load_magnetic(path / "magnetic_raw.csv")
    baseline, effective = _logged_b0(path)
    t = raw["t_available_ns"]
    field = raw["b_uT"].astype(float) - baseline
    finite = np.isfinite(field).all(1)
    eligible = finite & raw["sensor_valid"].all(1) & (t >= effective + 700_000_000)
    # Invalid observations are retained as reset/unknown frames, not imputed data.
    field[~finite] = 0.
    if not eligible.any():
        raise ValueError("no_eligible_post_B0_frames")
    row = {"path": path, "time": t, "field_uT": field, "eligible": eligible, "device": record["device"],
           "mode": record["mode"], "raw_rows": raw["raw_rows_total"], "invalid_time_rows": raw["invalid_time_rows"],
           "B0_effective_ns": effective}
    return row


def replay(row, frozen):
    x = causal_state_features(row["time"], row["field_uT"], row["eligible"])
    context, direction, binary = [m.predict(x[:, SCALAR_IDS]) for m in frozen["models"]]
    null = frozen["null"][row["device"]]
    absolute, dynamic = evidence_scores(row, null)
    response = confirm_evidence(row, absolute, dynamic, null)
    amplitude = np.linalg.norm(row["field_uT"], axis=1)
    fresh = 100.*x[:, 1] > frozen["fresh"][row["device"]]
    strength, _, _, _ = replay_rearm_phase(row["time"], context, direction, response, fresh, amplitude, row["eligible"], frozen["strength_policy"])
    phase, _, _ = replay_factorized(row["time"], strength, direction, binary, response, 100.*x[:, 1],
                                    frozen["change"][row["device"]], row["eligible"], frozen["phase_policy"])
    return {"strength": strength, "phase": phase, "response": response, "amplitude_uT": amplitude}


def post_group(row, record):
    """No output/model data are consulted, and none/manual does not mean quiet."""
    n, t = len(row["time"]), row["time"]
    labels = np.full(n, "unannotated", dtype="U24")
    q = np.full((n, 3), np.nan)
    reason = None
    if record["mode"] == "none":
        if record["none_is_protocol_control"]:
            labels[:] = "protocol_no_motion"
        return labels, q, reason
    try:
        meta = json.loads((row["path"] / "meta.json").read_text(encoding="utf-8"))
        segs = command_segments(read_rows(row["path"] / "stage_commands.csv"))
        ref = command_reference(t, segs, reference_origin(meta))
        q = ref["xyz_mm"].astype(float)
        depth, moving = q[:, 2], ref["command_moving"]
        known = np.isfinite(q).all(1)
        direction = np.zeros(n)
        for i, seg in enumerate(segs):
            if np.isfinite(seg["from_mm"][2]) and np.isfinite(seg["to_mm"][2]):
                direction[ref["command_segment_id"] == i] = seg["from_mm"][2] - seg["to_mm"][2]
        weak = known & (depth >= .02) & (depth < .2)
        labels[weak & moving & (direction > .02)] = "weak_loading"
        labels[weak & moving & (direction < -.02)] = "weak_withdrawal"
        labels[weak & ~moving & (ref["hold_age_s"] >= .1)] = "weak_hold"
        labels[known & (depth >= .4) & moving] = "supported_motion"
        labels[known & (depth >= .4) & ~moving & (ref["hold_age_s"] >= .1)] = "supported_hold"
        last_deep = np.maximum.accumulate(np.where(known & (depth >= .5), t, -1))
        safe = known & (depth <= -.5)
        labels[safe & (last_deep < 0)] = "pre_action_lift"
        labels[safe & (last_deep >= 0) & (t-last_deep <= 3_000_000_000)] = "lift_residue_0to3s"
        labels[safe & (last_deep >= 0) & (t-last_deep > 3_000_000_000)] = "late_lift"
    except (OSError, ValueError, KeyError) as e:
        reason = str(e)
    return labels, q, reason


def blocks(mask, time):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    gaps = (np.diff(ids) > 1) | (np.diff(time[ids]) <= 0) | (np.diff(time[ids]) > 250_000_000)
    return np.split(ids, np.flatnonzero(gaps)+1)


def metrics(row, result, labels, q):
    eligible = row["eligible"]
    groups = {}
    for name in GROUPS:
        all_group = labels == name
        m = eligible & all_group
        if not all_group.any():
            continue
        r = m & result["response"]
        groups[name] = {"all_frames": int(all_group.sum()), "eligible_frames": int(m.sum()), "response_frames": int(r.sum()),
            "strength_counts": np.bincount(result["strength"][m], minlength=7).tolist(),
            "phase_counts": np.bincount(result["phase"][m], minlength=6).tolist(),
            "responsive_phase_counts": np.bincount(result["phase"][r], minlength=6).tolist()}
    events = []
    for name, desired in (("weak_loading", 2), ("weak_withdrawal", 4)):
        for ids in blocks(labels == name, row["time"]):
            r = ids[eligible[ids] & result["response"][ids]]
            span = (int(row["time"][r[-1]])-int(row["time"][r[0]]))/1e9 if len(r) else 0.
            evaluable = len(r) >= 2 and span >= .03
            hits = r[result["phase"][r] == desired]
            events.append({"group": name, "all_frames": len(ids), "eligible_frames": int(eligible[ids].sum()),
                "responsive_frames": len(r), "responsive_span_s": span, "evaluable": bool(evaluable),
                "hit": bool(len(hits)), "delay_from_first_response_s": None if not len(hits) else (int(row["time"][hits[0]])-int(row["time"][r[0]]))/1e9,
                "command_xy_mm": q[ids[0], :2].tolist(), "start_time_ns": int(row["time"][ids[0]])})
    return groups, events


def scalar_rates(groups):
    def counts(names, key):
        arrays = [np.array(groups[name][key], float) for name in names if name in groups]
        return sum(arrays, np.zeros(7 if key == "strength_counts" else 6))
    rates = {}
    for name, code in (("weak_loading", 2), ("weak_withdrawal", 4)):
        v = counts((name,), "responsive_phase_counts")
        if v.sum():
            rates[name+"_responsive_frame_match"] = float(v[code]/v.sum())
    v = counts(("supported_motion", "supported_hold"), "strength_counts")
    if v.sum():
        rates["supported_state_preservation"] = float(v[6]/v.sum())
        rates["supported_as_weak"] = float(v[2:5].sum()/v.sum())
    for name, members in (("safe_lift", ("pre_action_lift", "lift_residue_0to3s", "late_lift")),
                          ("protocol_no_motion", ("protocol_no_motion",))):
        v = counts(members, "phase_counts")
        if v.sum():
            rates[name+"_loading_fraction"] = float(v[2]/v.sum())
            rates[name+"_any_action_fraction"] = float(v[2:5].sum()/v.sum())
    return rates


def pooled(items):
    g = {}; events = []
    for item in items:
        for name, values in item["groups"].items():
            slot = g.setdefault(name, {k: [0]*len(v) if isinstance(v, list) else 0 for k, v in values.items()})
            for k, v in values.items():
                slot[k] = (np.array(slot[k])+np.array(v)).tolist() if isinstance(v, list) else slot[k]+v
        events.extend(item["events"])
    er = {}
    for name in ("weak_loading", "weak_withdrawal"):
        v = [e for e in events if e["group"] == name]
        qualified = [e for e in v if e["evaluable"]]
        delay = [e["delay_from_first_response_s"] for e in qualified if e["hit"]]
        er[name] = {"all_runs": len(v), "evaluable_runs": len(qualified), "hits": sum(e["hit"] for e in qualified),
            "too_short_or_no_response_runs": len(v)-len(qualified),
            "delay_median_s": None if not delay else float(np.median(delay)),
            "delay_p95_s": None if not delay else float(np.percentile(delay, 95))}
    rng = np.random.default_rng(274)
    rates = [scalar_rates(i["groups"]) for i in items]
    macro = {}
    for name in sorted({k for r in rates for k in r}):
        values = np.array([r[name] for r in rates if name in r])
        bootstrap = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
        macro[name] = {"trials_with_denominator": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()), "trial_bootstrap_95pct": np.percentile(bootstrap, [2.5, 97.5]).tolist()}
    return {"trials": len(items), "raw_frames": sum(i["raw_rows"] for i in items),
            "replayed_frames": sum(i["frames"] for i in items), "eligible_frames": sum(i["eligible_frames"] for i in items),
            "groups": g, "pooled_frame_rates": scalar_rates(g), "trial_macro": macro, "weak_events": er}


def prefix_check(row, frozen, full):
    cut = len(row["time"])//2
    if cut < 2:
        return 0
    prefix = {k: (v[:cut] if isinstance(v, np.ndarray) and len(v) == len(row["time"]) else v) for k, v in row.items()}
    p = replay(prefix, frozen)
    for name in ("strength", "phase", "response"):
        np.testing.assert_array_equal(p[name], full[name][:cut])
    return 1


def perturb(row, condition, null, seed):
    rng = np.random.default_rng(seed)
    altered = dict(row)
    field = row["field_uT"].copy(); eligible = row["eligible"].copy()
    inverse = np.linalg.inv(null["whitener"])
    if condition.startswith("noise_"):
        level = 1. if condition == "noise_1x" else 2.
        field += level*(rng.standard_normal(field.shape) @ inverse)
    elif condition.startswith("gain_"):
        field *= .9 if condition == "gain_0p9" else 1.1
    elif condition == "B0_offset_1x":
        vector = rng.standard_normal(15)
        field += (vector / np.linalg.norm(vector) * np.sqrt(15)) @ inverse
    elif condition.startswith("drop_"):
        eligible &= rng.random(len(eligible)) >= (.01 if condition == "drop_1pct" else .05)
    elif condition.startswith("burst_"):
        duration = 100_000_000 if condition == "burst_100ms" else 300_000_000
        t = row["time"]
        eligible &= ~(((t-t[0]) % 5_000_000_000 >= 2_000_000_000) & ((t-t[0]) % 5_000_000_000 < 2_000_000_000+duration))
    elif condition.startswith("downsample_"):
        stride = int(condition.split("_")[-1])
        altered["time"] = row["time"][::stride]
        field, eligible = field[::stride], eligible[::stride]
    else:
        raise ValueError(condition)
    altered["field_uT"], altered["eligible"] = field, eligible
    return altered


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--robustness", action="store_true")
    args = parser.parse_args()
    inventory = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))["trials"]
    frozen, protocol = freeze()
    if args.robustness:
        return robustness(inventory, frozen, protocol)
    report_path = OUT / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    results, failures, exclusions, manifest, seen = [], [], [], [], {}
    streams = OUT / "streams"; streams.mkdir(exist_ok=True)
    checkpoints = OUT / "trial_reports"; checkpoints.mkdir(exist_ok=True)
    start = walltime.monotonic(); prefix = 0
    for index, rec in enumerate(inventory):
        key = hashlib.sha256(rec["path"].encode()).hexdigest()[:16]
        if not rec["initially_readable"]:
            exclusions.append({"path": rec["path"], "domain": domain(rec), "reason": "missing_raw_events_or_explicit_B0"}); continue
        hashes = [{"path": (ROOT / rec["path"] / f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT / rec["path"] / f)}
                  for f in RAW_FILES if (ROOT / rec["path"] / f).is_file()]
        manifest.extend(hashes)
        raw_sha = next(v["sha256"] for v in hashes if v["path"].endswith("magnetic_raw.csv"))
        if raw_sha in seen:
            exclusions.append({"path": rec["path"], "reason": "duplicate_raw_digest", "same_as": seen[raw_sha]}); continue
        seen[raw_sha] = rec["path"]
        file = checkpoints / f"{key}.json"
        if file.exists():
            item = json.loads(file.read_text(encoding="utf-8"))
            if item["input_hashes"] != hashes:
                raise ValueError("raw input changed since checkpoint")
            results.append(item); prefix += item["prefix_checked"]; continue
        try:
            row = load_record(rec)
            result = replay(row, frozen)
            # Only now may reference data be loaded for diagnostic grouping.
            labels, q, reference_error = post_group(row, rec)
            groups, events = metrics(row, result, labels, q)
            check = prefix_check(row, frozen, result) if len(results) % 25 == 0 else 0
            prefix += check
            item = {"path": rec["path"], "domain": domain(rec), "device": rec["device"], "date": rec["date"],
                "plan": rec["plan"], "mode": rec["mode"], "original_36_record_scope": rec["original_36_record_scope"],
                "key": key, "frames": len(row["time"]), "raw_rows": row["raw_rows"], "invalid_time_rows": row["invalid_time_rows"],
                "eligible_frames": int(row["eligible"].sum()), "ineligible_replayed_frames": int((~row["eligible"]).sum()),
                "parameters": rec["parameters"], "groups": groups, "events": events, "reference_error": reference_error,
                "input_hashes": hashes, "prefix_checked": check,
                "state_counts_all": np.bincount(result["strength"], minlength=7).tolist(),
                "phase_counts_all": np.bincount(result["phase"], minlength=6).tolist()}
            np.savez_compressed(streams / f"{key}.npz", time_ns=row["time"], eligible=row["eligible"],
                strength=result["strength"], phase=result["phase"], response=result["response"], amplitude_uT=result["amplitude_uT"],
                diagnostic_command_xyz_mm=q, diagnostic_labels=labels, physical_contact_truth=np.full(len(labels), -1, np.int8))
            save_json(file, item); results.append(item)
        except (OSError, ValueError, KeyError) as e:
            failures.append({"path": rec["path"], "domain": domain(rec), "reason": str(e)})
        if len(results) % 10 == 0:
            print(f"R274 replayed {len(results)} / {len(inventory)}, failures {len(failures)}, elapsed {walltime.monotonic()-start:.1f}s", flush=True)
    unique = {v["path"]: v for v in manifest}
    checks = [{**v, "unchanged": sha(ROOT/v["path"]) == v["sha256"]} for v in unique.values()]
    assert all(v["unchanged"] for v in checks)
    assert all(sha(ROOT/v["path"]) == v["sha256"] for v in protocol["selected_files"])
    new = [v for v in results if not v["original_36_record_scope"]]
    by_domain = {d: pooled([v for v in new if v["domain"] == d]) for d in sorted({v["domain"] for v in new})}
    by_date = {d: pooled([v for v in new if v["date"] == d]) for d in sorted({v["date"] for v in new})}
    by_mode = {d+":"+m: pooled([v for v in new if v["domain"] == d and v["mode"] == m])
        for d, m in sorted({(v["domain"], str(v["mode"])) for v in new})}
    report = {"schema": "R274-FrozenPhaseExpansion-1", "status": "offline frozen-weight audit, not a deployed gate or XYZ model",
        "candidate_trials": len(inventory), "new_scope_candidates": sum(not r["original_36_record_scope"] for r in inventory),
        "successful_trials": len(results), "new_scope_successful_trials": len(new), "original_scope_native_replays": len(results)-len(new),
        "new_raw_frames": sum(v["raw_rows"] for v in new), "all_replayed_frames": sum(v["frames"] for v in results),
        "new_eligible_frames": sum(v["eligible_frames"] for v in new), "prefix_checks": prefix,
        "input_file_hash_checks": len(checks), "all_input_files_unchanged": True, "all_frozen_models_codes_unchanged": True,
        "by_domain": by_domain, "by_date": by_date, "by_domain_mode": by_mode,
        "exclusions": exclusions, "failures": failures, "trials": results,
        "warning": "Earlier target data trained heads/null calibration: NOT zero-shot. New records untouched by state-layer fitting, but retrospectively evaluated and some inspected for earlier XYZ development. Two physical devices, repeated trials are not independent devices. Native clock differs from original UI prediction clock. Nominal weak-depth/direction groups are not physical contact ground truth.",
        "alignment": "no XYZ estimated or scored; no coordinate alignment. Recorded B0 subtraction only; no test-data learned baseline or gain correction."}
    save_json(OUT / "raw_input_manifest.json", checks)
    save_json(report_path, report)
    with (OUT / "per_trial_rates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        rows = [{"path": v["path"], "domain": v["domain"], "date": v["date"], "mode": v["mode"], "old_scope": v["original_36_record_scope"],
                 "raw_frames": v["raw_rows"], "eligible_frames": v["eligible_frames"], **scalar_rates(v["groups"])} for v in results]
        keys = sorted({k for v in rows for k in v}); writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k: report[k] for k in ("successful_trials", "new_scope_successful_trials", "new_raw_frames", "new_eligible_frames", "prefix_checks")}), flush=True)
    for d, v in by_domain.items():
        print(d, json.dumps({"trials": v["trials"], "rates": v["pooled_frame_rates"], "events": v["weak_events"]}), flush=True)


def robustness(inventory, frozen, protocol):
    file = OUT / "robustness_report.json"
    if file.exists():
        raise FileExistsError(file)
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    new = [v for v in report["trials"] if not v["original_36_record_scope"] and v["domain"] in ("source_5mm", "target_5mm")]
    selected, seen = [], set()
    for v in new:
        k = v["domain"], v["mode"]
        if k not in seen:
            seen.add(k); selected.append(v)
        if len(selected) >= 24:
            break
    lookup = {v["path"]: v for v in inventory}; comparisons = []
    for index, item in enumerate(selected):
        rec = lookup[item["path"]]; row = load_record(rec)
        base = replay(row, frozen); labels, q, _ = post_group(row, rec)
        g, e = metrics(row, base, labels, q)
        comparisons.append({"path": item["path"], "domain": item["domain"], "mode": item["mode"], "condition": "unperturbed",
            "groups": g, "events": e, "raw_rows": row["raw_rows"], "frames": len(row["time"]), "eligible_frames": int(row["eligible"].sum()),
            "unknown_all_fraction": float(np.mean(base["strength"] == 0)), "phase_disagreement_vs_base": 0.})
        for j, condition in enumerate(protocol["robustness_conditions"]):
            altered = perturb(row, condition, frozen["null"][row["device"]], 274_000+index*100+j)
            result = replay(altered, frozen)
            labs, reference, _ = post_group(altered, rec)
            groups, events = metrics(altered, result, labs, reference)
            step = int(condition.split("_")[-1]) if condition.startswith("downsample_") else 1
            comparisons.append({"path": item["path"], "domain": item["domain"], "mode": item["mode"], "condition": condition,
                "groups": groups, "events": events, "raw_rows": row["raw_rows"], "frames": len(altered["time"]),
                "eligible_frames": int(altered["eligible"].sum()), "unknown_all_fraction": float(np.mean(result["strength"] == 0)),
                "phase_disagreement_vs_base": float(np.mean(result["phase"] != base["phase"][::step]))})
        print("R274 robustness", index+1, "/", len(selected), item["domain"], item["mode"], flush=True)
    conditions = {}
    for c in ("unperturbed", *protocol["robustness_conditions"]):
        items = [v for v in comparisons if v["condition"] == c]
        conditions[c] = pooled(items)
        conditions[c]["trial_mean_unknown_all_fraction"] = float(np.mean([v["unknown_all_fraction"] for v in items]))
        conditions[c]["trial_mean_phase_disagreement_vs_base"] = float(np.mean([v["phase_disagreement_vs_base"] for v in items]))
    save_json(file, {"schema": "R274-FrozenSyntheticRobustness-1", "selected_real_trials": len(selected),
        "synthetic_conditions": len(protocol["robustness_conditions"]), "conditions": conditions, "trials": comparisons,
        "warning": "Synthetic stress, not additional independent experiments. Noise drawn from frozen earlier null covariance; no perturbation chosen using resulting accuracy. No test-time refit or compensation."})
    assert all(sha(ROOT/v["path"]) == v["sha256"] for v in protocol["selected_files"])


if __name__ == "__main__":
    main()
