"""SOURCE-only release of erroneous supported context, using frozen R288.

No new neural expert or model fitting. A confirmed residual score may
reclassify supported strength, but only source-qualified rules are frozen.
This is magnetic context, neither physical contact nor XYZ permission.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r288_ordered_action_context import (
    ROOT, INVENTORY, load_prior_args, prior_infer, protected_veto, load_source,
    source_partition, VAL_GAINS, TAU_S, augmented_signal_row, predict_sequence,
    load_model, score_prediction, aggregate, RAW_FILES, save_json, sha)
from training.probe_r288_ordered_action_context import OUT as PRIOR_OUT
from training.verify_r288_ordered_action_context import verify_freeze as prior_freeze

OUT = ROOT/"analysis/r291_source_supported_context_release_20260929"
SAFE = ("pre_action_lift", "lift_residue_0to3s", "late_lift", "protocol_no_motion")


def release_supported(row, prior, scores, limit=.995, confirm_s=.03):
    scores = np.asarray(scores, float); t = row["time"]; eligible = row["eligible"]; n = len(t)
    if scores.shape != (n, 2) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(abs(scores.sum(1)-1) > 1e-5):
        raise ValueError("aligned normalized frozen context scores required")
    if not 0 < limit < 1 or not np.isfinite(confirm_s) or confirm_s < 0: raise ValueError("valid causal policy required")
    negative = eligible & (scores[:, 0] >= limit)
    if n:
        reset = np.r_[True, ~negative[1:] | ~negative[:-1] | ~eligible[1:] | ~eligible[:-1] |
                      (np.diff(t) <= 0) | (np.diff(t) > 250_000_000)]
        starts = np.maximum.accumulate(np.where(reset, np.arange(n), 0))
        confirmed = negative & (np.arange(n)-starts >= 1) & (t-t[starts] >= confirm_s*1e9)
    else: confirmed = np.zeros(0, bool)
    changed = confirmed & (prior["strength"] == 6)
    strength = prior["strength"].copy(); phase = prior["phase"].copy()
    strength[changed] = np.where(row["response"][changed], 5, 1)
    phase[changed] = np.where(row["response"][changed], 5, 1)
    phase[~eligible] = 0
    return dict(prior, strength=strength, phase=phase, supported_context_released=changed,
                supported_release_residual_score=scores[:, 0])


def safety(row, old, new):
    labels, eligible = row["diagnostic_labels"], row["eligible"]
    deep = eligible & row["response"] & np.isin(labels, ("supported_motion", "supported_hold"))
    weak_l = eligible & row["response"] & (labels == "weak_loading") & (old["phase"] == 2)
    weak_w = eligible & row["response"] & (labels == "weak_withdrawal") & (old["phase"] == 4)
    deep_states = int((deep & (old["strength"] == 6) & (new["strength"] != 6)).sum())
    deep_actions = int((deep & np.isin(old["phase"], (2, 3, 4)) & ~np.isin(new["phase"], (2, 3, 4))).sum())
    weak_frames = int((weak_l & (new["phase"] != 2)).sum()+(weak_w & (new["phase"] != 4)).sum())
    oi, ni = score_prediction(row, old), score_prediction(row, new)
    oe = {(e["group"], e["start_time_ns"]): e for e in oi["events"]}
    lost_hits = sum(oe[(e["group"], e["start_time_ns"])]["hit"] and not e["hit"] for e in ni["events"])
    keys = ("safe_lift_loading_fraction", "safe_lift_any_action_fraction",
            "protocol_no_motion_loading_fraction", "protocol_no_motion_any_action_fraction")
    safe_ok = all(ni["rates"].get(k, 0.) <= oi["rates"].get(k, 0.)+1e-12 for k in keys)
    return {"supported_states_lost": deep_states, "supported_actions_lost": deep_actions,
        "correct_weak_frames_lost": weak_frames, "weak_hits_lost": int(lost_hits), "safe_nonincrease": bool(safe_ok),
        "qualified": bool(not deep_states and not deep_actions and not weak_frames and not lost_hits and safe_ok)}


def control_support(row, pred):
    use = row["eligible"] & np.isin(row["diagnostic_labels"], SAFE)
    n = int(use.sum()); false = int((use & (pred["strength"] == 6)).sum())
    return {"eligible_control_frames": n, "supported_control_frames": false,
        "fraction": float(false/n) if n else None}


def support_summary(rows, predictions):
    items = [dict(control_support(r, p), path=str(r["path"])) for r, p in zip(rows, predictions)]
    den = sum(i["eligible_control_frames"] for i in items); num = sum(i["supported_control_frames"] for i in items)
    valid = [i["fraction"] for i in items if i["fraction"] is not None]
    return {"control_frames": den, "supported_control_frames": num, "pooled_fraction": num/den if den else None,
        "trial_macro_fraction": float(np.mean(valid)) if valid else None, "trials": items}


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"limits": (.995, .999), "confirm_s": (.03, .1), "fit_rows": 0,
        "validation_gains": VAL_GAINS, "tau_s": TAU_S,
        "qualification": "all native SOURCE train/validation and all validation-gain Trials: zero old supported deep state/action loss, zero correct shallow frame/event loss, no added lift/control action; strict reduction in validation mean-gain control-supported fraction",
        "selection": "lowest SOURCE validation mean-gain Trial-macro control-supported fraction; native safety noninferior",
        "no_XYZ_arming": True, "semantics": "supported magnetic context can be mistaken, residual is not physical release truth",
        "base_models_unchanged": True, "no_target_retained_or_old_tip_selection": True}
    save_json(OUT/"declared_search.json", declared)
    p = prior_freeze(PRIOR_OUT); args, base = load_prior_args(); model = load_model(ROOT/p["selected"]["file"])
    def prepare(row):
        _, before = prior_infer(row, *args); s = p["selected"]
        scores = predict_sequence(row, model, s["spec"])
        row["prior"] = protected_veto(row, before, scores, s["limit"], s["confirm_s"])
        row["context_scores"] = scores
        return row
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and
        r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "whole_Trial": True, "target_opened": 0, "source_test_opened": 0})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        rows.append(prepare(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R291 SOURCE support release", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: support_summary(vr, [r["prior"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v["trial_macro_fraction"] for v in baseline.values()]))
    native_before = support_summary(rows, [r["prior"] for r in rows]); candidates = []
    for limit in declared["limits"]:
        for confirm in declared["confirm_s"]:
            native = [release_supported(r, r["prior"], r["context_scores"], limit, confirm) for r in rows]
            nc = [dict(safety(r, r["prior"], n), path=str(r["path"])) for r, n in zip(rows, native)]
            reports, strengths, gc = {}, {}, []
            for gain, vr in validation.items():
                preds = [release_supported(r, r["prior"], r["context_scores"], limit, confirm) for r in vr]
                reports[gain] = aggregate([score_prediction(r, n) for r, n in zip(vr, preds)])
                strengths[gain] = support_summary(vr, preds)
                gc += [dict(safety(r, r["prior"], n), path=str(r["path"]), gain=gain) for r, n in zip(vr, preds)]
            obj = float(np.mean([v["trial_macro_fraction"] for v in strengths.values()]))
            safe = all(c["qualified"] for c in nc+gc)
            candidates.append({"limit": limit, "confirm_s": confirm, "qualified_protection": bool(safe),
                "objective": obj, "previous_objective": previous, "gain_control_support": strengths,
                "gain_validation": reports, "native_control_support": support_summary(rows, native),
                "native_failures": [c for c in nc if not c["qualified"]], "gain_failures": [c for c in gc if not c["qualified"]],
                "accepted_improvement": bool(safe and obj < previous-1e-8)})
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R291-SourceSupportedRelease-1", "selected": selected,
        "accepted_improvement": bool(accepted), "candidates": candidates, "baseline_gain_control_support": baseline,
        "baseline_native_control_support": native_before, "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0, "new_model_fitting": False})
    tracked = [Path(__file__), OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json"]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R291-FrozenSupportedRelease-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "no_XYZ_arming": True, "base_models_unchanged": True, "strength_only_contextual_release": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R291 accepted", bool(accepted), "native controls", native_before["supported_control_frames"],
        "selected", None if selected is None else {k: selected[k] for k in ("limit", "confirm_s", "objective")}, flush=True)


if __name__ == "__main__": main()
