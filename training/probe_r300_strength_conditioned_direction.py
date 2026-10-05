"""Direction commitment conditioned on magnetic support, never command Z."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r299_direction_only_replacement import (
    ROOT, INVENTORY, REFERENCE_OUT, RELEASE_OUT, CONTEXT_OUT, NETWORK_OUT,
    load_prior_args, prior_infer, protected_veto, load_source, source_partition,
    VAL_GAINS, TAU_S, augmented_signal_row, predict_sequence, load_model,
    RAW_FILES, save_json, sha, decoupled_release, release_supported, protection,
    reference_freeze, release_freeze, context_freeze, network_freeze, MultiTaskMLP,
    objective, extract_direction, predict_direction, replace_direction)
from training.verify_r299_direction_only_replacement import verify_freeze as donor_rule_verify
from training.probe_r299_direction_only_replacement import OUT as DONOR_RULE_OUT

OUT = ROOT/"analysis/r300_source_strength_conditioned_direction_20260929"


def strength_conditioned_scores(scores, strength, supported_limit, other_limit):
    p = np.asarray(scores, float); strength = np.asarray(strength)
    if p.ndim != 2 or p.shape[1] != 3 or strength.shape != (len(p),): raise ValueError("aligned direction/support required")
    if not .5 < supported_limit <= other_limit < 1: raise ValueError("ordered magnetic commitment thresholds required")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(abs(p.sum(1)-1) > 1e-5): raise ValueError("normalized direction scores required")
    if not np.isin(strength, np.arange(7)).all(): raise ValueError("known magnetic strength states required")
    limit = np.where(strength == 6, supported_limit, other_limit)
    # Below the appropriate commitment floor the action is explicitly unknown.
    # These are decision scores, not recalibrated contact/direction posteriors.
    denied = np.max(p[:, :2], axis=1) < limit
    result = p.copy(); result[denied] = (0, 0, 1)
    return result


def replace(row, base, direction, context_scores, policy, context, release, reference):
    scores = strength_conditioned_scores(direction, base["strength"], policy["supported_limit"], policy["other_limit"])
    cp = {"direction": {"score_limit": policy["supported_limit"], "confirm_s": .03,
        "permit_residue": True, "require_signed_change": False}, "new_direction_residual_limit": policy["new_direction_residual_limit"]}
    result = replace_direction(row, base, scores, context_scores, cp, context, release, reference)
    result["direction_commitment_limit"] = np.where(base["strength"] == 6, policy["supported_limit"], policy["other_limit"])
    return result


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"fit_rows": 0, "checkpoint": "frozen R295 lowest SOURCE validation CE direction extraction",
        "threshold_pairs": ((.9, .97), (.9, .995), (.97, .995)), "new_direction_residual_limits": (.995, .999),
        "confirm_s": .03, "condition": "R282 magnetic strength state6; no command depth, device, trajectory template or future input",
        "other_states": "higher commitment floor, not forced physical noncontact",
        "fixed_context": "R288; R291 high-confidence release then R293 decoupled release",
        "validation_gains": VAL_GAINS, "tau_s": TAU_S, "no_XYZ_arming": True,
        "qualification": "all native SOURCE train/validation and gain validation Trials retain correct shallow frames/hits and deep states/actions, no added control actions/support; native objective noninferior; mean-gain objective improves >1e-4",
        "capacity": "no new parameters; not capacity-matched with R287", "strength_exact_reference": True,
        "target_fit_selection_rows": 0, "SOURCE_test_selection_rows": 0}
    save_json(OUT/"declared_search.json", declared)
    donor_rule_verify(DONOR_RULE_OUT); network_freeze(NETWORK_OUT)
    nr = json.loads((NETWORK_OUT/"training_report.json").read_text(encoding="utf-8")); head = min(nr["heads"], key=lambda h: h["validation_loss"])
    model = extract_direction(MultiTaskMLP.load(ROOT/head["file"])); file = OUT/"direction_extracted.npz"; model.save(file)
    ref, rel, ctx = reference_freeze(REFERENCE_OUT), release_freeze(RELEASE_OUT), context_freeze(CONTEXT_OUT)
    cm = load_model(ROOT/ctx["selected"]["file"]); args, base = load_prior_args()
    def prepare(row):
        row["base"], before = prior_infer(row, *args); c = ctx["selected"]
        scores = predict_sequence(row, cm, c["spec"])
        old = protected_veto(row, before, scores, c["limit"], c["confirm_s"])
        old = release_supported(row, old, scores, rel["selected"]["limit"], rel["selected"]["confirm_s"])
        row["reference"] = decoupled_release(row, old, scores, ref["selected"]["limit"], ref["selected"]["confirm_s"])
        row["context_scores"] = scores; row["direction_scores"] = predict_direction(row, model, head["spec"])
        return row
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and r["date"] >= "S20260926_A"
        and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs], "whole_Trial": True,
        "target_opened": 0, "source_test_opened": 0})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        rows.append(prepare(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R300 SOURCE support/direction", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: objective(vr, [r["reference"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v[0] for v in baseline.values()])); candidates = []
    for supported, other in declared["threshold_pairs"]:
        for rl in declared["new_direction_residual_limits"]:
            policy = {"supported_limit": supported, "other_limit": other, "new_direction_residual_limit": rl}
            def run(row):
                result = replace(row, row["base"], row["direction_scores"], row["context_scores"], policy,
                    ctx["selected"], rel["selected"], ref["selected"])
                np.testing.assert_array_equal(result["strength"], row["reference"]["strength"]); return result
            native = [run(r) for r in rows]
            nc = [dict(protection(r, r["reference"], n), path=str(r["path"])) for r, n in zip(rows, native)]
            reports, supports, objectives, gc = {}, {}, {}, []
            for g, vr in validation.items():
                preds = [run(r) for r in vr]; objectives[g], reports[g], supports[g] = objective(vr, preds)
                gc += [dict(protection(r, r["reference"], n), path=str(r["path"]), gain=g) for r, n in zip(vr, preds)]
            obj = float(np.mean(list(objectives.values()))); safe = all(c["qualified"] for c in nc+gc)
            native_ok = objectives["1.0"] <= baseline["1.0"][0]+1e-9
            candidates.append({"file": file.relative_to(ROOT).as_posix(), "spec": head["spec"], "policy": policy,
                "gain_validation": reports, "gain_control_support": supports, "gain_objectives": objectives, "objective": obj,
                "previous_objective": previous, "qualified_protection": bool(safe), "native_objective_noninferior": bool(native_ok),
                "native_failures": [c for c in nc if not c["qualified"]], "gain_failures": [c for c in gc if not c["qualified"]],
                "accepted_improvement": bool(safe and native_ok and obj < previous-1e-4)})
    accepted = [c for c in candidates if c["accepted_improvement"]]; selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R300-SourceStrengthConditionedDirection-1", "head": head, "candidates": candidates,
        "baseline_gain_validation": {g: v[1] for g, v in baseline.items()}, "baseline_gain_control_support": {g: v[2] for g, v in baseline.items()},
        "baseline_gain_objectives": {g: v[0] for g, v in baseline.items()}, "selected": selected, "accepted_improvement": bool(accepted),
        "elapsed_s": time.monotonic()-started, "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json", file]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R300-FrozenStrengthConditionedDirection-1", "selected": selected, "accepted_improvement": bool(accepted),
        "frozen_before_target_replay": True, "base_protocol": (REFERENCE_OUT/"freeze.json").relative_to(ROOT).as_posix(),
        "base_protocol_sha256": sha(REFERENCE_OUT/"freeze.json"), "network_protocol": (NETWORK_OUT/"freeze.json").relative_to(ROOT).as_posix(),
        "network_protocol_sha256": sha(NETWORK_OUT/"freeze.json"), "direction_protocol": (DONOR_RULE_OUT/"freeze.json").relative_to(ROOT).as_posix(),
        "direction_protocol_sha256": sha(DONOR_RULE_OUT/"freeze.json"), "no_XYZ_arming": True, "base_models_unchanged": True,
        "strength_exact_reference": True, "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R300 accepted", bool(accepted), "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__": main()
