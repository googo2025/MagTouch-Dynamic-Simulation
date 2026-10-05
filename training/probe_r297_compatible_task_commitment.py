"""SOURCE-only score compatibility, unchanged shared-network weights."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r295_matched_source_hard_context import (
    ROOT, INVENTORY, REFERENCE_OUT, RELEASE_OUT, CONTEXT_OUT,
    OUT as NETWORK_OUT, load_prior_args, prior_infer, protected_veto,
    load_source, source_partition, VAL_GAINS, TAU_S, augmented_signal_row,
    predict_sequence, predict_multitask, load_model, RAW_FILES, save_json, sha,
    replace_coverage, decoupled_release, release_supported, protection,
    reference_freeze, release_freeze, context_freeze, MultiTaskMLP, objective)
from training.verify_r295_matched_source_hard_context import verify_freeze as network_freeze

OUT = ROOT/"analysis/r297_source_compatible_task_commitment_20260929"


def compatible_scores(scores, kind):
    p = np.asarray(scores["direction"], float); c = np.asarray(scores["context"], float)
    if p.ndim != 2 or p.shape[1] != 3 or c.shape != (len(p), 2): raise ValueError("aligned independent task heads required")
    if any(not np.isfinite(v).all() or np.any(v < 0) or np.any(abs(v.sum(1)-1.) > 1e-5) for v in (p, c)):
        raise ValueError("normalized task scores required")
    if kind == "independent": return {"direction": p.copy(), "context": c.copy()}
    if kind != "compatible": raise ValueError("predeclared commitment rule required")
    result = p.copy(); result[:, :2] *= c[:, 1, None]
    result[:, 2] = np.maximum(0., 1.-result[:, :2].sum(1))
    return {"direction": result, "context": c.copy()}


def replace(row, base, scores, policy):
    result = replace_coverage(row, base, compatible_scores(scores, policy["score_kind"]), policy)
    result["compatible_active_direction_score"] = np.max(scores["direction"][:, :2], axis=1)*scores["context"][:, 1]
    return result


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"fit_rows": 0, "network_selection": "lowest SOURCE validation joint CE among the two frozen R295 heads",
        "score_kinds": ("independent", "compatible"), "direction_limits": (.75, .9, .97, .995),
        "context_limits": (.97, .995), "direction_confirm_s": .03, "context_confirm_s": .03,
        "release_limit": .995, "release_confirm_s": .03, "validation_gains": VAL_GAINS, "tau_s": TAU_S,
        "compatible_score": "direction loading/withdrawal scores times active-context score; neutral receives remaining mass",
        "interpretation": "bounded compatibility score, not a calibrated joint posterior or physical contact probability",
        "replacement": "same frozen shared network replaces R287/R288; retains R282 base and decoupled contextual release",
        "qualification": "all SOURCE native train/validation and validation gain Trials retain correct shallow frames/hits, nominal deep states/actions; control action/support nonincrease; native objective noninferior and mean gain improvement >1e-4",
        "selection": "SOURCE validation mean-gain policy objective +2*Trial-macro control-supported fraction",
        "no_target_or_SOURCE_test_selection": True, "no_XYZ_arming": True}
    save_json(OUT/"declared_search.json", declared)
    network_freeze(NETWORK_OUT)
    report = json.loads((NETWORK_OUT/"training_report.json").read_text(encoding="utf-8"))
    head = min(report["heads"], key=lambda h: h["validation_loss"])
    model = MultiTaskMLP.load(ROOT/head["file"])
    ref, release, context = reference_freeze(REFERENCE_OUT), release_freeze(RELEASE_OUT), context_freeze(CONTEXT_OUT)
    context_model = load_model(ROOT/context["selected"]["file"]); args, base = load_prior_args()
    def prepare(row):
        row["base"], row["before_context"] = prior_infer(row, *args)
        c = context["selected"]; scores = predict_sequence(row, context_model, c["spec"])
        old = protected_veto(row, row["before_context"], scores, c["limit"], c["confirm_s"])
        old = release_supported(row, old, scores, release["selected"]["limit"], release["selected"]["confirm_s"])
        row["reference"] = decoupled_release(row, old, scores, ref["selected"]["limit"], ref["selected"]["confirm_s"])
        row["task_scores"] = predict_multitask(row, model, head["spec"])
        return row
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and
        r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "whole_Trial": True, "target_opened": 0, "source_test_opened": 0})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        rows.append(prepare(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R297 SOURCE compatibility", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: objective(vr, [r["reference"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v[0] for v in baseline.values()])); candidates = []
    for kind in declared["score_kinds"]:
        for dl in declared["direction_limits"]:
            for cl in declared["context_limits"]:
                policy = {"score_kind": kind,
                    "direction": {"score_limit": dl, "confirm_s": .03, "permit_residue": True, "require_signed_change": False},
                    "context_limit": cl, "context_confirm_s": .03, "release_limit": .995, "release_confirm_s": .03}
                native = [replace(r, r["base"], r["task_scores"], policy) for r in rows]
                nc = [dict(protection(r, r["reference"], n), path=str(r["path"])) for r, n in zip(rows, native)]
                reports, supports, objectives, gc = {}, {}, {}, []
                for g, vr in validation.items():
                    preds = [replace(r, r["base"], r["task_scores"], policy) for r in vr]
                    objectives[g], reports[g], supports[g] = objective(vr, preds)
                    gc += [dict(protection(r, r["reference"], n), path=str(r["path"]), gain=g) for r, n in zip(vr, preds)]
                obj = float(np.mean(list(objectives.values()))); safe = all(c["qualified"] for c in nc+gc)
                native_ok = objectives["1.0"] <= baseline["1.0"][0]+1e-9
                candidates.append({"spec": head["spec"], "file": head["file"], "policy": policy,
                    "gain_validation": reports, "gain_control_support": supports, "gain_objectives": objectives,
                    "objective": obj, "previous_objective": previous, "qualified_protection": bool(safe),
                    "native_objective_noninferior": bool(native_ok), "native_failures": [c for c in nc if not c["qualified"]],
                    "gain_failures": [c for c in gc if not c["qualified"]],
                    "accepted_improvement": bool(safe and native_ok and obj < previous-1e-4)})
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R297-SourceCompatibleCommitment-1", "head": head, "candidates": candidates,
        "baseline_gain_validation": {g: v[1] for g, v in baseline.items()}, "baseline_gain_control_support": {g: v[2] for g, v in baseline.items()},
        "baseline_gain_objectives": {g: v[0] for g, v in baseline.items()}, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/probe_r294_matched_weak_coverage.py", ROOT/"training/multitask_response_mlp.py",
        OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json", ROOT/head["file"]]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R297-FrozenCompatibleCommitment-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (REFERENCE_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(REFERENCE_OUT/"freeze.json"),
        "network_protocol": (NETWORK_OUT/"freeze.json").relative_to(ROOT).as_posix(), "network_protocol_sha256": sha(NETWORK_OUT/"freeze.json"),
        "no_XYZ_arming": True, "base_models_unchanged": True, "strength_only_contextual_release": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R297 accepted", bool(accepted), "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__": main()
