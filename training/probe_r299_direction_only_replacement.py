"""SOURCE-only direction replacement, retaining the frozen residual context.

No new fitting: extract the direction head and normalization of the R295
checkpoint selected solely by SOURCE validation CE. Replace R287, not R288.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r297_compatible_task_commitment import (
    ROOT, INVENTORY, REFERENCE_OUT, RELEASE_OUT, CONTEXT_OUT, NETWORK_OUT,
    load_prior_args, prior_infer, protected_veto, load_source, source_partition,
    VAL_GAINS, TAU_S, augmented_signal_row, predict_sequence, load_model,
    RAW_FILES, save_json, sha, decoupled_release, release_supported, protection,
    reference_freeze, release_freeze, context_freeze, network_freeze, MultiTaskMLP,
    objective)
from training.probe_r292_shared_multitask_response import current_features, causal_taps
from training.probe_r278_independent_direction import supplement
from training.probe_r245_joint_contact_state import StateMLP

OUT = ROOT/"analysis/r299_source_direction_only_replacement_20260929"


def extract_direction(model):
    result = StateMLP(model.width, n_classes=3)
    result.mean = model.mean.copy(); result.scale = model.scale.copy()
    result.params = [v.copy() for v in model.params[:6]]
    if len(result.params) != 6 or result.params[-1].shape != (3,):
        raise ValueError("two-hidden-layer three-class frozen direction required")
    return result


def predict_direction(row, model, spec):
    values = current_features(row, spec["variant"])
    taps = causal_taps(row["time"], row["eligible"], 16, .1)
    result = []
    for start in range(0, len(taps), 512):
        ids = taps[start:start+512]; elapsed = np.zeros(ids.shape)
        elapsed[:, 1:] = np.diff(row["time"][ids], axis=1)/100_000_000.
        x = np.concatenate((values[ids], np.clip(elapsed, 0, 5)[..., None]), 2).reshape(len(ids), -1)
        result.append(model.predict(x))
    return np.vstack(result)


def contextual_confirmation(row, context_scores, limit, confirm_s):
    # Reuse the frozen causal confirmation, without inferring physical support.
    dummy = {"phase": np.zeros(len(row["time"]), int), "strength": np.full(len(row["time"]), 6)}
    return release_supported(row, dummy, context_scores, limit, confirm_s)["supported_context_released"]


def replace_direction(row, base, direction_scores, context_scores, policy, context, release, reference):
    proposed = supplement(row, base, direction_scores, policy["direction"])
    added = proposed["direction_supplemented"].copy()
    denied = added & contextual_confirmation(row, context_scores, policy["new_direction_residual_limit"], .03)
    phase = proposed["phase"].copy(); phase[denied] = base["phase"][denied]
    proposed = dict(proposed, phase=phase, direction_supplemented=added & ~denied)
    proposed = protected_veto(row, proposed, context_scores, context["limit"], context["confirm_s"])
    first = release_supported(row, proposed, context_scores, release["limit"], release["confirm_s"])
    result = decoupled_release(row, first, context_scores, reference["limit"], reference["confirm_s"])
    result["new_direction_added"] = added & ~denied
    result["new_direction_residual_denied"] = denied
    result["high_confidence_support_released"] = first["supported_context_released"]
    result["total_support_released"] = first["supported_context_released"] | result["contextual_support_released"]
    return result


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"fit_rows": 0, "network_selection": "lowest frozen R295 SOURCE validation joint CE",
        "extraction": "copy SOURCE mean/scale and first six parameter arrays; remove context output only",
        "direction_parameters": 27123, "context_retained": "frozen R288",
        "replacement": "R295 direction replaces R287; R282 base retained; no stacked new direction expert",
        "capacity": "not a capacity-matched ablation; extracted direction is larger than R287",
        "direction_limits": (.9, .97, .995), "new_direction_residual_limits": (.995, .999),
        "confirm_s": .03, "validation_gains": VAL_GAINS, "tau_s": TAU_S,
        "release_order": "R288 veto, R291 high-confidence phase/support release, R293 decoupled support release",
        "qualification": "all SOURCE native train/validation and validation-gain Trials retain exact correct weak frames/hits, deep states/actions; no increased lift/control actions or support; native objective noninferior; mean-gain objective improvement >1e-4",
        "no_target_or_SOURCE_test_selection": True, "no_XYZ_arming": True,
        "strength": "must exactly equal complete R293 on each frame"}
    save_json(OUT/"declared_search.json", declared)
    network_freeze(NETWORK_OUT)
    nr = json.loads((NETWORK_OUT/"training_report.json").read_text(encoding="utf-8"))
    head = min(nr["heads"], key=lambda h: h["validation_loss"])
    donor = MultiTaskMLP.load(ROOT/head["file"]); model = extract_direction(donor)
    assert sum(v.size for v in model.params) == declared["direction_parameters"]
    x = np.random.default_rng(299).normal(size=(1024, len(model.mean)))
    np.testing.assert_allclose(model.predict(x), donor.predict(x)["direction"], atol=1e-14, rtol=1e-14)
    file = OUT/"direction_extracted.npz"; model.save(file)
    ref = reference_freeze(REFERENCE_OUT); rel = release_freeze(RELEASE_OUT); ctx = context_freeze(CONTEXT_OUT)
    cm = load_model(ROOT/ctx["selected"]["file"]); args, base = load_prior_args()
    def prepare(row):
        row["base"], before = prior_infer(row, *args)
        c = ctx["selected"]; scores = predict_sequence(row, cm, c["spec"])
        old = protected_veto(row, before, scores, c["limit"], c["confirm_s"])
        old = release_supported(row, old, scores, rel["selected"]["limit"], rel["selected"]["confirm_s"])
        row["reference"] = decoupled_release(row, old, scores, ref["selected"]["limit"], ref["selected"]["confirm_s"])
        row["context_scores"] = scores; row["direction_scores"] = predict_direction(row, model, head["spec"])
        return row
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and
        r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "whole_Trial": True, "target_opened": 0, "source_test_opened": 0})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        rows.append(prepare(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R299 SOURCE direction", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: objective(vr, [r["reference"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v[0] for v in baseline.values()])); candidates = []
    for dl in declared["direction_limits"]:
        for rl in declared["new_direction_residual_limits"]:
            policy = {"direction": {"score_limit": dl, "confirm_s": .03, "permit_residue": True, "require_signed_change": False},
                "new_direction_residual_limit": rl}
            def run(row):
                result = replace_direction(row, row["base"], row["direction_scores"], row["context_scores"], policy,
                    ctx["selected"], rel["selected"], ref["selected"])
                np.testing.assert_array_equal(result["strength"], row["reference"]["strength"])
                return result
            native = [run(r) for r in rows]
            nc = [dict(protection(r, r["reference"], n), path=str(r["path"])) for r, n in zip(rows, native)]
            reports, supports, objectives, gc = {}, {}, {}, []
            for g, vr in validation.items():
                preds = [run(r) for r in vr]; objectives[g], reports[g], supports[g] = objective(vr, preds)
                gc += [dict(protection(r, r["reference"], n), path=str(r["path"]), gain=g) for r, n in zip(vr, preds)]
            obj = float(np.mean(list(objectives.values()))); safe = all(c["qualified"] for c in nc+gc)
            native_ok = objectives["1.0"] <= baseline["1.0"][0]+1e-9
            candidates.append({"file": file.relative_to(ROOT).as_posix(), "spec": head["spec"], "policy": policy,
                "gain_validation": reports, "gain_control_support": supports, "gain_objectives": objectives,
                "objective": obj, "previous_objective": previous, "qualified_protection": bool(safe),
                "native_objective_noninferior": bool(native_ok), "native_failures": [c for c in nc if not c["qualified"]],
                "gain_failures": [c for c in gc if not c["qualified"]],
                "accepted_improvement": bool(safe and native_ok and obj < previous-1e-4)})
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R299-SourceDirectionOnlyReplacement-1", "head": head, "candidates": candidates,
        "baseline_gain_validation": {g: v[1] for g, v in baseline.items()}, "baseline_gain_control_support": {g: v[2] for g, v in baseline.items()},
        "baseline_gain_objectives": {g: v[0] for g, v in baseline.items()}, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/multitask_response_mlp.py", OUT/"declared_search.json", OUT/"source_partition.json",
        OUT/"training_report.json", file, ROOT/head["file"]]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R299-FrozenDirectionOnlyReplacement-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (REFERENCE_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(REFERENCE_OUT/"freeze.json"),
        "network_protocol": (NETWORK_OUT/"freeze.json").relative_to(ROOT).as_posix(), "network_protocol_sha256": sha(NETWORK_OUT/"freeze.json"),
        "no_XYZ_arming": True, "base_models_unchanged": True, "strength_exact_reference": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R299 accepted", bool(accepted), "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__": main()
