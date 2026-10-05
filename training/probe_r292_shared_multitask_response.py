"""SOURCE-only shared-backbone, independent direction/context task losses.

One network replaces R287 + R288, compared against complete R291. Failure
to retain every SOURCE correct weak/deep response rejects the candidate.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r290_shared_action_context import (
    ROOT, INVENTORY, load_prior_args, prior_infer, protected_veto, load_source,
    source_partition, TRAIN_GAINS, VAL_GAINS, TAU_S, augmented_signal_row,
    predict_sequence, load_model, score_prediction, aggregate, policy_objective,
    RAW_FILES, save_json, sha, shared_dataset, shared_features, current_features, causal_taps,
    supplement)
from training.probe_r288_ordered_action_context import OUT as CONTEXT_OUT
from training.probe_r291_supported_context_release import (
    OUT as REFERENCE_OUT, release_supported, safety, control_support, support_summary)
from training.verify_r291_supported_context_release import verify_freeze as reference_freeze
from training.verify_r288_ordered_action_context import verify_freeze as context_freeze
from training.multitask_response_mlp import MultiTaskMLP

OUT = ROOT/"analysis/r292_source_shared_multitask_response_20260929"
SPECS = ({"name": "shared_lambda1", "variant": "global", "width": 32, "context_weight": 1.},
         {"name": "shared_lambda2", "variant": "global", "width": 32, "context_weight": 2.})


def predict_multitask(row, model, spec):
    value = current_features(row, spec["variant"]); taps = causal_taps(row["time"], row["eligible"], 16, .1)
    result = {"direction": [], "context": []}
    for start in range(0, len(taps), 512):
        ids = taps[start:start+512]; elapsed = np.zeros(ids.shape)
        elapsed[:, 1:] = np.diff(row["time"][ids], axis=1)/100_000_000.
        x = np.concatenate((value[ids], np.clip(elapsed, 0, 5)[..., None]), 2).reshape(len(ids), -1)
        p = model.predict(x)
        for key in result: result[key].append(p[key])
    return {key: np.vstack(values) for key, values in result.items()}


def replace_tasks(row, base, scores, policy):
    for key, dim in (("direction", 3), ("context", 2)):
        p = np.asarray(scores[key], float)
        if p.shape != (len(row["time"]), dim) or not np.isfinite(p).all() or np.any(p < 0) or np.any(abs(p.sum(1)-1) > 1e-5):
            raise ValueError("aligned normalized independent task scores required")
    proposed = supplement(row, base, scores["direction"], policy["direction"])
    proposed = protected_veto(row, proposed, scores["context"], policy["context_limit"], policy["context_confirm_s"])
    result = release_supported(row, proposed, scores["context"], policy["release_limit"], policy["release_confirm_s"])
    result["shared_direction_added"] = proposed["direction_supplemented"]
    result["shared_context_veto"] = proposed["ordered_context_veto"]
    return result


def preservation(row, old, new):
    check = safety(row, old, new); a, b = control_support(row, old), control_support(row, new)
    support_ok = b["supported_control_frames"] <= a["supported_control_frames"]
    check["supported_control_nonincrease"] = bool(support_ok)
    check["qualified"] = bool(check["qualified"] and support_ok)
    return check


def objective(rows, predictions):
    reports = aggregate([score_prediction(r, p) for r, p in zip(rows, predictions)])
    support = support_summary(rows, predictions)
    return policy_objective(reports)+2.*(support["trial_macro_fraction"] or 0.), reports, support


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"specs": SPECS, "seed": 292, "epochs": 80, "steps": 16, "step_s": .1,
        "effective_max_history_s": 6.5, "train_gains": TRAIN_GAINS, "validation_gains": VAL_GAINS,
        "tau_s": TAU_S, "cap_per_Trial_class": 150, "SOURCE_train_hard_negative_cap": 150,
        "direction_limits": (.9, .97, .995), "direction_confirm_s": .03,
        "context_limits": (.97, .995), "context_confirm_s": .03,
        "release_limit": .999, "release_confirm_s": .03,
        "architecture": "512-32-32 shared tanh trunk; separate 3-class direction and 2-class context softmax heads",
        "loss": "class/Trial-balanced direction CE + lambda*independently class/Trial-balanced context CE + .0005*sum(weight^2); no bias penalty",
        "optimization": "Adam lr .001, batch approximately 256, 80 epochs; SOURCE combined validation CE every 5 epochs selects checkpoint",
        "comparison": "identical SOURCE sample indices, task class weights, model parameter count, seed and updates; only context loss coefficient differs",
        "objective": "SOURCE validation policy_objective plus 2*Trial-macro control-supported fraction; average native/gain .5/1/2",
        "qualification": "every SOURCE native train/validation and validation gain Trial: exact correct weak frames/episode hits and supported deep states/actions retained, lift/control actions and supported-control frames nonincrease; native objective noninferior and mean-gain strict improvement >1e-4",
        "replacement": "R282 base plus ONE shared multi-task model replaces R287 and R288; R291 release rule remains source-screened; not another stacked expert",
        "semantics": "magnetic protocol-relative actions, not actual shallow contact or reliable XYZ",
        "target_and_retained_SOURCE_selection": 0, "no_XYZ_arming": True}
    save_json(OUT/"declared_search.json", declared)
    ref = reference_freeze(REFERENCE_OUT); cp = context_freeze(CONTEXT_OUT)
    context = load_model(ROOT/cp["selected"]["file"]); args, base = load_prior_args()
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and
        r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "target_opened": 0, "source_test_opened": 0, "whole_Trial": True})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    def prepare(row):
        row["base"], row["before_context"] = prior_infer(row, *args)
        s = cp["selected"]; scores = predict_sequence(row, context, s["spec"])
        old = protected_veto(row, row["before_context"], scores, s["limit"], s["confirm_s"])
        row["reference"] = release_supported(row, old, scores, ref["selected"]["limit"], ref["selected"]["confirm_s"])
        return row
    rows = []
    for i, rec in enumerate(recs):
        rows.append(prepare(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R292 SOURCE multitask load", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: objective(vr, [r["reference"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v[0] for v in baseline.values()]))
    heads, candidates, previous_manifest = [], [], None
    tx, ty, tw, train_ids = shared_dataset(rows, "global", base["null"]["source"])
    vx, vy, vw, val_ids = shared_dataset(rows, "global", base["null"]["source"], True)
    for spec in SPECS:
        manifest = {"train": train_ids, "validation": val_ids, "train_samples": len(ty), "validation_samples": len(vy)}
        if previous_manifest is not None: assert previous_manifest == manifest
        previous_manifest = manifest; save_json(OUT/f"samples_{spec['name']}.json", manifest)
        print("R292 SOURCE train", spec["name"], tx.shape, vx.shape, flush=True)
        model = MultiTaskMLP(spec["width"], 292, spec["context_weight"]).fit(tx, ty, tw, (vx, vy, vw), 80)
        file = OUT/f"multitask_{spec['name']}.npz"; model.save(file)
        heads.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "epoch": model.best_epoch,
            "validation_loss": model.best_loss, "history": model.history, "parameters": sum(int(v.size) for v in model.params)})
        native_scores = [predict_multitask(r, model, spec) for r in rows]
        gain_scores = {g: [predict_multitask(r, model, spec) for r in vr] for g, vr in validation.items()}
        for dl in declared["direction_limits"]:
            for cl in declared["context_limits"]:
                policy = {"direction": {"score_limit": dl, "confirm_s": .03, "permit_residue": True, "require_signed_change": False},
                    "context_limit": cl, "context_confirm_s": .03, "release_limit": .999, "release_confirm_s": .03}
                native = [replace_tasks(r, r["base"], s, policy) for r, s in zip(rows, native_scores)]
                nc = [dict(preservation(r, r["reference"], n), path=str(r["path"])) for r, n in zip(rows, native)]
                reports, controls, objectives, gc = {}, {}, {}, []
                for g, vr in validation.items():
                    preds = [replace_tasks(r, r["base"], s, policy) for r, s in zip(vr, gain_scores[g])]
                    objectives[g], reports[g], controls[g] = objective(vr, preds)
                    gc += [dict(preservation(r, r["reference"], n), path=str(r["path"]), gain=g) for r, n in zip(vr, preds)]
                obj = float(np.mean(list(objectives.values()))); safe = all(c["qualified"] for c in nc+gc)
                native_ok = objectives["1.0"] <= baseline["1.0"][0]+1e-9
                candidates.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "policy": policy,
                    "gain_validation": reports, "gain_control_support": controls, "gain_objectives": objectives,
                    "objective": obj, "previous_objective": previous, "qualified_protection": bool(safe),
                    "native_objective_noninferior": bool(native_ok),
                    "native_failures": [c for c in nc if not c["qualified"]], "gain_failures": [c for c in gc if not c["qualified"]],
                    "accepted_improvement": bool(safe and native_ok and obj < previous-1e-4)})
        print("R292 SOURCE model", spec["name"], "CE", model.best_loss, "accepted", sum(c["accepted_improvement"] for c in candidates), flush=True)
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R292-SharedMultitaskResponse-1", "heads": heads,
        "candidates": candidates, "baseline_gain_validation": {g: v[1] for g, v in baseline.items()},
        "baseline_gain_control_support": {g: v[2] for g, v in baseline.items()},
        "baseline_gain_objectives": {g: v[0] for g, v in baseline.items()}, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/multitask_response_mlp.py", OUT/"declared_search.json",
        OUT/"source_partition.json", OUT/"training_report.json"]+[ROOT/h["file"] for h in heads]+list(OUT.glob("samples_*.json"))
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R292-FrozenSharedMultitaskResponse-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (REFERENCE_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(REFERENCE_OUT/"freeze.json"),
        "no_XYZ_arming": True, "base_models_unchanged": True, "strength_only_contextual_release": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R292 accepted", bool(accepted), "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__": main()
