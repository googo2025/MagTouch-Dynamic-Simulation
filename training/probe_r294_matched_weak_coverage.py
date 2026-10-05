"""Matched SOURCE budget: per-view evidence and rare-action sampling.

Two equal-capacity multi-task models differ only in SOURCE training points.
Both use the same SOURCE validation points/checkpoint rule. No retained or
target records enter normalization, fitting, thresholds or model selection.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r292_shared_multitask_response import (
    ROOT, INVENTORY, load_prior_args, prior_infer, protected_veto, load_source,
    source_partition, TRAIN_GAINS, VAL_GAINS, TAU_S, augmented_signal_row,
    predict_sequence, load_model, score_prediction, aggregate, RAW_FILES, save_json,
    sha, shared_features, supplement, predict_multitask, objective)
from training.probe_r290_shared_action_context import common_ids, joint_labels
from training.probe_r291_supported_context_release import release_supported
from training.probe_r293_decoupled_support_context import (
    OUT as REFERENCE_OUT, decoupled_release, protection)
from training.probe_r291_supported_context_release import OUT as RELEASE_OUT
from training.probe_r288_ordered_action_context import OUT as CONTEXT_OUT
from training.verify_r293_decoupled_support_context import verify_freeze as reference_freeze
from training.verify_r291_supported_context_release import verify_freeze as release_freeze
from training.verify_r288_ordered_action_context import verify_freeze as context_freeze
from training.train_r275_source_only_response import assert_source
from training.audit_r274_frozen_phase_expansion import blocks
from training.multitask_response_mlp import MultiTaskMLP

OUT = ROOT/"analysis/r294_source_matched_weak_coverage_20260929"
SPECS = ({"name": "intersection_uniform", "sampling": "intersection_uniform", "variant": "global", "width": 48},
         {"name": "per_view_weak", "sampling": "per_view_weak", "variant": "global", "width": 48})


def uniform_ids(ids, count):
    ids = np.asarray(ids, int)
    if count < 0 or count > len(ids): raise ValueError("exact available budget required")
    if not count: return np.zeros(0, int)
    return ids[np.linspace(0, len(ids)-1, count, dtype=int)]


def stratified_ids(view, cls, count, weak_quota=70):
    """Same sample count; only this view's evidence qualifies shallow frames."""
    if view["device"] != "source" or view["split"] not in ("train", "validation"):
        raise ValueError("SOURCE development sampling only")
    if cls not in (0, 1): raise ValueError("rare action sampling is direction-specific")
    y = joint_labels(view); available = np.flatnonzero(y == cls)
    if count > len(available): raise ValueError("old intersection budget cannot exceed per-view evidence")
    group = "weak_loading" if cls == 0 else "weak_withdrawal"
    mask = (y == cls) & (view["diagnostic_labels"] == group)
    weak = np.flatnonzero(mask); quota = min(int(weak_quota), count, len(weak))
    episodes = blocks(mask, view["time"])
    if quota:
        group_ids = uniform_ids(np.arange(len(episodes)), min(quota, len(episodes)))
        chosen = np.array([episodes[g][(len(episodes[g])-1)//2] for g in group_ids], int)
        if len(chosen) < quota:
            chosen = np.r_[chosen, uniform_ids(weak[~np.isin(weak, chosen)], quota-len(chosen))]
    else: chosen = np.zeros(0, int)
    chosen = np.unique(np.r_[chosen, uniform_ids(available[~np.isin(available, chosen)], count-len(chosen))])
    assert len(chosen) == count and np.all(y[chosen] == cls)
    assert np.all(view["eligible"][chosen] & view["response"][chosen])
    return chosen


def coverage_dataset(rows, null, arm, validate=False):
    assert_source(rows)
    if any(r["split"] not in ("train", "validation") for r in rows): raise ValueError("reserved SOURCE excluded")
    if arm not in ("intersection_uniform", "per_view_weak"): raise ValueError("declared coverage arm required")
    split = "validation" if validate else "train"; gains = VAL_GAINS if validate else TRAIN_GAINS
    xs, ys, ws, manifest = [], [], [], []
    for row in rows:
        if row["split"] != split: continue
        views = [row if g == 1. else augmented_signal_row(row, g, null, TAU_S) for g in gains]
        common = common_ids(row, views)
        for g, view in zip(gains, views):
            for cls, fixed_ids in common:
                ids = (stratified_ids(view, cls, len(fixed_ids))
                       if cls in (0, 1) and (validate or arm == "per_view_weak") else fixed_ids)
                assert np.all(joint_labels(view)[ids] == cls)
                xs.append(shared_features(view, "global", ids).astype(np.float32))
                ys.append(np.full(len(ids), cls)); ws.append(np.full(len(ids), 1./len(ids)/len(gains)))
                group = "weak_loading" if cls == 0 else "weak_withdrawal" if cls == 1 else "none"
                weak = view["diagnostic_labels"] == group
                segments = blocks(weak & (joint_labels(view) == cls), view["time"]) if cls in (0, 1) else []
                manifest.append({"path": str(row["path"]), "gain": g, "class": cls, "n": len(ids),
                    "weak_frames": int(weak[ids].sum()), "responsive_weak_segments": len(segments),
                    "selected_weak_segments": int(sum(np.isin(s, ids).any() for s in segments)),
                    "indices_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()})
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(4):
        if not np.any(y == cls): raise ValueError("missing SOURCE class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    return x, y, w, manifest


def replace_coverage(row, base, scores, policy):
    for key, dim in (("direction", 3), ("context", 2)):
        p = np.asarray(scores[key], float)
        if p.shape != (len(row["time"]), dim) or not np.isfinite(p).all() or np.any(p < 0) or np.any(abs(p.sum(1)-1) > 1e-5):
            raise ValueError("aligned normalized frozen scores required")
    proposed = supplement(row, base, scores["direction"], policy["direction"])
    proposed = protected_veto(row, proposed, scores["context"], policy["context_limit"], policy["context_confirm_s"])
    result = decoupled_release(row, proposed, scores["context"], policy["release_limit"], policy["release_confirm_s"])
    result["shared_direction_added"] = proposed["direction_supplemented"]
    result["shared_context_veto"] = proposed["ordered_context_veto"]
    return result


def budget(manifest):
    return {(r["path"], r["gain"], r["class"]): r["n"] for r in manifest}


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"specs": SPECS, "seed": 294, "epochs": 80, "context_weight": 1.,
        "steps": 16, "step_s": .1, "effective_max_history_s": 6.5,
        "train_gains": TRAIN_GAINS, "validation_gains": VAL_GAINS, "tau_s": TAU_S,
        "weak_quota_per_Trial_class_view": 70, "cap_per_Trial_class": 150, "SOURCE_train_hard_negative_cap": 150,
        "validation_sampling": "shared per-view magnetic-evidence weak stratification; identical points/weights for both arms",
        "sample_budget": "each Trial/class/gain view exactly equals original intersection-uniform count; quiet and stationary indices unchanged",
        "architecture": "512-48-48 shared tanh, independent direction3/context2 heads; 27221 parameters each",
        "capacity_comparison": "equal between two R294 arms; approximately R287+R288 26917 parameters, not exact equality; cannot attribute differences from R292 only to coverage",
        "loss_optimization": "independently balanced task CE sum + .0005*weight squared; Adam .001, ~256 batch, 80 epochs; identical common SOURCE validation CE checkpoint every5",
        "direction_limits": (.9, .97, .995), "direction_confirm_s": .03,
        "context_limits": (.97, .995), "context_confirm_s": .03,
        "release_limit": .995, "release_confirm_s": .03,
        "objective": "SOURCE validation policy_objective + 2*Trial-macro control-supported fraction, mean gains .5/1/2",
        "qualification": "every SOURCE native train/validation and validation gain Trial retains exact correct weak frames/hits, nominal deep supported state/action; lift/control action and support nonincrease; native objective noninferior and mean-gain improvement >1e-4",
        "replacement": "one network replaces R287+R288, retains R282 base and decoupled support release; no appended expert",
        "semantics": "weak means action-reference range, not verified contact/depth; no XYZ arming",
        "no_target_or_retained_SOURCE_selection": True}
    save_json(OUT/"declared_search.json", declared)
    ref = reference_freeze(REFERENCE_OUT); release = release_freeze(RELEASE_OUT); context = context_freeze(CONTEXT_OUT)
    context_model = load_model(ROOT/context["selected"]["file"]); args, base = load_prior_args()
    def prepare(row):
        row["base"], row["before_context"] = prior_infer(row, *args)
        c = context["selected"]; scores = predict_sequence(row, context_model, c["spec"])
        old = protected_veto(row, row["before_context"], scores, c["limit"], c["confirm_s"])
        old = release_supported(row, old, scores, release["selected"]["limit"], release["selected"]["confirm_s"])
        row["reference"] = decoupled_release(row, old, scores, ref["selected"]["limit"], ref["selected"]["confirm_s"])
        return row
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and
        r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "whole_Trial": True, "target_opened": 0, "source_test_opened": 0})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        rows.append(prepare(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R294 SOURCE coverage load", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: objective(vr, [r["reference"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v[0] for v in baseline.values()]))
    vx, vy, vw, val_ids = coverage_dataset(rows, base["null"]["source"], "per_view_weak", True)
    save_json(OUT/"common_validation_samples.json", {"samples": val_ids, "n": len(vy)})
    heads, candidates, common_budget, common_y, common_w = [], [], None, None, None
    for spec in SPECS:
        tx, ty, tw, train_ids = coverage_dataset(rows, base["null"]["source"], spec["sampling"])
        if common_budget is None: common_budget, common_y, common_w = budget(train_ids), ty.copy(), tw.copy()
        else:
            assert common_budget == budget(train_ids)
            np.testing.assert_array_equal(ty, common_y); np.testing.assert_array_equal(tw, common_w)
        save_json(OUT/f"samples_{spec['name']}.json", {"train": train_ids, "validation": val_ids,
            "train_samples": len(ty), "validation_samples": len(vy)})
        print("R294 SOURCE train", spec["name"], tx.shape, vx.shape, flush=True)
        model = MultiTaskMLP(48, 294, 1.).fit(tx, ty, tw, (vx, vy, vw), 80)
        file = OUT/f"multitask_{spec['name']}.npz"; model.save(file)
        heads.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "epoch": model.best_epoch,
            "validation_loss": model.best_loss, "history": model.history, "parameters": sum(int(v.size) for v in model.params)})
        del tx, ty, tw
        native_scores = [predict_multitask(r, model, spec) for r in rows]
        gain_scores = {g: [predict_multitask(r, model, spec) for r in vr] for g, vr in validation.items()}
        for dl in declared["direction_limits"]:
            for cl in declared["context_limits"]:
                policy = {"direction": {"score_limit": dl, "confirm_s": .03, "permit_residue": True, "require_signed_change": False},
                    "context_limit": cl, "context_confirm_s": .03, "release_limit": .995, "release_confirm_s": .03}
                native = [replace_coverage(r, r["base"], s, policy) for r, s in zip(rows, native_scores)]
                nc = [dict(protection(r, r["reference"], n), path=str(r["path"])) for r, n in zip(rows, native)]
                reports, supports, objectives, gc = {}, {}, {}, []
                for g, vr in validation.items():
                    preds = [replace_coverage(r, r["base"], s, policy) for r, s in zip(vr, gain_scores[g])]
                    objectives[g], reports[g], supports[g] = objective(vr, preds)
                    gc += [dict(protection(r, r["reference"], n), path=str(r["path"]), gain=g) for r, n in zip(vr, preds)]
                obj = float(np.mean(list(objectives.values()))); safe = all(c["qualified"] for c in nc+gc)
                native_ok = objectives["1.0"] <= baseline["1.0"][0]+1e-9
                candidates.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "policy": policy,
                    "gain_validation": reports, "gain_control_support": supports, "gain_objectives": objectives,
                    "objective": obj, "previous_objective": previous, "qualified_protection": bool(safe),
                    "native_objective_noninferior": bool(native_ok), "native_failures": [c for c in nc if not c["qualified"]],
                    "gain_failures": [c for c in gc if not c["qualified"]],
                    "accepted_improvement": bool(safe and native_ok and obj < previous-1e-4)})
        print("R294 SOURCE arm", spec["name"], "CE", model.best_loss, "accepted", sum(c["accepted_improvement"] for c in candidates), flush=True)
    accepted = [c for c in candidates if c["accepted_improvement"]]; selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R294-MatchedWeakCoverage-1", "heads": heads, "candidates": candidates,
        "baseline_gain_validation": {g: v[1] for g, v in baseline.items()}, "baseline_gain_control_support": {g: v[2] for g, v in baseline.items()},
        "baseline_gain_objectives": {g: v[0] for g, v in baseline.items()}, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    helpers = ("training/multitask_response_mlp.py", "training/probe_r290_shared_action_context.py",
        "training/probe_r292_shared_multitask_response.py", "training/probe_r293_decoupled_support_context.py")
    tracked = [Path(__file__), *[ROOT/h for h in helpers], OUT/"declared_search.json", OUT/"source_partition.json",
        OUT/"training_report.json", OUT/"common_validation_samples.json", *[ROOT/h["file"] for h in heads], *OUT.glob("samples_*.json")]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R294-FrozenMatchedWeakCoverage-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (REFERENCE_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(REFERENCE_OUT/"freeze.json"),
        "no_XYZ_arming": True, "base_models_unchanged": True, "strength_only_contextual_release": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R294 accepted", bool(accepted), "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__": main()
