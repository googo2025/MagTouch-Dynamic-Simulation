"""Matched SOURCE error-mining intervention; no target-trained expert."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r294_matched_weak_coverage import (
    ROOT, INVENTORY, REFERENCE_OUT, RELEASE_OUT, CONTEXT_OUT,
    OUT as TEACHER_OUT, load_prior_args, prior_infer, protected_veto,
    load_source, source_partition, TRAIN_GAINS, VAL_GAINS, TAU_S,
    augmented_signal_row, predict_sequence, predict_multitask, load_model,
    score_prediction, aggregate, RAW_FILES, save_json, sha, shared_features,
    common_ids, joint_labels, coverage_dataset, stratified_ids, uniform_ids,
    replace_coverage, budget, decoupled_release, release_supported, protection,
    reference_freeze, release_freeze, context_freeze, assert_source, MultiTaskMLP, objective)
from training.verify_r294_matched_weak_coverage import verify_freeze as teacher_freeze
from training.audit_r274_frozen_phase_expansion import blocks

OUT = ROOT/"analysis/r295_source_matched_hard_context_20260929"
SPECS = ({"name": "weak_uniform_context", "sampling": "weak_uniform_context", "variant": "global", "width": 48},
         {"name": "weak_hard_context", "sampling": "weak_hard_context", "variant": "global", "width": 48})


def hard_ids(view, cls, fixed, scores, quota=70):
    """Replace at most half of a class budget using SOURCE train difficulty."""
    if view["device"] != "source" or view["split"] != "train":
        raise ValueError("hard mining is SOURCE train only")
    if cls not in (2, 3): raise ValueError("only static/quiet context sampling may change")
    fixed = np.asarray(fixed, int); y = joint_labels(view)
    if len(np.unique(fixed)) != len(fixed) or not np.all(y[fixed] == cls):
        raise ValueError("valid unique fixed class budget required")
    p, c = scores["direction"], scores["context"]
    if p.shape != (len(y), 3) or c.shape != (len(y), 2) or not np.isfinite(p).all() or not np.isfinite(c).all():
        raise ValueError("aligned frozen difficulty scores required")
    mask = y == cls
    if cls == 2: mask &= np.isfinite(view["q"]).all(1) & (view["q"][:, 2] >= .4)
    available = np.flatnonzero(mask)
    n = min(int(quota), len(fixed)//2, len(available))
    value = .5*np.max(p[:, :2], axis=1)+.5*(c[:, 0] if cls == 2 else c[:, 1])
    order = np.lexsort((available, -value[available])); chosen = available[order[:n]]
    rest = fixed[~np.isin(fixed, chosen)]
    selected = np.unique(np.r_[chosen, uniform_ids(rest, len(fixed)-len(chosen))])
    assert len(selected) == len(fixed) and np.all(y[selected] == cls)
    return selected, chosen


def training_dataset(rows, null, teacher, hard):
    assert_source(rows)
    if any(r["split"] not in ("train", "validation") for r in rows): raise ValueError("SOURCE test excluded")
    xs, ys, ws, manifest = [], [], [], []
    for row in rows:
        if row["split"] != "train": continue
        views = [row if g == 1. else augmented_signal_row(row, g, null, TAU_S) for g in TRAIN_GAINS]
        common = common_ids(row, views)
        for g, view in zip(TRAIN_GAINS, views):
            scores = predict_multitask(view, teacher, {"variant": "global"}) if hard else None
            for cls, fixed in common:
                ids = stratified_ids(view, cls, len(fixed)) if cls in (0, 1) else fixed
                chosen = np.zeros(0, int)
                if hard and cls in (2, 3): ids, chosen = hard_ids(view, cls, ids, scores)
                assert np.all(joint_labels(view)[ids] == cls)
                xs.append(shared_features(view, "global", ids).astype(np.float32))
                ys.append(np.full(len(ids), cls)); ws.append(np.full(len(ids), 1./len(ids)/len(TRAIN_GAINS)))
                group = "weak_loading" if cls == 0 else "weak_withdrawal" if cls == 1 else "none"
                weak = view["diagnostic_labels"] == group
                manifest.append({"path": str(row["path"]), "gain": g, "class": cls, "n": len(ids),
                    "weak_frames": int(weak[ids].sum()), "hard_context_points": len(chosen),
                    "indices_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                    "hard_indices_sha256": hashlib.sha256(chosen.astype("<i8").tobytes()).hexdigest()})
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(4):
        if not np.any(y == cls): raise ValueError("missing SOURCE class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    return x, y, w, manifest


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"specs": SPECS, "seed": 295, "epochs": 80, "context_weight": 1.,
        "train_gains": TRAIN_GAINS, "validation_gains": VAL_GAINS, "tau_s": TAU_S,
        "architecture": "512-48-48 shared tanh; independent direction3/context2; 27221 parameters",
        "training_sampling": "both arms retain R294 per-view shallow action points; only quiet/static context points change",
        "difficulty_teacher": "SOURCE-only frozen R294 per-view-weak head; training Trials only",
        "difficulty": ".5*max(direction loading/withdrawal score) + .5*context wrong-class score",
        "hard_context_quota": "at most 70 and half of each SOURCE train Trial/class/view budget; class2 mining only nominal Z>=.4 action-reference frames",
        "sample_budget": "identical count/y/task weights per Trial/class/gain; no added observations or duplicated frames",
        "validation": "identical fixed R294 stratified SOURCE validation points; no validation error mining",
        "optimization": "balanced task CE sum + .0005*weight squared; Adam .001, ~256 batch; 80 epochs, validation CE every5",
        "direction_limits": (.9, .97, .995), "context_limits": (.97, .995),
        "direction_confirm_s": .03, "context_confirm_s": .03, "release_limit": .995, "release_confirm_s": .03,
        "objective": "SOURCE validation policy_objective + 2*Trial-macro control-supported fraction, mean gains .5/1/2",
        "qualification": "every SOURCE native/gain Trial retains exact correct weak frames/hits and deep support/action; control actions/support nonincrease; native objective noninferior and mean gain improvement >1e-4",
        "no_target_or_SOURCE_test_selection": True, "no_XYZ_arming": True}
    save_json(OUT/"declared_search.json", declared)
    teacher_freeze(TEACHER_OUT)
    teacher = MultiTaskMLP.load(TEACHER_OUT/"multitask_per_view_weak.npz")
    ref, release, context = reference_freeze(REFERENCE_OUT), release_freeze(RELEASE_OUT), context_freeze(CONTEXT_OUT)
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
        if (i+1)%10 == 0: print("R295 SOURCE load", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: objective(vr, [r["reference"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v[0] for v in baseline.values()]))
    vx, vy, vw, val_ids = coverage_dataset(rows, base["null"]["source"], "per_view_weak", True)
    save_json(OUT/"common_validation_samples.json", {"samples": val_ids, "n": len(vy)})
    heads, candidates, common_budget, common_y, common_w = [], [], None, None, None
    for spec in SPECS:
        tx, ty, tw, train_ids = training_dataset(rows, base["null"]["source"], teacher, spec["name"] == "weak_hard_context")
        if common_budget is None: common_budget, common_y, common_w = budget(train_ids), ty.copy(), tw.copy()
        else:
            assert common_budget == budget(train_ids)
            np.testing.assert_array_equal(ty, common_y); np.testing.assert_array_equal(tw, common_w)
        save_json(OUT/f"samples_{spec['name']}.json", {"train": train_ids, "validation": val_ids,
            "train_samples": len(ty), "validation_samples": len(vy)})
        print("R295 SOURCE train", spec["name"], tx.shape, vx.shape, flush=True)
        model = MultiTaskMLP(48, 295, 1.).fit(tx, ty, tw, (vx, vy, vw), 80)
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
        print("R295 SOURCE arm", spec["name"], "CE", model.best_loss, "accepted", sum(c["accepted_improvement"] for c in candidates), flush=True)
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R295-MatchedHardContext-1", "heads": heads, "candidates": candidates,
        "baseline_gain_validation": {g: v[1] for g, v in baseline.items()}, "baseline_gain_control_support": {g: v[2] for g, v in baseline.items()},
        "baseline_gain_objectives": {g: v[0] for g, v in baseline.items()}, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/multitask_response_mlp.py", ROOT/"training/probe_r294_matched_weak_coverage.py",
        OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json", OUT/"common_validation_samples.json",
        *[ROOT/h["file"] for h in heads], *OUT.glob("samples_*.json")]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R295-FrozenMatchedHardContext-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (REFERENCE_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(REFERENCE_OUT/"freeze.json"),
        "teacher_protocol": (TEACHER_OUT/"freeze.json").relative_to(ROOT).as_posix(), "teacher_protocol_sha256": sha(TEACHER_OUT/"freeze.json"),
        "no_XYZ_arming": True, "base_models_unchanged": True, "strength_only_contextual_release": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R295 accepted", bool(accepted), "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__": main()
