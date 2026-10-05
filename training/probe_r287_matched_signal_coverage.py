"""Matched SOURCE augmentation control with identical points and budgets.

Five repeats of native features are compared with five response-gain views.
Training labels must remain supported by magnetic evidence in every view.
Every arm uses identical SOURCE points, class weights, architecture and seed.
The policy is frozen from SOURCE R286, not reselected using Yellow outcomes.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r286_signal_sequence import (
    ROOT, INVENTORY, PRIOR_OUT, source_partition, load_source, TRAIN_GAINS, VAL_GAINS, TAU_S,
    SPECS, augmented_signal_row, action_annotations, sequence_features, predict_sequence, add_gain_direction,
    temporal_freeze, direction_freeze, strength_freeze, load_frozen, load_temporal, load_model,
    temporal_infer, score_prediction, aggregate, policy_objective, qualifies, StateMLP, RAW_FILES, save_json, sha)
from training.verify_r286_signal_sequence import verify_freeze as augmentation_freeze
from training.train_r275_source_only_response import assert_source

OUT = ROOT/"analysis/r287_matched_source_signal_coverage_20260929"
AUG_OUT = ROOT/"analysis/r286_source_signal_sequence_20260929"
SEEDS = (287, 288)
SPEC = {"name": "ordered_ratio", "architecture": "MLP", "variant": "ratio", "width": 32}


def common_ids(views, cap=200):
    """Only same-label magnetic evidence windows; not gain-invented actions."""
    if not isinstance(cap, int) or cap <= 0: raise ValueError("positive SOURCE cap required")
    labels = np.vstack([action_annotations(v, "all_depth_action") for v in views])
    stable = np.all(labels == labels[:1], axis=0) & np.logical_and.reduce([v["eligible"] for v in views])
    result = []
    for cls in range(3):
        ids = np.flatnonzero(stable & (labels[0] == cls))
        ids = ids[::max(1, int(np.ceil(len(ids)/cap)))]
        if len(ids): result.append((cls, ids))
    return result


def matched_dataset(rows, null, validate=False, cap=200):
    assert_source(rows)
    if any(r["split"] not in ("train", "validation") for r in rows): raise ValueError("reserved SOURCE test excluded")
    native, gain, ys, ws, sample_ids = [], [], [], [], []
    split = "validation" if validate else "train"
    gains = VAL_GAINS if validate else TRAIN_GAINS
    for row in rows:
        if row["split"] != split: continue
        views = [augmented_signal_row(row, g, null, TAU_S) for g in gains]
        native_view = row if 1. in gains else augmented_signal_row(row, 1., null, TAU_S)
        for cls, ids in common_ids(views, cap):
            same = sequence_features(native_view, "ratio", 16, .1, ids).reshape(len(ids), -1)
            for g, view in zip(gains, views):
                gain.append(sequence_features(view, "ratio", 16, .1, ids).reshape(len(ids), -1))
                native.append(same); ys.append(np.full(len(ids), cls))
                ws.append(np.full(len(ids), 1./len(ids)/len(gains)))
                sample_ids.append({"path": str(row["path"]), "gain": g, "class": cls,
                    "n": len(ids), "indices_sha256": __import__("hashlib").sha256(ids.astype("<i8").tobytes()).hexdigest()})
    x0, x1, y, w = np.vstack(native), np.vstack(gain), np.concatenate(ys), np.concatenate(ws)
    for cls in range(3):
        if not np.any(y == cls): raise ValueError("missing SOURCE direction class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    assert x0.shape == x1.shape and x0.shape[1] == 256
    # Validation both arms see the very same gain stress views, not only native.
    if validate: x0 = x1
    return x0, x1, y, w, sample_ids


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    prior = temporal_freeze(PRIOR_OUT); aug = augmentation_freeze(AUG_OUT)
    policy = aug["selected"]["policy"]
    direction = direction_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength = strength_freeze(ROOT/Path(direction["base_protocol"]).parent)
    base, _ = load_frozen(); strength_models = [load_model(ROOT/f) for f in direction["baseline_models"]]
    direction_model = load_model(ROOT/direction["selected"]["file"]); temporal_model = load_temporal(prior["selected"])
    args = (temporal_model, prior, direction, direction_model, strength, strength_models, base)
    declared = {"architecture": SPEC, "parameters": 9379, "seeds": SEEDS, "epochs": 80,
        "cap_per_Trial_class_view": 200, "train_views": TRAIN_GAINS, "validation_views": VAL_GAINS, "tau_s": TAU_S,
        "matched_control": "native inputs repeated 5 times; identical source indices, labels, weights and update budget",
        "label_scope": "same source action annotations/evidence in every view; shallow command is not physical contact",
        "fixed_SOURCE_R286_policy": policy, "qualification": "native SOURCE train + validation per-Trial no added lift/control; all validation gains safety; strength unchanged; native objective noninferior; strict mean gain improvement vs R282 and matched native arm",
        "no_target_retained_source_previous_tip_selection": True}
    save_json(OUT/"declared_search.json", declared)
    recs = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    recs = [r for r in recs if r["device"] == "source" and r["date"] >= "S20260926_A" and
        r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "whole_Trial": True, "target_opened": 0, "source_test_opened": 0})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        row = load_source(rec, base["null"]["source"]); _, _, row["prior"] = temporal_infer(row, *args); rows.append(row)
        if (i+1) % 10 == 0: print("R287 SOURCE matched load", i+1, "/", len(recs), flush=True)
    native_items = [score_prediction(r, r["prior"]) for r in rows]
    validation = {}
    for g in VAL_GAINS:
        vr = [r for r in rows if r["split"] == "validation"] if g == 1. else [
            augmented_signal_row(r, g, base["null"]["source"], TAU_S) for r in rows if r["split"] == "validation"]
        if g != 1.:
            for r in vr: _, _, r["prior"] = temporal_infer(r, *args)
        old_items = [score_prediction(r, r["prior"]) for r in vr]
        validation[str(g)] = {"rows": vr, "items": old_items, "aggregate": aggregate(old_items)}
    previous = float(np.mean([policy_objective(v["aggregate"]) for v in validation.values()]))
    tx0, tx1, ty, tw, train_ids = matched_dataset(rows, base["null"]["source"])
    vx0, vx1, vy, vw, val_ids = matched_dataset(rows, base["null"]["source"], True)
    save_json(OUT/"matched_samples.json", {"train_samples": len(ty), "validation_samples": len(vy),
        "train": train_ids, "validation": val_ids, "identical_points_weights_classes_update_budget": True})
    print("R287 matched samples", tx0.shape, vx0.shape, flush=True)
    heads, candidates = [], []
    for seed in SEEDS:
        pair = []
        for arm, tx in (("native_repeat", tx0), ("gain_views", tx1)):
            model = StateMLP(32, seed=seed, n_classes=3).fit(tx, ty, tw, (vx1, vy, vw), epochs=80)
            file = OUT/f"direction_{arm}_seed{seed}.npz"; model.save(file)
            gain_reports, safe = {}, True
            for gain, value in validation.items():
                items = [score_prediction(r, add_gain_direction(r, r["prior"], predict_sequence(r, model, SPEC), policy)) for r in value["rows"]]
                gain_reports[gain] = aggregate(items); safe &= qualifies(items, value["items"])
            ni = [score_prediction(r, add_gain_direction(r, r["prior"], predict_sequence(r, model, SPEC), policy)) for r in rows]
            native_safe = qualifies(ni, native_items)
            obj = float(np.mean([policy_objective(v) for v in gain_reports.values()]))
            native_ok = policy_objective(gain_reports["1.0"]) <= policy_objective(validation["1.0"]["aggregate"])+1e-12
            item = {"arm": arm, "seed": seed, "spec": SPEC, "file": file.relative_to(ROOT).as_posix(), "policy": policy,
                "epoch": model.best_epoch, "validation_ce": model.best_loss, "history": model.history,
                "gain_validation": gain_reports, "gain_safety": bool(safe), "native_all_source_safety": bool(native_safe),
                "native_noninferior": bool(native_ok), "objective": obj, "previous_objective": previous,
                "native_train": aggregate([v for v, r in zip(ni, rows) if r["split"] == "train"])}
            heads.append(item); pair.append(item)
            print("R287 SOURCE", arm, seed, "epoch", model.best_epoch, "CE", round(model.best_loss, 6),
                "objective", round(obj, 6), "gain/native safety", bool(safe), bool(native_safe), flush=True)
        gain = dict(pair[1], matched_control_objective=pair[0]["objective"])
        gain["accepted_improvement"] = bool(gain["gain_safety"] and gain["native_all_source_safety"] and gain["native_noninferior"] and
            gain["objective"] < min(previous, pair[0]["objective"])-1e-4)
        candidates.append(gain)
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R287-MatchedSourceCoverage-1", "heads": heads, "candidates": candidates,
        "selected": selected, "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "baseline_gain_validation": {g: v["aggregate"] for g, v in validation.items()},
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/response_signal_augmentation.py", OUT/"declared_search.json",
        OUT/"source_partition.json", OUT/"matched_samples.json", OUT/"training_report.json"]+[ROOT/h["file"] for h in heads]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R287-FrozenMatchedSourceCoverage-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "policy_protocol": (AUG_OUT/"freeze.json").relative_to(ROOT).as_posix(), "policy_protocol_sha256": sha(AUG_OUT/"freeze.json"),
        "steps": 16, "step_s": .1, "no_XYZ_arming": True, "strength_frozen_R276": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked], "source_input_manifest": inputs})
    print("FROZEN R287 accepted", bool(accepted), "selected seed", None if not selected else selected["seed"], flush=True)


if __name__ == "__main__": main()
