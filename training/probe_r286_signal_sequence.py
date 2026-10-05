"""SOURCE-only ordered direction learning under causal response-gain proxies.

The low-frequency component is an augmentation proxy, not an identified
material response. All gains of a Trial stay in its fixed SOURCE split.
Yellow, retained SOURCE tests, and previous cross-tip stress records are
excluded from model/threshold/epoch selection. This layer cannot arm XYZ.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r282_causal_sequence import predict_sequence, load_temporal, sequence_features
from training.audit_r282_causal_sequence import verify_freeze as temporal_freeze, infer as temporal_infer
from training.probe_r278_independent_direction import (
    ROOT, INVENTORY, source_partition, load_source, action_annotations, supplement, qualifies)
from training.audit_r278_independent_direction import verify_freeze as direction_freeze
from training.audit_r276_condition_adaptation import verify_freeze as strength_freeze
from training.audit_r275_source_only_response import load_frozen
from training.probe_r276_condition_adaptation import score_prediction, aggregate, policy_objective
from training.audit_r274_frozen_phase_expansion import RAW_FILES, save_json
from training.probe_r245_joint_contact_state import StateMLP
from training.independent_response_direction import SupplementPolicy
from training.verify_r273_response_runtime import load_model
from training.build_trajectory_dataset_v1 import sha
from training.response_signal_augmentation import augmented_signal_row
from training.train_r275_source_only_response import assert_source

OUT = ROOT/"analysis/r286_source_signal_sequence_20260929"
PRIOR_OUT = ROOT/"analysis/r282_source_causal_sequence_20260929"
TRAIN_GAINS = (.5, .75, 1., 1.5, 2.)
VAL_GAINS = (.5, 1., 2.)
TAU_S = .05
SPECS = ({"name": "ordered_relative", "architecture": "MLP", "variant": "relative", "width": 32},
         {"name": "ordered_ratio", "architecture": "MLP", "variant": "ratio", "width": 32})


def add_gain_direction(row, prior, scores, policy):
    value = supplement(row, prior, scores, policy)
    value["gain_direction_supplemented"] = value["direction_supplemented"]
    value["gain_direction_context_conflict"] = value["direction_context_conflict"]
    value["direction_supplemented"] = prior["direction_supplemented"] | value["gain_direction_supplemented"]
    value["direction_context_conflict"] = prior["direction_context_conflict"] | value["gain_direction_context_conflict"]
    return value


def gain_sequence_dataset(rows, variant, null, validate=False, cap=250):
    """Equal Trial/gain/class weighting; no retained augmented raw streams."""
    assert_source(rows)
    if any(r["split"] not in ("train", "validation") for r in rows):
        raise ValueError("retained SOURCE tests cannot enter augmented fitting")
    if not isinstance(cap, int) or cap <= 0: raise ValueError("positive sample cap required")
    xs, ys, ws = [], [], []
    split = "validation" if validate else "train"
    gains = VAL_GAINS if validate else TRAIN_GAINS
    for row in rows:
        if row["split"] != split: continue
        for gain in gains:
            value = augmented_signal_row(row, gain, null, TAU_S)
            labels = action_annotations(value, "all_depth_action")
            for cls in range(3):
                ids = np.flatnonzero(value["eligible"] & (labels == cls))
                if not len(ids): continue
                ids = ids[::max(1, int(np.ceil(len(ids)/cap)))]
                xs.append(sequence_features(value, variant, 16, .1, ids).reshape(len(ids), -1))
                ys.append(np.full(len(ids), cls)); ws.append(np.full(len(ids), 1./len(ids)/len(gains)))
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(3):
        if not np.any(y == cls): raise ValueError("missing SOURCE action class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    return x, y, w


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    prior = temporal_freeze(PRIOR_OUT)
    direction = direction_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength = strength_freeze(ROOT/Path(direction["base_protocol"]).parent)
    base, _ = load_frozen(); strength_models = [load_model(ROOT/f) for f in direction["baseline_models"]]
    direction_model = load_model(ROOT/direction["selected"]["file"]); temporal_model = load_temporal(prior["selected"])
    args = (temporal_model, prior, direction, direction_model, strength, strength_models, base)
    declared = {"specs": SPECS, "seed": 286, "train_gains": TRAIN_GAINS, "validation_gains": VAL_GAINS,
        "tau_s": TAU_S, "steps": 16, "step_s": .1, "epochs": 80, "batch": 256,
        "learning_rate": .001, "weight_decay": .001, "cap_per_Trial_gain_class": 250,
        "score_limits": (.9, .97, .995), "confirm_s": (.015, .03), "signed_change": (False, True),
        "physical_scope": "causal response-gain augmentation proxy; fast residual unchanged, not new devices",
        "qualification": "each SOURCE validation Trial/gain no added lift/control action; native objective noninferior; mean gain trial-macro objective strictly improves >1e-4; all R282 action/strength preserved",
        "no_XYZ_arming": True, "no_target_or_retained_source_selection": True}
    save_json(OUT/"declared_search.json", declared)
    recs = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    recs = [r for r in recs if r["device"] == "source" and r["date"] >= "S20260926_A" and
        r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "target_opened": 0, "source_test_opened": 0, "whole_Trial": True})
    inputs = [ROOT/r["path"]/name for r in recs for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    input_manifest = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in inputs]
    rows = []
    for i, rec in enumerate(recs):
        rows.append(load_source(rec, base["null"]["source"]))
        if (i+1) % 10 == 0: print("R286 SOURCE load", i+1, "/", len(recs), flush=True)
    validation = {}
    for gain in VAL_GAINS:
        vr = [augmented_signal_row(r, gain, base["null"]["source"], TAU_S) for r in rows if r["split"] == "validation"]
        for r in vr: _, _, r["prior"] = temporal_infer(r, *args)
        items = [score_prediction(r, r["prior"]) for r in vr]
        validation[str(gain)] = {"rows": vr, "items": items, "aggregate": aggregate(items)}
        print("R286 SOURCE augmented control gain", gain, "ready", flush=True)
    previous_objective = float(np.mean([policy_objective(v["aggregate"]) for v in validation.values()]))
    heads, candidates = [], []
    for spec in SPECS:
        tx, ty, tw = gain_sequence_dataset(rows, spec["variant"], base["null"]["source"])
        vx, vy, vw = gain_sequence_dataset(rows, spec["variant"], base["null"]["source"], True)
        print("R286 SOURCE train", spec["name"], tx.shape, vx.shape, flush=True)
        model = StateMLP(32, seed=286, n_classes=3).fit(tx, ty, tw, (vx, vy, vw), epochs=80)
        file = OUT/f"direction_{spec['name']}.npz"; model.save(file)
        heads.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "epoch": model.best_epoch,
            "validation_ce": model.best_loss, "parameters": sum(int(p.size) for p in model.params),
            "train_counts": np.bincount(ty, minlength=3).tolist(), "validation_counts": np.bincount(vy, minlength=3).tolist(),
            "history": model.history})
        scores = {g: [predict_sequence(r, model, spec) for r in v["rows"]] for g, v in validation.items()}
        for limit in declared["score_limits"]:
            for confirm in declared["confirm_s"]:
                for signed in declared["signed_change"]:
                    policy = asdict(SupplementPolicy(limit, confirm, True, signed))
                    gs, qualified = {}, True
                    for gain, value in validation.items():
                        items = [score_prediction(r, add_gain_direction(r, r["prior"], s, policy)) for r, s in zip(value["rows"], scores[gain])]
                        qualified &= qualifies(items, value["items"]); gs[gain] = aggregate(items)
                    obj = float(np.mean([policy_objective(v) for v in gs.values()]))
                    native_ok = policy_objective(gs["1.0"]) <= policy_objective(validation["1.0"]["aggregate"])+1e-12
                    candidates.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "policy": policy,
                        "gain_validation": gs, "qualified_safety": bool(qualified), "native_noninferior": bool(native_ok),
                        "objective": obj, "previous_objective": previous_objective,
                        "accepted_improvement": bool(qualified and native_ok and obj < previous_objective-1e-4)})
        print("R286 SOURCE model", spec["name"], "epoch", model.best_epoch, "CE", round(model.best_loss, 6),
            "qualified improvements", sum(c["accepted_improvement"] for c in candidates), flush=True)
        del tx, ty, tw, vx, vy, vw
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R286-SourceSignalSequence-1", "heads": heads,
        "candidates": candidates, "baseline_gain_validation": {g: v["aggregate"] for g, v in validation.items()},
        "selected": selected, "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/response_signal_augmentation.py", ROOT/"training/probe_r282_causal_sequence.py",
        ROOT/"training/causal_response_sequence.py", ROOT/"training/independent_response_direction.py",
        OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json"]
    if selected: tracked.append(ROOT/selected["file"])
    for r in input_manifest: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R286-FrozenSourceSignalSequence-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "steps": 16, "step_s": .1, "tau_s": TAU_S, "no_XYZ_arming": True, "strength_frozen_R276": True,
        "preserve_all_R282_loading_withdrawal": True, "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked], "source_input_manifest": input_manifest})
    print("FROZEN R286 accepted", bool(accepted), "selected", None if not selected else {k: selected[k] for k in ("spec", "policy", "objective")}, flush=True)


if __name__ == "__main__": main()
