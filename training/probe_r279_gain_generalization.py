"""Source-only signal-gain augmentation; fixed R278 rules, no target selection.

Positive global field gains are a declared amplitude stress test, not new
physical devices or contact truth. Evidence is recomputed with the frozen
source noise model; unsupported nominal actions do not become positives.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r278_independent_direction import (
    ROOT, OUT as PRIOR_OUT, features as prior_features, action_annotations, supplement, qualifies)
from training.audit_r278_independent_direction import verify_freeze as prior_freeze
from training.probe_r276_condition_adaptation import INVENTORY, source_partition, load_source, score_prediction, aggregate, policy_objective
from training.audit_r276_condition_adaptation import verify_freeze as strength_freeze, infer as strength_infer
from training.audit_r275_source_only_response import load_frozen
from training.audit_r263_magnetic_evidence_uncertain_depth import evidence_scores, confirm_evidence
from training.audit_r274_frozen_phase_expansion import RAW_FILES, save_json
from training.probe_r269_hierarchical_response_state import dataset
from training.probe_r245_joint_contact_state import StateMLP, causal_state_features
from training.verify_r273_response_runtime import load_model
from training.build_trajectory_dataset_v1 import sha

OUT = ROOT/"analysis/r279_source_gain_generalization_20260929"
VALIDATION_GAINS = (.5, 1., 2.)
CANDIDATES = (("relative_wide", "relative", (.5, .75, 1., 1.5, 2.)),
    ("relative_narrow", "relative", (.75, 1., 1.5)), ("ratio_wide", "ratio", (.5, .75, 1., 1.5, 2.)))


def features(row, variant):
    value = prior_features(row, "relative")
    if variant == "relative": return value
    if variant == "ratio": return value[:, 16:]
    raise ValueError("only predeclared source feature variants allowed")


def gain_row(row, gain, base):
    if not np.isfinite(gain) or gain <= 0: raise ValueError("positive finite magnetic field gain required")
    value = dict(row, field_uT=row["field_uT"]*gain, gain=float(gain))
    value["state_x"] = causal_state_features(value["time"], value["field_uT"], value["eligible"])
    a, d = evidence_scores(value, base["null"]["source"])
    value["response"] = confirm_evidence(value, a, d, base["null"]["source"])
    value["amplitude_uT"] = np.linalg.norm(value["field_uT"], axis=1)
    return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); out = args.out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); start = time.monotonic()
    prior = prior_freeze(PRIOR_OUT); base, _ = load_frozen()
    strength_protocol = strength_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength_models = [load_model(ROOT/f) for f in prior["baseline_models"]]
    old_model = load_model(ROOT/prior["selected"]["file"]); policy = prior["selected"]["policy"]
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    records = [r for r in inventory if r["device"] == "source" and r["date"] >= "S20260926_A"
        and r["initially_readable"] and source_partition(r) != "test"]
    save_json(out/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in records],
        "source_test_opened": 0, "target_opened": 0, "same_partition_as_R276": True})
    save_json(out/"declared_search.json", {"validation_gains": VALIDATION_GAINS, "candidates": CANDIDATES,
        "policy_frozen_R278": policy, "physical_scope": "amplitude stress only, not simulated new contact/device",
        "selection": "per-Trial no added lift/control actions in each gain; original-source validation noninferiority; strictly better equally weighted mean gain objective"})
    rows = []
    for i, rec in enumerate(records):
        rows.append(load_source(rec, base["null"]["source"]))
        if (i+1) % 10 == 0: print("R279 source load", i+1, "/", len(records), flush=True)
    validation = {}
    for gain in VALIDATION_GAINS:
        vr = [gain_row(r, gain, base) for r in rows if r["split"] == "validation"]
        old_items = []
        for row in vr:
            row["strength_baseline"] = strength_infer(row, strength_models, strength_protocol, base)
            row["prior"] = supplement(row, row["strength_baseline"], old_model.predict(prior_features(row, "relative")), policy)
            old_items.append(score_prediction(row, row["prior"]))
        validation[str(gain)] = {"rows": vr, "items": old_items, "aggregate": aggregate(old_items)}
    reports = []
    for name, variant, gains in CANDIDATES:
        development = []
        for row in rows:
            if row["split"] == "train":
                for gain in gains:
                    value = gain_row(row, gain, base)
                    value["direction_y"] = action_annotations(value, "all_depth_action")
                    development.append(value)
            else:
                # All validation gains are held out from fit; CE epoch selection
                # uses equal source-Trial/class weighting across these gains.
                for gain in VALIDATION_GAINS:
                    value = gain_row(row, gain, base)
                    value["direction_y"] = action_annotations(value, "all_depth_action")
                    development.append(value)
        tx, ty, tw = dataset(development, "direction", variant, False, features)
        vx, vy, vw = dataset(development, "direction", variant, True, features)
        model = StateMLP(32, seed=279, n_classes=3).fit(tx, ty, tw, (vx, vy, vw), epochs=120)
        file = out/f"direction_{name}.npz"; model.save(file)
        gain_scores = {}; qualified = True
        for gain, value in validation.items():
            items = [score_prediction(r, supplement(r, r["strength_baseline"], model.predict(features(r, variant)), policy)) for r in value["rows"]]
            qualified &= qualifies(items, value["items"])
            gain_scores[gain] = aggregate(items)
        original_noninferior = policy_objective(gain_scores["1.0"]) <= policy_objective(validation["1.0"]["aggregate"])+1e-4
        mean_objective = float(np.mean([policy_objective(v) for v in gain_scores.values()]))
        previous_objective = float(np.mean([policy_objective(v["aggregate"]) for v in validation.values()]))
        accepted = bool(qualified and original_noninferior and mean_objective < previous_objective-1e-4)
        report = {"name": name, "variant": variant, "training_gains": gains, "input_features": tx.shape[1],
            "file": file.relative_to(ROOT).as_posix(), "epoch": model.best_epoch, "validation_ce": model.best_loss,
            "train_counts": np.bincount(ty, minlength=3).tolist(), "validation_counts": np.bincount(vy, minlength=3).tolist(),
            "gain_validation": gain_scores, "qualified_safety": bool(qualified), "original_noninferior": bool(original_noninferior),
            "objective": mean_objective, "previous_objective": previous_objective, "accepted_improvement": accepted,
            "history": model.history}
        reports.append(report)
        print("R279 source", name, "epoch", model.best_epoch, "CE", round(model.best_loss, 6), "safety", bool(qualified),
            "original_noninferior", bool(original_noninferior), "objective", round(mean_objective, 6), "baseline", round(previous_objective, 6), flush=True)
        # Avoid retaining all augmented native streams during the next model fit.
        del development, tx, ty, tw, vx, vy, vw
    accepted = [r for r in reports if r["accepted_improvement"]]
    selected = min(accepted, key=lambda r: r["objective"]) if accepted else None
    save_json(out/"training_report.json", {"schema": "R279-SourceGainGeneralization-1", "candidates": reports,
        "baseline_gain_validation": {g: v["aggregate"] for g, v in validation.items()},
        "selected": selected, "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-start,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), out/"declared_search.json", out/"source_partition.json", out/"training_report.json"]
    if selected: tracked.append(ROOT/selected["file"])
    inputs = [ROOT/r["path"]/name for r in records for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    save_json(out/"freeze.json", {"schema": "R279-FrozenSourceGain-1", "frozen_before_target_replay": True,
        "accepted_improvement": bool(accepted), "selected": selected, "policy": policy,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "strength_protocol": prior["base_protocol"], "baseline_models": prior["baseline_models"],
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "no_XYZ_arming": True, "strength_frozen_R276": True,
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked],
        "source_input_manifest": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in inputs]})
    print("FROZEN R279 accepted", bool(accepted), "selected", selected["name"] if selected else None, flush=True)


if __name__ == "__main__": main()
