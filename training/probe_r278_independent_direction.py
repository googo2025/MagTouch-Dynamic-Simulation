"""SOURCE-only direction recognition with explicit quiet/residual negatives.

R276 strength and existing action declarations stay frozen. Only causally
confirmed additional magnetic action phases are considered. No target or
reserved source test data is opened for training/selection. Never arm XYZ.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from training.probe_r276_condition_adaptation import (
    ROOT, INVENTORY, source_partition, load_source, run_scores, score_prediction, aggregate, policy_objective,
    invariant_features)
from training.audit_r276_condition_adaptation import verify_freeze, infer as infer_baseline
from training.audit_r275_source_only_response import load_frozen
from training.audit_r274_frozen_phase_expansion import RAW_FILES, save_json
from training.probe_r269_hierarchical_response_state import SCALAR_IDS, dataset
from training.probe_r245_joint_contact_state import StateMLP
from training.verify_r273_response_runtime import load_model
from training.independent_response_direction import SupplementPolicy, replay_supplement
from training.build_trajectory_dataset_v1 import sha

OUT = ROOT / "analysis/r278_source_independent_direction_20260929"


def action_annotations(row, scope):
    q, labels = row["q"], row["diagnostic_labels"]
    y = np.full(len(q), -1, int)
    safe = np.isin(labels, ("pre_action_lift", "lift_residue_0to3s", "late_lift", "protocol_no_motion"))
    # All safe quiet and lifted-residue records are explicit neutral controls,
    # including those without a magnetic response. They are not ignored.
    y[row["eligible"] & safe] = 2
    finite = np.isfinite(q).all(1)
    continuity = np.r_[False, (np.diff(row["time"]) > 0) & (np.diff(row["time"]) <= 250_000_000)]
    use = row["eligible"] & row["response"] & finite & np.r_[False, finite[:-1]] & continuity & (q[:, 2] >= .02)
    dz = np.r_[0., np.diff(q[:, 2])]
    y[use & (np.abs(dz) <= 1e-4)] = 2
    moving = use if scope == "all_depth_action" else use & (q[:, 2] < .4)
    y[moving & (dz > 1e-4)] = 0
    y[moving & (dz < -1e-4)] = 1
    return y


def features(row, variant):
    if variant == "scalar": return row["state_x"][:, SCALAR_IDS]
    if variant == "relative": return invariant_features(row["state_x"])
    raise ValueError("known source-defined feature variant required")


def supplement(row, baseline, scores, policy):
    phase, added, conflict = replay_supplement(row["time"], baseline["phase"], baseline["strength"],
        scores, row["response"], row["state_x"][:, 1]*100., row["eligible"], SupplementPolicy(**policy))
    return dict(baseline, phase=phase, direction_supplemented=added, direction_context_conflict=conflict)


def qualifies(items, baseline_items):
    # The unchanged strength layer guarantees deep/grid preservation exactly.
    # Compare every validation Trial, not just a pooled false-action average.
    limits = ("safe_lift_loading_fraction", "safe_lift_any_action_fraction",
        "protocol_no_motion_loading_fraction", "protocol_no_motion_any_action_fraction")
    for new, old in zip(items, baseline_items):
        for key in limits:
            if new["rates"].get(key, 0.) > old["rates"].get(key, 0.)+1e-12: return False
        if abs(new["rates"].get("supported_state_preservation", 0.)-old["rates"].get("supported_state_preservation", 0.)) > 1e-12:
            raise AssertionError("a phase supplement must not change strength")
    return True


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); out = args.out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); start = time.monotonic()
    prior_out = ROOT/"analysis/r276_source_condition_adaptation_20260929"
    prior = verify_freeze(prior_out); base, _ = load_frozen()
    baseline_models = [load_model(ROOT/f) for f in prior["models"]]
    records = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    records = [r for r in records if r["device"] == "source" and r["date"] >= "S20260926_A" and
        r["initially_readable"] and source_partition(r) != "test"]
    save_json(out/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r), "mode": r["mode"]} for r in records],
        "source_test_opened": 0, "target_opened": 0, "whole_trial": True, "same_partition_as_R276": True})
    rows = []
    for i, rec in enumerate(records):
        row = load_source(rec, base["null"]["source"])
        row["baseline"] = infer_baseline(row, baseline_models, prior, base)
        rows.append(row)
        if (i+1) % 10 == 0: print("source independent-direction load", i+1, "/", len(records), flush=True)
    validation = [r for r in rows if r["split"] == "validation"]
    baseline_items = [score_prediction(r, r["baseline"]) for r in validation]
    baseline_value = aggregate(baseline_items)
    reports, candidates = [], []
    # This search is declared before any target or retained source test replay.
    for scope in ("weak_action", "all_depth_action"):
        for row in rows: row["direction_y"] = action_annotations(row, scope)
        for variant in ("scalar", "relative"):
            tx, ty, tw = dataset(rows, "direction", variant, False, features)
            vx, vy, vw = dataset(rows, "direction", variant, True, features)
            model = StateMLP(32, seed=278, n_classes=3).fit(tx, ty, tw, (vx, vy, vw), epochs=120)
            file = out/f"direction_{scope}_{variant}.npz"; model.save(file)
            reports.append({"scope": scope, "variant": variant, "width": 32, "features": tx.shape[1],
                "epoch": model.best_epoch, "validation_ce": model.best_loss,
                "train_counts": np.bincount(ty, minlength=3).tolist(), "validation_counts": np.bincount(vy, minlength=3).tolist(),
                "history": model.history, "file": file.relative_to(ROOT).as_posix()})
            print("source independent-direction train", scope, variant, model.best_epoch, round(model.best_loss, 6), flush=True)
            scores = [model.predict(features(r, variant)) for r in validation]
            for limit in (.75, .9, .97, .995):
                for confirm in (.015, .03):
                    for residue in (False, True):
                        for signed in (False, True):
                            policy = asdict(SupplementPolicy(limit, confirm, residue, signed))
                            items = [score_prediction(r, supplement(r, r["baseline"], s, policy)) for r, s in zip(validation, scores)]
                            value = aggregate(items)
                            candidates.append({"scope": scope, "variant": variant, "file": file.relative_to(ROOT).as_posix(),
                                "policy": policy, "validation": value, "objective": policy_objective(value),
                                "qualified": qualifies(items, baseline_items),
                                "strict_improvement": policy_objective(value) < policy_objective(baseline_value)-1e-4})
            accepted = [c for c in candidates if c["qualified"] and c["strict_improvement"]]
            print("source independent-direction candidates", len(candidates), "qualified improvements", len(accepted), flush=True)
    accepted = [c for c in candidates if c["qualified"] and c["strict_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(out/"training_report.json", {"schema": "R278-IndependentSourceDirection-1", "heads": reports,
        "candidates": candidates, "baseline_R276_source_validation": baseline_value,
        "selected": selected, "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-start,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_opened": 0,
        "qualification": "per-source-validation-Trial no added safe-lift loading/any-action/control actions; exact R276 strength; trial-macro objective lower by >1e-4",
        "no_shallow_contact_or_XYZ_truth": True})
    tracked = [Path(__file__), ROOT/"training/independent_response_direction.py",
        out/"source_partition.json", out/"training_report.json"]
    if selected: tracked.append(ROOT/selected["file"])
    inputs = [ROOT/r["path"]/name for r in records for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    save_json(out/"freeze.json", {"schema": "R278-FrozenIndependentSourceDirection-1",
        "frozen_before_target_replay": True, "accepted_improvement": bool(accepted),
        "selected": selected, "baseline_models": prior["models"],
        "base_protocol": (prior_out/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(prior_out/"freeze.json"),
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked],
        "source_input_manifest": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in inputs],
        "no_XYZ_arming": True, "strength_frozen_R276": True})
    print("FROZEN R278", "accepted", bool(accepted), "selected", json.dumps(selected), flush=True)


if __name__ == "__main__": main()
