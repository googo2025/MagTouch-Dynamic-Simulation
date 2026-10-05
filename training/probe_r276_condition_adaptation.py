"""Condition-diverse SOURCE-only training/selection, before target replay.

Whole-Trial partitions are declared from source metadata. Shallow commands
are action annotations, never physical contact/XYZ targets. Existing R275
models, output streams, raw records, UI, and paper are not modified.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from training.train_r275_source_only_response import ROOT, INVENTORY, assert_source
from training.audit_r275_source_only_response import load_frozen
from training.audit_r274_frozen_phase_expansion import (
    load_record, post_group, save_json, metrics, scalar_rates, RAW_FILES, replay as old_replay)
from training.probe_r245_joint_contact_state import causal_state_features, StateMLP
from training.probe_r269_hierarchical_response_state import SCALAR_IDS, dataset
from training.audit_r263_magnetic_evidence_uncertain_depth import evidence_scores, confirm_evidence
from training.response_phase_state import PhasePolicy
from training.response_phase_state_v3 import replay_rearm_phase
from training.response_phase_state_v4 import replay_evidence_rearm
from training.factorized_response_phase import DirectionPolicy, replay_factorized
from training.diagnose_r274_whole_action import whole_actions, summary as whole_summary
from training.build_trajectory_dataset_v1 import sha

OUT = ROOT / "analysis/r276_source_condition_adaptation_20260929"


def source_partition(rec):
    if rec["device"] != "source" or rec["date"] < "S20260926_A":
        raise ValueError("only the current source assembly belongs to this development")
    n = int(Path(rec["path"]).name.split("_")[1]); plan = rec["plan"]
    if "R133_E_" in plan or "GUARD_ONLINE_PAIR" in plan:
        return "test"
    if "R136_" in plan and n == 31 or "R208_SOURCE" in plan and n in (35, 39):
        return "test"
    if "R132_A_HAD239_R130_NEURAL" in plan and n in (13, 16, 18, 20, 22, 24, 26, 28):
        return "validation"
    if "R132_C_HAD239" in plan and n in (58, 59, 62, 63, 65):
        return "validation"
    if "R133_D_" in plan and n in (88, 94, 100, 104, 108):
        return "validation"
    if "R133_RETRAIN" in plan and n in (9, 11, 12):
        return "validation"
    return "train"


def invariant_features(x):
    scalar = x[:, SCALAR_IDS]
    # Per-observation dimensionless changes: field gain/rotation invariant.
    # Fixed 20 uT floor prevents a vanishing baseline from exploding ratios.
    relative = scalar[:, 1:] / np.maximum(scalar[:, :1], .2)
    return np.column_stack((scalar, np.clip(relative, -10., 10.)))


def features(row, variant):
    x = row["state_x"]
    return x[:, SCALAR_IDS] if variant == "scalar" else invariant_features(x)


def load_source(rec, null):
    row = load_record(rec)
    assert_source([row])
    row["state_x"] = causal_state_features(row["time"], row["field_uT"], row["eligible"])
    a, d = evidence_scores(row, null)
    row["response"] = confirm_evidence(row, a, d, null)
    # Reference data is opened only after magnetic features are built.
    labels, q, error = post_group(row, rec)
    if error:
        raise ValueError(error)
    row["diagnostic_labels"], row["q"] = labels, q
    row["partition"] = "development" if source_partition(rec) != "test" else "evaluation"
    row["split"] = source_partition(rec)
    row["state_validate"] = np.full(len(row["time"]), row["split"] == "validation")
    row["amplitude_uT"] = np.linalg.norm(row["field_uT"], axis=1)
    y = np.full(len(q), -1, int); direction = y.copy()
    quiet = np.isin(labels, ("pre_action_lift", "lift_residue_0to3s", "late_lift", "protocol_no_motion"))
    y[row["eligible"] & quiet & ~row["response"]] = 0
    residue = np.isin(labels, ("lift_residue_0to3s", "late_lift"))
    y[row["eligible"] & residue & row["response"]] = 3
    finite = np.isfinite(q).all(1)
    use = row["eligible"] & row["response"] & finite
    y[use & (q[:, 2] >= .4)] = 2
    weak = use & (q[:, 2] >= .02) & (q[:, 2] < .4)
    y[weak] = 1
    dz = np.r_[0., np.diff(q[:, 2])]
    continuous = np.r_[False, (np.diff(row["time"]) > 0) & (np.diff(row["time"]) <= 250_000_000)]
    weak &= continuous & np.r_[False, finite[:-1]]
    direction[weak & (dz > 1e-4)] = 0
    direction[weak & (dz < -1e-4)] = 1
    direction[weak & (np.abs(dz) <= 1e-4)] = 2
    direction[(y == 2) & continuous & (np.abs(dz) <= 1e-4)] = 2
    direction[y == 3] = 2
    row["context_y"], row["direction_y"] = y, direction
    return row


def binary_dataset(rows, variant, validate=False):
    x, y, w = dataset(rows, "direction", variant, validate, features)
    take = y < 2
    x, y, w = x[take], y[take], w[take]
    w *= len(w)/w.sum()
    return x, y, w


def score_prediction(row, pred):
    groups, events = metrics(row, pred, row["diagnostic_labels"], row["q"])
    whole = whole_actions({"time_ns": row["time"], "diagnostic_command_xyz_mm": row["q"],
                           "eligible": row["eligible"], **pred})
    return {"path": row["path"].relative_to(ROOT).as_posix(), "split": row["split"],
            "mode": row["mode"], "frames": len(row["time"]), "raw_rows": row["raw_rows"],
            "eligible_frames": int(row["eligible"].sum()), "groups": groups, "events": events,
            "whole_actions": whole, "rates": scalar_rates(groups)}


def aggregate(items):
    groups = {}
    for item in items:
        for name, values in item["groups"].items():
            dest = groups.setdefault(name, {k: np.zeros(len(v), int) if isinstance(v, list) else 0 for k, v in values.items()})
            for key, value in values.items():
                dest[key] += np.array(value) if isinstance(value, list) else value
    rates = scalar_rates(groups)
    macro = {k: float(np.mean([v["rates"][k] for v in items if k in v["rates"]]))
             for k in sorted({k for v in items for k in v["rates"]})}
    whole = whole_summary([{"events": v["whole_actions"]} for v in items])
    return {"trials": len(items), "rates": rates, "trial_macro": macro, "whole": whole}


def run_scores(row, scores, base, policy):
    context, direction, binary = scores
    fresh = row["state_x"][:, 1]*100. > base["fresh"]["source"]
    sp = PhasePolicy(**policy["strength"])
    if policy["evidence_rearm"]:
        strength, rejected = replay_evidence_rearm(row["time"], context, direction, row["response"],
            fresh, row["amplitude_uT"], row["eligible"], sp)
    else:
        strength, _, _, _ = replay_rearm_phase(row["time"], context, direction, row["response"],
            fresh, row["amplitude_uT"], row["eligible"], sp)
        rejected = np.zeros(len(strength), bool)
    dp = DirectionPolicy(**policy["direction"])
    phase, _, conflict = replay_factorized(row["time"], strength, direction, binary, row["response"],
        100.*row["state_x"][:, 1], base["change"]["source"]*policy["change_multiplier"], row["eligible"], dp)
    return {"strength": strength, "phase": phase, "response": row["response"],
            "amplitude_uT": row["amplitude_uT"], "reentry_rejected": rejected,
            "residual_conflict": conflict}


def policy_objective(v):
    r, m = v["rates"], v["trial_macro"]
    weak = .5*(m.get("weak_loading_responsive_frame_match", 0.) + m.get("weak_withdrawal_responsive_frame_match", 0.))
    return float(1.-weak + 2.*(1.-m.get("supported_state_preservation", 0.)) +
                 10.*r.get("safe_lift_loading_fraction", 0.) + 2.*r.get("safe_lift_any_action_fraction", 0.))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); out = args.out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.monotonic(); base, old_protocol = load_frozen()
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    recs = [r for r in inventory if r["device"] == "source" and r["date"] >= "S20260926_A" and r["initially_readable"]]
    # No target raw record is opened by this program.
    partition = [{"path": r["path"], "split": source_partition(r), "mode": r["mode"]} for r in recs]
    save_json(out / "source_partition.json", {"records": partition, "target_fit_rows": 0,
        "rule": "whole Trial based on source plan/number only; R133-E/guard/R136-B/R208-fast reserved for source test",
        "native_availability_clock": True, "null": "frozen R275 SOURCE covariance/limits retained to isolate representation/coverage"})
    rows = []
    for i, rec in enumerate(recs):
        rows.append(load_source(rec, base["null"]["source"]))
        if (i+1) % 10 == 0:
            print("source load", i+1, "/", len(recs), flush=True)
    validation = [r for r in rows if r["split"] == "validation"]
    trained, head_reports = {}, []
    for variant in ("scalar", "relative"):
        models = []
        for head, classes in (("context", 4), ("direction", 3), ("conditional", 2)):
            getter = binary_dataset if head == "conditional" else lambda rr, vv, validate=False: dataset(rr, head, vv, validate, features)
            tx, ty, tw = getter(rows, variant); vx, vy, vw = getter(rows, variant, validate=True)
            candidates = []
            for width in (0, 32):
                model = StateMLP(width, seed=276, n_classes=classes).fit(tx, ty, tw, (vx, vy, vw), epochs=120)
                file = out / f"{head}_{variant}_width{width}.npz"; model.save(file)
                report = {"head": head, "variant": variant, "width": width, "features": tx.shape[1],
                    "epoch": model.best_epoch, "validation_ce": model.best_loss,
                    "train_counts": np.bincount(ty, minlength=classes).tolist(),
                    "validation_counts": np.bincount(vy, minlength=classes).tolist(),
                    "history": model.history, "file": file.relative_to(ROOT).as_posix()}
                head_reports.append(report); candidates.append((model.best_loss, model, report))
                print("source train", variant, head, width, model.best_epoch, round(model.best_loss, 5), flush=True)
            _, model, selected = min(candidates, key=lambda x: x[0]); models.append((model, selected))
        trained[variant] = models
    baseline = aggregate([score_prediction(r, old_replay(r, base)) for r in validation])
    candidates = []
    for variant, models in trained.items():
        scores = [[m.predict(features(r, variant)) for m, _ in models] for r in validation]
        for weak in (.60, .75):
            for guard in (False, True):
                for multiplier in (1., 0.):
                    for tail in (0., .20):
                        policy = {"strength": asdict(PhasePolicy(weak_limit=weak)),
                            "direction": asdict(DirectionPolicy(conditional_limit=.85, activity_limit=.15,
                                residue_tail_s=tail, reuse_confirmed_direction=True)),
                            "change_multiplier": multiplier, "evidence_rearm": guard}
                        items = [score_prediction(r, run_scores(r, s, base, policy)) for r, s in zip(validation, scores)]
                        val = aggregate(items)
                        item = {"variant": variant, "policy": policy, "validation": val,
                                "objective": policy_objective(val)}
                        # These checks are unchanged by target behavior.
                        rates = val["rates"]
                        item["qualified"] = bool(rates.get("supported_state_preservation", 0.) >= .95 and
                            rates.get("safe_lift_loading_fraction", 1.) <= .001 and
                            rates.get("protocol_no_motion_any_action_fraction", 1.) <= .001)
                        candidates.append(item)
                        print("source policy", variant, weak, guard, multiplier, tail,
                              round(item["objective"], 5), item["qualified"], flush=True)
    qualified = [c for c in candidates if c["qualified"]]
    selected = min(qualified or candidates, key=lambda c: c["objective"])
    reports = {"schema": "R276-SourceConditionAdaptation-1", "heads": head_reports,
               "source_validation_baseline_R275": baseline, "candidates": candidates, "selected": selected,
               "selection_qualified": bool(qualified), "elapsed_s": time.monotonic()-start,
               "target_fit_rows": 0, "target_normalization_rows": 0, "target_selection_rows": 0,
               "semantics": "response phase only; nominal weak command window not physical contact, no XYZ regression"}
    save_json(out / "training_report.json", reports)
    selected_models = [v[1]["file"] for v in trained[selected["variant"]]]
    raw_files = [ROOT / r["path"] / name for r in recs for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    tracked = [ROOT/v for v in selected_models] + [out / "training_report.json", out / "source_partition.json",
        Path(__file__), ROOT / "training/response_phase_state_v4.py"]
    protocol = {"schema": "R276-FrozenSourceAdaptation-1", "frozen_before_target_replay": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "variant": selected["variant"],
        "models": selected_models, "selected_policy": selected["policy"], "source_validation_qualified": bool(qualified),
        "base_protocol": str((ROOT / 'analysis/r275_source_only_zero_shot_20260929/freeze.json').relative_to(ROOT)),
        "base_protocol_sha256": sha(ROOT / 'analysis/r275_source_only_zero_shot_20260929/freeze.json'),
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked],
        "source_input_manifest": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in raw_files],
        "qualification": "retrospective cross-condition validation; previously viewed target behavior does not become prospective blind",
        "no_shallow_contact_or_XYZ_truth": True, "no_future_or_template_input": True}
    save_json(out / "freeze.json", protocol)
    print("FROZEN R276", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
