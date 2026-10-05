"""Source-only action-window versus quiet/residual evidence veto.

The binary annotations describe protocol-relative magnetic response, not
actual contact. Do not modify R276 strength, XYZ, or R278 frozen artifacts.
All model and confidence selection uses source development alone.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r279_gain_generalization import ROOT, INVENTORY, gain_row, features, VALIDATION_GAINS
from training.probe_r278_independent_direction import OUT as PRIOR_OUT, load_source, supplement
from training.audit_r278_independent_direction import verify_freeze as prior_freeze
from training.probe_r276_condition_adaptation import source_partition, aggregate, score_prediction, policy_objective
from training.audit_r276_condition_adaptation import verify_freeze as strength_freeze, infer as strength_infer
from training.audit_r275_source_only_response import load_frozen
from training.audit_r274_frozen_phase_expansion import RAW_FILES, save_json
from training.probe_r245_joint_contact_state import StateMLP
from training.verify_r273_response_runtime import load_model
from training.build_trajectory_dataset_v1 import sha

OUT = ROOT/"analysis/r280_source_action_context_20260929"
TRAIN_GAINS = (.5, .75, 1., 1.5, 2.)


def context_labels(row):
    labels, q = row["diagnostic_labels"], row["q"]
    y = np.full(len(q), -1, int)
    safe = np.isin(labels, ("pre_action_lift", "lift_residue_0to3s", "late_lift", "protocol_no_motion"))
    y[row["eligible"] & safe] = 0
    action = row["eligible"] & row["response"] & np.isfinite(q).all(1) & (q[:, 2] >= .02)
    y[action] = 1
    return y


def binary_dataset(rows, variant, validate):
    xs, ys, ws = [], [], []
    for row in rows:
        if (row["split"] == "validation") != validate: continue
        value = features(row, variant)
        for cls in range(2):
            ids = np.flatnonzero(row["eligible"] & (row["action_context_y"] == cls))
            if not len(ids): continue
            ids = ids[::max(1, int(np.ceil(len(ids)/500)))]
            xs.append(value[ids]); ys.append(np.full(len(ids), cls)); ws.append(np.full(len(ids), 1./len(ids)))
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(2):
        if not np.any(y == cls): raise ValueError("missing source action-context class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    return x, y, w


def veto_context(row, baseline, scores, limit, confirm_s):
    scores = np.asarray(scores, float); time = row["time"]; eligible = row["eligible"]; n = len(time)
    if scores.shape != (n, 2) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(np.abs(scores.sum(1)-1)>1e-5):
        raise ValueError("aligned normalized two-class response-context scores required")
    safe = eligible & (scores[:, 0] >= limit)
    reset = np.r_[True, ~safe[1:] | ~safe[:-1] | ~eligible[1:] | ~eligible[:-1] |
        (np.diff(time) <= 0) | (np.diff(time) > 250_000_000)]
    starts = np.maximum.accumulate(np.where(reset, np.arange(n), 0)) if n else np.zeros(0, int)
    confirmed = safe & (np.arange(n)-starts >= 1) & (time-time[starts] >= confirm_s*1e9)
    suppressed = confirmed & np.isin(baseline["phase"], (2, 3, 4))
    phase = baseline["phase"].copy()
    phase[suppressed] = np.where(row["response"][suppressed], 5, 1)
    phase[~eligible] = 0
    return dict(baseline, phase=phase, action_context_veto=suppressed, action_context_safe_score=scores[:, 0])


def supported_action_suppression(row, result):
    deep = row["eligible"] & row["response"] & np.isin(row["diagnostic_labels"], ("supported_motion", "supported_hold"))
    return int((deep & result["action_context_veto"]).sum())


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); start = time.monotonic()
    prior = prior_freeze(PRIOR_OUT); base, _ = load_frozen()
    strength_protocol = strength_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength_models = [load_model(ROOT/f) for f in prior["baseline_models"]]
    prior_model = load_model(ROOT/prior["selected"]["file"])
    declared = {"training_gains": TRAIN_GAINS, "validation_gains": VALIDATION_GAINS, "variants": ("relative", "ratio"),
        "safe_score_limits": (.9, .97, .995, .999), "confirm_s": (.03, .1, .3),
        "qualification": "ordinary source validation weak matching cannot fall; zero supported-action veto each gain; no added lift/control error; mean gain objective strictly lower"}
    save_json(OUT/"declared_search.json", declared)
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    records = [r for r in inventory if r["device"] == "source" and r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in records],
        "target_opened": 0, "source_test_opened": 0})
    raw_rows = []
    for i, rec in enumerate(records):
        raw_rows.append(load_source(rec, base["null"]["source"]))
        if (i+1) % 10 == 0: print("R280 source load", i+1, "/", len(records), flush=True)
    training, validation = [], {}
    for row in raw_rows:
        for gain in TRAIN_GAINS if row["split"] == "train" else VALIDATION_GAINS:
            value = gain_row(row, gain, base); value["action_context_y"] = context_labels(value)
            if row["split"] == "train": training.append(value)
            else: validation.setdefault(str(gain), []).append(value)
    old_scores = {}
    for gain, rows in validation.items():
        for row in rows:
            coarse = strength_infer(row, strength_models, strength_protocol, base)
            row["prior"] = supplement(row, coarse, prior_model.predict(features(row, "relative")), prior["selected"]["policy"])
        old_scores[gain] = aggregate([score_prediction(r, r["prior"]) for r in rows])
    development = training+[r for rows in validation.values() for r in rows]
    heads, candidates = [], []
    for variant in declared["variants"]:
        tx, ty, tw = binary_dataset(development, variant, False); vx, vy, vw = binary_dataset(development, variant, True)
        model = StateMLP(32, seed=280, n_classes=2).fit(tx, ty, tw, (vx, vy, vw), epochs=120)
        file = OUT/f"action_context_{variant}.npz"; model.save(file)
        heads.append({"variant": variant, "file": file.relative_to(ROOT).as_posix(), "input_features": tx.shape[1],
            "epoch": model.best_epoch, "validation_ce": model.best_loss, "history": model.history,
            "train_counts": np.bincount(ty, minlength=2).tolist(), "validation_counts": np.bincount(vy, minlength=2).tolist()})
        scores = {g: [model.predict(features(r, variant)) for r in rows] for g, rows in validation.items()}
        for limit in declared["safe_score_limits"]:
            for confirm in declared["confirm_s"]:
                results, deep_suppression = {}, {}
                for gain, rows in validation.items():
                    predictions = [veto_context(r, r["prior"], s, limit, confirm) for r, s in zip(rows, scores[gain])]
                    results[gain] = aggregate([score_prediction(r, p) for r, p in zip(rows, predictions)])
                    deep_suppression[gain] = sum(supported_action_suppression(r, p) for r, p in zip(rows, predictions))
                ordinary = results["1.0"]["rates"]; old = old_scores["1.0"]["rates"]
                weak_noninferior = all(ordinary.get(k, 0.) >= old.get(k, 0.)-1e-12 for k in
                    ("weak_loading_responsive_frame_match", "weak_withdrawal_responsive_frame_match"))
                deep_ok = not any(deep_suppression.values())
                objective = float(np.mean([policy_objective(r) for r in results.values()]))
                previous = float(np.mean([policy_objective(r) for r in old_scores.values()]))
                accepted = bool(weak_noninferior and deep_ok and objective < previous-1e-4)
                candidates.append({"variant": variant, "file": file.relative_to(ROOT).as_posix(), "limit": limit,
                    "confirm_s": confirm, "gain_validation": results, "supported_action_veto_frames": deep_suppression,
                    "ordinary_weak_noninferior": bool(weak_noninferior), "zero_deep_veto": bool(deep_ok),
                    "objective": objective, "previous_objective": previous, "accepted_improvement": accepted})
        print("R280 source model", variant, "epoch", model.best_epoch, "CE", round(model.best_loss, 6),
            "qualified", sum(c["accepted_improvement"] for c in candidates), flush=True)
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R280-SourceActionContext-1", "heads": heads,
        "candidates": candidates, "baseline_gain_validation": old_scores, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-start,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "semantics": "protocol-relative magnetic action response versus quiet/lift-memory; not physical contact"})
    tracked = [Path(__file__), OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json"]
    if selected: tracked.append(ROOT/selected["file"])
    inputs = [ROOT/r["path"]/name for r in records for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    save_json(OUT/"freeze.json", {"schema": "R280-FrozenSourceActionContext-1", "frozen_before_target_replay": True,
        "accepted_improvement": bool(accepted), "selected": selected,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "no_XYZ_arming": True, "strength_frozen_R276": True,
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked],
        "source_input_manifest": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in inputs]})
    print("FROZEN R280 accepted", bool(accepted), "selected", None if not selected else
        {k: selected[k] for k in ("variant", "limit", "confirm_s", "objective")}, flush=True)


if __name__ == "__main__": main()
