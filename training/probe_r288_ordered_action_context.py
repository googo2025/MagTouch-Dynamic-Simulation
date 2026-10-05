"""SOURCE-only ordered action context with protected supported responses.

Hard residual negatives are mined only from SOURCE training Trials. Labels
describe commanded action windows, not certain shallow contact or actual
tip depth. Neither context scores nor this veto arm XYZ. Yellow and reserved
SOURCE tests cannot enter fitting, threshold search or epoch selection.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r287_matched_signal_coverage import (
    ROOT, INVENTORY, TRAIN_GAINS, VAL_GAINS, TAU_S, source_partition, load_source,
    augmented_signal_row, sequence_features, predict_sequence, StateMLP,
    load_model, load_temporal, temporal_freeze, direction_freeze, strength_freeze,
    load_frozen, score_prediction, aggregate, policy_objective, RAW_FILES, save_json, sha)
from training.probe_r287_matched_signal_coverage import OUT as PRIOR_OUT
from training.verify_r287_matched_coverage import verify_freeze as prior_freeze
from training.audit_r286_signal_sequence import infer as prior_infer
from training.probe_r280_action_context import context_labels
from training.train_r275_source_only_response import assert_source

OUT = ROOT/"analysis/r288_source_ordered_action_context_20260929"
SPECS = ({"name": "ordered_ratio", "architecture": "MLP", "variant": "ratio", "width": 32},
         {"name": "ordered_relative", "architecture": "MLP", "variant": "relative", "width": 32})


def protected_veto(row, prior, scores, limit=.995, confirm_s=.1):
    """Causal rejection only; supported strength 6 can never be vetoed."""
    scores = np.asarray(scores, float)
    t, eligible = row["time"], row["eligible"]
    n = len(t)
    if scores.shape != (n, 2) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(abs(scores.sum(1)-1) > 1e-5):
        raise ValueError("aligned normalized context scores required")
    if not 0 < limit < 1 or not np.isfinite(confirm_s) or confirm_s < 0:
        raise ValueError("valid context threshold and confirmation required")
    safe = eligible & (scores[:, 0] >= limit) & (prior["strength"] != 6)
    if n:
        reset = np.r_[True, ~safe[1:] | ~safe[:-1] | ~eligible[1:] | ~eligible[:-1] |
            (np.diff(t) <= 0) | (np.diff(t) > 250_000_000)]
        starts = np.maximum.accumulate(np.where(reset, np.arange(n), 0))
        confirmed = safe & (np.arange(n)-starts >= 1) & (t-t[starts] >= confirm_s*1e9)
    else:
        confirmed = np.zeros(0, bool)
    suppressed = confirmed & np.isin(prior["phase"], (2, 3, 4))
    phase = prior["phase"].copy()
    phase[suppressed] = np.where(row["response"][suppressed], 5, 1)
    phase[~eligible] = 0
    return dict(prior, phase=phase, ordered_context_veto=suppressed,
                ordered_context_residual_score=scores[:, 0])


def context_ids(row, cap=300, hard_cap=300):
    """Fixed regular samples plus SOURCE-train-only magnetic hard negatives."""
    if row["device"] != "source" or row["split"] not in ("train", "validation"):
        raise ValueError("context fitting limited to SOURCE development")
    if min(cap, hard_cap) <= 0: raise ValueError("positive sampling caps required")
    labels = context_labels(row)
    answer = []
    for cls in range(2):
        ids = np.flatnonzero(row["eligible"] & (labels == cls))
        ids = ids[::max(1, int(np.ceil(len(ids)/cap)))]
        if row["split"] == "train" and cls == 0:
            hard = np.flatnonzero(row["eligible"] & (labels == 0) & np.isin(row["prior"]["phase"], (2, 3, 4)))
            hard = hard[::max(1, int(np.ceil(len(hard)/hard_cap)))]
            ids = np.unique(np.r_[ids, hard])
        if len(ids): answer.append((cls, ids))
    return answer


def context_dataset(rows, variant, validate=False):
    assert_source(rows)
    if any(r["split"] not in ("train", "validation") for r in rows):
        raise ValueError("reserved SOURCE tests excluded")
    xs, ys, ws, manifest = [], [], [], []
    split = "validation" if validate else "train"
    for row in rows:
        if row["split"] != split: continue
        for cls, ids in context_ids(row):
            xs.append(sequence_features(row, variant, 16, .1, ids).reshape(len(ids), -1))
            ys.append(np.full(len(ids), cls)); ws.append(np.full(len(ids), 1./len(ids)))
            manifest.append({"path": str(row["path"]), "gain": row["gain"], "class": cls, "n": len(ids),
                "indices_sha256": __import__("hashlib").sha256(ids.astype("<i8").tobytes()).hexdigest()})
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(2):
        if not np.any(y == cls): raise ValueError("missing SOURCE context class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    return x, y, w, manifest


def protection_check(row, prior, new):
    """Diagnostic eligibility only; q and action labels never enter veto."""
    same_strength = np.array_equal(prior["strength"], new["strength"])
    labels = row["diagnostic_labels"]
    deep = row["eligible"] & row["response"] & np.isin(labels, ("supported_motion", "supported_hold"))
    old, result = score_prediction(row, prior), score_prediction(row, new)
    weak_ok = all(result["rates"].get(k, 0.) >= old["rates"].get(k, 0.)-1e-12 for k in
                  ("weak_loading_responsive_frame_match", "weak_withdrawal_responsive_frame_match"))
    old_events = {(e["group"], e["start_time_ns"]): e for e in old["events"]}
    event_ok = all(not old_events[(e["group"], e["start_time_ns"])]["hit"] or e["hit"] for e in result["events"])
    count = int((deep & new["ordered_context_veto"]).sum())
    return {"strength_unchanged": bool(same_strength), "weak_frame_noninferior": bool(weak_ok),
            "weak_episodes_preserved": bool(event_ok), "supported_action_veto_frames": count,
            "qualified": bool(same_strength and weak_ok and event_ok and count == 0)}


def load_prior_args():
    p = prior_freeze(PRIOR_OUT)
    prior = temporal_freeze(ROOT/Path(p["base_protocol"]).parent)
    direction = direction_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength = strength_freeze(ROOT/Path(direction["base_protocol"]).parent)
    base, _ = load_frozen()
    args = (load_model(ROOT/p["selected"]["file"]), p, load_temporal(prior["selected"]), prior,
            direction, load_model(ROOT/direction["selected"]["file"]), strength,
            [load_model(ROOT/f) for f in direction["baseline_models"]], base)
    return args, base


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"specs": SPECS, "seed": 288, "epochs": 80, "steps": 16, "step_s": .1,
        "train_gains": TRAIN_GAINS, "validation_gains": VAL_GAINS, "tau_s": TAU_S,
        "safe_limits": (.9, .97, .995), "confirm_s": (.03, .1, .3),
        "regular_cap_per_Trial_gain_class": 300, "extra_negative_cap_SOURCE_train_only": 300,
        "protected_strength": 6, "batch": 256, "learning_rate": .001, "weight_decay": .001,
        "qualification": "every native SOURCE train/validation and validation-gain Trial: strength unchanged, zero supported action veto, no lost weak frame matches or episode hits; native validation and mean-gain objective strict improvement",
        "semantics": "protocol-relative action window vs magnetic residual; not shallow contact truth or XYZ",
        "no_target_retained_SOURCE_or_old_tip_selection": True}
    save_json(OUT/"declared_search.json", declared)
    args, base = load_prior_args()
    recs = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    recs = [r for r in recs if r["device"] == "source" and r["date"] >= "S20260926_A" and
            r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "target_opened": 0, "source_test_opened": 0, "whole_Trial": True})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        row = load_source(rec, base["null"]["source"]); _, row["prior"] = prior_infer(row, *args)
        row["gain"] = 1.; rows.append(row)
        if (i+1)%10 == 0: print("R288 SOURCE context", i+1, "/", len(recs), flush=True)
    development = []
    for row in rows:
        for gain in TRAIN_GAINS if row["split"] == "train" else VAL_GAINS:
            value = row if gain == 1. else augmented_signal_row(row, gain, base["null"]["source"], TAU_S)
            value["gain"] = gain
            if gain != 1.: _, value["prior"] = prior_infer(value, *args)
            development.append(value)
    validation = {str(g): [r for r in development if r["split"] == "validation" and r["gain"] == g] for g in VAL_GAINS}
    baseline = {g: aggregate([score_prediction(r, r["prior"]) for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([policy_objective(v) for v in baseline.values()]))
    heads, candidates = [], []
    for spec in SPECS:
        tx, ty, tw, train_ids = context_dataset(development, spec["variant"])
        vx, vy, vw, val_ids = context_dataset(development, spec["variant"], True)
        save_json(OUT/f"samples_{spec['name']}.json", {"train": train_ids, "validation": val_ids,
            "train_samples": len(ty), "validation_samples": len(vy), "hard_negative_mining_SOURCE_train_only": True})
        print("R288 SOURCE train", spec["name"], tx.shape, vx.shape, flush=True)
        model = StateMLP(32, seed=288, n_classes=2).fit(tx, ty, tw, (vx, vy, vw), epochs=80)
        file = OUT/f"context_{spec['name']}.npz"; model.save(file)
        heads.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "epoch": model.best_epoch,
            "validation_ce": model.best_loss, "history": model.history, "parameters": sum(int(p.size) for p in model.params)})
        native_scores = [predict_sequence(r, model, spec) for r in rows]
        gain_scores = {g: [predict_sequence(r, model, spec) for r in vr] for g, vr in validation.items()}
        for limit in declared["safe_limits"]:
            for confirm in declared["confirm_s"]:
                native_checks = [protection_check(r, r["prior"], protected_veto(r, r["prior"], s, limit, confirm))
                                 for r, s in zip(rows, native_scores)]
                gain_reports, gain_checks = {}, []
                for g, vr in validation.items():
                    predictions = [protected_veto(r, r["prior"], s, limit, confirm) for r, s in zip(vr, gain_scores[g])]
                    gain_reports[g] = aggregate([score_prediction(r, p) for r, p in zip(vr, predictions)])
                    gain_checks += [dict(protection_check(r, r["prior"], p), path=str(r["path"]), gain=g) for r, p in zip(vr, predictions)]
                obj = float(np.mean([policy_objective(v) for v in gain_reports.values()]))
                safe = all(c["qualified"] for c in native_checks+gain_checks)
                strict_native = policy_objective(gain_reports["1.0"]) < policy_objective(baseline["1.0"])-1e-6
                candidates.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "limit": limit,
                    "confirm_s": confirm, "gain_validation": gain_reports, "objective": obj,
                    "previous_objective": previous, "qualified_protection": bool(safe), "native_strict_improvement": bool(strict_native),
                    "native_failures": [dict(c, path=str(r["path"])) for c, r in zip(native_checks, rows) if not c["qualified"]],
                    "gain_failures": [c for c in gain_checks if not c["qualified"]],
                    "accepted_improvement": bool(safe and strict_native and obj < previous-1e-4)})
        print("R288 SOURCE model", spec["name"], "epoch", model.best_epoch, "CE", round(model.best_loss, 6),
              "accepted", sum(c["accepted_improvement"] for c in candidates), flush=True)
        del tx, ty, tw, vx, vy, vw
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R288-SourceOrderedActionContext-1", "heads": heads,
        "candidates": candidates, "baseline_gain_validation": baseline, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/causal_response_sequence.py", OUT/"declared_search.json",
        OUT/"source_partition.json", OUT/"training_report.json"] + [ROOT/h["file"] for h in heads] + list(OUT.glob("samples_*.json"))
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R288-FrozenOrderedContext-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "steps": 16, "step_s": .1, "no_XYZ_arming": True, "strength_frozen_R276": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked], "source_input_manifest": inputs})
    print("FROZEN R288 accepted", bool(accepted), "selected", None if not selected else
          {k: selected[k] for k in ("spec", "limit", "confirm_s", "objective")}, flush=True)


if __name__ == "__main__": main()
