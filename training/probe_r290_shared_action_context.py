"""SOURCE-only shared four-class magnetic action/context replacement.

Replace R287 direction + R288 residual head, rather than append another
expert. Frozen R282/R276 remain the base. Commands label SOURCE action
windows only; inference never reads commands, device, path or future data.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r288_ordered_action_context import (
    ROOT, INVENTORY, load_prior_args, prior_infer, protected_veto, load_source,
    source_partition, TRAIN_GAINS, VAL_GAINS, TAU_S, augmented_signal_row,
    predict_sequence, load_model, StateMLP, score_prediction, aggregate,
    policy_objective, RAW_FILES, save_json, sha)
from training.probe_r288_ordered_action_context import OUT as REFERENCE_OUT
from training.verify_r288_ordered_action_context import verify_freeze as reference_freeze
from training.probe_r278_independent_direction import action_annotations, features, supplement, qualifies
from training.probe_r280_action_context import context_labels
from training.causal_response_sequence import causal_taps
from training.response_spatial_features import spatial_features
from training.train_r275_source_only_response import assert_source

OUT = ROOT/"analysis/r290_source_shared_action_context_20260929"
SPECS = ({"name": "shared_global", "variant": "global", "width": 32},
         {"name": "shared_sensor", "variant": "sensor", "width": 32})


def joint_labels(row):
    """loading, withdrawal, stationary action, protocol quiet/residual."""
    y = action_annotations(row, "all_depth_action")
    y[context_labels(row) == 0] = 3
    return y


def current_features(row, variant):
    global_x = features(row, "relative")
    if variant == "global": return global_x
    if variant != "sensor": raise ValueError("declared shared representation required")
    local = spatial_features(row["time"], row["field_uT"], row["eligible"], "local_relative")
    return np.column_stack((global_x, local[:, 16:]))


def shared_features(row, variant, ids=None):
    value = current_features(row, variant)
    taps = causal_taps(row["time"], row["eligible"], 16, .1)
    if ids is not None: taps = taps[np.asarray(ids, int)]
    elapsed = np.zeros(taps.shape)
    elapsed[:, 1:] = np.diff(row["time"][taps], axis=1)/100_000_000.
    return np.concatenate((value[taps], np.clip(elapsed, 0, 5)[..., None]), 2).reshape(len(taps), -1)


def predict_shared(row, model, spec):
    # Compute magnetic summaries once, not once per 512-frame batch.
    value = current_features(row, spec["variant"])
    taps = causal_taps(row["time"], row["eligible"], 16, .1)
    result = []
    for start in range(0, len(taps), 512):
        ids = taps[start:start+512]; elapsed = np.zeros(ids.shape)
        elapsed[:, 1:] = np.diff(row["time"][ids], axis=1)/100_000_000.
        x = np.concatenate((value[ids], np.clip(elapsed, 0, 5)[..., None]), 2)
        result.append(model.predict(x.reshape(len(x), -1)))
    return np.vstack(result)


def replace_heads(row, base, scores, policy):
    scores = np.asarray(scores, float)
    if scores.shape != (len(row["time"]), 4) or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(abs(scores.sum(1)-1) > 1e-5):
        raise ValueError("aligned normalized four-class shared scores required")
    # Do not renormalize away the quiet class to fabricate action confidence.
    direction = np.column_stack((scores[:, :2], scores[:, 2]+scores[:, 3]))
    proposed = supplement(row, base, direction, policy["direction"])
    added = proposed["direction_supplemented"].copy()
    context = np.column_stack((scores[:, 3], 1.-scores[:, 3]))
    result = protected_veto(row, proposed, context, policy["context_limit"], policy["context_confirm_s"])
    result["shared_direction_added"] = added
    result["shared_context_veto"] = result["ordered_context_veto"]
    return result


def common_ids(row, views, cap=150, hard_cap=150):
    if row["device"] != "source" or row["split"] not in ("train", "validation"):
        raise ValueError("only SOURCE development can enter shared fitting")
    if cap <= 0 or hard_cap <= 0: raise ValueError("positive sample caps required")
    labels = np.vstack([joint_labels(v) for v in views])
    stable = np.all(labels == labels[:1], 0) & np.logical_and.reduce([v["eligible"] for v in views])
    answer = []
    for cls in range(4):
        ids = np.flatnonzero(stable & (labels[0] == cls))
        ids = ids[::max(1, int(np.ceil(len(ids)/cap)))]
        if cls == 3 and row["split"] == "train":
            hard = np.flatnonzero(stable & (labels[0] == 3) & np.isin(row["before_context"]["phase"], (2, 3, 4)))
            hard = hard[::max(1, int(np.ceil(len(hard)/hard_cap)))]
            ids = np.unique(np.r_[ids, hard])
        if len(ids): answer.append((cls, ids))
    return answer


def shared_dataset(rows, variant, null, validate=False):
    assert_source(rows)
    if any(r["split"] not in ("train", "validation") for r in rows): raise ValueError("reserved SOURCE test excluded")
    split = "validation" if validate else "train"; gains = VAL_GAINS if validate else TRAIN_GAINS
    xs, ys, ws, manifest = [], [], [], []
    for row in rows:
        if row["split"] != split: continue
        views = [row if g == 1. else augmented_signal_row(row, g, null, TAU_S) for g in gains]
        selected = common_ids(row, views)
        for g, view in zip(gains, views):
            for cls, ids in selected:
                xs.append(shared_features(view, variant, ids).astype(np.float32))
                ys.append(np.full(len(ids), cls)); ws.append(np.full(len(ids), 1./len(ids)/len(gains)))
                manifest.append({"path": str(row["path"]), "gain": g, "class": cls, "n": len(ids),
                    "indices_sha256": hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest()})
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(4):
        if not np.any(y == cls): raise ValueError("missing SOURCE shared class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    return x, y, w, manifest


def preservation(row, old, new):
    labels = row["diagnostic_labels"]; eligible = row["eligible"]
    weak_l = eligible & row["response"] & (labels == "weak_loading") & (old["phase"] == 2)
    weak_w = eligible & row["response"] & (labels == "weak_withdrawal") & (old["phase"] == 4)
    weak_lost = int((weak_l & (new["phase"] != 2)).sum()+(weak_w & (new["phase"] != 4)).sum())
    deep = eligible & row["response"] & np.isin(labels, ("supported_motion", "supported_hold"))
    deep_lost = int((deep & np.isin(old["phase"], (2, 3, 4)) & ~np.isin(new["phase"], (2, 3, 4))).sum())
    oi, ni = score_prediction(row, old), score_prediction(row, new)
    prior_events = {(e["group"], e["start_time_ns"]): e for e in oi["events"]}
    lost_events = sum(prior_events[(e["group"], e["start_time_ns"])]["hit"] and not e["hit"] for e in ni["events"])
    safety = qualifies([ni], [oi]); same = np.array_equal(old["strength"], new["strength"])
    return {"weak_correct_frames_lost": weak_lost, "supported_action_frames_lost": deep_lost,
        "weak_episode_hits_lost": int(lost_events), "safe_nonincrease": bool(safety), "strength_unchanged": bool(same),
        "qualified": bool(not weak_lost and not deep_lost and not lost_events and safety and same)}


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"specs": SPECS, "seed": 290, "epochs": 80, "steps": 16, "step_s": .1,
        "effective_max_history_s": 6.5, "train_gains": TRAIN_GAINS, "validation_gains": VAL_GAINS,
        "tau_s": TAU_S, "cap_per_Trial_class": 150, "SOURCE_train_hard_negative_cap": 150,
        "direction_limits": (.9, .97, .995), "direction_confirm_s": .03,
        "context_limits": (.97, .995), "context_confirm_s": .03, "require_signed_change": False,
        "comparison": "identical SOURCE indices/class weights/update budget; same hidden width but more sensor parameters; not capacity-matched",
        "qualification": "every native SOURCE train/validation and validation gain Trial: no lost correct weak frames/episodes or nominal supported actions, unchanged strength, no added lift/control; native and mean-gain objective strict improvement vs complete R288",
        "replacement": "R282 base + one joint head replaces R287 direction and R288 context; no expert stacking",
        "semantics": "magnetic protocol-relative actions, not actual shallow contact/XYZ", "no_target_selection": True}
    save_json(OUT/"declared_search.json", declared)
    p = reference_freeze(REFERENCE_OUT); context = load_model(ROOT/p["selected"]["file"])
    args, base = load_prior_args()
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and
            r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "target_opened": 0, "source_test_opened": 0, "whole_Trial": True})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    def priors(row):
        row["base"], row["before_context"] = prior_infer(row, *args)
        s = p["selected"]
        row["reference"] = protected_veto(row, row["before_context"], predict_sequence(row, context, s["spec"]), s["limit"], s["confirm_s"])
        return row
    rows = []
    for i, rec in enumerate(recs):
        rows.append(priors(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R290 SOURCE shared load", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else priors(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
                          for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    old_reports = {g: aggregate([score_prediction(r, r["reference"]) for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([policy_objective(v) for v in old_reports.values()]))
    heads, candidates, reference_ids = [], [], None
    for spec in SPECS:
        tx, ty, tw, train_ids = shared_dataset(rows, spec["variant"], base["null"]["source"])
        vx, vy, vw, val_ids = shared_dataset(rows, spec["variant"], base["null"]["source"], True)
        ids = {"train": train_ids, "validation": val_ids}
        if reference_ids is None: reference_ids = ids
        else: assert ids == reference_ids, "representation comparison sampling drift"
        save_json(OUT/f"samples_{spec['name']}.json", dict(ids, train_samples=len(ty), validation_samples=len(vy)))
        print("R290 SOURCE train", spec["name"], tx.shape, vx.shape, flush=True)
        model = StateMLP(spec["width"], seed=290, n_classes=4).fit(tx, ty, tw, (vx, vy, vw), epochs=80)
        file = OUT/f"joint_{spec['name']}.npz"; model.save(file)
        heads.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "epoch": model.best_epoch,
            "validation_ce": model.best_loss, "history": model.history, "parameters": sum(int(v.size) for v in model.params)})
        del tx, ty, tw, vx, vy, vw
        native_scores = [predict_shared(r, model, spec) for r in rows]
        gain_scores = {g: [predict_shared(r, model, spec) for r in vr] for g, vr in validation.items()}
        for dl in declared["direction_limits"]:
            for cl in declared["context_limits"]:
                policy = {"direction": {"score_limit": dl, "confirm_s": .03, "permit_residue": True, "require_signed_change": False},
                    "context_limit": cl, "context_confirm_s": .03}
                native = [replace_heads(r, r["base"], s, policy) for r, s in zip(rows, native_scores)]
                nc = [dict(preservation(r, r["reference"], n), path=str(r["path"])) for r, n in zip(rows, native)]
                reports, gc = {}, []
                for g, vr in validation.items():
                    predictions = [replace_heads(r, r["base"], s, policy) for r, s in zip(vr, gain_scores[g])]
                    reports[g] = aggregate([score_prediction(r, n) for r, n in zip(vr, predictions)])
                    gc += [dict(preservation(r, r["reference"], n), path=str(r["path"]), gain=g) for r, n in zip(vr, predictions)]
                obj = float(np.mean([policy_objective(v) for v in reports.values()]))
                safe = all(c["qualified"] for c in nc+gc)
                native_ok = policy_objective(reports["1.0"]) < policy_objective(old_reports["1.0"])-1e-6
                candidates.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "policy": policy,
                    "gain_validation": reports, "objective": obj, "previous_objective": previous,
                    "qualified_protection": bool(safe), "native_strict_improvement": bool(native_ok),
                    "native_failures": [c for c in nc if not c["qualified"]], "gain_failures": [c for c in gc if not c["qualified"]],
                    "accepted_improvement": bool(safe and native_ok and obj < previous-1e-4)})
        print("R290 SOURCE model", spec["name"], "CE", model.best_loss, "accepted", sum(c["accepted_improvement"] for c in candidates), flush=True)
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R290-SharedActionContext-1", "heads": heads, "candidates": candidates,
        "baseline_gain_validation": old_reports, "selected": selected, "accepted_improvement": bool(accepted),
        "elapsed_s": time.monotonic()-started, "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/response_spatial_features.py", OUT/"declared_search.json",
        OUT/"source_partition.json", OUT/"training_report.json"]+[ROOT/h["file"] for h in heads]+list(OUT.glob("samples_*.json"))
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R290-FrozenSharedActionContext-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (REFERENCE_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(REFERENCE_OUT/"freeze.json"),
        "no_XYZ_arming": True, "strength_frozen_R276": True, "target_fit_rows": 0, "target_selection_rows": 0,
        "source_test_selection_rows": 0, "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked],
        "source_input_manifest": inputs})
    print("FROZEN R290 accepted", bool(accepted), "elapsed", round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__": main()
