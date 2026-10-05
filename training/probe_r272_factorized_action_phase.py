"""Separate conditional action direction from frozen R271 strength/context.

No shallow XYZ training. All model/policy selection uses earlier development
validation. R208 is a retrospective same-stream check, not a blind test.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import numpy as np

from training.probe_r269_hierarchical_response_state import (
    ROOT, load_rows, prepare_hierarchy, head_features, StateMLP, stage_summary)
from training.factorized_response_phase import DirectionPolicy, replay_factorized
from training.audit_r271_response_phase import timing, plot_trace

FROZEN = ROOT / "analysis/r271_rearm_phase_continuity_20260929"
OUT = ROOT / "analysis/r272_factorized_action_phase_20260929"


def dataset(rows, validate=False):
    xs, ys, ws = [], [], []
    for row in rows:
        if row["partition"] != "development":
            continue
        part = row["state_validate"] if validate else ~row["state_validate"]
        for cls in (0, 1):
            ids = np.flatnonzero(row["eligible"] & part & (row["direction_y"] == cls))
            if not len(ids):
                continue
            ids = ids[::max(1, int(np.ceil(len(ids) / 500)))]
            xs.append(head_features(row, "scalar")[ids])
            ys.append(np.full(len(ids), cls))
            ws.append(np.full(len(ids), 1. / len(ids)))
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in (0, 1):
        if not np.any(y == cls):
            raise ValueError("both conditional phase classes required")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w) / w.sum()
    return x, y, w


def metrics(rows, frozen, conditional, limits, policy):
    counts = {k: [0, 0] for k in ("loading", "withdrawal", "residue", "quiet", "supported_hold")}
    for row, a, binary in zip(rows, frozen, conditional):
        if row["partition"] != "development":
            continue
        phase, _, _ = replay_factorized(row["time"], a["state"], a["direction_scores"], binary,
                                       row["response"], row["state_x"][:, 1] * 100.,
                                       limits[row["device"]], row["eligible"], policy)
        use = row["eligible"] & row["state_validate"]
        for name, mask, matched in (
            ("loading", row["direction_y"] == 0, phase == 2),
            ("withdrawal", row["direction_y"] == 1, phase == 4),
            ("residue", row["context_y"] == 3, np.isin(phase, (2, 4))),
            ("quiet", row["context_y"] == 0, np.isin(phase, (2, 3, 4))),
            ("supported_hold", (row["context_y"] == 2) & (row["direction_y"] == 2), np.isin(phase, (2, 4)))):
            m = use & mask
            counts[name][0] += int(np.sum(m & matched)); counts[name][1] += int(m.sum())
    rates = {k: a / max(b, 1) for k, (a, b) in counts.items()}
    qualified = rates["residue"] <= .01 and rates["quiet"] <= .001 and rates["supported_hold"] <= .01
    return {"qualified_direction_validation": qualified,
            "objective": 2. - rates["loading"] - rates["withdrawal"] + 4.*rates["residue"] +
            4.*rates["supported_hold"] + rates["quiet"], "rates": rates, "counts_numerator_denominator": counts}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = load_rows(); prepare_hierarchy(rows)
    OUT.mkdir(parents=True)
    train, validation = dataset(rows), dataset(rows, validate=True)
    models, records = [], []
    for width in (0, 32):
        model = StateMLP(width, seed=272, n_classes=2)
        model.fit(*train, validation=validation, epochs=120)
        model.save(OUT / f"conditional_direction_width{width}.npz")
        records.append({"width": width, "validation_loss": model.best_loss, "best_epoch": model.best_epoch,
                        "training_class_frames": np.bincount(train[1], minlength=2).tolist(),
                        "validation_class_frames": np.bincount(validation[1], minlength=2).tolist(),
                        "history": model.history})
        models.append(model)
        print("conditional phase", width, model.best_epoch, model.best_loss, flush=True)
    model = min(models, key=lambda m: m.best_loss)
    limits = {}
    for device in ("source", "target"):
        values = []
        for row in rows:
            if row["partition"] != "development" or row["device"] != device:
                continue
            safe = np.isin(row["labels"], ("unloaded_trial", "precontact_lift", "lift_residual", "postcontact_lift"))
            ids = np.flatnonzero(row["eligible"] & ~row["state_validate"] & safe)[::4]
            values.extend(np.abs(100. * row["state_x"][ids, 1]).tolist())
        limits[device] = max(1., float(np.percentile(values, 95.)))
    frozen, scores = [], []
    for row in rows:
        trial = int(row["path"].name.split("_")[1])
        with np.load(FROZEN / f"trial_{trial:06d}_state.npz", allow_pickle=False) as a:
            record = {k: a[k].copy() for k in a.files}
        np.testing.assert_array_equal(record["time_ns"], row["time"])
        frozen.append(record)
        scores.append(model.predict(head_features(row, "scalar")))
    candidates = [(DirectionPolicy(conditional_limit=c, activity_limit=a, residue_tail_s=t), None)
                  for c in (.85, .95) for a in (.15, .35, .60) for t in (0., .10, .20)]
    candidates = [(p, metrics(rows, frozen, scores, limits, p)) for p, _ in candidates]
    qualified = [v for v in candidates if v[1]["qualified_direction_validation"]]
    policy, check = min(qualified or candidates, key=lambda v: v[1]["objective"])
    pooled, trials = {}, []
    selected_plot = {}
    for row, a, binary in zip(rows, frozen, scores):
        phase, proposed, conflicts = replay_factorized(row["time"], a["state"], a["direction_scores"], binary,
            row["response"], row["state_x"][:, 1] * 100., limits[row["device"]], row["eligible"], policy)
        n = int(row["path"].name.split("_")[1])
        np.savez_compressed(OUT / f"trial_{n:06d}_phase.npz", time_ns=row["time"], strength_state=a["state"],
                            phase=phase, proposal=proposed, residual_conflict=conflicts,
                            conditional_direction_scores=binary, response=row["response"], eligible=row["eligible"],
                            physical_contact_truth=np.full(len(phase), -1, np.int8))
        groups = stage_summary(row, phase)
        dest = pooled.setdefault(row["partition"] + "_" + row["device"], {})
        for label, value in groups.items():
            acc = dest.setdefault(label, {"n": 0, "response": 0, "states_all": [0]*7, "states_response": [0]*7})
            acc["n"] += value["n"]; acc["response"] += value["response"]
            for k in ("states_all", "states_response"):
                acc[k] = (np.array(acc[k]) + np.array(value[k])).tolist()
        events = timing(row, phase)
        trials.append({"trial": row["path"].name, "device": row["device"], "partition": row["partition"],
                       "mode": row["mode"], "groups": groups, "events": events,
                       "residual_conflict_frames": int(np.sum(conflicts & row["eligible"])), "raw_sha256": row["raw_sha256"]})
        if row["partition"] == "evaluation" and np.any(row["labels"] == "shallow_descent"):
            selected_plot.setdefault(row["device"], (row, a, phase))
    report = {"schema": "MagTouch-R272-FactorizedPhase-1", "status": "offline shadow phase; no gate/XYZ/hardware/UI/paper change",
              "frozen_strength_input": str(FROZEN.relative_to(ROOT)), "training": records, "selected_width": model.width,
              "policy": asdict(policy), "earlier_validation": check,
              "activity_change_limits_uT": limits, "activity_calibration": "95% absolute 0.1s amplitude change in earlier training safe windows",
              "policy_candidates": [{"policy": asdict(p), "validation": v} for p, v in candidates],
              "pooled": pooled, "trials": trials,
              "semantics": "Strength/context and conditional action direction are separate. Conditional binary scores are not contact probabilities. Residual conflict stays explicit. Unknown physical contact on all frames. Target earlier data used; not zero-shot. R208 already viewed; retrospective only."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("direction policy", asdict(policy), check, "change limits", limits, flush=True)
    for key in ("evaluation_source", "evaluation_target"):
        print(key, {k: pooled[key].get(k) for k in ("shallow_descent", "shallow_ascent", "lift_residual", "postcontact_lift", "deep_still")}, flush=True)


if __name__ == "__main__":
    main()
