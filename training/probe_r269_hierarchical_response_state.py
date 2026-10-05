"""Train separate response-context and motion-direction heads, offline only.

No raw data/model/report overwrites, physical shallow-contact assertions,
XYZ regression, or hardware/UI operations. Context/phase targets are command
grouping annotations, not physical contact or shallow depth truth.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import numpy as np

from training.audit_r226_shallow_history_gate import ROOT, _auc
from training.audit_r227_assisted_contact_gate import load_rows
from training.probe_r228_shallow_inverse import reference
from training.probe_r245_joint_contact_state import prepare, StateMLP
from training.audit_r263_magnetic_evidence_uncertain_depth import OUT as EVIDENCE
from training.response_phase_state import PhasePolicy, STATE_NAMES, replay_phase

OUT = ROOT / "analysis/r269_hierarchical_response_state_20260929"
CONTEXT = ("safe_quiet_response", "weak_action_response", "supported_action_response", "safely_lifted_residue")
DIRECTION = ("loading_action", "withdrawal_action", "neutral_or_lift_memory")
SCALAR_IDS = list(range(10)) + list(range(40, 43)) + list(range(58, 61))


def prepare_hierarchy(rows):
    prepare(rows)
    for row in rows:
        trial = int(row["path"].name.split("_")[1])
        with np.load(EVIDENCE / f"trial_{trial:06d}_evidence.npz", allow_pickle=False) as a:
            assert np.array_equal(a["time_ns"], row["time"])
            row["response"] = a["magnetic_response_evidence"].copy()
        y = np.full(len(row["time"]), -1, int)
        safe = np.isin(row["labels"], ("unloaded_trial", "precontact_lift", "lift_residual", "postcontact_lift"))
        y[row["eligible"] & safe & ~row["response"]] = 0
        y[row["eligible"] & row["response"] & np.isin(row["labels"], ("deep_contact", "deep_still"))] = 2
        y[row["eligible"] & row["response"] & np.isin(row["labels"], ("lift_residual", "postcontact_lift"))] = 3
        direction = np.full(len(y), -1, int)
        if row["mode"] != "none":
            q = reference(row)
            finite = np.isfinite(q).all(1)
            weak = row["eligible"] & row["response"] & finite & (q[:, 2] >= .02) & (q[:, 2] < .4)
            y[weak] = 1
            dz = np.zeros(len(q))
            previous_finite = np.r_[False, finite[:-1]]
            dz[1:] = q[1:, 2] - q[:-1, 2]
            continuous = np.r_[False, (np.diff(row["time"]) > 0) & (np.diff(row["time"]) <= 250_000_000)]
            candidate = weak & previous_finite & continuous
            direction[candidate & (dz > 1e-4)] = 0
            direction[candidate & (dz < -1e-4)] = 1
            direction[candidate & (np.abs(dz) <= 1e-4)] = 2
            # Pure XY/deep holds provide negative direction controls, not XYZ features.
            direction[(y == 2) & previous_finite & continuous & (np.abs(dz) <= 1e-4)] = 2
        direction[y == 3] = 2
        row["context_y"], row["direction_y"] = y, direction


def head_features(row, variant):
    return row["state_x"] if variant == "vector" else row["state_x"][:, SCALAR_IDS]


def dataset(rows, head, variant, validate=False, feature_getter=head_features):
    xs, ys, ws = [], [], []
    classes = 4 if head == "context" else 3
    for row in rows:
        if row["partition"] != "development":
            continue
        part = row["state_validate"] if validate else ~row["state_validate"]
        for cls in range(classes):
            ids = np.flatnonzero(row["eligible"] & part & (row[head + "_y"] == cls))
            if not len(ids):
                continue
            ids = ids[::max(1, int(np.ceil(len(ids) / 500)))]
            xs.append(feature_getter(row, variant)[ids])
            ys.append(np.full(len(ids), cls))
            ws.append(np.full(len(ids), 1. / len(ids)))
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(classes):
        if not np.any(y == cls):
            raise ValueError(f"missing {head} class {cls}")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w) / w.sum()
    return x, y, w


def train_heads(rows, output_dir=OUT, feature_getter=head_features,
                variants=("scalar", "vector"), seed=269):
    records, chosen = {}, {}
    for head in ("context", "direction"):
        records[head] = {}
        candidates = []
        for variant in variants:
            x, y, w = dataset(rows, head, variant, feature_getter=feature_getter)
            vx, vy, vw = dataset(rows, head, variant, validate=True, feature_getter=feature_getter)
            for width in (0, 32):
                model = StateMLP(width, seed=seed, n_classes=4 if head == "context" else 3)
                model.fit(x, y, w, (vx, vy, vw), epochs=120)
                key = f"{variant}_width{width}"
                model.save(output_dir / f"{head}_{key}.npz")
                item = {"feature_count": x.shape[1], "width": width, "best_epoch": model.best_epoch,
                        "validation_loss": model.best_loss, "training_frames": len(y),
                        "training_class_counts": np.bincount(y, minlength=model.n_classes).tolist(),
                        "validation_frames": len(vy), "validation_class_counts": np.bincount(vy, minlength=model.n_classes).tolist(),
                        "history": model.history}
                records[head][key] = item
                candidates.append((model.best_loss, model, variant, key))
                print(head, key, model.best_epoch, round(model.best_loss, 5), flush=True)
        _, model, variant, key = min(candidates, key=lambda item: item[0])
        chosen[head] = (model, variant, key)
        records[head]["selected"] = key
    return records, chosen


def stage_summary(row, states, *, mask=None):
    results = {}
    subset = row["eligible"] if mask is None else row["eligible"] & mask
    for label in ("shallow_descent", "shallow_ascent", "deep_contact", "deep_still", "lift_residual", "postcontact_lift", "unloaded_trial", "precontact_lift"):
        m = subset & (row["labels"] == label)
        if not m.any():
            continue
        r = m & row["response"]
        results[label] = {"n": int(m.sum()), "response": int(r.sum()),
                          "states_all": np.bincount(states[m], minlength=7).tolist(),
                          "states_response": np.bincount(states[r], minlength=7).tolist()}
    return results


def policy_metrics(rows, scores, policy, validation=True):
    counts = np.zeros((4, 7), int)
    for row, (c, d) in zip(rows, scores):
        if validation and row["partition"] != "development":
            continue
        states, _, _ = replay_phase(row["time"], c, d, row["response"], row["eligible"], policy)
        m = row["eligible"] & (row["context_y"] >= 0)
        if validation:
            m &= row["state_validate"]
        np.add.at(counts, (row["context_y"][m], states[m]), 1)
    denom = np.maximum(counts.sum(1), 1)
    weak_rate = float(counts[1, 2:5].sum() / denom[1])
    residue_false = float(counts[3, 2:5].sum() / denom[3])
    supported_false = float(counts[2, 2:5].sum() / denom[2])
    quiet_false = float(counts[0, 2:5].sum() / denom[0])
    supported_rate = float(counts[2, 6] / denom[2])
    objective = (1. - weak_rate) + 2. * residue_false + 2. * supported_false + quiet_false + .25 * (1. - supported_rate)
    qualified = residue_false <= .01 and supported_false <= .01 and quiet_false <= .001 and weak_rate >= .60
    return {"objective": objective, "qualified_earlier_validation": bool(qualified),
            "weak_context_coverage": weak_rate, "residue_false_weak_rate": residue_false,
            "supported_false_weak_rate": supported_false, "quiet_false_weak_rate": quiet_false,
            "supported_state_coverage": supported_rate, "confusion_context_rows_state_columns": counts.tolist()}


def event_diagnostics(row, states):
    events = []
    # References only identify diagnostic episodes AFTER replay; no backdating.
    for label, desired in (("shallow_descent", 2), ("shallow_ascent", 4)):
        ids = np.flatnonzero(row["eligible"] & (row["labels"] == label))
        for block in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
            if not len(block):
                continue
            responsive = block[row["response"][block]]
            correct = block[states[block] == desired]
            first = int(correct[0]) if len(correct) else None
            events.append({"stage": label, "start_time_ns": int(row["time"][block[0]]),
                           "end_time_ns": int(row["time"][block[-1]]), "frames": len(block),
                           "response_frames": len(responsive), "matched_direction_frames": len(correct),
                           "first_match_time_ns": None if first is None else int(row["time"][first]),
                           "delay_from_stage_start_s": None if first is None else (int(row["time"][first]) - int(row["time"][block[0]])) / 1e9})
    return events


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = load_rows()
    prepare_hierarchy(rows)
    OUT.mkdir(parents=True)
    records, models = train_heads(rows)
    scores = [(models["context"][0].predict(head_features(row, models["context"][1])),
               models["direction"][0].predict(head_features(row, models["direction"][1]))) for row in rows]
    policies = []
    for weak in (.60, .75, .90):
        for delay in (.03, .06):
            p = PhasePolicy(weak_limit=weak, confirm_s=delay)
            policies.append((p, policy_metrics(rows, scores, p)))
    qualified = [item for item in policies if item[1]["qualified_earlier_validation"]]
    selected, validation = min(qualified or policies, key=lambda item: item[1]["objective"])
    pooled, trials = {}, []
    for row, (c, d) in zip(rows, scores):
        state, proposal, carried = replay_phase(row["time"], c, d, row["response"], row["eligible"], selected)
        trial = int(row["path"].name.split("_")[1])
        np.savez_compressed(OUT / f"trial_{trial:06d}_state.npz", time_ns=row["time"],
                            context_scores=c, direction_scores=d, response=row["response"],
                            state=state, proposal=proposal, carried=carried, eligible=row["eligible"],
                            physical_contact_truth=np.full(len(state), -1, np.int8))
        grouped = stage_summary(row, state)
        target = pooled.setdefault(row["partition"] + "_" + row["device"], {})
        for label, counts in grouped.items():
            dest = target.setdefault(label, {"n": 0, "response": 0, "states_all": [0]*7, "states_response": [0]*7})
            dest["n"] += counts["n"]; dest["response"] += counts["response"]
            for key in ("states_all", "states_response"):
                dest[key] = (np.array(dest[key]) + np.array(counts[key])).tolist()
        trials.append({"trial": row["path"].name, "device": row["device"], "partition": row["partition"],
                       "mode": row["mode"], "groups": grouped, "events": event_diagnostics(row, state),
                       "raw_sha256": row["raw_sha256"]})
    report = {"schema": "MagTouch-R269-HierarchicalResponseState-1", "status": "offline shadow-state prototype; no UI/gate/XYZ/hardware/paper changes",
              "context_names": CONTEXT, "direction_names": DIRECTION, "state_names": STATE_NAMES,
              "head_training": records, "policy": asdict(selected), "validation": validation,
              "policy_candidates": [{"policy": asdict(p), "validation": r} for p, r in policies],
              "pooled": pooled, "trials": trials,
              "warning": "Shallow stage annotations are not physical contact truth. No shallow XYZ labels/regression. Both device development data used: not zero-shot. R208 already inspected and used only for retrospective audit; models/policy selected by earlier validation. Quiet response is not a physical no-contact assertion.",
              "XYZ_preservation": "No XYZ are generated, modified or suppressed by this shadow state. This is not proof of decoder compatibility under a changed gate."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("policy", asdict(selected), validation, flush=True)
    for name, labels in pooled.items():
        print(name, {k: labels.get(k) for k in ("shallow_descent", "shallow_ascent", "deep_contact", "deep_still", "lift_residual", "postcontact_lift", "unloaded_trial")}, flush=True)


if __name__ == "__main__":
    main()
