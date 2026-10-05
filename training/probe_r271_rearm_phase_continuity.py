"""Offline continuity refinement using frozen R269 heads; no XYZ/UI change."""
from __future__ import annotations

from dataclasses import asdict
import json
import numpy as np

from training.probe_r269_hierarchical_response_state import (
    ROOT, load_rows, prepare_hierarchy, stage_summary, event_diagnostics, CONTEXT, DIRECTION)
from training.response_phase_state import PhasePolicy, STATE_NAMES
from training.response_phase_state_v3 import replay_rearm_phase

OUT = ROOT / "analysis/r271_rearm_phase_continuity_20260929"
HEADS = ROOT / "analysis/r269_hierarchical_response_state_20260929"


def metrics(rows, scores, policy):
    counts = np.zeros((4, 7), int)
    for row, (c, d) in zip(rows, scores):
        if row["partition"] != "development":
            continue
        states, _, _, _ = replay_rearm_phase(row["time"], c, d, row["response"], row["fresh_rise"], row["amplitude_uT"], row["eligible"], policy)
        use = row["eligible"] & row["state_validate"] & (row["context_y"] >= 0)
        np.add.at(counts, (row["context_y"][use], states[use]), 1)
    denom = np.maximum(counts.sum(1), 1)
    weak = float(counts[1, 2:5].sum() / denom[1]); residue = float(counts[3, 2:5].sum() / denom[3])
    deep = float(counts[2, 2:5].sum() / denom[2]); quiet = float(counts[0, 2:5].sum() / denom[0])
    supported = float(counts[2, 6] / denom[2])
    return {"objective": 1. - weak + 2.*residue + 2.*deep + quiet + .25*(1.-supported),
            "qualified_earlier_validation": residue <= .01 and deep <= .01 and quiet <= .001 and weak >= .60,
            "weak_context_coverage": weak, "residue_false_weak_rate": residue,
            "supported_false_weak_rate": deep, "quiet_false_weak_rate": quiet,
            "supported_state_coverage": supported, "confusion_context_rows_state_columns": counts.tolist()}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = load_rows(); prepare_hierarchy(rows)
    limits = {}
    for device in ("source", "target"):
        changes = []
        for row in rows:
            if row["partition"] != "development" or row["device"] != device:
                continue
            safe = np.isin(row["labels"], ("unloaded_trial", "precontact_lift", "lift_residual", "postcontact_lift"))
            ids = np.flatnonzero(row["eligible"] & ~row["state_validate"] & safe)[::4]
            changes.extend((100. * row["state_x"][ids, 1]).tolist())
        limits[device] = max(1., float(np.percentile(changes, 99.5)))
    scores = []
    for row in rows:
        n = int(row["path"].name.split("_")[1])
        with np.load(HEADS / f"trial_{n:06d}_state.npz", allow_pickle=False) as a:
            assert np.array_equal(a["time_ns"], row["time"])
            scores.append((a["context_scores"].copy(), a["direction_scores"].copy()))
        row["fresh_rise"] = 100. * row["state_x"][:, 1] > limits[row["device"]]
    policies = [PhasePolicy(weak_limit=w, confirm_s=t) for w in (.60, .75, .90) for t in (.03, .06)]
    candidates = [(p, metrics(rows, scores, p)) for p in policies]
    qualified = [c for c in candidates if c[1]["qualified_earlier_validation"]]
    selected, validation = min(qualified or candidates, key=lambda item: item[1]["objective"])
    OUT.mkdir(parents=True)
    pooled, trials = {}, []
    for row, (c, d) in zip(rows, scores):
        states, proposals, carried, guarded = replay_rearm_phase(row["time"], c, d, row["response"], row["fresh_rise"], row["amplitude_uT"], row["eligible"], selected)
        n = int(row["path"].name.split("_")[1])
        np.savez_compressed(OUT / f"trial_{n:06d}_state.npz", time_ns=row["time"], context_scores=c,
                            direction_scores=d, state=states, proposal=proposals, carried=carried,
                            continuity_guard=guarded, fresh_rise=row["fresh_rise"], response=row["response"],
                            eligible=row["eligible"], physical_contact_truth=np.full(len(states), -1, np.int8))
        groups = stage_summary(row, states)
        dest = pooled.setdefault(row["partition"] + "_" + row["device"], {})
        for label, value in groups.items():
            acc = dest.setdefault(label, {"n": 0, "response": 0, "states_all": [0]*7, "states_response": [0]*7})
            acc["n"] += value["n"]; acc["response"] += value["response"]
            for key in ("states_all", "states_response"):
                acc[key] = (np.array(acc[key]) + np.array(value[key])).tolist()
        trials.append({"trial": row["path"].name, "partition": row["partition"], "device": row["device"],
                       "groups": groups, "events": event_diagnostics(row, states), "raw_sha256": row["raw_sha256"],
                       "continuity_guard_frames": int(np.sum(guarded & row["eligible"]))})
    report = {"schema": "MagTouch-R271-RearmPhaseContinuity-1", "status": "offline shadow state; no UI, XYZ, gate, hardware or paper change",
              "context_names": CONTEXT, "direction_names": DIRECTION, "state_names": STATE_NAMES,
              "frozen_head_source": str(HEADS.relative_to(ROOT)), "policy": asdict(selected), "fresh_rise_limits_uT": limits,
              "fresh_rise_calibration": "99.5 percentile positive 0.1s amplitude change in earlier training safe-lift records, min1uT",
              "validation": validation, "policy_candidates": [{"policy": asdict(p), "validation": v} for p, v in candidates],
              "pooled": pooled, "trials": trials,
              "warning": "Response/action states, not physical shallow-contact truth. Target development data used, not zero-shot. R208 already viewed. Policy picked by earlier validation. Continuity-guarded state is separate from raw classifier confidence. XYZ/gate never altered."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("selected", limits, validation, flush=True)
    for k in ("evaluation_source", "evaluation_target"):
        print(k, pooled[k], flush=True)


if __name__ == "__main__":
    main()
