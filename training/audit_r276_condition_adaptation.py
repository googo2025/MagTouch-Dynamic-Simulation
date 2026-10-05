"""Evaluate frozen condition-adapted observer, never fit on target records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from training.probe_r276_condition_adaptation import (
    ROOT, OUT, INVENTORY, source_partition, features, run_scores, score_prediction, aggregate)
from training.audit_r275_source_only_response import load_frozen
from training.audit_r274_frozen_phase_expansion import load_record, post_group, replay as old_replay, save_json, RAW_FILES
from training.audit_r263_magnetic_evidence_uncertain_depth import evidence_scores, confirm_evidence
from training.probe_r245_joint_contact_state import causal_state_features
from training.verify_r273_response_runtime import load_model
from training.build_trajectory_dataset_v1 import sha


def verify_freeze(out):
    p = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    assert p["frozen_before_target_replay"] and p["target_fit_rows"] == p["target_selection_rows"] == 0
    assert sha(ROOT / p["base_protocol"]) == p["base_protocol_sha256"]
    for record in p["tracked"]:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("frozen input changed: " + record["path"])
    return p


def magnetic_row(rec, base):
    row = load_record(rec)
    row["state_x"] = causal_state_features(row["time"], row["field_uT"], row["eligible"])
    absolute, dynamic = evidence_scores(row, base["null"]["source"])
    row["response"] = confirm_evidence(row, absolute, dynamic, base["null"]["source"])
    row["amplitude_uT"] = np.linalg.norm(row["field_uT"], axis=1)
    return row


def infer(row, models, protocol, base):
    scores = [m.predict(features(row, protocol["variant"])) for m in models]
    return run_scores(row, scores, base, protocol["selected_policy"])


def check_prefix(row, prediction, models, protocol, base):
    cuts = sorted({min(512, len(row["time"])), len(row["time"])//2, len(row["time"])-1} - {0})
    for cut in cuts:
        small = {k: v[:cut] if isinstance(v, np.ndarray) and v.shape[0] == len(row["time"]) else v for k, v in row.items()}
        small["state_x"] = causal_state_features(small["time"], small["field_uT"], small["eligible"])
        result = infer(small, models, protocol, base)
        for key in ("strength", "phase", "response", "reentry_rejected", "residual_conflict"):
            np.testing.assert_array_equal(result[key], prediction[key][:cut])
    # A device string cannot select a new normalizer, policy, or parameter.
    other = dict(row, device="target" if row["device"] == "source" else "source")
    result = infer(other, models, protocol, base)
    for key in ("strength", "phase", "response"):
        np.testing.assert_array_equal(result[key], prediction[key])
    return len(cuts)


def cohort(rec):
    if rec["device"] == "source":
        return "source_" + source_partition(rec)
    if "R208_TARGET" in rec["plan"]:
        return "target_complex_history"
    if rec["original_36_record_scope"]:
        return "target_prior_inspected"
    return "target_expanded_49"


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); out = args.out
    p = verify_freeze(out)
    if (out / "evaluation_report.json").exists():
        raise FileExistsError(out / "evaluation_report.json")
    base, _ = load_frozen(); models = [load_model(ROOT / f) for f in p["models"]]
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    selected = [r for r in inventory if r["device"] in ("source", "target") and
                r["date"] >= "S20260926_A" and r["initially_readable"]]
    before = [{"path": (ROOT/r["path"]/name).relative_to(ROOT).as_posix(),
               "sha256": sha(ROOT/r["path"]/name)} for r in selected for name in RAW_FILES
              if (ROOT/r["path"]/name).is_file()]
    save_json(out / "evaluation_inputs.json", {"frozen_model_sha256": sha(out/"freeze.json"), "files": before})
    streams = out / "streams"; streams.mkdir(exist_ok=True)
    items, failures = [], []; prefix_checks = 0; start = time.monotonic()
    for i, rec in enumerate(selected):
        row = magnetic_row(rec, base)
        # No command / depth / trajectory type / domain identifier is consulted.
        baseline, prediction = old_replay(row, base), infer(row, models, p, base)
        if i % 12 == 0 or "R208_" in rec["plan"]:
            prefix_checks += check_prefix(row, prediction, models, p, base)
        # Only now load command annotations for diagnostic grouping.
        row["diagnostic_labels"], row["q"], error = post_group(row, rec)
        if error:
            failures.append({"path": rec["path"], "reason": error})
        row["split"] = cohort(rec)
        old, new = score_prediction(row, baseline), score_prediction(row, prediction)
        key = hashlib.sha256(rec["path"].encode()).hexdigest()[:16]
        np.savez_compressed(streams/f"{key}.npz", time_ns=row["time"], eligible=row["eligible"],
            diagnostic_command_xyz_mm=row["q"], diagnostic_labels=row["diagnostic_labels"],
            old_strength=baseline["strength"], old_phase=baseline["phase"], **prediction,
            physical_contact_truth=np.full(len(row["time"]), -1, np.int8))
        items.append({"path": rec["path"], "key": key, "device": rec["device"], "mode": rec["mode"],
                      "plan": rec["plan"], "cohort": row["split"], "old": old, "new": new})
        if (i+1) % 10 == 0:
            print("frozen adaptation replay", i+1, "/", len(selected), round(time.monotonic()-start, 1), flush=True)
    cohorts = {c: {version: aggregate([v[version] for v in items if v["cohort"] == c]) for version in ("old", "new")}
               for c in sorted({v["cohort"] for v in items})}
    by_mode = {c+":"+mode: {version: aggregate([v[version] for v in items if v["cohort"] == c and v["mode"] == mode])
                  for version in ("old", "new")}
               for c, mode in sorted({(v["cohort"], v["mode"]) for v in items})}
    for record in before:
        if sha(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("raw input changed")
    verify_freeze(out); load_frozen()
    report = {"schema": "R276-FrozenConditionAdaptationAudit-1", "records": len(items), "cohorts": cohorts,
        "by_cohort_mode": by_mode, "prefix_checks": prefix_checks, "domain_invariance_checks": sum(i % 12 == 0 or "R208_" in r["plan"] for i, r in enumerate(selected)),
        "failures": failures, "raw_inputs_unchanged": len(before), "target_fit_rows": 0,
        "target_normalization_or_policy_selection_rows": 0, "trials": items,
        "semantics": "response-phase observer only, no XYZ/true-contact recall; same raw streams and SOURCE null model as R275; target retrospective zero-label transfer; UI/hardware/paper unchanged"}
    save_json(out / "evaluation_report.json", report)
    csv_rows = [{"path": i["path"], "cohort": i["cohort"], "mode": i["mode"], "version": version,
                 **i[version]["rates"]} for i in items for version in ("old", "new")]
    with (out/"per_trial_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({k for row in csv_rows for k in row})); writer.writeheader(); writer.writerows(csv_rows)
    for c, summary in cohorts.items():
        print(c, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
