"""Frozen source-only R278 replay; target is reporting only, never selection."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from training.probe_r278_independent_direction import ROOT, OUT, features, supplement
from training.probe_r276_condition_adaptation import INVENTORY, score_prediction, aggregate
from training.audit_r276_condition_adaptation import verify_freeze as previous_freeze, magnetic_row, infer as baseline_infer, cohort
from training.audit_r275_source_only_response import load_frozen
from training.audit_r274_frozen_phase_expansion import post_group, RAW_FILES, save_json
from training.probe_r245_joint_contact_state import causal_state_features
from training.verify_r273_response_runtime import load_model
from training.build_trajectory_dataset_v1 import sha


def verify_freeze(out):
    p = json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    assert p["frozen_before_target_replay"] and p["no_XYZ_arming"] and p["strength_frozen_R276"]
    assert p["target_fit_rows"] == p["target_selection_rows"] == p["source_test_selection_rows"] == 0
    assert sha(ROOT/p["base_protocol"]) == p["base_protocol_sha256"]
    previous_freeze(ROOT/Path(p["base_protocol"]).parent)
    for r in p["tracked"]:
        if sha(ROOT/r["path"]) != r["sha256"]: raise ValueError("frozen input changed: "+r["path"])
    return p


def infer(row, model, p, base, prior, prior_models):
    baseline = baseline_infer(row, prior_models, prior, base)
    result = supplement(row, baseline, model.predict(features(row, p["selected"]["variant"])), p["selected"]["policy"])
    return baseline, result


def check_prefix(row, result, model, p, base, prior, prior_models):
    cuts = sorted({min(512, len(row["time"])), len(row["time"])//2, len(row["time"])-1}-{0})
    for cut in cuts:
        small = {k: v[:cut] if isinstance(v, np.ndarray) and v.shape[0] == len(row["time"]) else v for k, v in row.items()}
        small["state_x"] = causal_state_features(small["time"], small["field_uT"], small["eligible"])
        _, pred = infer(small, model, p, base, prior, prior_models)
        for key in ("strength", "phase", "response", "direction_supplemented", "direction_context_conflict"):
            np.testing.assert_array_equal(pred[key], result[key][:cut])
    swapped = dict(row, device="target" if row["device"] == "source" else "source", mode="changed_template",
        diagnostic_command_xyz_mm=np.full((len(row["time"]), 3), 999.))
    _, pred = infer(swapped, model, p, base, prior, prior_models)
    for key in ("strength", "phase", "response", "direction_supplemented"):
        np.testing.assert_array_equal(pred[key], result[key])
    return len(cuts)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(); out = args.out; p = verify_freeze(out)
    if not p["accepted_improvement"]: raise ValueError("source validation has no qualified improvement; no target replay")
    if (out/"evaluation_report.json").exists(): raise FileExistsError(out/"evaluation_report.json")
    base, _ = load_frozen(); prior = previous_freeze(ROOT/Path(p["base_protocol"]).parent)
    prior_models = [load_model(ROOT/f) for f in p["baseline_models"]]
    model = load_model(ROOT/p["selected"]["file"])
    records = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    records = [r for r in records if r["device"] in ("source", "target") and r["date"] >= "S20260926_A" and r["initially_readable"]]
    before = [{"path": (ROOT/r["path"]/name).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/name)}
        for r in records for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    save_json(out/"evaluation_inputs.json", {"files": before, "freeze_sha256": sha(out/"freeze.json")})
    streams = out/"streams"; streams.mkdir(exist_ok=True)
    start = time.monotonic(); items = []; failures = []; prefix_checks = 0; invariance_checks = 0
    for i, rec in enumerate(records):
        row = magnetic_row(rec, base)
        old, new = infer(row, model, p, base, prior, prior_models)
        np.testing.assert_array_equal(new["strength"], old["strength"])
        np.testing.assert_array_equal(new["phase"][np.isin(old["phase"], (2, 4))], old["phase"][np.isin(old["phase"], (2, 4))])
        if i % 12 == 0 or "R208_" in rec["plan"]:
            prefix_checks += check_prefix(row, new, model, p, base, prior, prior_models); invariance_checks += 1
        # Only after predictions, open command annotations for diagnostic bins.
        row["diagnostic_labels"], row["q"], error = post_group(row, rec)
        if error: failures.append({"path": rec["path"], "reason": error})
        row["split"] = cohort(rec)
        key = hashlib.sha256(rec["path"].encode()).hexdigest()[:16]
        np.savez_compressed(streams/f"{key}.npz", time_ns=row["time"], eligible=row["eligible"],
            diagnostic_command_xyz_mm=row["q"], diagnostic_labels=row["diagnostic_labels"],
            old_strength=old["strength"], old_phase=old["phase"], **new,
            physical_contact_truth=np.full(len(row["time"]), -1, np.int8))
        items.append({"path": rec["path"], "key": key, "device": rec["device"], "mode": rec["mode"],
            "plan": rec["plan"], "cohort": row["split"], "old": score_prediction(row, old), "new": score_prediction(row, new),
            "supplemented_frames": int(new["direction_supplemented"].sum()),
            "context_conflict_frames": int(new["direction_context_conflict"].sum())})
        if (i+1) % 10 == 0: print("R278 frozen same-stream audit", i+1, "/", len(records), round(time.monotonic()-start, 1), flush=True)
    cohorts = {c: {v: aggregate([i[v] for i in items if i["cohort"] == c]) for v in ("old", "new")}
        for c in sorted({i["cohort"] for i in items})}
    by_mode = {c+":"+m: {v: aggregate([i[v] for i in items if i["cohort"] == c and i["mode"] == m]) for v in ("old", "new")}
        for c, m in sorted({(i["cohort"], i["mode"]) for i in items})}
    for r in before:
        if sha(ROOT/r["path"]) != r["sha256"]: raise ValueError("raw input changed")
    verify_freeze(out); load_frozen()
    save_json(out/"evaluation_report.json", {"schema": "R278-FrozenIndependentSourceDirectionAudit-1", "records": len(items),
        "cohorts": cohorts, "by_cohort_mode": by_mode, "trials": items, "failures": failures,
        "prefix_checks": prefix_checks, "domain_command_template_invariance_checks": invariance_checks,
        "raw_inputs_unchanged": len(before), "target_fit_rows": 0, "target_selection_rows": 0,
        "all_strength_states_identical_R276": True, "all_baseline_declared_actions_preserved": True,
        "semantics": "magnetic response direction, not true contact or shallow XYZ; source-only frozen selection, retrospective target reporting"})
    csv_rows = [{"path": i["path"], "cohort": i["cohort"], "mode": i["mode"], "version": v, **i[v]["rates"]}
        for i in items for v in ("old", "new")]
    with (out/"per_trial_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for i in csv_rows for k in i})); writer.writeheader(); writer.writerows(csv_rows)
    for c, value in cohorts.items(): print(c, json.dumps({v: value[v]["rates"] for v in ("old", "new")}), flush=True)


if __name__ == "__main__": main()
