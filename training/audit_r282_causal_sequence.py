"""Replay SOURCE-selected temporal observer, then read evaluation labels."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r282_causal_sequence import ROOT, OUT, PRIOR_OUT, predict_sequence, add_temporal, load_temporal
from training.probe_r276_condition_adaptation import INVENTORY, score_prediction, aggregate
from training.audit_r278_independent_direction import verify_freeze as prior_freeze, infer as prior_infer
from training.audit_r276_condition_adaptation import verify_freeze as strength_freeze, magnetic_row, cohort
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
    prior_freeze(ROOT/Path(p["base_protocol"]).parent)
    for record in p["tracked"]: assert sha(ROOT/record["path"]) == record["sha256"], record["path"]
    return p


def infer(row, model, p, prior, prior_model, strength, strength_models, base):
    baseline, old = prior_infer(row, prior_model, prior, base, strength, strength_models)
    selected = p["selected"]
    new = add_temporal(row, old, predict_sequence(row, model, selected["spec"], p["steps"], p["step_s"]), selected["policy"])
    return baseline, old, new


def check_prefix(row, result, *args):
    cuts = sorted({min(512, len(row["time"])), len(row["time"])//2, len(row["time"])-1}-{0})
    for cut in cuts:
        small = {k: v[:cut] if isinstance(v, np.ndarray) and v.shape[0] == len(row["time"]) else v for k, v in row.items()}
        small["state_x"] = causal_state_features(small["time"], small["field_uT"], small["eligible"])
        _, _, pred = infer(small, *args)
        for key in ("strength", "phase", "response", "temporal_direction_supplemented", "direction_supplemented"):
            np.testing.assert_array_equal(pred[key], result[key][:cut])
    swapped = dict(row, device="changed_device", mode="changed_template", q=np.full((len(row["time"]), 3), 999.),
        diagnostic_labels=np.full(len(row["time"]), "fake_contact"))
    _, _, pred = infer(swapped, *args)
    for key in ("strength", "phase", "response", "direction_supplemented"):
        np.testing.assert_array_equal(pred[key], result[key])
    return len(cuts)


def main():
    p = verify_freeze(OUT)
    if not p["accepted_improvement"]: raise ValueError("no SOURCE-qualified temporal improvement")
    if (OUT/"evaluation_report.json").exists(): raise FileExistsError(OUT/"evaluation_report.json")
    base, _ = load_frozen(); prior = prior_freeze(PRIOR_OUT)
    strength = strength_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength_models = [load_model(ROOT/f) for f in prior["baseline_models"]]
    prior_model = load_model(ROOT/prior["selected"]["file"]); model = load_temporal(p["selected"])
    args = (model, p, prior, prior_model, strength, strength_models, base)
    recs = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    recs = [r for r in recs if r["device"] in ("source", "target") and r["date"] >= "S20260926_A" and r["initially_readable"]]
    inputs = [{"path": (ROOT/r["path"]/name).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/name)}
        for r in recs for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    save_json(OUT/"evaluation_inputs.json", {"files": inputs, "freeze_sha256": sha(OUT/"freeze.json")})
    streams = OUT/"streams"; streams.mkdir(exist_ok=True)
    start = time.monotonic(); items = []; failures = []; prefix = 0; invariance = 0
    for i, rec in enumerate(recs):
        row = magnetic_row(rec, base); baseline, old, new = infer(row, *args)
        np.testing.assert_array_equal(new["strength"], old["strength"])
        action = np.isin(old["phase"], (2, 4)); np.testing.assert_array_equal(new["phase"][action], old["phase"][action])
        if i % 12 == 0 or "R208_" in rec["plan"]:
            prefix += check_prefix(row, new, *args); invariance += 1
        # Only after inference, open command references for diagnostic groups.
        row["diagnostic_labels"], row["q"], error = post_group(row, rec)
        if error: failures.append({"path": rec["path"], "reason": error})
        row["split"] = cohort(rec); key = hashlib.sha256(rec["path"].encode()).hexdigest()[:16]
        np.savez_compressed(streams/f"{key}.npz", time_ns=row["time"], eligible=row["eligible"],
            diagnostic_command_xyz_mm=row["q"], diagnostic_labels=row["diagnostic_labels"],
            baseline_phase=baseline["phase"], old_phase=old["phase"], old_strength=old["strength"], **new,
            physical_contact_truth=np.full(len(row["time"]), -1, np.int8))
        items.append({"path": rec["path"], "key": key, "device": rec["device"], "mode": rec["mode"],
            "plan": rec["plan"], "cohort": row["split"], "baseline_R276": score_prediction(row, baseline),
            "old": score_prediction(row, old), "new": score_prediction(row, new),
            "temporal_added_frames": int(new["temporal_direction_supplemented"].sum()),
            "temporal_context_conflict_frames": int(new["temporal_direction_context_conflict"].sum())})
        if (i+1) % 10 == 0: print("R282 frozen native same-stream replay", i+1, "/", len(recs), round(time.monotonic()-start, 1), flush=True)
    versions = ("baseline_R276", "old", "new")
    cohorts = {c: {v: aggregate([i[v] for i in items if i["cohort"] == c]) for v in versions} for c in sorted({i["cohort"] for i in items})}
    by_mode = {c+":"+m: {v: aggregate([i[v] for i in items if i["cohort"] == c and i["mode"] == m]) for v in versions}
        for c, m in sorted({(i["cohort"], i["mode"]) for i in items})}
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    verify_freeze(OUT); load_frozen()
    save_json(OUT/"evaluation_report.json", {"schema": "R282-FrozenCausalSequenceAudit-1", "records": len(items),
        "selected_spec": p["selected"]["spec"], "cohorts": cohorts, "by_cohort_mode": by_mode, "trials": items,
        "failures": failures, "prefix_checks": prefix, "domain_command_template_invariance_checks": invariance,
        "raw_inputs_unchanged": len(inputs), "target_fit_rows": 0, "target_selection_rows": 0,
        "all_strength_states_identical_R276": True, "all_R278_declared_actions_preserved": True,
        "semantics": "causal magnetic direction only, not actual contact or XYZ; SOURCE-selected frozen retrospective target audit"})
    csv_rows = [{"path": i["path"], "cohort": i["cohort"], "mode": i["mode"], "version": v, **i[v]["rates"]} for i in items for v in versions]
    with (OUT/"per_trial_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for i in csv_rows for k in i})); writer.writeheader(); writer.writerows(csv_rows)
    for c, value in cohorts.items(): print(c, json.dumps({v: value[v]["rates"] for v in versions}), flush=True)


if __name__ == "__main__": main()
