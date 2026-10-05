"""Native same-stream audit after SOURCE-only response-gain freezing."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r286_signal_sequence import ROOT, OUT, PRIOR_OUT, add_gain_direction
from training.verify_r286_signal_sequence import verify_freeze
from training.probe_r282_causal_sequence import predict_sequence, load_temporal
from training.audit_r282_causal_sequence import verify_freeze as temporal_freeze, infer as temporal_infer
from training.audit_r278_independent_direction import verify_freeze as direction_freeze
from training.audit_r276_condition_adaptation import verify_freeze as strength_freeze, magnetic_row, cohort
from training.audit_r275_source_only_response import load_frozen
from training.probe_r276_condition_adaptation import INVENTORY, score_prediction, aggregate
from training.audit_r274_frozen_phase_expansion import RAW_FILES, post_group, save_json
from training.probe_r245_joint_contact_state import causal_state_features
from training.verify_r273_response_runtime import load_model
from training.build_trajectory_dataset_v1 import sha


def infer(row, model, p, *prior_args):
    _, _, old = temporal_infer(row, *prior_args)
    selected = p["selected"]
    value = add_gain_direction(row, old, predict_sequence(row, model, selected["spec"], p["steps"], p["step_s"]), selected["policy"])
    return old, value


def check_prefix(row, result, *args):
    cuts = sorted({min(512, len(row["time"])), len(row["time"])//2, len(row["time"])-1}-{0})
    for cut in cuts:
        small = {k: v[:cut] if isinstance(v, np.ndarray) and v.shape[0] == len(row["time"]) else v for k, v in row.items()}
        small["state_x"] = causal_state_features(small["time"], small["field_uT"], small["eligible"])
        _, pred = infer(small, *args)
        for key in ("phase", "strength", "response", "gain_direction_supplemented", "direction_supplemented"):
            np.testing.assert_array_equal(pred[key], result[key][:cut])
    other = dict(row, device="changed_device", mode="changed_path", q=np.full((len(row["time"]), 3), 999.),
        diagnostic_labels=np.full(len(row["time"]), "fake_contact"))
    _, pred = infer(other, *args)
    for key in ("phase", "strength", "response", "gain_direction_supplemented"):
        np.testing.assert_array_equal(pred[key], result[key])
    return len(cuts)


def main():
    p = verify_freeze(OUT)
    if not p["accepted_improvement"]: raise ValueError("no SOURCE-qualified candidate: native tests/Yellow not opened")
    if (OUT/"evaluation_report.json").exists(): raise FileExistsError(OUT/"evaluation_report.json")
    prior = temporal_freeze(PRIOR_OUT)
    direction = direction_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength = strength_freeze(ROOT/Path(direction["base_protocol"]).parent)
    base, _ = load_frozen(); strength_models = [load_model(ROOT/f) for f in direction["baseline_models"]]
    direction_model = load_model(ROOT/direction["selected"]["file"]); temporal_model = load_temporal(prior["selected"])
    model = load_model(ROOT/p["selected"]["file"])
    args = (model, p, temporal_model, prior, direction, direction_model, strength, strength_models, base)
    recs = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    recs = [r for r in recs if r["device"] in ("source", "target") and r["date"] >= "S20260926_A" and r["initially_readable"]]
    files = [{"path": (ROOT/r["path"]/name).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/name)}
        for r in recs for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    save_json(OUT/"evaluation_inputs.json", {"files": files, "freeze_sha256": sha(OUT/"freeze.json")})
    streams = OUT/"streams"; streams.mkdir(exist_ok=True)
    started = time.monotonic(); items = []; failures = []; prefixes = 0; invariance = 0
    for i, rec in enumerate(recs):
        row = magnetic_row(rec, base); old, new = infer(row, *args)
        np.testing.assert_array_equal(new["strength"], old["strength"])
        action = np.isin(old["phase"], (2, 4)); np.testing.assert_array_equal(new["phase"][action], old["phase"][action])
        if i % 12 == 0 or "R208_" in rec["plan"]:
            prefixes += check_prefix(row, new, *args); invariance += 1
        row["diagnostic_labels"], row["q"], error = post_group(row, rec)
        if error: failures.append({"path": rec["path"], "reason": error})
        row["split"] = cohort(rec); key = hashlib.sha256(rec["path"].encode()).hexdigest()[:16]
        np.savez_compressed(streams/f"{key}.npz", time_ns=row["time"], eligible=row["eligible"],
            diagnostic_command_xyz_mm=row["q"], diagnostic_labels=row["diagnostic_labels"],
            old_phase=old["phase"], old_strength=old["strength"], **new,
            physical_contact_truth=np.full(len(row["time"]), -1, np.int8))
        items.append({"path": rec["path"], "key": key, "device": rec["device"], "mode": rec["mode"],
            "plan": rec["plan"], "cohort": row["split"], "old": score_prediction(row, old), "new": score_prediction(row, new),
            "gain_added_frames": int(new["gain_direction_supplemented"].sum())})
        if (i+1) % 10 == 0: print("R286 frozen native replay", i+1, "/", len(recs), round(time.monotonic()-started, 1), flush=True)
    versions = ("old", "new")
    cohorts = {c: {v: aggregate([i[v] for i in items if i["cohort"] == c]) for v in versions} for c in sorted({i["cohort"] for i in items})}
    by_mode = {c+":"+m: {v: aggregate([i[v] for i in items if i["cohort"] == c and i["mode"] == m]) for v in versions}
        for c, m in sorted({(i["cohort"], i["mode"]) for i in items})}
    for r in files: assert sha(ROOT/r["path"]) == r["sha256"]
    verify_freeze(OUT)
    save_json(OUT/"evaluation_report.json", {"schema": "R286-FrozenNativeAudit-1", "records": len(items),
        "cohorts": cohorts, "by_cohort_mode": by_mode, "trials": items, "failures": failures,
        "prefix_checks": prefixes, "domain_command_template_invariance_checks": invariance,
        "raw_inputs_unchanged": len(files), "target_fit_rows": 0, "target_selection_rows": 0,
        "all_R282_actions_strength_preserved": True,
        "semantics": "magnetic-response direction, not physical contact/XYZ; source frozen retrospective target audit"})
    csv_rows = [{"path": i["path"], "cohort": i["cohort"], "mode": i["mode"], "version": v, **i[v]["rates"]} for i in items for v in versions]
    with (OUT/"per_trial_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for i in csv_rows for k in i})); writer.writeheader(); writer.writerows(csv_rows)
    for c, value in cohorts.items(): print(c, json.dumps({v: value[v]["rates"] for v in versions}), flush=True)


if __name__ == "__main__": main()
