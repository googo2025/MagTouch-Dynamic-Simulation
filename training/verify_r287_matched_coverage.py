"""Check matched source budgets, fixed policy and frozen raw lineage."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
import numpy as np
from training.probe_r287_matched_signal_coverage import ROOT, OUT, PRIOR_OUT, AUG_OUT
from training.audit_r282_causal_sequence import verify_freeze as temporal_freeze
from training.verify_r286_signal_sequence import verify_freeze as gain_freeze
from training.audit_r274_frozen_phase_expansion import save_json
from training.build_trajectory_dataset_v1 import sha


def verify_freeze(out):
    p = json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    assert p["frozen_before_target_replay"] and p["no_XYZ_arming"] and p["strength_frozen_R276"]
    assert p["target_fit_rows"] == p["target_selection_rows"] == p["source_test_selection_rows"] == 0
    assert sha(ROOT/p["base_protocol"]) == p["base_protocol_sha256"]
    assert sha(ROOT/p["policy_protocol"]) == p["policy_protocol_sha256"]
    temporal_freeze(ROOT/Path(p["base_protocol"]).parent)
    gain = gain_freeze(ROOT/Path(p["policy_protocol"]).parent)
    for r in p["tracked"]: assert sha(ROOT/r["path"]) == r["sha256"], r["path"]
    if p["selected"]: assert p["selected"]["policy"] == gain["selected"]["policy"]
    return p


def main():
    dest = OUT/"final_verification.json"
    if dest.exists(): raise FileExistsError(dest)
    p = verify_freeze(OUT); raw = p["source_input_manifest"]
    for r in raw: assert sha(ROOT/r["path"]) == r["sha256"]
    partition = json.loads((OUT/"source_partition.json").read_text(encoding="utf-8"))["records"]
    assert len(partition) == 78 and sum(r["split"] == "train" for r in partition) == 57
    assert {r["split"] for r in partition} == {"train", "validation"}
    assert not any("yellow" in r["path"].lower() or "yelow" in r["path"].lower() for r in partition)
    matched = json.loads((OUT/"matched_samples.json").read_text(encoding="utf-8"))
    for scope, views in (("train", 5), ("validation", 3)):
        by_group = {}
        for row in matched[scope]: by_group.setdefault((row["path"], row["class"]), []).append(row)
        for rows in by_group.values():
            assert len(rows) == views and len({r["indices_sha256"] for r in rows}) == 1
            assert len({r["n"] for r in rows}) == 1
        assert sum(r["n"] for r in matched[scope]) == matched[scope+"_samples"]
    frames, records = 0, 0
    if p["accepted_improvement"]:
        s = p["selected"]
        assert s["gain_safety"] and s["native_all_source_safety"] and s["native_noninferior"]
        assert s["objective"] < min(s["previous_objective"], s["matched_control_objective"])-1e-4
        report = json.loads((OUT/"evaluation_report.json").read_text(encoding="utf-8"))
        raw = json.loads((OUT/"evaluation_inputs.json").read_text(encoding="utf-8"))["files"]
        for r in raw: assert sha(ROOT/r["path"]) == r["sha256"]
        for item in report["trials"]:
            with np.load(OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as a, np.load(PRIOR_OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as b:
                for key in ("time_ns", "eligible", "response", "strength"): np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_array_equal(a["old_phase"], b["phase"])
                known = np.isin(a["old_phase"], (2, 4)); np.testing.assert_array_equal(a["phase"][known], a["old_phase"][known])
                np.testing.assert_array_equal(a["gain_direction_supplemented"], a["phase"] != a["old_phase"])
                assert np.all(a["physical_contact_truth"] == -1) and not any("predicted_xyz" in k for k in a.files)
                frames += len(a["phase"])
        records = len(report["trials"])
    else:
        assert p["selected"] is None and not (OUT/"evaluation_report.json").exists()
    names = ("training.test_r287_matched_coverage", "training.test_r286_signal_sequence", "training.test_r286_signal_augmentation",
        "training.test_r284_temporal_context", "training.test_r282_causal_sequence", "training.test_r281_local_direction",
        "training.test_r280_action_context", "training.test_r279_gain_generalization", "training.test_r278_independent_direction",
        "training.test_r277_spatial_features", "training.test_r276_condition_adaptation", "training.test_r275_source_only",
        "training.test_r274_frozen_expansion", "training.test_response_phase_observer", "training.test_r263_weak_label_evidence",
        "training.test_r245_joint_state", "training.test_r227_gate_assist")
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); assert result.wasSuccessful()
    save_json(dest, {"schema": "R287-MatchedCoverageVerification-1", "tests_run": result.testsRun, "tests_passed": True,
        "matched_source_budget_verified": True, "raw_inputs_unchanged": len(raw), "accepted_improvement": p["accepted_improvement"],
        "audit_records": records, "audit_frames": frames, "target_fit_or_selection_rows": 0,
        "no_XYZ_UI_hardware_paper_changes": True, "verification_code_sha256": sha(Path(__file__))})
    print("Verified R287", result.testsRun, "tests; accepted", p["accepted_improvement"], flush=True)


if __name__ == "__main__": main()
