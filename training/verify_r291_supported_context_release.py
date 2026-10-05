"""Check SOURCE-selected conditional release without altering base models."""
import json
from pathlib import Path
import unittest
import numpy as np
from training.probe_r291_supported_context_release import ROOT, OUT, PRIOR_OUT
from training.verify_r288_ordered_action_context import verify_freeze as prior_freeze
from training.audit_r274_frozen_phase_expansion import save_json
from training.build_trajectory_dataset_v1 import sha


def verify_freeze(out):
    p = json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    assert p["frozen_before_target_replay"] and p["no_XYZ_arming"] and p["base_models_unchanged"]
    assert p["strength_only_contextual_release"]
    assert p["target_fit_rows"] == p["target_selection_rows"] == p["source_test_selection_rows"] == 0
    assert sha(ROOT/p["base_protocol"]) == p["base_protocol_sha256"]
    prior_freeze(ROOT/Path(p["base_protocol"]).parent)
    for r in p["tracked"]: assert sha(ROOT/r["path"]) == r["sha256"], r["path"]
    return p


def main():
    dest = OUT/"final_verification.json"
    if dest.exists(): raise FileExistsError(dest)
    p = verify_freeze(OUT); raw = p["source_input_manifest"]
    for r in raw: assert sha(ROOT/r["path"]) == r["sha256"]
    records = json.loads((OUT/"source_partition.json").read_text(encoding="utf-8"))["records"]
    assert len(records) == 78 and sum(r["split"] == "train" for r in records) == 57
    assert {r["split"] for r in records} == {"train", "validation"}
    assert not any("yellow" in r["path"].lower() or "yelow" in r["path"].lower() for r in records)
    frames = count = 0
    if p["accepted_improvement"]:
        s = p["selected"]; assert s["qualified_protection"] and not s["native_failures"] and not s["gain_failures"]
        assert s["objective"] < s["previous_objective"]-1e-8
        report = json.loads((OUT/"evaluation_report.json").read_text(encoding="utf-8"))
        raw = json.loads((OUT/"evaluation_inputs.json").read_text(encoding="utf-8"))["files"]
        for r in raw: assert sha(ROOT/r["path"]) == r["sha256"]
        for item in report["trials"]:
            with np.load(OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as a, np.load(PRIOR_OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as b:
                for key in ("time_ns", "eligible", "response"): np.testing.assert_array_equal(a[key], b[key])
                np.testing.assert_array_equal(a["old_phase"], b["phase"]); np.testing.assert_array_equal(a["old_strength"], b["strength"])
                released = a["supported_context_released"]
                assert not np.any(released & (a["old_strength"] != 6))
                np.testing.assert_array_equal(released, a["strength"] != a["old_strength"])
                np.testing.assert_array_equal(a["phase"][~released], a["old_phase"][~released])
                assert np.all(a["physical_contact_truth"] == -1) and not any("predicted_xyz" in key for key in a.files)
                frames += len(a["phase"])
        count = len(report["trials"])
    else:
        assert p["selected"] is None and not (OUT/"evaluation_report.json").exists()
    names = ("training.test_r291_supported_context_release", "training.test_r290_shared_action_context",
        "training.test_r288_ordered_action_context", "training.test_r287_matched_coverage",
        "training.test_r286_signal_sequence", "training.test_r286_signal_augmentation", "training.test_r284_temporal_context",
        "training.test_r282_causal_sequence", "training.test_r281_local_direction", "training.test_r280_action_context",
        "training.test_r279_gain_generalization", "training.test_r278_independent_direction", "training.test_r277_spatial_features",
        "training.test_r276_condition_adaptation", "training.test_r275_source_only", "training.test_r274_frozen_expansion",
        "training.test_response_phase_observer", "training.test_r263_weak_label_evidence", "training.test_r245_joint_state",
        "training.test_r227_gate_assist")
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); assert result.wasSuccessful()
    save_json(dest, {"schema": "R291-SupportedReleaseVerification-1", "tests_run": result.testsRun, "tests_passed": True,
        "raw_inputs_unchanged": len(raw), "accepted_improvement": p["accepted_improvement"], "audit_records": count,
        "audit_frames": frames, "target_fit_selection_rows": 0, "base_models_unchanged": True,
        "no_XYZ_UI_hardware_paper_changes": True, "verification_code_sha256": sha(Path(__file__))})
    print("Verified R291", result.testsRun, "tests; accepted", p["accepted_improvement"], flush=True)


if __name__ == "__main__": main()
