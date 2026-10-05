"""Frozen SOURCE task compatibility without changing neural weights."""
import json
from pathlib import Path
import unittest
import numpy as np
from training.probe_r297_compatible_task_commitment import ROOT, OUT, REFERENCE_OUT
from training.verify_r294_matched_weak_coverage import verify_freeze as base_verify
from training.verify_r295_matched_source_hard_context import verify_freeze as network_verify
from training.build_trajectory_dataset_v1 import sha
from training.audit_r274_frozen_phase_expansion import save_json


def verify_freeze(out):
    p = base_verify(out)
    assert sha(ROOT/p["network_protocol"]) == p["network_protocol_sha256"]
    network_verify(ROOT/Path(p["network_protocol"]).parent)
    return p


def main():
    dest = OUT/"final_verification.json"
    if dest.exists(): raise FileExistsError(dest)
    p = verify_freeze(OUT); files = p["source_input_manifest"]
    for r in files: assert sha(ROOT/r["path"]) == r["sha256"]
    records = json.loads((OUT/"source_partition.json").read_text(encoding="utf-8"))["records"]
    assert len(records) == 78 and sum(r["split"] == "train" for r in records) == 57
    assert {r["split"] for r in records} == {"train", "validation"}
    assert not any("yellow" in r["path"].lower() or "yelow" in r["path"].lower() for r in records)
    report = json.loads((OUT/"training_report.json").read_text(encoding="utf-8"))
    network_out = ROOT/Path(p["network_protocol"]).parent
    network_report = json.loads((network_out/"training_report.json").read_text(encoding="utf-8"))
    assert report["head"] == min(network_report["heads"], key=lambda h: h["validation_loss"])
    count = frames = 0
    if not p["accepted_improvement"]:
        assert p["selected"] is None and not (OUT/"evaluation_report.json").exists()
    else:
        s = p["selected"]
        assert s["qualified_protection"] and s["native_objective_noninferior"]
        assert not s["native_failures"] and not s["gain_failures"]
        if (OUT/"evaluation_report.json").exists():
            audit = json.loads((OUT/"evaluation_report.json").read_text(encoding="utf-8"))
            files = json.loads((OUT/"evaluation_inputs.json").read_text(encoding="utf-8"))["files"]
            for r in files: assert sha(ROOT/r["path"]) == r["sha256"]
            for item in audit["trials"]:
                with np.load(OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as a, np.load(REFERENCE_OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as b:
                    for key in ("time_ns", "eligible", "response"): np.testing.assert_array_equal(a[key], b[key])
                    np.testing.assert_array_equal(a["old_phase"], b["phase"])
                    np.testing.assert_array_equal(a["old_strength"], b["strength"])
                    assert not np.any(a["contextual_support_released"] & (a["base_strength"] != 6))
                    keep = ~a["contextual_support_released"]
                    np.testing.assert_array_equal(a["strength"][keep], a["base_strength"][keep])
                    assert np.all(a["physical_contact_truth"] == -1) and not any("predicted_xyz" in k for k in a.files)
                    frames += len(a["phase"])
            count = len(audit["trials"])
    names = ("training.test_r297_compatible_task_commitment", "training.test_quiet_magnetic_innovation",
        "training.test_r295_matched_source_hard_context", "training.test_r294_matched_weak_coverage", "training.test_r293_decoupled_support_context",
        "training.test_r292_shared_multitask_response", "training.test_r291_supported_context_release",
        "training.test_r290_shared_action_context", "training.test_r288_ordered_action_context",
        "training.test_r287_matched_coverage", "training.test_r286_signal_sequence", "training.test_r286_signal_augmentation",
        "training.test_r284_temporal_context", "training.test_r282_causal_sequence", "training.test_r281_local_direction",
        "training.test_r280_action_context", "training.test_r279_gain_generalization", "training.test_r278_independent_direction",
        "training.test_r277_spatial_features", "training.test_r276_condition_adaptation", "training.test_r275_source_only",
        "training.test_r274_frozen_expansion", "training.test_response_phase_observer", "training.test_r263_weak_label_evidence",
        "training.test_r245_joint_state", "training.test_r227_gate_assist")
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names))
    assert result.wasSuccessful()
    save_json(dest, {"schema": "R297-CompatibleCommitmentVerification-1", "tests_run": result.testsRun,
        "tests_passed": True, "source_records": len(records), "raw_inputs_unchanged": len(files),
        "network_weights_unchanged": True, "SOURCE_only_network_and_rule_selection": True,
        "accepted_improvement": p["accepted_improvement"], "audit_records": count, "audit_frames": frames,
        "target_fit_selection_rows": 0, "no_XYZ_UI_hardware_paper_changes": True,
        "verification_code_sha256": sha(Path(__file__))})
    print("Verified R297", result.testsRun, "tests; accepted", p["accepted_improvement"], flush=True)


if __name__ == "__main__": main()
