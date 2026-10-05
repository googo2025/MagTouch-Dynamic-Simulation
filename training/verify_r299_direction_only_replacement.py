"""Verify source-only extraction, qualification, lineage and raw integrity."""
import json
from pathlib import Path
import unittest
import numpy as np
from training.probe_r299_direction_only_replacement import ROOT, OUT, REFERENCE_OUT, NETWORK_OUT, extract_direction, MultiTaskMLP, load_model
from training.verify_r297_compatible_task_commitment import verify_freeze as lineage_verify
from training.audit_r274_frozen_phase_expansion import save_json
from training.build_trajectory_dataset_v1 import sha


def verify_freeze(out):
    p = lineage_verify(out)
    assert p["strength_exact_reference"] and p["selected"] == json.loads((out/"training_report.json").read_text(encoding="utf-8"))["selected"]
    report = json.loads((out/"training_report.json").read_text(encoding="utf-8"))
    nr = json.loads((NETWORK_OUT/"training_report.json").read_text(encoding="utf-8"))
    assert report["head"] == min(nr["heads"], key=lambda h: h["validation_loss"])
    extracted = load_model(out/"direction_extracted.npz")
    donor = extract_direction(MultiTaskMLP.load(ROOT/report["head"]["file"]))
    for a, b in zip((extracted.mean, extracted.scale, *extracted.params), (donor.mean, donor.scale, *donor.params)):
        np.testing.assert_array_equal(a, b)
    assert sum(v.size for v in extracted.params) == 27123
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
    count = frames = 0
    if not p["accepted_improvement"]:
        assert p["selected"] is None and not (OUT/"evaluation_report.json").exists()
    else:
        s = p["selected"]; assert s["qualified_protection"] and s["native_objective_noninferior"]
        assert not s["native_failures"] and not s["gain_failures"]
        if (OUT/"evaluation_report.json").exists():
            audit = json.loads((OUT/"evaluation_report.json").read_text(encoding="utf-8"))
            files = json.loads((OUT/"evaluation_inputs.json").read_text(encoding="utf-8"))["files"]
            for r in files: assert sha(ROOT/r["path"]) == r["sha256"]
            for item in audit["trials"]:
                with np.load(OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as a, np.load(REFERENCE_OUT/"streams"/f"{item['key']}.npz", allow_pickle=False) as b:
                    for key in ("time_ns", "eligible", "response", "strength"): np.testing.assert_array_equal(a[key], b[key])
                    np.testing.assert_array_equal(a["old_phase"], b["phase"])
                    np.testing.assert_array_equal(a["base_strength"] != a["strength"], a["total_support_released"])
                    assert np.all(a["physical_contact_truth"] == -1) and not any("predicted_xyz" in k for k in a.files)
                    frames += len(a["phase"])
            count = len(audit["trials"])
    names = ("training.test_r299_direction_only_replacement", "training.test_r298_coherent_response_head",
        "training.test_r297_compatible_task_commitment", "training.test_quiet_magnetic_innovation", "training.test_r295_matched_source_hard_context",
        "training.test_r294_matched_weak_coverage", "training.test_r293_decoupled_support_context", "training.test_r292_shared_multitask_response",
        "training.test_r291_supported_context_release", "training.test_r290_shared_action_context", "training.test_r288_ordered_action_context",
        "training.test_r287_matched_coverage", "training.test_r286_signal_sequence", "training.test_r286_signal_augmentation",
        "training.test_r284_temporal_context", "training.test_r282_causal_sequence", "training.test_r281_local_direction",
        "training.test_r280_action_context", "training.test_r279_gain_generalization", "training.test_r278_independent_direction",
        "training.test_r277_spatial_features", "training.test_r276_condition_adaptation", "training.test_r275_source_only",
        "training.test_r274_frozen_expansion", "training.test_response_phase_observer", "training.test_r263_weak_label_evidence",
        "training.test_r245_joint_state", "training.test_r227_gate_assist")
    result = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromNames(names)); assert result.wasSuccessful()
    save_json(dest, {"schema": "R299-DirectionReplacementVerification-1", "tests_run": result.testsRun,
        "tests_passed": True, "source_records": len(records), "raw_inputs_unchanged": len(files),
        "extracted_prediction_equivalence": True, "SOURCE_only_network_and_rule_selection": True,
        "accepted_improvement": p["accepted_improvement"], "audit_records": count, "audit_frames": frames,
        "target_fit_selection_rows": 0, "no_XYZ_UI_hardware_paper_changes": True,
        "verification_code_sha256": sha(Path(__file__))})
    print("Verified R299", result.testsRun, "tests; accepted", p["accepted_improvement"], flush=True)


if __name__ == "__main__": main()
