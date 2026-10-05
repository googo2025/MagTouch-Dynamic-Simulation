"""Verify two SOURCE coverage arms without claiming identical sample points."""
import json
from pathlib import Path
import unittest
from training.probe_r294_matched_weak_coverage import ROOT, OUT, budget
from training.verify_r293_decoupled_support_context import verify_freeze as reference_freeze
from training.audit_r274_frozen_phase_expansion import save_json
from training.build_trajectory_dataset_v1 import sha


def verify_freeze(out):
    p = json.loads((out/"freeze.json").read_text(encoding="utf-8"))
    assert p["frozen_before_target_replay"] and p["no_XYZ_arming"] and p["base_models_unchanged"]
    assert p["target_fit_rows"] == p["target_selection_rows"] == p["source_test_selection_rows"] == 0
    assert sha(ROOT/p["base_protocol"]) == p["base_protocol_sha256"]
    reference_freeze(ROOT/Path(p["base_protocol"]).parent)
    for r in p["tracked"]: assert sha(ROOT/r["path"]) == r["sha256"], r["path"]
    return p


def main():
    dest = OUT/"final_verification.json"
    if dest.exists(): raise FileExistsError(dest)
    p = verify_freeze(OUT)
    files = p["source_input_manifest"]
    for r in files: assert sha(ROOT/r["path"]) == r["sha256"]
    records = json.loads((OUT/"source_partition.json").read_text(encoding="utf-8"))["records"]
    assert len(records) == 78 and sum(r["split"] == "train" for r in records) == 57
    assert {r["split"] for r in records} == {"train", "validation"}
    assert not any("yellow" in r["path"].lower() or "yelow" in r["path"].lower() for r in records)
    a, b = [json.loads((OUT/f"samples_{name}.json").read_text(encoding="utf-8"))
            for name in ("intersection_uniform", "per_view_weak")]
    assert budget(a["train"]) == budget(b["train"])
    assert a["validation"] == b["validation"]
    assert a["train_samples"] == b["train_samples"] == 94340
    assert a["validation_samples"] == b["validation_samples"] == 21549
    for old, new in zip(a["train"], b["train"]):
        assert (old["path"], old["gain"], old["class"]) == (new["path"], new["gain"], new["class"])
        if old["class"] in (2, 3): assert old == new
    report = json.loads((OUT/"training_report.json").read_text(encoding="utf-8"))
    assert {h["parameters"] for h in report["heads"]} == {27221}
    assert p["selected"] == report["selected"]
    if not p["accepted_improvement"]:
        assert p["selected"] is None and not (OUT/"evaluation_report.json").exists()
    else:
        s = p["selected"]
        assert s["qualified_protection"] and s["native_objective_noninferior"]
        assert not s["native_failures"] and not s["gain_failures"]
    names = ("training.test_r294_matched_weak_coverage", "training.test_r293_decoupled_support_context",
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
    save_json(dest, {"schema": "R294-MatchedWeakCoverageVerification-1", "tests_run": result.testsRun,
        "tests_passed": True, "source_records": len(records), "raw_inputs_unchanged": len(files),
        "matched_counts_parameters_update_budget": True, "same_SOURCE_validation_points": True,
        "training_points_intentionally_differ": True, "accepted_improvement": p["accepted_improvement"],
        "target_fit_selection_rows": 0, "no_XYZ_UI_hardware_paper_changes": True,
        "verification_code_sha256": sha(Path(__file__))})
    print("Verified R294", result.testsRun, "tests; accepted", p["accepted_improvement"], flush=True)


if __name__ == "__main__": main()
