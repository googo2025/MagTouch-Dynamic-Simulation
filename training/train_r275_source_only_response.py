"""Source-only response observer refit. No target records opened during fitting.

All three heads, normalizers, null covariance, evidence limits, activity limits
and policies are selected using source development data only. Target data may
be replayed only after the hashed freeze file exists. This is a retrospective
zero-label target transfer, not an unseen-device prospective experiment.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from training.audit_r226_shallow_history_gate import ROOT, load_trial
from training.audit_r263_magnetic_evidence_uncertain_depth import null_fit, evidence_scores, confirm_evidence
from training.probe_r245_joint_contact_state import prepare, StateMLP
from training.probe_r269_hierarchical_response_state import train_heads, head_features
from training.probe_r228_shallow_inverse import reference
from training.probe_r271_rearm_phase_continuity import metrics as strength_metrics
from training.probe_r272_factorized_action_phase import dataset as binary_dataset, metrics as phase_metrics
from training.response_phase_state import PhasePolicy
from training.response_phase_state_v3 import replay_rearm_phase
from training.factorized_response_phase import DirectionPolicy
from training.build_trajectory_dataset_v1 import sha
from training.audit_r274_frozen_phase_expansion import save_json, RAW_FILES

OUT = ROOT / "analysis/r275_source_only_zero_shot_20260929"
INVENTORY = ROOT / "analysis/r274_frozen_phase_expansion_20260929/inventory.json"


def assert_source(rows):
    if not rows or any(r["device"] != "source" or "had-Yelow" in str(r["path"]) or
                       "yellow" in str(r["path"]).lower() for r in rows):
        raise ValueError("source-only fitting: target or empty input rejected")


def load_source_development():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    # Select before opening raw/labels. Never call the mixed-device load_rows().
    selected = [r for r in inventory if r["original_36_record_scope"] and r["device"] == "source"
                and any(s in r["plan"] for s in ("R133_RETRAIN", "R136_"))]
    rows = [load_trial(ROOT / r["path"], "development") for r in selected]
    assert_source(rows)
    return rows


def label_rows(rows, null):
    assert_source(rows)
    for row in rows:
        absolute, dynamic = evidence_scores(row, null)
        row["response"] = confirm_evidence(row, absolute, dynamic, null)
        y = np.full(len(row["time"]), -1, int)
        safe = np.isin(row["labels"], ("unloaded_trial", "precontact_lift", "lift_residual", "postcontact_lift"))
        y[row["eligible"] & safe & ~row["response"]] = 0
        y[row["eligible"] & row["response"] & np.isin(row["labels"], ("deep_contact", "deep_still"))] = 2
        y[row["eligible"] & row["response"] & np.isin(row["labels"], ("lift_residual", "postcontact_lift"))] = 3
        d = np.full(len(y), -1, int)
        if row["mode"] != "none":
            q = reference(row); finite = np.isfinite(q).all(1)
            weak = row["eligible"] & row["response"] & finite & (q[:, 2] >= .02) & (q[:, 2] < .4)
            y[weak] = 1
            dz = np.r_[0., np.diff(q[:, 2])]
            prior = np.r_[False, finite[:-1]]
            continuous = np.r_[False, (np.diff(row["time"]) > 0) & (np.diff(row["time"]) <= 250_000_000)]
            use = weak & prior & continuous
            d[use & (dz > 1e-4)] = 0
            d[use & (dz < -1e-4)] = 1
            d[use & (np.abs(dz) <= 1e-4)] = 2
            d[(y == 2) & prior & continuous & (np.abs(dz) <= 1e-4)] = 2
        d[y == 3] = 2
        row["context_y"], row["direction_y"] = y, d


def source_limits(rows):
    assert_source(rows)
    values = []
    for row in rows:
        safe = np.isin(row["labels"], ("unloaded_trial", "precontact_lift", "lift_residual", "postcontact_lift"))
        ids = np.flatnonzero(row["eligible"] & ~row["state_validate"] & safe)[::4]
        values.extend((100. * row["state_x"][ids, 1]).tolist())
    return max(1., float(np.percentile(values, 99.5))), max(1., float(np.percentile(np.abs(values), 95.)))


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    start = time.monotonic()
    rows = load_source_development(); prepare(rows)
    null = null_fit(rows, "source"); label_rows(rows, null)
    np.savez_compressed(OUT / "source_null_model.npz", **null)
    records, selected = train_heads(rows, output_dir=OUT, variants=("scalar",), seed=275)
    train, val = binary_dataset(rows), binary_dataset(rows, validate=True)
    candidates, binary_records = [], []
    for width in (0, 32):
        model = StateMLP(width, seed=275, n_classes=2).fit(*train, validation=val, epochs=120)
        model.save(OUT / f"conditional_direction_width{width}.npz")
        item = {"width": width, "best_epoch": model.best_epoch, "validation_loss": model.best_loss,
                "training_counts": np.bincount(train[1], minlength=2).tolist(),
                "validation_counts": np.bincount(val[1], minlength=2).tolist(), "history": model.history}
        binary_records.append(item); candidates.append(model)
        print("source binary", width, model.best_loss, flush=True)
    binary = min(candidates, key=lambda m: m.best_loss)
    fresh, change = source_limits(rows)
    scores = [(selected["context"][0].predict(head_features(r, "scalar")),
               selected["direction"][0].predict(head_features(r, "scalar"))) for r in rows]
    for row in rows:
        row["fresh_rise"] = 100. * row["state_x"][:, 1] > fresh
    scandidates = []
    for w in (.60, .75, .90):
        for delay in (.03, .06):
            policy = PhasePolicy(weak_limit=w, confirm_s=delay)
            scandidates.append((policy, strength_metrics(rows, scores, policy)))
    passing = [v for v in scandidates if v[1]["qualified_earlier_validation"]]
    spolicy, svalidation = min(passing or scandidates, key=lambda v: v[1]["objective"])
    frozen_strength = []
    for row, (c, d) in zip(rows, scores):
        state, _, _, _ = replay_rearm_phase(row["time"], c, d, row["response"], row["fresh_rise"],
                                           row["amplitude_uT"], row["eligible"], spolicy)
        frozen_strength.append({"state": state, "direction_scores": d})
    bscores = [binary.predict(head_features(r, "scalar")) for r in rows]
    pcandidates = []
    for confidence in (.85, .95):
        for activity in (.15, .35, .60):
            for tail in (0., .10, .20):
                p = DirectionPolicy(conditional_limit=confidence, activity_limit=activity,
                                    residue_tail_s=tail, reuse_confirmed_direction=True)
                v = phase_metrics(rows, frozen_strength, bscores, {"source": change}, p)
                v["qualified_direction_validation"] &= v["rates"]["loading"] >= .60 and v["rates"]["withdrawal"] >= .60
                pcandidates.append((p, v))
    qualified = [v for v in pcandidates if v[1]["qualified_direction_validation"]]
    ppolicy, pvalidation = min(qualified or pcandidates, key=lambda v: v[1]["objective"])
    manifest = []
    for row in rows:
        for name in RAW_FILES + ("live_predictions.csv",):
            file = row["path"] / name
            if file.is_file():
                manifest.append({"path": file.relative_to(ROOT).as_posix(), "sha256": sha(file)})
    model_files = [OUT / f"{name}_{selected[name][2]}.npz" for name in ("context", "direction")]
    model_files += [OUT / f"conditional_direction_width{binary.width}.npz", OUT / "source_null_model.npz"]
    report = {"schema": "R275-SourceOnlyRefit-1", "heads": records, "conditional": binary_records,
              "selected_binary_width": binary.width, "strength_policy": asdict(spolicy), "phase_policy": asdict(ppolicy),
              "fresh_rise_limit_uT": fresh, "activity_change_limit_uT": change,
              "strength_validation": svalidation, "phase_validation": pvalidation,
              "strength_candidates": [{"policy": asdict(p), "validation": v} for p, v in scandidates],
              "phase_candidates": [{"policy": asdict(p), "validation": v} for p, v in pcandidates],
              "source_trials": [{"path": r["path"].relative_to(ROOT).as_posix(), "train_frames": int(np.sum(r["eligible"] & ~r["state_validate"])),
                                 "validation_frames": int(np.sum(r["eligible"] & r["state_validate"]))} for r in rows],
              "target_fit_rows": 0, "target_normalization_rows": 0, "target_noise_rows": 0, "target_selection_rows": 0,
              "selection_split": "same source development scope as preceding work; whole Trials 9/12/31 validation, grid Trial11 complete XY-node hash holdout; no frame-random split",
              "input_manifest": manifest, "elapsed_s": time.monotonic()-start}
    save_json(OUT / "training_report.json", report)
    protocol = {"schema": "R275-FrozenSourceOnlyProtocol-1", "frozen_before_target_replay": True,
                "target_fit_rows": 0, "source_null_applies_to_both_devices": True,
                "target_session_B0": "recorded unloaded B0 subtraction is input acquisition, no fitted target noise/scale/threshold/XYZ alignment",
                "models": [v.relative_to(ROOT).as_posix() for v in model_files[:3]],
                "null": model_files[3].relative_to(ROOT).as_posix(),
                "strength_policy": asdict(spolicy), "phase_policy": asdict(ppolicy),
                "fresh_rise_limit_uT": fresh, "activity_change_limit_uT": change,
                "conditional_next_step_acceptance": {"whole_responsive_loading_fraction_min": .95,
                    "whole_responsive_withdrawal_fraction_min": .95, "supported_preservation_min": .95,
                    "safe_lift_loading_fraction_max": .001, "protocol_no_motion_any_action_fraction_max": .001},
                "acceptance_scope": "new-scope source/target 5mm separately; narrow-window results and nonresponse counts still mandatory, acceptance is not physical contact certification or deployment",
                "evaluation_scope": "R274 inventory retained including missing-file and baseline exclusions; source training records reported separately",
                "no_future_or_template": True, "no_shallow_XYZ_truth": True,
                "qualification": "retrospective zero-label target transfer, target previously inspected; source-only weights/selection do not erase historical knowledge of target behavior"}
    tracked = model_files + [OUT / "training_report.json", Path(__file__),
        ROOT / "training/probe_r245_joint_contact_state.py", ROOT / "training/audit_r263_magnetic_evidence_uncertain_depth.py",
        ROOT / "training/response_phase_state.py", ROOT / "training/response_phase_state_v3.py", ROOT / "training/factorized_response_phase.py"]
    protocol["selected_files"] = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked]
    save_json(OUT / "freeze.json", protocol)
    print("FROZEN source-only", json.dumps({"source_trials": len(rows), "fresh": fresh, "change": change,
          "strength": svalidation, "phase": pvalidation}), flush=True)


if __name__ == "__main__":
    main()
