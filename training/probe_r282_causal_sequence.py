"""SOURCE-only bounded causal GRU direction search, with ordered-MLP control."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r278_independent_direction import (
    ROOT, INVENTORY, source_partition, load_source, action_annotations, supplement, qualifies, OUT as PRIOR_OUT)
from training.audit_r278_independent_direction import verify_freeze as prior_freeze, infer as prior_infer
from training.audit_r276_condition_adaptation import verify_freeze as strength_freeze
from training.audit_r275_source_only_response import load_frozen
from training.probe_r276_condition_adaptation import score_prediction, aggregate, policy_objective
from training.audit_r274_frozen_phase_expansion import RAW_FILES, save_json
from training.probe_r245_joint_contact_state import StateMLP
from training.independent_response_direction import SupplementPolicy
from training.verify_r273_response_runtime import load_model
from training.build_trajectory_dataset_v1 import sha
from training.causal_response_sequence import sequence_features, CausalGRU

OUT = ROOT/"analysis/r282_source_causal_sequence_20260929"
SPECS = ({"name": "gru_ratio", "architecture": "GRU", "variant": "ratio", "width": 16},
         {"name": "gru_relative", "architecture": "GRU", "variant": "relative", "width": 16},
         {"name": "ordered_mlp_relative", "architecture": "MLP", "variant": "relative", "width": 32})


def sequence_dataset(rows, variant, validate=False, steps=16, step_s=.1):
    xs, ys, ws = [], [], []
    for row in rows:
        if row["split"] != ("validation" if validate else "train"): continue
        for cls in range(3):
            ids = np.flatnonzero(row["eligible"] & (row["direction_y"] == cls))
            if not len(ids): continue
            ids = ids[::max(1, int(np.ceil(len(ids)/500)))]
            xs.append(sequence_features(row, variant, steps, step_s, ids))
            ys.append(np.full(len(ids), cls)); ws.append(np.full(len(ids), 1./len(ids)))
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(3):
        if not np.any(y == cls): raise ValueError("missing SOURCE sequence direction class")
        w[y == cls] /= w[y == cls].sum()
    w *= len(w)/w.sum()
    return x, y, w


def predict_sequence(row, model, spec, steps=16, step_s=.1):
    chunks = []
    for start in range(0, len(row["time"]), 512):
        x = sequence_features(row, spec["variant"], steps, step_s,
            np.arange(start, min(start+512, len(row["time"]))))
        if spec["architecture"] == "MLP": x = x.reshape(len(x), -1)
        chunks.append(model.predict(x))
    return np.vstack(chunks)


def add_temporal(row, prior, scores, policy):
    value = supplement(row, prior, scores, policy)
    value["prior_direction_supplemented"] = prior["direction_supplemented"]
    value["prior_direction_context_conflict"] = prior["direction_context_conflict"]
    value["temporal_direction_supplemented"] = value["direction_supplemented"]
    value["temporal_direction_context_conflict"] = value["direction_context_conflict"]
    value["direction_supplemented"] = prior["direction_supplemented"] | value["temporal_direction_supplemented"]
    value["direction_context_conflict"] = prior["direction_context_conflict"] | value["temporal_direction_context_conflict"]
    return value


def load_temporal(selected):
    return CausalGRU.load(ROOT/selected["file"]) if selected["spec"]["architecture"] == "GRU" else load_model(ROOT/selected["file"])


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); start = time.monotonic()
    prior = prior_freeze(PRIOR_OUT); base, _ = load_frozen()
    strength = strength_freeze(ROOT/Path(prior["base_protocol"]).parent)
    strength_models = [load_model(ROOT/f) for f in prior["baseline_models"]]
    prior_model = load_model(ROOT/prior["selected"]["file"])
    declared = {"specs": SPECS, "seed": 282, "steps": 16, "step_s": .1, "past_horizon_s": 1.5,
        "epochs": 80, "batch": 256, "learning_rate": .001, "weight_decay": .001,
        "score_limits": (.75, .9, .97, .995), "confirm_s": (.015, .03), "signed_change": (False, True),
        "permit_residue": True, "preserve_all_R278_loading_withdrawal": True,
        "qualification": "each SOURCE validation Trial no added lift/control actions; identical strength; strictly lower trial-macro objective"}
    save_json(OUT/"declared_search.json", declared)
    records = json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"]
    records = [r for r in records if r["device"] == "source" and r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in records],
        "target_opened": 0, "source_test_opened": 0, "whole_trial": True})
    rows = []
    for i, rec in enumerate(records):
        row = load_source(rec, base["null"]["source"])
        _, row["prior"] = prior_infer(row, prior_model, prior, base, strength, strength_models)
        row["direction_y"] = action_annotations(row, "all_depth_action"); rows.append(row)
        if (i+1) % 10 == 0: print("R282 SOURCE sequence load", i+1, "/", len(records), flush=True)
    validation = [r for r in rows if r["split"] == "validation"]
    baseline_items = [score_prediction(r, r["prior"]) for r in validation]; baseline = aggregate(baseline_items)
    heads, candidates = [], []
    for spec in SPECS:
        tx, ty, tw = sequence_dataset(rows, spec["variant"]); vx, vy, vw = sequence_dataset(rows, spec["variant"], True)
        print("R282 train", spec["name"], "train/validation", len(ty), len(vy), "features", tx.shape[2], flush=True)
        model_start = time.monotonic()
        if spec["architecture"] == "GRU":
            model = CausalGRU(spec["width"], seed=282).fit(tx, ty, tw, (vx, vy, vw), epochs=80)
        else:
            tx = tx.reshape(len(tx), -1); vx = vx.reshape(len(vx), -1)
            model = StateMLP(spec["width"], seed=282, n_classes=3).fit(tx, ty, tw, (vx, vy, vw), epochs=80)
        file = OUT/f"direction_{spec['name']}.npz"; model.save(file)
        heads.append({"spec": spec, "epoch": model.best_epoch, "validation_ce": model.best_loss,
            "parameters": sum(int(p.size) for p in model.params), "history": model.history,
            "file": file.relative_to(ROOT).as_posix(), "elapsed_s": time.monotonic()-model_start,
            "train_counts": np.bincount(ty, minlength=3).tolist(), "validation_counts": np.bincount(vy, minlength=3).tolist()})
        scores = [predict_sequence(r, model, spec) for r in validation]
        for limit in declared["score_limits"]:
            for confirm in declared["confirm_s"]:
                for signed in declared["signed_change"]:
                    policy = asdict(SupplementPolicy(limit, confirm, True, signed))
                    items = [score_prediction(r, add_temporal(r, r["prior"], s, policy)) for r, s in zip(validation, scores)]
                    value = aggregate(items); qualified = qualifies(items, baseline_items); objective = policy_objective(value)
                    candidates.append({"spec": spec, "file": file.relative_to(ROOT).as_posix(), "policy": policy,
                        "validation": value, "objective": objective, "qualified": bool(qualified),
                        "accepted_improvement": bool(qualified and objective < policy_objective(baseline)-1e-4)})
        print("R282 SOURCE model", spec["name"], "epoch", model.best_epoch, "CE", round(model.best_loss, 6),
            "qualified improvements", sum(c["accepted_improvement"] for c in candidates), flush=True)
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R282-SourceCausalSequence-1", "heads": heads,
        "candidates": candidates, "baseline_R278_source_validation": baseline, "selected": selected,
        "accepted_improvement": bool(accepted), "elapsed_s": time.monotonic()-start,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    tracked = [Path(__file__), ROOT/"training/causal_response_sequence.py", ROOT/"training/independent_response_direction.py",
        OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json"]
    if selected: tracked.append(ROOT/selected["file"])
    inputs = [ROOT/r["path"]/name for r in records for name in RAW_FILES if (ROOT/r["path"]/name).is_file()]
    save_json(OUT/"freeze.json", {"schema": "R282-FrozenSourceCausalSequence-1", "frozen_before_target_replay": True,
        "accepted_improvement": bool(accepted), "selected": selected, "steps": 16, "step_s": .1,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "no_XYZ_arming": True, "strength_frozen_R276": True, "preserve_all_R278_loading_withdrawal": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in tracked],
        "source_input_manifest": [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha(p)} for p in inputs]})
    print("FROZEN R282 accepted", bool(accepted), "selected", None if not selected else
        {k: selected[k] for k in ("spec", "policy", "objective")}, flush=True)


if __name__ == "__main__": main()
