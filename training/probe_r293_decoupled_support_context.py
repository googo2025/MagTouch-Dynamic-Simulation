"""SOURCE-qualified support release without clobbering directional evidence.

Strength confidence and action direction are distinct. Confirmed magnetic
residual context may revoke support, while existing loading/withdrawal
phases remain observations of magnetic change, not contact/XYZ permission.
"""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
from training.probe_r291_supported_context_release import (
    ROOT, INVENTORY, load_prior_args, prior_infer, protected_veto, load_source,
    source_partition, VAL_GAINS, TAU_S, augmented_signal_row, predict_sequence,
    load_model, score_prediction, aggregate, RAW_FILES, save_json, sha,
    release_supported, safety, control_support, support_summary)
from training.probe_r291_supported_context_release import OUT as PRIOR_OUT
from training.probe_r288_ordered_action_context import OUT as CONTEXT_OUT
from training.verify_r291_supported_context_release import verify_freeze as prior_freeze
from training.verify_r288_ordered_action_context import verify_freeze as context_freeze

OUT = ROOT/"analysis/r293_source_decoupled_support_context_20260929"


def decoupled_release(row, prior, scores, limit=.995, confirm_s=.03):
    # The R291 confirmation is causal, reset-aware and independent of labels.
    result = release_supported(row, prior, scores, limit, confirm_s)
    changed = result["supported_context_released"]
    direction = changed & np.isin(prior["phase"], (2, 4))
    phase = result["phase"].copy(); phase[direction] = prior["phase"][direction]
    # Do not turn an existing unknown phase into a certain action/context.
    phase[prior["phase"] == 0] = 0; phase[~row["eligible"]] = 0
    return dict(result, phase=phase, contextual_support_released=changed,
                direction_preserved_on_release=direction)


def protection(row, old, new):
    c = safety(row, old, new); a, b = control_support(row, old), control_support(row, new)
    support_ok = b["supported_control_frames"] <= a["supported_control_frames"]
    c["supported_control_nonincrease"] = bool(support_ok)
    c["qualified"] = bool(c["qualified"] and support_ok)
    return c


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir(parents=True); started = time.monotonic()
    declared = {"limits": (.995, .999), "confirm_s": (.03, .1), "fit_rows": 0,
        "validation_gains": VAL_GAINS, "tau_s": TAU_S,
        "qualification": "every native SOURCE train/validation and validation gain Trial: no lost deep supported state/action, exact correct weak frames and episode hits retained, no added lift/control action, control-supported frames nonincrease; strict mean-gain control-support reduction",
        "selection": "lowest SOURCE validation mean-gain Trial-macro control-supported fraction",
        "direction_preservation": "retain only already-existing loading/withdrawal phase 2/4 on strength release; no new action is created",
        "semantics": "magnetic directional evidence remains possible in residue; action direction is not physical contact nor reliable XYZ",
        "no_XYZ_arming": True, "base_models_unchanged": True, "target_and_retained_SOURCE_selection": 0}
    save_json(OUT/"declared_search.json", declared)
    p = prior_freeze(PRIOR_OUT); context = context_freeze(CONTEXT_OUT)
    model = load_model(ROOT/context["selected"]["file"]); args, base = load_prior_args()
    def prepare(row):
        _, before = prior_infer(row, *args); s = context["selected"]
        scores = predict_sequence(row, model, s["spec"])
        old = protected_veto(row, before, scores, s["limit"], s["confirm_s"])
        row["prior"] = release_supported(row, old, scores, p["selected"]["limit"], p["selected"]["confirm_s"])
        row["context_scores"] = scores
        return row
    recs = [r for r in json.loads(INVENTORY.read_text(encoding="utf-8"))["trials"] if r["device"] == "source" and
        r["date"] >= "S20260926_A" and r["initially_readable"] and source_partition(r) != "test"]
    save_json(OUT/"source_partition.json", {"records": [{"path": r["path"], "split": source_partition(r)} for r in recs],
        "whole_Trial": True, "target_opened": 0, "source_test_opened": 0})
    inputs = [{"path": (ROOT/r["path"]/f).relative_to(ROOT).as_posix(), "sha256": sha(ROOT/r["path"]/f)} for r in recs for f in RAW_FILES]
    rows = []
    for i, rec in enumerate(recs):
        rows.append(prepare(load_source(rec, base["null"]["source"])))
        if (i+1)%10 == 0: print("R293 SOURCE context/direction", i+1, "/", len(recs), flush=True)
    validation = {str(g): [r if g == 1. else prepare(augmented_signal_row(r, g, base["null"]["source"], TAU_S))
        for r in rows if r["split"] == "validation"] for g in VAL_GAINS}
    baseline = {g: support_summary(vr, [r["prior"] for r in vr]) for g, vr in validation.items()}
    previous = float(np.mean([v["trial_macro_fraction"] for v in baseline.values()]))
    native_before = support_summary(rows, [r["prior"] for r in rows]); candidates = []
    for limit in declared["limits"]:
        for confirm in declared["confirm_s"]:
            native = [decoupled_release(r, r["prior"], r["context_scores"], limit, confirm) for r in rows]
            nc = [dict(protection(r, r["prior"], n), path=str(r["path"])) for r, n in zip(rows, native)]
            reports, controls, gc = {}, {}, []
            for g, vr in validation.items():
                preds = [decoupled_release(r, r["prior"], r["context_scores"], limit, confirm) for r in vr]
                reports[g] = aggregate([score_prediction(r, n) for r, n in zip(vr, preds)])
                controls[g] = support_summary(vr, preds)
                gc += [dict(protection(r, r["prior"], n), path=str(r["path"]), gain=g) for r, n in zip(vr, preds)]
            obj = float(np.mean([v["trial_macro_fraction"] for v in controls.values()])); safe = all(c["qualified"] for c in nc+gc)
            candidates.append({"limit": limit, "confirm_s": confirm, "objective": obj, "previous_objective": previous,
                "qualified_protection": bool(safe), "gain_control_support": controls, "gain_validation": reports,
                "native_control_support": support_summary(rows, native),
                "native_direction_preserved_on_release_frames": sum(int(n["direction_preserved_on_release"].sum()) for n in native),
                "native_failures": [c for c in nc if not c["qualified"]], "gain_failures": [c for c in gc if not c["qualified"]],
                "accepted_improvement": bool(safe and obj < previous-1e-8)})
    accepted = [c for c in candidates if c["accepted_improvement"]]
    selected = min(accepted, key=lambda c: c["objective"]) if accepted else None
    save_json(OUT/"training_report.json", {"schema": "R293-DecoupledSupportContext-1", "selected": selected,
        "accepted_improvement": bool(accepted), "candidates": candidates, "baseline_gain_control_support": baseline,
        "baseline_native_control_support": native_before, "elapsed_s": time.monotonic()-started,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0, "new_model_fitting": False})
    tracked = [Path(__file__), OUT/"declared_search.json", OUT/"source_partition.json", OUT/"training_report.json"]
    for r in inputs: assert sha(ROOT/r["path"]) == r["sha256"]
    save_json(OUT/"freeze.json", {"schema": "R293-FrozenDecoupledSupportContext-1", "selected": selected,
        "accepted_improvement": bool(accepted), "frozen_before_target_replay": True,
        "base_protocol": (PRIOR_OUT/"freeze.json").relative_to(ROOT).as_posix(), "base_protocol_sha256": sha(PRIOR_OUT/"freeze.json"),
        "no_XYZ_arming": True, "base_models_unchanged": True, "strength_only_contextual_release": True,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "tracked": [{"path": f.relative_to(ROOT).as_posix(), "sha256": sha(f)} for f in tracked], "source_input_manifest": inputs})
    print("FROZEN R293", bool(accepted), "SOURCE native controls", native_before["supported_control_frames"],
        "selected", None if selected is None else {k: selected[k] for k in ("limit", "confirm_s", "objective")}, flush=True)


if __name__ == "__main__": main()
