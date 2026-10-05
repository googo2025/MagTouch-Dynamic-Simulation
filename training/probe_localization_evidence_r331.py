"""Source-only provisional-position permission probe; no UI or hardware IO.

R330 stays immutable. An unshown candidate tracker reads every eligible magnetic
response, including weak observations. Source command references enter only
after causal feature extraction to label reliable calibrated-range candidates.
Extreme-shallow commands are NEVER XYZ/contact supervision. This executable
never opens Target files; the cache helper can be reused only by a separate
frozen-evaluation entry. The score is not physical-contact truth/confidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from training.audit_unified_tactile_r330 import ROOT, OUT as PRIOR, selected_records, create_freeze
from training.audit_r274_frozen_phase_expansion import load_record, save_json, RAW_FILES
from training.probe_r276_condition_adaptation import source_partition, invariant_features
from training.probe_r245_joint_contact_state import causal_state_features, StateMLP
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
from training.unified_response_runtime_r330 import digest

OUT = ROOT/"analysis/r331_source_localization_evidence_20260929"


def declaration():
    return {"schema": "R331-SourceLocalizationPermissionProbe-1",
        "parent": "immutable R330 integrated audit", "no_target_fit_or_selection": True,
        "records": "all 20 Source Trials from fixed R330 manifest; whole-Trial R276 partitions",
        "candidate": "frozen R133 magnetic decoder, causal raw stream; provisional, never a held stale point",
        "features": "31 causal amplitude-relative features + 17 magnetic-fit/local-Jacobian/candidate features + 7 strength and 6 phase one-hots",
        "positive": "Source calibrated-action reference z>=0.4 mm and candidate XY error<=0.8 mm, Z absolute error<=0.25 mm",
        "negative": "known safe-lift/protocol controls, or calibrated-range candidate failing position criteria",
        "ignored": "nominal z<0.4 mm and unknown reference are not XYZ or physical-contact labels",
        "scope": "a localization-permission prototype, not a new XYZ network or a contact classifier",
        "model_widths": [0, 24], "epochs": 100, "seed": 331,
        "score_limits": [.5, .7, .85, .9, .95, .975, .99, .995, .999],
        "confirmation_s": [.0, .03],
        "acceptance": "zero added Source validation lift/control outputs; added calibrated-range outputs >=95% meet XYZ criteria; any supported R330 XYZ preserved exactly",
        "selection": "maximum added reliable Source validation outputs among qualified candidates; smaller model then higher score limit on ties",
        "target_selection_rows": 0, "source_test_selection_rows": 0,
        "prospective_blind": False}


def ensure_declaration():
    OUT.mkdir(parents=True, exist_ok=True)
    file = OUT/"declared_protocol.json"; value = declaration()
    if file.exists():
        if json.loads(file.read_text(encoding="utf-8")) != value:
            raise ValueError("Declared source search changed")
    else: save_json(file, value)
    create_freeze(selected_records())


def geometric_features(atlas, field, xyz, memory, correction, basis, strength):
    """Current-field/candidate features only; no label or future observation."""
    finite = np.isfinite(xyz).all(1)
    point = np.where(finite[:, None], xyz, [0., 0., .4])
    raw = atlas.features(field)
    phi = raw-strength*memory@basis.T
    uncorrected = point.copy(); uncorrected[:, :2] -= correction
    model_phi = atlas.features(atlas.forward(uncorrected))
    final_phi = atlas.features(atlas.forward(point))
    residual = phi-model_phi
    jac = np.empty((len(point), 16, 3))
    for axis, step in enumerate((.05, .05, .02)):
        plus, minus = uncorrected.copy(), uncorrected.copy()
        plus[:, axis] += step; minus[:, axis] -= step
        jac[:, :, axis] = (atlas.features(atlas.forward(plus))-atlas.features(atlas.forward(minus)))/(2*step)
    singular = np.linalg.svd(jac, compute_uv=False)
    fit = np.linalg.norm(residual, axis=1)
    position_uncertainty_proxy = fit/np.maximum(singular[:, -1], 1e-4)
    # A residual/Jacobian quotient is a proxy, NOT a calibrated error bound.
    geometry = np.column_stack((
        point[:, 0]/10., point[:, 1]/10., point[:, 2]/3.,
        np.linalg.norm(point[:, :2], axis=1)/10.,
        fit, np.linalg.norm(phi-final_phi, axis=1),
        np.linalg.norm(raw-model_phi, axis=1),
        residual[:, -1], np.linalg.norm(memory, axis=1),
        np.linalg.norm(correction, axis=1),
        np.log(np.maximum(singular, 1e-4)),
        np.log(np.maximum(singular[:, 0]/np.maximum(singular[:, -1], 1e-4), 1.)),
        np.log1p(position_uncertainty_proxy),
        (point[:, 2] <= atlas.z[0]+.025).astype(float), finite.astype(float)))
    assert geometry.shape[1] == 17
    return np.clip(geometry, -20., 20.)


def cache(rec, prior_row):
    file = OUT/"cache"/(Path(prior_row["stream"]).stem+".npz")
    inputs = [{"path": (ROOT/rec["path"]/f).relative_to(ROOT).as_posix(),
               "sha256": digest(ROOT/rec["path"]/f)} for f in RAW_FILES]
    if file.exists():
        side = json.loads(file.with_suffix(".json").read_text(encoding="utf-8"))
        if side["raw_manifest"] != inputs: raise ValueError("Raw inputs changed")
        return side
    row = load_record(rec)
    with np.load(PRIOR/prior_row["stream"], allow_pickle=False) as a:
        t = a["time_ns"].copy(); eligible = a["eligible"].copy()
        strength, phase, responsive = a["strength"].copy(), a["phase"].copy(), a["response"].copy()
        groups, q = a["diagnostic_group"].copy(), a["q"].copy()
        base_valid, base_xyz = a["xyz_valid"].copy(), a["xyz_mm"].copy()
    np.testing.assert_array_equal(t, row["time"])
    decoder = CoverageGuardedNeuralDecoder.from_report()
    xyz = np.full((len(t), 3), np.nan); memory = np.zeros((len(t), 2)); correction = np.zeros((len(t), 2))
    started = time.monotonic()
    for i, (now, b, ok, response, state) in enumerate(zip(t, row["field_uT"], eligible, responsive, strength)):
        if not ok or not response:
            decoder.reset(); continue
        # Residual candidates are computed for negative training diagnostics,
        # but their state is never allowed to arm XYZ by the output rule.
        if state in (1, 5): decoder.reset()
        candidate = decoder.predict_frame(b, time_ns=now, b0_id=1, stream_id=0, contact_active=True)
        xyz[i] = candidate["xyz_mm"]; memory[i] = candidate["memory_state"]
        correction[i] = candidate["xy_neural_correction_mm"]
        if (i+1)%5000 == 0: print("candidate", rec["path"].split("/")[-1][:28], i+1, "/", len(t), flush=True)
    state_x = causal_state_features(t, row["field_uT"], eligible)
    geom = geometric_features(decoder.atlas, row["field_uT"], xyz, memory, correction, decoder.basis, decoder.strength)
    x = np.column_stack((invariant_features(state_x), geom, np.eye(7)[strength], np.eye(6)[phase]))
    assert x.shape == (len(t), 61) and np.isfinite(x).all()
    # Only after causal candidates/features: reference-based development labels.
    safe = np.isin(groups, ("pre_action_lift", "lift_residue_0to3s", "late_lift", "protocol_no_motion"))
    calibrated = np.isfinite(q).all(1) & (q[:, 2] >= .4)
    reliable = (calibrated & np.isfinite(xyz).all(1) &
        (np.linalg.norm(xyz[:, :2]-q[:, :2], axis=1) <= .8) & (np.abs(xyz[:, 2]-q[:, 2]) <= .25))
    y = np.full(len(t), -1, int); y[eligible & safe] = 0
    y[eligible & calibrated] = reliable[eligible & calibrated].astype(int)
    np.savez_compressed(file, time_ns=t, eligible=eligible, x=x, y=y, q=q, safe=safe,
        calibrated=calibrated, reliable=reliable, strength=strength, phase=phase, response=responsive,
        candidate_xyz_mm=xyz, base_valid=base_valid, base_xyz_mm=base_xyz)
    for item in inputs:
        if digest(ROOT/item["path"]) != item["sha256"]: raise ValueError("Raw Trial changed")
    result = {"path": rec["path"], "split": source_partition(rec) if rec["device"] == "source" else "target_evaluation", "mode": rec["mode"],
        "cache": file.relative_to(ROOT).as_posix(), "raw_manifest": inputs,
        "frames": len(t), "candidate_frames": int(np.isfinite(xyz).all(1).sum()),
        "label_counts": {str(k): int((y == k).sum()) for k in (-1, 0, 1)},
        "wall_seconds": time.monotonic()-started}
    save_json(file.with_suffix(".json"), result)
    return result


def records():
    return [r for r in selected_records() if r["device"] == "source"]


def dataset(caches):
    xx, yy, ww = [], [], []
    for c in caches:
        with np.load(ROOT/c["cache"], allow_pickle=False) as a:
            ids = np.flatnonzero(a["eligible"] & (a["y"] >= 0))[::4]
            x, y = a["x"][ids], a["y"][ids]
        if not len(y): continue
        weight = np.zeros(len(y))
        for k in (0, 1):
            mask = y == k
            if mask.any(): weight[mask] = 1./mask.sum()
        xx.append(x); yy.append(y); ww.append(weight)
    w = np.concatenate(ww); w *= len(w)/w.sum()
    return np.concatenate(xx), np.concatenate(yy), w


def permit(t, eligible, response, strength, score, candidate, limit, confirm_s):
    from training.unified_response_runtime_r330 import CausalConfirmation
    confirm = CausalConfirmation(confirm_s)
    result = np.zeros(len(t), bool)
    for i in range(len(t)):
        good = (eligible[i] and response[i] and strength[i] not in (1, 5) and
                np.isfinite(candidate[i]).all() and np.all(np.abs(candidate[i, :2]) <= 10.) and
                0 <= candidate[i, 2] <= 3. and score[i] >= limit)
        result[i] = bool(good) if confirm_s == 0 else confirm.step(int(t[i]), bool(good))
    return result


def evaluate(caches, model, limit, confirm_s):
    results = []
    for c in caches:
        with np.load(ROOT/c["cache"], allow_pickle=False) as a:
            new = permit(a["time_ns"], a["eligible"], a["response"], a["strength"],
                model.predict(a["x"])[:, 1], a["candidate_xyz_mm"], limit, confirm_s) & ~a["base_valid"]
            added = new & a["calibrated"]
            results.append({"path": c["path"], "added_calibrated": int(added.sum()),
                "added_reliable": int((added & a["reliable"]).sum()),
                "added_unreliable": int((added & ~a["reliable"]).sum()),
                "added_lift_control": int((new & a["safe"]).sum()),
                "added_unknown_or_shallow": int((new & ~a["safe"] & ~a["calibrated"]).sum())})
    total = {k: sum(r[k] for r in results) for k in results[0] if k != "path"}
    total["reliable_fraction"] = total["added_reliable"]/max(total["added_calibrated"], 1)
    total["qualified"] = bool(total["added_lift_control"] == 0 and total["added_calibrated"] > 0 and
                               total["reliable_fraction"] >= .95)
    return {"summary": total, "trials": results}


def train():
    recs = records()
    caches = [json.loads((OUT/"cache"/(Path(r["stream"]).stem+".json")).read_text(encoding="utf-8"))
              for r in json.loads((PRIOR/"report_source.json").read_text(encoding="utf-8"))["rows"]
              if source_partition(next(c for c in recs if c["path"] == r["path"])) != "test"]
    tr = [c for c in caches if c["split"] == "train"]; va = [c for c in caches if c["split"] == "validation"]
    tx, ty, tw = dataset(tr); vx, vy, vw = dataset(va)
    if (OUT/"freeze.json").exists(): raise FileExistsError("R331 selection already frozen")
    candidates = []; specs = []
    for width in declaration()["model_widths"]:
        model = StateMLP(width, seed=331, n_classes=2).fit(tx, ty, tw, (vx, vy, vw), epochs=100)
        file = OUT/f"permission_width{width}.npz"; model.save(file)
        specs.append({"width": width, "epoch": model.best_epoch, "loss": model.best_loss,
                      "file": file.relative_to(ROOT).as_posix()})
        for limit in declaration()["score_limits"]:
            for confirm in declaration()["confirmation_s"]:
                val = evaluate(va, model, limit, confirm)
                candidates.append({"width": width, "score_limit": limit, "confirmation_s": confirm, "validation": val})
        print("Source permission fit", specs[-1], flush=True)
    qualified = [c for c in candidates if c["validation"]["summary"]["qualified"]]
    selected = min(qualified, key=lambda c: (-c["validation"]["summary"]["added_reliable"], c["width"], -c["score_limit"], c["confirmation_s"])) if qualified else None
    report = {"models": specs, "selected": selected, "candidates": candidates,
        "train_samples": len(tx), "validation_samples": len(vx), "source_trials": caches,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "scope": "provisional candidate permission; selected validation not blind; no UI update"}
    save_json(OUT/"training_report.json", report)
    selected_file = next(s["file"] for s in specs if selected is not None and s["width"] == selected["width"]) if selected else None
    paths = [Path(__file__), OUT/"declared_protocol.json", OUT/"training_report.json",
             PRIOR/"freeze.json", ROOT/"training/probe_r245_joint_contact_state.py"]+[ROOT/s["file"] for s in specs]
    save_json(OUT/"freeze.json", {"schema": "R331-FrozenSourcePermission-1", "selected": selected,
        "model": selected_file, "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "frozen_before_target_replay": True, "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
        "source_feature_caches": [{"path": c["cache"], "sha256": digest(ROOT/c["cache"])} for c in caches]})
    print("FROZEN source candidate permission", None if selected is None else selected["validation"]["summary"], flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("cache", "train"), default="cache")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    args = parser.parse_args(); ensure_declaration(); (OUT/"cache").mkdir(exist_ok=True)
    if args.stage == "train": return train()
    old_rows = {r["path"]: r for r in json.loads((PRIOR/"report_source.json").read_text(encoding="utf-8"))["rows"]}
    recs = [r for r in records() if args.split == "all" or source_partition(r) == args.split]
    for i, r in enumerate(recs):
        side = cache(r, old_rows[r["path"]]); print(f'{i+1}/{len(recs)} {side["split"]} {Path(r["path"]).name[:35]} candidate={side["candidate_frames"]} sec={side["wall_seconds"]:.1f}', flush=True)


if __name__ == "__main__": main()
