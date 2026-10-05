"""Source-only depth-policy development followed by frozen cross-device audit.

Saved online XY and validity remain unchanged: this isolates a Z-decoder
ablation, not a full integrated R332 replay or a new online experiment.
No Target data are opened by `select`; all fitting/selection uses six declared
whole Source Trials. Extreme-shallow commands are not depth/contact labels.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd

from training.probe_had239_local_forward_r104 import ROOT, sha
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
from training.retrain_r133_coverage_xy import load_new
from training.build_trajectory_dataset_v1 import command_reference, command_segments, read_rows, reference_origin
from training.analyze_r332_online_acceptance_r333 import movement_mask, scoring, runs
from training.z_evidence_runtime_r334 import DepthPolicy, MagneticDepthTracker, depth_profiles

OUT = ROOT/"analysis/r334_z_evidence_20260929"
SOURCE_D = ROOT/"data/S20260927_A/Plan_20260927_213250_R133_D_HAD239_COVERAGE_TRAIN__mag_5x5-had-239"
VALIDATION_IDS = (90, 93, 96, 104, 108, 109)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def declaration():
    return {"schema": "MagTouch-R334-SourceDepthEvidenceProtocol-1",
        "validation_source_trials": list(VALIDATION_IDS),
        "selection": "six predeclared complete Source D Trials: three depths of circles, XYZ ramp, nonmonotonic Z, static grid",
        "candidate_memory_strength": [0., .25, .5, 1.], "candidate_xy": ["guard", "final"],
        "candidate_continuity": [0., .002, .01, .03], "field_veto_ratio": 2.,
        "objective": "macro(Z centered MAE + .2 raw MAE + .4 adjacent-step P95 + .1 max step) at nominal >=0.4",
        "qualification": "no static grid or real-Z-changing Trial raw/centered MAE worsening >0.03 mm",
        "basis": "frozen Source R122 two-mode magnetic memory; no memory refit",
        "target_fit_rows": 0, "target_selection_rows": 0,
        "preserved": "recorded XY and validity, every accepted frame, no template, no coordinate EMA",
        "reference": "command action/reference only, not enforced XYZ at extreme shallow depths",
        "prospective_blind": False, "new_online_test": False,
        "scope": "Z-only conditional-field ablation; saved XY/history with unchanged permissions, not full engine replay"}


def prepare(path, decoder):
    data = load_new(path, decoder.atlas, decoder.basis, decoder.strength, min_core_frames=0)
    seg = command_segments(read_rows(path/"stage_commands.csv")) if (path/"stage_commands.csv").stat().st_size > 100 else []
    if seg:
        ref = command_reference(data["time"], seg, reference_origin(data["meta"]))
        phase = np.array([seg[j]["phase"] if j >= 0 else "before_stage" for j in ref["command_segment_id"]])
        if data["mode"] == "waypoint_xyz":
            action = np.array([bool(re.fullmatch(r"waypoint_\d{3}", p)) for p in phase])
        else:
            action = movement_mask(data["mode"], phase)
        known = np.isfinite(ref["xyz_mm"]).all(1)
        action &= known
    else:
        phase = np.full(len(data["time"]), "none"); action = np.zeros(len(phase), bool)
        ref = {"xyz_mm": np.full((len(phase), 3), np.nan), "command_moving": np.zeros(len(phase), bool),
               "command_segment_id": np.full(len(phase), -1)}
    f = pd.read_csv(path/"live_predictions.csv", usecols=["extra_outputs_json"])
    extra = [json.loads(s) for s in f.extra_outputs_json]
    epochs = np.array([e.get("baseline_epoch", 0) for e in extra])
    data["final_xy"] = np.array([e.get("unaligned_xyz_mm", data["live"][i].tolist())[:2]
                                 for i, e in enumerate(extra)])
    data["guard_xy"] = data["final_xy"]-data["logged_correction"]
    # The amplitude cross-check exposes B0 reconstruction error; it is not a
    # score-based frame filter. Historical baseline windows remain causal.
    expected = np.array([e.get("amplitude_uT", e.get("response_observation", {}).get("amplitude_uT", np.nan)) for e in extra])
    good = data["valid"] & np.isfinite(expected)
    delta = np.abs(np.linalg.norm(data["observed"][good], axis=1)-expected[good])
    data.update(q=data["truth"], xyz=data["live"], action=action,
                calibrated_action=action & (data["truth"][:, 2] >= .4),
                phase=phase, moving=ref["command_moving"], segment=ref["command_segment_id"], epoch=epochs,
                b0_amplitude_check={"n": int(len(delta)), "median_uT": float(np.median(delta)) if len(delta) else None,
                                    "p95_uT": float(np.quantile(delta, .95)) if len(delta) else None})
    data["constant"] = action & np.isclose(data["truth"][:, 2], data["meta"]["trial_parameters"]["z_indent_mm"], atol=1e-5)
    return data


def summary(data, xyz):
    a = {"q": data["q"], "xyz": xyz, "valid": data["valid"]}
    out = scoring(a, data["calibrated_action"])
    dt = np.diff(data["time"])/1e9
    pairs = (data["calibrated_action"][1:] & data["calibrated_action"][:-1] &
             data["valid"][1:] & data["valid"][:-1] & (dt > 0) & (dt <= .25))
    # Residual steps remove real prescribed depth changes from this statistic.
    dz = np.abs(np.diff(xyz[:, 2]-data["q"][:, 2]))
    if pairs.any():
        out.update(z_step_p95_mm=float(np.quantile(dz[pairs], .95)),
                   z_step_max_mm=float(dz[pairs].max()), z_over02=int((dz[pairs] > .2).sum()),
                   z_pair_count=int(pairs.sum()))
    if data["mode"] in ("circle_scan", "angle_scan"):
        c = data["constant"]
        cp = c[1:] & c[:-1] & pairs
        out["constant_depth"] = scoring(a, c)
        if cp.any():
            out["constant_depth"].update(z_step_p95_mm=float(np.quantile(dz[cp], .95)),
                z_step_max_mm=float(dz[cp].max()), z_over02=int((dz[cp] > .2).sum()), z_pair_count=int(cp.sum()))
    return out


def profiles(data, decoder, memory, position):
    good = data["valid"]
    policy = DepthPolicy(memory_strength=memory, position_source=position)
    z, cost, phi = depth_profiles(decoder.atlas, data["observed"][good],
        data[position+"_xy"][good], data["memory"][good], decoder.basis, policy)
    return z, cost, phi


def apply_profiles(data, profile, policy):
    z, cost, phi = profile
    tracker = MagneticDepthTracker(policy)
    out = data["live"].copy(); cursor = 0
    chosen = np.zeros(len(out), bool); veto = chosen.copy()
    for i, (t, good) in enumerate(zip(data["time"], data["valid"])):
        if not good:
            tracker.reset(); continue
        result = tracker.step_profile(t, z, cost[cursor], phi[cursor], token=int(data["epoch"][i]))
        out[i, 2] = result["z_mm"]
        chosen[i], veto[i] = result["used_continuity"], result["field_veto"]
        cursor += 1
    assert cursor == len(cost)
    np.testing.assert_array_equal(out[:, :2], data["live"][:, :2])
    return out, chosen, veto


def objective(row):
    if not row.get("output_frames", 0): return float("inf")
    return (row["z_centered_mae_mm"]+.2*row["z_raw_mae_mm"]+
            .4*row.get("z_step_p95_mm", 0.)+.1*row.get("z_step_max_mm", 0.))


def source_validation():
    return [next(SOURCE_D.glob(f"Trial_{n:06d}_*")) for n in VALIDATION_IDS]


def select():
    OUT.mkdir(exist_ok=True)
    file = OUT/"declared_protocol.json"; declared = declaration()
    if file.exists():
        assert json.loads(file.read_text(encoding="utf-8")) == declared
    else: dump(file, declared)
    decoder = CoverageGuardedNeuralDecoder.from_report()
    values = [prepare(p, decoder) for p in source_validation()]
    base = [summary(d, d["live"]) for d in values]
    print("Source validation baseline", [(p.name[:12], round(r["z_raw_mae_mm"], 3), round(r["z_centered_mae_mm"], 3)) for p, r in zip(source_validation(), base)], flush=True)
    candidates = []
    for memory in declared["candidate_memory_strength"]:
        for position in declared["candidate_xy"]:
            current = [profiles(d, decoder, memory, position) for d in values]
            for continuity in declared["candidate_continuity"]:
                policy = DepthPolicy(memory_strength=memory, position_source=position, continuity=continuity)
                rows = [summary(d, apply_profiles(d, p, policy)[0]) for d, p in zip(values, current)]
                qualified = all(rows[j][key] <= base[j][key]+.03 for j in (3, 4, 5)
                                for key in ("z_raw_mae_mm", "z_centered_mae_mm"))
                value = float(np.mean([objective(r) for r in rows]))
                candidates.append({"policy": asdict(policy), "qualified": qualified, "objective": value, "trials": rows})
                print("R334 Source", memory, position, continuity, round(value, 4), qualified, flush=True)
    baseline_objective = float(np.mean([objective(r) for r in base]))
    qualified = [c for c in candidates if c["qualified"] and c["objective"] < baseline_objective]
    selected = min(qualified, key=lambda c: c["objective"]) if qualified else None
    result = {"schema": "R334-SourceDepthSelection-1", "baseline_objective": baseline_objective,
        "baseline": base, "candidates": candidates, "selected": selected,
        "target_fit_rows": 0, "target_selection_rows": 0,
        "raw_manifest": [{"path": (d["path"]/name).relative_to(ROOT).as_posix(), "sha256": sha(d["path"]/name)}
                         for d in values for name in ("meta.json", "live_predictions.csv", "magnetic_raw.csv", "events.csv", "stage_commands.csv")],
        "b0_amplitude_checks": [d["b0_amplitude_check"] for d in values]}
    dump(OUT/"source_selection.json", result)
    if selected:
        dump(OUT/"freeze.json", {"schema": "R334-FrozenSourceDepthPolicy-1", "policy": selected["policy"],
            "selection_sha256": sha(OUT/"source_selection.json"), "runtime_sha256": sha(ROOT/"training/z_evidence_runtime_r334.py"),
            "target_fit_rows": 0, "target_selection_rows": 0, "prospective_blind": False})
    print("SELECTED", None if selected is None else selected["policy"], "baseline", baseline_objective, flush=True)


def records(device):
    result = []
    for path in sorted((ROOT/"data").glob("S2026092[789]*")):
        for p in sorted(path.rglob("meta.json")):
            is_target = "Yelow" in str(p)
            if is_target != (device == "target"): continue
            # Declared complete holdout cohorts, not favourable individual Trials.
            keep = ("R133_E_" in str(p) or "R133_F_" in str(p) or "R133_RETRAIN_" in str(p)
                    or "R136_" in str(p) or "R208_" in str(p) or "R332_" in str(p))
            if not keep: continue
            m = json.loads(p.read_text(encoding="utf-8"))
            mode = m.get("trial_parameters", {}).get("command_mode")
            if mode not in ("angle_scan", "circle_scan", "grid_points", "compound", "z_path", "waypoint_xyz"): continue
            if not (p.parent/"live_predictions.csv").exists(): continue
            result.append(p.parent)
    return result


def evaluate(device):
    freeze = json.loads((OUT/"freeze.json").read_text(encoding="utf-8"))
    assert freeze["runtime_sha256"] == sha(ROOT/"training/z_evidence_runtime_r334.py")
    assert freeze["selection_sha256"] == sha(OUT/"source_selection.json")
    policy = DepthPolicy(**freeze["policy"])
    decoder = CoverageGuardedNeuralDecoder.from_report()
    rows, errors = [], []
    for p in records(device):
        try:
            d = prepare(p, decoder)
            ph = profiles(d, decoder, policy.memory_strength, policy.position_source)
            xyz, continuity, veto = apply_profiles(d, ph, policy)
            old, new = summary(d, d["live"]), summary(d, xyz)
            stem = sha(p/"meta.json")[:16]
            np.savez_compressed(OUT/"streams"/(stem+".npz"), time=d["time"], q=d["q"], valid=d["valid"],
                old=d["live"], new=xyz, action=d["action"], calibrated=d["calibrated_action"],
                constant=d["constant"], phase=d["phase"], continuity=continuity, veto=veto)
            rows.append({"path": p.relative_to(ROOT).as_posix(), "device": device,
                "trial": int(d["meta"]["trial_parameters"]["trial_id"]), "mode": d["mode"],
                "stream": (OUT/"streams"/(stem+".npz")).relative_to(ROOT).as_posix(),
                "old": old, "new": new, "b0_amplitude_check": d["b0_amplitude_check"],
                "xy_and_validity_unchanged": True, "continuity_frames": int(continuity.sum()), "field_veto_frames": int(veto.sum()),
                "raw_manifest": [{"path": (p/f).relative_to(ROOT).as_posix(), "sha256": sha(p/f)}
                                 for f in ("meta.json", "live_predictions.csv", "magnetic_raw.csv", "events.csv", "stage_commands.csv")]})
            print(device, p.name[:12], d["mode"], round(old.get("z_centered_mae_mm",0),3),
                "->", round(new.get("z_centered_mae_mm",0),3), flush=True)
        except (ValueError, KeyError, StopIteration) as e:
            errors.append({"path": str(p.relative_to(ROOT)), "error": str(e)})
            print("UNAVAILABLE", p.name[:12], str(e), flush=True)
    metrics = {}
    for key in ("z_raw_mae_mm", "z_centered_mae_mm", "z_step_p95_mm", "z_step_max_mm"):
        use = [r for r in rows if key in r["old"] and key in r["new"]]
        metrics[key] = {k: float(np.mean([r[k][key] for r in use])) for k in ("old", "new")} if use else None
    dump(OUT/f"report_{device}.json", {"schema": "R334-FrozenDepthAblation-1", "policy": freeze,
        "new_online_test": False, "full_r332_replayed": False, "target_fit_rows": 0,
        "target_selection_rows": 0, "rows": rows, "unavailable": errors, "trial_macro": metrics})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("task", choices=("select", "source", "target"))
    args = parser.parse_args(); OUT.mkdir(exist_ok=True); (OUT/"streams").mkdir(exist_ok=True)
    if args.task == "select": select()
    else: evaluate(args.task)


if __name__ == "__main__": main()
