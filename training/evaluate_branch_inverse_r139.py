#!/usr/bin/env python3
"""Replay branch-aware magnetic inverse on recorded command-labeled Trials.

No acquisition/UI imports.  These S26 Trials have already been inspected, so
this is development evidence, not independent prospective validation.
"""
from __future__ import annotations

import json

import numpy as np

from training.branch_aware_inverse_r138 import BranchAwareInverse
from training.build_trajectory_dataset_v1 import (
    assemble, command_segments, load_magnetic, read_rows, reference_origin,
)
from training.diagnose_r126_latest_four_live_r130 import circle_shape
from training.fit_had239_old1mm_field_memory_r131 import (
    OUT as OLD_REPORT, PHYSICAL_ID as OLD_ID, old1mm_atlas,
)
from training.probe_had239_field_memory_r122 import (
    causal_memory, feature_to_field, observed_memory,
)
from training.probe_had239_local_forward_r104 import (
    ForwardAtlas, ROOT, load_profile, score, sha,
)

OUT = ROOT / "analysis/r139_branch_inverse_20260926/report_v2.json"
FIVE_CONFIG = ROOT / "config/had239_legacy_formal_continuous_r92_20260926.json"
FIVE_KEY = "5x5_had239_final5mm_old_formal_r92"
FIVE_ID = "magnetic_layer_v1_5*5-had-239"
FIVE_REPORT = ROOT / "analysis/r122_had239_field_memory_20260926/report.json"


def trial(number: int, physical_id: str, training_contact_z: float,
          atlas: ForwardAtlas) -> dict:
    matches = list((ROOT / "data/S20260926_A").glob(f"Trial_{number:06d}_*"))
    if len(matches) != 1:
        raise ValueError(matches)
    path = matches[0]
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta["status"] != "completed" or meta["acquisition"][
            "magnetic_layer_profile_id"] != physical_id:
        raise ValueError(f"wrong physical assembly {path}")
    data, _ = assemble(load_magnetic(path / "magnetic_raw.csv"),
                       command_segments(read_rows(path / "stage_commands.csv")),
                       read_rows(path / "events.csv"), reference_origin(meta))
    choose = data["input_valid"].astype(bool) & (
        np.arange(len(data["t_ns"])) % 4 == 0)
    observed = data["delta_b_uT"][choose].astype(float)
    truth = data["xyz_mm"][choose].astype(float)
    truth[:, 2] += training_contact_z - float(reference_origin(meta)[2])
    supported = (np.isfinite(truth).all(axis=1) &
                 (truth[:, 2] >= atlas.z[0] - 1e-5) &
                 (truth[:, 2] <= atlas.z[-1] + 1e-5) &
                 (np.abs(truth[:, :2]) <= 10 + 1e-5).all(axis=1))
    supervised = data["xyz_train_mask"][choose].astype(bool) & supported
    moving = supervised & data["command_moving"][choose].astype(bool)
    hold = (supervised & ~moving &
            (data["hold_age_s"][choose] >= .2) &
            (data["contact_command"][choose] == 1))
    base = atlas.predict(observed)
    phi = atlas.features(observed)
    feature = np.column_stack((base, phi, phi - atlas.features(atlas.forward(base))))
    return {"trial": number, "family": meta["trial_parameters"]["command_mode"],
            "source": path.relative_to(ROOT).as_posix(),
            "raw_sha256": sha(path / "magnetic_raw.csv"),
            "feature": feature, "base": base, "truth": truth,
            "moving": moving, "hold": hold, "time_ns": data["t_ns"][choose],
            "b0_id": data["b0_id"][choose],
            "stream_id": data["stream_id"][choose],
            "contact": data["contact_command"][choose] == 1}


def corrected_fields(row: dict, atlas: ForwardAtlas, report: dict) -> np.ndarray:
    basis = np.asarray(report["basis"], float)
    selected = report["selection"]["selected"]
    h_obs = observed_memory(atlas, basis, row["feature"], row["base"])
    state = causal_memory(h_obs, row["time_ns"], row["b0_id"],
                          row["stream_id"], float(selected["retention"]))
    phi = row["feature"][:, 3:19] - float(selected["strength"]) * (
        state @ basis.T)
    return feature_to_field(atlas, phi)


def jump_count(pred: np.ndarray, t_ns: np.ndarray, moving: np.ndarray) -> dict:
    eligible = (moving[1:] & moving[:-1] &
                (np.diff(t_ns) > 0) & (np.diff(t_ns) < 150_000_000))
    jump = np.linalg.norm(np.diff(pred[:, :2], axis=0), axis=1)[eligible]
    return {"pairs": int(len(jump)), "over_1mm": int((jump > 1).sum()),
            "over_2mm": int((jump > 2).sum()),
            "max_mm": float(jump.max()) if len(jump) else None}


def run_case(row: dict, atlas: ForwardAtlas, field: np.ndarray,
             global_xyz: np.ndarray, penalty: float, innovation_gain: float) -> dict:
    tracker = BranchAwareInverse(atlas, spatial_penalty=penalty,
                                 innovation_gain=innovation_gain)
    estimate = np.full_like(global_xyz, np.nan)
    used_local = 0
    for i in range(len(field)):
        out = tracker.predict_frame(
            field[i], time_ns=int(row["time_ns"][i]),
            b0_id=int(row["b0_id"][i]),
            stream_id=int(row["stream_id"][i]),
            contact_active=bool(row["contact"][i]), global_xyz=global_xyz[i])
        if out is not None:
            estimate[i] = out["xyz_mm"]
            used_local += int(out["used_local_branch"])
    if not np.isfinite(estimate[row["moving"]]).all():
        raise ValueError(f"missing moving estimates in Trial {row['trial']}")
    result = {"moving": score(estimate[row["moving"]], row["truth"][row["moving"]]),
              "jumps": jump_count(estimate, row["time_ns"], row["moving"]),
              "used_local_frames": used_local}
    if row["hold"].any() and np.isfinite(estimate[row["hold"]]).all():
        result["hold"] = score(estimate[row["hold"]], row["truth"][row["hold"]])
    if row["family"] == "circle_scan":
        result["fourfold_mm"] = circle_shape(
            estimate[row["moving"]], row["truth"][row["moving"]]
        )["fourfold_amplitude_mm"]
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    five_profile = load_profile(FIVE_CONFIG, FIVE_KEY)
    samples = five_profile["samples"]
    xyz = np.asarray([[s["x"], s["y"], s["depth"]] for s in samples], float)
    delta = np.asarray([s["delta"] for s in samples], float)
    five_atlas = ForwardAtlas(xyz, delta, "unit_logamp")
    old_atlas, _ = old1mm_atlas()
    report = {"schema": "MagTouch-R139-BranchInverseDevelopment-1",
              "warning": "S26 already-viewed command trajectories; not prospective validation or measured tip position",
              "domains": {}}
    for name, atlas, config, model_path, physical_id, contact_z, numbers in (
            ("had239_5mm", five_atlas, FIVE_CONFIG, FIVE_REPORT,
             FIVE_ID, 23.70, (11, 12, 13, 14, 15)),
            ("had239_old1mm", old_atlas,
             ROOT / "config/had239_old1mm_121pt_continuous_augmented_r65_20260924.json",
             OLD_REPORT, OLD_ID, 16.0, (1, 3, 4, 5, 6, 7, 9, 10))):
        model = json.loads(model_path.read_text(encoding="utf-8"))
        rows = []
        for number in numbers:
            row = trial(number, physical_id, contact_z, atlas)
            field = corrected_fields(row, atlas, model)
            global_xyz = atlas.predict(field)
            cases = {}
            for penalty, gain in ((0.0, 2.0), (.01, 2.0),
                                  (.03, 2.0), (.1, 2.0),
                                  (.3, 2.0)):
                key = f"penalty_{penalty:g}_gain_{gain:g}"
                cases[key] = run_case(row, atlas, field, global_xyz, penalty, gain)
                metric = cases[key]["moving"]
                print("[R139]", name, number, key,
                      round(metric["xy_mean_mm"], 3),
                      cases[key]["jumps"]["over_1mm"], flush=True)
            rows.append({"trial": number, "family": row["family"],
                         "source": row["source"], "raw_sha256": row["raw_sha256"],
                         "n_moving": int(row["moving"].sum()),
                         "n_hold": int(row["hold"].sum()), "cases": cases})
        report["domains"][name] = {"config": str(config.relative_to(ROOT)),
                                    "config_sha256": sha(config),
                                    "model_report": str(model_path.relative_to(ROOT)),
                                    "model_sha256": sha(model_path),
                                    "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
