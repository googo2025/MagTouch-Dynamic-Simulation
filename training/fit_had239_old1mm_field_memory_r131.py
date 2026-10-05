#!/usr/bin/env python3
"""R131 old-1mm had-239 field-memory fit; never opens acquisition hardware.

Train only on 2026-09-24 old-1mm Trials 5--7, select coefficients on Trial 8,
and audit 2026-09-26 Trials 3--8 without using them for fitting/selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.build_trajectory_dataset_v1 import (
    assemble, command_segments, load_magnetic, read_rows, reference_origin,
)
from training.probe_had239_local_forward_r104 import (
    ForwardAtlas, ROOT, load_profile, score, sha,
)
from training.probe_had239_field_memory_r122 import (
    RANK, causal_memory, corrected, memory_basis, observed_memory,
)

STATIC_CONFIG = ROOT / "config/had239_old1mm_121pt_continuous_augmented_r65_20260924.json"
STATIC_KEY = "5x5_had239_old1mm_121pt_continuous_augmented_r65"
GATE_CONFIG = ROOT / "config/had239_legacy_formal_continuous_r92_20260926.json"
GATE_KEY = "5x5_had239_old1mm_old_formal_r92"
PHYSICAL_ID = "magnetic_layer_v1_5*5-had-old1mmTouch"
OUT = ROOT / "analysis/r131_had239_old1mm_field_memory_20260926/report_v3.json"
FULL_LEVELS = (.2, .5, .8, 1.2, 1.6)
TRAIN = (("S20260924_A", 5), ("S20260924_A", 6), ("S20260924_A", 7))
VALIDATION = (("S20260924_A", 8),)
TEST = tuple(("S20260926_A", n) for n in (3, 4, 5, 6, 7, 8))
STATIC_AUDIT = (("S20260926_A", 1), ("S20260926_A", 9))
RETENTIONS = (0.0, 0.5, 0.8, 0.95)
STRENGTHS = (0.0, 0.25, 0.5, 1.0)


def old1mm_atlas() -> tuple[ForwardAtlas, dict]:
    profile = load_profile(STATIC_CONFIG, STATIC_KEY)
    xyz = np.asarray([[sample["x"], sample["y"], sample["depth"]]
                      for sample in profile["samples"]], float)
    field = np.asarray([sample["delta"] for sample in profile["samples"]], float)
    keep = np.isin(np.round(xyz[:, 2], 3), FULL_LEVELS)
    selected = xyz[keep]
    if len(selected) != 121 * len(FULL_LEVELS):
        raise ValueError("old1mm atlas is not a complete 121-by-5 grid")
    if not np.allclose(np.sort(np.unique(selected[:, 2])), FULL_LEVELS):
        raise ValueError("unexpected old1mm depth levels")
    atlas = ForwardAtlas(selected, field[keep], "unit_logamp",
                         allow_irregular_z=True)
    return atlas, {"source_config": str(STATIC_CONFIG.relative_to(ROOT)),
                   "source_config_sha256": sha(STATIC_CONFIG),
                   "gate_config": str(GATE_CONFIG.relative_to(ROOT)),
                   "gate_config_sha256": sha(GATE_CONFIG),
                   "points": int(len(selected)), "complete_depths_mm": list(FULL_LEVELS),
                   "training_contact_z_mm": 16.0}


def trial_path(session: str, number: int) -> Path:
    matches = list((ROOT / "data" / session).glob(f"Trial_{number:06d}_*"))
    if len(matches) != 1:
        raise ValueError(matches)
    path = matches[0]
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if (meta.get("status") != "completed" or
            meta["acquisition"]["magnetic_layer_profile_id"] != PHYSICAL_ID or
            abs(float(reference_origin(meta)[2]) - 16.0) > .01):
        raise ValueError(f"different physical configuration: {path}")
    return path


def read_trial(atlas: ForwardAtlas, session: str, number: int) -> dict:
    path = trial_path(session, number)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    data, _ = assemble(load_magnetic(path / "magnetic_raw.csv"),
                       command_segments(read_rows(path / "stage_commands.csv")),
                       read_rows(path / "events.csv"), reference_origin(meta))
    choose = data["input_valid"].astype(bool) & (np.arange(len(data["t_ns"])) % 4 == 0)
    observed = data["delta_b_uT"][choose].astype(float)
    truth = data["xyz_mm"][choose].astype(float)
    truth[:, 2] += 16.0 - float(reference_origin(meta)[2])
    supported = (np.isfinite(truth).all(axis=1) &
                 (truth[:, 2] >= atlas.z[0] - 1e-5) &
                 (truth[:, 2] <= atlas.z[-1] + 1e-5) &
                 (np.abs(truth[:, :2]) <= 10.0 + 1e-5).all(axis=1))
    supervised = data["xyz_train_mask"][choose].astype(bool) & supported
    moving = supervised & data["command_moving"][choose].astype(bool)
    static_grid = (supervised & ~moving &
                   (data["hold_age_s"][choose] >= .2) &
                   (data["contact_command"][choose] == 1))
    base = atlas.predict(observed)
    phi = atlas.features(observed)
    projected = atlas.features(atlas.forward(base))
    return {"session": session, "trial": number,
            "family": meta["trial_parameters"]["command_mode"],
            "source": path.relative_to(ROOT).as_posix(),
            "meta_sha256": sha(path / "meta.json"),
            "raw_sha256": sha(path / "magnetic_raw.csv"),
            "contact_z_mm": float(reference_origin(meta)[2]),
            "nominal_depth_mm": float(meta["trial_parameters"]["z_indent_mm"]),
            "n_input_10hz": int(len(observed)),
            "n_supervised_in_domain": int(supervised.sum()),
            "n_moving_in_domain": int(moving.sum()),
            "n_static_grid_frames": int(static_grid.sum()),
            "feature": np.column_stack((base, phi, phi - projected)),
            "base": base, "truth": truth, "supervised": supervised,
            "moving": moving, "static_grid": static_grid,
            "time_ns": data["t_ns"][choose],
            "b0_id": data["b0_id"][choose],
            "stream_id": data["stream_id"][choose]}


def metrics(row: dict, estimate: np.ndarray) -> dict:
    both = row["supervised"]
    moving = row["moving"]
    still = both & ~moving
    return {"moving": score(estimate[moving], row["truth"][moving]) if moving.any() else None,
            "static": score(estimate[still], row["truth"][still]) if still.any() else None,
            "grid_hold": score(estimate[row["static_grid"]], row["truth"][row["static_grid"]])
            if row["static_grid"].any() else None,
            "contact_all": score(estimate[both], row["truth"][both]) if both.any() else None}


def compact(row: dict, base: np.ndarray, estimate: np.ndarray) -> dict:
    return {key: row[key] for key in (
        "session", "trial", "family", "source", "meta_sha256", "raw_sha256",
        "contact_z_mm", "nominal_depth_mm", "n_input_10hz",
        "n_supervised_in_domain", "n_moving_in_domain", "n_static_grid_frames")} | {
            "static_atlas": metrics(row, base),
            "field_memory": metrics(row, estimate)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    gate = load_profile(GATE_CONFIG, GATE_KEY)
    if (gate.get("physical_profile_id") != PHYSICAL_ID or
            abs(float(gate["training_contact_z_mm"]) - 16.0) > 1e-6):
        raise ValueError("wrong old1mm gate profile")
    atlas, source = old1mm_atlas()
    groups = {"train": [read_trial(atlas, *key) for key in TRAIN],
              "validation": [read_trial(atlas, *key) for key in VALIDATION],
              "test": [read_trial(atlas, *key) for key in TEST],
              "static_audit": [read_trial(atlas, *key) for key in STATIC_AUDIT]}
    basis, basis_info = memory_basis(atlas, groups["train"])
    for rows in groups.values():
        for row in rows:
            row["observed_memory"] = observed_memory(
                atlas, basis, row["feature"], row["base"])
    candidates = []
    for retention in RETENTIONS:
        for row in groups["validation"]:
            row["state"] = causal_memory(
                row["observed_memory"], row["time_ns"],
                row["b0_id"], row["stream_id"], retention)
        for strength in STRENGTHS:
            trial_metrics = []
            for row in groups["validation"]:
                estimate = corrected(atlas, row, row["state"], basis, strength)
                trial_metrics.append(metrics(row, estimate))
            xy = float(np.mean([value["moving"]["xy_mean_mm"]
                                for value in trial_metrics]))
            z = float(np.mean([value["moving"]["z_mae_mm"]
                               for value in trial_metrics]))
            candidates.append({"retention": retention, "strength": strength,
                               "objective_xy_plus_0p2z_mm": xy + .2 * z,
                               "metrics": trial_metrics})
    selected = min(candidates, key=lambda value: value["objective_xy_plus_0p2z_mm"])
    comparison = {}
    for name, rows in groups.items():
        comparison[name] = []
        for row in rows:
            state = causal_memory(row["observed_memory"], row["time_ns"],
                                  row["b0_id"], row["stream_id"],
                                  selected["retention"])
            estimate = corrected(atlas, row, state, basis, selected["strength"])
            result = compact(row, row["base"], estimate)
            comparison[name].append(result)
            print("[R131]", name, row["trial"], row["family"],
                  "XY", round(result["static_atlas"]["moving"]["xy_mean_mm"], 3),
                  "->", round(result["field_memory"]["moving"]["xy_mean_mm"], 3),
                  flush=True)
    report = {"schema": "MagTouch-R131-Old1mmFieldMemory-3",
              "supersedes": "report_v2.json: adds independent next-day grid-hold audit; report.json incorrectly excluded z=1.6 nominal frames",
              "scope": "offline 10Hz field-memory fit, no contact gate or UI yet",
              "warning": "motion truth is interpolated stage command, not independently measured tip position; S24 development data was previously viewed; S26 test data not used for model selection",
              "physical_profile_id": PHYSICAL_ID,
              "source": source,
              "basis_rank": RANK,
              "basis": basis.tolist(),
              "basis_training": basis_info,
              "selection": {"candidates": candidates, "selected": selected},
              "comparisons": comparison}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
