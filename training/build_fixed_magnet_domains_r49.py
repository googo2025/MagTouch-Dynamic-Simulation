"""Snapshot the four verified fixed-magnet acquisition domains for R49.

This builder never edits source Trials.  It records only completed 121-point
grid Trials and deliberately excludes aborted/recording folders.  Absolute and
relative eligibility are explicit so later training cannot silently mix
post-remount or changed-indenter absolute magnetic values into the canonical
absolute calibration.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_r47_depth_sweep_model import (
    _next_stage,
    _read_csv,
    _read_json,
    _stable_median_window,
    _vector_rows,
)


BATCHES: list[dict[str, Any]] = [
    {
        "id": "A_algorithm_regression_canonical",
        "session": "S20260921_A",
        "container": "session",
        "trials": list(range(17, 32)),
        "assembly_state": "original_fixed_silicone_same_mount_same_tip",
        "recorded_material": "silicone_v3",
        "contact_z_mm": 26.5,
        "role": "canonical_absolute_train_and_R48_source",
        "absolute_train": True,
        "relative_train": True,
        "notes": "121x15 depth sweep; 0.2 mm remains below-range diagnostic.",
    },
    {
        "id": "B_crossnight_validation",
        "session": "S20260922_A",
        "container": "session",
        "trials": list(range(1, 15)),
        "assembly_state": "same_mount_same_tip_crossnight",
        "recorded_material": "silicone_v3",
        "contact_z_mm": 26.5,
        "role": "independent_crossnight_absolute_and_relative_validation",
        "absolute_train": False,
        "relative_train": False,
        "notes": "Complete 0.4-3.0 mm validation sweep; never used for fitting.",
    },
    {
        "id": "C_remount_same_tip",
        "session": "S20260922_A",
        "container": "session",
        "trials": [18, 19, 20],
        "assembly_state": "sensor_removed_remounted_tip_removed_remounted_same_tip",
        "recorded_material": "silicone_v3",
        "contact_z_mm": 26.15,
        "role": "relative_train_absolute_transfer_diagnostic",
        "absolute_train": False,
        "relative_train": True,
        "notes": "Completed 0.8/1.8/2.8 mm after centre, plane and five-point recalibration. Trials 16-17 were aborted and excluded.",
    },
    {
        "id": "D_changed_tip_plan_snapshot",
        "session": "S20260922_A",
        "container": "Plan_20260922_184401_MagTouch_FixedSilicone_121_DepthSweep_0p2_3p0",
        "trials": list(range(26, 41)),
        "assembly_state": "same_fixed_magnet_changed_tip_recalibrated_plane",
        "recorded_material": "silicone_v4",
        "contact_z_mm": 23.7,
        "role": "relative_train_absolute_isolated_new_tip_domain",
        "absolute_train": False,
        "relative_train": True,
        "notes": "Rolling plan snapshot: include completed depths only; recording/pending Trials are excluded.",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trial_number(path: Path) -> int:
    return int(path.name.split("_", 2)[1])


def locate_trials(data_root: Path, spec: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    base = data_root / spec["session"]
    if spec["container"] != "session":
        base = base / spec["container"]
    wanted = set(int(value) for value in spec["trials"])
    completed: list[Path] = []
    excluded: list[dict[str, Any]] = []
    for path in sorted(base.glob("Trial_*_WITH_PRED_grid_points_*"), key=trial_number):
        number = trial_number(path)
        if number not in wanted:
            continue
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            excluded.append({"trial": path.name, "reason": "missing_meta"})
            continue
        meta = _read_json(meta_path)
        status = str(meta.get("status", "unknown"))
        files = [path / name for name in ("events.csv", "stage_commands.csv", "magnetic_raw.csv")]
        if status != "completed" or any(not item.is_file() or item.stat().st_size == 0 for item in files):
            excluded.append({"trial": path.name, "reason": f"status_or_files:{status}"})
            continue
        completed.append(path)
    return completed, excluded


def extract_contacts(trial: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = _read_json(trial / "meta.json")
    events = [(int(row["t_perf_ns"]), row) for row in _read_csv(trial / "events.csv")]
    begins = [(timestamp, row) for timestamp, row in events if row.get("event") == "GRID_POINT"]
    stages = sorted((int(row["t_perf_ns"]), row)
                    for row in _read_csv(trial / "stage_commands.csv"))
    times, fields, magnetic_qc = _vector_rows(trial / "magnetic_raw.csv")
    trial_end = max(times[-1] + 1,
                    max((timestamp for timestamp, _ in events), default=times[-1]) + 1,
                    max((timestamp for timestamp, _ in stages), default=times[-1]) + 1)
    contacts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, (event_time, row) in enumerate(begins):
        end_time = begins[index + 1][0] if index + 1 < len(begins) else trial_end
        try:
            payload = json.loads(row.get("value") or "{}")
            indent_time, _ = _next_stage(stages, event_time, end_time, "grid_indent")
            hold_time, _ = _next_stage(stages, indent_time, end_time,
                                       "grid_hold", "HOST_DWELL")
            b0, b0_frames, _, _, b0_fallback = _stable_median_window(
                times, fields, indent_time, "pre-indent B0")
            steady, steady_frames, _, _, steady_fallback = _stable_median_window(
                times, fields, hold_time, "steady hold")
            delta = steady - b0
            contacts.append({
                "x": float(payload["offset_x_mm"]),
                "y": float(payload["offset_y_mm"]),
                "z": float(payload["depth_mm"]),
                "delta": np.asarray(delta, dtype=np.float32),
                "b0": np.asarray(b0, dtype=np.float32),
                "amplitude_uT": float(np.linalg.norm(delta)),
                "b0_frames": int(b0_frames),
                "steady_frames": int(steady_frames),
                "fallback": bool(b0_fallback or steady_fallback),
            })
        except Exception as exc:
            errors.append({"event_index": index, "error": str(exc)})
    params = meta["trial_parameters"]
    acquisition = meta.get("acquisition", {})
    report = {
        "trial": trial.name,
        "trial_number": trial_number(trial),
        "created_local": meta.get("created_local"),
        "status": meta.get("status"),
        "depth_mm": float(params["z_indent_mm"]),
        "contacts": len(contacts),
        "errors": errors,
        "magnetic_qc": magnetic_qc,
        "material_id": params.get("material_id"),
        "magnetic_layer_id": params.get("magnetic_layer_id"),
        "profile_id": acquisition.get("magnetic_layer_profile_id"),
        "contact_z_mm": acquisition.get("printer_reference", {}).get("contact_z_mm"),
        "center_xy_mm": [params.get("x_mm"), params.get("y_mm")],
        "meta_sha256": sha256(trial / "meta.json"),
        "source_files": {
            name: {"size": (trial / name).stat().st_size,
                   "mtime_ns": (trial / name).stat().st_mtime_ns}
            for name in ("events.csv", "stage_commands.csv", "magnetic_raw.csv")
        },
    }
    return contacts, report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "datasets" / "MagTouch_FixedMagnet_R49_20260922")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists; use a new versioned directory: {output}")
    output.mkdir(parents=True)

    deltas: list[np.ndarray] = []
    targets: list[list[float]] = []
    b0_values: list[np.ndarray] = []
    batch_index: list[int] = []
    trial_ids: list[int] = []
    absolute_train: list[bool] = []
    relative_train: list[bool] = []
    working_range: list[bool] = []
    index_rows: list[dict[str, Any]] = []
    batch_reports: list[dict[str, Any]] = []

    for batch_id, spec in enumerate(BATCHES):
        trials, excluded = locate_trials(args.data_root, spec)
        reports = []
        start = len(deltas)
        for trial in trials:
            contacts, report = extract_contacts(trial)
            if report["errors"] or len(contacts) != 121:
                excluded.append({"trial": trial.name,
                                 "reason": f"incomplete_contacts:{len(contacts)}",
                                 "errors": report["errors"]})
                continue
            reports.append(report)
            for contact_index, contact in enumerate(contacts):
                depth = float(contact["z"])
                is_working = depth >= 0.4 - 1e-8
                deltas.append(contact["delta"])
                b0_values.append(contact["b0"])
                targets.append([contact["x"], contact["y"], depth])
                batch_index.append(batch_id)
                trial_ids.append(report["trial_number"])
                absolute_train.append(bool(spec["absolute_train"] and is_working))
                relative_train.append(bool(spec["relative_train"] and is_working))
                working_range.append(is_working)
                index_rows.append({
                    "sample_index": len(deltas) - 1,
                    "batch_id": spec["id"],
                    "trial": report["trial_number"],
                    "contact_index": contact_index,
                    "x_mm": contact["x"], "y_mm": contact["y"], "z_mm": depth,
                    "amplitude_uT": contact["amplitude_uT"],
                    "absolute_train": bool(spec["absolute_train"] and is_working),
                    "relative_train": bool(spec["relative_train"] and is_working),
                    "working_range": is_working,
                })
            print(f"[R49] {spec['id']} {trial.name}: {len(contacts)} contacts", flush=True)
        batch_reports.append({
            **spec,
            "batch_index": batch_id,
            "completed_trials_at_snapshot": len(reports),
            "completed_trial_numbers": [row["trial_number"] for row in reports],
            "depths_mm": sorted({row["depth_mm"] for row in reports}),
            "samples": len(deltas) - start,
            "trial_reports": reports,
            "excluded": excluded,
        })

    delta = np.stack(deltas).astype(np.float32)
    xyz = np.asarray(targets, dtype=np.float32)
    batches = np.asarray(batch_index, dtype=np.int16)
    abs_mask = np.asarray(absolute_train, dtype=bool)
    rel_mask = np.asarray(relative_train, dtype=bool)
    work_mask = np.asarray(working_range, dtype=bool)
    if delta.shape[1:] != (15,) or xyz.shape[1:] != (3,):
        raise ValueError(f"unexpected shapes delta={delta.shape}, xyz={xyz.shape}")
    if not np.isfinite(delta).all() or not np.isfinite(xyz).all():
        raise ValueError("non-finite R49 contact dataset")
    train_delta = delta[abs_mask]
    normalization = {
        "fit_scope": "batch A absolute training contacts only",
        "delta_mean_uT": train_delta.mean(axis=0).tolist(),
        "delta_std_uT": np.maximum(train_delta.std(axis=0), 1.0).tolist(),
    }
    np.savez_compressed(
        output / "contact_medians.npz", delta_b_uT=delta, xyz_mm=xyz,
        b0_uT=np.stack(b0_values).astype(np.float32), batch_index=batches,
        trial_id=np.asarray(trial_ids, dtype=np.int16),
        absolute_train_mask=abs_mask, relative_train_mask=rel_mask,
        working_range_mask=work_mask,
    )
    write_csv(output / "contact_index.csv", index_rows)
    manifest = {
        "schema": "MagTouch-FixedMagnet-Domains-R49-1.0",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "snapshot_complete",
        "source_data_mutated": False,
        "sample_count": len(delta),
        "absolute_train_samples": int(abs_mask.sum()),
        "relative_train_samples": int(rel_mask.sum()),
        "normalization": normalization,
        "batches": batch_reports,
        "policy": {
            "absolute": "Only canonical batch A is eligible for absolute-model fitting; B is validation; C/D are diagnostics.",
            "relative": "A/C/D may train relative displacement; B remains crossnight validation.",
            "below_range": "0.2 mm is retained as diagnostic and excluded from fit masks.",
            "incomplete": "Aborted, recording and pending Trials are excluded from this immutable snapshot.",
        },
        "limitations": [
            "Grid contacts contain strong Z press/release information but no continuous lateral path.",
            "Arbitrary dXY claims require later straight/eight-direction/circle/XYZ trajectories from the same final tip state.",
            "Commanded printer coordinates are labels, not independent metrology.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "samples": len(delta),
        "absolute_train": int(abs_mask.sum()), "relative_train": int(rel_mask.sum()),
        "batches": [{"id": row["id"], "trials": row["completed_trial_numbers"],
                     "depths": row["depths_mm"], "samples": row["samples"]}
                    for row in batch_reports],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
