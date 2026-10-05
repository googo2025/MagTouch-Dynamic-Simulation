#!/usr/bin/env python3
"""R108: apply a prior five-point XY registration to later-session models.

Only Trial 11's center and four cardinal stable contacts determine the rigid
transform. The remaining Trial 11 points, Trial 21, and Trials 12--15 are
evaluation data, never inputs to registration. This is an offline diagnostic;
the registration is not inferred from movement commands at inference time.
"""
from __future__ import annotations

import json

import numpy as np

from training.audit_had239_atlas_later_session_r106 import (
    CONFIG, OUT as R106_OUT, PROFILE_KEY, R50_CONFIG, R50_KEY, ROOT,
    ForwardAtlas, assemble, command_segments, direct_predict, extract_contacts,
    load_magnetic, load_profile, read_rows, reference_origin,
    r50_predict, samples, score, sha, trial_path,
)

OUT = ROOT / "analysis/r108_had239_atlas_fivepoint_20260926/report.json"
LANDMARKS = {(0., 0.), (-4., 0.), (4., 0.), (0., -4.), (0., 4.)}


def rigid_xy(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row-vector rotation and translation, without scale or shear."""
    a, b = pred[:, :2], truth[:, :2]
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    rot = u @ np.diag([1., np.linalg.det(u @ vt)]) @ vt
    return rot, bc - ac @ rot


def predict_all(r50: dict, r92: dict, atlases: dict,
                delta: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "r50_static": r50_predict(r50, delta),
        "r92_old_formula": direct_predict(r92, delta),
        **{name: atlas.predict(delta) for name, atlas in atlases.items()},
    }


def evaluate(preds: dict[str, np.ndarray], truth: np.ndarray,
             transforms: dict[str, tuple[np.ndarray, np.ndarray]],
             mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(truth), bool)
    out = {}
    for name, pred in preds.items():
        rot, shift = transforms[name]
        adjusted = pred.copy()
        adjusted[:, :2] = pred[:, :2] @ rot + shift
        out[name] = {"before": score(pred[mask], truth[mask]),
                     "after": score(adjusted[mask], truth[mask])}
    return out


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"R108 exists: {OUT}")
    r50 = load_profile(R50_CONFIG, R50_KEY)
    r92 = load_profile(CONFIG, PROFILE_KEY)
    xyz, field = samples(r92)
    atlases = {name: ForwardAtlas(xyz, field, name)
               for name in ("raw", "unit_logamp")}

    calibration_path = trial_path(11)
    contacts, calibration_info = extract_contacts(calibration_path)
    if calibration_info["errors"] or len(contacts) != 25:
        raise ValueError(calibration_info)
    delta = np.asarray([c["delta"] for c in contacts], float)
    truth = np.asarray([[c["x"], c["y"], c["z"]] for c in contacts], float)
    truth[:, 2] += r92["training_contact_z_mm"] - calibration_info["contact_z_mm"]
    landmark = np.asarray([(c["x"], c["y"]) in LANDMARKS for c in contacts])
    if landmark.sum() != 5:
        raise ValueError("missing or duplicate calibration landmarks")
    initial = predict_all(r50, r92, atlases, delta)
    transforms = {name: rigid_xy(pred[landmark], truth[landmark])
                  for name, pred in initial.items()}
    records = [{
        "trial": 11, "role": "calibration_grid_20_nonlandmarks_only",
        "source": calibration_path.relative_to(ROOT).as_posix(),
        "meta_sha256": sha(calibration_path / "meta.json"),
        "n": int((~landmark).sum()),
        "models": evaluate(initial, truth, transforms, ~landmark),
    }]
    for number in (12, 13, 14, 15, 21):
        path = trial_path(number)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if number == 21:
            contacts, info = extract_contacts(path)
            if info["errors"] or len(contacts) != 25:
                raise ValueError(info)
            d = np.asarray([c["delta"] for c in contacts], float)
            target = np.asarray([[c["x"], c["y"], c["z"]] for c in contacts], float)
            target[:, 2] += r92["training_contact_z_mm"] - info["contact_z_mm"]
            role = "later_grid"
        else:
            data, _ = assemble(load_magnetic(path / "magnetic_raw.csv"),
                               command_segments(read_rows(path / "stage_commands.csv")),
                               read_rows(path / "events.csv"), reference_origin(meta))
            eligible = (data["input_valid"].astype(bool) &
                        data["xyz_train_mask"].astype(bool) &
                        data["command_moving"].astype(bool) &
                        (np.arange(len(data["t_ns"])) % 4 == 0))
            d = data["delta_b_uT"][eligible].astype(float)
            target = data["xyz_mm"][eligible].astype(float)
            target[:, 2] += r92["training_contact_z_mm"] - float(reference_origin(meta)[2])
            role = "later_motion"
        records.append({
            "trial": number, "role": role,
            "source": path.relative_to(ROOT).as_posix(),
            "meta_sha256": sha(path / "meta.json"), "n": len(target),
            "models": evaluate(predict_all(r50, r92, atlases, d),
                               target, transforms),
        })
        print("[R108]", number, {name: round(value["after"]["xy_mean_mm"], 3)
              for name, value in records[-1]["models"].items()}, flush=True)
    result = {
        "schema": "MagTouch-R108-FivePointRegistration-1",
        "scope": "offline prior five-point XY rigid registration; no scale, shear, Z fit, gate, time model or UI",
        "warning": "stage-command reference, not measured tip; the later Trials were previously viewed in development",
        "contact_z_training_mm": r92["training_contact_z_mm"],
        "calibration_contact_z_mm": calibration_info["contact_z_mm"],
        "r50_config_sha256": sha(R50_CONFIG),
        "r92_config_sha256": sha(CONFIG),
        "r106_report_sha256": sha(R106_OUT),
        "landmark_xy_mm": sorted([list(p) for p in LANDMARKS]),
        "transforms": {name: {
            "rotation_row_xy": rot.tolist(), "translation_xy_mm": shift.tolist(),
            "angle_deg": float(np.rad2deg(np.arctan2(rot[0, 1], rot[0, 0]))),
        } for name, (rot, shift) in transforms.items()},
        "records": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
