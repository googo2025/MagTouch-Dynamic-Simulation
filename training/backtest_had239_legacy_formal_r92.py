#!/usr/bin/env python3
"""Held-out static and causal-stream backtest of the R92 old-formal refits.

This script never reads acquisition devices. Retrospective command-interpolated
motion references are labelled as such and never described as saved-live R92.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code")]
from analysis.r31_continuous_fit import predict_rbf, predict_ridge  # noqa: E402
from replay_backtest import JUMP_HYSTERESIS, NormalizedFit, load_profile  # noqa: E402
from training.build_fixed_magnet_domains_r49 import extract_contacts  # noqa: E402
from training.build_trajectory_dataset_v1 import (  # noqa: E402
    assemble, command_segments, load_magnetic, read_rows, reference_origin,
)

CONFIG = ROOT / "config/had239_legacy_formal_continuous_r92_20260926.json"
OUTPUT = ROOT / "analysis/r92_had239_legacy_formal_20260926/backtest.json"
FIVE_DYNAMIC = ROOT / "datasets/MagTouch_FinalTip_Trajectory_R52_20260922"
RUNTIME_TRIALS_5MM = {41, 50, 54}
RUNTIME_TRIALS_1MM = {3, 4, 8}


def one_dir(session: str, trial: int) -> Path:
    matches = list((ROOT / "data" / session).glob(f"Trial_{trial:06d}_*"))
    if len(matches) != 1:
        raise ValueError(f"expected one {session} Trial {trial}, got {len(matches)}")
    return matches[0]


def stat(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0}
    xy = np.linalg.norm(values[:, :2], axis=1)
    xyz = np.linalg.norm(values, axis=1)
    return {"n": int(len(values)), "xy_mean_mm": float(xy.mean()),
            "xy_p95_mm": float(np.percentile(xy, 95)),
            "z_mae_mm": float(np.abs(values[:, 2]).mean()),
            "xyz_mean_mm": float(xyz.mean()),
            "xyz_p95_mm": float(np.percentile(xyz, 95))}


def direct_predict(profile: dict, delta: np.ndarray) -> np.ndarray:
    result = np.full((len(delta), 3), np.nan)
    amp = np.linalg.norm(delta, axis=1)
    good = np.isfinite(delta).all(axis=1) & (amp > 1e-9)
    for start in range(0, len(delta), 512):
        ids = np.flatnonzero(good[start:start + 512]) + start
        if not len(ids):
            continue
        d = delta[ids].astype(np.float64)
        u = d / np.linalg.norm(d, axis=1)[:, None]
        result[ids, :2] = predict_rbf(profile["continuous_xy_model"], u)
        result[ids, 2] = predict_ridge(profile["continuous_z_model"], d)
    return result


def check_meta(path: Path, expected_profile: str) -> dict:
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8-sig"))
    actual = meta["acquisition"]["magnetic_layer_profile_id"]
    if meta["status"] != "completed" or actual != expected_profile:
        raise ValueError(f"wrong physical assembly/status for {path}: {actual}")
    return meta


def static_backtest(profile: dict, session: str, trial: int) -> dict:
    path = one_dir(session, trial)
    check_meta(path, profile["physical_profile_id"])
    contacts, extraction = extract_contacts(path)
    if extraction["errors"] or not contacts:
        raise ValueError(f"contact extraction failed: {path}: {extraction['errors']}")
    delta = np.asarray([c["delta"] for c in contacts], float)
    nominal = np.asarray([[c["x"], c["y"], c["z"]] for c in contacts], float)
    prediction = direct_predict(profile, delta)
    adjusted = nominal.copy()
    adjusted[:, 2] += profile["training_contact_z_mm"] - extraction["contact_z_mm"]
    return {
        "source": str(path.relative_to(ROOT)), "contacts": len(contacts),
        "contact_z_mm": extraction["contact_z_mm"],
        "training_contact_z_mm": profile["training_contact_z_mm"],
        "nominal_depth_mm": extraction["depth_mm"],
        "adjusted_depth_mm": float(adjusted[0, 2]),
        "domain_status": ("inside" if profile["calibration_domain_mm"]["depth"][0]
                          <= adjusted[0, 2] <= profile["calibration_domain_mm"]["depth"][1]
                          else "outside_training_depth"),
        "nominal": stat(prediction - nominal),
        "contact_z_adjusted": stat(prediction - adjusted),
        "truth_source": "recorded stationary grid command; not external tip metrology",
    }


def replay(profile: dict, data: dict, runtime: bool) -> dict:
    t = data["t_ns"]
    delta = data["delta_b_uT"].astype(np.float64)
    target = data["xyz_mm"].astype(np.float64)
    valid = data["input_valid"].astype(bool)
    label = data["xyz_train_mask"].astype(bool)
    moving = data["command_moving"].astype(bool)
    eligible = valid & label & moving & (np.arange(len(t)) % 4 == 0)
    negative = (valid & ~label & (np.arange(len(t)) % 4 == 0) &
                (target[:, 2] <= -.3))
    direct = direct_predict(profile, delta)
    out = {"eligible_moving_frames": int(eligible.sum()),
           "direct_regression_no_gate": stat(direct[eligible] - target[eligible]),
           "negative_frames": int(negative.sum())}
    if not runtime:
        return out
    defaults = profile["defaults"]
    engine = NormalizedFit(profile, Path("__r92_offline_only__"),
                           defaults["threshold"], defaults["min_cosine"],
                           defaults["smoothing_ms"], defaults["contact_threshold"],
                           JUMP_HYSTERESIS)
    engine.set_baseline(np.zeros(15))
    errors = []
    negative_outputs = 0
    rejects: Counter[str] = Counter()
    b0_id, stream_id = data["b0_id"], data["stream_id"]
    for i in range(len(t)):
        if i and (b0_id[i] != b0_id[i - 1] or stream_id[i] != stream_id[i - 1]):
            engine.set_baseline(np.zeros(15))
        if not valid[i] or not np.isfinite(delta[i]).all():
            continue
        pred = engine.push(int(t[i]), delta[i])
        if eligible[i]:
            if pred["valid"]:
                errors.append(np.asarray(pred["xyz"], float) - target[i])
            else:
                rejects[str(pred["status"])] += 1
        if negative[i] and pred["valid"]:
            negative_outputs += 1
    error_array = np.asarray(errors, float).reshape(-1, 3)
    out["runtime_old_normalizedfit"] = {
        **stat(error_array), "eligible": int(eligible.sum()),
        "coverage": len(error_array) / int(eligible.sum()) if eligible.any() else None,
        "negative_outputs": negative_outputs,
        "negative_false_positive_rate": negative_outputs / int(negative.sum())
        if negative.any() else None,
        "rejections": dict(rejects),
    }
    return out


def load_old1mm_motion(path: Path, expected_profile: str) -> dict:
    meta = check_meta(path, expected_profile)
    origin = reference_origin(meta)
    if abs(float(origin[2]) - 16.0) > 1e-6:
        raise ValueError(f"old 1mm trial has different actual contact Z: {path}")
    data, _ = assemble(load_magnetic(path / "magnetic_raw.csv"),
                       command_segments(read_rows(path / "stage_commands.csv")),
                       read_rows(path / "events.csv"), origin)
    return data


def main(config: Path = CONFIG, output: Path = OUTPUT, version: str = "r92") -> None:
    config = (ROOT / config).resolve() if not config.is_absolute() else config.resolve()
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if output.exists():
        raise FileExistsError("R92 backtest exists; refusing to overwrite")
    five = load_profile(config, f"5x5_had239_final5mm_old_formal_{version}")
    one = load_profile(config, f"5x5_had239_old1mm_old_formal_{version}")
    static_five = [static_backtest(five, "S20260923_A", n) for n in (1, 3)]
    static_one = [static_backtest(one, "S20260924_A", n) for n in (1, 2, 3)]
    static_one += [static_backtest(one, "S20260926_A", n) for n in (1, 2)]
    dynamic_five = []
    entries = [json.loads(line) for line in (FIVE_DYNAMIC / "index.jsonl")
               .read_text(encoding="utf-8").splitlines() if line]
    for entry in entries:
        path = Path(entry["source"])
        check_meta(path, five["physical_profile_id"])
        with np.load(FIVE_DYNAMIC / entry["sequence"], allow_pickle=False) as archive:
            data = {key: archive[key] for key in (
                "t_ns", "delta_b_uT", "xyz_mm", "input_valid",
                "xyz_train_mask", "command_moving", "b0_id", "stream_id")}
        result = replay(five, data, int(entry["trial"]) in RUNTIME_TRIALS_5MM)
        dynamic_five.append({"trial": int(entry["trial"]), "family": entry["family"],
                             "split_from_prior_R52": entry["split"],
                             "source": str(path.relative_to(ROOT)), **result})
        print("[R92 5mm MOTION]", entry["trial"],
              round(result["direct_regression_no_gate"]["xy_mean_mm"], 3), flush=True)
    dynamic_one = []
    for n in range(3, 9):
        path = one_dir("S20260926_A", n)
        data = load_old1mm_motion(path, one["physical_profile_id"])
        result = replay(one, data, n in RUNTIME_TRIALS_1MM)
        dynamic_one.append({"trial": n, "family": path.name.split("_")[4],
                            "source": str(path.relative_to(ROOT)), **result})
        print("[R92 1mm MOTION]", n,
              round(result["direct_regression_no_gate"]["xy_mean_mm"], 3), flush=True)
    report = {
        "schema": "MagTouch-LegacyFormalBacktest-1",
        "model_config": str(config.relative_to(ROOT)),
        "scope": f"retrospective old-formal refit; none of these Trials saved live {version.upper()} predictions",
        "reference_warning": "motion XYZ is host-command-interpolated, not external encoder/tip metrology",
        "had239_final5mm": {"static_external": static_five,
                             "dynamic_external": dynamic_five},
        "had239_old1mm": {"static_external": static_one,
                           "dynamic_external": dynamic_one},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--version", choices=("r92", "r93"), default="r92")
    args = parser.parse_args()
    main(args.config, args.output, args.version)
