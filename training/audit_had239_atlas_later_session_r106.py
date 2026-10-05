#!/usr/bin/env python3
"""R106: same-frame R50/R92/R104 comparison on later had-239 5mm records.

This uses raw magnetic data and command interpolation only. All models are
single-frame here; no contact gate or dynamic state is evaluated. The physical
profile and Contact Z are verified before each comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code")]
from replay_backtest import load_profile  # noqa: E402
from training.backtest_had239_legacy_formal_r92 import direct_predict  # noqa: E402
from training.build_fixed_magnet_domains_r49 import extract_contacts  # noqa: E402
from training.build_trajectory_dataset_v1 import (  # noqa: E402
    assemble, command_segments, load_magnetic, read_rows, reference_origin,
)
from training.probe_had239_local_forward_r104 import (  # noqa: E402
    CONFIG, PROFILE_KEY, ForwardAtlas, samples, score, sha,
)

OUT = ROOT / "analysis/r106_had239_atlas_later_session_20260926/report.json"
R50_CONFIG = ROOT / "config/replay_fit_presets_r50.json"
R50_KEY = "5x5_final_tip_depth_sweep_121pt_continuous"
SESSION = ROOT / "data/S20260924_A"
TRIALS = (11, 12, 13, 14, 15, 21)


def krr_predict(model: dict, delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, float)
    amplitude = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(amplitude[:, None], 1e-12)
    kind = model["feature"]
    if kind == "shape":
        feature = unit
    elif kind == "raw":
        feature = delta
    elif kind == "fixed_fusion":
        logamp = np.log(np.maximum(amplitude, 1e-8))
        feature = np.column_stack((unit, logamp, logamp ** 2,
                                   unit * logamp[:, None]))
    else:
        raise ValueError(kind)
    center = np.asarray(model["centers"], float)
    query = ((feature - np.asarray(model["mean"], float)) /
             np.asarray(model["scale"], float))
    coef = np.asarray(model["coef"], float)
    output_mean = np.asarray(model["output_mean"], float)
    result = np.empty((len(query), coef.shape[1]))
    center_norm = np.sum(center ** 2, axis=1)
    for lo in range(0, len(query), 256):
        hi = min(lo + 256, len(query))
        q = query[lo:hi]
        sq = np.maximum(np.sum(q ** 2, axis=1)[:, None] +
                        center_norm[None, :] - 2 * q @ center.T, 0.)
        kernel = np.exp(-float(model["gamma"]) * sq /
                        float(model["bandwidth"]))
        result[lo:hi] = kernel @ coef + output_mean
    return result


def r50_predict(profile: dict, delta: np.ndarray) -> np.ndarray:
    xy = krr_predict(profile["working_range_xy_model"], delta)
    z = krr_predict(profile["working_range_z_model"], delta)
    if xy.shape != (len(delta), 2) or z.shape != (len(delta), 1):
        raise ValueError("R50 model output shape mismatch")
    return np.column_stack((xy, z))


def trial_path(number: int) -> Path:
    paths = list(SESSION.glob(f"Trial_{number:06d}_*"))
    if len(paths) != 1:
        raise ValueError(paths)
    path = paths[0]
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if (meta["status"] != "completed" or
        meta["acquisition"]["magnetic_layer_profile_id"] !=
            "magnetic_layer_v1_5*5-had-239" or
        abs(float(meta["acquisition"]["printer_reference"]["contact_z_mm"])
            - 23.360001) > .01):
        raise ValueError(f"different physical/calibration state: {path}")
    return path


def evaluate(r50: dict, r92: dict, atlases: dict[str, ForwardAtlas],
             delta: np.ndarray, xyz: np.ndarray) -> dict:
    return {
        "r50_static": score(r50_predict(r50, delta), xyz),
        "r92_old_formula": score(direct_predict(r92, delta), xyz),
        **{name: score(atlas.predict(delta), xyz)
           for name, atlas in atlases.items()},
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"R106 result exists: {OUT}")
    r50 = load_profile(R50_CONFIG, R50_KEY)
    r92 = load_profile(CONFIG, PROFILE_KEY)
    xyz, delta = samples(r92)
    atlases = {name: ForwardAtlas(xyz, delta, name)
               for name in ("raw", "unit_logamp")}
    static, moving = [], []
    for number in TRIALS:
        path = trial_path(number)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        common = {
            "trial": number, "source": path.relative_to(ROOT).as_posix(),
            "meta_sha256": sha(path / "meta.json"),
            "motion": meta["trial_parameters"]["command_mode"],
            "contact_z_mm": meta["acquisition"]["printer_reference"]["contact_z_mm"],
        }
        if number in (11, 21):
            contacts, extraction = extract_contacts(path)
            if extraction["errors"] or len(contacts) != 25:
                raise ValueError(extraction)
            d = np.asarray([c["delta"] for c in contacts], float)
            target = np.asarray([[c["x"], c["y"], c["z"]] for c in contacts], float)
            target[:, 2] += 23.70 - extraction["contact_z_mm"]
            static.append({
                **common, "n": len(contacts),
                "adjusted_depth_mm": float(target[0, 2]),
                "models": evaluate(r50, r92, atlases, d, target),
            })
            continue
        data, _ = assemble(load_magnetic(path / "magnetic_raw.csv"),
                           command_segments(read_rows(path / "stage_commands.csv")),
                           read_rows(path / "events.csv"), reference_origin(meta))
        eligible = (data["input_valid"].astype(bool) &
                    data["xyz_train_mask"].astype(bool) &
                    data["command_moving"].astype(bool) &
                    (np.arange(len(data["t_ns"])) % 4 == 0))
        d = data["delta_b_uT"][eligible].astype(float)
        target = data["xyz_mm"][eligible].astype(float)
        target[:, 2] += 23.70 - float(reference_origin(meta)[2])
        moving.append({
            **common, "n": len(target),
            "adjusted_depth_nominal_mm": float(meta["trial_parameters"]["z_indent_mm"] +
                                                23.70 - reference_origin(meta)[2]),
            "models": evaluate(r50, r92, atlases, d, target),
        })
        print("[R106]", number, len(target),
              {name: round(value["xy_mean_mm"], 3)
               for name, value in moving[-1]["models"].items()}, flush=True)
    result = {
        "schema": "MagTouch-R106-LaterHad239SameFrameStatic-1",
        "scope": "offline direct regression only; no contact gate, temporal model or UI",
        "reference_warning": "command-interpolated motion XYZ, not measured tip; contact Z labels shifted +0.339999 mm into training coordinate system",
        "r50_config_sha256": sha(R50_CONFIG),
        "r92_config_sha256": sha(CONFIG),
        "later_grid": static, "later_motion": moving,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
