#!/usr/bin/env python3
"""R116: guarded causal ridge correction on top of a continuous field atlas.

Static field atlas comes only from Trials 27--40. Residual coefficients use
the 10 complete train Trials in R52; lambda/correction cap use only its four
validation Trials. Five test Trials are scored once after selection. All model
features are current/past magnetic input and current/past atlas estimates,
never stage commands, future frames, or trajectory family.
"""
from __future__ import annotations

import json

import numpy as np

from training.probe_had239_local_forward_r104 import (
    CONFIG, INDEX, PROFILE_KEY, ROOT, SEQUENCES, ForwardAtlas,
    load_profile, samples, score, sha,
)

OUT = ROOT / "analysis/r116_had239_atlas_causal_residual_20260926/report.json"
LAGS = (1, 3, 5)  # 10Hz analysis: 0.1/0.3/0.5s, all causal.
LAMBDA_GRID = (.001, .01, .1, 1.)
CAP_GRID_MM = (2., 4., None)


def feature_matrix(atlas: ForwardAtlas, observed: np.ndarray,
                   predicted: np.ndarray, time_ns: np.ndarray,
                   b0_id: np.ndarray, stream_id: np.ndarray) -> np.ndarray:
    phi = atlas.features(observed)
    field_residual = phi - atlas.features(atlas.forward(predicted))
    columns = [predicted, phi, field_residual]
    n = len(observed)
    for lag in LAGS:
        previous = np.arange(n) - lag
        valid = previous >= 0
        source = np.maximum(previous, 0)
        valid &= (b0_id == b0_id[source]) & (stream_id == stream_id[source])
        valid &= (time_ns - time_ns[source]) <= int(0.15 * lag * 1e9)
        valid &= (time_ns - time_ns[source]) > 0
        change_field = phi - phi[source]
        change_position = predicted - predicted[source]
        change_field[~valid] = 0.
        change_position[~valid] = 0.
        columns.extend((change_field, change_position, valid[:, None].astype(float)))
    return np.column_stack(columns)


def load_split(atlas: ForwardAtlas) -> dict[str, list[dict]]:
    rows = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines()
            if line]
    result = {"train": [], "validation": [], "test": []}
    if {key: sum(row["split"] == key for row in rows) for key in result} != {
            "train": 10, "validation": 4, "test": 5}:
        raise ValueError("R52 split changed")
    for row in rows:
        with np.load(SEQUENCES / row["sequence"], allow_pickle=False) as data:
            choose = (data["input_valid"].astype(bool) &
                      (np.arange(len(data["t_ns"])) % 4 == 0))
            observed = data["delta_b_uT"][choose].astype(float)
            truth = data["xyz_mm"][choose].astype(float)
            t = data["t_ns"][choose]
            b0 = data["b0_id"][choose]
            stream = data["stream_id"][choose]
            supervised = data["xyz_train_mask"][choose].astype(bool)
            moving = supervised & data["command_moving"][choose].astype(bool)
        base = atlas.predict(observed)
        feature = feature_matrix(atlas, observed, base, t, b0, stream)
        result[row["split"]].append({
            "trial": int(row["trial"]), "family": row["family"],
            "source": row["source"], "sequence_sha256": sha(SEQUENCES / row["sequence"]),
            "feature": feature, "base": base, "truth": truth,
            "supervised": supervised, "moving": moving,
        })
    return result


def fit(train: list[dict], strength: float) -> dict:
    features = np.vstack([row["feature"][row["supervised"]] for row in train])
    residual = np.vstack([(row["truth"] - row["base"])[row["supervised"]]
                          for row in train])
    mean = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), .01)
    x = np.column_stack((np.ones(len(features)), (features - mean) / scale))
    n = len(x)
    penalty = np.eye(x.shape[1]) * strength
    penalty[0, 0] = 0.
    coef = np.linalg.solve((x.T @ x) / n + penalty,
                           (x.T @ residual) / n)
    return {"mean": mean, "scale": scale, "coef": coef,
            "strength": strength, "fit_frames": n}


def infer(model: dict, row: dict, cap_mm: float | None) -> np.ndarray:
    x = (row["feature"] - model["mean"]) / model["scale"]
    correction = model["coef"][0] + x @ model["coef"][1:]
    if cap_mm is not None:
        xy_norm = np.linalg.norm(correction[:, :2], axis=1)
        correction[:, :2] *= (np.minimum(1., cap_mm /
                                         np.maximum(xy_norm, 1e-12)))[:, None]
        correction[:, 2] = np.clip(correction[:, 2], -1., 1.)
    return row["base"] + correction


def evaluate(rows: list[dict], model: dict | None,
             cap_mm: float | None = None) -> list[dict]:
    result = []
    for row in rows:
        pred = row["base"] if model is None else infer(model, row, cap_mm)
        result.append({
            "trial": row["trial"], "family": row["family"],
            "source": row["source"], "sequence_sha256": row["sequence_sha256"],
            "n_moving": int(row["moving"].sum()),
            "moving": score(pred[row["moving"]], row["truth"][row["moving"]]),
            "n_contact_all": int(row["supervised"].sum()),
            "contact_all": score(pred[row["supervised"]], row["truth"][row["supervised"]]),
        })
    return result


def objective(reports: list[dict]) -> float:
    return float(np.mean([r["moving"]["xyz_mean_mm"] +
                          .1 * r["moving"]["xy_p95_mm"] for r in reports]))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"R116 exists: {OUT}")
    profile = load_profile(CONFIG, PROFILE_KEY)
    xyz, delta = samples(profile)
    atlas = ForwardAtlas(xyz, delta, "unit_logamp")
    splits = load_split(atlas)
    validation_base = evaluate(splits["validation"], None)
    candidates = []
    for strength in LAMBDA_GRID:
        model = fit(splits["train"], strength)
        for cap in CAP_GRID_MM:
            val = evaluate(splits["validation"], model, cap)
            candidates.append({"strength": strength, "cap_xy_mm": cap,
                               "objective": objective(val), "validation": val})
            print("[R116 val]", strength, cap, round(objective(val), 4), flush=True)
    chosen = min(candidates, key=lambda c: c["objective"])
    model = fit(splits["train"], chosen["strength"])
    test_base = evaluate(splits["test"], None)
    test_model = evaluate(splits["test"], model, chosen["cap_xy_mm"])
    print("[R116 selected]", chosen["strength"], chosen["cap_xy_mm"])
    for baseline, new in zip(test_base, test_model):
        print("[R116 test]", baseline["trial"],
              round(baseline["moving"]["xy_mean_mm"], 3),
              round(new["moving"]["xy_mean_mm"], 3),
              "p95", round(baseline["moving"]["xy_p95_mm"], 3),
              round(new["moving"]["xy_p95_mm"], 3), flush=True)
    result = {
        "schema": "MagTouch-R116-AtlasCausalRidgeResidual-1",
        "scope": "offline 10Hz causal magnetic-history residual; no gate/UI/hardware",
        "warning": "R52 test Trials were previously viewed during development; test excluded from this coefficient and hyperparameter fit, but not a new blind test",
        "source_config_sha256": sha(CONFIG),
        "source_index_sha256": sha(INDEX),
        "feature_definition": "current atlas xyz, current unit+logamp, current atlas field residual; 0.1/0.3/0.5s past magnetic/atlas differences with causal same-B0/stream masks",
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "lambda_grid": LAMBDA_GRID, "xy_cap_grid_mm": CAP_GRID_MM,
        "validation_objective": "macro_trial_mean(xyz_mean_mm + 0.1*xy_p95_mm) on moving command-contact frames",
        "validation_base": validation_base,
        "validation_candidates": candidates,
        "selected": {"strength": chosen["strength"],
                     "cap_xy_mm": chosen["cap_xy_mm"],
                     "validation_objective": chosen["objective"]},
        "selected_model": {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in model.items()},
        "test_base": test_base, "test_selected": test_model,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
