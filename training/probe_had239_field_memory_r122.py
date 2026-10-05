#!/usr/bin/env python3
"""R122: offline low-rank magnetic-memory probe for the had-239 5 mm tip.

The model corrects the *magnetic observation*, never adds an XYZ residual.
R52 complete-Trial train/validation/test partitions are retained.  Later
S20260924_A Trials 11--15/21 are an already-viewed external audit, not a
hyperparameter selection set.  This module does not import UI/hardware code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.audit_had239_atlas_fivepoint_r108 import LANDMARKS, rigid_xy
from training.audit_had239_atlas_later_session_r106 import (
    assemble, command_segments, load_magnetic, read_rows, reference_origin,
    trial_path,
)
from training.fit_had239_atlas_causal_residual_r116 import (
    CONFIG, INDEX, PROFILE_KEY, ROOT, ForwardAtlas, load_profile,
    load_split, samples, score, sha,
)

OUT = ROOT / "analysis/r122_had239_field_memory_20260926/report.json"
RANK = 2
LAGS = (0.0, 0.5, 0.8, 0.95)
STRENGTHS = (0.25, 0.5, 1.0)
RIDGE = 0.02
H_CAP = 1.5


def memory_basis(atlas: ForwardAtlas, train: list[dict]) -> tuple[np.ndarray, dict]:
    """Equal-Trial covariance of field residuals relative to the static map."""
    covariance = np.zeros((16, 16), float)
    counts = {}
    for row in train:
        truth = row["truth"]
        usable = (row["supervised"] & row["moving"] &
                  (truth[:, 2] >= atlas.z[0]) &
                  (truth[:, 2] <= atlas.z[-1]))
        if usable.sum() < 20:
            continue
        observed = row["feature"][:, 3:19]
        predicted = atlas.features(atlas.forward(truth))
        residual = observed - predicted
        steady = (row["supervised"] & ~row["moving"] &
                  (truth[:, 2] >= atlas.z[0]))
        if steady.sum() >= 20:
            residual -= np.median(residual[steady], axis=0)
        part = residual[usable]
        # Each complete Trial has equal weight regardless of frame count.
        covariance += part.T @ part / len(part)
        counts[str(row["trial"])] = int(len(part))
    if not counts:
        raise ValueError("no supervised moving samples in R52 training split")
    covariance /= len(counts)
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalue)[::-1]
    explained = eigenvalue[order] / max(eigenvalue.sum(), 1e-12)
    return eigenvector[:, order[:RANK]], {
        "trial_frame_counts": counts,
        "explained_variance_first_six": explained[:6].tolist(),
        "rank2_variance": float(explained[:RANK].sum()),
    }


def local_jacobian(atlas: ForwardAtlas, point: np.ndarray) -> np.ndarray:
    """Jacobian of the same unit+log-amplitude geometry used by R104."""
    columns = []
    for column, step in enumerate((0.1, 0.1, 0.05)):
        plus, minus = point.copy(), point.copy()
        plus[:, column] += step
        minus[:, column] -= step
        columns.append((atlas.features(atlas.forward(plus)) -
                        atlas.features(atlas.forward(minus))) / (2 * step))
    return np.stack(columns, axis=2)


def observed_memory(atlas: ForwardAtlas, basis: np.ndarray,
                    feature: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Infer memory only from the field component distinguishable from XYZ."""
    phi = feature[:, 3:19]
    residual = phi - atlas.features(atlas.forward(base))
    jacobian = local_jacobian(atlas, base)
    jtj = np.einsum("nfi,nfj->nij", jacobian, jacobian)
    trace = np.trace(jtj, axis1=1, axis2=2)
    jtj[:, range(3), range(3)] += (.02 * trace / 3 + 1e-7)[:, None]
    # Project both the current residual and the learned memory modes outside
    # the local position tangent.  Do not mistake tangent ambiguity for h.
    jr = np.einsum("nfi,nf->ni", jacobian, residual)
    ju = np.einsum("nfi,fk->nik", jacobian, basis)
    tangent_r = np.einsum("nfi,ni->nf", jacobian,
                          np.linalg.solve(jtj, jr[..., None])[..., 0])
    tangent_u = np.einsum("nfi,nik->nfk", jacobian,
                          np.linalg.solve(jtj, ju))
    orthogonal_r = residual - tangent_r
    orthogonal_u = basis[None, :, :] - tangent_u
    utu = np.einsum("nfk,nfl->nkl", orthogonal_u, orthogonal_u)
    utr = np.einsum("nfk,nf->nk", orthogonal_u, orthogonal_r)
    utu[:, range(RANK), range(RANK)] += RIDGE
    h = np.linalg.solve(utu, utr[..., None])[..., 0]
    return np.clip(h, -H_CAP, H_CAP)


def causal_memory(h_obs: np.ndarray, time_ns: np.ndarray,
                  b0_id: np.ndarray, stream_id: np.ndarray,
                  retention: float) -> np.ndarray:
    """Causal state; stopping motion never resets it, a new stream/B0 does."""
    state = np.zeros_like(h_obs)
    for i in range(len(h_obs)):
        same = (i > 0 and b0_id[i] == b0_id[i - 1] and
                stream_id[i] == stream_id[i - 1] and
                0 < time_ns[i] - time_ns[i - 1] <= 250_000_000)
        previous = state[i - 1] if same else np.zeros(RANK)
        state[i] = retention * previous + (1.0 - retention) * h_obs[i]
    return state


def feature_to_field(atlas: ForwardAtlas, feature: np.ndarray) -> np.ndarray:
    unit = feature[:, :15]
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-9)
    log_amplitude = np.clip(feature[:, 15], -4.0, 4.0)
    amplitude = atlas.amplitude_scale * np.exp(2 * log_amplitude)
    return unit * amplitude[:, None]


def corrected(atlas: ForwardAtlas, row: dict, state: np.ndarray,
              basis: np.ndarray, strength: float) -> np.ndarray:
    phi = row["feature"][:, 3:19] - strength * (state @ basis.T)
    return atlas.predict(feature_to_field(atlas, phi))


def row_metrics(row: dict, pred: np.ndarray) -> dict:
    mask = row["supervised"]
    moving = row["moving"]
    still = mask & ~moving
    return {
        "trial": row["trial"], "family": row["family"],
        "moving": score(pred[moving], row["truth"][moving]),
        "static": score(pred[still], row["truth"][still]) if still.any() else None,
        "contact_all": score(pred[mask], row["truth"][mask]),
    }


def objective(metrics: list[dict]) -> float:
    return float(np.mean([
        r["moving"]["xy_mean_mm"] + .1 * r["moving"]["xy_p95_mm"] +
        (.5 * r["static"]["xyz_mean_mm"] if r["static"] else 0)
        for r in metrics
    ]))


def with_sequence_metadata(rows: dict[str, list[dict]]) -> None:
    index = {int(json.loads(line)["trial"]): json.loads(line)
             for line in INDEX.read_text(encoding="utf-8").splitlines() if line}
    for split in rows.values():
        for row in split:
            with np.load(INDEX.parent / index[row["trial"]]["sequence"],
                         allow_pickle=False) as data:
                choose = (data["input_valid"].astype(bool) &
                          (np.arange(len(data["t_ns"])) % 4 == 0))
                row["time_ns"] = data["t_ns"][choose]
                row["b0_id"] = data["b0_id"][choose]
                row["stream_id"] = data["stream_id"][choose]


def external_row(number: int, atlas: ForwardAtlas,
                 training_contact_z: float, stride: int = 4) -> dict:
    path = trial_path(number)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    data, _ = assemble(load_magnetic(path / "magnetic_raw.csv"),
                       command_segments(read_rows(path / "stage_commands.csv")),
                       read_rows(path / "events.csv"), reference_origin(meta))
    choose = (data["input_valid"].astype(bool) &
              (np.arange(len(data["t_ns"])) % stride == 0))
    observed = data["delta_b_uT"][choose].astype(float)
    base = atlas.predict(observed)
    phi = atlas.features(observed)
    feature = np.column_stack((base, phi, phi -
                               atlas.features(atlas.forward(base))))
    truth = data["xyz_mm"][choose].astype(float)
    truth[:, 2] += training_contact_z - float(reference_origin(meta)[2])
    supervised = data["xyz_train_mask"][choose].astype(bool)
    moving = supervised & data["command_moving"][choose].astype(bool)
    return {
        "trial": number, "family": json.loads((path / "meta.json").read_text(
            encoding="utf-8")).get("action", "external"),
        "source": path.relative_to(ROOT).as_posix(),
        "source_sha256": sha(path / "magnetic_raw.csv"),
        "feature": feature, "base": base, "truth": truth,
        "supervised": supervised, "moving": moving,
        "static_grid": (supervised & ~moving &
                        (data["hold_age_s"][choose] >= .2)),
        "time_ns": data["t_ns"][choose],
        "b0_id": data["b0_id"][choose],
        "stream_id": data["stream_id"][choose],
    }


def grid_centers(row: dict, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = row["truth"]
    positions = np.unique(np.round(truth[row["static_grid"], :2], 3), axis=0)
    if len(positions) != 25:
        raise ValueError(f"Trial {row['trial']}: expected 25 grid holds")
    target, values = [], []
    for xy in positions:
        mask = (row["static_grid"] &
                np.all(np.isclose(truth[:, :2], xy, atol=.001), axis=1))
        target.append(np.median(truth[mask], axis=0))
        values.append(np.median(pred[mask], axis=0))
    return np.asarray(target), np.asarray(values)


def external_audit(atlas: ForwardAtlas, basis: np.ndarray,
                   retention: float, strength: float,
                   training_contact_z: float) -> dict:
    rows = {n: external_row(n, atlas, training_contact_z)
            for n in (11, 12, 13, 14, 15, 21)}
    predictions = {}
    for number, row in rows.items():
        h_obs = observed_memory(atlas, basis, row["feature"], row["base"])
        state = causal_memory(h_obs, row["time_ns"], row["b0_id"],
                              row["stream_id"], retention)
        predictions[number] = {
            "base": row["base"],
            "corrected": corrected(atlas, row, state, basis, strength),
        }
    target, _ = grid_centers(rows[11], predictions[11]["base"])
    landmark = np.asarray([tuple(xy) in LANDMARKS for xy in target[:, :2]])
    if landmark.sum() != 5:
        raise ValueError("missing Trial 11 five-point landmarks")
    transforms = {}
    for name in ("base", "corrected"):
        _, points = grid_centers(rows[11], predictions[11][name])
        transforms[name] = rigid_xy(points[landmark], target[landmark])
    result = {"transforms": {}, "trials": []}
    for name, (rot, shift) in transforms.items():
        result["transforms"][name] = {
            "rotation_row_xy": rot.tolist(), "translation_xy_mm": shift.tolist()}
    for number, row in rows.items():
        trial = {"trial": number, "source": row["source"],
                 "raw_sha256": row["source_sha256"], "models": {}}
        for name, pred in predictions[number].items():
            rot, shift = transforms[name]
            pred = pred.copy()
            pred[:, :2] = pred[:, :2] @ rot + shift
            if number in (11, 21):
                target, points = grid_centers(row, pred)
                keep = np.asarray([tuple(xy) not in LANDMARKS
                                   for xy in target[:, :2]]) if number == 11 \
                    else np.ones(len(target), bool)
                trial["models"][name] = score(points[keep], target[keep])
            else:
                moving = row["moving"]
                trial["models"][name] = score(pred[moving],
                                              row["truth"][moving])
        result["trials"].append(trial)
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"R122 report exists: {OUT}")
    profile = load_profile(CONFIG, PROFILE_KEY)
    xyz, field = samples(profile)
    atlas = ForwardAtlas(xyz, field, "unit_logamp")
    rows = load_split(atlas)
    with_sequence_metadata(rows)
    basis, basis_info = memory_basis(atlas, rows["train"])
    # Observation and causal state do not depend on correction strength.
    for split in rows.values():
        for row in split:
            row["h_obs"] = observed_memory(atlas, basis, row["feature"],
                                            row["base"])
    base_validation = [row_metrics(row, row["base"])
                       for row in rows["validation"]]
    candidates = []
    for retention in LAGS:
        for split in rows.values():
            for row in split:
                row["state"] = causal_memory(
                    row["h_obs"], row["time_ns"], row["b0_id"],
                    row["stream_id"], retention)
        for strength in STRENGTHS:
            metrics = [row_metrics(row, corrected(atlas, row, row["state"],
                                                  basis, strength))
                       for row in rows["validation"]]
            candidates.append({"retention": retention, "strength": strength,
                               "objective": objective(metrics), "trials": metrics})
    chosen = min(candidates, key=lambda x: x["objective"])
    retention, strength = chosen["retention"], chosen["strength"]
    comparisons = {}
    for split_name in ("train", "validation", "test"):
        details = []
        for row in rows[split_name]:
            state = causal_memory(row["h_obs"], row["time_ns"], row["b0_id"],
                                  row["stream_id"], retention)
            pred = corrected(atlas, row, state, basis, strength)
            details.append({"source": row["source"],
                            "base": row_metrics(row, row["base"]),
                            "candidate": row_metrics(row, pred)})
        comparisons[split_name] = details
    external = external_audit(atlas, basis, retention, strength,
                              profile["training_contact_z_mm"])
    report = {
        "schema": "MagTouch-R122-LowRankFieldMemory-1",
        "scope": "offline 10Hz causal field correction; no contact gate/UI/hardware",
        "warning": "position truth is interpolated stage command, not measured contact pose; R52 test and S24 external Trials were previously viewed",
        "sources": {"static_config": str(CONFIG.relative_to(ROOT)),
                    "static_config_sha256": sha(CONFIG),
                    "trajectory_index": str(INDEX.relative_to(ROOT)),
                    "trajectory_index_sha256": sha(INDEX)},
        "basis": basis.tolist(), "basis_training": basis_info,
        "basis_rank": RANK, "orthogonal_ridge": RIDGE,
        "memory_observation_cap": H_CAP,
        "selection": {"baseline_objective": objective(base_validation),
                      "candidates": candidates,
                      "selected": {"retention": retention,
                                   "strength": strength,
                                   "objective": chosen["objective"]}},
        "r52_split": comparisons,
        "external_fivepoint": external,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print("[R122] basis variance", round(basis_info["rank2_variance"], 3))
    print("[R122] validation", round(objective(base_validation), 3),
          "->", round(chosen["objective"], 3),
          "retention", retention, "strength", strength)
    for row in external["trials"]:
        print("[R122] external", row["trial"],
              round(row["models"]["base"]["xy_mean_mm"], 3),
              "->", round(row["models"]["corrected"]["xy_mean_mm"], 3))
    print(OUT)


if __name__ == "__main__":
    main()
