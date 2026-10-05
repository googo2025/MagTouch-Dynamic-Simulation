#!/usr/bin/env python3
"""Train and frozen-backtest neural decoders on the R47 121-point sweep.

This is an offline-only experiment.  The frozen 85-position spatial holdout and
the 3.0-mm depth endpoint are never read by architecture selection, early
stopping, feature normalization, or target normalization.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.r47_depth_sweep_121_fit import anchor_mask, validate_depth_sweep_dataset
from tools.build_r47_depth_sweep_model import extract_trial_windows, find_trials


ARCHITECTURES = ("direct_mlp_15", "shape_depth_dual")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def engineered_features(kind: str, delta: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float32)
    if kind == "direct_mlp_15":
        return delta
    if kind == "shape_depth_dual":
        amplitude = np.linalg.norm(delta, axis=1, keepdims=True)
        unit = delta / np.maximum(amplitude, 1e-6)
        return np.concatenate((unit, delta, np.log1p(amplitude)), axis=1)
    raise ValueError(f"unknown architecture {kind!r}")


class DirectMLP15(nn.Module):
    """eFlesh-comparable 15 -> 128 -> 128 -> XYZ decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class ShapeDepthDualMLP(nn.Module):
    """Separate direction/shape and raw-amplitude encoders with joint heads."""

    def __init__(self) -> None:
        super().__init__()
        self.shape = nn.Sequential(
            nn.Linear(15, 96), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(96, 64), nn.GELU(),
        )
        self.depth = nn.Sequential(
            nn.Linear(16, 64), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(64, 32), nn.GELU(),
        )
        self.joint = nn.Sequential(
            nn.Linear(96, 128), nn.GELU(), nn.Dropout(0.05),
        )
        self.xy = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))
        self.z = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shape = self.shape(values[:, :15])
        depth = self.depth(values[:, 15:])
        joint = self.joint(torch.cat((shape, depth), dim=1))
        return torch.cat((self.xy(joint), self.z(joint)), dim=1)


def make_model(kind: str) -> nn.Module:
    if kind == "direct_mlp_15":
        return DirectMLP15()
    if kind == "shape_depth_dual":
        return ShapeDepthDualMLP()
    raise ValueError(kind)


def load_contacts(session: Path, first: int = 17, last: int = 31
                  ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    contacts: list[dict[str, Any]] = []
    trial_reports: list[dict[str, Any]] = []
    b0_by_trial: list[np.ndarray] = []
    for trial in find_trials(session, first, last):
        extracted, report = extract_trial_windows(trial)
        if report["errors"] or len(extracted) != 121:
            raise ValueError(f"{trial.name}: expected 121 clean contacts; report={report}")
        contacts.extend(extracted)
        trial_reports.append(report)
        b0_by_trial.append(np.asarray(extracted[0]["b0"], dtype=np.float64))
        print(f"[EXTRACT] {trial.name}: {len(extracted)} contacts", flush=True)

    samples = [item["sample"] for item in contacts]
    audit = validate_depth_sweep_dataset(samples)
    delta = np.asarray([sample["delta"] for sample in samples], dtype=np.float32)
    target = np.asarray(
        [[sample["x"], sample["y"], sample["depth"]] for sample in samples],
        dtype=np.float32,
    )
    if delta.shape != (1815, 15) or target.shape != (1815, 3):
        raise ValueError(f"unexpected shapes: delta={delta.shape}, target={target.shape}")
    if not np.isfinite(delta).all() or not np.isfinite(target).all():
        raise ValueError("non-finite neural sample")

    b0 = np.stack(b0_by_trial)
    reference = np.median(b0, axis=0)
    b0_l2 = np.linalg.norm(b0 - reference, axis=1)
    b0_channel = np.max(np.abs(b0 - reference), axis=1)
    metadata = {
        "dataset_audit": audit,
        "trial_reports": trial_reports,
        "first_contact_b0_reference_uT": reference.tolist(),
        "first_contact_b0_l2_uT": b0_l2.tolist(),
        "first_contact_b0_max_channel_uT": b0_channel.tolist(),
        "first_contact_b0_l2_max_uT": float(np.max(b0_l2)),
        "first_contact_b0_max_channel_max_uT": float(np.max(b0_channel)),
    }
    return delta, target, samples, metadata


def split_masks(target: np.ndarray) -> dict[str, np.ndarray]:
    xy = target[:, :2]
    depth = target[:, 2]
    anchor = anchor_mask(xy)
    in_working = depth >= 0.4 - 1e-6
    interior_depth = depth < 3.0 - 1e-6

    anchor_axes = [sorted(set(float(value) for value in xy[anchor, axis]))
                   for axis in range(2)]
    if [len(values) for values in anchor_axes] != [6, 6]:
        raise ValueError(f"expected 6x6 anchors, got {[len(v) for v in anchor_axes]}")
    x_index = {round(value, 6): index for index, value in enumerate(anchor_axes[0])}
    y_index = {round(value, 6): index for index, value in enumerate(anchor_axes[1])}
    inner_val_position = np.asarray([
        bool(is_anchor and ((x_index[round(float(x), 6)]
                             + 2 * y_index[round(float(y), 6)]) % 4 == 0))
        for (x, y), is_anchor in zip(xy, anchor)
    ])

    masks = {
        "selection_train": anchor & ~inner_val_position & in_working & interior_depth,
        "selection_val": anchor & inner_val_position & in_working & interior_depth,
        "refit_train": anchor & in_working & interior_depth,
        "spatial_holdout": ~anchor & in_working & interior_depth,
        "endpoint_holdout": in_working & ~interior_depth,
        "frozen_union": ((~anchor & in_working & interior_depth)
                         | (in_working & ~interior_depth)),
        "below_range_diagnostic": depth < 0.4 - 1e-6,
    }
    expected = {
        "selection_train": 351,
        "selection_val": 117,
        "refit_train": 468,
        "spatial_holdout": 1105,
        "endpoint_holdout": 121,
        "frozen_union": 1226,
        "below_range_diagnostic": 121,
    }
    observed = {name: int(mask.sum()) for name, mask in masks.items()}
    if observed != expected:
        raise ValueError(f"split mismatch: observed={observed}, expected={expected}")
    if np.any(masks["refit_train"] & masks["frozen_union"]):
        raise ValueError("frozen holdout leakage")
    return masks


def fit_normalization(features: np.ndarray, target: np.ndarray,
                      mask: np.ndarray) -> dict[str, np.ndarray]:
    x_mean = features[mask].mean(axis=0)
    x_scale = features[mask].std(axis=0)
    x_scale[x_scale < 1e-6] = 1.0
    y_mean = target[mask].mean(axis=0)
    y_scale = target[mask].std(axis=0)
    y_scale[y_scale < 1e-6] = 1.0
    return {"x_mean": x_mean, "x_scale": x_scale,
            "y_mean": y_mean, "y_scale": y_scale}


def normalize(features: np.ndarray, target: np.ndarray,
              norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return ((features - norm["x_mean"]) / norm["x_scale"],
            (target - norm["y_mean"]) / norm["y_scale"])


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = prediction - target
    abs_error = np.abs(error)
    xy_error = np.linalg.norm(error[:, :2], axis=1)
    xyz_error = np.linalg.norm(error, axis=1)
    return {
        "n": int(len(target)),
        "axis_mae_mm": abs_error.mean(axis=0).tolist(),
        "axis_rmse_mm": np.sqrt(np.mean(error * error, axis=0)).tolist(),
        "axis_bias_mm": error.mean(axis=0).tolist(),
        "xy_mean_mm": float(xy_error.mean()),
        "xy_rmse_mm": float(np.sqrt(np.mean(xy_error * xy_error))),
        "xy_p95_mm": float(np.percentile(xy_error, 95)),
        "z_mae_mm": float(abs_error[:, 2].mean()),
        "z_rmse_mm": float(np.sqrt(np.mean(error[:, 2] ** 2))),
        "z_p95_mm": float(np.percentile(abs_error[:, 2], 95)),
        "xyz_mean_mm": float(xyz_error.mean()),
        "xyz_rmse_mm": float(np.sqrt(np.mean(xyz_error * xyz_error))),
        "xyz_p95_mm": float(np.percentile(xyz_error, 95)),
        "xyz_max_mm": float(xyz_error.max()),
    }


@torch.no_grad()
def predict(model: nn.Module, features: np.ndarray,
            norm: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    model.eval()
    x = (features - norm["x_mean"]) / norm["x_scale"]
    result: list[np.ndarray] = []
    for start in range(0, len(x), 512):
        values = torch.as_tensor(x[start:start + 512], dtype=torch.float32, device=device)
        result.append(model(values).cpu().numpy())
    normalized = np.concatenate(result)
    return normalized * norm["y_scale"] + norm["y_mean"]


def train_selection(kind: str, features: np.ndarray, target: np.ndarray,
                    train_mask: np.ndarray, val_mask: np.ndarray, seed: int,
                    device: torch.device, max_epochs: int, patience: int,
                    batch_size: int, learning_rate: float) -> dict[str, Any]:
    set_seed(seed)
    norm = fit_normalization(features, target, train_mask)
    x, y = normalize(features, target, norm)
    x_train = torch.as_tensor(x[train_mask], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y[train_mask], dtype=torch.float32, device=device)
    model = make_model(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1009)
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            batch_x = x_train[index]
            # Small standardized-domain jitter prevents exact lookup while
            # remaining far below the observed cross-position signal changes.
            batch_x = batch_x + torch.randn_like(batch_x) * 0.01
            batch_y = y_train[index]
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x)
            loss = torch.nn.functional.smooth_l1_loss(output, batch_y, beta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_prediction = predict(model, features[val_mask], norm, device)
        score = metrics(val_prediction, target[val_mask])["xyz_mean_mm"]
        if score < best_score - 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("selection training produced no checkpoint")
    model.load_state_dict(best_state)
    val_prediction = predict(model, features[val_mask], norm, device)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "metrics": metrics(val_prediction, target[val_mask]),
    }


def train_refit(kind: str, features: np.ndarray, target: np.ndarray,
                train_mask: np.ndarray, seed: int, epochs: int,
                device: torch.device, batch_size: int, learning_rate: float
                ) -> tuple[nn.Module, dict[str, np.ndarray]]:
    set_seed(seed)
    norm = fit_normalization(features, target, train_mask)
    x, y = normalize(features, target, norm)
    x_train = torch.as_tensor(x[train_mask], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y[train_mask], dtype=torch.float32, device=device)
    model = make_model(kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2017)
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size].to(device)
            batch_x = x_train[index] + torch.randn_like(x_train[index]) * 0.01
            batch_y = y_train[index]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.smooth_l1_loss(
                model(batch_x), batch_y, beta=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model.eval(), norm


def serializable_norm(norm: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {name: value.tolist() for name, value in norm.items()}


def aggregate(rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return [float(np.mean(values)), float(np.std(values))]


def write_per_depth(path: Path, target: np.ndarray, predictions: dict[str, np.ndarray],
                    frozen_mask: np.ndarray) -> None:
    fields = ["architecture", "seed", "depth_mm", "n", "xy_mean_mm",
              "xy_p95_mm", "z_mae_mm", "xyz_mean_mm", "xyz_p95_mm"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, prediction in predictions.items():
            architecture, seed = key.split(":", 1)
            for depth in sorted(set(float(value) for value in target[frozen_mask, 2])):
                mask = frozen_mask & np.isclose(target[:, 2], depth)
                row = metrics(prediction[mask], target[mask])
                writer.writerow({
                    "architecture": architecture, "seed": seed,
                    "depth_mm": depth, "n": row["n"],
                    "xy_mean_mm": row["xy_mean_mm"],
                    "xy_p95_mm": row["xy_p95_mm"],
                    "z_mae_mm": row["z_mae_mm"],
                    "xyz_mean_mm": row["xyz_mean_mm"],
                    "xyz_p95_mm": row["xyz_p95_mm"],
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=ROOT / "data/S20260921_A")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "models/r48_static_grid_nn_20260922")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260920, 20260921, 20260922])
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    delta, target, samples, data_audit = load_contacts(args.session)
    masks = split_masks(target)
    role_order = ("selection_train", "selection_val", "spatial_holdout",
                  "endpoint_holdout", "below_range_diagnostic")
    split_labels = [
        next(name for name in role_order if masks[name][index])
        for index in range(len(target))
    ]
    np.savez_compressed(
        args.output / "dataset_contact_medians.npz",
        delta_b_uT=delta, xyz_mm=target,
        trial=np.asarray([sample["trial"] for sample in samples]),
        split=np.asarray(split_labels),
    )

    selection: dict[str, list[dict[str, Any]]] = {}
    feature_cache = {kind: engineered_features(kind, delta) for kind in ARCHITECTURES}
    for kind in ARCHITECTURES:
        selection[kind] = []
        for seed in args.seeds:
            row = train_selection(
                kind, feature_cache[kind], target,
                masks["selection_train"], masks["selection_val"], seed,
                device, args.max_epochs, args.patience, args.batch_size,
                args.learning_rate)
            selection[kind].append(row)
            print(f"[SELECT] {kind} seed={seed} epoch={row['best_epoch']} "
                  f"xyz={row['metrics']['xyz_mean_mm']:.4f}", flush=True)

    selected = min(
        ARCHITECTURES,
        key=lambda kind: np.mean([row["metrics"]["xyz_mean_mm"]
                                  for row in selection[kind]]),
    )
    print(f"[SELECTED] {selected}", flush=True)

    refits: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ARCHITECTURES}
    all_predictions: dict[str, np.ndarray] = {}
    for kind in ARCHITECTURES:
        for selection_row in selection[kind]:
            seed = int(selection_row["seed"])
            model, norm = train_refit(
                kind, feature_cache[kind], target, masks["refit_train"], seed,
                int(selection_row["best_epoch"]), device, args.batch_size,
                args.learning_rate)
            prediction = predict(model, feature_cache[kind], norm, device)
            all_predictions[f"{kind}:{seed}"] = prediction
            subsets = {
                name: metrics(prediction[mask], target[mask])
                for name, mask in masks.items()
                if name not in {"selection_train", "selection_val"}
            }
            checkpoint = args.output / f"{kind}_seed_{seed}_frozen.pt"
            torch.save({
                "schema": "MagTouch-R48-StaticGridNN-1.0",
                "architecture": kind,
                "seed": seed,
                "epochs": int(selection_row["best_epoch"]),
                "model_state": {name: value.detach().cpu()
                                for name, value in model.state_dict().items()},
                "normalization": serializable_norm(norm),
                "input": "15D delta-B" if kind == "direct_mlp_15"
                         else "15D unit shape + 15D delta-B + log(1+amplitude)",
                "output": "absolute [x,y,z] mm in the recorded printer-relative frame",
                "training_scope": "36 anchor positions, depth 0.4-2.8 mm",
                "frozen_test_not_used_for_selection": True,
            }, checkpoint)
            row = {
                "seed": seed,
                "epochs": int(selection_row["best_epoch"]),
                "selection_validation": selection_row["metrics"],
                "checkpoint": str(checkpoint.resolve()),
                "metrics": subsets,
            }
            refits[kind].append(row)
            frozen = subsets["frozen_union"]
            print(f"[FROZEN] {kind} seed={seed} "
                  f"xy={frozen['xy_mean_mm']:.4f} z={frozen['z_mae_mm']:.4f} "
                  f"xyz={frozen['xyz_mean_mm']:.4f}", flush=True)

    selected_rows = refits[selected]
    selected_seed = min(
        selection[selected], key=lambda row: row["metrics"]["xyz_mean_mm"])["seed"]
    result = {
        "schema": "MagTouch-R48-StaticGridNN-Backtest-1.0",
        "created_local": datetime.now().isoformat(timespec="seconds"),
        "offline_only": True,
        "hardware_acquisition_called": False,
        "source_session": str(args.session.resolve()),
        "source_trials": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        "working_range_mm": [0.4, 3.0],
        "below_range_diagnostic_mm": 0.2,
        "split_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "split_policy": {
            "architecture_selection": "27 anchor positions, depth 0.4-2.8 mm",
            "inner_validation": "9 disjoint anchor positions, depth 0.4-2.8 mm",
            "refit": "all 36 anchor positions, depth 0.4-2.8 mm",
            "frozen_spatial": "85 interleaved positions, depth 0.4-2.8 mm",
            "frozen_endpoint": "all 121 positions at 3.0 mm",
            "frozen_data_used_for_selection": False,
        },
        "data_audit": data_audit,
        "selection": selection,
        "selected_architecture": selected,
        "selected_seed_by_inner_validation": int(selected_seed),
        "refits": refits,
        "selected_three_seed_summary": {
            "frozen_union_xy_mean_mm": aggregate(
                selected_rows, ("metrics", "frozen_union", "xy_mean_mm")),
            "frozen_union_z_mae_mm": aggregate(
                selected_rows, ("metrics", "frozen_union", "z_mae_mm")),
            "frozen_union_xyz_mean_mm": aggregate(
                selected_rows, ("metrics", "frozen_union", "xyz_mean_mm")),
            "spatial_xy_mean_mm": aggregate(
                selected_rows, ("metrics", "spatial_holdout", "xy_mean_mm")),
            "spatial_z_mae_mm": aggregate(
                selected_rows, ("metrics", "spatial_holdout", "z_mae_mm")),
            "endpoint_xy_mean_mm": aggregate(
                selected_rows, ("metrics", "endpoint_holdout", "xy_mean_mm")),
            "endpoint_z_mae_mm": aggregate(
                selected_rows, ("metrics", "endpoint_holdout", "z_mae_mm")),
        },
        "limitations": [
            "One commanded contact exists per XY-depth combination; no repeatability claim.",
            "Labels are printer commands, not independent local metrology.",
            "The frozen test measures same-session interpolation and a depth endpoint only.",
            "Dynamic trajectories, random off-grid contacts and remount transfer remain untested.",
            "No checkpoint is promoted to the live UI by this offline experiment.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_per_depth(args.output / "per_depth_metrics.csv", target, all_predictions,
                    masks["frozen_union"])
    print(json.dumps(result["selected_three_seed_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
