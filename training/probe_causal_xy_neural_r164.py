#!/usr/bin/env python3
"""Small causal residual-MLP probe for had-239/5 mm XY shape.

Complete Trials are split chronologically. The already-viewed 30--32 set is
never used for weight/epoch/model selection. This is a diagnostic probe, not
a deployable or blind validation. Uses live gate and stage-command reference;
never imports or controls acquisition hardware.
"""
from __future__ import annotations

import json

import numpy as np

from training.diagnose_r126_latest_four_live_r130 import circle_shape, xy_summary
from training.evidence_guarded_branch_r162 import EvidenceGuardedBranchInverse
from training.evaluate_guarded_branch_r163 import load
from training.evaluate_branch_inverse_r139 import FIVE_CONFIG, FIVE_KEY, FIVE_REPORT
from training.had239_causal_xy_features_r165 import offline_features
from training.had239_causal_xy_mlp_runtime_r165 import ARTIFACT, METADATA, CausalXYMLP
from training.probe_had239_local_forward_r104 import ForwardAtlas, ROOT, load_profile, sha

OUT = ROOT / "analysis/r164_causal_xy_neural_probe_20260926/report_v6.json"
TRAIN = (11, 12, 13, 14, 15, 24, 25, 26)
VALIDATE = (27, 28, 29)
TEST = (30, 31, 32)


def features(data, atlas):
    if "_xy_features" not in data:
        data["_xy_features"] = offline_features(data, atlas)
    return data["_xy_features"]


def selected(data, atlas, stride=4):
    x = features(data, atlas)
    core = data["core"]
    choose = np.flatnonzero(core)[::stride]
    y = data["truth"][choose, :2] - data["guarded"][choose, :2]
    return x[choose], np.clip(y, -3., 3.)


def trial_metrics(model, data, atlas, base=None):
    core = data["core"]
    x = features(data, atlas)
    pred = data["guarded"].copy() if base is None else base.copy()
    pred[core, :2] += model.predict(x[core])
    current = data["guarded"] if base is None else base
    metric = {"trial": int(data["path"].name.split("_")[1]),
              "mode": data["mode"],
              "baseline": xy_summary(current[core], data["truth"][core]),
              "model": xy_summary(pred[core], data["truth"][core])}
    pair = core[1:] & core[:-1] & (np.diff(data["time"]) > 0) & (
        np.diff(data["time"]) < 120_000_000)
    if np.any(pair):
        steps = np.linalg.norm(np.diff(pred[:, :2], axis=0), axis=1)[pair]
        metric["jumps_over_1mm"] = int((steps > 1).sum())
        metric["max_jump_mm"] = float(steps.max())
    if data["mode"] == "circle_scan":
        metric["baseline_circle"] = circle_shape(current[core], data["truth"][core])
        metric["model_circle"] = circle_shape(pred[core], data["truth"][core])
    return metric


def guarded_positions(data, atlas):
    out = np.full_like(data["global"], np.nan)
    tracker = EvidenceGuardedBranchInverse(
        atlas, spatial_penalty=.03, innovation_gain=2.0,
        activation_distance_mm=1.0, max_local_cost_ratio=4.0)
    for i in range(len(out)):
        if not data["valid"][i]:
            tracker.reset()
            continue
        row = tracker.predict_frame(
            data["corrected"][i], time_ns=int(data["time"][i]),
            b0_id=0, stream_id=0, global_xyz=data["global"][i])
        out[i] = row["xyz_mm"]
    # R125 freezes depth to the static atlas; the stored live Z is the exact
    # depth consumed by the deployed branch+memory decoder.
    out[data["valid"], 2] = data["live"][data["valid"], 2]
    return out


class Ridge:
    def __init__(self, penalty):
        self.penalty = float(penalty)

    def fit(self, x, y):
        design = np.column_stack((np.ones(len(x)), x))
        reg = np.eye(design.shape[1]) * self.penalty
        reg[0, 0] = 0
        self.weight = np.linalg.solve(design.T @ design + reg, design.T @ y)
        return self

    def predict(self, x):
        return np.column_stack((np.ones(len(x)), x)) @ self.weight


class MLP:
    def __init__(self, width=32, seed=7):
        self.width = width
        self.rng = np.random.default_rng(seed)

    def fit(self, x, y, valid, atlas):
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), .25)
        train = (x - self.mean) / self.scale
        n, d = train.shape
        w1 = self.rng.normal(0, np.sqrt(2/d), (d, self.width))
        b1 = np.zeros(self.width)
        w2 = self.rng.normal(0, np.sqrt(2/self.width), (self.width, self.width))
        b2 = np.zeros(self.width)
        w3 = self.rng.normal(0, .03, (self.width, 2))
        b3 = np.zeros(2)
        params = [w1, b1, w2, b2, w3, b3]
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        step = 0
        best = np.inf
        best_params = None
        best_epoch = None
        history = []
        for epoch in range(1, 121):
            order = self.rng.permutation(n)
            for idx in np.array_split(order, max(1, (n + 255) // 256)):
                xx, yy = train[idx], y[idx]
                h1 = np.tanh(xx @ w1 + b1)
                h2 = np.tanh(h1 @ w2 + b2)
                out = h2 @ w3 + b3
                residual = out - yy
                delta3 = 2 * residual / len(idx)
                g3 = h2.T @ delta3 + 0.001 * w3
                gb3 = delta3.sum(axis=0)
                delta2 = (delta3 @ w3.T) * (1 - h2*h2)
                g2 = h1.T @ delta2 + 0.001 * w2
                gb2 = delta2.sum(axis=0)
                delta1 = (delta2 @ w2.T) * (1 - h1*h1)
                g1 = xx.T @ delta1 + 0.001 * w1
                gb1 = delta1.sum(axis=0)
                step += 1
                for p, grad, m, v in zip(params, (g1, gb1, g2, gb2, g3, gb3), first, second):
                    m *= .9
                    m += .1 * grad
                    v *= .999
                    v += .001 * grad*grad
                    p -= .001 * (m / (1 - .9**step)) / (
                        np.sqrt(v / (1 - .999**step)) + 1e-8)
            if epoch % 5 == 0:
                self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = params
                rows = [trial_metrics(self, data, atlas) for data in valid]
                objective = float(np.mean([row["model"]["xy_mean_mm"] for row in rows]))
                history.append({"epoch": epoch, "validation_trial_mean_xy_mm": objective})
                if objective < best:
                    best, best_epoch = objective, epoch
                    best_params = [p.copy() for p in params]
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = best_params
        self.best_epoch = best_epoch
        self.best_validation = best
        self.history = history
        return self

    def predict(self, x):
        x = (x - self.mean) / self.scale
        return np.tanh(np.tanh(x @ self.w1 + self.b1) @ self.w2 + self.b2) @ self.w3 + self.b3


def main():
    if OUT.exists() or ARTIFACT.exists() or METADATA.exists():
        raise FileExistsError("R165 training output already exists")
    profile = load_profile(FIVE_CONFIG, FIVE_KEY)
    samples = profile["samples"]
    xyz = np.asarray([[s["x"], s["y"], s["depth"]] for s in samples])
    delta = np.asarray([s["delta"] for s in samples])
    atlas = ForwardAtlas(xyz, delta, "unit_logamp")
    memory_report = json.loads(FIVE_REPORT.read_text(encoding="utf-8"))
    basis = np.asarray(memory_report["basis"], float)
    strength = float(memory_report["selection"]["selected"]["strength"])
    data = {n: load(n, atlas, basis, strength) for n in TRAIN + VALIDATE + TEST}
    for row in data.values():
        row["guarded"] = guarded_positions(row, atlas)
    train_x, train_y = [], []
    for n in TRAIN:
        x, y = selected(data[n], atlas)
        train_x.append(x)
        train_y.append(y)
    # Physically valid no-motion atlas states regularize the spatial mapping;
    # only measured static atlas samples, never command trajectories, are used.
    synthetic = np.column_stack((xyz[::3] / np.array([10., 10., 3.]),
                                 atlas.features(delta[::3]),
                                 np.zeros((len(xyz[::3]), 34))))
    train_x.append(synthetic)
    train_y.append(np.zeros((len(synthetic), 2)))
    x = np.vstack(train_x)
    y = np.vstack(train_y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("non-finite training features")
    candidates = {
        "ridge_10": Ridge(10).fit(x, y),
        "ridge_100": Ridge(100).fit(x, y),
        "mlp32": MLP(32).fit(x, y, [data[n] for n in VALIDATE], atlas),
        "mlp64": MLP(64).fit(x, y, [data[n] for n in VALIDATE], atlas),
    }
    results = {}
    for name, model in candidates.items():
        val = [trial_metrics(model, data[n], atlas) for n in VALIDATE]
        test = [trial_metrics(model, data[n], atlas) for n in TEST]
        results[name] = {"validation": val, "test": test,
                         "selected_epoch": getattr(model, "best_epoch", None),
                         "epoch_history": getattr(model, "history", None)}
        print(name, "validation", [(r["trial"], round(r["model"]["xy_mean_mm"], 3)) for r in val],
              "test", [(r["trial"], round(r["model"]["xy_mean_mm"], 3)) for r in test], flush=True)
    report = {"schema": "MagTouch-R164-CausalXYResidualProbe-1",
              "warning": "Diagnostic only. Previously viewed stage-command trajectories are not independently measured tip positions or blind validation; this probe does not include branch tracking or contact-gate evaluation.",
              "train_trials": TRAIN, "validation_trials": VALIDATE,
              "test_trials": TEST, "train_samples": int(len(x)),
              "train_real_samples": int(len(x) - len(synthetic)),
              "static_anchor_samples": int(len(synthetic)),
              "feature_description": "global XYZ, current corrected 16D unit+logamp field, 2D magnetic memory, past-only 0.1/0.4s 16D field differences",
              "source_sha256": {str(n): sha(data[n]["path"] / "live_predictions.csv") for n in data},
              "results": results,
              "branch_guard_max_local_cost_ratio": 4.0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected_model = candidates["mlp64"]
    mlp32_score = float(np.mean([v["model"]["xy_mean_mm"] for v in results["mlp32"]["validation"]]))
    mlp64_score = float(np.mean([v["model"]["xy_mean_mm"] for v in results["mlp64"]["validation"]]))
    if mlp64_score > mlp32_score:
        raise ValueError("MLP64 is not the validation winner")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ARTIFACT, mean=selected_model.mean, scale=selected_model.scale,
                        w1=selected_model.w1, b1=selected_model.b1,
                        w2=selected_model.w2, b2=selected_model.b2,
                        w3=selected_model.w3, b3=selected_model.b3)
    info = {"schema": "MagTouch-R165-CausalXYMLP-1",
            "artifact_sha256": sha(ARTIFACT), "report_sha256": sha(OUT),
            "static_config_sha256": sha(FIVE_CONFIG),
            "memory_report_sha256": sha(FIVE_REPORT),
            "train_trials": TRAIN, "validation_trials": VALIDATE,
            "test_trials": TEST, "selected_epoch": selected_model.best_epoch,
            "selected_width": 64,
            "selection_rule": "Lowest mean trial XY on validation Trials 27-29; test Trials 30-32 excluded",
            "validation_mlp32_mean_xy_mm": mlp32_score,
            "validation_mlp64_mean_xy_mm": mlp64_score,
            "use": "experimental had-239/5mm only, applied after R162 ratio4 guard; corrects XY only",
            "source_prediction_sha256": report["source_sha256"]}
    METADATA.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen = CausalXYMLP()
    test_feature = features(data[30], atlas)[data[30]["core"]][:8]
    mismatch = float(np.max(np.abs(frozen.predict(test_feature) - selected_model.predict(test_feature))))
    if mismatch > 1e-10:
        raise ValueError(f"frozen inference mismatch {mismatch}")
    print(OUT, flush=True)
    print(METADATA, flush=True)


if __name__ == "__main__":
    main()
