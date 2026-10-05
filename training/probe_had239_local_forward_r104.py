#!/usr/bin/env python3
"""R104: offline, non-neural forward-field interpolation/inversion probe.

No acquisition or UI code is imported. The 36->85 comparison uses the exact
R92 position split. R92 and the forward atlas are scored on identical samples.
The five trajectory Trials were viewed during prior development; this is a
diagnostic comparison, not a new blind validation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code")]
from replay_backtest import load_profile  # noqa: E402
from training.backtest_had239_legacy_formal_r92 import direct_predict  # noqa: E402
from training.build_fixed_magnet_domains_r49 import extract_contacts  # noqa: E402

CONFIG = ROOT / "config/had239_legacy_formal_continuous_r92_20260926.json"
INDEX = ROOT / "datasets/MagTouch_FinalTip_Trajectory_R52_20260922/index.jsonl"
SEQUENCES = INDEX.parent
OUT = ROOT / "analysis/r104_had239_local_forward_20260926/report.json"
PROFILE_KEY = "5x5_had239_final5mm_old_formal_r92"
MOTION_TRIALS = (41, 42, 50, 54, 59)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    xy = np.linalg.norm(err[:, :2], axis=1)
    return {
        "n": len(err), "xy_mean_mm": float(xy.mean()),
        "xy_p95_mm": float(np.percentile(xy, 95)),
        "z_mae_mm": float(np.abs(err[:, 2]).mean()),
        "xyz_mean_mm": float(np.linalg.norm(err, axis=1).mean()),
        "signed_mean_xyz_mm": err.mean(axis=0).tolist(),
    }


class ForwardAtlas:
    """Piecewise trilinear 3D magnetic atlas on the measured XYZ lattice."""

    def __init__(self, xyz: np.ndarray, delta: np.ndarray, metric: str,
                 *, allow_irregular_z: bool = False, smooth_xy: bool = False,
                 spatial_ridge: float = 0.0):
        self.x = np.unique(xyz[:, 0])
        self.y = np.unique(xyz[:, 1])
        self.z = np.unique(xyz[:, 2])
        self.field = np.full((len(self.x), len(self.y), len(self.z), 15), np.nan)
        for p, b in zip(xyz, delta):
            i = int(np.searchsorted(self.x, p[0]))
            j = int(np.searchsorted(self.y, p[1]))
            k = int(np.searchsorted(self.z, p[2]))
            if not np.allclose((self.x[i], self.y[j], self.z[k]), p, atol=1e-6):
                raise ValueError(f"off-grid position {p}")
            if np.isfinite(self.field[i, j, k]).any():
                raise ValueError(f"duplicate atlas position {p}")
            self.field[i, j, k] = b
        if (not np.isfinite(self.field).all() or
                (not allow_irregular_z and
                 not np.allclose(np.diff(self.z), .2, atol=1e-6))):
            raise ValueError("atlas not a complete regular 121-point/depth grid")
        if not np.isfinite(spatial_ridge) or spatial_ridge < 0:
            raise ValueError("invalid spatial ridge")
        self.spatial_ridge = float(spatial_ridge)
        if self.spatial_ridge:
            size = len(self.x) * len(self.y)
            laplacian = np.zeros((size, size), float)
            for ix in range(len(self.x)):
                for iy in range(len(self.y)):
                    idx = ix * len(self.y) + iy
                    for nx, ny in ((ix - 1, iy), (ix + 1, iy),
                                   (ix, iy - 1), (ix, iy + 1)):
                        if 0 <= nx < len(self.x) and 0 <= ny < len(self.y):
                            other = nx * len(self.y) + ny
                            laplacian[idx, idx] += 1
                            laplacian[idx, other] -= 1
            grouped = self.field.reshape(size, -1)
            self.field = np.linalg.solve(
                np.eye(size) + self.spatial_ridge * laplacian,
                grouped).reshape(self.field.shape)
        if smooth_xy not in (False, True, "pchip"):
            raise ValueError("smooth_xy must be False, True or pchip")
        self.smooth_xy = smooth_xy
        if self.smooth_xy and (not np.allclose(np.diff(self.x), np.diff(self.x)[0]) or
                               not np.allclose(np.diff(self.y), np.diff(self.y)[0])):
            raise ValueError("smooth XY requires regular grid axes")
        self.metric = metric
        self.amplitude_scale = float(np.median(np.linalg.norm(delta, axis=1)))
        self.channel_scale = np.maximum(np.std(delta, axis=0), 10.)
        self.candidates = self._candidate_positions()
        self.candidate_field = self.forward(self.candidates)
        self.candidate_features = self.features(self.candidate_field).astype(np.float32)
        self.candidate_norm = np.sum(self.candidate_features ** 2, axis=1)

    def _candidate_positions(self) -> np.ndarray:
        x = np.arange(self.x[0], self.x[-1] + .0001, 1.)
        y = np.arange(self.y[0], self.y[-1] + .0001, 1.)
        z = self.z
        return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

    def forward(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float).reshape(-1, 3)
        indices, t = [], []
        for col, axis in enumerate((self.x, self.y, self.z)):
            q = np.clip(p[:, col], axis[0], axis[-1])
            lower = np.clip(np.searchsorted(axis, q, side="right") - 1,
                            0, len(axis) - 2)
            indices.append(lower)
            t.append(((q - axis[lower]) / (axis[lower + 1] - axis[lower]))[:, None])
        i, j, k = indices
        a, b, c = t
        if self.smooth_xy:
            offsets = np.array([-1, 0, 1, 2])
            xi = np.clip(i[:, None] + offsets[None, :], 0, len(self.x) - 1)
            yj = np.clip(j[:, None] + offsets[None, :], 0, len(self.y) - 1)
            if self.smooth_xy == "pchip":
                def hermite(values: np.ndarray, u: np.ndarray,
                            lower: np.ndarray, axis_size: int,
                            spacing: float) -> np.ndarray:
                    d0 = (values[:, 1] - values[:, 0]) / spacing
                    d1 = (values[:, 2] - values[:, 1]) / spacing
                    d2 = (values[:, 3] - values[:, 2]) / spacing
                    def harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
                        same = (left * right) > 0
                        denominator = np.where(same, left + right, 1.)
                        return np.where(same, 2 * left * right / denominator, 0.)
                    m1 = np.where(lower[:, None] == 0, d1,
                                  harmonic(d0, d1))
                    m2 = np.where(lower[:, None] == axis_size - 2, d1,
                                  harmonic(d1, d2))
                    s = u[:, None]
                    s2, s3 = s*s, s*s*s
                    return ((2*s3 - 3*s2 + 1) * values[:, 1] +
                            (s3 - 2*s2 + s) * spacing * m1 +
                            (-2*s3 + 3*s2) * values[:, 2] +
                            (s3 - s2) * spacing * m2)
                out = np.zeros((len(p), 15), float)
                for dz in (0, 1):
                    y_values = []
                    for iy in range(4):
                        samples = self.field[xi, yj[:, iy, None], k[:, None] + dz]
                        y_values.append(hermite(
                            samples, a[:, 0], i, len(self.x),
                            float(self.x[1] - self.x[0])))
                    plane = hermite(np.stack(y_values, axis=1), b[:, 0],
                                    j, len(self.y), float(self.y[1] - self.y[0]))
                    out += (c if dz else 1 - c) * plane
                return out
            def weights(u: np.ndarray) -> np.ndarray:
                u2, u3 = u * u, u * u * u
                return np.column_stack((-.5*u + u2 - .5*u3,
                                        1 - 2.5*u2 + 1.5*u3,
                                        .5*u + 2*u2 - 1.5*u3,
                                        -.5*u2 + .5*u3))
            wx, wy = weights(a[:, 0]), weights(b[:, 0])
            out = np.zeros((len(p), 15), float)
            for dz in (0, 1):
                plane = np.zeros_like(out)
                for iy in range(4):
                    stripe = np.zeros_like(out)
                    for ix in range(4):
                        stripe += wx[:, ix, None] * self.field[
                            xi[:, ix], yj[:, iy], k + dz]
                    plane += wy[:, iy, None] * stripe
                out += (c if dz else 1 - c) * plane
            return out
        out = np.zeros((len(p), 15), float)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    weight = (a if dx else 1 - a) * (b if dy else 1 - b) * (
                        c if dz else 1 - c)
                    out += weight * self.field[i + dx, j + dy, k + dz]
        return out

    def features(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, float)
        if self.metric == "raw":
            return delta / self.amplitude_scale
        if self.metric == "channel_scaled":
            return delta / self.channel_scale
        if self.metric == "unit_logamp":
            amplitude = np.linalg.norm(delta, axis=1)
            unit = delta / np.maximum(amplitude[:, None], 1e-6)
            logamp = .5 * np.log(np.maximum(amplitude, 1.) /
                                self.amplitude_scale)
            return np.column_stack((unit, logamp))
        raise ValueError(self.metric)

    def predict(self, delta: np.ndarray, refine: bool = True) -> np.ndarray:
        delta = np.asarray(delta, float).reshape(-1, 15)
        features = self.features(delta).astype(np.float32)
        best = np.empty(len(delta), np.int32)
        for lo in range(0, len(delta), 256):
            hi = min(lo + 256, len(delta))
            part = features[lo:hi]
            distance = (np.sum(part ** 2, axis=1)[:, None] +
                        self.candidate_norm[None, :] -
                        2 * part @ self.candidate_features.T)
            best[lo:hi] = np.argmin(distance, axis=1)
        point = self.candidates[best].copy()
        if not refine:
            return point
        current_feature = self.features(self.forward(point))
        current_cost = np.sum((features - current_feature) ** 2, axis=1)
        for _ in range(2):
            jacobian = np.empty((len(point), features.shape[1], 3), float)
            for column, step in enumerate((.05, .05, .02)):
                plus = point.copy()
                minus = point.copy()
                plus[:, column] += step
                minus[:, column] -= step
                jacobian[:, :, column] = (
                    self.features(self.forward(plus)) -
                    self.features(self.forward(minus))) / (2 * step)
            residual = features - current_feature
            jtj = np.einsum("nfi,nfj->nij", jacobian, jacobian)
            jtr = np.einsum("nfi,nf->ni", jacobian, residual)
            damping = .01 * np.trace(jtj, axis1=1, axis2=2) / 3 + 1e-7
            jtj[:, range(3), range(3)] += damping[:, None]
            delta_p = np.linalg.solve(jtj, jtr[..., None])[..., 0]
            delta_p = np.clip(delta_p, [-.5, -.5, -.1], [.5, .5, .1])
            proposal = point + delta_p
            for column, axis in enumerate((self.x, self.y, self.z)):
                proposal[:, column] = np.clip(proposal[:, column], axis[0], axis[-1])
            next_feature = self.features(self.forward(proposal))
            next_cost = np.sum((features - next_feature) ** 2, axis=1)
            accept = next_cost < current_cost
            point[accept] = proposal[accept]
            current_feature[accept] = next_feature[accept]
            current_cost[accept] = next_cost[accept]
        return point


def samples(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = profile["samples"]
    xyz = np.asarray([[s["x"], s["y"], s["depth"]] for s in rows], float)
    delta = np.asarray([s["delta"] for s in rows], float)
    if len(rows) != 1694 or xyz.shape != (1694, 3) or delta.shape != (1694, 15):
        raise ValueError("unexpected had-239 5mm source sample geometry")
    return xyz, delta


def external_static(profile: dict, atlases: dict[str, ForwardAtlas]) -> list[dict]:
    results = []
    for n in (1, 3):
        paths = list((ROOT / "data/S20260923_A").glob(f"Trial_{n:06d}_*"))
        if len(paths) != 1:
            raise ValueError(paths)
        path = paths[0]
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if (meta["status"] != "completed" or
            meta["acquisition"]["magnetic_layer_profile_id"] !=
                "magnetic_layer_v1_5*5-had-239"):
            raise ValueError(f"wrong physical profile: {path}")
        contacts, extraction = extract_contacts(path)
        if extraction["errors"] or not contacts:
            raise ValueError(extraction)
        delta = np.asarray([c["delta"] for c in contacts], float)
        xyz = np.asarray([[c["x"], c["y"], c["z"]] for c in contacts], float)
        xyz[:, 2] += profile["training_contact_z_mm"] - extraction["contact_z_mm"]
        results.append({
            "trial": n, "source": path.relative_to(ROOT).as_posix(),
            "source_meta_sha256": sha(path / "meta.json"),
            "contact_z_mm": extraction["contact_z_mm"],
            "n": len(contacts), "r92": score(direct_predict(profile, delta), xyz),
            "atlases": {name: score(model.predict(delta), xyz)
                        for name, model in atlases.items()},
        })
    return results


def motion(profile: dict, atlases: dict[str, ForwardAtlas]) -> list[dict]:
    index = {int(row["trial"]): row for line in INDEX.read_text(encoding="utf-8").splitlines()
             if line for row in [json.loads(line)]}
    results = []
    for number in MOTION_TRIALS:
        row = index[number]
        if row["split"] != "test":
            raise ValueError(f"Trial {number} is not prior test split")
        src = Path(row["source"])
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        if (meta["acquisition"]["magnetic_layer_profile_id"] !=
                "magnetic_layer_v1_5*5-had-239"):
            raise ValueError(f"wrong physical profile: {src}")
        with np.load(SEQUENCES / row["sequence"], allow_pickle=False) as a:
            eligible = (a["input_valid"].astype(bool) &
                        a["xyz_train_mask"].astype(bool) &
                        a["command_moving"].astype(bool) &
                        (np.arange(len(a["t_ns"])) % 4 == 0))
            d = a["delta_b_uT"][eligible].astype(float)
            xyz = a["xyz_mm"][eligible].astype(float)
        results.append({
            "trial": number, "family": row["family"], "n": len(xyz),
            "source": src.relative_to(ROOT).as_posix(),
            "source_meta_sha256": sha(src / "meta.json"),
            "sequence_sha256": sha(SEQUENCES / row["sequence"]),
            "r92": score(direct_predict(profile, d), xyz),
            "atlases": {name: score(model.predict(d), xyz)
                        for name, model in atlases.items()},
        })
        print("[R104 motion]", number, results[-1]["r92"]["xy_mean_mm"],
              {name: round(val["xy_mean_mm"], 3)
               for name, val in results[-1]["atlases"].items()}, flush=True)
    return results


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"R104 report already exists: {OUT}")
    profile = load_profile(CONFIG, PROFILE_KEY)
    xyz, delta = samples(profile)
    anchor = np.isin(xyz[:, 0], [-10, -6, -2, 2, 6, 10]) & np.isin(
        xyz[:, 1], [-10, -6, -2, 2, 6, 10])
    if anchor.sum() != 36 * 14 or (~anchor).sum() != 85 * 14:
        raise ValueError("frozen spatial split changed")
    target = xyz[~anchor]
    metric_names = ("raw", "channel_scaled", "unit_logamp")
    frozen = {}
    for name in metric_names:
        atlas = ForwardAtlas(xyz[anchor], delta[anchor], name)
        pred = atlas.predict(delta[~anchor])
        frozen[name] = score(pred, target)
        print("[R104 frozen]", name, frozen[name], flush=True)
    all_atlases = {name: ForwardAtlas(xyz, delta, name) for name in metric_names}
    static = external_static(profile, all_atlases)
    dynamic = motion(profile, all_atlases)
    result = {
        "schema": "MagTouch-R104-LocalForwardInterpolationProbe-1",
        "scope": "offline diagnostic only; no model selected or deployed",
        "model_definition": "3D trilinear XYZ->15D delta-B atlas, 1mm XY x measured-depth global inverse search, 2 damped local Gauss-Newton steps; metrics raw/channel-scaled/unit-plus-log-amplitude",
        "reference_warning": "motion truth is stage command interpolation, not measured tip XYZ; prior test Trials were viewed during model development",
        "config": CONFIG.relative_to(ROOT).as_posix(),
        "config_sha256": sha(CONFIG),
        "source_contacts": len(xyz),
        "anchor_contacts": int(anchor.sum()),
        "unseen_contacts": int((~anchor).sum()),
        "r92_frozen_old_formula": profile["validation"]["frozen_spatial"],
        "frozen_spatial": frozen,
        "crossday_static": static,
        "motion_prior_holdout": dynamic,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
