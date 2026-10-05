"""Direct source-data magnetic fingerprint calibration for MagTouch.

The routines in this module deliberately do not call an XY predictor.  They
compare a small set of current, B0-subtracted 15-channel contact fingerprints
with the raw ``samples[].delta`` vectors stored in the selected fit profile.
The fitted rigid transform maps coordinates in the source/training frame to
coordinates in the current printer frame.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_vector(value: Any, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"磁场指纹必须是 {size} 维有限数值")
    return result


def build_source_bank(profile: Mapping[str, Any], depth_mm: float) -> dict[str, Any]:
    """Build one source fingerprint per XY node at ``depth_mm``.

    Repeated samples at an identical XY/depth are reduced by their component
    median.  If the requested depth lies between two acquired depths, the two
    raw delta vectors are linearly interpolated.  Extrapolation is rejected.
    """
    depth = float(depth_mm)
    if not math.isfinite(depth):
        raise ValueError("校准压深必须为有限数值")
    grouped: dict[tuple[float, float], dict[float, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in profile.get("samples", []):
        try:
            xy = (round(float(row["x"]), 8), round(float(row["y"]), 8))
            z = round(float(row["depth"]), 8)
            delta = _finite_vector(row["delta"], 15)
        except (KeyError, TypeError, ValueError):
            continue
        grouped[xy][z].append(delta)
    if len(grouped) < 5:
        raise ValueError("所选模型源数据不足5个有效XY位置，不能进行五点源指纹校准")

    nodes: list[tuple[float, float]] = []
    vectors: list[np.ndarray] = []
    modes: list[str] = []
    used_depths: list[list[float]] = []
    for xy, by_depth in grouped.items():
        levels = sorted((float(z), np.median(np.vstack(rows), axis=0))
                        for z, rows in by_depth.items())
        exact = [item for item in levels if abs(item[0] - depth) <= 1e-7]
        if exact:
            vector = exact[0][1]
            mode = "exact"
            pair = [exact[0][0]]
        else:
            lower = [item for item in levels if item[0] < depth]
            upper = [item for item in levels if item[0] > depth]
            if not lower or not upper:
                continue
            lo_z, lo_v = lower[-1]
            hi_z, hi_v = upper[0]
            weight = (depth - lo_z) / (hi_z - lo_z)
            vector = (1.0 - weight) * lo_v + weight * hi_v
            mode = "depth_interpolated"
            pair = [lo_z, hi_z]
        if float(np.linalg.norm(vector)) <= 1e-8:
            continue
        nodes.append(xy)
        vectors.append(vector)
        modes.append(mode)
        used_depths.append(pair)
    if len(nodes) < 5:
        available = sorted({float(z) for by_depth in grouped.values() for z in by_depth})
        raise ValueError(
            f"源数据在 {depth:g} mm 不能构造至少5个XY指纹；可用压深={available}"
        )
    xy_array = np.asarray(nodes, dtype=float)
    vector_array = np.vstack(vectors)
    nearest = []
    for index, point in enumerate(xy_array):
        distance = np.linalg.norm(xy_array - point, axis=1)
        distance[index] = np.inf
        finite = distance[np.isfinite(distance)]
        if len(finite):
            nearest.append(float(np.min(finite)))
    spacing = float(np.median(nearest)) if nearest else 1.0
    return {
        "xy": xy_array,
        "vectors": vector_array,
        "depth_mm": depth,
        "depth_modes": modes,
        "used_depths_mm": used_depths,
        "spacing_mm": max(spacing, 0.25),
        "domain_mm": {
            "x": [float(np.min(xy_array[:, 0])), float(np.max(xy_array[:, 0]))],
            "y": [float(np.min(xy_array[:, 1])), float(np.max(xy_array[:, 1]))],
        },
    }


def _interpolate_bank(bank: Mapping[str, Any], query_xy: np.ndarray) -> np.ndarray | None:
    nodes = np.asarray(bank["xy"], dtype=float)
    vectors = np.asarray(bank["vectors"], dtype=float)
    query = np.asarray(query_xy, dtype=float).reshape(2)
    domain = bank["domain_mm"]
    margin = 0.25 * float(bank["spacing_mm"])
    if not (float(domain["x"][0]) - margin <= query[0] <= float(domain["x"][1]) + margin and
            float(domain["y"][0]) - margin <= query[1] <= float(domain["y"][1]) + margin):
        return None
    distance2 = np.sum((nodes - query[None, :]) ** 2, axis=1)
    nearest = int(np.argmin(distance2))
    if distance2[nearest] <= 1e-12:
        return vectors[nearest].copy()
    count = min(8, len(nodes))
    selected = np.argpartition(distance2, count - 1)[:count]
    sigma = max(0.75 * float(bank["spacing_mm"]), 0.35)
    weights = np.exp(-distance2[selected] / (2.0 * sigma * sigma))
    total = float(np.sum(weights))
    if total <= 1e-12:
        return vectors[nearest].copy()
    return np.sum(vectors[selected] * (weights / total)[:, None], axis=0)


def _candidate_metrics(bank: Mapping[str, Any], reference_xy: np.ndarray,
                       current_vectors: np.ndarray, angle_deg: float,
                       tx_mm: float, ty_mm: float) -> dict[str, Any] | None:
    angle = math.radians(float(angle_deg))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[c, -s], [s, c]], dtype=float)
    translation = np.asarray([tx_mm, ty_mm], dtype=float)
    # q_current = R * p_source + t  ->  p_source = R.T * (q_current - t)
    source_xy = (reference_xy - translation[None, :]) @ rotation
    source_vectors = []
    for point in source_xy:
        vector = _interpolate_bank(bank, point)
        if vector is None:
            return None
        source_vectors.append(vector)
    source = np.vstack(source_vectors)
    current_norm = np.linalg.norm(current_vectors, axis=1)
    source_norm = np.linalg.norm(source, axis=1)
    if np.any(current_norm <= 1e-8) or np.any(source_norm <= 1e-8):
        return None
    cosine = np.sum(current_vectors * source, axis=1) / (current_norm * source_norm)
    ratios = current_norm / source_norm
    scale = float(np.median(ratios))
    if not math.isfinite(scale) or scale <= 1e-8:
        return None
    amplitude_log_residual = np.abs(np.log(np.maximum(ratios / scale, 1e-12)))
    vector_relative_residual = np.linalg.norm(
        current_vectors - scale * source, axis=1
    ) / current_norm
    loss = (float(np.mean(1.0 - np.clip(cosine, -1.0, 1.0))) +
            0.15 * float(np.median(amplitude_log_residual)) +
            0.20 * float(np.median(vector_relative_residual)))
    return {
        "loss": loss,
        "source_xy": source_xy,
        "source_vectors": source,
        "cosine": cosine,
        "amplitude_scale": scale,
        "amplitude_ratios": ratios,
        "amplitude_log_residual": amplitude_log_residual,
        "vector_relative_residual": vector_relative_residual,
    }


def _axis_values(center: float, radius: float, step: float) -> list[float]:
    count = int(round((2.0 * radius) / step))
    return [center - radius + step * index for index in range(count + 1)]


def fit_source_fingerprint_alignment(
        profile: Mapping[str, Any], captures: Sequence[Mapping[str, Any]],
        depth_mm: float, translation_limit_mm: float = 6.0,
        rotation_limit_deg: float = 12.0) -> dict[str, Any]:
    """Fit source-frame -> current-printer-frame rigid alignment from raw delta-B."""
    if len(captures) < 5:
        raise ValueError("源指纹校准至少需要中心＋四方向共5个点")
    reference_xy = np.asarray(
        [[float(row["reference_xy_mm"][0]), float(row["reference_xy_mm"][1])]
         for row in captures], dtype=float)
    current = np.vstack([_finite_vector(row["current_delta_uT"], 15)
                         for row in captures])
    if len(np.unique(np.round(reference_xy, 6), axis=0)) < 5:
        raise ValueError("五点源指纹校准的打印机坐标必须互不相同")
    bank = build_source_bank(profile, depth_mm)
    t_limit = float(translation_limit_mm)
    r_limit = float(rotation_limit_deg)
    if not (0.25 <= t_limit <= 10.0 and 0.5 <= r_limit <= 30.0):
        raise ValueError("源指纹搜索范围不安全")

    best: tuple[float, float, float, dict[str, Any]] | None = None
    levels = [
        ((0.0, 0.0, 0.0), (t_limit, t_limit, r_limit), (1.0, 1.0, 2.0)),
        (None, (1.0, 1.0, 2.0), (0.25, 0.25, 0.5)),
        (None, (0.25, 0.25, 0.5), (0.05, 0.05, 0.1)),
    ]
    for center_spec, radius, step in levels:
        if center_spec is None:
            if best is None:
                raise RuntimeError("源指纹粗搜索没有有效候选")
            center = (best[0], best[1], best[2])
        else:
            center = center_spec
        for tx in _axis_values(center[0], radius[0], step[0]):
            if abs(tx) > t_limit + 1e-9:
                continue
            for ty in _axis_values(center[1], radius[1], step[1]):
                if abs(ty) > t_limit + 1e-9:
                    continue
                for angle in _axis_values(center[2], radius[2], step[2]):
                    if abs(angle) > r_limit + 1e-9:
                        continue
                    metrics = _candidate_metrics(
                        bank, reference_xy, current, angle, tx, ty
                    )
                    if metrics is None:
                        continue
                    if best is None or float(metrics["loss"]) < float(best[3]["loss"]):
                        best = (float(tx), float(ty), float(angle), metrics)
    if best is None:
        raise RuntimeError("源指纹搜索失败：候选点超出源数据覆盖范围")
    tx, ty, angle, metrics = best
    cosine = np.asarray(metrics["cosine"], dtype=float)
    relative = np.asarray(metrics["vector_relative_residual"], dtype=float)
    ratios = np.asarray(metrics["amplitude_ratios"], dtype=float)
    scale = float(metrics["amplitude_scale"])
    scale_spread = float(np.median(np.abs(ratios / scale - 1.0)))
    median_cosine = float(np.median(cosine))
    min_cosine = float(np.min(cosine))
    median_relative = float(np.median(relative))
    if median_cosine >= 0.96 and min_cosine >= 0.85 and median_relative <= 0.40:
        quality = "pass"
    elif median_cosine >= 0.85 and min_cosine >= 0.60 and median_relative <= 0.75:
        quality = "warning"
    else:
        quality = "fail"
    warnings: list[str] = []
    if quality == "warning":
        warnings.append("源指纹只达到警告级匹配，建议用独立点复核")
    elif quality == "fail":
        warnings.append("源指纹不能由统一平移/旋转可靠解释，未自动应用校准")
    if not 0.65 <= scale <= 1.50:
        warnings.append(f"整体幅值比={scale:.3f}，疑似高度/预压/灵敏度变化")
    if scale_spread > 0.25:
        warnings.append(f"五点幅值比离散={scale_spread:.3f}，疑似倾斜或非刚性变化")
    if abs(tx) >= t_limit - 0.11 or abs(ty) >= t_limit - 0.11:
        warnings.append("平移结果接近搜索边界")
    if abs(angle) >= r_limit - 0.21:
        warnings.append("旋转结果接近搜索边界")

    source_xy = np.asarray(metrics["source_xy"], dtype=float)
    nodes = np.asarray(bank["xy"], dtype=float)
    direct_rows = []
    for index, (capture, matched) in enumerate(zip(captures, source_xy)):
        nearest = int(np.argmin(np.linalg.norm(nodes - matched[None, :], axis=1)))
        direct_rows.append({
            "label": str(capture.get("label", index)),
            "current_reference_xy_mm": reference_xy[index].tolist(),
            "matched_source_xy_mm": matched.tolist(),
            "nearest_source_node_xy_mm": nodes[nearest].tolist(),
            "cosine_similarity": float(cosine[index]),
            "amplitude_ratio": float(ratios[index]),
            "vector_relative_residual": float(relative[index]),
        })
    return {
        "strategy": "source_delta_b_fingerprint_rigid_v1",
        "rotation_deg": angle,
        "offset_mm": [tx, ty],
        "depth_mm": float(depth_mm),
        "source_node_count": int(len(nodes)),
        "source_grid_spacing_mm": float(bank["spacing_mm"]),
        "source_domain_mm": bank["domain_mm"],
        "depth_interpolated_node_count": int(sum(
            mode == "depth_interpolated" for mode in bank["depth_modes"]
        )),
        "loss": float(metrics["loss"]),
        "quality": quality,
        "median_cosine": median_cosine,
        "minimum_cosine": min_cosine,
        "amplitude_scale": scale,
        "amplitude_scale_median_relative_spread": scale_spread,
        "median_vector_relative_residual": median_relative,
        "warnings": warnings,
        "point_matches": direct_rows,
    }

