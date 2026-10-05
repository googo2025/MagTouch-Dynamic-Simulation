"""Raw 15-channel magnetic helpers for conservative contact-Z calibration.

This module deliberately contains no model or contact-classifier dependency.  It
operates on the five sensors' 15 converted raw magnetic channels (uT) before any
learned XYZ regression.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence


CHANNEL_COUNT = 15


def flatten_group_ut(data: dict) -> tuple[float, ...] | None:
    """Return S1..S5 XYZ in a stable 15-channel order, or None if incomplete."""
    values: list[float] = []
    for index in range(5):
        row = data.get(index) if isinstance(data, dict) else None
        if not row:
            return None
        try:
            values.extend((float(row["x_ut"]), float(row["y_ut"]), float(row["z_ut"])))
        except (KeyError, TypeError, ValueError):
            return None
    if len(values) != CHANNEL_COUNT or not all(math.isfinite(value) for value in values):
        return None
    return tuple(values)


def _checked(samples: Iterable[Sequence[float]]) -> list[tuple[float, ...]]:
    rows = [tuple(float(value) for value in sample) for sample in samples]
    if not rows:
        raise ValueError("raw magnetic sample window is empty")
    if any(len(row) != CHANNEL_COUNT for row in rows):
        raise ValueError(f"raw magnetic samples must contain {CHANNEL_COUNT} channels")
    if any(not all(math.isfinite(value) for value in row) for row in rows):
        raise ValueError("raw magnetic samples contain non-finite values")
    return rows


def delta_amplitudes_ut(
    samples: Iterable[Sequence[float]], baseline_ut: Sequence[float]
) -> list[float]:
    rows = _checked(samples)
    baseline = tuple(float(value) for value in baseline_ut)
    if len(baseline) != CHANNEL_COUNT:
        raise ValueError(f"baseline must contain {CHANNEL_COUNT} channels")
    return [
        math.sqrt(sum((value - origin) ** 2 for value, origin in zip(row, baseline)))
        for row in rows
    ]


def raw_baseline_statistics(
    samples: Iterable[Sequence[float]], minimum_threshold_ut: float
) -> dict:
    """Build a robust B0 and an adaptive threshold from a verified unloaded window."""
    rows = _checked(samples)
    if len(rows) < 8:
        raise ValueError("at least 8 unloaded raw frames are required")
    minimum = float(minimum_threshold_ut)
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("minimum raw contact threshold must be positive")
    columns = list(zip(*rows))
    baseline = tuple(float(statistics.median(column)) for column in columns)
    residuals = sorted(delta_amplitudes_ut(rows, baseline))
    p95 = residuals[min(len(residuals) - 1, max(0, math.ceil(0.95 * len(residuals)) - 1))]
    median = float(statistics.median(residuals))
    mad = float(statistics.median(abs(value - median) for value in residuals))
    # The absolute 3 uT floor prevents a numerically perfect/simulated baseline from
    # creating an unrealistically sensitive automatic motion threshold.
    adaptive = p95 + max(3.0, 6.0 * 1.4826 * mad)
    threshold = max(minimum, adaptive)
    return {
        "baseline_ut": list(baseline),
        "frames": len(rows),
        "noise_median_ut": median,
        "noise_p95_ut": float(p95),
        "noise_mad_ut": mad,
        "minimum_threshold_ut": minimum,
        "adaptive_threshold_ut": float(adaptive),
        "threshold_ut": float(threshold),
    }


def raw_contact_window(
    samples: Iterable[Sequence[float]],
    baseline_ut: Sequence[float],
    threshold_ut: float,
    confirm_frames: int = 4,
) -> dict:
    """Confirm contact only when the newest N raw frames all exceed threshold."""
    threshold = float(threshold_ut)
    confirm = int(confirm_frames)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("raw contact threshold must be positive")
    if confirm < 2:
        raise ValueError("raw contact confirmation requires at least 2 frames")
    amplitudes = delta_amplitudes_ut(samples, baseline_ut)
    tail = amplitudes[-confirm:]
    triggered = len(tail) == confirm and all(value >= threshold for value in tail)
    return {
        "triggered": bool(triggered),
        "frames": len(amplitudes),
        "confirm_frames": confirm,
        "threshold_ut": threshold,
        "last_amplitude_ut": float(amplitudes[-1]),
        "tail_min_ut": float(min(tail)),
        "tail_median_ut": float(statistics.median(tail)),
        "peak_ut": float(max(amplitudes)),
    }
