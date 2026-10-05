"""Advisory magnetic-layer installation fingerprint helpers.

The comparison is intentionally non-blocking in R37.  It is a reference/QC
signal for experiments; a future application build may opt into enforcement.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


def _vector_15(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (15,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite 5x3/15-channel values")
    return array


def robust_fingerprint(samples: Iterable[Any]) -> list[list[float]]:
    """Return the channel-wise median of recent unloaded 5x3 field samples."""
    rows = [_vector_15(sample, "sample") for sample in samples]
    if len(rows) < 10:
        raise ValueError("至少需要10帧有效无载磁场样本 / need at least 10 unloaded samples")
    return np.median(np.asarray(rows, dtype=float), axis=0).reshape(5, 3).tolist()


def compare_fingerprint(current: Any, config: dict[str, Any] | None) -> dict[str, Any]:
    """Compare against a configured reference without authorizing a hard stop."""
    cfg = config if isinstance(config, dict) else {}
    reference = cfg.get("reference_b0_5x3_uT")
    if not cfg.get("enabled", True) or reference is None:
        return {
            "configured": False,
            "severity": "unconfigured",
            "advisory_only": True,
            "blocks_experiment": False,
            "message": "未配置安装指纹参考；不影响实验运行",
        }
    current_vector = _vector_15(current, "current fingerprint")
    reference_vector = _vector_15(reference, "reference fingerprint")
    delta = current_vector - reference_vector
    l2 = float(np.linalg.norm(delta))
    rms = float(math.sqrt(float(np.mean(delta * delta))))
    max_channel = float(np.max(np.abs(delta)))
    warn_l2 = float(cfg.get("warn_l2_uT", 500.0))
    warn_channel = float(cfg.get("warn_max_channel_uT", 300.0))
    warning = l2 > warn_l2 or max_channel > warn_channel
    message = (
        f"安装指纹{'偏离参考' if warning else '接近参考'}："
        f"L2={l2:.1f}/{warn_l2:.1f} µT，单通道最大={max_channel:.1f}/{warn_channel:.1f} µT；"
        "R37仅提示，不阻止采集或实验"
    )
    return {
        "configured": True,
        "severity": "warning" if warning else "ok",
        "advisory_only": True,
        "blocks_experiment": False,
        "l2_uT": l2,
        "rms_per_channel_uT": rms,
        "max_abs_channel_uT": max_channel,
        "warn_l2_uT": warn_l2,
        "warn_max_channel_uT": warn_channel,
        "message": message,
    }
