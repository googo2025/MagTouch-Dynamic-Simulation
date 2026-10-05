#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Case-preserving online adapters for H1, H2-R15, V16, R18 and R56 checkpoints.

The upper UI consumes ``raw_fields`` without renaming model outputs.  The lower
plot and 3-D view consume ``visual_xyz_mm`` through an explicit adapter:

* H1 defaults to its absolute (x, y, z) output.
* A future dual-head H2 defaults to absolute (X, Y, Z).  Existing H2-R15
  checkpoints that only expose relative fields automatically fall back to
  (dX, dY, dZ).

The mapping is configurable, but the original fields and every additional
model output remain available in ``raw_fields`` / ``extra_outputs``.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


_CHECKPOINT_PATH_COMPAT_LOCK = threading.Lock()


def _torch_load_cross_platform(torch, path: Path) -> Any:
    """Load Linux/Windows checkpoints whose metadata contains concrete pathlib paths.

    PyTorch pickles arbitrary metadata in addition to tensors.  A checkpoint saved
    on Ubuntu may therefore contain ``pathlib.PosixPath`` inside ``args``; creating
    that concrete class on Windows raises ``NotImplementedError`` before the model
    record can be inspected.  Temporarily map only the foreign concrete path class
    while unpickling, then restore pathlib immediately.  The lock prevents another
    loader thread from observing the short-lived compatibility mapping.
    """
    with _CHECKPOINT_PATH_COMPAT_LOCK:
        original_posix = pathlib.PosixPath
        original_windows = pathlib.WindowsPath
        try:
            if os.name == "nt":
                pathlib.PosixPath = original_windows
            else:
                pathlib.WindowsPath = original_posix
            return torch.load(path, map_location="cpu", weights_only=False)
        finally:
            pathlib.PosixPath = original_posix
            pathlib.WindowsPath = original_windows


RAW_FIELD_ORDER = ("x", "y", "z", "dx", "dy", "mode5", "Q", "X", "Y", "Z", "dX", "dY", "dZ")
MODE5_NAMES = ("none", "press", "slide_x", "slide_y", "compound")
DEFAULT_OUTPUT_MAPPING = {
    "h1_reg": ["x", "y", "z", "dx", "dy"],
    "h1_mode": "mode5",
    "h1_reliability": "Q",
    "h2_absolute": ["X", "Y", "Z"],
    "h2_reg": ["dX", "dY", "dZ"],
    "visual": {"x": "auto", "y": "auto", "z": "auto"},
}


def normalize_output_mapping(value: str | Mapping[str, Any] | None) -> dict[str, Any]:
    """Parse and validate the small, JSON-serializable output mapping."""
    if value is None or value == "":
        user: Mapping[str, Any] = {}
    elif isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Output mapping JSON must be an object")
        user = parsed
    elif isinstance(value, Mapping):
        user = value
    else:
        raise ValueError("Output mapping must be JSON text or a mapping")

    out = json.loads(json.dumps(DEFAULT_OUTPUT_MAPPING))
    for key in ("h1_reg", "h2_absolute", "h2_reg"):
        if key in user:
            names = user[key]
            expected = 5 if key == "h1_reg" else 3
            if not isinstance(names, list) or len(names) != expected or not all(isinstance(x, str) and x for x in names):
                raise ValueError(f"{key} must contain exactly {expected} non-empty field names")
            out[key] = list(names)
    for key in ("h1_mode", "h1_reliability"):
        if key in user:
            if not isinstance(user[key], str) or not user[key]:
                raise ValueError(f"{key} must be a non-empty field name")
            out[key] = user[key]
    if "visual" in user:
        visual = user["visual"]
        if not isinstance(visual, Mapping):
            raise ValueError("visual must be an object with x/y/z source fields")
        for axis in "xyz":
            if axis in visual:
                if not isinstance(visual[axis], str) or not visual[axis]:
                    raise ValueError(f"visual.{axis} must be a non-empty field name or auto")
                out["visual"][axis] = visual[axis]
    return out


def default_output_mapping_json() -> str:
    return json.dumps(DEFAULT_OUTPUT_MAPPING, ensure_ascii=False, separators=(",", ":"))


def _to_python(value: Any) -> Any:
    """Detach tensors and retain nested extra outputs for UI inspection."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _reliability(raw: np.ndarray) -> np.ndarray:
    """Axis-level Q policy used by the H1/R15 dataset builders."""
    raw = np.asarray(raw, dtype=np.float32)
    a = np.abs(raw)
    q = np.ones_like(a, dtype=np.float32)
    near95 = 0.95 * 32768.0
    critical = 32000.0
    hard = 32760.0
    middle = (a > near95) & (a < critical)
    q[middle] = (critical - a[middle]) / (critical - near95)
    q[a >= critical] = 0.05
    q[a >= hard] = 0.0
    q[~np.isfinite(raw)] = 0.0
    if raw.shape[0] > 1:
        jumps = np.abs(np.diff(raw, axis=0)) > 60000.0
        for index in np.argwhere(jumps):
            frame = int(index[0])
            lo = max(0, frame - 1)
            hi = min(len(q), frame + 4)
            q[(slice(lo, hi), *tuple(int(i) for i in index[1:]))] = 0.0
    return q


def _scalar(value: Any) -> float | None:
    if isinstance(value, (int, float, np.number)) and np.isfinite(float(value)):
        return float(value)
    return None


def _visual_sources(family: str, mapping: Mapping[str, Any], available: Mapping[str, Any]) -> tuple[str, str, str]:
    if family == "H1":
        defaults = ("x", "y", "z")
    else:
        absolute = tuple(str(x) for x in mapping.get("h2_absolute", ("X", "Y", "Z")))
        defaults = absolute if all(_scalar(available.get(name)) is not None for name in absolute) else ("dX", "dY", "dZ")
    visual = mapping.get("visual", {})
    return tuple(
        defaults[i] if str(visual.get(axis, "auto")) == "auto" else str(visual[axis])
        for i, axis in enumerate("xyz")
    )


@dataclass(frozen=True)
class UnifiedPrediction:
    family: str
    raw_fields: dict[str, Any]
    extra_outputs: dict[str, Any]
    visual_xyz_mm: tuple[float, float, float]
    visual_sources: tuple[str, str, str]
    norm_mm: float
    moving: bool
    q_mean: float
    q_min: float
    history_frames: int
    history_required: int
    t_ns: int
    interaction_valid: bool = True
    normal_state: str = "unknown"
    tangential_motion: bool = False
    speed_mm_s: float = 0.0

    # Compatibility with the existing plot/3-D path.  These are adapter values,
    # never aliases injected into ``raw_fields``.
    @property
    def dx_mm(self) -> float:
        return self.visual_xyz_mm[0]

    @property
    def dy_mm(self) -> float:
        return self.visual_xyz_mm[1]

    @property
    def dz_mm(self) -> float:
        return self.visual_xyz_mm[2]


class _LiveBase:
    GRID_NS = 25_000_000

    def __init__(self) -> None:
        self.reset_all()

    def reset_all(self) -> None:
        self.b0: np.ndarray | None = None
        self.baseline_target_frames = 40
        self.baseline_frames: list[np.ndarray] = []
        self._B: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._raw: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._t: deque[int] = deque(maxlen=self.seq_len)
        self._previous: tuple[int, np.ndarray, np.ndarray] | None = None
        self._next_grid_ns: int | None = None
        self._last_motion_sample: tuple[int, np.ndarray] | None = None
        if hasattr(self, "interaction"):
            self.interaction.reset()

    def begin_baseline(self, seconds: float = 1.0) -> None:
        self.reset_all()
        self.baseline_target_frames = max(8, int(round(float(seconds) * 40.0)))

    @property
    def baseline_progress(self) -> tuple[int, int]:
        return len(self.baseline_frames), self.baseline_target_frames

    @property
    def ready(self) -> bool:
        return self.b0 is not None and len(self._B) >= self.seq_len

    @staticmethod
    def _parse_group(data: dict[int, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
        if any(i not in data for i in range(5)):
            return None
        try:
            b = np.asarray([[data[i]["x_ut"], data[i]["y_ut"], data[i]["z_ut"]] for i in range(5)], dtype=np.float32)
            raw = np.asarray([[data[i]["x_raw"], data[i]["y_raw"], data[i]["z_raw"]] for i in range(5)], dtype=np.float32)
        except (KeyError, TypeError, ValueError):
            return None
        return (b, raw) if np.isfinite(b).all() and np.isfinite(raw).all() else None

    def push_group(self, data: dict[int, dict[str, Any]], t_ns: int) -> list[UnifiedPrediction]:
        parsed = self._parse_group(data)
        if parsed is None:
            return []
        b, raw = parsed
        t_ns = int(t_ns)
        if self._previous is None:
            self._previous = (t_ns, b, raw)
            self._next_grid_ns = t_ns + self.GRID_NS
            return self._append_grid(t_ns, b, raw)
        t0, b0, raw0 = self._previous
        if t_ns <= t0:
            return []
        predictions: list[UnifiedPrediction] = []
        assert self._next_grid_ns is not None
        while self._next_grid_ns <= t_ns:
            alpha = float(self._next_grid_ns - t0) / float(t_ns - t0)
            alpha = min(1.0, max(0.0, alpha))
            bi = b0 + (b - b0) * alpha
            rawi = raw0 if alpha < 0.5 else raw
            predictions.extend(self._append_grid(self._next_grid_ns, bi.astype(np.float32), rawi.astype(np.float32)))
            self._next_grid_ns += self.GRID_NS
        self._previous = (t_ns, b, raw)
        return predictions

    def _append_grid(self, t_ns: int, b: np.ndarray, raw: np.ndarray) -> list[UnifiedPrediction]:
        if self.b0 is None:
            self.baseline_frames.append(np.asarray(b, dtype=np.float32))
            if len(self.baseline_frames) < self.baseline_target_frames:
                return []
            self.b0 = np.nanmedian(np.stack(self.baseline_frames, axis=0), axis=0).astype(np.float32)
            for frame in self.baseline_frames[-self.seq_len:]:
                self._B.append(frame)
                self._raw.append(raw.copy())
                self._t.append(int(t_ns))
            return []
        self._B.append(np.asarray(b, dtype=np.float32))
        self._raw.append(np.asarray(raw, dtype=np.float32))
        self._t.append(int(t_ns))
        if len(self._B) < self.seq_len:
            return []
        return [self._infer()]

    def _mapped_prediction(
        self,
        raw_fields: dict[str, Any],
        extra_outputs: dict[str, Any],
        q_input: np.ndarray,
    ) -> UnifiedPrediction:
        available = dict(extra_outputs)
        available.update(raw_fields)
        sources = _visual_sources(self.family, self.output_mapping, available)
        values = tuple(_scalar(available.get(source)) for source in sources)
        missing = [source for source, value in zip(sources, values) if value is None]
        if missing:
            raise ValueError("Visual mapping refers to missing/non-scalar field(s): " + ", ".join(missing))
        xyz = tuple(float(v) for v in values if v is not None)
        # Display coordinates may be absolute.  Keep the derived threshold state
        # tied to displacement/indentation fields so a non-zero contact location
        # is never mislabeled as motion merely because it is away from the origin.
        motion_sources = ("dx", "dy", "z") if self.family == "H1" else ("dX", "dY", "dZ")
        motion_values = tuple(_scalar(available.get(source)) for source in motion_sources)
        motion_xyz = xyz if any(value is None for value in motion_values) else tuple(float(value) for value in motion_values)
        motion_vector = np.asarray(motion_xyz, dtype=np.float32)
        norm = float(np.linalg.norm(motion_vector))
        velocity = np.zeros(3, dtype=np.float32)
        if self._last_motion_sample is not None:
            previous_t, previous_vector = self._last_motion_sample
            dt = max((int(self._t[-1]) - previous_t) / 1e9, 1e-6)
            velocity = (motion_vector - previous_vector) / dt
        self._last_motion_sample = (int(self._t[-1]), motion_vector.copy())
        speed = float(np.linalg.norm(velocity))
        speed_threshold = float(getattr(self, "motion_speed_epsilon_mm_s", 0.05))
        if velocity[2] > speed_threshold:
            normal_state = "press"
        elif velocity[2] < -speed_threshold:
            normal_state = "release"
        else:
            normal_state = "hold"
        extra_outputs.update({
            "adapter.motion_velocity_xyz_mm_s": _to_python(velocity),
            "adapter.motion_speed_mm_s": speed,
            "adapter.motion_state_semantics": "Causal velocity of predicted displacement; not displacement-from-zero threshold.",
        })
        return UnifiedPrediction(
            self.family, raw_fields, extra_outputs, xyz, sources, norm,
            speed > speed_threshold, float(np.mean(q_input[-1])),
            float(np.min(q_input[-1])), len(self._B), self.seq_len,
            int(self._t[-1]), True, normal_state,
            float(np.linalg.norm(velocity[:2])) > speed_threshold, speed,
        )


class _LiveH1Predictor(_LiveBase):
    family = "H1"

    def __init__(self, path: Path, record: dict[str, Any], output_mapping: Mapping[str, Any], device: str | None):
        import torch
        try:
            from .model_magtouch_ra_hw5 import MagTouchRAHardware5
        except ImportError:  # pragma: no cover
            from model_magtouch_ra_hw5 import MagTouchRAHardware5

        self.torch = torch
        self.path = path
        self.args = record.get("args")
        if not isinstance(self.args, dict):
            raise ValueError("H1 checkpoint has invalid args")
        self.feature = str(self.args.get("feature", "F5"))
        if self.feature not in {f"F{i}" for i in range(6)}:
            raise ValueError(f"Unsupported H1 feature: {self.feature}")
        self.policy = str(self.args.get("policy", "raw15"))
        if self.policy not in {"raw15", "clean14", "mask15"}:
            raise ValueError(f"Unsupported H1 sensor policy: {self.policy}")
        self.history = int(self.args.get("T", 32))
        self.seq_len = max(1, self.history)
        self.motion_epsilon_mm = float(self.args.get("motion_epsilon_mm", 0.02))
        self.output_mapping = dict(output_mapping)
        self.scaler_median = np.asarray(record["scaler_median"], dtype=np.float32)
        self.scaler_scale = np.asarray(record["scaler_scale"], dtype=np.float32)
        self.reg_scale = np.asarray(record.get("reg_scale", np.ones(5)), dtype=np.float32)
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model = MagTouchRAHardware5(
            int(self.scaler_median.shape[-1]), self.seq_len,
            int(self.args.get("d_model", 128)), int(self.args.get("layers", 3)),
            4, 0.1, not bool(self.args.get("disable_reliability", False)),
        ).to(self.device)
        self.model.load_state_dict(record["state_dict"])
        self.model.eval()
        self.source_dataset = str(self.args.get("features", ""))
        super().__init__()

    @property
    def model_summary(self) -> str:
        return f"{self.path.name} | H1 {self.feature}/{self.policy} | T{self.seq_len} | {self.device.type}"

    @property
    def visual_scale(self) -> np.ndarray:
        if self.reg_scale.size >= 5:
            return np.asarray([self.reg_scale[3], self.reg_scale[4], self.reg_scale[2]], dtype=np.float32)
        return np.ones(3, dtype=np.float32)

    def _features(self) -> tuple[np.ndarray, np.ndarray]:
        B = np.stack(self._B, axis=0).astype(np.float32)
        raw = np.stack(self._raw, axis=0).astype(np.float32)
        dB = B - np.asarray(self.b0, dtype=np.float32)[None, :, :]
        med = np.nanmedian(dB, axis=1).astype(np.float32)
        rel_median = (dB - med[:, None, :]).astype(np.float32)
        rel_s5 = np.zeros_like(dB)
        rel_s5[:, :4, :] = dB[:, :4, :] - dB[:, 4:5, :]
        derivative = np.concatenate([np.zeros((1, 5, 3), np.float32), np.diff(dB, axis=0) * 40.0], axis=0)
        common = np.repeat(med[:, None, :], 5, axis=1).astype(np.float32)
        q = _reliability(raw).astype(np.float32)
        if self.feature == "F0":
            parts, signal_indices = [B], [0]
        elif self.feature == "F1":
            parts, signal_indices = [dB], [0]
        elif self.feature == "F2":
            parts, signal_indices = [dB, rel_s5], [0, 1]
        elif self.feature == "F3":
            parts, signal_indices = [dB, rel_median], [0, 1]
        else:
            parts, signal_indices = [dB, rel_median, derivative, common], [0, 1, 2]
            if self.feature == "F5":
                parts.append(q.copy())

        if self.policy == "clean14":
            for index in signal_indices:
                parts[index][:, 3, 1] = 0.0
            q[:, 3, 1] = 0.0
            if self.feature == "F5":
                parts[-1] = q.copy()
        elif self.policy == "mask15":
            if self.feature == "F2":
                parts[0] *= q
                q_rel = np.zeros_like(q)
                q_rel[:, :4, :] = np.minimum(q[:, :4, :], q[:, 4:5, :])
                parts[1] *= q_rel
            else:
                for index in signal_indices:
                    parts[index] *= q

        features = np.concatenate(parts, axis=-1).astype(np.float32)
        if features.shape[1:] != self.scaler_median.shape or self.scaler_median.shape != self.scaler_scale.shape:
            raise ValueError(
                f"H1 feature/scaler mismatch: live={features.shape[1:]} scaler={self.scaler_median.shape}"
            )
        return features, q

    def _infer(self) -> UnifiedPrediction:
        x, q_input = self._features()
        x = (x - self.scaler_median[None, :, :]) / self.scaler_scale[None, :, :]
        tx = self.torch.from_numpy(x[None, ...]).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tx)
            reg = output.reg.float().cpu().numpy()[0]
            logits = output.cls_logits.float().cpu()
            probabilities = self.torch.softmax(logits, dim=-1).numpy()[0]
        raw: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for name, value in zip(self.output_mapping["h1_reg"], reg):
            target = raw if name in RAW_FIELD_ORDER else extras
            target[name] = float(value)
        mode_index = int(np.argmax(probabilities))
        mode_value = {
            "index": mode_index,
            "label": MODE5_NAMES[mode_index] if 0 <= mode_index < len(MODE5_NAMES) else str(mode_index),
            "probability": float(probabilities[mode_index]),
        }
        mode_name = self.output_mapping["h1_mode"]
        (raw if mode_name in RAW_FIELD_ORDER else extras)[mode_name] = mode_value
        predicted_q = _to_python(output.reliability[0, -1]) if output.reliability is not None else None
        q_name = self.output_mapping["h1_reliability"]
        if predicted_q is not None:
            (raw if q_name in RAW_FIELD_ORDER else extras)[q_name] = predicted_q
        extras.update({
            "model.cls_logits": _to_python(logits[0]),
            "model.mode5_probabilities": {name: float(probabilities[i]) for i, name in enumerate(MODE5_NAMES)},
            "model.extras": _to_python(output.extras),
            "adapter.input_Q_last": _to_python(q_input[-1]),
            "adapter.input_Q_mean": float(np.mean(q_input[-1])),
            "adapter.input_Q_min": float(np.min(q_input[-1])),
        })
        return self._mapped_prediction(raw, extras, q_input)


class _LiveH2R15Predictor(_LiveBase):
    family = "H2-R15"

    def __init__(self, path: Path, record: dict[str, Any], output_mapping: Mapping[str, Any], device: str | None):
        import torch
        try:
            from .model_relative_tactile_hw5 import RelativeTactileHW5
        except ImportError:  # pragma: no cover
            from model_relative_tactile_hw5 import RelativeTactileHW5

        self.torch = torch
        self.path = path
        self.args = record.get("args")
        if not isinstance(self.args, dict):
            raise ValueError("H2-R15 checkpoint has invalid args")
        self.feature = str(self.args.get("feature", "R1"))
        if self.feature not in {"R0", "R1", "R2", "R3"}:
            raise ValueError(f"Unsupported H2-R15 feature: {self.feature}")
        self.history = int(record.get("history_frames", self.args.get("history", 64)))
        self.seq_len = 1 if self.history == 0 else self.history
        self.motion_epsilon_mm = float(self.args.get("motion_epsilon_mm", 0.02))
        self.use_q = bool(self.args.get("use_q", True))
        self.output_mapping = dict(output_mapping)
        self.scaler_median = np.asarray(record["scaler_median"], dtype=np.float32)
        self.scaler_scale = np.asarray(record["scaler_scale"], dtype=np.float32)
        self.reg_scale = np.asarray(record["reg_scale"], dtype=np.float32)
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model = RelativeTactileHW5(
            int(self.scaler_median.shape[-1]), self.feature, self.seq_len,
            int(self.args.get("d_model", 128)), int(self.args.get("layers", 3)),
            4, 0.1, self.use_q,
        ).to(self.device)
        self.model.load_state_dict(record["state_dict"])
        self.model.eval()
        self.source_dataset = str(self.args.get("dataset", ""))
        super().__init__()

    @property
    def model_summary(self) -> str:
        return (
            f"{self.path.name} | H2-R15 {self.feature} | H{self.history} "
            f"({self.history / 40.0:.2f}s) | Q={int(self.use_q)} | {self.device.type}"
        )

    @property
    def visual_scale(self) -> np.ndarray:
        return np.asarray(self.reg_scale[:3], dtype=np.float32)

    def _features(self) -> tuple[np.ndarray, np.ndarray]:
        B = np.stack(self._B, axis=0).astype(np.float32)
        raw = np.stack(self._raw, axis=0).astype(np.float32)
        dB = B - np.asarray(self.b0, dtype=np.float32)[None, :, :]
        med = np.nanmedian(dB, axis=1).astype(np.float32)
        rel = (dB - med[:, None, :]).astype(np.float32)
        derivative = np.concatenate([np.zeros((1, 5, 3), np.float32), np.diff(dB, axis=0) * 40.0], axis=0)
        common = np.repeat(med[:, None, :], 5, axis=1).astype(np.float32)
        q = _reliability(raw).astype(np.float32)
        feature = {
            "R0": dB,
            "R1": np.concatenate([dB, rel], axis=-1),
            "R2": np.concatenate([dB, rel, derivative, common], axis=-1),
            "R3": np.concatenate([dB, rel, derivative, common, q], axis=-1),
        }[self.feature]
        return feature.astype(np.float32), q

    def _infer(self) -> UnifiedPrediction:
        x, q_input = self._features()
        x = (x - self.scaler_median[None, :, :]) / self.scaler_scale[None, :, :]
        tx = self.torch.from_numpy(x[None, ...]).to(self.device)
        tq = self.torch.from_numpy(q_input[None, ...]).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tx, tq)
            normalized = output.dxyz_norm.float().cpu().numpy()[0]
        millimetres = normalized * self.reg_scale
        raw: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        for name, value in zip(self.output_mapping["h2_reg"], millimetres):
            target = raw if name in RAW_FIELD_ORDER else extras
            target[name] = float(value)
        extras.update({
            "model.dxyz_normalized": _to_python(normalized),
            "model.extras": _to_python(output.extras),
            "adapter.input_Q_last": _to_python(q_input[-1]),
            "adapter.input_Q_mean": float(np.mean(q_input[-1])),
            "adapter.input_Q_min": float(np.min(q_input[-1])),
        })
        return self._mapped_prediction(raw, extras, q_input)


class _LiveR18Predictor(_LiveH2R15Predictor):
    """R18.1 joint absolute XYZ + relative dXYZ + mode5 live adapter."""
    family = "H2-Unified-R18"

    def __init__(self, path: Path, record: dict[str, Any], output_mapping: Mapping[str, Any], device: str | None):
        import torch
        try:
            from .model_unified_magtouch_r18_hw5 import UnifiedMagTouchHW5
        except ImportError:  # pragma: no cover
            from model_unified_magtouch_r18_hw5 import UnifiedMagTouchHW5

        self.checkpoint_schema = str(record.get("schema", ""))
        if self.checkpoint_schema not in {
            "MagTouch-Unified-R18-checkpoint-1.0",
            "MagTouch-Unified-R18-checkpoint-1.1",
        }:
            raise ValueError(f"Unsupported R18 checkpoint schema: {self.checkpoint_schema or 'missing'}")
        # R18-1.0 was trained with per-Trial centred absolute XY labels.  Its
        # relative dXYZ and mode5 heads remain loadable, but the absolute head
        # must never be presented as a valid global contact coordinate.
        self.legacy_bad_absolute = self.checkpoint_schema.endswith("1.0")
        self.torch = torch
        self.path = path
        self.args = record.get("args")
        if not isinstance(self.args, dict):
            raise ValueError("R18 checkpoint has invalid args")
        self.feature = str(self.args.get("feature", "R1"))
        if self.feature not in {"R0", "R1", "R2", "R3"}:
            raise ValueError(f"Unsupported R18 feature: {self.feature}")
        self.history = int(record.get("history_frames", self.args.get("history", 64)))
        self.seq_len = 1 if self.history == 0 else self.history
        self.motion_epsilon_mm = float(self.args.get("motion_epsilon_mm", 0.02))
        self.use_q = bool(self.args.get("use_q", True))
        self.objective = str(record.get("objective", "unified"))
        if self.objective not in {"unified", "coordinate_only"}:
            raise ValueError(f"Unsupported R18 objective: {self.objective}")
        # R18 native absolute fields are lower-case.  Do not let a persisted V16
        # X/Y/Z mapping silently rename the raw R18 head.
        self.output_mapping = dict(output_mapping)
        self.output_mapping["h2_absolute"] = ["x", "y", "z"]
        self.scaler_median = np.asarray(record["scaler_median"], dtype=np.float32)
        self.scaler_scale = np.asarray(record["scaler_scale"], dtype=np.float32)
        self.fixed_reg_scale = np.asarray(record["fixed_reg_scale"], dtype=np.float32)
        self.relative_reg_scale = np.asarray(record["relative_reg_scale"], dtype=np.float32)
        self.reg_scale = self.relative_reg_scale
        self.mode_names = list(record.get("mode_names", ["noncontact", "press", "slide", "hold", "release"]))
        if self.mode_names != ["noncontact", "press", "slide", "hold", "release"]:
            raise ValueError("R18 mode5 classes are incompatible")
        self.coordinate_frame = record.get("coordinate_frame", {})
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model = UnifiedMagTouchHW5(
            int(self.scaler_median.shape[-1]), self.feature, self.seq_len,
            int(self.args.get("d_model", 128)), int(self.args.get("layers", 3)),
            4, 0.1, self.use_q, len(self.mode_names),
        ).to(self.device)
        self.model.load_state_dict(record["state_dict"])
        self.model.eval()
        self.source_dataset = str(self.args.get("dataset", ""))
        _LiveBase.__init__(self)

    @property
    def model_summary(self) -> str:
        frame_id = self.coordinate_frame.get("id", "unknown-frame") if isinstance(self.coordinate_frame, dict) else str(self.coordinate_frame)
        legacy = " | LEGACY 1.0: unsafe absolute xyz suppressed; dXYZ+mode5 only" if self.legacy_bad_absolute else ""
        return (f"{self.path.name} | H2-Unified-R18 {self.objective} {self.feature} "
                f"| H{self.history} | {frame_id} | {self.device.type}{legacy}")

    @property
    def visual_scale(self) -> np.ndarray:
        return np.maximum(self.fixed_reg_scale, self.relative_reg_scale)

    def _infer(self) -> UnifiedPrediction:
        x, q_input = self._features()
        if x.shape[1:] != self.scaler_median.shape:
            raise ValueError(f"R18 feature/scaler mismatch: live={x.shape[1:]} scaler={self.scaler_median.shape}")
        x = (x - self.scaler_median[None, ...]) / self.scaler_scale[None, ...]
        tx = self.torch.from_numpy(x[None, ...]).to(self.device)
        tq = self.torch.from_numpy(q_input[None, ...]).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tx, tq)
            fixed_mm = output.fixed_xyz_norm.float().cpu().numpy()[0] * self.fixed_reg_scale
            relative_mm = output.dxyz_norm.float().cpu().numpy()[0] * self.relative_reg_scale
            probabilities = self.torch.softmax(output.mode_logits.float(), dim=-1).cpu().numpy()[0]
        raw: dict[str, Any] = {}
        extras: dict[str, Any] = {
            "model.coordinate_frame": _to_python(self.coordinate_frame),
            "model.objective": self.objective,
            "model.extras": _to_python(output.extras),
            "adapter.input_Q_last": _to_python(q_input[-1]),
            "adapter.input_Q_mean": float(np.mean(q_input[-1])),
            "adapter.input_Q_min": float(np.min(q_input[-1])),
        }
        if self.legacy_bad_absolute:
            extras.update({
                "model.legacy_fixed_xyz_unsafe_mm": _to_python(fixed_mm),
                "adapter.absolute_output_suppressed": (
                    "R18 checkpoint-1.0 used per-Trial-centred absolute XY labels; "
                    "raw x/y/z are intentionally unavailable. Retrain with R18.1 for global xyz."
                ),
            })
        else:
            for name, value in zip(self.output_mapping["h2_absolute"], fixed_mm):
                (raw if name in RAW_FIELD_ORDER else extras)[name] = float(value)
        for name, value in zip(self.output_mapping["h2_reg"], relative_mm):
            (raw if name in RAW_FIELD_ORDER else extras)[name] = float(value)
        mode_index = int(np.argmax(probabilities))
        valid = True
        if self.objective == "unified":
            raw["mode5"] = {"index": mode_index, "label": self.mode_names[mode_index],
                            "probability": float(probabilities[mode_index])}
            extras["model.mode5_probabilities"] = {
                name: float(probabilities[i]) for i, name in enumerate(self.mode_names)}
            valid = mode_index != 0
            extras["adapter.validity"] = "Predicted noncontact hides 3D; raw coordinate heads remain available."
        else:
            extras["adapter.validity"] = "coordinate_only: mode head is not used as a contact gate"
        prediction = self._mapped_prediction(raw, extras, q_input)
        return UnifiedPrediction(
            prediction.family, prediction.raw_fields, prediction.extra_outputs,
            prediction.visual_xyz_mm, prediction.visual_sources, prediction.norm_mm,
            prediction.moving, prediction.q_mean, prediction.q_min,
            prediction.history_frames, prediction.history_required, prediction.t_ns, valid,
            prediction.normal_state, prediction.tangential_motion, prediction.speed_mm_s,
        )


class _LiveH2SpatialPredictor(_LiveH2R15Predictor):
    family = "H2-Spatial-R16"

    def __init__(self, path: Path, record: dict[str, Any], output_mapping: Mapping[str, Any], device: str | None):
        import torch
        try:
            from .model_spatial_relative_hw5 import SpatialRelativeHW5
            from .trajectory_state import CausalInteractionState
        except ImportError:  # pragma: no cover
            from model_spatial_relative_hw5 import SpatialRelativeHW5
            from trajectory_state import CausalInteractionState

        self.torch = torch
        self.path = path
        self.args = record.get("args")
        if not isinstance(self.args, dict):
            raise ValueError("V16 checkpoint has invalid args")
        self.feature = str(self.args.get("feature", "R1"))
        if self.feature not in {"R0", "R1", "R2", "R3"}:
            raise ValueError(f"Unsupported V16 feature: {self.feature}")
        self.history = int(record.get("history_frames", self.args.get("history", 64)))
        self.seq_len = 1 if self.history == 0 else self.history
        self.motion_epsilon_mm = float(self.args.get("motion_epsilon_mm", 0.02))
        self.motion_speed_epsilon_mm_s = float(self.args.get("motion_speed_epsilon_mm_s", 0.05))
        self.use_q = bool(self.args.get("use_q", True))
        self.objective = str(record.get("objective", self.args.get("objective", "joint")))
        self.output_mapping = dict(output_mapping)
        self.scaler_median = np.asarray(record["scaler_median"], dtype=np.float32)
        self.scaler_scale = np.asarray(record["scaler_scale"], dtype=np.float32)
        self.fixed_reg_scale = np.asarray(record["fixed_reg_scale"], dtype=np.float32)
        self.relative_reg_scale = np.asarray(record["relative_reg_scale"], dtype=np.float32)
        self.reg_scale = self.relative_reg_scale
        self.coordinate_frame = record.get("coordinate_frame", {})
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model = SpatialRelativeHW5(
            int(self.scaler_median.shape[-1]), self.feature, self.seq_len,
            int(self.args.get("d_model", 128)), int(self.args.get("layers", 3)),
            4, 0.1, self.use_q,
        ).to(self.device)
        self.model.load_state_dict(record["state_dict"])
        self.model.eval()
        self.source_dataset = str(self.args.get("dataset", ""))
        self.interaction = CausalInteractionState(
            enter_mm=max(self.motion_epsilon_mm * 2.0, 0.04),
            exit_mm=max(self.motion_epsilon_mm, 0.02),
            speed_mm_s=self.motion_speed_epsilon_mm_s,
        )
        _LiveBase.__init__(self)

    @property
    def model_summary(self) -> str:
        frame_id = self.coordinate_frame.get("id", "unknown-frame") if isinstance(self.coordinate_frame, dict) else str(self.coordinate_frame)
        return (
            f"{self.path.name} | H2-Spatial-R16 {self.objective} {self.feature} "
            f"| H{self.history} | {frame_id} | {self.device.type}"
        )

    @property
    def visual_scale(self) -> np.ndarray:
        if self.objective in {"fixed_only", "joint"}:
            return np.maximum(self.fixed_reg_scale, self.relative_reg_scale)
        return self.relative_reg_scale

    def _infer(self) -> UnifiedPrediction:
        x, q_input = self._features()
        x = (x - self.scaler_median[None, :, :]) / self.scaler_scale[None, :, :]
        tx = self.torch.from_numpy(x[None, ...]).to(self.device)
        tq = self.torch.from_numpy(q_input[None, ...]).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tx, tq)
            fixed_normalized = output.fixed_xyz_norm.float().cpu().numpy()[0]
            relative_normalized = output.dxyz_norm.float().cpu().numpy()[0]
        fixed_mm = fixed_normalized * self.fixed_reg_scale
        relative_mm = relative_normalized * self.relative_reg_scale
        relative_trained = self.objective in {"relative_only", "joint"}
        fixed_trained = self.objective in {"fixed_only", "joint"}
        state = self.interaction.update(relative_mm, int(self._t[-1])) if relative_trained else None
        raw: dict[str, Any] = {}
        extras: dict[str, Any] = {
            "model.objective": self.objective,
            "model.fixed_head_trained": fixed_trained,
            "model.relative_head_trained": relative_trained,
            "model.coordinate_frame": _to_python(self.coordinate_frame),
            "model.extras": _to_python(output.extras),
            "adapter.input_Q_last": _to_python(q_input[-1]),
            "adapter.input_Q_mean": float(np.mean(q_input[-1])),
            "adapter.input_Q_min": float(np.min(q_input[-1])),
        }
        if fixed_trained:
            extras.update({
                "model.fixed_xyz_mm": _to_python(fixed_mm),
                "model.fixed_xyz_normalized": _to_python(fixed_normalized),
            })
        if relative_trained:
            extras.update({
                "model.relative_dxyz_mm": _to_python(relative_mm),
                "model.relative_dxyz_normalized": _to_python(relative_normalized),
            })
        if relative_trained:
            for name, value in zip(self.output_mapping["h2_reg"], relative_mm):
                (raw if name in RAW_FIELD_ORDER else extras)[name] = float(value)
        if fixed_trained and (state is None or state.valid):
            for name, value in zip(self.output_mapping["h2_absolute"], fixed_mm):
                (raw if name in RAW_FIELD_ORDER else extras)[name] = float(value)
            if state is None:
                extras["adapter.fixed_output_validity"] = (
                    "V16 fixed-only model head; shown directly without a causal interaction gate. "
                    "Interpret outside the trained aligned-interaction region with caution."
                )
        elif fixed_trained:
            extras["adapter.fixed_output_suppressed"] = (
                "No causal interaction validity yet; raw X/Y/Z remain '--'. "
                "The numeric model head is retained under model.fixed_xyz_mm."
            )
        if state is not None:
            extras.update({
                "adapter.interaction_valid": state.valid,
                "adapter.interaction_validity_semantics": "Causal hysteresis over predicted R15 relative displacement; not an independently measured contact label.",
                "adapter.normal_state": state.normal_state,
                "adapter.tangential_motion": state.tangential_motion,
                "adapter.velocity_xyz_mm_s": list(state.velocity_xyz_mm_s),
                "adapter.speed_mm_s": state.speed_mm_s,
            })
        if not raw:
            raise RuntimeError("V16 checkpoint produced no mapped live output")
        prediction = self._mapped_prediction(raw, extras, q_input)
        if state is None:
            return prediction
        return UnifiedPrediction(
            prediction.family, prediction.raw_fields, prediction.extra_outputs,
            prediction.visual_xyz_mm, prediction.visual_sources, prediction.norm_mm,
            state.moving, prediction.q_mean, prediction.q_min,
            prediction.history_frames, prediction.history_required, prediction.t_ns,
            state.valid, state.normal_state, state.tangential_motion, state.speed_mm_s,
        )


class _LiveFinalTipR56Predictor(_LiveBase):
    """Live adapter for the R56 absolute final-tip trajectory checkpoints.

    R56 keeps the R54 checkpoint schema for backward compatibility.  Unlike the
    older R15 adapter it publishes an absolute ``X/Y/Z`` head.  Its 1023-frame
    causal receptive field must not be confused with B0 acquisition duration:
    a one-second robust B0 is sufficient, then the UI explicitly blends from
    the frozen R48 static observer to the temporal head during warm-up.
    """

    family = "FinalTip-R56"
    recommended_baseline_seconds = 1.0

    def __init__(self, path: Path, record: dict[str, Any],
                 output_mapping: Mapping[str, Any], device: str | None):
        import torch
        try:
            from .model_final_tip_trajectory_r54 import load_checkpoint_r54
        except ImportError:  # pragma: no cover
            from model_final_tip_trajectory_r54 import load_checkpoint_r54

        self.torch = torch
        self.path = path
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        self.model = load_checkpoint_r54(path, self.device).eval()
        self.metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        self.model_version = str(self.metadata.get("model_version", "R56-compatible"))
        self.seq_len = int(self.model.config.receptive_field)
        self.history = self.seq_len
        self.minimum_history_frames = min(255, self.seq_len)
        self.transition_frames = 40
        self.motion_epsilon_mm = 0.02
        self.motion_speed_epsilon_mm_s = 0.05
        self.output_mapping = dict(output_mapping)
        # Keep the new absolute trajectory fields unambiguous and case-sensitive.
        self.output_mapping["h2_absolute"] = ["X", "Y", "Z"]
        self.reg_scale = np.asarray([10.0, 10.0, 3.0], dtype=np.float32)
        self.source_dataset = str(self.metadata.get("dataset", ""))
        self._frames_since_baseline = 0
        super().__init__()

    def reset_all(self) -> None:
        self._frames_since_baseline = 0
        super().reset_all()

    @property
    def ready(self) -> bool:
        # The frozen R48 observation is available immediately after B0.  The
        # history_ready flag in extra outputs separately reports temporal warm-up.
        return self.b0 is not None and len(self._B) > 0

    @property
    def model_summary(self) -> str:
        return (
            f"{self.path.name} | {self.model_version} | absolute X/Y/Z | "
            f"RF{self.seq_len}, temporal warm-up {self.minimum_history_frames} frames | {self.device.type}"
        )

    @property
    def visual_scale(self) -> np.ndarray:
        return self.reg_scale.copy()

    def _append_grid(self, t_ns: int, b: np.ndarray, raw: np.ndarray) -> list[UnifiedPrediction]:
        if self.b0 is None:
            self.baseline_frames.append(np.asarray(b, dtype=np.float32))
            if len(self.baseline_frames) < self.baseline_target_frames:
                return []
            self.b0 = np.nanmedian(np.stack(self.baseline_frames, axis=0), axis=0).astype(np.float32)
            # Preserve the known stationary prefix as zero-delta temporal context.
            # It reduces cold-start transients without extending B0 collection.
            for frame in self.baseline_frames[-self.seq_len:]:
                self._B.append(frame)
                self._raw.append(raw.copy())
                self._t.append(int(t_ns))
            self._frames_since_baseline = len(self._B)
            return []
        self._B.append(np.asarray(b, dtype=np.float32))
        self._raw.append(np.asarray(raw, dtype=np.float32))
        self._t.append(int(t_ns))
        self._frames_since_baseline += 1
        return [self._infer()]

    def _infer(self) -> UnifiedPrediction:
        B = np.stack(self._B, axis=0).astype(np.float32)
        raw_sensor = np.stack(self._raw, axis=0).astype(np.float32)
        delta = (B - np.asarray(self.b0, dtype=np.float32)[None, :, :]).reshape(len(B), 15)
        q_input = _reliability(raw_sensor).astype(np.float32)
        tensor = self.torch.from_numpy(delta[None, ...]).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
            model_xyz = output["xyz_mm"][0, -1]
            static_xyz = output["static_xyz_mm"][0, -1]
            correction = output["correction_xyz_mm"][0, -1]
            log_sigma = output["log_sigma_xyz"][0, -1]
        ready = self._frames_since_baseline >= self.minimum_history_frames
        if ready:
            blend = min(
                1.0,
                (self._frames_since_baseline - self.minimum_history_frames + 1)
                / float(self.transition_frames),
            )
        else:
            blend = 0.0
        published = static_xyz + blend * (model_xyz - static_xyz)
        xyz = published.detach().float().cpu().numpy()
        raw_fields = {name: float(value) for name, value in zip(("X", "Y", "Z"), xyz)}
        contact = bool(np.isfinite(xyz[2]) and float(xyz[2]) > self.motion_epsilon_mm)
        extras: dict[str, Any] = {
            "model.model_version": self.model_version,
            "model.xyz_mm": _to_python(model_xyz),
            "model.static_xyz_mm": _to_python(static_xyz),
            "model.correction_xyz_mm": _to_python(correction),
            "model.log_sigma_xyz": _to_python(log_sigma),
            "adapter.history_ready": ready,
            "adapter.temporal_blend": float(blend),
            "adapter.minimum_history_frames": self.minimum_history_frames,
            "adapter.receptive_field_frames": self.seq_len,
            "adapter.warmup_semantics": (
                "Absolute R48 static output is published first; the causal R56 correction "
                "is blended in after the explicit history warm-up."
            ),
            "adapter.input_Q_last": _to_python(q_input[-1]),
            "adapter.input_Q_mean": float(np.mean(q_input[-1])),
            "adapter.input_Q_min": float(np.min(q_input[-1])),
            "display.contact": contact,
            "display.contact_reason": "R56 absolute Z > 0.02 mm inferred contact gate",
        }
        return self._mapped_prediction(raw_fields, extras, q_input)


class UnifiedLivePredictor:
    """Auto-detect supported legacy checkpoints and final-tip R56 checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        family: str = "auto",
        output_mapping: str | Mapping[str, Any] | None = None,
        device: str | None = None,
    ) -> None:
        import torch

        path = Path(checkpoint_path).expanduser().resolve()
        record = _torch_load_cross_platform(torch, path)
        if not isinstance(record, dict):
            raise ValueError("Checkpoint must contain a dictionary")
        schema = str(record.get("schema", ""))
        is_final_tip = schema == "MagTouch-FinalTip-Trajectory-R54-1.0"
        if not is_final_tip:
            required = {"state_dict", "args", "scaler_median", "scaler_scale", "reg_scale"}
            missing = sorted(required - set(record))
            if missing:
                raise ValueError("Unsupported checkpoint; missing: " + ", ".join(missing))
        mapping = normalize_output_mapping(output_mapping)
        requested = str(family or "auto").strip().lower()
        args = record.get("args") if isinstance(record.get("args"), dict) else {}
        feature = str(args.get("feature", ""))
        if requested in {"h1", "h1-r13"}:
            detected = "H1"
        elif requested in {"h2", "h2-r15", "r15"}:
            detected = "H2-R15"
        elif requested in {"h2-spatial", "h2-spatial-r16", "r16", "v16"}:
            detected = "H2-Spatial-R16"
        elif requested in {"h2-unified-r18", "r18"}:
            detected = "H2-Unified-R18"
        elif requested in {"finaltip-r56", "final-tip-r56", "r56", "r54"}:
            detected = "FinalTip-R56"
        elif requested == "auto":
            if is_final_tip:
                detected = "FinalTip-R56"
            elif schema.startswith("MagTouch-Unified-R18"):
                detected = "H2-Unified-R18"
            elif schema.startswith("MagTouch-SpatialRelative-V16") or "fixed_reg_scale" in record:
                detected = "H2-Spatial-R16"
            elif feature.startswith("F"):
                detected = "H1"
            elif feature.startswith("R") or "history_frames" in record:
                detected = "H2-R15"
            else:
                keys = tuple(str(k) for k in record["state_dict"])
                if any(k.startswith("heads.") for k in keys):
                    detected = "H1"
                elif any(k.startswith("xyz.") for k in keys):
                    detected = "H2-R15"
                else:
                    raise ValueError("Cannot auto-detect checkpoint family")
        else:
            raise ValueError(
                "Model family must be auto, H1, H2-R15, H2-Spatial-R16, "
                "H2-Unified-R18, or FinalTip-R56"
            )
        if detected == "H1":
            self.impl = _LiveH1Predictor(path, record, mapping, device)
        elif detected == "H2-Spatial-R16":
            self.impl = _LiveH2SpatialPredictor(path, record, mapping, device)
        elif detected == "H2-Unified-R18":
            self.impl = _LiveR18Predictor(path, record, mapping, device)
        elif detected == "FinalTip-R56":
            self.impl = _LiveFinalTipR56Predictor(path, record, mapping, device)
        else:
            self.impl = _LiveH2R15Predictor(path, record, mapping, device)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.impl, name)

    @property
    def family(self) -> str:
        return self.impl.family

    @property
    def visual_scale(self) -> np.ndarray:
        return self.impl.visual_scale

    def begin_baseline(self, seconds: float = 1.0) -> None:
        self.impl.begin_baseline(seconds)

    def push_group(self, data: dict[int, dict[str, Any]], t_ns: int) -> list[UnifiedPrediction]:
        return self.impl.push_group(data, t_ns)
