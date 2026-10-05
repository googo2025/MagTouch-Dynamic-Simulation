#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Causal state/validity adapter for dual-coordinate online output.

This is an interpretable runtime baseline, not a trained contact classifier.
It uses only current/past relative predictions and timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InteractionState:
    valid: bool
    moving: bool
    normal_state: str
    tangential_motion: bool
    velocity_xyz_mm_s: tuple[float, float, float]
    speed_mm_s: float


class CausalInteractionState:
    def __init__(self, enter_mm: float = 0.04, exit_mm: float = 0.02,
                 speed_mm_s: float = 0.05, release_frames: int = 4):
        if exit_mm > enter_mm:
            raise ValueError("exit_mm must not exceed enter_mm")
        self.enter_mm = float(enter_mm)
        self.exit_mm = float(exit_mm)
        self.speed_threshold = float(speed_mm_s)
        self.release_frames = max(1, int(release_frames))
        self.reset()

    def reset(self) -> None:
        self.active = False
        self._quiet = 0
        self._previous: tuple[int, np.ndarray] | None = None

    def update(self, dxyz_mm: np.ndarray, t_ns: int) -> InteractionState:
        vector = np.asarray(dxyz_mm, dtype=np.float32).reshape(3)
        norm = float(np.linalg.norm(vector))
        if not self.active and norm >= self.enter_mm:
            self.active = True
            self._quiet = 0
        elif self.active:
            if norm <= self.exit_mm:
                self._quiet += 1
                if self._quiet >= self.release_frames:
                    self.active = False
                    self._quiet = 0
            else:
                self._quiet = 0

        velocity = np.zeros(3, dtype=np.float32)
        if self._previous is not None:
            previous_t, previous_vector = self._previous
            dt = max((int(t_ns) - previous_t) / 1e9, 1e-6)
            velocity = (vector - previous_vector) / dt
        self._previous = (int(t_ns), vector.copy())
        speed = float(np.linalg.norm(velocity))
        tangential = float(np.linalg.norm(velocity[:2])) > self.speed_threshold
        if velocity[2] > self.speed_threshold:
            normal = "press"
        elif velocity[2] < -self.speed_threshold:
            normal = "release"
        else:
            normal = "hold"
        return InteractionState(
            valid=self.active,
            moving=self.active and speed > self.speed_threshold,
            normal_state=normal if self.active else "invalid",
            tangential_motion=self.active and tangential,
            velocity_xyz_mm_s=tuple(float(v) for v in velocity),
            speed_mm_s=speed,
        )
