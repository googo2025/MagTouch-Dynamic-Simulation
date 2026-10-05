#!/usr/bin/env python3
"""Causal magnetic inverse with fast position and slower field-memory state.

The slow magnetic state is owned by R122/R131.  This module only resolves
ambiguous *position* branches after magnetic-memory correction.  It never
averages XYZ, fits a path shape, uses a stage command, or examines future data.
"""
from __future__ import annotations

import numpy as np

from training.probe_had239_local_forward_r104 import ForwardAtlas


class BranchAwareInverse:
    def __init__(self, atlas: ForwardAtlas, *, spatial_penalty: float = .005,
                 innovation_gain: float = 2.0, max_gap_ns: int = 250_000_000,
                 activation_distance_mm: float = 1.0):
        self.atlas = atlas
        self.spatial_penalty = float(spatial_penalty)
        self.innovation_gain = float(innovation_gain)
        self.max_gap_ns = int(max_gap_ns)
        self.activation_distance_mm = float(activation_distance_mm)
        if (self.spatial_penalty < 0 or self.innovation_gain < 0 or
                self.activation_distance_mm < 0):
            raise ValueError("invalid branch-tracking penalty")
        self.reset()

    def reset(self) -> None:
        self.previous: np.ndarray | None = None
        self.previous_phi: np.ndarray | None = None
        self.previous_time_ns: int | None = None
        self.previous_b0_id: int | None = None
        self.previous_stream_id: int | None = None

    def _field_cost(self, point: np.ndarray, phi: np.ndarray) -> float:
        predicted = self.atlas.features(self.atlas.forward(point[None, :]))[0]
        return float(np.sum((phi - predicted) ** 2))

    def _local_refine(self, start: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Damped GN from the previous physical branch, with descent checks."""
        point = start.astype(float).copy()
        current_cost = self._field_cost(point, phi)
        for _ in range(4):
            jacobian = np.empty((len(phi), 3), float)
            for col, step in enumerate((.05, .05, .02)):
                plus, minus = point.copy(), point.copy()
                plus[col] += step
                minus[col] -= step
                jacobian[:, col] = (
                    self.atlas.features(self.atlas.forward(plus[None, :]))[0] -
                    self.atlas.features(self.atlas.forward(minus[None, :]))[0]
                ) / (2 * step)
            residual = phi - self.atlas.features(
                self.atlas.forward(point[None, :]))[0]
            jtj = jacobian.T @ jacobian
            damping = .01 * np.trace(jtj) / 3 + 1e-7
            jtj.flat[::4] += damping
            step_xyz = np.linalg.solve(jtj, jacobian.T @ residual)
            step_xyz = np.clip(step_xyz, [-.5, -.5, -.1], [.5, .5, .1])
            proposal = point + step_xyz
            for col, axis in enumerate((self.atlas.x, self.atlas.y,
                                        self.atlas.z)):
                proposal[col] = np.clip(proposal[col], axis[0], axis[-1])
            cost = self._field_cost(proposal, phi)
            if cost < current_cost:
                point, current_cost = proposal, cost
            else:
                break
        return point

    def predict_frame(self, field: np.ndarray, *, time_ns: int, b0_id: int,
                      stream_id: int, contact_active: bool = True,
                      global_xyz: np.ndarray | None = None) -> dict | None:
        if not contact_active:
            self.reset()
            return None
        field = np.asarray(field, float)
        if field.shape != (15,) or not np.isfinite(field).all():
            raise ValueError("expected 15 finite magnetic channels")
        phi = self.atlas.features(field[None, :])[0]
        global_xyz = (self.atlas.predict(field[None, :])[0] if global_xyz is None
                      else np.asarray(global_xyz, float))
        if global_xyz.shape != (3,) or not np.isfinite(global_xyz).all():
            raise ValueError("invalid global atlas estimate")
        now = int(time_ns)
        gap = None if self.previous_time_ns is None else now - self.previous_time_ns
        continuing = (self.previous is not None and gap is not None and
                      0 < gap <= self.max_gap_ns and
                      self.previous_b0_id == b0_id and
                      self.previous_stream_id == stream_id)
        local_xyz = global_xyz.copy()
        selected = global_xyz.copy()
        global_cost = self._field_cost(global_xyz, phi)
        local_cost = global_cost
        branch_switched = False
        if (continuing and np.linalg.norm(global_xyz[:2] - self.previous[:2]) >
                self.activation_distance_mm):
            local_xyz = self._local_refine(self.previous, phi)
            local_cost = self._field_cost(local_xyz, phi)
            # A large magnetic innovation can indicate genuinely fast motion.
            # Reduce the *soft* spatial prior in that case; never hard-clamp
            # velocity or take an EMA of position.
            innovation = float(np.linalg.norm(phi - self.previous_phi))
            dt = gap / 1e9
            scale = (max(dt, .025) / .1) ** 2 * (
                1 + self.innovation_gain * innovation) ** 2
            local_travel = float(np.sum((local_xyz[:2] - self.previous[:2]) ** 2))
            global_travel = float(np.sum((global_xyz[:2] - self.previous[:2]) ** 2))
            local_objective = local_cost + self.spatial_penalty * local_travel / scale
            global_objective = global_cost + self.spatial_penalty * global_travel / scale
            if local_objective < global_objective:
                selected = local_xyz
            else:
                branch_switched = global_travel > 1.0
        self.previous = selected.copy()
        self.previous_phi = phi.copy()
        self.previous_time_ns = now
        self.previous_b0_id = int(b0_id)
        self.previous_stream_id = int(stream_id)
        return {"xyz_mm": selected, "global_xyz_mm": global_xyz,
                "local_xyz_mm": local_xyz,
                "global_field_cost": global_cost,
                "local_field_cost": local_cost,
                "used_local_branch": bool(continuing and not branch_switched and
                                          not np.array_equal(selected, global_xyz)),
                "global_branch_switch_accepted": bool(branch_switched)}
