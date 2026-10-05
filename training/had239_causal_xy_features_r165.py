"""Past-only features for the had-239/5 mm XY residual experiment."""
from __future__ import annotations

from collections import deque

import numpy as np


class CausalXYFeatureHistory:
    delays_ns = (100_000_000, 400_000_000)
    max_gap_ns = 250_000_000

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque()
        self.last_time_ns = None
        self.last_b0_id = None
        self.last_stream_id = None

    def push(self, *, time_ns, b0_id, stream_id, xyz, phi, memory):
        t = int(time_ns)
        point = np.asarray(xyz, float)
        field_feature = np.asarray(phi, float)
        state = np.asarray(memory, float)
        if (point.shape != (3,) or field_feature.shape != (16,) or
                state.shape != (2,) or not np.isfinite(point).all() or
                not np.isfinite(field_feature).all() or not np.isfinite(state).all()):
            raise ValueError("invalid causal XY feature input")
        if self.last_time_ns is not None and (
                t <= self.last_time_ns or t - self.last_time_ns > self.max_gap_ns or
                int(b0_id) != self.last_b0_id or int(stream_id) != self.last_stream_id):
            self.reset()
        blocks = [point / np.array([10., 10., 3.]), field_feature,
                  state / 1.5]
        for delay in self.delays_ns:
            previous = next((v for when, v in reversed(self.history)
                             if when <= t - delay), None)
            blocks.append(field_feature - previous if previous is not None
                          else np.zeros(16))
        vector = np.concatenate(blocks)
        self.history.append((t, field_feature.copy()))
        while self.history and self.history[0][0] < t - 600_000_000:
            self.history.popleft()
        self.last_time_ns = t
        self.last_b0_id = int(b0_id)
        self.last_stream_id = int(stream_id)
        return vector


def offline_features(data, atlas):
    """Apply the exact runtime feature function to stored accepted frames."""
    t = data["time"]
    valid = data["valid"]
    points = data["guarded"]
    phi = np.full((len(t), 16), np.nan)
    phi[valid] = atlas.features(data["corrected"][valid])
    out = np.zeros((len(t), 53))
    history = CausalXYFeatureHistory()
    for i in range(len(t)):
        if not valid[i]:
            history.reset()
            continue
        out[i] = history.push(time_ns=t[i], b0_id=0, stream_id=0,
                              xyz=points[i], phi=phi[i],
                              memory=data["memory"][i])
    return out
