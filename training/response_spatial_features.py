"""Causal per-sensor rotation-invariant response features, no learned fit.

Each sensor stays in its physical order; axes may rotate per sensor without
changing these norm/dot features. Feature histories never cross invalid
samples, repeated timestamps or a >0.25 s gap. No command/domain input.
"""
from __future__ import annotations

import numpy as np
from training.probe_r245_joint_contact_state import causal_state_features
from training.probe_r269_hierarchical_response_state import SCALAR_IDS


def spatial_features(time, field, eligible, variant="local_absolute"):
    time = np.asarray(time, np.int64)
    field = np.asarray(field, float)
    eligible = np.asarray(eligible, bool)
    scalar = causal_state_features(time, field, eligible)[:, SCALAR_IDS]
    if variant == "scalar":
        return scalar
    if variant not in ("local_absolute", "local_relative"):
        raise ValueError("known spatial feature variant required")
    n = len(time); triads = field.reshape(n, 5, 3)
    overall = np.linalg.norm(field, axis=1)
    sensor = np.linalg.norm(triads, axis=2)
    reset = ~eligible.copy(); reset[0] = True
    reset[1:] |= (~eligible[:-1]) | (np.diff(time) > 250_000_000) | (np.diff(time) <= 0)
    start = np.maximum.accumulate(np.where(reset, np.arange(n), 0))
    shape = sensor / np.maximum(overall[:, None], 1.)
    pieces = [scalar]
    if variant == "local_absolute":
        pieces.append(sensor/100.)
    pieces.append(shape)
    for lag in (100_000_000, 300_000_000, 700_000_000, 2_000_000_000, 5_000_000_000):
        ids = np.minimum(np.maximum(np.searchsorted(time, time-lag, side="right")-1, start), np.arange(n))
        prior = triads[ids]; delta = triads-prior
        changes = np.column_stack((sensor-np.linalg.norm(prior, axis=2),
            np.linalg.norm(delta, axis=2),
            np.sum(triads*delta, axis=2)/np.maximum(sensor, 1.)))
        if variant == "local_absolute":
            changes /= 100.
        else:
            # A fixed magnetic floor, not a target-fitted amplitude scale.
            changes = np.clip(changes/np.maximum(overall[:, None], 20.), -10., 10.)
        pieces.append(changes)
    return np.column_stack(pieces)
