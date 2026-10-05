#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V16 dual-coordinate MagTouch model.

Both heads consume the same DeltaB-derived history.  Stage/printer values are
targets only and are never model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn

from model_magtouch_ra_hw5 import HW5_SENSOR_XY, AlternatingTransformer, CoordinatePositionEncoder, AttnPool
from model_relative_tactile_hw5 import DeterministicAxisQGate


@dataclass
class SpatialRelativeOutput:
    fixed_xyz_norm: torch.Tensor
    dxyz_norm: torch.Tensor
    extras: Dict[str, torch.Tensor]


class SpatialRelativeHW5(nn.Module):
    def __init__(self, in_dim: int, feature: str = "R1", timesteps: int = 64,
                 d_model: int = 128, layers: int = 3, heads: int = 4,
                 dropout: float = 0.1, use_q: bool = True):
        super().__init__()
        self.use_q = bool(use_q)
        self.feature = feature
        timesteps = max(1, int(timesteps))
        self.register_buffer("sensor_xy", HW5_SENSOR_XY.clone(), persistent=True)
        self.qgate = DeterministicAxisQGate(feature) if use_q else None
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.qproj = nn.Linear(3, d_model) if use_q else None
        self.coord = CoordinatePositionEncoder(d_model)
        self.time = nn.Parameter(torch.randn(1, timesteps, 1, d_model) * 0.02)
        self.trunk = AlternatingTransformer(d_model, layers, heads, dropout, self.sensor_xy, 16.0)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttnPool(d_model, heads)
        self.fixed_xyz = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3)
        )
        self.relative_xyz = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3)
        )

    def forward(self, x: torch.Tensor, q_axis: torch.Tensor) -> SpatialRelativeOutput:
        if x.ndim != 4 or x.shape[2] != 5:
            raise ValueError(f"Expected [B,T,5,D], got {tuple(x.shape)}")
        if q_axis.shape[-2:] != (5, 3):
            raise ValueError(f"Expected Q [B,T,5,3], got {tuple(q_axis.shape)}")
        y = self.qgate(x, q_axis) if self.qgate is not None else x
        h = self.proj(y) + self.time[:, :x.shape[1]] + self.coord(self.sensor_xy).view(1, 1, 5, -1)
        if self.qproj is not None:
            h = h + self.qproj(q_axis)
        h = self.norm(self.trunk(h))
        pooled = self.pool(h)
        return SpatialRelativeOutput(
            fixed_xyz_norm=self.fixed_xyz(pooled),
            dxyz_norm=self.relative_xyz(pooled),
            extras={"pooled": pooled},
        )
