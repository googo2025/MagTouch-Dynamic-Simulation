"""Geometry-aware causal point tracker for the five-sensor MagTouch array.

The five magnetometers are treated as irregular spatial tokens at their real
coordinates.  A small graph-attention encoder produces one embedding per
magnetic frame.  The absolute head follows a YOLO/CenterNet-like formulation:
coarse XY cell probabilities plus a continuous offset and Z estimate.  The
temporal path is strictly causal and predicts displacement relative to the
causally captured contact-event reference.

No command, trajectory-family, target-position, trial, or domain identifier is
accepted by ``forward``.  Those values are labels/audit metadata only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SENSOR_XY_MM = torch.tensor([
    [-7.594, 7.594],
    [-7.594, -7.594],
    [7.594, 7.594],
    [7.594, -7.594],
    [0.0, 1.406],
], dtype=torch.float32)


@dataclass(frozen=True)
class PointTrackR49Config:
    grid_size: int = 5
    workspace_half_mm: float = 10.0
    node_hidden: int = 32
    frame_hidden: int = 64
    attention_heads: int = 4
    attention_layers: int = 2
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    dropout: float = 0.05


class SpatialAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        attended, _ = self.attention(normalized, normalized, normalized,
                                     need_weights=False)
        value = value + attended
        return value + self.ffn(self.norm2(value))


class GeometryFrameEncoder(nn.Module):
    """Encode one or more 15-D magnetic frames using real sensor geometry."""

    def __init__(self, config: PointTrackR49Config) -> None:
        super().__init__()
        self.config = config
        sensor_xy = SENSOR_XY_MM / config.workspace_half_mm
        self.register_buffer("sensor_xy", sensor_xy)
        self.node_input = nn.Sequential(
            # scaled raw XYZ, global unit-response XYZ slice, local log amplitude,
            # and the fixed physical XY coordinate.
            nn.Linear(9, config.node_hidden), nn.GELU(),
            nn.Linear(config.node_hidden, config.node_hidden),
        )
        self.blocks = nn.ModuleList([
            SpatialAttentionBlock(config.node_hidden, config.attention_heads,
                                  config.dropout)
            for _ in range(config.attention_layers)
        ])
        self.pool = nn.Sequential(
            nn.Linear(config.node_hidden * 2, config.frame_hidden), nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, delta_b: torch.Tensor, mean: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        if delta_b.shape[-1] != 15:
            raise ValueError(f"expected 15 magnetic channels, got {delta_b.shape}")
        leading = delta_b.shape[:-1]
        flat = delta_b.reshape(-1, 15)
        scaled = ((flat - mean) / scale).reshape(-1, 5, 3)
        amplitude = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
        unit = (flat / amplitude.clamp_min(1e-6)).reshape(-1, 5, 3)
        local_amplitude = torch.linalg.vector_norm(flat.reshape(-1, 5, 3),
                                                   dim=-1, keepdim=True)
        local_log_amplitude = torch.log1p(local_amplitude) / 8.0
        coordinates = self.sensor_xy.unsqueeze(0).expand(len(flat), -1, -1)
        nodes = self.node_input(torch.cat(
            [scaled, unit, local_log_amplitude, coordinates], dim=-1))
        for block in self.blocks:
            nodes = block(nodes)
        pooled = torch.cat([nodes.mean(dim=1), nodes.amax(dim=1)], dim=-1)
        return self.pool(pooled).reshape(*leading, self.config.frame_hidden)


class CausalResidualBlock(nn.Module):
    def __init__(self, hidden: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=dilation)
        # Per-time channel normalization is causal.  GroupNorm on [B,C,T]
        # would aggregate over the complete T axis and leak future samples.
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        convolved = self.conv(F.pad(value, (self.pad, 0)))
        normalized = self.norm(convolved.transpose(1, 2)).transpose(1, 2)
        return value + self.dropout(F.gelu(normalized))


class CausalTemporalPyramid(nn.Module):
    """Dilated TCN with a 127-frame receptive field for the default config."""

    def __init__(self, config: PointTrackR49Config) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            CausalResidualBlock(config.frame_hidden, dilation, config.dropout)
            for dilation in config.tcn_dilations
        ])
        self.output_norm = nn.LayerNorm(config.frame_hidden)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        value = frames.transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        return self.output_norm(value.transpose(1, 2))


class GridPointHead(nn.Module):
    """Coarse XY probability field plus local XY and Z regressions."""

    def __init__(self, hidden: int, grid_size: int,
                 workspace_half_mm: float) -> None:
        super().__init__()
        self.grid_size = int(grid_size)
        self.cell_count = self.grid_size ** 2
        centers = torch.linspace(-workspace_half_mm + workspace_half_mm / grid_size,
                                 workspace_half_mm - workspace_half_mm / grid_size,
                                 grid_size)
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        self.register_buffer("cell_centers", torch.stack([xx.flatten(), yy.flatten()], -1))
        self.offset_limit_mm = float(workspace_half_mm / grid_size)
        self.head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(),
            nn.Linear(128, self.cell_count * 4),
        )

    def forward(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.head(embedding).reshape(*embedding.shape[:-1], self.cell_count, 4)
        logits = raw[..., 0]
        offset = torch.tanh(raw[..., 1:3]) * self.offset_limit_mm
        z = raw[..., 3:4]
        cell_xyz = torch.cat([
            self.cell_centers + offset,
            z,
        ], dim=-1)
        probability = torch.softmax(logits, dim=-1)
        xyz = torch.sum(probability.unsqueeze(-1) * cell_xyz, dim=-2)
        winner = logits.argmax(dim=-1)
        gather = winner[..., None, None].expand(*winner.shape, 1, 3)
        hard_xyz = torch.gather(cell_xyz, -2, gather).squeeze(-2)
        return {
            "cell_logits": logits,
            "cell_probability": probability,
            "cell_offset_xy_mm": offset,
            "cell_z_mm": z.squeeze(-1),
            "cell_xyz_mm": cell_xyz,
            "absolute_xyz_mm": xyz,
            "hard_absolute_xyz_mm": hard_xyz,
            "winning_cell": winner,
        }


class GeometryPointTrackR49(nn.Module):
    """Single-contact absolute-origin and event-relative causal tracker."""

    def __init__(self, delta_mean: torch.Tensor | list[float],
                 delta_scale: torch.Tensor | list[float],
                 config: PointTrackR49Config | None = None) -> None:
        super().__init__()
        self.config = config or PointTrackR49Config()
        mean = torch.as_tensor(delta_mean, dtype=torch.float32).reshape(15)
        scale = torch.as_tensor(delta_scale, dtype=torch.float32).reshape(15).clamp_min(1.0)
        self.register_buffer("delta_mean", mean)
        self.register_buffer("delta_scale", scale)
        self.spatial = GeometryFrameEncoder(self.config)
        self.temporal = CausalTemporalPyramid(self.config)
        self.absolute_head = GridPointHead(
            self.config.frame_hidden, self.config.grid_size,
            self.config.workspace_half_mm)
        self.relative_head = nn.Sequential(
            nn.Linear(self.config.frame_hidden * 3, 96), nn.GELU(),
            nn.Dropout(self.config.dropout), nn.Linear(96, 3),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(self.config.frame_hidden, 32), nn.GELU(), nn.Linear(32, 1))
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.config.frame_hidden, 32), nn.GELU(), nn.Linear(32, 3))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def encode_frames(self, delta_b: torch.Tensor) -> torch.Tensor:
        return self.spatial(delta_b, self.delta_mean, self.delta_scale)

    def relative_from_embeddings(self, current: torch.Tensor,
                                 reference: torch.Tensor) -> torch.Tensor:
        value = torch.cat([current, reference, current - reference], dim=-1)
        zero = torch.cat([reference, reference,
                          torch.zeros_like(reference)], dim=-1)
        return self.relative_head(value) - self.relative_head(zero)

    def forward_static(self, delta_b: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode_frames(delta_b)
        result = self.absolute_head(embedding)
        result["embedding"] = embedding
        result["contact_logit"] = self.contact_head(embedding).squeeze(-1)
        result["log_sigma_xyz"] = self.uncertainty_head(embedding).clamp(-5.0, 3.0)
        return result

    def forward(self, delta_b: torch.Tensor, event_start: torch.Tensor,
                event_active: torch.Tensor) -> dict[str, torch.Tensor]:
        if delta_b.ndim != 3:
            raise ValueError("temporal input must have shape [batch,time,15]")
        frame = self.encode_frames(delta_b)
        temporal = self.temporal(frame)
        static = self.absolute_head(frame)
        batch, steps, _ = temporal.shape
        reference = torch.zeros_like(temporal[:, 0])
        origin = torch.zeros((batch, 3), dtype=temporal.dtype, device=temporal.device)
        reference_valid = torch.zeros(batch, dtype=torch.bool, device=temporal.device)
        relative_steps, origin_steps, valid_steps = [], [], []
        for step in range(steps):
            start = event_start[:, step].bool()
            active = event_active[:, step].bool()
            reference = torch.where(start[:, None], temporal[:, step], reference)
            origin = torch.where(start[:, None], static["hard_absolute_xyz_mm"][:, step], origin)
            reference_valid = torch.where(start, torch.ones_like(reference_valid),
                                          reference_valid) & active
            displacement = self.relative_from_embeddings(temporal[:, step], reference)
            displacement = torch.where(reference_valid[:, None], displacement,
                                       torch.zeros_like(displacement))
            relative_steps.append(displacement)
            origin_steps.append(origin)
            valid_steps.append(reference_valid)
        relative = torch.stack(relative_steps, dim=1)
        origins = torch.stack(origin_steps, dim=1)
        valid = torch.stack(valid_steps, dim=1)
        return {
            **static,
            "frame_embedding": frame,
            "temporal_embedding": temporal,
            "event_origin_xyz_mm": origins,
            "event_dxyz_mm": relative,
            "event_xyz_mm": origins + relative,
            "event_output_valid": valid,
            "contact_logit": self.contact_head(frame).squeeze(-1),
            "log_sigma_xyz": self.uncertainty_head(temporal).clamp(-5.0, 3.0),
        }


def cell_targets(xy_mm: torch.Tensor, head: GridPointHead,
                 gaussian_sigma_mm: float = 2.5) -> dict[str, torch.Tensor]:
    distance2 = torch.sum((xy_mm.unsqueeze(-2) - head.cell_centers) ** 2, dim=-1)
    soft = torch.softmax(-distance2 / (2.0 * gaussian_sigma_mm ** 2), dim=-1)
    index = distance2.argmin(dim=-1)
    center = head.cell_centers[index]
    return {"soft": soft, "index": index, "offset_xy_mm": xy_mm - center}


def static_grid_loss(outputs: dict[str, torch.Tensor], target_xyz: torch.Tensor,
                     head: GridPointHead) -> dict[str, torch.Tensor]:
    targets = cell_targets(target_xyz[..., :2], head)
    log_probability = torch.log_softmax(outputs["cell_logits"], dim=-1)
    heatmap = -(targets["soft"] * log_probability).sum(dim=-1).mean()
    index2 = targets["index"][..., None, None].expand(*targets["index"].shape, 1, 2)
    chosen_offset = torch.gather(outputs["cell_offset_xy_mm"], -2, index2).squeeze(-2)
    index1 = targets["index"][..., None]
    chosen_z = torch.gather(outputs["cell_z_mm"], -1, index1).squeeze(-1)
    offset = F.smooth_l1_loss(chosen_offset, targets["offset_xy_mm"], beta=0.25)
    depth = F.smooth_l1_loss(chosen_z, target_xyz[..., 2], beta=0.1)
    continuous = F.smooth_l1_loss(
        outputs["absolute_xyz_mm"], target_xyz, beta=0.25)
    hard = F.smooth_l1_loss(outputs["hard_absolute_xyz_mm"], target_xyz, beta=0.25)
    total = heatmap + offset + 2.0 * depth + 0.5 * continuous + 0.25 * hard
    return {"total": total, "heatmap": heatmap, "offset": offset,
            "depth": depth, "continuous": continuous, "hard": hard}


def checkpoint_payload(model: GeometryPointTrackR49,
                       metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "MagTouch-GeometryPointTrack-R49-1.0",
        "state_dict": model.state_dict(),
        "model_config": model.config.__dict__,
        "delta_mean": model.delta_mean.detach().cpu().tolist(),
        "delta_scale": model.delta_scale.detach().cpu().tolist(),
        "metadata": metadata,
    }
