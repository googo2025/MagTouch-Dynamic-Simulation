"""R50: global shape-amplitude branch fused with R49 sensor geometry tokens."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from training.model_geometry_pointtrack_r49 import (
    GeometryFrameEncoder,
    GeometryPointTrackR49,
    PointTrackR49Config,
)


class GeometryGlobalFrameEncoder(GeometryFrameEncoder):
    """Preserve the complete ordered 15-channel response alongside the graph."""

    def __init__(self, config: PointTrackR49Config) -> None:
        super().__init__(config)
        self.global_shape_amplitude = nn.Sequential(
            nn.Linear(31, 96), nn.GELU(), nn.Dropout(config.dropout),
            nn.Linear(96, config.frame_hidden), nn.GELU(),
        )
        self.graph_global_fusion = nn.Sequential(
            nn.Linear(config.frame_hidden * 2, 96), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(96, config.frame_hidden),
        )

    def forward(self, delta_b: torch.Tensor, mean: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        graph = super().forward(delta_b, mean, scale)
        leading = delta_b.shape[:-1]
        flat = delta_b.reshape(-1, 15)
        scaled = (flat - mean) / scale
        amplitude = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
        unit = flat / amplitude.clamp_min(1e-6)
        global_features = torch.cat([unit, scaled, torch.log1p(amplitude) / 8.0], dim=-1)
        global_embedding = self.global_shape_amplitude(global_features).reshape(
            *leading, self.config.frame_hidden)
        return self.graph_global_fusion(torch.cat([graph, global_embedding], dim=-1))


class GeometryPointTrackR50(GeometryPointTrackR49):
    """R49 heads/TCN with a dual global+geometry spatial encoder and refinement."""

    def __init__(self, delta_mean: torch.Tensor | list[float],
                 delta_scale: torch.Tensor | list[float],
                 config: PointTrackR49Config | None = None) -> None:
        super().__init__(delta_mean, delta_scale, config)
        self.spatial = GeometryGlobalFrameEncoder(self.config)
        self.absolute_refinement = nn.Sequential(
            nn.Linear(self.config.frame_hidden, 64), nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward_static(self, delta_b: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode_frames(delta_b)
        result = self.absolute_head(embedding)
        limit = embedding.new_tensor([2.0, 2.0, 0.5])
        refinement = torch.tanh(self.absolute_refinement(embedding)) * limit
        result["grid_absolute_xyz_mm"] = result["absolute_xyz_mm"]
        result["grid_hard_absolute_xyz_mm"] = result["hard_absolute_xyz_mm"]
        result["absolute_refinement_mm"] = refinement
        result["absolute_xyz_mm"] = result["absolute_xyz_mm"] + refinement
        result["hard_absolute_xyz_mm"] = result["hard_absolute_xyz_mm"] + refinement
        result["embedding"] = embedding
        result["contact_logit"] = self.contact_head(embedding).squeeze(-1)
        result["log_sigma_xyz"] = self.uncertainty_head(embedding).clamp(-5.0, 3.0)
        return result

    def forward(self, delta_b: torch.Tensor, event_start: torch.Tensor,
                event_active: torch.Tensor) -> dict[str, torch.Tensor]:
        # Keep the inherited causal relative path but use the refined R50
        # absolute event origins.  The explicit loop is small (T<=128).
        if delta_b.ndim != 3:
            raise ValueError("temporal input must have shape [batch,time,15]")
        frame = self.encode_frames(delta_b)
        temporal = self.temporal(frame)
        static = self.forward_static(delta_b)
        batch, steps, _ = temporal.shape
        reference = torch.zeros_like(temporal[:, 0])
        origin = torch.zeros((batch, 3), dtype=temporal.dtype, device=temporal.device)
        valid_state = torch.zeros(batch, dtype=torch.bool, device=temporal.device)
        relative_steps, origin_steps, valid_steps = [], [], []
        for step in range(steps):
            start = event_start[:, step].bool()
            active = event_active[:, step].bool()
            reference = torch.where(start[:, None], temporal[:, step], reference)
            origin = torch.where(start[:, None], static["hard_absolute_xyz_mm"][:, step], origin)
            valid_state = torch.where(start, torch.ones_like(valid_state), valid_state) & active
            displacement = self.relative_from_embeddings(temporal[:, step], reference)
            displacement = torch.where(valid_state[:, None], displacement,
                                       torch.zeros_like(displacement))
            relative_steps.append(displacement)
            origin_steps.append(origin)
            valid_steps.append(valid_state)
        relative = torch.stack(relative_steps, dim=1)
        origins = torch.stack(origin_steps, dim=1)
        return {
            **static,
            "frame_embedding": frame,
            "temporal_embedding": temporal,
            "event_origin_xyz_mm": origins,
            "event_dxyz_mm": relative,
            "event_xyz_mm": origins + relative,
            "event_output_valid": torch.stack(valid_steps, dim=1),
        }


def checkpoint_payload_r50(model: GeometryPointTrackR50,
                           metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "MagTouch-GeometryPointTrack-R50-1.0",
        "state_dict": model.state_dict(),
        "model_config": model.config.__dict__,
        "delta_mean": model.delta_mean.detach().cpu().tolist(),
        "delta_scale": model.delta_scale.detach().cpu().tolist(),
        "metadata": metadata,
    }

