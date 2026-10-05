"""R51 evidence-selected hybrid: frozen R48 origin + R50 causal displacement."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from training.model_geometry_pointtrack_r49 import PointTrackR49Config
from training.model_geometry_pointtrack_r50 import GeometryPointTrackR50
from training.train_static_grid_nn_r48 import ShapeDepthDualMLP


class FrozenR48Absolute(nn.Module):
    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("architecture") != "shape_depth_dual":
            raise ValueError("R51 requires the R48 shape_depth_dual checkpoint")
        self.model = ShapeDepthDualMLP()
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        normalization = payload["normalization"]
        for name in ("x_mean", "x_scale", "y_mean", "y_scale"):
            self.register_buffer(name, torch.as_tensor(
                normalization[name], dtype=torch.float32))

    def forward(self, delta_b: torch.Tensor) -> torch.Tensor:
        leading = delta_b.shape[:-1]
        flat = delta_b.reshape(-1, 15)
        amplitude = torch.linalg.vector_norm(flat, dim=-1, keepdim=True)
        unit = flat / amplitude.clamp_min(1e-6)
        features = torch.cat([unit, flat, torch.log1p(amplitude)], dim=-1)
        normalized = (features - self.x_mean) / self.x_scale
        prediction = self.model(normalized) * self.y_scale + self.y_mean
        return prediction.reshape(*leading, 3)


class HybridPointTrackR51(nn.Module):
    """Causal event tracker; no command or trajectory labels enter inference."""

    def __init__(self, r48_checkpoint: str | Path,
                 r50_checkpoint: str | Path) -> None:
        super().__init__()
        self.absolute_origin = FrozenR48Absolute(r48_checkpoint)
        payload = torch.load(r50_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema") != "MagTouch-GeometryPointTrack-R50-1.0":
            raise ValueError("R51 requires an R50 checkpoint")
        config_values = dict(payload["model_config"])
        config_values["tcn_dilations"] = tuple(config_values["tcn_dilations"])
        self.relative_tracker = GeometryPointTrackR50(
            payload["delta_mean"], payload["delta_scale"],
            PointTrackR49Config(**config_values))
        self.relative_tracker.load_state_dict(payload["state_dict"])
        self.relative_tracker.eval()
        self.source_metadata = payload.get("metadata", {})

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def forward(self, delta_b: torch.Tensor, event_start: torch.Tensor,
                event_active: torch.Tensor) -> dict[str, torch.Tensor]:
        if delta_b.ndim != 3 or delta_b.shape[-1] != 15:
            raise ValueError("expected [batch,time,15] delta-B")
        static_xyz = self.absolute_origin(delta_b)
        frame = self.relative_tracker.encode_frames(delta_b)
        temporal = self.relative_tracker.temporal(frame)
        batch, steps, _ = temporal.shape
        reference = torch.zeros_like(temporal[:, 0])
        origin = torch.zeros((batch, 3), dtype=temporal.dtype, device=temporal.device)
        valid_state = torch.zeros(batch, dtype=torch.bool, device=temporal.device)
        origins, displacements, valid_steps = [], [], []
        for step in range(steps):
            start = event_start[:, step].bool()
            active = event_active[:, step].bool()
            reference = torch.where(start[:, None], temporal[:, step], reference)
            origin = torch.where(start[:, None], static_xyz[:, step], origin)
            valid_state = torch.where(start, torch.ones_like(valid_state), valid_state) & active
            displacement = self.relative_tracker.relative_from_embeddings(
                temporal[:, step], reference)
            displacement = torch.where(valid_state[:, None], displacement,
                                       torch.zeros_like(displacement))
            origins.append(origin)
            displacements.append(displacement)
            valid_steps.append(valid_state)
        event_origin = torch.stack(origins, dim=1)
        event_dxyz = torch.stack(displacements, dim=1)
        return {
            "static_r48_xyz_mm": static_xyz,
            "event_origin_xyz_mm": event_origin,
            "event_dxyz_mm": event_dxyz,
            "xyz_mm": event_origin + event_dxyz,
            "event_output_valid": torch.stack(valid_steps, dim=1),
            "contact_logit": self.relative_tracker.contact_head(frame).squeeze(-1),
            "log_sigma_xyz": self.relative_tracker.uncertainty_head(temporal).clamp(-5.0, 3.0),
        }


def checkpoint_payload_r51(model: HybridPointTrackR51,
                           metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "MagTouch-HybridPointTrack-R51-1.0",
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }

