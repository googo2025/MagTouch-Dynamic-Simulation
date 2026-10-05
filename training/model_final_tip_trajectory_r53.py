"""R53 final-tip causal trajectory regressor.

The model treats five 3-axis magnetometers as a compact spatiotemporal signal,
not as a natural camera grid.  It retains the final-tip R52 static observation
and learns a bounded, causal history-dependent correction from raw shape,
amplitude and multi-scale temporal differences.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.model_hybrid_pointtrack_r51 import FrozenR48Absolute


@dataclass(frozen=True)
class FinalTipTrajectoryR53Config:
    hidden: int = 96
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    kernel_size: int = 3
    dropout: float = 0.08
    correction_limit_xyz_mm: tuple[float, float, float] = (5.0, 5.0, 1.2)

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, width: int, dilation: int, kernel_size: int,
                 dropout: float) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            width, width, kernel_size, dilation=dilation, groups=width)
        self.mix = nn.Conv1d(width, width * 2, 1)
        self.project = nn.Conv1d(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        convolved = self.depthwise(nn.functional.pad(value, (self.left_padding, 0)))
        left, gate = self.mix(convolved).chunk(2, dim=1)
        update = torch.tanh(left) * torch.sigmoid(gate)
        update = self.dropout(self.project(update))
        return self.norm((residual + update).transpose(1, 2)).transpose(1, 2)


class FinalTipTrajectoryR53(nn.Module):
    feature_dim = 79

    def __init__(self, static_checkpoint: str | Path,
                 delta_mean: list[float] | torch.Tensor,
                 delta_scale: list[float] | torch.Tensor,
                 config: FinalTipTrajectoryR53Config | None = None) -> None:
        super().__init__()
        self.config = config or FinalTipTrajectoryR53Config()
        self.static_observer = FrozenR48Absolute(static_checkpoint)
        for parameter in self.static_observer.parameters():
            parameter.requires_grad_(False)
        mean = torch.as_tensor(delta_mean, dtype=torch.float32).reshape(15)
        scale = torch.as_tensor(delta_scale, dtype=torch.float32).reshape(15).clamp_min(1.0)
        self.register_buffer("delta_mean", mean)
        self.register_buffer("delta_scale", scale)
        self.input_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.config.hidden),
            nn.LayerNorm(self.config.hidden), nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.temporal = nn.ModuleList([
            CausalDepthwiseBlock(self.config.hidden, dilation,
                                 self.config.kernel_size, self.config.dropout)
            for dilation in self.config.dilations
        ])
        self.correction_head = nn.Sequential(
            nn.Linear(self.config.hidden, 64), nn.GELU(), nn.Linear(64, 3))
        self.increment_head = nn.Sequential(
            nn.Linear(self.config.hidden, 48), nn.GELU(), nn.Linear(48, 3))
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.config.hidden, 32), nn.GELU(), nn.Linear(32, 3))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def train(self, mode: bool = True) -> "FinalTipTrajectoryR53":
        super().train(mode)
        # The R52 observation is a frozen deterministic baseline; parent
        # train() must not reactivate its Dropout layers.
        self.static_observer.eval()
        return self

    def _lag_difference(self, scaled: torch.Tensor, lag: int) -> torch.Tensor:
        result = torch.zeros_like(scaled)
        if scaled.shape[1] > lag:
            result[:, lag:] = scaled[:, lag:] - scaled[:, :-lag]
        return result

    def features(self, delta_b: torch.Tensor,
                 static_xyz: torch.Tensor) -> torch.Tensor:
        scaled = (delta_b - self.delta_mean) / self.delta_scale
        amplitude = torch.linalg.vector_norm(delta_b, dim=-1, keepdim=True)
        unit = delta_b / amplitude.clamp_min(1e-6)
        differences = [self._lag_difference(scaled, lag) for lag in (1, 4, 16)]
        static_scaled = static_xyz / static_xyz.new_tensor([10.0, 10.0, 3.0])
        return torch.cat([
            scaled, unit, torch.log1p(amplitude) / 8.0,
            *differences, static_scaled,
        ], dim=-1)

    def forward(self, delta_b: torch.Tensor) -> dict[str, torch.Tensor]:
        if delta_b.ndim != 3 or delta_b.shape[-1] != 15:
            raise ValueError("R53 expects [batch,time,15] delta-B")
        # Source bundles retain pre-B0/invalid frames as NaN for provenance.
        # Validity masks govern supervision; the network receives a neutral
        # finite placeholder so an invalid prefix cannot poison later states.
        delta_b = torch.nan_to_num(delta_b, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            static_xyz = self.static_observer(delta_b)
        encoded = self.input_encoder(self.features(delta_b, static_xyz))
        temporal = encoded.transpose(1, 2)
        for block in self.temporal:
            temporal = block(temporal)
        temporal = temporal.transpose(1, 2)
        limit = temporal.new_tensor(self.config.correction_limit_xyz_mm)
        correction = torch.tanh(self.correction_head(temporal)) * limit
        return {
            "xyz_mm": static_xyz + correction,
            "static_xyz_mm": static_xyz,
            "correction_xyz_mm": correction,
            "increment_1_mm": self.increment_head(temporal),
            "log_sigma_xyz": self.uncertainty_head(temporal).clamp(-5.0, 3.0),
            "temporal_embedding": temporal,
        }


def checkpoint_payload_r53(model: FinalTipTrajectoryR53,
                           static_checkpoint: str | Path,
                           metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "MagTouch-FinalTip-Trajectory-R53-1.0",
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "delta_mean": model.delta_mean.detach().cpu().tolist(),
        "delta_scale": model.delta_scale.detach().cpu().tolist(),
        "static_checkpoint": str(Path(static_checkpoint).resolve()),
        "metadata": metadata,
    }


def load_checkpoint_r53(path: str | Path,
                        device: torch.device | str = "cpu") -> FinalTipTrajectoryR53:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "MagTouch-FinalTip-Trajectory-R53-1.0":
        raise ValueError("not an R53 checkpoint")
    config = dict(payload["model_config"])
    config["dilations"] = tuple(config["dilations"])
    config["correction_limit_xyz_mm"] = tuple(config["correction_limit_xyz_mm"])
    model = FinalTipTrajectoryR53(
        payload["static_checkpoint"], payload["delta_mean"],
        payload["delta_scale"], FinalTipTrajectoryR53Config(**config))
    model.load_state_dict(payload["model_state"])
    return model.to(device)
