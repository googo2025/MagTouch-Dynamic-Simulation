"""R54 long-context final-tip trajectory model with an explicit event dXYZ head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from training.model_final_tip_trajectory_r53 import (
    FinalTipTrajectoryR53,
    FinalTipTrajectoryR53Config,
)


@dataclass(frozen=True)
class FinalTipTrajectoryR54Config(FinalTipTrajectoryR53Config):
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class FinalTipTrajectoryR54(FinalTipTrajectoryR53):
    def __init__(self, static_checkpoint: str | Path,
                 delta_mean: list[float] | torch.Tensor,
                 delta_scale: list[float] | torch.Tensor,
                 config: FinalTipTrajectoryR54Config | None = None) -> None:
        super().__init__(static_checkpoint, delta_mean, delta_scale,
                         config or FinalTipTrajectoryR54Config())
        width = self.config.hidden
        self.relative_head = nn.Sequential(
            nn.Linear(width * 3, 128), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))

    def relative_from_embeddings(self, current: torch.Tensor,
                                 reference: torch.Tensor) -> torch.Tensor:
        difference = current - reference
        value = torch.cat([current, reference, difference], dim=-1)
        zero = torch.cat([reference, reference, torch.zeros_like(reference)], dim=-1)
        return self.relative_head(value) - self.relative_head(zero)

    def forward(self, delta_b: torch.Tensor,
                event_reference_index: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        output = super().forward(delta_b)
        batch, steps, _ = output["temporal_embedding"].shape
        if event_reference_index is None:
            output["event_dxyz_mm"] = torch.zeros(
                batch, steps, 3, device=delta_b.device, dtype=delta_b.dtype)
            output["event_output_valid"] = torch.zeros(
                batch, steps, device=delta_b.device, dtype=torch.bool)
            return output
        if tuple(event_reference_index.shape) != (batch, steps):
            raise ValueError("event_reference_index must have shape [batch,time]")
        valid = (event_reference_index >= 0) & (event_reference_index < steps)
        safe = event_reference_index.clamp(0, max(steps - 1, 0)).long()
        reference = torch.gather(
            output["temporal_embedding"], 1,
            safe[..., None].expand(batch, steps, self.config.hidden))
        relative = self.relative_from_embeddings(output["temporal_embedding"], reference)
        output["event_dxyz_mm"] = torch.where(
            valid[..., None], relative, torch.zeros_like(relative))
        output["event_output_valid"] = valid
        return output


def checkpoint_payload_r54(model: FinalTipTrajectoryR54,
                           static_checkpoint: str | Path,
                           metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "MagTouch-FinalTip-Trajectory-R54-1.0",
        "model_state": model.state_dict(),
        "model_config": asdict(model.config),
        "delta_mean": model.delta_mean.detach().cpu().tolist(),
        "delta_scale": model.delta_scale.detach().cpu().tolist(),
        "static_checkpoint": str(Path(static_checkpoint).resolve()),
        "metadata": metadata,
    }


def load_checkpoint_r54(path: str | Path,
                        device: torch.device | str = "cpu") -> FinalTipTrajectoryR54:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "MagTouch-FinalTip-Trajectory-R54-1.0":
        raise ValueError("not an R54 checkpoint")
    config = dict(payload["model_config"])
    config["dilations"] = tuple(config["dilations"])
    config["correction_limit_xyz_mm"] = tuple(config["correction_limit_xyz_mm"])
    model = FinalTipTrajectoryR54(
        payload["static_checkpoint"], payload["delta_mean"], payload["delta_scale"],
        FinalTipTrajectoryR54Config(**config))
    model.load_state_dict(payload["model_state"])
    return model.to(device)
