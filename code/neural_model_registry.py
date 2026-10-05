"""Trusted neural checkpoints exposed by both live and replay UIs."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LIVE_NEURAL_MODEL_OPTIONS = {
    "R56-soup｜验证集优选融合（推荐）": (
        PROJECT_ROOT / "models" / "r56_weight_soup_20260923" / "best.pt"
    ),
    "R56-seed20260930｜最佳单模型": (
        PROJECT_ROOT / "models" / "r56_event_relative_seed20260930_20260923" / "best.pt"
    ),
}


def resolve_neural_model_selection(value: str | Path) -> Path:
    """Resolve a trusted UI label or an explicitly browsed checkpoint path."""
    selected = str(value).strip()
    return Path(LIVE_NEURAL_MODEL_OPTIONS.get(selected, selected)).expanduser()


def neural_model_family(value: str | Path, fallback: str = "auto") -> str:
    """Return the forced family only for a registry-owned model option."""
    return "FinalTip-R56" if str(value).strip() in LIVE_NEURAL_MODEL_OPTIONS else fallback
