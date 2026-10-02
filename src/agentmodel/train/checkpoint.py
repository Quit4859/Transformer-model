"""Portable checkpoints for model weights and resumable training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .loop import Trainer


def save_checkpoint(path: str | Path, trainer: Trainer) -> None:
    """Save a checkpoint atomically enough for a single-process training run."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "config": trainer.cfg.to_dict(),
        "config_hash": trainer.cfg.config_hash(),
        "model": trainer.model.state_dict(),
        "adamw": trainer.adamw.state_dict(),
        "muon": trainer.muon.state_dict() if trainer.muon is not None else None,
        "step_index": trainer.step_index,
        "tokens_seen": trainer.tokens_seen,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_checkpoint(path: str | Path, trainer: Trainer, *, strict_config: bool = True) -> dict[str, Any]:
    """Restore a checkpoint into ``trainer`` and return its metadata."""
    payload = torch.load(path, map_location=trainer.device, weights_only=False)
    if payload.get("version") != 1:
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')!r}")
    if strict_config and payload.get("config_hash") != trainer.cfg.config_hash():
        raise ValueError("checkpoint configuration does not match the trainer configuration")
    trainer.model.load_state_dict(payload["model"])
    trainer.adamw.load_state_dict(payload["adamw"])
    if trainer.muon is not None and payload.get("muon") is not None:
        trainer.muon.load_state_dict(payload["muon"])
    trainer.step_index = int(payload.get("step_index", 0))
    trainer.tokens_seen = int(payload.get("tokens_seen", 0))
    return {
        "version": payload["version"],
        "config_hash": payload.get("config_hash", ""),
        "step_index": trainer.step_index,
        "tokens_seen": trainer.tokens_seen,
    }
