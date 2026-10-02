"""Evaluation helpers for packed language-model batches."""

from __future__ import annotations

import math

import torch

from ..data.packing import PackedBatch
from .loop import Trainer


@torch.no_grad()
def evaluate(trainer: Trainer, batches: list[PackedBatch]) -> dict[str, float]:
    """Return mean token loss and perplexity over evaluation batches."""
    if not batches:
        raise ValueError("batches must not be empty")
    was_training = trainer.model.training
    trainer.model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        loss, stats = trainer.forward_loss(batch)
        tokens = int(stats["valid_tokens"])
        total_loss += float(loss) * tokens
        total_tokens += tokens
    if was_training:
        trainer.model.train()
    if total_tokens == 0:
        raise ValueError("evaluation batches contain no valid target tokens")
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss)}
