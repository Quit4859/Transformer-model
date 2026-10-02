"""One optimizer step: forward, loss, backward, Muon/AdamW, clip, step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..data.packing import PackedBatch
from ..model.config import Config
from ..model.transformer import Transformer
from .autocast import autocast_ctx
from .losses import count_valid, entropy_of, token_cross_entropy
from .optim_setup import (
    apply_lr_scale,
    build_optimizers,
    clip_grad_norm,
    lr_scale,
    param_groups_lr_base,
)


@dataclass
class StepStats:
    step: int
    loss: float
    lr_scale: float
    grad_norm: float | None
    valid_tokens: int
    aux_loss: float = 0.0
    entropy: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "loss": round(self.loss, 6),
            "lr_scale": round(self.lr_scale, 6),
            "grad_norm": None if self.grad_norm is None else round(self.grad_norm, 6),
            "valid_tokens": self.valid_tokens,
            "aux_loss": round(self.aux_loss, 6),
            "entropy": round(self.entropy, 4),
            **self.extra,
        }


class Trainer:
    def __init__(self, cfg: Config, model: Transformer | None = None) -> None:
        self.cfg = cfg
        torch.manual_seed(cfg.train.seed)
        self.model = model or Transformer(cfg.model)
        self.model.train()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.adamw, self.muon = build_optimizers(self.model, cfg.optim)
        self.base_lr = param_groups_lr_base(cfg.optim)
        self.step_index = 0
        self.tokens_seen = 0

    def forward_loss(self, batch: PackedBatch) -> tuple[torch.Tensor, dict[str, float]]:
        batch = batch.to(self.device)
        with autocast_ctx(self.cfg.precision.compute_dtype):
            out = self.model(
                batch.input_ids,
                gradient_checkpointing=self.cfg.train.gradient_checkpointing,
            )
        loss = token_cross_entropy(out.logits, batch.targets)
        stats = {
            "valid_tokens": float(count_valid(batch.targets)),
            "entropy": float(entropy_of(out.logits).detach()),
        }
        if out.aux_loss is not None:
            stats["aux_loss"] = float(out.aux_loss.detach())
            loss = loss + out.aux_loss
        return loss, stats

    def optimizer_step(self) -> float:
        opts = [self.adamw, self.muon]
        gnorm = clip_grad_norm(self.model, self.cfg.optim.grad_clip)
        apply_lr_scale(opts, self.current_lr_scale, self.base_lr)
        for opt in opts:
            if opt is not None:
                opt.step()
        return float(gnorm)

    @property
    def current_lr_scale(self) -> float:
        return lr_scale(self.step_index, self.cfg.optim)

    def zero_grad(self) -> None:
        for opt in (self.adamw, self.muon):
            if opt is not None:
                opt.zero_grad(set_to_none=True)

    def micro_step(self, batch: PackedBatch, accumulate: bool = True) -> StepStats:
        loss, stats = self.forward_loss(batch)
        (loss / self.cfg.train.grad_accum_steps).backward()
        self.tokens_seen += int(batch.num_real_tokens)
        # grad_norm is only measurable on the micro-step that actually steps.
        gnorm: float | None = None
        if not accumulate or (self.step_index + 1) % self.cfg.train.grad_accum_steps == 0:
            gnorm = self.optimizer_step()
            self.zero_grad()
        return StepStats(
            step=self.step_index,
            loss=float(loss.detach()),
            lr_scale=self.current_lr_scale,
            grad_norm=gnorm,
            valid_tokens=int(stats["valid_tokens"]),
            aux_loss=stats.get("aux_loss", 0.0),
            entropy=stats.get("entropy", 0.0),
        )

    def advance(self) -> None:
        self.step_index += 1
