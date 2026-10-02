"""Optimizer construction: Muon for 2D hidden weights, AdamW for everything else.

Split rule: a parameter goes to Muon when it is exactly 2D, is a float param,
and lives on a hidden-hidden projection. Embeddings, the LM head, norms, biases
and any 1D parameter stay on AdamW.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
from torch.optim import Optimizer

from ..model.config import OptimConfig
from .optim import Muon


def is_muon_param(name: str, p: torch.nn.Parameter) -> bool:
    if p.ndim != 2:
        return False
    if not p.is_floating_point():
        return False
    if p.shape[0] < 16 or p.shape[1] < 16:
        return False
    return True


def split_params(
    model: nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("router.weight") or "router.bias" in name:
            adamw_params.append(p)
        elif is_muon_param(name, p):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


def build_optimizers(
    model: nn.Module, cfg: OptimConfig
) -> tuple[Optimizer, Optimizer | None]:
    muon_params, adamw_params = split_params(model)
    adamw = torch.optim.AdamW(
        adamw_params,
        lr=cfg.lr,
        betas=tuple(cfg.adamw_betas),
        weight_decay=cfg.weight_decay,
        eps=1e-8,
        fused=False,
    )
    muon = (
        Muon(
            muon_params,
            lr=cfg.muon_lr,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
        if muon_params and cfg.optimizer == "muon_adamw"
        else None
    )
    return adamw, muon


def lr_scale(step: int, cfg: OptimConfig) -> float:
    """Return a multiplier in (0, 1] applied to every group's base lr."""
    warmup = max(1, cfg.warmup_steps)
    if step < warmup:
        return (step + 1) / warmup
    if cfg.schedule == "constant":
        return 1.0
    decay_steps = max(warmup + 1, cfg.decay_steps)
    progress = min(1.0, (step - warmup) / (decay_steps - warmup))
    if cfg.schedule == "cosine":
        import math as _math

        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (
            1.0 + _math.cos(_math.pi * progress)
        )
    if cfg.schedule == "wsd":
        # stable plateau, then linear decay to min_lr_ratio
        if progress < 0.5:
            return 1.0
        t = (progress - 0.5) / 0.5
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1.0 - t)
    return 1.0


def apply_lr_scale(optimizers: Iterable, scale: float, base: dict) -> None:
    for opt in optimizers:
        if opt is None:
            continue
        for group in opt.param_groups:
            key = "muon_lr" if isinstance(opt, Muon) else "lr"
            group["lr"] = base[key] * scale


def param_groups_lr_base(cfg: OptimConfig) -> dict[str, float]:
    return {"lr": cfg.lr, "muon_lr": cfg.muon_lr}


def clip_grad_norm(model: nn.Module, max_norm: float) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def num_flops_per_token(cfg) -> float:
    """Approximate non-embedding FLOPs per token (6 * params as the rule of thumb)."""
    return 6.0 * cfg.n_params(non_embedding=True)
