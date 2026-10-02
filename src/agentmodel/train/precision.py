"""Precision helpers: dtype resolution and an optional EMA of weights."""

from __future__ import annotations

import torch
import torch.nn as nn

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3": getattr(torch, "float8_e4m3fn", None),
    "fp8_e5m2": getattr(torch, "float8_e5m2", None),
}


def resolve_dtype(name: str) -> torch.dtype:
    dtype = DTYPES.get(name)
    if dtype is None:
        raise ValueError(f"unknown dtype {name}; known: {sorted(DTYPES)}")
    return dtype


class EMA:
    """Exponential moving average of model parameters, fp32 master copy."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point and k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict({k: v.clone() for k, v in self.shadow.items()}, strict=False)
