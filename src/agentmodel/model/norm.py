"""RMSNorm, including the zero-centered variant used by gated attention stacks.

`zero_centered_qk_norm` subtracts 1.0 from the learned gain instead of
initializing it to 1.0. Both are used in 2026-era hybrid stacks; the
zero-centered form is a little more stable inside mixed attention layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, zero_centered: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.weight * x
        if self.zero_centered:
            out = out - self.weight
        return out.to(dtype)

    def extra_repr(self) -> str:
        return f"eps={self.eps}, zero_centered={self.zero_centered}"


class LayerNorm(nn.Module):
    """Kept for ablation runs; not used by the default recipe."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
