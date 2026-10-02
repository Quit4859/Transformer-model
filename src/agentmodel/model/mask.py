"""Attention masks: full causal and sliding-window causal.

Masks are built as additive float bias tensors of shape [1, 1, T, T] so they
compose with `F.scaled_dot_product_attention` without needing an is_causal flag,
which is required when some layers are local and some are global.
"""

from __future__ import annotations

import torch


def sliding_window_causal_mask(
    t: int,
    window: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Additive mask where position i may attend to j <= i and i - j < window.

    The diagonal is always included, so a window smaller than 1 is meaningless.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    idx = torch.arange(t, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)  # [j, i]
    distance = idx.unsqueeze(1) - idx.unsqueeze(0)  # [i, j] = i - j
    allowed = causal & (distance < window)
    mask = torch.zeros(t, t, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None, :, :]


def full_causal_mask(
    t: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    idx = torch.arange(t, device=device)
    causal = idx.unsqueeze(0) <= idx.unsqueeze(1)
    mask = torch.zeros(t, t, device=device, dtype=dtype)
    mask.masked_fill_(~causal, torch.finfo(dtype).min)
    return mask[None, None, :, :]


def causal_mask(
    t: int,
    window: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return (
        full_causal_mask(t, device, dtype)
        if window is None
        else sliding_window_causal_mask(t, window, device, dtype)
    )


def kv_window_length(local_window: int) -> int:
    return max(1, local_window)
