"""Rotary position embeddings: RoPE, partial RoPE, and YaRN context scaling.

Partial RoPE applies rotation to only the first `partial_rope_dim` channels of
each head and leaves the remainder untouched. YaRN rescales the inverse
frequencies instead of interpolating positions, so a model trained at 2k can be
served at 128k without architectural changes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_inv_freq(
    head_dim: int,
    theta: float,
    partial_rope_dim: int = 0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    scaling_type: str = "none",
    factor: float = 1.0,
    original_max_seq_len: int = 2048,
) -> torch.Tensor:
    """Return inverse frequencies of shape [rope_dim // 2]."""
    rope_dim = head_dim if partial_rope_dim <= 0 else partial_rope_dim
    if rope_dim % 2 != 0:
        raise ValueError(f"rotary dim must be even, got {rope_dim}")
    half = rope_dim // 2

    # YaRN blends between linear and wavelength-based scaling per dimension so
    # high-frequency dims keep their original period.
    if scaling_type == "yarn" and factor != 1.0:
        inv = yarn_inv_freq(
            half, theta, factor, original_max_seq_len, device=device, dtype=dtype
        )
        return inv

    inv = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype).float() / half))
    if scaling_type == "linear" and factor != 1.0:
        inv = inv / factor
    return inv


def yarn_inv_freq(
    half: int,
    theta: float,
    factor: float,
    original_max_seq_len: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    attn_factor: float = 1.0,
) -> torch.Tensor:
    base = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=dtype).float() / half))
    # Linear ramp over the dimension index from slow to fast correction.
    ramp = torch.linspace(0, 1, half, device=device, dtype=dtype)
    low = math.floor(beta_slow)
    high = math.ceil(beta_fast)
    correction = (ramp * (high - low) + low).clamp(max=beta_fast / beta_fast * high)
    # correction ~ 32 at dim 0, ~1 at dim half-1
    correction = (1.0 / factor) + (correction - 1.0 / correction) * (1.0 / factor)
    inv = base / correction
    return inv * (attn_factor + 0.0)


def yarn_mscale(attn_factor: float = 1.0) -> float:
    return attn_factor


def apply_rope(
    x: torch.Tensor,
    inv_freq: torch.Tensor,
    positions: torch.Tensor,
    partial_rope_dim: int = 0,
) -> torch.Tensor:
    """Rotate `x` of shape [B, H, T, D] using raw inverse frequencies."""
    freqs = torch.einsum("bi,j->bij", positions.float(), inv_freq.float())
    cos = freqs.cos().to(x.dtype).unsqueeze(1)
    sin = freqs.sin().to(x.dtype).unsqueeze(1)
    return apply_rope_with_cos_sin(x, cos, sin, partial_rope_dim)


def apply_rope_with_cos_sin(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    partial_rope_dim: int = 0,
) -> torch.Tensor:
    """Rotate `x` of shape [B, H, T, D] given precomputed cos/sin.

    Only the leading `partial_rope_dim` channels are rotated when that is
    non-zero; the tail is copied through unchanged.
    """
    b, h, t, d = x.shape
    rope_dim = d if partial_rope_dim <= 0 else min(partial_rope_dim, d)
    if rope_dim == 0:
        return x

    xf = x[..., :rope_dim].float()
    x1, x2 = xf[..., ::2], xf[..., 1::2]
    cos = cos[..., : rope_dim // 2].float()
    sin = sin[..., : rope_dim // 2].float()
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    rotated = torch.stack((o1, o2), dim=-1).flatten(-2)

    if rope_dim < d:
        return torch.cat((rotated.to(x.dtype), x[..., rope_dim:]), dim=-1)
    return rotated.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        theta: float = 500_000.0,
        partial_rope_dim: int = 0,
        max_seq_len: int = 2048,
        scaling_type: str = "none",
        factor: float = 1.0,
        original_max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.partial_rope_dim = partial_rope_dim
        self.max_seq_len = max_seq_len
        inv_freq = build_inv_freq(
            head_dim,
            theta,
            partial_rope_dim,
            scaling_type=scaling_type,
            factor=factor,
            original_max_seq_len=original_max_seq_len,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("scaling_factor", torch.tensor(float(factor)), persistent=False)
        self._cached_len = 0
        self._cached_cos: torch.Tensor | None = None
        self._cached_sin: torch.Tensor | None = None

    def _cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cached_cos is None or self._cached_len < seq_len or self._cached_cos.device != device:
            pos = torch.arange(seq_len, device=device).unsqueeze(0)
            freqs = torch.einsum("bi,j->bij", pos.float(), self.inv_freq.to(device).float())
            self._cached_cos = freqs.cos().to(dtype)
            self._cached_sin = freqs.sin().to(dtype)
            self._cached_len = seq_len
        return self._cached_cos[:, :seq_len], self._cached_sin[:, :seq_len]

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) broadcastable to [B, 1, T, rope_dim/2]."""
        _, _, t, _ = x.shape
        if positions is None:
            cos, sin = self._cos_sin(t, x.device, x.dtype)
            return cos.unsqueeze(1), sin.unsqueeze(1)
        freqs = torch.einsum("bi,j->bij", positions.float(), self.inv_freq.to(x.device).float())
        cos = freqs.cos().to(x.dtype).unsqueeze(1)
        sin = freqs.sin().to(x.dtype).unsqueeze(1)
        return cos, sin

    def rotate(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """Rotate `x` of shape [B, H, T, D] in place-free fashion.

        Named `rotate` rather than `apply` so it does not shadow `nn.Module.apply`.
        """
        cos, sin = self.forward(x, positions)
        return apply_rope_with_cos_sin(x, cos, sin, self.partial_rope_dim)


def rotate_half_legacy(x: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX style interleaved-half rotation, for comparison against apply_rope."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)
