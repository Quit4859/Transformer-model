"""Grouped-query attention with RoPE, optional QK-Norm, output gating and SWA.

Grouped-query attention keeps `n_head` query heads but only `n_kv_head` key/value
heads, cutting KV-cache bytes by `n_kv_groups`. The reference path here uses
`F.scaled_dot_product_attention`; FlashAttention-3 and Triton kernels can be
swapped in behind the same interface once the reference path is gradient-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .mask import causal_mask
from .norm import RMSNorm
from .rope import RotaryEmbedding


@dataclass
class LayerKVCache:
    k: torch.Tensor  # [B, H_kv, T, D]
    v: torch.Tensor  # [B, H_kv, T, D]


class KVCache:
    """Simple append-only cache. A paged cache belongs here for long contexts."""

    def __init__(self, sliding_window: int | None = None) -> None:
        self.layers: list[LayerKVCache] = []
        self.sliding_window = sliding_window
        self.seen = 0

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.layers):
            self.layers.append(LayerKVCache(k=k, v=v))
        else:
            cur = self.layers[layer_idx]
            cur.k = torch.cat((cur.k, k), dim=2)
            cur.v = torch.cat((cur.v, v), dim=2)
        k_all, v_all = self.layers[layer_idx].k, self.layers[layer_idx].v
        if self.sliding_window is not None and k_all.shape[2] > self.sliding_window:
            k_all = k_all[:, :, -self.sliding_window :, :]
            v_all = v_all[:, :, -self.sliding_window :, :]
        return k_all, v_all

    def len(self, layer_idx: int) -> int:
        """Number of key positions already cached for a layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].k.shape[2]

    def get(self, layer_idx: int) -> LayerKVCache:
        return self.layers[layer_idx]

    def reset(self) -> None:
        self.layers.clear()
        self.seen = 0


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, H_kv, T, D] -> [B, H_kv * n_rep, T, D]."""
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class GatedAttention(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_kv_groups

        self.q_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.d_model, bias=False)

        self.use_gate = cfg.gated_attention and cfg.attn_output_gate
        if self.use_gate:
            self.gate_proj = nn.Linear(cfg.d_model, self.n_head * self.head_dim, bias=False)

        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.zero_centered_qk_norm)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.zero_centered_qk_norm)
        else:
            self.q_norm = self.k_norm = nn.Identity()

        self.rope = RotaryEmbedding(
            head_dim=self.head_dim,
            theta=cfg.rope_theta,
            partial_rope_dim=cfg.partial_rope_dim,
            max_seq_len=cfg.max_seq_len,
            scaling_type=cfg.rope_scaling.type,
            factor=cfg.rope_scaling.factor,
            original_max_seq_len=cfg.rope_scaling.original_max_seq_len,
        )

    def _cached_mask(self, q_pos: torch.Tensor, kv_len: int, dtype: torch.dtype) -> torch.Tensor:
        """Additive [B, 1, T, kv_len] mask for cached decode/prefill.

        Query position i may attend to key j when j <= pos_i and, for sliding
        window layers, pos_i - j < window.
        """
        b, t = q_pos.shape
        device = q_pos.device
        keys = torch.arange(kv_len, device=device)
        window = self.sliding_window if self.sliding_window is not None else kv_len
        allowed = (keys.view(1, 1, kv_len) <= q_pos.unsqueeze(-1)) & (
            q_pos.unsqueeze(-1) - keys.view(1, 1, kv_len) < window
        )
        mask = torch.zeros(b, 1, t, kv_len, device=device, dtype=dtype)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope.rotate(q, positions)
        k = self.rope.rotate(k, positions)

        if cache is not None and use_cache:
            offset = cache.len(self.layer_idx)
            k, v = cache.append(self.layer_idx, k, v)
            kv_len = k.shape[2]
            if positions is None:
                positions = (
                    torch.arange(offset, offset + t, device=x.device).unsqueeze(0).expand(b, t)
                )
            if mask is None:
                mask = self._cached_mask(q_pos=positions, kv_len=kv_len, dtype=q.dtype)
        elif mask is None:
            if self.sliding_window is not None and self.sliding_window < t:
                mask = causal_mask(t, self.sliding_window, x.device, q.dtype)
            else:
                mask = causal_mask(t, None, x.device, q.dtype)

        if mask.shape[0] == 1:
            mask = mask.expand(b, -1, -1, -1)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).reshape(b, t, self.n_head * self.head_dim)

        if self.use_gate:
            gate = torch.sigmoid(self.gate_proj(x)).view(b, t, self.n_head * self.head_dim)
            attn = attn * gate

        return self.o_proj(attn)
