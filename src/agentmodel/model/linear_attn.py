"""Gated DeltaNet: linear-attention block with a gated delta-rule state update.

This is the linear sequence mixer used by 2026-era hybrid stacks (Qwen3-Next,
gpt-oss) at a 3:1 ratio against gated full attention. It is O(T) in sequence
length and carries state in a fixed-size matrix, so it never grows a KV cache.

Reference recurrence, per head, with S as a [D, D] matrix whose columns are
value slots:

    S_t = a_t * S_{t-1} + beta_t * k_t (v_t - S_{t-1} k_t)^T
    o_t = q_t S_t

`beta_t` is the delta-rule write strength and `a_t` is the decay gate, both in
(0, 1) so state forgets exponentially. This is the plain-torch reference path;
a chunk-parallel Triton kernel (the associative/UT transform) is a later phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import KVCache
from .config import ModelConfig


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        d = cfg.d_model
        self.head_dim = cfg.head_dim
        self.num_q_heads = cfg.n_head
        self.num_k_heads = cfg.n_kv_head if cfg.n_kv_head > 0 else cfg.n_head
        self.num_v_heads = cfg.n_head

        self.q_proj = nn.Linear(d, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.num_k_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.num_v_heads * self.head_dim, bias=False)
        self.b_proj = nn.Linear(d, self.num_v_heads, bias=False)
        self.a_proj = nn.Linear(d, self.num_v_heads, bias=False)
        self.o_proj = nn.Linear(self.num_v_heads * self.head_dim, d, bias=False)

    def _heads(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        b, t, d = x.shape

        q = l2norm(self._heads(self.q_proj(x), self.num_q_heads))
        k = l2norm(self._heads(self.k_proj(x), self.num_k_heads))
        v = self._heads(self.v_proj(x), self.num_v_heads)
        beta = torch.sigmoid(self.b_proj(x))  # [B, T, Hv] write strength
        a = torch.sigmoid(self.a_proj(x))  # [B, T, Hv] decay gate

        if self.num_v_heads % self.num_q_heads == 0:
            q = q.repeat_interleave(self.num_v_heads // self.num_q_heads, dim=2)
        if self.num_v_heads % self.num_k_heads == 0:
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        out = self._sequential_recurrence(q, k, v, beta, a)
        out = out.reshape(b, t, self.num_v_heads * self.head_dim)
        return self.o_proj(out)

    def _sequential_recurrence(
        self,
        q: torch.Tensor,  # [B, T, H, D]
        k: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, H, D]
        beta: torch.Tensor,  # [B, T, H]
        a: torch.Tensor,  # [B, T, H]
    ) -> torch.Tensor:
        b, t, h, dh = q.shape
        out = torch.zeros_like(q)
        s = q.new_zeros(b, h, dh, dh)
        for i in range(t):
            q_i = q[:, i]  # [B, H, D]
            k_i = k[:, i]
            v_i = v[:, i]
            beta_i = beta[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
            a_i = a[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

            s = a_i * s
            # delta rule: S += beta * (v - S k) k^T, an outer product
            sk = torch.matmul(s, k_i.unsqueeze(-1)).squeeze(-1)  # [B, H, D]
            delta = (v_i - sk).unsqueeze(-1)  # [B, H, D, 1]
            s = s + beta_i * delta * k_i.unsqueeze(-2)
            out[:, i] = torch.matmul(q_i.unsqueeze(-2), s).squeeze(-2)
        return out
