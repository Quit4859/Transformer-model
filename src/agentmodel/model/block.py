"""Decoder block composition.

Residual + attention-then-FFN skeleton is unchanged from the 2017 transformer;
every component inside it is swapped for a modern equivalent.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .attention import GatedAttention, KVCache
from .config import ModelConfig
from .ffn import MoEFFN, SwiGLU
from .linear_attn import GatedDeltaNet
from .norm import RMSNorm


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.is_moe = cfg.n_expert > 0
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_ffn = RMSNorm(cfg.d_model, cfg.norm_eps)

        if layer_type == "full":
            self.attn: nn.Module = GatedAttention(
                cfg, layer_idx, sliding_window=None
            )
        elif layer_type == "swa":
            self.attn = GatedAttention(cfg, layer_idx, sliding_window=cfg.local_window)
        elif layer_type == "delta":
            self.attn = GatedDeltaNet(cfg, layer_idx)
        else:
            raise ValueError(f"unknown layer_type {layer_type}")

        if self.is_moe:
            self.ffn: nn.Module = MoEFFN(cfg)
        else:
            self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor | None = None,
        cache: KVCache | None = None,
        mask: torch.Tensor | None = None,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        aux: torch.Tensor | None = None

        def attn_part(h: torch.Tensor) -> torch.Tensor:
            if self.layer_type == "delta":
                return self.attn(h, cache=cache, use_cache=use_cache)
            return self.attn(h, positions=positions, cache=cache, mask=mask, use_cache=use_cache)

        def ffn_part(h: torch.Tensor) -> Any:
            out = self.ffn(h)
            if self.is_moe:
                return out
            return out, None

        if gradient_checkpointing and self.training:
            a = cp.checkpoint(attn_part, x, use_reentrant=False)
        else:
            a = attn_part(x)
        x = x + a

        if gradient_checkpointing and self.training:
            f_out = cp.checkpoint(ffn_part, x, use_reentrant=False)
        else:
            f_out = ffn_part(x)
        if isinstance(f_out, tuple):
            f_out, aux = f_out
        x = x + f_out
        return x, aux
