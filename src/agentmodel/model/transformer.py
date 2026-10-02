"""Full decoder-only transformer stack."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import KVCache
from .block import Block
from .config import ModelConfig
from .norm import RMSNorm


@dataclass
class ModelOutput:
    logits: torch.Tensor
    aux_loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    cache: KVCache | None = None


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_types = cfg.layer_types()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg, i, lt) for i, lt in enumerate(self.layer_types)]
        )
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        if cfg.logit_scale:
            self.logit_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("logit_scale", torch.tensor(1.0), persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 1.0 / math.sqrt(module.in_features) if module.in_features > 0 else 0.02
            if module.out_features >= 2 * module.in_features:
                std = 0.02 / math.sqrt(2 * module.in_features)
            nn.init.normal_(module.weight, mean=0.0, std=min(std, 0.05))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        positions: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        gradient_checkpointing: bool = False,
        last_logit_only: bool = False,
    ) -> ModelOutput:
        b, t = input_ids.shape
        x = self.embed(input_ids)
        if positions is None:
            offset = 0
            if cache is not None and cache.layers:
                offset = cache.layers[0].k.shape[2]
            positions = torch.arange(offset, offset + t, device=input_ids.device).unsqueeze(0)
            positions = positions.expand(b, t)

        aux_total: torch.Tensor | None = None
        for block in self.blocks:
            x, aux = block(
                x,
                positions=positions,
                cache=cache,
                use_cache=use_cache,
                gradient_checkpointing=gradient_checkpointing,
            )
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux

        if last_logit_only:
            x = x[:, -1:, :]
        x = self.norm_out(x)
        logits = self.lm_head(x)
        if self.cfg.logit_scale:
            logits = logits * torch.exp(self.logit_scale)
        return ModelOutput(logits=logits, aux_loss=aux_total, cache=cache)

    def num_params(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed.weight.numel()
        return total

    def new_cache(self) -> KVCache:
        return KVCache()


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Shifted-token CE. `logits` is [B, T, V]; `targets` is already shifted."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )
