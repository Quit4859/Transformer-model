"""Feed-forward blocks: dense SwiGLU and DeepSeekMoE-style sparse MoE.

SwiGLU keeps three matrices instead of two, so d_ff is scaled to ~8/3 d_model to
hold the parameter count constant against a plain GELU MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, d_ff = cfg.d_model, cfg.d_ff
        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoERouter(nn.Module):
    """Aux-loss-free load balancing: bias is nudged toward experts chosen less."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.n_expert, cfg.d_model))
        nn.init.normal_(self.weight, std=0.02)
        self.register_buffer("bias", torch.zeros(cfg.n_expert), persistent=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        logits = F.linear(flat.float(), self.weight.float(), self.bias.float())
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class MoEFFN(nn.Module):
    """Top-k routed experts plus always-on shared experts.

    Returns the auxiliary load-balancing loss alongside the output so the
    training loop can add it without needing a second forward pass.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = MoERouter(cfg)
        self.experts = nn.ModuleList([SwiGLU(cfg) for _ in range(cfg.n_expert)])
        self.shared_experts = nn.ModuleList(
            [SwiGLU(cfg) for _ in range(cfg.n_shared_expert)]
        )
        self.n_expert_used = cfg.n_expert_used

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        probs, logits = self.router(x)
        topk_w, topk_i = torch.topk(probs, self.n_expert_used, dim=-1)  # [B*T, k]
        out = torch.zeros_like(x.reshape(-1, d))

        flat_x = x.reshape(-1, d)
        expert_mask = F.one_hot(topk_i, num_classes=self.cfg.n_expert).float()  # [N, k, E]
        hits = expert_mask.sum(dim=(0, 1))  # [E]

        for e, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(topk_i == e)
            if token_idx.numel() == 0:
                continue
            w = topk_w[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, expert(flat_x[token_idx]) * w.to(x.dtype))

        for shared in self.shared_experts:
            out = out + shared(flat_x)

        aux = self._aux_loss(logits, topk_i, hits, self.n_expert_used)
        return out.view(b, t, d), aux

    def _aux_loss(self, logits: torch.Tensor, topk_i: torch.Tensor, hits: torch.Tensor, k: int) -> torch.Tensor:
        n_tokens = topk_i.shape[0]
        n_expert = self.cfg.n_expert
        frac_tokens = hits / max(1, n_tokens * k)
        mean_router = F.softmax(logits, dim=-1).mean(dim=0)
        loss = n_expert * torch.sum(frac_tokens * mean_router)
        return self.cfg.moe_aux_loss_coeff * loss

    @torch.no_grad()
    def load_balance_bias_step(self, scores: torch.Tensor) -> None:
        """Nudge router bias so under-used experts get sampled more.

        scores: [E] fraction of tokens routed to each expert over the last step.
        """
        err = self.cfg.n_expert * scores
        self.router.bias -= 1e-3 * err.sign()
