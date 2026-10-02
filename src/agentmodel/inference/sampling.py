"""Inference-time sampling and generation for the agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..model.attention import KVCache
from ..model.transformer import Transformer


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 disables
    max_new_tokens: int = 256
    stop: tuple[str, ...] = ()


def filter_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Apply temperature, top-k and nucleus filtering to `[vocab]` logits."""
    out = logits
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive; use greedy decoding at 1e-6")
    if cfg.temperature != 1.0:
        out = out / cfg.temperature
    if cfg.top_k and cfg.top_k > 0:
        k = min(cfg.top_k, out.shape[-1])
        threshold = torch.topk(out, k).values[-1]
        out = out.masked_fill(out < threshold, float("-inf"))
    if 0.0 < cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(out, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) > cfg.top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        out = torch.full_like(out, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return out


def sample_token(logits: torch.Tensor, cfg: SamplingConfig, generator=None) -> int:
    probs = torch.softmax(filter_logits(logits, cfg), dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


@dataclass
class Generator:
    """Batched sampling with a per-stream KV cache and stop sequences."""

    model: Transformer
    cfg: SamplingConfig = field(default_factory=SamplingConfig)
    tokenizer: object | None = None
    cache: KVCache | None = None

    def reset(self) -> KVCache:
        self.cache = KVCache()
        return self.cache

    @torch.no_grad()
    def generate(self, prompt: str, **overrides) -> str:
        cfg = SamplingConfig(**{**self.cfg.__dict__, **overrides})
        if self.tokenizer is None:
            raise ValueError("a tokenizer is required to decode generated ids")
        ids = self.tokenizer.encode(prompt)
        cache = self.reset()
        out = self.model(torch.tensor([ids]), cache=cache, use_cache=True)
        generated: list[int] = []
        for _ in range(cfg.max_new_tokens):
            token = sample_token(out.logits[0, -1], cfg)
            generated.append(token)
            if any(s and _ends_with(self.tokenizer, generated, s) for s in cfg.stop):
                break
            step = self.model(torch.tensor([[token]]), cache=cache, use_cache=True)
            out = step
        return self.tokenizer.decode(generated)


def _ends_with(tokenizer, ids: list[int], stop: str) -> bool:
    stop_ids = tokenizer.encode(stop)
    if not stop_ids or len(stop_ids) > len(ids):
        return False
    return ids[-len(stop_ids) :] == stop_ids