"""Muon: momentum orthogonalized by Newton-Schulz, applied to 2D hidden weights.

AdamW keeps ownership of embeddings, norms, biases and any 1D parameter.
Everything that is a 2D weight matrix on a hidden-hidden path goes to Muon.
The Newton-Schulz iteration approximates the orthogonal polar factor without
forming it, so the cost is a handful of small matmuls per parameter.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonalization Q = U V^T of G = U S V^T."""
    if G.ndim < 2:
        raise ValueError("newtonschulz requires a >=2D tensor")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = G.to(torch.float32)
    transposed = G.size(0) > G.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        aa = x @ x.T  # [m, m]
        b_mat = b * aa + c * (aa @ aa)  # [m, m]
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Single-GPU Muon. Distributed variants shard the Newton-Schulz matmul."""

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        ns_steps: int = 5,
        mu_dtype: torch.dtype | None = None,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            mu_dtype=mu_dtype,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - mom)
                update = g.lerp_(buf, mom) if nesterov else buf
                if wd:
                    p.mul_(1.0 - lr * wd)
                if p.ndim == 2:
                    orig_shape = update.shape
                    m, n = orig_shape
                    side = max(m, n)
                    transposed = n > m
                    u = update.T if transposed else update
                    u = zeropower_via_newtonschulz5(u, steps=group["ns_steps"])
                    u = u.T if transposed else u
                    update = u.view(orig_shape)
                p.add_(update.reshape(p.shape), alpha=-lr)
        return loss
