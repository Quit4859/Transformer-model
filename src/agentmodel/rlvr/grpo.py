"""GRPO: critic-free policy optimization from grouped rollout rewards.

Advantages are estimated by comparing rollouts of the same prompt against each
other, so no value network is trained and none of its memory or instability is
inherited. The cost is that a prompt whose rollouts all score the same produces
zero signal, which `dynamic_sampling` filters out.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rewards import reward_spread


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize rewards within each leading group.

    Groups with zero variance normalize to zero, which is the correct answer:
    nothing distinguishes those rollouts.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [groups, samples]")
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def clipped_policy_loss(
    log_ratio: torch.Tensor, advantages: torch.Tensor, clip_ratio: float = 0.2
) -> torch.Tensor:
    """PPO-style clipped surrogate, asymmetric in DAPO's favor of exploration."""
    return masked_mean(clipped_policy_terms(log_ratio, advantages, clip_ratio), torch.ones_like(log_ratio))


def clipped_policy_terms(
    log_ratio: torch.Tensor, advantages: torch.Tensor, clip_ratio: float = 0.2
) -> torch.Tensor:
    """Per-token clipped surrogate, unmasked, so callers can weight tokens."""
    if log_ratio.shape != advantages.shape:
        raise ValueError("log_ratio and advantages must have equal shapes")
    if clip_ratio <= 0:
        raise ValueError("clip_ratio must be positive")
    ratio = log_ratio.exp()
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    return -torch.minimum(ratio * advantages, clipped * advantages)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `values` over positions where `mask` is set."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have equal shapes")
    total = mask.sum()
    if total == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / total


@dataclass
class GroupBatch:
    """One prompt sampled several times, with per-token behaviour and rewards."""

    log_probs: torch.Tensor  # [samples, tokens] under the training policy
    old_log_probs: torch.Tensor  # [samples, tokens] under the rollout policy
    rewards: torch.Tensor  # [samples]
    mask: torch.Tensor  # [samples, tokens], 1 on generated tokens

    def log_ratio(self) -> torch.Tensor:
        return self.log_probs - self.old_log_probs

    def advantages(self) -> torch.Tensor:
        # A single prompt's rollouts form one group; normalizing across them is
        # what makes the advantage critic-free.
        normalized = group_advantages(self.rewards.unsqueeze(0))
        return normalized.squeeze(0).unsqueeze(-1).expand_as(self.log_probs)


def grpo_loss(batch: GroupBatch, clip_ratio: float = 0.2) -> torch.Tensor:
    """Token-mean clipped surrogate over every generated token in the group."""
    return masked_mean(
        clipped_policy_terms(batch.log_ratio(), batch.advantages(), clip_ratio), batch.mask
    )


def dynamic_sampling(groups: list[list[float]], min_spread: float = 1e-6) -> tuple[list[int], list[float]]:
    """Drop prompts whose rollout rewards carry no signal.

    Returns the surviving prompt indices and the reward spread of each survivor,
    which is logged as the curriculum signal: a prompt trending toward zero
    spread is one the model has saturated and is ready to be replaced.
    """
    kept: list[int] = []
    spreads: list[float] = []
    for index, rewards in enumerate(groups):
        spread = reward_spread(rewards)
        if spread > min_spread:
            kept.append(index)
            spreads.append(spread)
    return kept, spreads