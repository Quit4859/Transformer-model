"""Losses: token cross-entropy, MoE aux loss, and MTP loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Next-token CE. `logits` [B, T, V]; `targets` [B, T] already shifted."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def count_valid(targets: torch.Tensor, ignore_index: int = -100) -> int:
    return int((targets != ignore_index).sum().item())


PAD_DOC_ID = -1


def doc_boundary_mask(sequence_ids: torch.Tensor) -> torch.Tensor:
    """Positions where a new packed document begins, excluding padding."""
    if sequence_ids.dim() != 2:
        raise ValueError("sequence_ids must be [B, T]")
    is_start = torch.zeros_like(sequence_ids, dtype=torch.bool)
    is_start[:, 0] = sequence_ids[:, 0] != PAD_DOC_ID
    if sequence_ids.shape[1] > 1:
        is_start[:, 1:] = sequence_ids[:, 1:] != sequence_ids[:, :-1]
    return is_start & (sequence_ids != PAD_DOC_ID)


def padding_mask(sequence_ids: torch.Tensor) -> torch.Tensor:
    """Positions that are padding rather than real document tokens."""
    return sequence_ids == PAD_DOC_ID


def mtp_loss(
    main_logits: torch.Tensor,
    extra_logits: torch.Tensor,
    targets: torch.Tensor,
    shift: int = 1,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Multi-token prediction loss over `shift+1`, `shift+2`, ... heads.

    extra_logits: list of [B, T, V] heads, head j predicts token t+j.
    """
    if not extra_logits:
        raise ValueError("extra_logits must be non-empty")
    losses = []
    for j, head in enumerate(extra_logits):
        step = shift + j
        if step >= targets.shape[1]:
            break
        losses.append(
            token_cross_entropy(
                head[:, :-step, :], targets[:, step:], ignore_index=ignore_index
            )
        )
    if not losses:
        raise ValueError("no valid MTP shifts for the given targets")
    return torch.stack(losses).mean()


def entropy_of(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1).mean()
