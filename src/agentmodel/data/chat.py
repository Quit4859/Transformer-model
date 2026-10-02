"""Chat formatting and assistant-only supervision for SFT."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from .tokenizer import CodeAwareBPETokenizer


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class EncodedConversation:
    input_ids: list[int]
    loss_mask: list[bool]


class ChatTemplate:
    """A deterministic role-delimited template that preserves message text."""

    roles = frozenset({"system", "user", "assistant", "tool"})

    def __init__(self, *, bos: str = "", eos: str = "") -> None:
        self.bos = bos
        self.eos = eos

    def render(self, messages: Iterable[ChatMessage]) -> str:
        rendered = self.bos
        for message in messages:
            if message.role not in self.roles:
                raise ValueError(f"unsupported chat role: {message.role}")
            rendered += f"<|{message.role}|>\n{message.content}\n"
        return rendered + self.eos

    def encode(
        self, messages: Iterable[ChatMessage], tokenizer: CodeAwareBPETokenizer
    ) -> EncodedConversation:
        ids: list[int] = []
        mask: list[bool] = []
        if self.bos:
            bos_ids = tokenizer.encode(self.bos)
            ids.extend(bos_ids)
            mask.extend([False] * len(bos_ids))
        for message in messages:
            if message.role not in self.roles:
                raise ValueError(f"unsupported chat role: {message.role}")
            role_text = f"<|{message.role}|>\n"
            role_ids = tokenizer.encode(role_text)
            ids.extend(role_ids)
            mask.extend([False] * len(role_ids))
            content_ids = tokenizer.encode(message.content + "\n")
            ids.extend(content_ids)
            mask.extend([message.role == "assistant"] * len(content_ids))
        eos_ids = tokenizer.encode(self.eos) if self.eos else []
        ids.extend(eos_ids)
        mask.extend([False] * len(eos_ids))
        return EncodedConversation(ids, mask)


def assistant_targets(
    encoded: EncodedConversation,
    *,
    pad_to: int | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inputs and next-token targets masked to assistant responses."""
    if len(encoded.input_ids) != len(encoded.loss_mask):
        raise ValueError("input_ids and loss_mask must have equal lengths")
    ids = encoded.input_ids
    targets = [ignore_index] * len(ids)
    for index in range(len(ids) - 1):
        if encoded.loss_mask[index]:
            targets[index] = ids[index + 1]
    if pad_to is not None:
        if pad_to < len(ids):
            raise ValueError("pad_to cannot be shorter than the conversation")
        ids = ids + [0] * (pad_to - len(ids))
        targets = targets + [ignore_index] * (pad_to - len(targets))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def messages_from_dicts(messages: Iterable[Mapping[str, str]]) -> list[ChatMessage]:
    result = [ChatMessage(str(item["role"]), str(item["content"])) for item in messages]
    if not result:
        raise ValueError("conversation must contain at least one message")
    return result
