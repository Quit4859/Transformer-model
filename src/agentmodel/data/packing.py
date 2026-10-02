"""Deterministic data plumbing shared by pretraining, SFT and RL.

Packing concatenates multiple documents up to a target length so every token is
trained on, and records a per-position document id so losses can be masked at
boundaries and so cross-document attention can be blocked later.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch


@dataclass
class PackedBatch:
    input_ids: torch.Tensor  # [B, T]
    targets: torch.Tensor  # [B, T] already shifted, pad = ignore_index
    doc_ids: torch.Tensor  # [B, T], -1 marks padding
    attention_mask: torch.Tensor  # [B, T] bool, True on real tokens
    num_real_tokens: int

    def to(self, device) -> "PackedBatch":
        return PackedBatch(
            input_ids=self.input_ids.to(device),
            targets=self.targets.to(device),
            doc_ids=self.doc_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            num_real_tokens=self.num_real_tokens,
        )


def pack_documents(
    docs: list[list[int]],
    seq_len: int,
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> PackedBatch:
    """Concatenate `docs` into fixed-length `seq_len` rows.

    Each document is separated by an EOS token so the model learns to stop.
    Document ids restart at each boundary; padding is doc id -1.
    """
    stream: list[int] = []
    doc_stream: list[int] = []
    for doc_id, doc in enumerate(docs):
        if len(doc) == 0:
            continue
        stream.extend(doc)
        doc_stream.extend([doc_id] * len(doc))
    if not stream:
        empty = torch.zeros((1, seq_len), dtype=torch.long)
        return PackedBatch(
            input_ids=empty,
            targets=empty.clone(),
            doc_ids=torch.full((1, seq_len), -1, dtype=torch.long),
            attention_mask=torch.zeros((1, seq_len), dtype=torch.bool),
            num_real_tokens=0,
        )

    num_rows = (len(stream) + seq_len - 1) // seq_len
    pad = num_rows * seq_len - len(stream)
    stream.extend([pad_token_id] * pad)
    doc_stream.extend([-1] * pad)

    ids = torch.tensor(stream, dtype=torch.long).view(num_rows, seq_len)
    doc_ids = torch.tensor(doc_stream, dtype=torch.long).view(num_rows, seq_len)
    attention_mask = doc_ids != -1

    targets = torch.full_like(ids, ignore_index)
    valid = attention_mask.clone()
    valid[:, 1:] = attention_mask[:, 1:] & attention_mask[:, :-1]
    targets[:, :-1] = torch.where(valid[:, :-1], ids[:, 1:], targets[:, :-1])
    targets[:, -1] = ignore_index

    return PackedBatch(
        input_ids=ids,
        targets=targets,
        doc_ids=doc_ids,
        attention_mask=attention_mask,
        num_real_tokens=int(attention_mask.sum().item()),
    )


def iter_jsonl(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_manifest(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@dataclass
class ShardManifest:
    path: str
    weight: float = 1.0
    tokens: int = 0
    dedup_hash: str = ""
    meta: dict = field(default_factory=dict)
