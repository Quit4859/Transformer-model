"""Supervised fine-tuning on chat and tool-use traces.

Loss is masked to assistant tokens only. Training on tool *results* teaches the
model to hallucinate outputs it never observed, so those positions are ignored.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from ..config import Config, load_config
from ..data.chat import ChatTemplate, messages_from_dicts
from ..data.tokenizer import CodeAwareBPETokenizer
from ..data.traces import filter_passing_traces
from .loop import Trainer


def load_passing_traces(path: str | Path) -> list[dict]:
    traces = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}") from exc
            traces.append(trace)
    return filter_passing_traces(traces)


def encode_traces(
    traces: list[dict], tokenizer: CodeAwareBPETokenizer, template: ChatTemplate | None = None
) -> list[dict[str, list[int]]]:
    template = template or ChatTemplate()
    return [
        {
            "input_ids": encoded.input_ids,
            "targets": [
                token if mask else -100
                for token, mask in zip(
                    encoded.input_ids[1:] + [-100], encoded.loss_mask, strict=True
                )
            ],
        }
        for trace in traces
        for encoded in [template.encode(messages_from_dicts(trace["messages"]), tokenizer)]
    ]


def collate(rows: list[dict[str, list[int]]], seq_len: int, pad_token_id: int = 0):
    """Pad encoded conversations into an input/target pair for a forward pass."""
    if not rows:
        raise ValueError("no rows to collate")
    inputs, targets = [], []
    for row in rows:
        ids = row["input_ids"][:seq_len]
        tgts = row["targets"][:seq_len]
        pad = seq_len - len(ids)
        inputs.append(ids + [pad_token_id] * pad)
        targets.append(tgts + [-100] * pad)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def sft_batches(
    rows: list[dict[str, list[int]]],
    seq_len: int,
    micro_batch_size: int,
    pad_token_id: int = 0,
):
    """Yield microbatches, dropping any batch that supervises no tokens."""
    for start in range(0, len(rows), micro_batch_size):
        chunk = rows[start : start + micro_batch_size]
        inputs, targets = collate(chunk, seq_len, pad_token_id)
        if (targets != -100).sum() == 0:
            continue
        yield inputs, targets


def run_sft(
    cfg: Config,
    rows: list[dict[str, list[int]]],
    steps: int,
    template: ChatTemplate | None = None,
) -> list[dict]:
    """Fine-tune on pre-encoded conversations and return the logged history."""
    if not rows:
        raise ValueError("no SFT rows")
    sft_cfg = replace(cfg, train=replace(cfg.train, grad_accum_steps=1))
    trainer = Trainer(sft_cfg)
    batches = list(
        sft_batches(
            rows,
            sft_cfg.data.sequence_length,
            sft_cfg.train.micro_batch_size,
            sft_cfg.data.pad_token_id,
        )
    )
    if not batches:
        raise ValueError("every conversation had no supervised tokens")
    history = []
    for step in range(steps):
        inputs, targets = batches[step % len(batches)]
        loss = trainer.supervised_loss(inputs, targets)
        loss.backward()
        grad_norm = trainer.optimizer_step()
        trainer.zero_grad()
        trainer.advance()
        rec = {
            "step": step,
            "loss": round(float(loss.detach()), 6),
            "grad_norm": round(grad_norm, 6),
            "supervised_tokens": int((targets != -100).sum()),
        }
        history.append(rec)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="supervised fine-tuning on tool traces")
    parser.add_argument("config")
    parser.add_argument("tokenizer")
    parser.add_argument("traces")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--train", action="store_true", help="run training instead of encoding")
    parser.add_argument("--out", default="runs/sft")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tokenizer = CodeAwareBPETokenizer.load(args.tokenizer)
    traces = load_passing_traces(args.traces)
    if not traces:
        raise SystemExit("no valid traces with passing tests")
    rows = encode_traces(traces, tokenizer)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if not args.train:
        with (output / "train.jsonl").open("w", encoding="utf-8") as stream:
            for record in rows:
                stream.write(json.dumps(record) + "\n")
        print(f"encoded {len(rows)} traces -> {output / 'train.jsonl'}")
        return
    history = run_sft(cfg, rows, args.steps)
    with (output / "history.jsonl").open("w", encoding="utf-8") as stream:
        for record in history:
            stream.write(json.dumps(record) + "\n")
    print(f"trained {len(history)} steps -> {output}")


if __name__ == "__main__":
    main()
