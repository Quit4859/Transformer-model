"""Pretraining entrypoint: pack a corpus, run steps, log, checkpoint.

The overfit mode is the Phase-0 acceptance gate: a tiny model driven to near
zero loss on a single tiny batch means the whole stack is wired correctly.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ..config import Config, load_config
from ..data.packing import PackedBatch, pack_documents
from ..model.transformer import Transformer
from .loop import Trainer


def synthetic_docs(n_docs: int, doc_len: int, vocab: int, eos: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    docs = []
    for _ in range(n_docs):
        # small-vocab alphabet keeps the task learnable on CPU
        toks = torch.randint(3, max(4, vocab), (doc_len,), generator=g).tolist()
        docs.append(toks + [eos])
    return docs


def build_batch(cfg: Config, seed: int = 0, seq_len: int | None = None) -> PackedBatch:
    seq_len = seq_len or cfg.data.sequence_length
    docs = synthetic_docs(
        n_docs=8,
        doc_len=max(8, seq_len // 4),
        vocab=cfg.model.vocab_size,
        eos=cfg.data.eos_token_id,
        seed=seed,
    )
    return pack_documents(
        docs,
        seq_len,
        pad_token_id=cfg.data.pad_token_id,
    )


def run(
    cfg: Config,
    steps: int,
    out_dir: str,
    log_every: int = 10,
    seq_len: int | None = None,
) -> list[dict]:
    """Train on a synthetic packed batch and log the loss curve.

    `seq_len` shortens the sequence, which keeps the Phase-0 overfit gate fast
    on CPU while still exercising every layer, the mask path, both optimizers
    and the scheduler.
    """
    seq_len = seq_len or cfg.data.sequence_length
    os.makedirs(out_dir, exist_ok=True)
    model = Transformer(cfg.model)
    trainer = Trainer(cfg, model=model)
    batch = build_batch(cfg, seq_len=seq_len)
    print(
        f"params={trainer.model.num_params():,} config_hash={cfg.config_hash()} "
        f"seq_len={seq_len} layers={cfg.model.layer_types()}"
    )
    history = []
    for i in range(steps):
        stats = trainer.micro_step(batch)
        trainer.advance()
        if i % log_every == 0 or i == steps - 1:
            rec = stats.as_dict()
            history.append(rec)
            gnorm = "  n/a " if rec["grad_norm"] is None else f"{rec['grad_norm']:.3f}"
            print(
                f"step {rec['step']:5d} loss {rec['loss']:.4f} "
                f"lr {rec['lr_scale']:.3f} gnorm {gnorm}"
            )
    with open(os.path.join(out_dir, "history.jsonl"), "w") as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="pretrain a small model")
    p.add_argument("config", help="path to a yaml config")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--out", default="runs/pretrain")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    run(cfg, args.steps, args.out, args.log_every, args.seq_len)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
