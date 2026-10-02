"""Pretraining entrypoint: pack a corpus, run steps, log, checkpoint.

The overfit mode is the Phase-0 acceptance gate: a tiny model driven to near
zero loss on a single tiny batch means the whole stack is wired correctly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace

import torch

from ..config import Config, load_config
from ..data.packing import PackedBatch, pack_documents
from ..model.transformer import Transformer
from .checkpoint import load_checkpoint, save_checkpoint
from .eval import evaluate
from .loop import Trainer


def synthetic_docs(n_docs: int, doc_len: int, vocab: int, eos: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    docs = []
    for _ in range(n_docs):
        # small-vocab alphabet keeps the task learnable on CPU
        toks = torch.randint(3, max(4, vocab), (doc_len,), generator=g).tolist()
        docs.append(toks + [eos])
    return docs


def load_tokenized_documents(path: str) -> list[list[int]]:
    """Load token-id documents from JSONL records containing ``tokens``."""
    documents: list[list[int]] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            tokens = record.get("tokens", record.get("input_ids")) if isinstance(record, dict) else record
            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                raise ValueError(f"line {line_number} of {path} must contain an integer token list")
            if tokens:
                documents.append(tokens)
    if not documents:
        raise ValueError(f"{path} contains no tokenized documents")
    return documents


def build_batch(
    cfg: Config,
    seed: int = 0,
    seq_len: int | None = None,
    documents: list[list[int]] | None = None,
) -> PackedBatch:
    seq_len = seq_len or cfg.data.sequence_length
    docs = documents or synthetic_docs(
        n_docs=8, doc_len=max(8, seq_len // 4), vocab=cfg.model.vocab_size,
        eos=cfg.data.eos_token_id, seed=seed
    )
    return pack_documents(
        docs,
        seq_len,
        pad_token_id=cfg.data.pad_token_id,
    )


def run_overfit_gate(
    cfg: Config,
    documents: list[list[int]],
    *,
    steps: int = 100,
    heldout_fraction: float = 0.1,
    learning_rate: float = 1.0,
) -> dict[str, float]:
    """Run the nano acceptance gate on a fixed shard and return measured metrics."""
    if not documents:
        raise ValueError("documents must not be empty")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between 0 and 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    split = max(1, min(len(documents) - 1, round(len(documents) * (1 - heldout_fraction))))
    train_docs, heldout_docs = documents[:split], documents[split:]
    gate_cfg = replace(
        cfg,
        precision=replace(cfg.precision, compute_dtype="fp32"),
        optim=replace(
            cfg.optim,
            lr=learning_rate,
            schedule="constant",
            warmup_steps=1,
        ),
        train=replace(cfg.train, grad_accum_steps=1),
    )
    trainer = Trainer(gate_cfg)
    train_batch = pack_documents(
        train_docs,
        gate_cfg.data.sequence_length,
        pad_token_id=gate_cfg.data.pad_token_id,
    )
    heldout_batch = pack_documents(
        heldout_docs,
        gate_cfg.data.sequence_length,
        pad_token_id=gate_cfg.data.pad_token_id,
    )
    first_loss = float(trainer.forward_loss(train_batch)[0].detach())
    finite_gradients = True
    for _ in range(steps):
        stats = trainer.micro_step(train_batch)
        trainer.advance()
        finite_gradients &= stats.grad_norm is None or math.isfinite(stats.grad_norm)
    with torch.no_grad():
        heldout_loss = float(trainer.forward_loss(heldout_batch)[0].detach())
    if not finite_gradients or not torch.isfinite(torch.tensor(heldout_loss)):
        raise RuntimeError("nano overfit gate produced non-finite gradients or loss")
    return {
        "first_loss": first_loss,
        "heldout_loss": heldout_loss,
        "finite_gradients": float(finite_gradients),
    }


def split_documents(
    documents: list[list[int]], heldout_fraction: float = 0.1, seed: int = 0
) -> tuple[list[list[int]], list[list[int]]]:
    """Partition documents into train and heldout sets without overlap.

    The heldout set must be disjoint from training data for a validation loss to
    mean anything, so it is drawn as whole documents rather than token slices.
    """
    if not documents:
        raise ValueError("documents must not be empty")
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be between 0 and 1")
    order = torch.randperm(len(documents), generator=torch.Generator().manual_seed(seed))
    heldout_count = min(len(documents) - 1, max(1, round(len(documents) * heldout_fraction)))
    heldout_idx = set(order[:heldout_count].tolist())
    train = [doc for i, doc in enumerate(documents) if i not in heldout_idx]
    heldout = [doc for i, doc in enumerate(documents) if i in heldout_idx]
    if not train or not heldout:
        raise ValueError("split produced an empty train or heldout set")
    return train, heldout


def run(
    cfg: Config,
    steps: int,
    out_dir: str,
    log_every: int = 10,
    seq_len: int | None = None,
    documents: list[list[int]] | None = None,
    resume: str | None = None,
    heldout_fraction: float = 0.1,
    min_improvement: float = 0.05,
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
    if resume:
        metadata = load_checkpoint(resume, trainer)
        print(
            f"resumed checkpoint={resume} step={metadata['step_index']} "
            f"tokens={metadata['tokens_seen']}"
        )
    batch = build_batch(cfg, seq_len=seq_len, documents=documents)
    print(
        f"params={trainer.model.num_params():,} config_hash={cfg.config_hash()} "
        f"seq_len={seq_len} layers={cfg.model.layer_types()}"
    )
    # Validation documents are disjoint from the training batch, so the reported
    # loss reflects generalization rather than memorization of the trained tokens.
    heldout_batch = None
    if documents:
        _, heldout_docs = split_documents(documents, heldout_fraction)
        heldout_batch = build_batch(
            cfg, seed=1234, seq_len=seq_len, documents=heldout_docs
        )
        baseline = evaluate(trainer, [heldout_batch])
        print(
            f"heldout baseline loss {baseline['loss']:.4f} "
            f"ppl {baseline['perplexity']:.2f} (random init)"
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
    history_path = os.path.join(out_dir, "history.jsonl")
    mode = "a" if resume and os.path.exists(history_path) else "w"
    with open(history_path, mode) as f:
        for rec in history:
            f.write(json.dumps(rec) + "\n")
    save_checkpoint(os.path.join(out_dir, "checkpoint.pt"), trainer)
    eval_metrics = evaluate(trainer, [heldout_batch] if heldout_batch else [batch])
    if heldout_batch is not None:
        # The acceptance gate: a model that cannot beat its own random-init loss
        # on unseen documents has not learned anything transferable. A relative
        # margin is required because a handful of steps lowers loss by less than
        # run-to-run noise, which would make the gate pass without learning.
        improvement = (baseline["loss"] - eval_metrics["loss"]) / baseline["loss"]
        eval_metrics["baseline_loss"] = baseline["loss"]
        eval_metrics["relative_improvement"] = improvement
        eval_metrics["beats_random_init"] = float(improvement >= min_improvement)
        print(
            f"heldout loss {eval_metrics['loss']:.4f} "
            f"ppl {eval_metrics['perplexity']:.2f} "
            f"baseline {baseline['loss']:.4f} "
            f"improvement {improvement:+.2%} "
            f"gate={'PASS' if eval_metrics['beats_random_init'] else 'FAIL'}"
        )
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2, sort_keys=True)
    return history


def main() -> None:
    p = argparse.ArgumentParser(description="pretrain a small model")
    p.add_argument("config", help="path to a yaml config")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--out", default="runs/pretrain")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--data", default=None, help="tokenized JSONL with tokens or input_ids fields")
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.1,
        help="fraction of documents reserved for validation loss",
    )
    p.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="relative heldout loss reduction required to pass the gate",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    documents = load_tokenized_documents(args.data) if args.data else None
    run(
        cfg,
        args.steps,
        args.out,
        args.log_every,
        args.seq_len,
        documents=documents,
        resume=args.resume,
        heldout_fraction=args.heldout_fraction,
        min_improvement=args.min_improvement,
    )
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
