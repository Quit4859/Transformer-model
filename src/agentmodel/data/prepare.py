"""Read a jsonl corpus, filter it, tokenize, pack and write binary shards."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import torch

from .dedup import MinHashDeduplicator
from .packing import ShardManifest, pack_documents, write_manifest
from .quality import benchmark_decontaminate
from .tokenizer import CodeAwareBPETokenizer


@dataclass
class PrepareStats:
    documents_in: int = 0
    dropped_dup: int = 0
    dropped_contam: int = 0
    kept: int = 0
    tokens: int = 0
    rows: int = 0


def read_texts(path: str, text_field: str = "text") -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            value = record.get(text_field)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def load_eval_texts(paths: list[str]) -> list[str]:
    """Read evaluation reference files, one document per line."""
    texts: list[str] = []
    for path in paths:
        with open(path) as f:
            texts.extend(line.strip() for line in f if line.strip())
    return texts


def prepare(
    corpus: str,
    out_dir: str,
    seq_len: int,
    vocab_size: int = 8192,
    dedup_threshold: float = 0.8,
    eval_files: list[str] | None = None,
    ngram: int = 13,
    text_field: str = "text",
    save_tokenizer: bool = True,
) -> PrepareStats:
    stats = PrepareStats()
    texts = read_texts(corpus, text_field)
    stats.documents_in = len(texts)
    if not texts:
        raise ValueError(f"no usable documents in {corpus}")

    dedup = MinHashDeduplicator(threshold=dedup_threshold)
    deduped = dedup.deduplicate(texts)
    stats.dropped_dup = len(deduped.duplicate_indices)
    docs = deduped.documents

    if eval_files:
        eval_texts = load_eval_texts(eval_files)
        filtered = benchmark_decontaminate(docs, eval_texts, ngram_size=ngram)
        stats.dropped_contam = len(filtered.contaminated_indices)
        docs = filtered.documents

    stats.kept = len(docs)
    if not docs:
        raise ValueError("every document was filtered out; check the thresholds")

    if save_tokenizer or not os.path.exists(os.path.join(out_dir, "tokenizer.json")):
        tokenizer = CodeAwareBPETokenizer(vocab_size=vocab_size).train(docs)
        os.makedirs(out_dir, exist_ok=True)
        tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    else:
        tokenizer = CodeAwareBPETokenizer.load(os.path.join(out_dir, "tokenizer.json"))

    encoded = [tokenizer.encode(doc) + [tokenizer.special_tokens["<eos>"]] for doc in docs]
    batch = pack_documents(encoded, seq_len=seq_len, pad_token_id=tokenizer.special_tokens["<pad>"])
    stats.tokens = batch.num_real_tokens
    stats.rows = int(batch.input_ids.shape[0])

    os.makedirs(out_dir, exist_ok=True)
    shard_path = os.path.join(out_dir, "train.bin")
    batch.input_ids.to(torch.int16).numpy().tofile(shard_path)
    batch.targets.to(torch.int16).numpy().tofile(shard_path + ".targets")
    batch.doc_ids.to(torch.int16).numpy().tofile(shard_path + ".docs")

    manifest = ShardManifest(
        path=shard_path,
        weight=1.0,
        tokens=stats.tokens,
        meta={
            "documents_in": stats.documents_in,
            "dropped_dup": stats.dropped_dup,
            "dropped_contam": stats.dropped_contam,
            "rows": stats.rows,
            "seq_len": seq_len,
            "dedup_threshold": dedup_threshold,
        },
    )
    write_manifest(os.path.join(out_dir, "manifest.jsonl"), [asdict(manifest)])
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="prepare a training corpus")
    p.add_argument("corpus", help="jsonl file, one document per line")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--dedup-threshold", type=float, default=0.8)
    p.add_argument("--eval-files", nargs="*", default=None)
    p.add_argument("--ngram", type=int, default=13)
    p.add_argument("--text-field", default="text")
    args = p.parse_args()
    stats = prepare(
        args.corpus,
        args.out,
        args.seq_len,
        vocab_size=args.vocab_size,
        dedup_threshold=args.dedup_threshold,
        eval_files=args.eval_files,
        ngram=args.ngram,
        text_field=args.text_field,
    )
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()