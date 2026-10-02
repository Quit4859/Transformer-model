"""Command-line entrypoint for model, training, and data utilities."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .model.transformer import Transformer


def _info(config_path: str) -> int:
    cfg = load_config(config_path)
    model = Transformer(cfg.model)
    print(f"params: {model.num_params():,}")
    print(f"config_hash: {cfg.config_hash()}")
    return 0


def _pretrain(args: argparse.Namespace) -> int:
    from .train.pretrain import load_tokenized_documents, run

    cfg = load_config(args.config)
    documents = load_tokenized_documents(args.data) if args.data else None
    run(
        cfg,
        args.steps,
        args.out,
        log_every=args.log_every,
        seq_len=args.seq_len,
        documents=documents,
        resume=args.resume,
        heldout_fraction=args.heldout_fraction,
        min_improvement=args.min_improvement,
    )
    return 0


def _sft(args: argparse.Namespace) -> int:
    from .data.tokenizer import CodeAwareBPETokenizer
    from .train.sft import encode_traces, load_passing_traces

    tokenizer = CodeAwareBPETokenizer.load(args.tokenizer)
    records = encode_traces(load_passing_traces(args.traces), tokenizer)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records to {args.out}")
    return 0


def _check(args: argparse.Namespace) -> int:
    """Validate a config and optionally run the repository test suite."""
    cfg = load_config(args.config)
    model = Transformer(cfg.model)
    print(f"config: {args.config}")
    print(f"config_hash: {cfg.config_hash()}")
    print(f"params: {model.num_params():,}")
    if args.skip_tests:
        return 0
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], check=False)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentmodel")
    subparsers = parser.add_subparsers(dest="command")

    info = subparsers.add_parser("info", help="show model size and configuration hash")
    info.add_argument("config", nargs="?", default="configs/pretrain_nano.yaml")
    info.set_defaults(handler=_info)

    pretrain = subparsers.add_parser("pretrain", help="run synthetic or tokenized-JSONL training")
    pretrain.add_argument("config")
    pretrain.add_argument("--steps", type=int, default=200)
    pretrain.add_argument("--out", default="runs/pretrain")
    pretrain.add_argument("--log-every", type=int, default=10)
    pretrain.add_argument("--seq-len", type=int)
    pretrain.add_argument("--data")
    pretrain.add_argument("--resume")
    pretrain.add_argument("--heldout-fraction", type=float, default=0.1)
    pretrain.add_argument("--min-improvement", type=float, default=0.05)
    pretrain.set_defaults(handler=_pretrain)

    sft = subparsers.add_parser("sft-encode", help="encode passing traces for SFT")
    sft.add_argument("tokenizer")
    sft.add_argument("traces")
    sft.add_argument("--out", default="runs/sft/train.jsonl")
    sft.set_defaults(handler=_sft)

    check = subparsers.add_parser("check", help="validate configuration and run tests")
    check.add_argument("--config", default="configs/pretrain_nano.yaml")
    check.add_argument("--skip-tests", action="store_true")
    check.set_defaults(handler=_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.command:
        print("agentmodel CLI", file=sys.stderr)
        return _info("configs/pretrain_nano.yaml")
    return args.handler(args.config) if args.command == "info" else args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
