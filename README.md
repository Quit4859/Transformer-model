# AgentModel

Decoder-only Transformer training project.

## Quick start

```bash
python -m pytest -q
python -m agentmodel.train.pretrain configs/pretrain_nano.yaml --steps 200
agentmodel info configs/pretrain_nano.yaml
```

Pretraining writes `history.jsonl`, `eval.json`, and `checkpoint.pt` to the
output directory. To train from tokenized JSONL, provide records containing a
`tokens` or `input_ids` integer list:

```bash
python -m agentmodel.train.pretrain configs/pretrain_nano.yaml \
  --data data/train.jsonl --out runs/pretrain
```

The unified CLI also exposes the same workflows:

```bash
agentmodel pretrain configs/pretrain_nano.yaml --steps 200
agentmodel sft-encode tokenizer.json traces.jsonl --out runs/sft/train.jsonl
```

The current implementation includes the model core, training loop, tokenizer,
deduplication and data-quality pipeline, checkpointing, and evaluation
utilities. Larger SFT, agent-harness, RLVR, and distributed-scaling phases
remain sequenced in `plans/phases.md`.
