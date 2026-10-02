# AgentModel

[![tests](https://github.com/Quit4859/Transformer-modle/actions/workflows/tests.yml/badge.svg)](https://github.com/Quit4859/Transformer-modle/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/pytorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ruff](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![code style](https://img.shields.io/badge/code%20style-ruff-4b8b29?logo=ruff&logoColor=white)](https://github.com/astral-sh/ruff)
[![cpu only](https://img.shields.io/badge/runs%20on-CPU%20only-6f42c1?logo=linux&logoColor=white)](https://www.kernel.org)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?logo=github&logoColor=white)](https://makeapullrequest.com)
[![contributors](https://img.shields.io/github/contributors/Quit4859/Transformer-modle?logo=github&logoColor=white)](https://github.com/Quit4859/Transformer-modle/graphs/contributors)
[![last commit](https://img.shields.io/github/last-commit/Quit4859/Transformer-modle?logo=git&logoColor=white)](https://github.com/Quit4859/Transformer-modle/commits/main)

A decoder-only Transformer, trained from scratch in PyTorch, with the training
infrastructure needed to turn it into a tool-using coding agent.

The goal is not a single model file. It is the full path from raw text to an
agent that runs code: tokenizer, data pipeline, optimizer, trainer, inference
loop, SFT on tool traces, and a reinforcement-learning stage that rewards
verified outcomes rather than human labels.

![Six-layer decoder-only transformer. Five sliding-window attention layers
followed by one global attention layer. Each layer contains RMSNorm,
grouped-query attention with eight query heads sharing two key-value heads,
QK-Norm and partial RoPE, and a SwiGLU feed-forward block, with residual
connections bypassing each sub-block.](src/img/architecture.png)

## Status

The core is complete and runs end to end on CPU. Pretraining, validation,
supervised fine-tuning, the agent harness, and GRPO are all implemented and
covered by tests. The known gaps are listed under
[Not yet built](#not-yet-built).

## Quick start

The package installs as editable, so every command below runs from the
repository root with no further setup.

```bash
pip install -e . --no-deps

python3 -m pytest tests/ -q
agentmodel pretrain configs/pretrain_nano.yaml \
  --steps 60 --seq-len 128 --out runs/demo --log-every 10
```

That trains a 1.4M-parameter model for 60 steps and drops loss from 6.93 to
4.31 in roughly 50 seconds. It writes `history.jsonl`, `eval.json`, and
`checkpoint.pt` into the output directory.

If `agentmodel` is not on your PATH, use the module form, which is otherwise
identical:

```bash
python3 -m agentmodel.cli pretrain configs/pretrain_nano.yaml --steps 60 --out runs/demo
```

## Commands

```bash
agentmodel info configs/pretrain_nano.yaml         # parameter count and config hash
agentmodel check --config configs/pretrain_nano.yaml   # validate config and model
agentmodel check --skip-tests                      # validation only, skip the suite
agentmodel pretrain <config> [options]
agentmodel sft-encode tokenizer.json traces.jsonl --out runs/sft.jsonl
```

Every subcommand accepts `--help`.

### Training on your own data

Pass a JSONL file where each line carries an integer list under `tokens` or
`input_ids`:

```bash
agentmodel pretrain configs/pretrain_nano.yaml \
  --steps 150 --seq-len 64 --data corpus.jsonl --out runs/real
```

Supplying real documents enables the validation gate, which is the point of the
exercise. A fraction of whole documents is held out of training, the loss on
those unseen tokens is measured before training starts and again afterwards, and
the run reports whether the model beat its own random initialization:

```
heldout baseline loss 6.9325 ppl 1025.07 (random init)
step    20 loss 6.7913 lr 0.420 gnorm   n/a
step    39 loss 6.3887 lr 0.800 gnorm 0.416
heldout loss 6.3429 ppl 568.46 baseline 6.9325 improvement +8.50% gate=PASS
```

The gate requires a 5% relative reduction by default. That margin is not
decoration: two optimizer steps move heldout loss by 0.04%, which would satisfy
a naive "better than baseline" check while the model had learned nothing.
Raise or lower it with `--min-improvement`, and change the split with
`--heldout-fraction`.

![Training curve showing held-out validation loss falling from 6.93 to about
1.83 over 150 steps against a flat random-initialization baseline at 6.93. A
vertical marker near step 2 notes that the acceptance gate requires 5%
improvement, well past the point where the two curves first separate.](src/img/training-curve.png)

Without `--data`, training falls back to synthetic documents and there is no
heldout set, so the gate does not run.

### Resuming

```bash
agentmodel pretrain configs/pretrain_nano.yaml \
  --out runs/real --resume runs/real/checkpoint.pt --steps 150 --data corpus.jsonl
```

History is appended rather than overwritten, and the validation gate is
recomputed against the restored model.

## Configurations

Configs are YAML with an `extends` key for inheritance, resolved by
`agentmodel.config.load_config`. Every run config extends `base.yaml`.

- `configs/base.yaml` holds the full default surface for a 12-layer, 512-wide
  model, with each architectural choice documented inline.
- `configs/pretrain_nano.yaml` is a 6-layer, 1.4M-parameter model for CPU
  smoke tests.
- `configs/pretrain_nano10m.yaml` is a 12-layer, 9.77M-parameter model.
- `configs/sft_small.yaml` covers the fine-tuning stage.

`agentmodel info` prints the parameter count and a hash of the resolved
config, which makes it easy to confirm that a run used the settings you think
it did.

## Architecture

The model follows current practice for small decoder-only stacks rather than
the original 2017 Transformer.

Attention is grouped-query, so eight query heads share two key-value heads and
the KV cache shrinks accordingly. Queries and keys pass through RMSNorm before
rotary position encoding, which keeps attention logits bounded at long context.
Global layers gate their attention output. RoPE is applied to only part of each
head's dimension, leaving the rest unrotated, and can be extended with YaRN
when a run needs more context than it was trained on.

Layers alternate between sliding-window and global attention in a 5:1 ratio,
so most layers attend locally while a minority provide global receptive field.
A `hybrid` pattern substitutes gated linear attention for the local layers.

The feed-forward block is SwiGLU. Setting `n_expert` switches it to a
DeepSeekMoE router with shared experts and an auxiliary load-balancing loss.

## Training

![End-to-end training pipeline. Raw JSONL documents pass through
deduplication and quality filters, then a byte-level BPE tokenizer learned
from scratch, then sequence packing with cross-document attention masked.
The packed batches branch to two optimizers, Muon with Newton-Schulz
orthogonalization for the 2D hidden weights and AdamW for embeddings, norms,
biases and scalars, which converge on the trainer. The run ends in held-out
evaluation against a random-initialization baseline, with a checkpoint and
resume loop.](src/img/training-pipeline.png)

Muon, momentum orthogonalized by a Newton-Schulz iteration, updates the 2D
hidden weights. AdamW keeps embeddings, norms, biases, and scalars. The split
is automatic and is chosen because orthogonalizing a matrix gives every
direction the same scale, which suits large hidden weight matrices but would
distort an embedding table whose rows have very different norms.

Learning rate follows a warmup-stable-decay schedule. Precision is selectable,
an exponential moving average of weights is maintained, and gradient
checkpointing trades compute for memory. FLOPs per step are estimated from the
config so throughput can be compared across runs.

## Layout

```
src/agentmodel/
  model/       attention, RoPE, masks, SwiGLU and MoE, Gated DeltaNet, assembly
  data/        tokenizer, packing, mixture sampling, dedup, quality filters, chat
  train/       Muon and AdamW, trainer loop, losses, checkpointing, eval, SFT
  agent/       tool definitions, parser, context window, agent loop
  rlvr/        GRPO with verifiable rewards
  inference/   sampling and decoding
  scale/       int8 quantization and export
```

## Not yet built

These are known gaps rather than hidden ones:

- Muon applies no RMS-matching rescale to its update, so effective learning
  rate varies with matrix aspect ratio. Reference implementations scale the
  orthogonalized update to match Adam's RMS.
- GRPO is implemented as verified math with unit tests, but no training loop
  drives it and there is no execution sandbox, so rewards cannot yet come from
  real tool runs.
- Single-process only. No FSDP or tensor parallelism.
- Byte-level BPE trained from scratch; no pretrained tokenizer import.
- Attention runs in eager PyTorch with no fused kernels or FP8 path.

## Testing

130 tests covering model shapes and gradients, mask correctness, optimizer
behavior, packing and masking, checkpoint round-trips, resume, the validation
gate, sampling, quantization, SFT encoding, GRPO, and the CLI.

```bash
python3 -m pytest tests/ -q
python3 -m ruff check src tests
```

The suite runs on CPU in under a minute. Ruff is configured in
`pyproject.toml` under the `E`, `F`, `I`, `UP`, and `B` rule sets.

## Requirements

Python 3.10 or newer and PyTorch. The optional `train` extra pulls in
`datasets`, `tokenizers`, and `safetensors` for real corpus preparation; `dev`
adds `pytest` and `ruff`. Everything in the quick start runs without either.

Development here has been on CPU-only hardware, which is why the committed
configs are small. Training a 10M-parameter model is comfortable on CPU; the
architecture is intended to scale to GPUs that this repository has not been
tested on.
