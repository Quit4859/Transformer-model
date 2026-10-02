import json
from dataclasses import replace

import torch

from agentmodel.config import load_config
from agentmodel.data.packing import pack_documents
from agentmodel.model.config import ModelConfig
from agentmodel.train.checkpoint import load_checkpoint, save_checkpoint
from agentmodel.train.eval import evaluate
from agentmodel.train.loop import Trainer
from agentmodel.train.pretrain import run, split_documents, synthetic_docs


def _config():
    cfg = load_config("configs/pretrain_nano.yaml")
    cfg.model = ModelConfig(
        vocab_size=16,
        n_layer=1,
        d_model=16,
        n_head=2,
        n_kv_head=1,
        d_ff=32,
        max_seq_len=8,
        partial_rope_dim=4,
        layer_pattern="global",
        local_window=8,
    )
    cfg.data.sequence_length = 8
    cfg.optim = replace(cfg.optim, optimizer="adamw")
    return cfg


def test_checkpoint_round_trip_and_evaluation(tmp_path):
    cfg = _config()
    trainer = Trainer(cfg)
    batch = pack_documents([[3, 4, 5, 2]], 8)
    trainer.micro_step(batch, accumulate=False)
    trainer.advance()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trainer)

    restored = Trainer(cfg)
    metadata = load_checkpoint(path, restored)
    assert metadata["step_index"] == 1
    assert restored.tokens_seen == trainer.tokens_seen
    metrics = evaluate(restored, [batch])
    assert metrics["loss"] > 0
    assert torch.isfinite(torch.tensor(metrics["perplexity"]))


def test_pretraining_can_resume_from_checkpoint(tmp_path):
    cfg = _config()
    output = tmp_path / "run"
    run(cfg, steps=1, out_dir=str(output), seq_len=8, log_every=1)
    checkpoint = output / "checkpoint.pt"
    run(cfg, steps=1, out_dir=str(output), seq_len=8, log_every=1, resume=str(checkpoint))
    lines = (output / "history.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_split_documents_is_disjoint_and_deterministic():
    docs = synthetic_docs(20, 8, 32, 2, seed=1)
    train, heldout = split_documents(docs, heldout_fraction=0.25)
    assert len(train) == 15 and len(heldout) == 5
    assert not {id(d) for d in train} & {id(d) for d in heldout}
    assert split_documents(docs, 0.25) == (train, heldout)


def test_split_documents_rejects_bad_fraction():
    docs = synthetic_docs(4, 8, 32, 2, seed=1)
    for fraction in (0.0, 1.0, -0.5, 2.0):
        try:
            split_documents(docs, fraction)
        except ValueError:
            continue
        raise AssertionError(f"heldout_fraction={fraction} must be rejected")


def test_single_document_cannot_be_split():
    docs = synthetic_docs(1, 8, 32, 2, seed=1)
    try:
        split_documents(docs, 0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("a one-document corpus has no valid split")


def test_run_reports_baseline_and_gate_on_heldout(tmp_path):
    """The validation gate must compare against random init on unseen tokens."""
    cfg = _config()
    docs = synthetic_docs(16, 8, cfg.model.vocab_size, cfg.data.eos_token_id, seed=3)
    run(cfg, steps=1, out_dir=str(tmp_path / "r"), seq_len=8, log_every=100, documents=docs)
    metrics = json.loads((tmp_path / "r" / "eval.json").read_text())
    assert "baseline_loss" in metrics
    assert "relative_improvement" in metrics
    assert metrics["beats_random_init"] in (0.0, 1.0)


def test_gate_requires_a_real_margin_not_just_noise(tmp_path):
    """One optimizer step must not be able to satisfy the acceptance gate."""
    cfg = _config()
    docs = synthetic_docs(16, 8, cfg.model.vocab_size, cfg.data.eos_token_id, seed=3)
    run(cfg, steps=1, out_dir=str(tmp_path / "g"), seq_len=8, log_every=100, documents=docs)
    metrics = json.loads((tmp_path / "g" / "eval.json").read_text())
    assert metrics["beats_random_init"] == 0.0
    assert metrics["relative_improvement"] < 0.05
