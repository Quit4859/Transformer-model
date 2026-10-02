from dataclasses import replace

import torch

from agentmodel.config import load_config
from agentmodel.data.packing import pack_documents
from agentmodel.model.config import ModelConfig
from agentmodel.train.checkpoint import load_checkpoint, save_checkpoint
from agentmodel.train.eval import evaluate
from agentmodel.train.loop import Trainer
from agentmodel.train.pretrain import run


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
