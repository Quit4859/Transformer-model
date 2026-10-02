import torch

from agentmodel.config import load_config
from agentmodel.model.config import ModelConfig
from agentmodel.train.pretrain import run_overfit_gate


def test_overfit_gate_reports_finite_training_metrics():
    cfg = load_config("configs/pretrain_nano.yaml")
    cfg.model = ModelConfig(
        vocab_size=16,
        n_layer=1,
        d_model=16,
        n_head=2,
        n_kv_head=1,
        d_ff=32,
        max_seq_len=16,
        partial_rope_dim=4,
        layer_pattern="global",
        local_window=8,
    )
    cfg.data.sequence_length = 16
    docs = [[3, 4, 5, 6, 7, 2]] * 10

    metrics = run_overfit_gate(cfg, docs, steps=2, learning_rate=1.0)

    assert metrics["finite_gradients"] == 1.0
    assert torch.isfinite(torch.tensor(metrics["heldout_loss"]))
