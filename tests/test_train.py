import torch

from agentmodel.config import load_config
from agentmodel.data.packing import pack_documents
from agentmodel.model.config import ModelConfig
from agentmodel.model.transformer import Transformer
from agentmodel.train.losses import count_valid, doc_boundary_mask, token_cross_entropy
from agentmodel.train.loop import Trainer
from agentmodel.train.pretrain import build_batch, synthetic_docs
from agentmodel.train.precision import EMA, resolve_dtype


def test_pack_shapes_and_targets():
    docs = [[5, 6, 7], [8, 9], [10]]
    batch = pack_documents(docs, seq_len=8, pad_token_id=0)
    assert batch.input_ids.shape == (1, 8)
    assert batch.targets.shape == (1, 8)
    assert batch.doc_ids.shape == (1, 8)
    assert batch.input_ids[0, :3].tolist() == [5, 6, 7]


def test_pack_masks_padding_targets():
    docs = [[5, 6, 7]]
    batch = pack_documents(docs, seq_len=8, pad_token_id=0)
    # only real positions get a target, and the last column never does
    valid = batch.targets != -100
    assert not valid[:, -1].any()
    assert valid[0, :2].tolist() == [True, True]


def test_pack_records_doc_boundaries():
    docs = [[1, 2], [3, 4]]
    batch = pack_documents(docs, seq_len=8, pad_token_id=0)
    assert batch.doc_ids[0, 0].item() == 0
    assert batch.doc_ids[0, 2].item() == 1
    assert (batch.doc_ids[0, 4:] == -1).all()


def test_pack_multiple_rows():
    docs = [list(range(10, 25)) for _ in range(4)]
    batch = pack_documents(docs, seq_len=8, pad_token_id=0)
    assert batch.input_ids.shape[0] > 1
    assert batch.num_real_tokens <= batch.input_ids.numel()


def test_doc_boundary_mask():
    seq = torch.tensor([[0, 0, 1, 1, 2]])
    starts = doc_boundary_mask(seq)
    assert starts[0].tolist() == [True, False, True, False, True]


def test_count_valid():
    targets = torch.tensor([[1, -100, 3]])
    assert count_valid(targets) == 2


def test_token_cross_entropy_ignores_pad():
    logits = torch.zeros(1, 3, 8)
    targets = torch.tensor([[0, -100, 5]])
    loss = token_cross_entropy(logits, targets)
    assert abs(loss.item() - torch.log(torch.tensor(8.0)).item()) < 1e-4


def test_synthetic_docs_respect_vocab():
    docs = synthetic_docs(4, 12, vocab=50, eos=2, seed=0)
    assert len(docs) == 4
    for d in docs:
        assert d[-1] == 2
        assert all(3 <= t < 50 for t in d[:-1])


def test_trainer_reduces_loss():
    cfg = load_config("configs/pretrain_nano.yaml")
    cfg.model = ModelConfig(
        vocab_size=64,
        n_layer=2,
        d_model=32,
        n_head=4,
        n_kv_head=2,
        d_ff=88,
        max_seq_len=64,
        partial_rope_dim=8,
        layer_pattern="swa",
        local_window=16,
    )
    cfg.train.grad_accum_steps = 1
    cfg.optim.warmup_steps = 5
    cfg.optim.decay_steps = 40
    trainer = Trainer(cfg)
    batch = build_batch(cfg, seq_len=64)
    first = trainer.micro_step(batch).loss
    trainer.advance()
    for _ in range(25):
        last = trainer.micro_step(batch).loss
        trainer.advance()
    assert last < first


def test_trainer_grad_norm_only_on_step_micro():
    cfg = load_config("configs/pretrain_nano.yaml")
    cfg.train.grad_accum_steps = 2
    trainer = Trainer(cfg)
    batch = build_batch(cfg, seq_len=32)
    s0 = trainer.micro_step(batch)
    trainer.advance()
    s1 = trainer.micro_step(batch)
    trainer.advance()
    assert s0.grad_norm is None
    assert s1.grad_norm is not None


def test_ema_tracks_weights():
    model = Transformer(ModelConfig(
        vocab_size=32, n_layer=1, d_model=16, n_head=2, n_kv_head=1,
        d_ff=32, max_seq_len=16, partial_rope_dim=4, layer_pattern="global",
        local_window=8,
    ))
    ema = EMA(model, decay=0.9)
    before = ema.state_dict()["norm_out.weight"].clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    after = ema.state_dict()["norm_out.weight"]
    assert not torch.allclose(before, after)


def test_resolve_dtype():
    assert resolve_dtype("fp32") is torch.float32
    assert resolve_dtype("bf16") is torch.bfloat16
    try:
        resolve_dtype("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")