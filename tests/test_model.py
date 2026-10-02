import math

import torch

from agentmodel.config import load_config
from agentmodel.model.attention import GatedAttention
from agentmodel.model.config import ModelConfig
from agentmodel.model.rope import build_inv_freq
from agentmodel.model.transformer import Transformer
from agentmodel.train.optim import zeropower_via_newtonschulz5
from agentmodel.train.optim_setup import build_optimizers, lr_scale


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=64,
        n_layer=2,
        d_model=32,
        n_head=4,
        n_kv_head=2,
        d_ff=88,
        max_seq_len=64,
        partial_rope_dim=8,
        local_window=16,
        layer_pattern="swa",
    )
    base.update(kw)
    return ModelConfig(**base)


def test_param_count_matches_config():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    assert abs(model.num_params() - cfg.n_params()) < cfg.d_model * 4


def test_forward_shape():
    model = Transformer(tiny_cfg())
    ids = torch.randint(0, 64, (2, 16))
    out = model(ids)
    assert out.logits.shape == (2, 16, 64)
    assert torch.isfinite(out.logits).all()


def test_all_layer_patterns_forward():
    for pattern in ("global", "swa", "hybrid"):
        model = Transformer(tiny_cfg(layer_pattern=pattern, n_layer=4))
        ids = torch.randint(0, 64, (2, 32))
        out = model(ids)
        assert out.logits.shape == (2, 32, 64)
        assert torch.isfinite(out.logits).all()


def test_swa_respects_window():
    cfg = tiny_cfg(layer_pattern="global", n_layer=1)
    attn = GatedAttention(cfg, 0, sliding_window=4).eval()
    x = torch.randn(1, 16, cfg.d_model)
    out_a = attn(x)
    x_b = x.clone()
    x_b[:, 0, :] += 1.0
    out_b = attn(x_b)
    assert torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-4)
    assert not torch.allclose(out_a[:, 2], out_b[:, 2], atol=1e-4)


def test_layer_patterns_resolve():
    assert tiny_cfg(layer_pattern="global", n_layer=4).layer_types() == ["full"] * 4
    swa = tiny_cfg(layer_pattern="swa", n_layer=11, swa_every=5).layer_types()
    assert swa.count("full") == 4  # i=0, i=4, i=9, and the final layer
    assert swa.count("swa") == 7
    assert swa[-1] == "full"
    hybrid = tiny_cfg(layer_pattern="hybrid", n_layer=11, swa_every=5).layer_types()
    assert hybrid.count("delta") == 7 and hybrid.count("full") == 4
    assert tiny_cfg(layer_pattern="hybrid", n_layer=4).layer_types() == [
        "full",
        "delta",
        "delta",
        "full",
    ]


def test_global_layer_sees_whole_prefix():
    cfg = tiny_cfg(layer_pattern="global", n_layer=1)
    attn = GatedAttention(cfg, 0, sliding_window=None).eval()
    x = torch.randn(1, 16, cfg.d_model)
    out_a = attn(x)
    x_b = x.clone()
    x_b[:, 0, :] += 1.0
    out_b = attn(x_b)
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-4)


def test_moe_forward_and_aux():
    cfg = tiny_cfg(n_expert=4, n_expert_used=2, n_shared_expert=1, n_layer=2)
    model = Transformer(cfg)
    ids = torch.randint(0, 64, (2, 16))
    out = model(ids)
    assert out.logits.shape == (2, 16, 64)
    assert out.aux_loss is not None and torch.isfinite(out.aux_loss)


def test_gradients_flow():
    model = Transformer(tiny_cfg())
    ids = torch.randint(0, 64, (2, 16))
    loss = torch.nn.functional.cross_entropy(
        out_logits := model(ids).logits.reshape(-1, 64), torch.randint(0, 64, (32,))
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_gradcheck_small():
    """Numerical gradcheck on the embedding of a 1-layer stack.

    Tolerances are loose because the reference SDPA kernel in float64 still
    carries a few ulps of error at this size.
    """
    cfg = ModelConfig(
        vocab_size=16,
        n_layer=1,
        d_model=8,
        n_head=2,
        n_kv_head=1,
        d_ff=24,
        max_seq_len=16,
        partial_rope_dim=0,
        layer_pattern="global",
        local_window=4,
    )
    model = Transformer(cfg).double()
    ids = torch.randint(0, 16, (1, 4))
    assert torch.autograd.gradcheck(
        lambda _e: model(ids).logits.sum(),
        (model.embed.weight,),
        eps=1e-6,
        atol=1e-3,
        rtol=1e-2,
        fast_mode=True,
    )


def test_gradcheck_delta_net():
    cfg = ModelConfig(
        vocab_size=16,
        n_layer=2,
        d_model=8,
        n_head=2,
        n_kv_head=1,
        d_ff=24,
        max_seq_len=16,
        partial_rope_dim=0,
        layer_pattern="hybrid",
        local_window=4,
    )
    model = Transformer(cfg).double()
    ids = torch.randint(0, 16, (1, 4))
    assert torch.autograd.gradcheck(
        lambda _e: model(ids).logits.sum(),
        (model.embed.weight,),
        eps=1e-6,
        atol=1e-3,
        rtol=1e-2,
        fast_mode=True,
    )


def test_newtonschulz_flattens_singular_values():
    """The NS5 quintic is a polynomial approximation, not exact orthogonalization.

    It clusters the singular values near 1 instead of driving them all to 1, so
    the property to assert is that conditioning collapses, not that q.T @ q == I.
    """
    # A random matrix is already near-semi-orthogonal, so build a badly
    # conditioned one: singular values spread over three decades.
    u, _ = torch.linalg.qr(torch.randn(64, 32))
    v, _ = torch.linalg.qr(torch.randn(32, 32))
    g = u @ torch.diag(torch.logspace(0, -3, 32)) @ v.T
    s_in = torch.linalg.svdvals(g)
    q = zeropower_via_newtonschulz5(g, steps=5)
    s_out = torch.linalg.svdvals(q)
    assert s_out.shape == s_in.shape
    cond_in = s_in.max() / s_in.min()
    cond_out = s_out.max() / s_out.min()
    assert cond_in > 100
    assert cond_out < cond_in / 10
    assert s_out.max() < 1.5
    assert s_out.min() > 0.25
    assert torch.isfinite(q).all()


def test_newtonschulz_preserves_shape_and_dtype():
    g = torch.randn(48, 24, dtype=torch.float64)
    q = zeropower_via_newtonschulz5(g, steps=5)
    assert q.shape == g.shape and q.dtype == g.dtype


def test_param_split_routes_2d_to_muon():
    model = Transformer(tiny_cfg())
    adamw, muon = build_optimizers(model, load_config("configs/pretrain_nano.yaml").optim)
    assert muon is not None
    muon_ids = {id(p) for g in muon.param_groups for p in g["params"]}
    adamw_ids = {id(p) for g in adamw.param_groups for p in g["params"]}
    assert not (muon_ids & adamw_ids)
    assert len(muon_ids) > 0 and len(adamw_ids) > 0


def test_lr_schedule_warmup_and_decay():
    from agentmodel.train.optim_setup import OptimConfig

    cfg = OptimConfig(warmup_steps=10, decay_steps=100, schedule="wsd", min_lr_ratio=0.1)
    assert abs(lr_scale(0, cfg) - 0.1) < 1e-6
    assert abs(lr_scale(9, cfg) - 1.0) < 1e-6
    assert lr_scale(10, cfg) == 1.0
    assert abs(lr_scale(100, cfg) - 0.1) < 1e-6


def test_rope_inv_freq_shape_and_monotonic():
    inv = build_inv_freq(head_dim=32, theta=500000.0, partial_rope_dim=8)
    assert inv.shape == (4,)
    assert (inv[1:] < inv[:-1]).all()


def test_kv_cache_decode_matches_full_forward():
    """Prefill + one cached decode step must reproduce the full forward pass."""
    model = Transformer(tiny_cfg(n_layer=2, layer_pattern="global")).eval()
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        full = model(ids).logits[0, 6]  # position 6, the last one prefilled
        cache = model.new_cache()
        prefill = model(ids[:, :7], cache=cache, use_cache=True).logits[0, -1]
        assert torch.allclose(prefill, full, atol=1e-4)
    with torch.no_grad():
        step = model(ids[:, 7:8], cache=cache, use_cache=True)
        decode = step.logits[0, -1]
        full_step = model(ids).logits[0, 7]
    assert cache.layers[0].k.shape[2] == 8
    assert torch.allclose(decode, full_step, atol=1e-4)


def test_kv_cache_grows_by_one_token_per_step():
    model = Transformer(tiny_cfg(n_layer=1, layer_pattern="global")).eval()
    cache = model.new_cache()
    with torch.no_grad():
        model(torch.randint(0, 64, (1, 4)), cache=cache, use_cache=True)
        assert cache.layers[0].k.shape[2] == 4
        model(torch.randint(0, 64, (1, 1)), cache=cache, use_cache=True)
        assert cache.layers[0].k.shape[2] == 5


def test_tied_embeddings_share_weight():
    cfg = tiny_cfg(tie_embeddings=True)
    model = Transformer(cfg)
    assert model.lm_head.weight is model.embed.weight
