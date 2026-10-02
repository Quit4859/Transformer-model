import torch

from agentmodel.inference.sampling import SamplingConfig, filter_logits, sample_token
from agentmodel.model.config import ModelConfig
from agentmodel.model.transformer import Transformer


def logits(values: list[float]) -> torch.Tensor:
    return torch.tensor(values)


def test_filter_logits_applies_temperature():
    filtered = filter_logits(logits([1.0, 1.0]), SamplingConfig(temperature=2.0))
    assert torch.allclose(filtered, torch.tensor([0.5, 0.5]))


def test_filter_logits_top_k_keeps_only_k_finest():
    filtered = filter_logits(logits([1.0, 2.0, 3.0]), SamplingConfig(top_k=1))
    assert torch.isinf(filtered[0]) and torch.isinf(filtered[1])
    assert not torch.isinf(filtered[2])


def test_filter_logits_top_p_narrows_support():
    filtered = filter_logits(logits([10.0, 0.0, 0.0]), SamplingConfig(top_p=0.5))
    assert torch.isinf(filtered[1]) and torch.isinf(filtered[2])


def test_filter_logits_rejects_zero_temperature():
    try:
        filter_logits(logits([1.0]), SamplingConfig(temperature=0.0))
    except ValueError:
        pass
    else:
        raise AssertionError("temperature must be validated")


def test_sample_token_respects_top_k():
    counts = {}
    for _ in range(50):
        counts[sample_token(logits([0.0, 0.0, 5.0]), SamplingConfig(top_k=1))] = 1
    assert set(counts) == {2}


def test_sample_token_is_seed_reproducible():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = sample_token(logits([1.0, 2.0, 3.0]), SamplingConfig(), generator=g1)
    b = sample_token(logits([1.0, 2.0, 3.0]), SamplingConfig(), generator=g2)
    assert a == b


def test_generator_requires_tokenizer():
    from agentmodel.inference.sampling import Generator

    model = Transformer(ModelConfig(
        vocab_size=32, n_layer=1, d_model=16, n_head=2, n_kv_head=1,
        d_ff=32, max_seq_len=32, partial_rope_dim=4, layer_pattern="global",
        local_window=8,
    ))
    try:
        Generator(model=model).generate("hi")
    except ValueError:
        pass
    else:
        raise AssertionError("generator must require a tokenizer")