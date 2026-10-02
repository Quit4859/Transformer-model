import torch

from agentmodel.config import load_config
from agentmodel.data.tokenizer import CodeAwareBPETokenizer
from agentmodel.inference.sampling import Generator, SamplingConfig, filter_logits, sample_token
from agentmodel.model.config import ModelConfig
from agentmodel.model.transformer import Transformer


def logits(values: list[float]) -> torch.Tensor:
    return torch.tensor(values)


def _tiny_tokenizer() -> CodeAwareBPETokenizer:
    return CodeAwareBPETokenizer(vocab_size=300).train(
        ["hello world", "the quick brown fox", "café"]
    )


def _tiny_model(vocab_size: int) -> Transformer:
    return Transformer(
        ModelConfig(
            vocab_size=vocab_size,
            n_layer=1,
            d_model=16,
            n_head=2,
            n_kv_head=1,
            d_ff=32,
            max_seq_len=32,
            partial_rope_dim=4,
            layer_pattern="global",
            local_window=8,
        )
    ).eval()


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
    seen = {
        sample_token(logits([0.0, 0.0, 5.0]), SamplingConfig(top_k=1)) for _ in range(50)
    }
    assert seen == {2}


def test_sample_token_is_seed_reproducible():
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    assert sample_token(logits([1.0, 2.0, 3.0]), SamplingConfig(), generator=g1) == sample_token(
        logits([1.0, 2.0, 3.0]), SamplingConfig(), generator=g2
    )


def test_generator_requires_tokenizer():
    try:
        Generator(model=_tiny_model(300)).generate("hi")
    except ValueError:
        pass
    else:
        raise AssertionError("generator must require a tokenizer")


def test_decode_stream_holds_back_incomplete_rune():
    """Regression guard: byte-level BPE emits half a character mid-stream."""
    tok = _tiny_tokenizer()
    raw = "café".encode()  # b"caf\\xc3\\xa9"
    ids = [tok._vocab[bytes([b])] for b in raw]
    # cut -> (complete text, bytes held back until the rune can be finished)
    expected = {1: ("c", b""), 2: ("ca", b""), 3: ("caf", b""), 4: ("caf", b"\xc3")}
    for cut, (want_text, want_carry) in expected.items():
        text, carry = tok.decode_stream(ids[:cut])
        assert (text, carry) == (want_text, want_carry), f"cut={cut}"
        # Nothing is lost or duplicated: text + carry reconstitutes the input.
        assert text.encode("utf-8") + carry == raw[:cut]


def test_decode_stream_is_exact_once_complete():
    tok = _tiny_tokenizer()
    raw = "café".encode()
    ids = [tok._vocab[bytes([b])] for b in raw]
    text, carry = tok.decode_stream(ids)
    assert text == "café"
    assert carry == b""


def test_generate_survives_random_tokens():
    """The generator must not crash on invalid UTF-8 from an untrained model."""
    tok = _tiny_tokenizer()
    model = _tiny_model(len(tok._tokens))
    generator = Generator(
        model=model, tokenizer=tok, cfg=SamplingConfig(max_new_tokens=16, temperature=1.0)
    )
    torch.manual_seed(0)
    out = generator.generate("hello")
    assert isinstance(out, str)
    assert generator.cache is not None


def test_generate_returns_full_text_for_a_deterministic_model():
    """A model biased toward real tokens must decode them without error."""
    tok = _tiny_tokenizer()
    model = _tiny_model(len(tok._tokens))
    target = tok.encode("hello world")
    with torch.no_grad():
        for token in target:
            model.lm_head.weight[token] += 8.0
    generator = Generator(
        model=model, tokenizer=tok, cfg=SamplingConfig(max_new_tokens=3, temperature=0.01)
    )
    out = generator.generate("")
    assert isinstance(out, str) and out != ""
    assert generator.cache is not None