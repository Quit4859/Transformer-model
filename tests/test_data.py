from agentmodel.config import load_config
from agentmodel.data.mixture import Mixture, MixtureSampler
from agentmodel.data.packing import pack_documents
from agentmodel.data.prepare import read_texts
from agentmodel.data.quality import benchmark_decontaminate
from agentmodel.data.tokenizer import CodeAwareBPETokenizer
from agentmodel.model.transformer import Transformer


def test_tokenizer_round_trips_exact_bytes():
    tok = CodeAwareBPETokenizer(vocab_size=300).train(["def f(x):\n    return x + 1\n"])
    text = "def add(a, b):\n    # café naïve\n    return a + b\n"
    assert tok.decode(tok.encode(text)) == text


def test_tokenizer_handles_non_ascii_without_loss():
    tok = CodeAwareBPETokenizer(vocab_size=260).train(["日本語のテキスト"])
    assert tok.decode(tok.encode("日本語")) == "日本語"


def test_tokenizer_is_deterministic():
    corpus = ["alpha beta", "beta gamma"] * 5
    a = CodeAwareBPETokenizer(vocab_size=270).train(corpus).encode("alpha beta gamma")
    b = CodeAwareBPETokenizer(vocab_size=270).train(corpus).encode("alpha beta gamma")
    assert a == b


def test_tokenizer_save_load_round_trip(tmp_path):
    tok = CodeAwareBPETokenizer(vocab_size=280).train(["class A:\n    pass\n"])
    path = tmp_path / "tok.json"
    tok.save(path)
    reloaded = CodeAwareBPETokenizer.load(path)
    assert reloaded.encode("class A:\n    pass\n") == tok.encode("class A:\n    pass\n")


def test_tokenizer_compresses_code_better_than_raw_bytes():
    corpus = ["def add(value):\n    return value + 1\n"] * 8
    tok = CodeAwareBPETokenizer(vocab_size=512).train(corpus)
    assert len(tok.encode(corpus[0])) < len(corpus[0].encode("utf-8"))


def test_tokenizer_add_special_tokens():
    tok = CodeAwareBPETokenizer(vocab_size=260)
    ids = tok.encode("x", add_special_tokens=True)
    assert ids[0] == tok.special_tokens["<bos>"]
    assert ids[-1] == tok.special_tokens["<eos>"]


def test_decontam_flags_high_overlap_document():
    result = benchmark_decontaminate(
        ["noise noise noise the quick brown fox jumps over the lazy dog today"],
        ["the quick brown fox jumps over the lazy dog today"],
        ngram_size=5,
        threshold=0.8,
    )
    assert result.contaminated_indices == [0]
    assert result.documents == []


def test_decontam_keeps_clean_documents():
    result = benchmark_decontaminate(
        ["completely different content about databases and kernels"],
        ["alpha beta gamma delta epsilon"],
        ngram_size=5,
        threshold=0.8,
    )
    assert result.contaminated_indices == []
    assert len(result.documents) == 1
    assert result.overlap_scores[0] == 0.0


def test_decontam_ignores_reformatting():
    result = benchmark_decontaminate(
        ["alpha   beta\n\tgamma  delta epsilon"],
        ["alpha beta gamma delta epsilon"],
        ngram_size=5,
        threshold=0.9,
    )
    assert result.contaminated_indices == [0]


def test_mixture_weights_follow_temperature():
    mixtures = [Mixture("a", 9.0), Mixture("b", 1.0)]
    flat = MixtureSampler(mixtures, temperature=1.0).probabilities()
    assert abs(flat[0] - 0.9) < 1e-6
    sharp = MixtureSampler(mixtures, temperature=4.0).probabilities()
    assert sharp[0] > flat[0]


def test_mixture_sampling_is_seed_reproducible():
    mixtures = [Mixture("a"), Mixture("b")]
    assert MixtureSampler(mixtures, seed=3).sample(10) == MixtureSampler(mixtures, seed=3).sample(10)


def test_prepare_reads_texts(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"text": "hello"}\n{"text": "  "}\n{"text": "world"}\n')
    assert read_texts(str(path)) == ["hello", "world"]


def test_nano10m_config_is_about_10m_params():
    cfg = load_config("configs/pretrain_nano10m.yaml").model
    model = Transformer(cfg)
    assert 8e6 < model.num_params() < 14e6
    assert abs(model.num_params() - cfg.n_params()) <= cfg.d_model


def test_overfit_gate_drives_nano_to_low_loss():
    """Phase 1 acceptance: a small model memorizes one small batch."""
    cfg = load_config("configs/pretrain_nano10m.yaml")
    cfg.model.n_layer = 2
    cfg.model.d_model = 64
    cfg.model.n_head = 4
    cfg.model.d_ff = 176
    cfg.train.grad_accum_steps = 1
    cfg.optim.warmup_steps = 10
    cfg.optim.decay_steps = 120

    from agentmodel.train.loop import Trainer

    trainer = Trainer(cfg)
    tok = CodeAwareBPETokenizer(vocab_size=cfg.model.vocab_size)
    docs = [tok.encode("def add(value):\n    return value + 1\n") for _ in range(4)]
    batch = pack_documents(docs, cfg.data.sequence_length, pad_token_id=0)

    first = trainer.micro_step(batch).loss
    trainer.advance()
    for _ in range(60):
        last = trainer.micro_step(batch).loss
        trainer.advance()
    assert last < first * 0.5