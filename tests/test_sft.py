import torch

from agentmodel.config import load_config
from agentmodel.data.chat import ChatMessage, ChatTemplate, assistant_targets
from agentmodel.data.tokenizer import CodeAwareBPETokenizer
from agentmodel.data.traces import filter_passing_traces
from agentmodel.model.transformer import Transformer
from agentmodel.train.sft import collate, encode_traces, run_sft, sft_batches


def _tokenizer() -> CodeAwareBPETokenizer:
    return CodeAwareBPETokenizer(vocab_size=300).train(
        ["read the file", "the file says hello", '{"name":"read"}']
    )


def _trace(content: str = "hello") -> dict:
    return {
        "tests_passed": True,
        "messages": [
            {"role": "user", "content": "read the file"},
            {"role": "assistant", "content": content},
        ],
    }


def test_chat_template_preserves_content_and_masks_only_assistant():
    tokenizer = _tokenizer()
    messages = [ChatMessage("user", "read the file"), ChatMessage("assistant", "hello")]
    template = ChatTemplate()
    encoded = template.encode(messages, tokenizer)

    assert template.render(messages).endswith("hello\n")
    inputs, targets = assistant_targets(encoded)
    assert inputs.numel() == targets.numel()
    supervised = (targets != -100).sum().item()
    assert 0 < supervised < targets.numel()


def test_chat_template_rejects_unknown_role():
    template = ChatTemplate()
    try:
        template.encode([ChatMessage("wizard", "hi")], _tokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("unknown role should raise")


def test_chat_template_handles_tool_role():
    template = ChatTemplate()
    encoded = template.encode(
        [ChatMessage("user", "q"), ChatMessage("tool", "a.txt"), ChatMessage("assistant", "ok")],
        _tokenizer(),
    )
    assert len(encoded.input_ids) == len(encoded.loss_mask)


def test_assistant_targets_pad_to_length():
    template = ChatTemplate()
    encoded = template.encode([ChatMessage("assistant", "hello")], _tokenizer())
    inputs, targets = assistant_targets(encoded, pad_to=200)
    assert inputs.shape == targets.shape == (200,)


def test_trace_filter_requires_valid_passing_trace():
    traces = [
        {"messages": [{"role": "user", "content": "x"}], "tests_passed": True},
        {"messages": [{"role": "user", "content": "x"}], "tests_passed": False},
        {"messages": [{"role": "bad", "content": "x"}], "tests_passed": True},
        {"messages": [], "tests_passed": True},
        "not a dict",
    ]
    assert len(filter_passing_traces(traces)) == 1


def test_encode_traces_shifts_targets():
    rows = encode_traces([_trace()], _tokenizer())
    assert rows[0]["targets"][-1] == -100
    assert any(t != -100 for t in rows[0]["targets"])


def test_collate_pads_and_masks():
    rows = encode_traces([_trace("a"), _trace("bb")], _tokenizer())
    inputs, targets = collate(rows, seq_len=64)
    assert inputs.shape == targets.shape == (2, 64)
    assert (targets[:, -1] == -100).all()


def test_sft_batches_skips_rows_with_no_supervision():
    rows = [{"input_ids": [1, 2, 3], "targets": [-100, -100, -100]}]
    assert list(sft_batches(rows, seq_len=8, micro_batch_size=1)) == []


def test_run_sft_reduces_loss_on_one_batch():
    cfg = load_config("configs/pretrain_nano.yaml")
    cfg.model = type(cfg.model)(
        vocab_size=300, n_layer=2, d_model=64, n_head=4, n_kv_head=2,
        d_ff=176, max_seq_len=128, partial_rope_dim=16, layer_pattern="global",
        local_window=32,
    )
    cfg.data.sequence_length = 64
    rows = encode_traces([_trace("hello"), _trace("hello world")], _tokenizer())

    history = run_sft(cfg, rows, steps=30)
    assert history[-1]["loss"] < history[0]["loss"]
    assert history[-1]["supervised_tokens"] > 0
    assert torch.isfinite(torch.tensor(history[-1]["loss"]))


def test_sft_only_supervises_assistant_tokens():
    """Regression guard: training on tool results teaches hallucination."""
    template = ChatTemplate()
    tokenizer = _tokenizer()
    messages = [
        ChatMessage("user", "read"),
        ChatMessage("tool", "file contents"),
        ChatMessage("assistant", "ok"),
    ]
    encoded = template.encode(messages, tokenizer)
    assert any(encoded.loss_mask), "assistant turn should be supervised"
    assert sum(encoded.loss_mask) == len(tokenizer.encode("ok\n"))

    _, targets = assistant_targets(encoded)
    expected = [t for t, m in zip(encoded.input_ids, encoded.loss_mask) if m][1:]
    assert targets[targets != -100].tolist() == expected