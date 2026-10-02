"""A deterministic, code-aware byte-level BPE tokenizer.

The reference implementation intentionally has no compiled dependency. It
keeps whitespace, identifiers, numbers, and punctuation in separate
pre-tokenization spans, while retaining byte fallback for every UTF-8 string.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

_CODE_SPAN = re.compile(r"\s+|[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^A-Za-z_0-9\s]+")
_BYTE_VOCAB_SIZE = 256


def _spans(text: str) -> list[bytes]:
    """Split source into mergeable code-aware spans without losing characters."""
    out: list[bytes] = []
    end = 0
    for match in _CODE_SPAN.finditer(text):
        if match.start() > end:
            out.append(text[end : match.start()].encode("utf-8"))
        out.append(match.group(0).encode("utf-8"))
        end = match.end()
    if end < len(text):
        out.append(text[end:].encode("utf-8"))
    return [span for span in out if span]


def _merge_pair(tokens: list[bytes], pair: tuple[bytes, bytes], merged: bytes) -> list[bytes]:
    out: list[bytes] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


@dataclass
class CodeAwareBPETokenizer:
    """Train and apply a compact byte-level BPE tokenizer for source code."""

    vocab_size: int = 8192
    special_tokens: dict[str, int] = field(
        default_factory=lambda: {"<pad>": 0, "<bos>": 1, "<eos>": 2}
    )
    _vocab: dict[bytes, int] = field(default_factory=dict, init=False, repr=False)
    _tokens: list[bytes] = field(default_factory=list, init=False, repr=False)
    _merges: list[tuple[bytes, bytes]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.vocab_size < _BYTE_VOCAB_SIZE + len(self.special_tokens):
            raise ValueError("vocab_size must leave room for all byte and special tokens")
        self._reset_base_vocab()

    def _reset_base_vocab(self) -> None:
        self._vocab = {}
        self._tokens = []
        for _ in range(len(self.special_tokens)):
            self._tokens.append(b"")
        for value in range(_BYTE_VOCAB_SIZE):
            self._vocab[bytes([value])] = len(self._tokens)
            self._tokens.append(bytes([value]))
        self._merges = []

    @property
    def merges(self) -> tuple[tuple[bytes, bytes], ...]:
        return tuple(self._merges)

    def train(self, texts: Iterable[str], min_frequency: int = 2) -> CodeAwareBPETokenizer:
        """Learn the most frequent in-span byte pairs until the vocabulary is full."""
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")
        spans = [
            [[bytes([value]) for value in span] for span in _spans(text)]
            for text in texts
        ]
        self._reset_base_vocab()
        while len(self._tokens) < self.vocab_size:
            counts: Counter[tuple[bytes, bytes]] = Counter()
            for document in spans:
                for span in document:
                    counts.update(zip(span, span[1:], strict=False))
            if not counts:
                break
            pair, frequency = counts.most_common(1)[0]
            if frequency < min_frequency:
                break
            merged = pair[0] + pair[1]
            if merged in self._vocab:
                break
            self._vocab[merged] = len(self._tokens)
            self._tokens.append(merged)
            self._merges.append(pair)
            spans = [
                [_merge_pair(span, pair, merged) for span in document]
                for document in spans
            ]
        return self

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token ids, preserving every input byte."""
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.special_tokens["<bos>"])
        for span in _spans(text):
            pieces = [bytes([value]) for value in span]
            for pair in self._merges:
                merged = pair[0] + pair[1]
                pieces = _merge_pair(pieces, pair, merged)
            ids.extend(self._vocab[piece] for piece in pieces)
        if add_special_tokens:
            ids.append(self.special_tokens["<eos>"])
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Decode ids back to the original Unicode text."""
        return self.decode_bytes(ids, skip_special_tokens).decode("utf-8")

    def decode_bytes(self, ids: Iterable[int], skip_special_tokens: bool = True) -> bytes:
        """Decode ids to raw bytes without attempting UTF-8 assembly."""
        special_ids = set(self.special_tokens.values())
        return b"".join(
            self._tokens[token_id]
            for token_id in ids
            if 0 <= token_id < len(self._tokens)
            and (not skip_special_tokens or token_id not in special_ids)
        )

    def decode_stream(
        self, ids: Iterable[int], skip_special_tokens: bool = True
    ) -> tuple[str, bytes]:
        """Decode a partial token stream, holding back an incomplete final rune.

        Byte-level BPE can emit half a multi-byte character mid-stream, so
        strict decoding would raise on every generation. Bytes that cannot yet
        form a valid character are returned as a carry buffer to prepend to the
        next chunk instead of being dropped or replaced.
        """
        raw = self.decode_bytes(ids, skip_special_tokens)
        for cut in range(0, min(4, len(raw)) + 1):
            try:
                return raw[: len(raw) - cut].decode("utf-8"), raw[len(raw) - cut :]
            except UnicodeDecodeError:
                continue
        return "", raw

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "tokens": [token.hex() for token in self._tokens],
            "merges": [[left.hex(), right.hex()] for left, right in self._merges],
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> CodeAwareBPETokenizer:
        payload = json.loads(Path(path).read_text())
        tokenizer = cls(payload["vocab_size"], payload["special_tokens"])
        tokenizer._tokens = [bytes.fromhex(token) for token in payload["tokens"]]
        tokenizer._vocab = {
            token: token_id
            for token_id, token in enumerate(tokenizer._tokens)
            if token and token_id >= len(tokenizer.special_tokens)
        }
        tokenizer._merges = [
            (bytes.fromhex(left), bytes.fromhex(right)) for left, right in payload["merges"]
        ]
        return tokenizer
