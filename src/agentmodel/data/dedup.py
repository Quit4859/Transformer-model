"""Deterministic MinHash near-duplicate filtering for text and source documents."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence


_MAX_HASH = (1 << 64) - 1


def _shingles(text: str, size: int) -> set[str]:
    if size < 1:
        raise ValueError("shingle_size must be positive")
    if len(text) <= size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _hash_shingle(shingle: str, seed: int) -> int:
    digest = hashlib.blake2b(
        shingle.encode("utf-8"), digest_size=8, person=seed.to_bytes(8, "little")
    )
    return int.from_bytes(digest.digest(), "little")


@dataclass(frozen=True)
class DedupResult:
    documents: list[str]
    kept_indices: list[int]
    duplicate_indices: list[int]


class MinHashDeduplicator:
    """Keep the first document from each MinHash similarity cluster."""

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 64,
        shingle_size: int = 5,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        if shingle_size < 1:
            raise ValueError("shingle_size must be positive")
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.seed = seed
        self._salts = tuple(seed + index * 0x9E3779B1 for index in range(num_perm))

    def signature(self, text: str) -> tuple[int, ...]:
        """Return a deterministic MinHash signature for one document."""
        shingles = _shingles(text, self.shingle_size)
        return tuple(
            min(_hash_shingle(shingle, salt) for shingle in shingles) for salt in self._salts
        )

    def similarity(self, left: Sequence[int], right: Sequence[int]) -> float:
        """Estimate Jaccard similarity from two equal-length signatures."""
        if len(left) != len(right):
            raise ValueError("signatures must have equal length")
        if not left:
            return 1.0
        return sum(a == b for a, b in zip(left, right)) / len(left)

    def deduplicate(self, documents: Iterable[str]) -> DedupResult:
        """Remove documents whose MinHash similarity reaches `threshold`."""
        kept: list[str] = []
        kept_indices: list[int] = []
        duplicate_indices: list[int] = []
        signatures: list[tuple[int, ...]] = []
        for index, document in enumerate(documents):
            signature = self.signature(document)
            if any(self.similarity(signature, prior) >= self.threshold for prior in signatures):
                duplicate_indices.append(index)
                continue
            kept.append(document)
            kept_indices.append(index)
            signatures.append(signature)
        return DedupResult(kept, kept_indices, duplicate_indices)
