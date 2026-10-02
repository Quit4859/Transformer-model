"""Corpus quality filters used before packing training documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_SPACE = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """Normalize source/text enough to match formatting-only variants."""
    return _SPACE.sub(" ", text).strip()


def _ngrams(text: str, size: int) -> set[str]:
    normalized = normalize_for_matching(text)
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {normalized[i : i + size] for i in range(len(normalized) - size + 1)}


@dataclass(frozen=True)
class DecontaminationResult:
    documents: list[str]
    contaminated_indices: list[int]
    overlap_scores: dict[int, float]


def benchmark_decontaminate(
    documents: Iterable[str],
    benchmark_documents: Iterable[str],
    *,
    ngram_size: int = 13,
    threshold: float = 0.8,
) -> DecontaminationResult:
    """Remove corpus documents with high character-ngram overlap to a benchmark."""
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    benchmark_ngrams = set()
    for benchmark in benchmark_documents:
        benchmark_ngrams.update(_ngrams(benchmark, ngram_size))

    kept: list[str] = []
    contaminated: list[int] = []
    scores: dict[int, float] = {}
    for index, document in enumerate(documents):
        document_ngrams = _ngrams(document, ngram_size)
        score = (
            len(document_ngrams & benchmark_ngrams) / len(document_ngrams)
            if document_ngrams
            else 0.0
        )
        scores[index] = score
        if score >= threshold:
            contaminated.append(index)
        else:
            kept.append(document)
    return DecontaminationResult(kept, contaminated, scores)
