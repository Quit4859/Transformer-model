"""Weighted mixture sampling over corpus sources.

Sharp mixtures over-represent small high-quality sources; flat mixtures
over-represent bulk web text. `temperature` interpolates between them: 1.0
reproduces the declared weights, values below 1 flatten toward uniform, values
above 1 sharpen.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mixture:
    """One corpus source and its relative sampling weight."""

    name: str
    weight: float = 1.0
    tokens: int = 0


@dataclass
class MixtureSampler:
    """Sample source names in proportion to ``weight ** temperature``."""

    mixtures: list[Mixture]
    seed: int = 0
    temperature: float = 1.0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if any(m.weight <= 0 for m in self.mixtures):
            raise ValueError("mixture weights must be positive")
        self._rng = random.Random(self.seed)

    def probabilities(self) -> list[float]:
        weights = [m.weight**self.temperature for m in self.mixtures]
        total = sum(weights)
        return [w / total for w in weights]

    def sample(self, n: int = 1) -> list[str]:
        if not self.mixtures:
            raise ValueError("no mixtures registered")
        if n < 1:
            raise ValueError("n must be positive")
        names = [m.name for m in self.mixtures]
        probs = self.probabilities()
        picked = self._rng.choices(names, weights=probs, k=n)
        return list(picked)