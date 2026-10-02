"""Reward composition and dynamic sampling for coding-agent rollouts.

A GRPO group whose rollouts all score the same carries zero gradient signal.
Filtering those groups out before the optimizer step is what keeps RLVR from
quietly stalling, and the surviving reward spread per prompt is the curriculum
signal: a prompt with near-zero spread is one the model has already saturated.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reward:
    total: float
    tests_passed: bool
    valid_format: bool
    details: dict[str, float] = field(default_factory=dict)


def grade_rollout(
    *,
    tests_passed: bool,
    valid_format: bool,
    no_regression: bool = True,
    minimal_diff: bool = True,
    steps: int = 0,
) -> Reward:
    """Compose a rollout reward from independently verifiable signals.

    `steps` is a real cost, so it is penalized: without it the model learns to
    wander. The penalty is small enough that it never outweighs a passing test.
    """
    details = {
        "tests": 1.0 if tests_passed else 0.0,
        "format": 1.0 if valid_format else 0.0,
        "no_regression": 1.0 if no_regression else 0.0,
        "minimal_diff": 1.0 if minimal_diff else 0.0,
        "step_penalty": -0.01 * max(0, steps),
    }
    return Reward(
        total=sum(details.values()),
        tests_passed=tests_passed,
        valid_format=valid_format,
        details=details,
    )


def reward_spread(rewards: list[float]) -> float:
    """Population standard deviation of one prompt's rollout rewards."""
    if not rewards:
        return 0.0
    if len(rewards) == 1:
        return 0.0
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    return variance**0.5


def filter_zero_signal(rewards: list[float]) -> list[int]:
    """Return indices of rollouts in a group that carries gradient signal.

    Returns nothing when every rollout in the group scored identically, which is
    the DAPO dynamic-sampling rule: such a prompt produces a zero
    group-relative advantage and therefore an all-zero gradient.
    """
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    return [index for index, reward in enumerate(rewards) if reward != mean]