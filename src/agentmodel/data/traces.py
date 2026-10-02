"""Validation and filtering for generated tool-use traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class TraceResult:
    trace: dict[str, Any]
    tests_passed: bool
    valid: bool


def validate_trace(trace: Any) -> bool:
    if not isinstance(trace, dict) or not isinstance(trace.get("messages"), list):
        return False
    if not trace["messages"]:
        return False
    return all(
        isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant", "tool"}
        and isinstance(message.get("content"), str)
        for message in trace["messages"]
    )


def filter_passing_traces(traces: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep structurally valid traces whose harness reports passing tests."""
    return [
        trace
        for trace in traces
        if validate_trace(trace) and trace.get("tests_passed") is True
    ]
