"""The agent loop: generate, parse tool calls, execute, feed results back.

This is the harness half of "Claude Code type". The model's job is to decide
what to call; the loop's job is to make every call safe, bounded, observable
and recoverable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .context import Context
from .parser import ToolCall, ToolCallParser
from .tools import PermissionError, WorkspaceTools

MAX_TOOL_OUTPUT_CHARS = 8000


class LanguageModel(Protocol):
    """Minimal contract the loop needs from a generator."""

    def generate(
        self, prompt: str, *, stop: tuple[str, ...], max_new_tokens: int
    ) -> str: ...


@dataclass
class LoopLimits:
    max_iterations: int = 12
    max_tool_output_chars: int = MAX_TOOL_OUTPUT_CHARS
    max_new_tokens: int = 1024


@dataclass
class StepRecord:
    iteration: int
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    results: list[dict[str, str]] = field(default_factory=list)


def truncate(text: str, limit: int) -> str:
    """Cap tool output and say so, rather than silently dropping the tail."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return f"{text[:limit]}\n... [{dropped} characters truncated]"


class AgentLoop:
    def __init__(
        self,
        model: LanguageModel,
        tools: WorkspaceTools,
        limits: LoopLimits | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.limits = limits or LoopLimits()
        if (
            self.limits.max_iterations < 1
            or self.limits.max_tool_output_chars < 1
            or self.limits.max_new_tokens < 1
        ):
            raise ValueError("loop limits must be positive")
        self.transcript: list[StepRecord] = []
        self._handlers = {
            "read": tools.read,
            "write": tools.write,
            "edit": tools.edit,
            "diff": tools.diff,
            "glob": tools.glob,
            "grep": tools.grep,
            "bash": tools.bash,
            "git": tools.git,
        }

    def _invoke(self, call: ToolCall) -> dict[str, str]:
        handler = self._handlers.get(call.name)
        if handler is None:
            return {"call": call.name, "result": f"unknown tool: {call.name}", "error": "true"}
        try:
            value = handler(**call.arguments)
        except (TypeError, ValueError, PermissionError, OSError) as exc:
            return {
                "call": call.name,
                "result": f"{type(exc).__name__}: {exc}",
                "error": "true",
            }
        text = value if isinstance(value, str) else str(value)
        return {"call": call.name, "result": truncate(text, self.limits.max_tool_output_chars)}

    def run(self, task: str, render: Callable[[Context], str]) -> str:
        context = Context()
        context.append("user", task)
        stop = ("</tool_call>", "")

        for iteration in range(self.limits.max_iterations):
            text = self.model.generate(
                render(context), stop=stop, max_new_tokens=self.limits.max_new_tokens
            )
            parser = ToolCallParser()
            looks_like_call = text.lstrip().startswith("{")
            try:
                calls = parser.feed(text)
                if looks_like_call:
                    parser.finish()
            except ValueError as exc:
                context.append("tool", f"malformed tool call: {exc}")
                self.transcript.append(StepRecord(iteration, text))
                continue

            if not calls:
                self.transcript.append(StepRecord(iteration, text))
                return text

            results = [self._invoke(call) for call in calls]
            context.append("assistant", text)
            for call, result in zip(calls, results, strict=True):
                context.append("tool", f"{call.name} -> {result['result']}")
            self.transcript.append(StepRecord(iteration, text, calls, results))

        return "stopped: iteration budget exhausted"