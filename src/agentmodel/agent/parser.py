"""Incremental parsing of JSON tool calls from model output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


class ToolCallParser:
    """Parse one JSON object incrementally without losing partial chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> list[ToolCall]:
        self._buffer += chunk
        calls: list[ToolCall] = []
        decoder = json.JSONDecoder()
        while self._buffer:
            self._buffer = self._buffer.lstrip()
            if not self._buffer:
                break
            try:
                value, end = decoder.raw_decode(self._buffer)
            except json.JSONDecodeError:
                break
            self._buffer = self._buffer[end:]
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                raise ValueError("tool call must be an object with a string name")
            arguments = value.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("tool call arguments must be an object")
            calls.append(ToolCall(value["name"], arguments))
        return calls

    def finish(self) -> None:
        if self._buffer.strip():
            raise ValueError("incomplete tool call")
