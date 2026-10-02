"""Bounded conversation state and deterministic context compaction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Context:
    messages: list[dict[str, str]] = field(default_factory=list)
    max_messages: int = 40

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.compact()

    def compact(self) -> None:
        if len(self.messages) <= self.max_messages:
            return
        keep = max(1, self.max_messages // 2)
        summary = "\n".join(
            f"{message['role']}: {message['content'][:300]}" for message in self.messages[:-keep]
        )
        self.messages = [{"role": "system", "content": "Compacted context:\n" + summary}] + self.messages[-keep:]
