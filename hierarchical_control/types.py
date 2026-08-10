from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Usage:
    extra_tokens: int = 0
    extra_calls: int = 0

    def add(self, tokens: int, calls: int = 1) -> None:
        if tokens < 0 or calls < 0:
            raise ValueError("Usage increments must be non-negative")
        self.extra_tokens += int(tokens)
        self.extra_calls += int(calls)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Usage":
        return cls(int(payload.get("extra_tokens", 0)), int(payload.get("extra_calls", 0)))


@dataclass
class AgentState:
    query: str
    current_answer: str
    history: list[dict[str, str]]
    role: str = "critic"
    round_index: int = 0
    collaboration_steps: int = 0
    usage: Usage = field(default_factory=Usage)
    terminated: bool = False
    termination_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "AgentState":
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentState":
        data = copy.deepcopy(payload)
        data["usage"] = Usage.from_dict(data.get("usage", {}))
        return cls(**data)


@dataclass(frozen=True)
class CompletionResult:
    content: str
    completion_tokens: int
    prompt_tokens: int | None = None
    total_tokens: int | None = None
    usage_reported: bool = False
