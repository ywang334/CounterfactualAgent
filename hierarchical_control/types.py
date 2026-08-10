from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Usage:
    """Additional collaboration usage.

    Hard budgets constrain ``extra_completion_tokens`` and ``extra_calls``.
    The legacy ``extra_tokens`` name meant completion tokens; it remains a
    read/property alias but is not emitted by v2 serialization.
    """

    extra_prompt_tokens: int = 0
    extra_completion_tokens: int = 0
    extra_total_tokens: int = 0
    extra_calls: int = 0

    def __post_init__(self) -> None:
        values = (
            self.extra_prompt_tokens,
            self.extra_completion_tokens,
            self.extra_total_tokens,
            self.extra_calls,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("Usage values must be non-negative integers")
        if self.extra_total_tokens != self.extra_prompt_tokens + self.extra_completion_tokens:
            raise ValueError("Usage total tokens must equal prompt tokens plus completion tokens")

    @property
    def extra_tokens(self) -> int:
        """Legacy alias: historically this field represented completion tokens."""
        return self.extra_completion_tokens

    def add(
        self,
        completion_tokens: int,
        calls: int = 1,
        *,
        prompt_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
        values = (prompt_tokens, completion_tokens, total_tokens, calls)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise ValueError("Usage increments must be non-negative")
        if total_tokens != prompt_tokens + completion_tokens:
            raise ValueError("Usage total tokens must equal prompt tokens plus completion tokens")
        self.extra_prompt_tokens += prompt_tokens
        self.extra_completion_tokens += completion_tokens
        self.extra_total_tokens += total_tokens
        self.extra_calls += calls

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Usage":
        if not isinstance(payload, dict):
            raise ValueError("Usage payload must be an object")
        completion = payload.get(
            "extra_completion_tokens",
            payload.get("extra_tokens", 0),
        )
        if (
            "extra_completion_tokens" in payload
            and "extra_tokens" in payload
            and payload["extra_completion_tokens"] != payload["extra_tokens"]
        ):
            raise ValueError(
                "Conflicting extra_completion_tokens and legacy extra_tokens values"
            )
        prompt = payload.get("extra_prompt_tokens", 0)
        total = payload.get("extra_total_tokens", prompt + completion)
        calls = payload.get("extra_calls", 0)
        return cls(
            extra_prompt_tokens=prompt,
            extra_completion_tokens=completion,
            extra_total_tokens=total,
            extra_calls=calls,
        )


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
