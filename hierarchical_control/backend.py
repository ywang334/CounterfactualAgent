from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import CompletionResult


class LLMBackend(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        purpose: str,
    ) -> CompletionResult: ...


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


@dataclass
class MockBackend:
    """Deterministic backend for offline tests only.

    It intentionally charges the requested completion cap. This makes budget
    boundary tests deterministic and must not be interpreted as realistic token
    usage. Answers carry a synthetic quality marker consumed by MockEvaluator.
    """

    calls: list[dict[str, object]] = field(default_factory=list)
    mock_only: bool = True

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        purpose: str,
    ) -> CompletionResult:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        query = _last_user_text(messages)
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        self.calls.append({"purpose": purpose, "max_tokens": max_tokens, "history_size": len(messages)})
        if purpose == "solver":
            content = f"MOCK_ANSWER id={digest} quality=0"
        elif purpose == "critic":
            quality = _extract_quality(messages[-1].get("content", "")) if messages else 0
            content = f"MOCK_CRITIQUE id={digest} observed_quality={quality}"
        elif purpose == "refiner":
            if max_tokens <= 64:
                quality = 1
            elif max_tokens <= 192:
                quality = 2
            else:
                quality = 3
            content = f"MOCK_ANSWER id={digest} quality={quality}"
        else:
            content = f"MOCK_OUTPUT id={digest} quality=0"
        return CompletionResult(content=content, completion_tokens=max_tokens)


def _extract_quality(text: str) -> int:
    match = re.search(r"(?:^|\s)quality=(\d+)", text)
    return int(match.group(1)) if match else 0


class OpenAIBackend:
    """Unified OpenAI-compatible backend for vLLM and hosted APIs."""

    mock_only = False

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional runtime
            raise RuntimeError("Install the 'openai' package to use OpenAIBackend") from exc
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.extra_body = dict(extra_body or {})

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        purpose: str,
    ) -> CompletionResult:
        request_options: dict[str, Any] = {}
        if self.extra_body:
            request_options["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=self.temperature,
            **request_options,
        )
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        usage_reported = completion_tokens is not None and prompt_tokens is not None
        if completion_tokens is None:
            # Only a fallback for non-conforming OpenAI-compatible servers.
            completion_tokens = max(1, len(content.encode("utf-8")) // 4)
        if usage_reported and total_tokens is None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        return CompletionResult(
            content=content,
            completion_tokens=int(completion_tokens),
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            usage_reported=usage_reported,
        )
