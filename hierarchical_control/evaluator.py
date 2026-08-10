from __future__ import annotations

import re
from typing import Any, Protocol


class Evaluator(Protocol):
    def evaluate(self, query: str, answer: str, example: dict[str, Any]) -> tuple[float, bool]: ...


class MockEvaluator:
    """Toy evaluator paired exclusively with MockBackend."""

    mock_only = True

    def evaluate(self, query: str, answer: str, example: dict[str, Any]) -> tuple[float, bool]:
        required = int(example.get("difficulty", 1))
        match = re.search(r"(?:^|\s)quality=(\d+)", answer)
        achieved = int(match.group(1)) if match else 0
        if required <= 0:
            return 1.0, True
        score = min(float(achieved) / float(required), 1.0)
        return score, achieved >= required


class ExactMatchEvaluator:
    """Minimal non-mock evaluator; replace with a benchmark-specific evaluator."""

    mock_only = False

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split())

    def evaluate(self, query: str, answer: str, example: dict[str, Any]) -> tuple[float, bool]:
        if "expected_answer" not in example:
            raise ValueError("ExactMatchEvaluator requires an 'expected_answer' field")
        success = self._normalize(answer) == self._normalize(str(example["expected_answer"]))
        return float(success), success
