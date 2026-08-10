from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import LLMBackend
from .config import ExperimentConfig
from .io_utils import write_jsonl
from .types import CompletionResult


ANSWER_LETTERS = ("A", "B", "C", "D")
FINAL_ANSWER_PATTERN = re.compile(r"FINAL_ANSWER: ([A-D])")


@dataclass(frozen=True)
class LogiQAExample:
    question_id: str | int
    passage: str
    question: str
    options: tuple[str, str, str, str]
    gold: str

    def problem_text(self) -> str:
        options = "\n".join(f"{letter}. {option}" for letter, option in zip(ANSWER_LETTERS, self.options))
        return f"Passage:\n{self.passage}\n\nQuestion:\n{self.question}\n\nOptions:\n{options}"


def _parse_record(payload: dict[str, Any], line_number: int) -> LogiQAExample:
    required = {"id", "answer", "text", "question", "options"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"LogiQA line {line_number} is missing fields: {sorted(missing)}")
    answer = payload["answer"]
    if isinstance(answer, bool) or not isinstance(answer, int) or answer not in range(4):
        raise ValueError(f"LogiQA line {line_number} has invalid integer answer: {answer!r}")
    options = payload["options"]
    if not isinstance(options, list) or len(options) != 4 or not all(isinstance(x, str) for x in options):
        raise ValueError(f"LogiQA line {line_number} must contain exactly four string options")
    if not all(isinstance(payload[name], str) for name in ("text", "question")):
        raise ValueError(f"LogiQA line {line_number} has non-string text or question")
    return LogiQAExample(
        question_id=payload["id"],
        passage=payload["text"],
        question=payload["question"],
        options=tuple(options),  # type: ignore[arg-type]
        gold=ANSWER_LETTERS[answer],
    )


def load_logiqa_dev(
    data_path: str | Path,
    limit: int = 50,
    seed: int = 20260810,
) -> list[LogiQAExample]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"LogiQA dev file not found: {path}")
    examples: list[LogiQAExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"LogiQA line {line_number} must be a JSON object")
            examples.append(_parse_record(payload, line_number))
    if not examples:
        raise ValueError(f"LogiQA dev file is empty: {path}")
    count = min(limit, len(examples))
    indices = random.Random(seed).sample(range(len(examples)), count)
    return [examples[index] for index in indices]


def extract_final_answer(output: str) -> str | None:
    lines = output.rstrip().splitlines()
    if not lines:
        return None
    match = FINAL_ANSWER_PATTERN.fullmatch(lines[-1])
    return match.group(1) if match else None


def build_solver_messages(problem: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a logical reasoning Solver. Analyze the passage and options carefully, but "
                "keep the entire response under 220 words and do not restate the full problem. You "
                "MUST reserve the final line for the answer. The final non-empty line must contain "
                "exactly FINAL_ANSWER: X, where X is one uppercase letter A, B, C, or D. Do not put "
                "Markdown, bold markers, backticks, bullets, or punctuation on that final line."
            ),
        },
        {"role": "user", "content": problem},
    ]


def build_critic_messages(problem: str, solver_output: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a logical reasoning Critic. Inspect the proposed solution for invalid "
                "inferences, missed constraints, or option-selection errors. Give actionable analysis "
                "to a Refiner in under 180 words; do not restate the full problem. Do not assume "
                "access to any answer key."
            ),
        },
        {
            "role": "user",
            "content": f"{problem}\n\nSolver output:\n{solver_output}",
        },
    ]


def build_refiner_messages(
    problem: str,
    solver_output: str,
    critic_output: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a logical reasoning Refiner. Re-solve the problem using the Solver output "
                "and Critic feedback, but keep the entire response under 220 words and do not restate "
                "the full problem. You MUST reserve the final line for the answer. The final non-empty "
                "line must contain exactly FINAL_ANSWER: X, where X is one uppercase letter A, B, C, "
                "or D. Do not put Markdown, bold markers, backticks, bullets, or punctuation on that "
                "final line."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{problem}\n\nSolver output:\n{solver_output}"
                f"\n\nCritic feedback:\n{critic_output}"
            ),
        },
    ]


def _usage(result: CompletionResult, purpose: str) -> dict[str, int]:
    if not result.usage_reported or result.prompt_tokens is None or result.total_tokens is None:
        raise RuntimeError(
            f"OpenAI-compatible backend did not report real token usage for {purpose}; "
            "Pilot refuses estimated usage"
        )
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
    }


def _add_usage(*parts: dict[str, int]) -> dict[str, int]:
    return {
        name: sum(part[name] for part in parts)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _timed_complete(
    backend: LLMBackend,
    messages: list[dict[str, str]],
    max_tokens: int,
    purpose: str,
    question_id: str | int,
) -> tuple[CompletionResult, float]:
    started = time.perf_counter()
    try:
        result = backend.complete(messages, max_tokens, purpose)
    except Exception as exc:
        raise RuntimeError(
            f"Real OpenAI-compatible service call failed for question {question_id!r} "
            f"during {purpose}; no Mock fallback was used: {exc}"
        ) from exc
    return result, time.perf_counter() - started


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _build_summary(
    predictions: list[dict[str, Any]],
    data_path: str | Path,
    requested_limit: int,
    seed: int,
    config: ExperimentConfig,
    backend: LLMBackend,
) -> dict[str, Any]:
    sample_count = len(predictions)
    solver_correct = sum(row["solver_correct"] for row in predictions)
    refiner_correct = sum(row["refiner_correct"] for row in predictions)
    corrected = sum((not row["solver_correct"]) and row["refiner_correct"] for row in predictions)
    degraded = sum(row["solver_correct"] and (not row["refiner_correct"]) for row in predictions)
    unchanged = sample_count - corrected - degraded
    solver_parse_failures = sum(row["solver_parse_failure"] for row in predictions)
    refiner_parse_failures = sum(row["refiner_parse_failure"] for row in predictions)
    either_parse_failures = sum(
        row["solver_parse_failure"] or row["refiner_parse_failure"] for row in predictions
    )
    solver_total_tokens = [row["usage"]["solver_only"]["total_tokens"] for row in predictions]
    collaborative_total_tokens = [
        row["usage"]["solver_critic_refiner"]["total_tokens"] for row in predictions
    ]
    extra_tokens = [
        row["usage"]["extra_collaboration"]["total_tokens"] for row in predictions
    ]
    solver_latency = [row["latency_seconds"]["solver_only"] for row in predictions]
    collaborative_latency = [
        row["latency_seconds"]["solver_critic_refiner"] for row in predictions
    ]
    return {
        "mock_only": False,
        "data_path": str(Path(data_path).resolve()),
        "requested_limit": requested_limit,
        "samples": sample_count,
        "seed": seed,
        "backend": {
            "base_url": getattr(backend, "base_url", None),
            "model": getattr(backend, "model", None),
        },
        "temperature": float(getattr(backend, "temperature", 0.0)),
        "generation_caps": dict(config.pilot_generation_caps),
        "request_extra_body": dict(config.pilot_request_extra_body),
        "solver_only": {
            "correct": solver_correct,
            "accuracy": solver_correct / sample_count,
            "parse_failures": solver_parse_failures,
            "average_total_tokens": _mean(solver_total_tokens),
            "average_calls": 1.0,
            "average_latency_seconds": _mean(solver_latency),
        },
        "solver_critic_refiner": {
            "correct": refiner_correct,
            "accuracy": refiner_correct / sample_count,
            "parse_failures": refiner_parse_failures,
            "average_total_tokens": _mean(collaborative_total_tokens),
            "average_extra_collaboration_tokens": _mean(extra_tokens),
            "average_calls": 3.0,
            "average_extra_calls": 2.0,
            "average_latency_seconds": _mean(collaborative_latency),
        },
        "transitions": {
            "corrected": corrected,
            "degraded": degraded,
            "unchanged": unchanged,
        },
        "parse_failures": {
            "solver": solver_parse_failures,
            "refiner": refiner_parse_failures,
            "either": either_parse_failures,
        },
    }


def run_logiqa_pilot(
    data_path: str | Path,
    output_dir: str | Path,
    backend: LLMBackend,
    config: ExperimentConfig,
    limit: int = 50,
    seed: int = 20260810,
) -> dict[str, Any]:
    if bool(getattr(backend, "mock_only", True)):
        raise ValueError("pilot-logiqa requires a real OpenAI-compatible backend; Mock is forbidden")
    examples = load_logiqa_dev(data_path, limit=limit, seed=seed)
    predictions: list[dict[str, Any]] = []
    caps = config.pilot_generation_caps
    for example in examples:
        problem = example.problem_text()
        solver, solver_latency = _timed_complete(
            backend, build_solver_messages(problem), caps["solver"], "solver", example.question_id
        )
        solver_usage = _usage(solver, "solver")
        critic, critic_latency = _timed_complete(
            backend,
            build_critic_messages(problem, solver.content),
            caps["critic"],
            "critic",
            example.question_id,
        )
        critic_usage = _usage(critic, "critic")
        refiner, refiner_latency = _timed_complete(
            backend,
            build_refiner_messages(problem, solver.content, critic.content),
            caps["refiner"],
            "refiner",
            example.question_id,
        )
        refiner_usage = _usage(refiner, "refiner")
        solver_answer = extract_final_answer(solver.content)
        refiner_answer = extract_final_answer(refiner.content)
        collaboration_extra = _add_usage(critic_usage, refiner_usage)
        collaboration_total = _add_usage(solver_usage, collaboration_extra)
        predictions.append(
            {
                "question_id": example.question_id,
                "gold": example.gold,
                "solver_answer": solver_answer,
                "refiner_answer": refiner_answer,
                "solver_correct": solver_answer == example.gold,
                "refiner_correct": refiner_answer == example.gold,
                "solver_parse_failure": solver_answer is None,
                "refiner_parse_failure": refiner_answer is None,
                "raw_outputs": {
                    "solver": solver.content,
                    "critic": critic.content,
                    "refiner": refiner.content,
                },
                "usage": {
                    "solver": solver_usage,
                    "critic": critic_usage,
                    "refiner": refiner_usage,
                    "solver_only": solver_usage,
                    "extra_collaboration": collaboration_extra,
                    "solver_critic_refiner": collaboration_total,
                    "calls": {"solver_only": 1, "extra_collaboration": 2, "solver_critic_refiner": 3},
                },
                "latency_seconds": {
                    "solver": solver_latency,
                    "critic": critic_latency,
                    "refiner": refiner_latency,
                    "solver_only": solver_latency,
                    "extra_collaboration": critic_latency + refiner_latency,
                    "solver_critic_refiner": solver_latency + critic_latency + refiner_latency,
                },
                "mock_only": False,
            }
        )
    summary = _build_summary(predictions, data_path, limit, seed, config, backend)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "predictions.jsonl", predictions)
    temporary = target / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target / "summary.json")
    return summary
