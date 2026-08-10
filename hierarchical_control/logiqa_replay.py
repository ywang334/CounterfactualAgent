from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backend import LLMBackend
from .io_utils import read_jsonl
from .logiqa_audit import tolerant_final_answer
from .logiqa_pilot import ANSWER_LETTERS, LogiQAExample, extract_final_answer, load_logiqa_dev
from .logiqa_prompts import (
    PROMPT_VERSIONS,
    build_versioned_critic_messages,
    build_versioned_refiner_messages,
)
from .types import CompletionResult


SAFE_KEEP_REVIEW = """QUESTION_POLARITY: UNKNOWN
CONSTRAINT_AUDIT: Critic protocol validation failed; preserve the Solver answer.
DECISIVE_ERROR: NONE
ALTERNATIVE_VERIFICATION: NONE
VERDICT: KEEP
PROPOSED_ANSWER: NONE"""


@dataclass(frozen=True)
class ReplaySettings:
    source_summary_path: Path
    data_path: Path | None
    requested_limit: int
    seed: int
    base_url: str
    model: str
    temperature: float
    critic_max_tokens: int
    refiner_max_tokens: int
    extra_body: dict[str, Any]


@dataclass(frozen=True)
class CriticProtocol:
    parsed_verdict: str | None
    proposed_answer: str | None
    parse_failure: bool
    effective_verdict: str
    effective_proposed_answer: str | None
    review_for_refiner: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _single_field(output: str, field: str) -> str | None:
    matches = re.findall(
        rf"^{re.escape(field)}:[ \t]*(.*?)[ \t]*$",
        output,
        flags=re.MULTILINE,
    )
    return matches[0] if len(matches) == 1 else None


def parse_critic_protocol(output: str) -> CriticProtocol:
    verdict = _single_field(output, "VERDICT")
    proposed = _single_field(output, "PROPOSED_ANSWER")
    valid = (
        (verdict == "KEEP" and proposed == "NONE")
        or (verdict == "REVISE" and proposed in ANSWER_LETTERS)
    )
    if not valid:
        return CriticProtocol(
            parsed_verdict=verdict,
            proposed_answer=proposed,
            parse_failure=True,
            effective_verdict="KEEP",
            effective_proposed_answer=None,
            review_for_refiner=SAFE_KEEP_REVIEW,
        )
    return CriticProtocol(
        parsed_verdict=verdict,
        proposed_answer=None if proposed == "NONE" else proposed,
        parse_failure=False,
        effective_verdict=verdict,
        effective_proposed_answer=None if proposed == "NONE" else proposed,
        review_for_refiner=output,
    )


def parse_refiner_protocol(output: str) -> dict[str, str | None]:
    validation = _single_field(output, "CRITIQUE_VALIDATION")
    decision = _single_field(output, "REFINEMENT_DECISION")
    if validation not in {"VALID", "INVALID", "NOT_APPLICABLE"}:
        validation = None
    if decision not in {"KEEP_ORIGINAL", "APPLY_REVISION"}:
        decision = None
    return {
        "critique_validation": validation,
        "refinement_decision": decision,
    }


def load_logiqa_replay_settings(predictions_path: str | Path) -> ReplaySettings:
    source = Path(predictions_path)
    summary_path = source.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Pilot summary is required to reuse backend and sampling settings: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mock_only") is not False:
        raise ValueError("Prompt replay requires a real, mock_only=false Pilot source")
    backend = summary.get("backend")
    caps = summary.get("generation_caps")
    extra_body = summary.get("request_extra_body")
    if not isinstance(backend, dict) or not isinstance(caps, dict):
        raise ValueError("Pilot summary is missing backend or generation_caps settings")
    base_url = backend.get("base_url")
    model = backend.get("model")
    if not isinstance(base_url, str) or not base_url or not isinstance(model, str) or not model:
        raise ValueError("Pilot summary has invalid backend base_url or model")
    critic_cap = caps.get("critic")
    refiner_cap = caps.get("refiner")
    if (
        isinstance(critic_cap, bool)
        or not isinstance(critic_cap, int)
        or critic_cap <= 0
        or isinstance(refiner_cap, bool)
        or not isinstance(refiner_cap, int)
        or refiner_cap <= 0
    ):
        raise ValueError("Pilot summary has invalid Critic/Refiner generation caps")
    if not isinstance(extra_body, dict):
        raise ValueError("Pilot summary request_extra_body must be an object")
    data_path_value = summary.get("data_path")
    data_path = Path(data_path_value) if isinstance(data_path_value, str) else None
    requested_limit = summary.get("requested_limit", summary.get("samples"))
    seed = summary.get("seed")
    if (
        isinstance(requested_limit, bool)
        or not isinstance(requested_limit, int)
        or requested_limit <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("Pilot summary has invalid requested_limit or seed")
    temperature = summary.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("Pilot summary has invalid temperature")
    return ReplaySettings(
        source_summary_path=summary_path.resolve(),
        data_path=data_path,
        requested_limit=requested_limit,
        seed=seed,
        base_url=base_url,
        model=model,
        temperature=float(temperature),
        critic_max_tokens=critic_cap,
        refiner_max_tokens=refiner_cap,
        extra_body=dict(extra_body),
    )


def _id_key(question_id: str | int) -> str:
    return json.dumps([type(question_id).__name__, question_id], ensure_ascii=False)


def _usage_dict(payload: Any, purpose: str) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Missing real usage for {purpose}")
    values: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Missing or invalid real {name} for {purpose}")
        values[name] = value
    if values["total_tokens"] != values["prompt_tokens"] + values["completion_tokens"]:
        raise RuntimeError(f"Inconsistent real token usage for {purpose}")
    return values


def _completion_usage(result: CompletionResult, purpose: str) -> dict[str, int]:
    if (
        not result.usage_reported
        or result.prompt_tokens is None
        or result.total_tokens is None
    ):
        raise RuntimeError(
            f"OpenAI-compatible backend did not report real token usage for {purpose}; "
            "prompt replay refuses estimated usage"
        )
    return _usage_dict(
        {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
        purpose,
    )


def _latency_value(payload: Any, purpose: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)) or payload < 0:
        raise RuntimeError(f"Missing or invalid real latency for {purpose}")
    return float(payload)


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


def _source_problem(
    row: dict[str, Any],
    examples: dict[str, LogiQAExample],
) -> tuple[dict[str, Any], str]:
    rendered = row.get("problem_and_choices")
    problem = row.get("problem")
    if isinstance(rendered, str) and rendered:
        if isinstance(problem, dict):
            return dict(problem), rendered
        return {"rendered": rendered}, rendered
    key = _id_key(row["question_id"])
    if key not in examples:
        raise ValueError(f"Cannot reconstruct problem for question {row['question_id']!r}")
    example = examples[key]
    structured = {
        "passage": example.passage,
        "question": example.question,
        "options": {
            letter: option for letter, option in zip(ANSWER_LETTERS, example.options)
        },
    }
    return structured, example.problem_text()


def _load_source_rows(
    predictions_path: Path,
    settings: ReplaySettings,
) -> list[dict[str, Any]]:
    rows = read_jsonl(predictions_path)
    if not rows:
        raise ValueError(f"Pilot predictions are empty: {predictions_path}")
    seen: set[str] = set()
    needs_reconstruction = any(
        not isinstance(row.get("problem_and_choices"), str) for row in rows
    )
    examples: dict[str, LogiQAExample] = {}
    if needs_reconstruction:
        if settings.data_path is None:
            raise ValueError("Pilot records omit original queries and summary has no data_path")
        sampled = load_logiqa_dev(
            settings.data_path,
            limit=settings.requested_limit,
            seed=settings.seed,
        )
        examples = {_id_key(example.question_id): example for example in sampled}
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Pilot prediction {index} must be an object")
        question_id = row.get("question_id")
        if isinstance(question_id, bool) or not isinstance(question_id, (str, int)):
            raise ValueError(f"Pilot prediction {index} has invalid question_id")
        key = _id_key(question_id)
        if key in seen:
            raise ValueError(f"Duplicate Pilot question_id: {question_id!r}")
        seen.add(key)
        gold = row.get("gold")
        if gold not in ANSWER_LETTERS:
            raise ValueError(f"Pilot prediction {question_id!r} has invalid gold")
        raw_outputs = row.get("raw_outputs")
        if not isinstance(raw_outputs, dict):
            raise ValueError(f"Pilot prediction {question_id!r} has no raw_outputs")
        solver_output = raw_outputs.get("solver")
        minimal_refiner_output = raw_outputs.get("refiner")
        if not isinstance(solver_output, str) or not isinstance(minimal_refiner_output, str):
            raise ValueError(
                f"Pilot prediction {question_id!r} lacks Solver or minimal_v1 Refiner output"
            )
        usage = row.get("usage")
        latency = row.get("latency_seconds")
        if not isinstance(usage, dict) or not isinstance(latency, dict):
            raise RuntimeError(f"Pilot prediction {question_id!r} lacks real cost records")
        solver_usage = _usage_dict(usage.get("solver"), f"saved Solver {question_id!r}")
        minimal_usage = _usage_dict(
            usage.get("solver_critic_refiner"),
            f"saved minimal_v1 Full {question_id!r}",
        )
        solver_latency = _latency_value(
            latency.get("solver"),
            f"saved Solver {question_id!r}",
        )
        minimal_latency = _latency_value(
            latency.get("solver_critic_refiner"),
            f"saved minimal_v1 Full {question_id!r}",
        )
        problem, rendered = _source_problem(row, examples)
        if key in examples and examples[key].gold != gold:
            raise ValueError(
                f"Gold mismatch while reconstructing question {question_id!r}"
            )
        prepared.append(
            {
                "source": row,
                "question_id": question_id,
                "gold": gold,
                "problem": problem,
                "problem_and_choices": rendered,
                "solver_output": solver_output,
                "minimal_refiner_output": minimal_refiner_output,
                "solver_usage": solver_usage,
                "minimal_usage": minimal_usage,
                "solver_latency": solver_latency,
                "minimal_latency": minimal_latency,
            }
        )
    return prepared


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_path(directory: Path, question_id: str | int) -> Path:
    digest = hashlib.sha256(_id_key(question_id).encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def _load_checkpoint(
    path: Path,
    question_id: str | int,
    prompt_version: str,
    source_predictions: str,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "question_id": question_id,
            "prompt_version": prompt_version,
            "source_predictions": source_predictions,
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if (
        checkpoint.get("question_id") != question_id
        or checkpoint.get("prompt_version") != prompt_version
        or checkpoint.get("source_predictions") != source_predictions
    ):
        raise ValueError(f"Replay checkpoint does not match current run: {path}")
    return checkpoint


def _prepare_output(
    source_predictions: Path,
    output_dir: Path,
    prompt_version: str,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    target_predictions = output_dir / "predictions.jsonl"
    if target_predictions.resolve() == source_predictions.resolve():
        raise ValueError("Replay output cannot overwrite the source minimal_v1 predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    completed: dict[str, dict[str, Any]] = {}
    if target_predictions.exists():
        for row in read_jsonl(target_predictions):
            if (
                row.get("prompt_version") != prompt_version
                or row.get("prompt_development") is not True
                or row.get("source_predictions") != str(source_predictions.resolve())
            ):
                raise ValueError(
                    "Existing replay predictions belong to a different run; refusing overwrite"
                )
            key = _id_key(row.get("question_id"))
            if key in completed:
                raise ValueError(f"Duplicate completed replay sample: {row.get('question_id')!r}")
            completed[key] = row
    elif summary_path.exists() or report_path.exists():
        raise ValueError(
            "Output directory contains non-resumable artifacts; refusing to overwrite them"
        )
    return target_predictions, completed


def _tolerant_payload(output: str) -> dict[str, Any]:
    parsed = tolerant_final_answer(output)
    return {
        "answer": parsed.answer,
        "match_count": parsed.match_count,
        "conflict": parsed.conflict,
        "matches": list(parsed.matches),
    }


def _transition(solver_answer: str | None, full_answer: str | None, gold: str) -> str:
    solver_correct = solver_answer == gold
    full_correct = full_answer == gold
    if solver_correct and full_correct:
        return "correct_to_correct"
    if solver_correct:
        return "correct_to_wrong"
    if full_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _build_prediction(
    item: dict[str, Any],
    prompt_version: str,
    source_predictions: str,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    critic = checkpoint["critic"]
    refiner = checkpoint["refiner"]
    solver_strict = extract_final_answer(item["solver_output"])
    solver_tolerant = _tolerant_payload(item["solver_output"])
    refiner_strict = extract_final_answer(refiner["raw_output"])
    refiner_tolerant = _tolerant_payload(refiner["raw_output"])
    critic_usage = _usage_dict(
        critic["usage"],
        f"checkpoint Critic {item['question_id']!r}",
    )
    refiner_usage = _usage_dict(
        refiner["usage"],
        f"checkpoint Refiner {item['question_id']!r}",
    )
    collaboration_usage = _add_usage(critic_usage, refiner_usage)
    complete_usage = _add_usage(item["solver_usage"], collaboration_usage)
    critic_latency = _latency_value(
        critic["latency_seconds"],
        f"checkpoint Critic {item['question_id']!r}",
    )
    refiner_latency = _latency_value(
        refiner["latency_seconds"],
        f"checkpoint Refiner {item['question_id']!r}",
    )
    collaboration_latency = critic_latency + refiner_latency
    complete_latency = item["solver_latency"] + collaboration_latency
    protocol = critic["protocol"]
    effective_verdict = protocol["effective_verdict"]
    protocol_violation = (
        effective_verdict == "KEEP"
        and solver_tolerant["answer"] is not None
        and refiner_tolerant["answer"] is not None
        and solver_tolerant["answer"] != refiner_tolerant["answer"]
    )
    transition = _transition(
        solver_tolerant["answer"],
        refiner_tolerant["answer"],
        item["gold"],
    )
    return {
        "question_id": item["question_id"],
        "gold": item["gold"],
        "prompt_version": prompt_version,
        "prompt_development": True,
        "deployable_result": False,
        "solver_reused": True,
        "solver_called": False,
        "mock_only": False,
        "source_predictions": source_predictions,
        "problem": item["problem"],
        "problem_and_choices": item["problem_and_choices"],
        "solver": {
            "raw_output": item["solver_output"],
            "strict_answer": solver_strict,
            "tolerant": solver_tolerant,
            "usage": item["solver_usage"],
            "calls_in_replay": 0,
            "recorded_latency_seconds": item["solver_latency"],
        },
        "critic": {
            "raw_output": critic["raw_output"],
            "parsed_verdict": protocol["parsed_verdict"],
            "proposed_answer": protocol["proposed_answer"],
            "effective_verdict": effective_verdict,
            "effective_proposed_answer": protocol["effective_proposed_answer"],
            "review_sent_to_refiner": protocol["review_for_refiner"],
            "usage": critic_usage,
            "calls": 1,
            "latency_seconds": critic_latency,
        },
        "refiner": {
            "raw_output": refiner["raw_output"],
            "critique_validation": refiner["protocol"]["critique_validation"],
            "refinement_decision": refiner["protocol"]["refinement_decision"],
            "strict_answer": refiner_strict,
            "tolerant": refiner_tolerant,
            "usage": refiner_usage,
            "calls": 1,
            "latency_seconds": refiner_latency,
        },
        "critic_parse_failure": bool(protocol["parse_failure"]),
        "refiner_protocol_violation": protocol_violation,
        "strict": {
            "solver_answer": solver_strict,
            "full_answer": refiner_strict,
            "solver_parse_failure": solver_strict is None,
            "full_parse_failure": refiner_strict is None,
        },
        "tolerant": {
            "solver_answer": solver_tolerant["answer"],
            "full_answer": refiner_tolerant["answer"],
            "solver_parse_failure": solver_tolerant["answer"] is None,
            "full_parse_failure": refiner_tolerant["answer"] is None,
        },
        "transition": transition,
        "usage": {
            "solver_reused": item["solver_usage"],
            "critic": critic_usage,
            "refiner": refiner_usage,
            "replay_collaboration": collaboration_usage,
            "complete_v2": complete_usage,
        },
        "calls": {
            "solver_in_replay": 0,
            "critic": 1,
            "refiner": 1,
            "replay_actual": 2,
            "complete_workflow_equivalent": 3,
        },
        "latency_seconds": {
            "solver_recorded": item["solver_latency"],
            "critic": critic_latency,
            "refiner": refiner_latency,
            "replay_actual": collaboration_latency,
            "complete_v2": complete_latency,
        },
    }


def _answer_pair(output: str) -> tuple[str | None, str | None]:
    return extract_final_answer(output), tolerant_final_answer(output).answer


def _answer_metrics(
    records: list[tuple[str | int, str, str | None, str | None]],
) -> dict[str, Any]:
    count = len(records)
    correct_ids = [question_id for question_id, gold, _, answer in records if answer == gold]
    parse_failure_ids = [
        question_id for question_id, _, _, answer in records if answer is None
    ]
    return {
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / count,
        "parse_failures": len(parse_failure_ids),
        "parse_failure_ids": parse_failure_ids,
        "correct_ids": correct_ids,
    }


def _strategy_metrics(
    ids_gold_answers: list[
        tuple[str | int, str, str | None, str | None, str | None, str | None]
    ],
) -> dict[str, Any]:
    strict_records = [
        (question_id, gold, solver_strict, full_strict)
        for question_id, gold, solver_strict, _, full_strict, _ in ids_gold_answers
    ]
    tolerant_records = [
        (question_id, gold, solver_tolerant, full_tolerant)
        for question_id, gold, _, solver_tolerant, _, full_tolerant in ids_gold_answers
    ]
    strict = _answer_metrics(strict_records)
    tolerant = _answer_metrics(tolerant_records)
    transitions = {
        name: {"count": 0, "sample_ids": []}
        for name in (
            "correct_to_correct",
            "correct_to_wrong",
            "wrong_to_correct",
            "wrong_to_wrong",
        )
    }
    for question_id, gold, solver_answer, full_answer in tolerant_records:
        name = _transition(solver_answer, full_answer, gold)
        transitions[name]["count"] += 1
        transitions[name]["sample_ids"].append(question_id)
    corrected_ids = transitions["wrong_to_correct"]["sample_ids"]
    degraded_ids = transitions["correct_to_wrong"]["sample_ids"]
    unchanged_ids = (
        transitions["correct_to_correct"]["sample_ids"]
        + transitions["wrong_to_wrong"]["sample_ids"]
    )
    return {
        "strict": strict,
        "tolerant": tolerant,
        "transitions": transitions,
        "corrected": len(corrected_ids),
        "corrected_ids": corrected_ids,
        "degraded": len(degraded_ids),
        "degraded_ids": degraded_ids,
        "unchanged": len(unchanged_ids),
        "unchanged_ids": unchanged_ids,
    }


def _aggregate_cost(
    usages: list[dict[str, int]],
    latencies: list[float],
    calls_per_sample: int,
) -> dict[str, Any]:
    count = len(usages)
    totals = {
        name: sum(usage[name] for usage in usages)
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "samples": count,
        "total_usage": totals,
        "average_usage": {
            name: totals[name] / count for name in totals
        },
        "total_calls": calls_per_sample * count,
        "average_calls": float(calls_per_sample),
        "total_latency_seconds": sum(latencies),
        "average_latency_seconds": sum(latencies) / count,
    }


def _build_summary(
    source_items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    settings: ReplaySettings,
    source_predictions: str,
    prompt_version: str,
) -> dict[str, Any]:
    solver_pairs = []
    minimal_pairs = []
    structured_pairs = []
    for item in source_items:
        solver_strict, solver_tolerant = _answer_pair(item["solver_output"])
        minimal_strict, minimal_tolerant = _answer_pair(item["minimal_refiner_output"])
        solver_pairs.append(
            (
                item["question_id"],
                item["gold"],
                solver_strict,
                solver_tolerant,
                solver_strict,
                solver_tolerant,
            )
        )
        minimal_pairs.append(
            (
                item["question_id"],
                item["gold"],
                solver_strict,
                solver_tolerant,
                minimal_strict,
                minimal_tolerant,
            )
        )
    for row in predictions:
        structured_pairs.append(
            (
                row["question_id"],
                row["gold"],
                row["strict"]["solver_answer"],
                row["tolerant"]["solver_answer"],
                row["strict"]["full_answer"],
                row["tolerant"]["full_answer"],
            )
        )
    solver_metrics = _strategy_metrics(solver_pairs)
    minimal_metrics = _strategy_metrics(minimal_pairs)
    structured_metrics = _strategy_metrics(structured_pairs)
    effective_keep = sum(
        row["critic"]["effective_verdict"] == "KEEP" for row in predictions
    )
    effective_revise = sum(
        row["critic"]["effective_verdict"] == "REVISE" for row in predictions
    )
    parse_failures = sum(row["critic_parse_failure"] for row in predictions)
    valid_keep = sum(
        (not row["critic_parse_failure"])
        and row["critic"]["parsed_verdict"] == "KEEP"
        for row in predictions
    )
    valid_revise = sum(
        (not row["critic_parse_failure"])
        and row["critic"]["parsed_verdict"] == "REVISE"
        for row in predictions
    )
    false_revise_ids = [
        row["question_id"]
        for row in predictions
        if not row["critic_parse_failure"]
        and row["critic"]["parsed_verdict"] == "REVISE"
        and row["tolerant"]["solver_answer"] == row["gold"]
    ]
    helpful_revise_ids = [
        row["question_id"]
        for row in predictions
        if not row["critic_parse_failure"]
        and row["critic"]["parsed_verdict"] == "REVISE"
        and row["tolerant"]["solver_answer"] != row["gold"]
        and row["tolerant"]["full_answer"] == row["gold"]
    ]
    revise_refiner_keep_ids = [
        row["question_id"]
        for row in predictions
        if not row["critic_parse_failure"]
        and row["critic"]["parsed_verdict"] == "REVISE"
        and row["refiner"]["refinement_decision"] == "KEEP_ORIGINAL"
    ]
    protocol_violation_ids = [
        row["question_id"] for row in predictions if row["refiner_protocol_violation"]
    ]
    source_solver_usages = [item["solver_usage"] for item in source_items]
    source_solver_latencies = [item["solver_latency"] for item in source_items]
    source_minimal_usages = [item["minimal_usage"] for item in source_items]
    source_minimal_latencies = [item["minimal_latency"] for item in source_items]
    critic_usages = [row["usage"]["critic"] for row in predictions]
    critic_latencies = [row["latency_seconds"]["critic"] for row in predictions]
    refiner_usages = [row["usage"]["refiner"] for row in predictions]
    refiner_latencies = [row["latency_seconds"]["refiner"] for row in predictions]
    replay_usages = [row["usage"]["replay_collaboration"] for row in predictions]
    replay_latencies = [row["latency_seconds"]["replay_actual"] for row in predictions]
    complete_usages = [row["usage"]["complete_v2"] for row in predictions]
    complete_latencies = [row["latency_seconds"]["complete_v2"] for row in predictions]
    costs = {
        "solver_only": _aggregate_cost(source_solver_usages, source_solver_latencies, 1),
        "minimal_v1_full": _aggregate_cost(
            source_minimal_usages,
            source_minimal_latencies,
            3,
        ),
        "structured_v2_critic": _aggregate_cost(critic_usages, critic_latencies, 1),
        "structured_v2_refiner": _aggregate_cost(refiner_usages, refiner_latencies, 1),
        "structured_v2_replay_actual": _aggregate_cost(
            replay_usages,
            replay_latencies,
            2,
        ),
        "structured_v2_complete": _aggregate_cost(
            complete_usages,
            complete_latencies,
            3,
        ),
    }
    comparison = []
    for name, metrics, cost_key in (
        ("Solver Only", solver_metrics, "solver_only"),
        ("Full minimal_v1", minimal_metrics, "minimal_v1_full"),
        ("Full structured_v2", structured_metrics, "structured_v2_complete"),
    ):
        comparison.append(
            {
                "strategy": name,
                "strict_accuracy": metrics["strict"]["accuracy"],
                "strict_parse_failures": metrics["strict"]["parse_failures"],
                "tolerant_accuracy": metrics["tolerant"]["accuracy"],
                "tolerant_parse_failures": metrics["tolerant"]["parse_failures"],
                "corrected": metrics["corrected"],
                "degraded": metrics["degraded"],
                "unchanged": metrics["unchanged"],
                "average_total_tokens": costs[cost_key]["average_usage"]["total_tokens"],
                "average_calls": costs[cost_key]["average_calls"],
                "average_latency_seconds": costs[cost_key]["average_latency_seconds"],
            }
        )
    return {
        "prompt_version": prompt_version,
        "prompt_development": True,
        "deployable_result": False,
        "solver_reused": True,
        "solver_called": False,
        "mock_only": False,
        "samples": len(predictions),
        "source_predictions": source_predictions,
        "source_summary": str(settings.source_summary_path),
        "development_set_warning": (
            "These 50 samples are a prompt development set and are not a final test result."
        ),
        "backend": {
            "base_url": settings.base_url,
            "model": settings.model,
            "temperature": settings.temperature,
            "extra_body": settings.extra_body,
        },
        "generation_caps": {
            "critic": settings.critic_max_tokens,
            "refiner": settings.refiner_max_tokens,
        },
        "strategies": {
            "solver_only": solver_metrics,
            "minimal_v1_full": minimal_metrics,
            "structured_v2_full": structured_metrics,
        },
        "critic": {
            "keep": valid_keep,
            "revise": valid_revise,
            "effective_keep": effective_keep,
            "effective_revise": effective_revise,
            "parse_failures": parse_failures,
            "false_revise": len(false_revise_ids),
            "false_revise_ids": false_revise_ids,
            "helpful_revise": len(helpful_revise_ids),
            "helpful_revise_ids": helpful_revise_ids,
            "revise_then_refiner_keep_original": len(revise_refiner_keep_ids),
            "revise_then_refiner_keep_original_ids": revise_refiner_keep_ids,
        },
        "refiner_protocol_violation": {
            "count": len(protocol_violation_ids),
            "sample_ids": protocol_violation_ids,
        },
        "costs": costs,
        "comparison": comparison,
    }


def _build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# LogiQA prompt replay: " + summary["prompt_version"],
        "",
        "**Prompt development set only. deployable_result=false; this is not a final test result.**",
        "",
        f"Samples: {summary['samples']}",
        "Solver reused: true; Solver called: false",
        "",
        "## Strategy comparison",
        "",
        "| Strategy | Strict acc. | Strict parse failures | Tolerant acc. | Tolerant parse failures | Corrected | Degraded | Unchanged | Avg tokens | Avg calls | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparison"]:
        lines.append(
            f"| {row['strategy']} | {row['strict_accuracy']:.4f} | "
            f"{row['strict_parse_failures']} | {row['tolerant_accuracy']:.4f} | "
            f"{row['tolerant_parse_failures']} | {row['corrected']} | "
            f"{row['degraded']} | {row['unchanged']} | "
            f"{row['average_total_tokens']:.2f} | {row['average_calls']:.2f} | "
            f"{row['average_latency_seconds']:.4f} |"
        )
    critic = summary["critic"]
    lines.extend(
        [
            "",
            "## Critic and Refiner protocol",
            "",
            f"- Valid KEEP: {critic['keep']}",
            f"- Valid REVISE: {critic['revise']}",
            f"- Critic parse failures: {critic['parse_failures']}",
            f"- Effective KEEP after safe fallback: {critic['effective_keep']}",
            f"- false_revise: {critic['false_revise']}",
            f"- helpful_revise: {critic['helpful_revise']}",
            (
                "- REVISE followed by Refiner KEEP_ORIGINAL: "
                f"{critic['revise_then_refiner_keep_original']}"
            ),
            (
                "- Refiner protocol violations: "
                f"{summary['refiner_protocol_violation']['count']}"
            ),
            "",
            "## Real token and latency cost",
            "",
            "| Component | Prompt tokens | Completion tokens | Total tokens | Calls | Wall latency (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    cost_labels = (
        ("structured_v2_critic", "Critic"),
        ("structured_v2_refiner", "Refiner"),
        ("structured_v2_replay_actual", "Replay actual (Critic + Refiner)"),
        ("structured_v2_complete", "Complete structured_v2 including saved Solver"),
    )
    for key, label in cost_labels:
        cost = summary["costs"][key]
        usage = cost["total_usage"]
        lines.append(
            f"| {label} | {usage['prompt_tokens']} | {usage['completion_tokens']} | "
            f"{usage['total_tokens']} | {cost['total_calls']} | "
            f"{cost['total_latency_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "All usage values are reported by the real OpenAI-compatible backend. "
            "No missing token usage was estimated.",
            "",
        ]
    )
    return "\n".join(lines)


def run_logiqa_prompt_replay(
    predictions_path: str | Path,
    prompt_version: str,
    output_dir: str | Path,
    backend: LLMBackend,
) -> dict[str, Any]:
    if prompt_version not in PROMPT_VERSIONS:
        raise ValueError(
            f"Unknown LogiQA prompt version {prompt_version!r}; expected {PROMPT_VERSIONS}"
        )
    if bool(getattr(backend, "mock_only", True)):
        raise ValueError(
            "replay-logiqa-prompts requires a real-compatible backend; Mock is forbidden"
        )
    source_predictions_path = Path(predictions_path).resolve()
    settings = load_logiqa_replay_settings(source_predictions_path)
    source_items = _load_source_rows(source_predictions_path, settings)
    target = Path(output_dir)
    target_predictions, completed = _prepare_output(
        source_predictions_path,
        target,
        prompt_version,
    )
    source_predictions_text = str(source_predictions_path)
    checkpoint_dir = target / ".replay_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for item in source_items:
        key = _id_key(item["question_id"])
        if key in completed:
            continue
        checkpoint_path = _checkpoint_path(checkpoint_dir, item["question_id"])
        checkpoint = _load_checkpoint(
            checkpoint_path,
            item["question_id"],
            prompt_version,
            source_predictions_text,
        )
        if "critic" not in checkpoint:
            critic_result, critic_latency = _timed_complete(
                backend,
                build_versioned_critic_messages(
                    item["problem_and_choices"],
                    item["solver_output"],
                    prompt_version,
                ),
                settings.critic_max_tokens,
                "critic",
                item["question_id"],
            )
            critic_usage = _completion_usage(
                critic_result,
                f"Critic question {item['question_id']!r}",
            )
            protocol = parse_critic_protocol(critic_result.content)
            checkpoint["critic"] = {
                "raw_output": critic_result.content,
                "protocol": protocol.to_dict(),
                "usage": critic_usage,
                "latency_seconds": critic_latency,
            }
            _write_json_atomic(checkpoint_path, checkpoint)
        if "refiner" not in checkpoint:
            critic_review = checkpoint["critic"]["protocol"]["review_for_refiner"]
            refiner_result, refiner_latency = _timed_complete(
                backend,
                build_versioned_refiner_messages(
                    item["problem_and_choices"],
                    item["solver_output"],
                    critic_review,
                    prompt_version,
                ),
                settings.refiner_max_tokens,
                "refiner",
                item["question_id"],
            )
            refiner_usage = _completion_usage(
                refiner_result,
                f"Refiner question {item['question_id']!r}",
            )
            checkpoint["refiner"] = {
                "raw_output": refiner_result.content,
                "protocol": parse_refiner_protocol(refiner_result.content),
                "usage": refiner_usage,
                "latency_seconds": refiner_latency,
            }
            _write_json_atomic(checkpoint_path, checkpoint)
        prediction = _build_prediction(
            item,
            prompt_version,
            source_predictions_text,
            checkpoint,
        )
        _append_jsonl(target_predictions, prediction)
        completed[key] = prediction
        checkpoint_path.unlink()
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    ordered_predictions = [
        completed[_id_key(item["question_id"])] for item in source_items
    ]
    if len(ordered_predictions) != len(source_items):
        raise RuntimeError("Replay did not complete every source sample")
    summary = _build_summary(
        source_items,
        ordered_predictions,
        settings,
        source_predictions_text,
        prompt_version,
    )
    _write_json_atomic(target / "summary.json", summary)
    _write_text_atomic(target / "report.md", _build_report(summary))
    return summary
