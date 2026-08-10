from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import LLMBackend
from .io_utils import read_jsonl
from .logiqa_audit import exact_mcnemar, tolerant_final_answer
from .logiqa_pilot import (
    ANSWER_LETTERS,
    LogiQAExample,
    _parse_record,
    build_solver_messages,
    extract_final_answer,
)
from .logiqa_prompts import (
    MINIMAL_V1,
    STRUCTURED_V2,
    build_versioned_critic_messages,
    build_versioned_refiner_messages,
)
from .logiqa_replay import parse_critic_protocol, parse_refiner_protocol
from .prompt_stability_audit import (
    CRITIC_PROTOCOL_CLASSES,
    binary_confusion_matrix,
    classify_structured_critic,
    compare_id_sets,
    minimum_cost_posthoc_oracle,
)
from .types import CompletionResult


VALIDATION_SEED = 20260811
VALIDATION_SAMPLES = 100
TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True)
class ValidationSettings:
    source_summary_path: Path
    data_path: Path
    base_url: str
    model: str
    temperature: float
    solver_max_tokens: int
    critic_max_tokens: int
    refiner_max_tokens: int
    extra_body: dict[str, Any]


def _id_key(question_id: str | int) -> str:
    return json.dumps([type(question_id).__name__, question_id], ensure_ascii=False)


def _valid_id(question_id: Any) -> bool:
    return not isinstance(question_id, bool) and isinstance(question_id, (str, int))


def load_validation_settings(pilot_predictions: str | Path) -> ValidationSettings:
    predictions = Path(pilot_predictions).resolve()
    summary_path = predictions.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Pilot summary is required to reuse model settings: {summary_path}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("mock_only") is not False:
        raise ValueError("Policy validation requires a real, mock_only=false Pilot source")
    backend = summary.get("backend")
    caps = summary.get("generation_caps")
    extra_body = summary.get("request_extra_body")
    data_path = summary.get("data_path")
    temperature = summary.get("temperature")
    if not isinstance(backend, dict) or not isinstance(caps, dict):
        raise ValueError("Pilot summary is missing backend or generation_caps")
    if not isinstance(extra_body, dict):
        raise ValueError("Pilot summary request_extra_body must be an object")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("Pilot summary is missing data_path")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("Pilot summary has invalid temperature")
    base_url = backend.get("base_url")
    model = backend.get("model")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("Pilot summary has invalid backend base_url")
    if not isinstance(model, str) or not model:
        raise ValueError("Pilot summary has invalid backend model")
    validated_caps: dict[str, int] = {}
    for role in ("solver", "critic", "refiner"):
        value = caps.get(role)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Pilot summary has invalid {role} generation cap")
        validated_caps[role] = value
    resolved_data_path = Path(data_path).resolve()
    if not resolved_data_path.is_file():
        raise FileNotFoundError(f"Pilot LogiQA dev file not found: {resolved_data_path}")
    return ValidationSettings(
        source_summary_path=summary_path.resolve(),
        data_path=resolved_data_path,
        base_url=base_url,
        model=model,
        temperature=float(temperature),
        solver_max_tokens=validated_caps["solver"],
        critic_max_tokens=validated_caps["critic"],
        refiner_max_tokens=validated_caps["refiner"],
        extra_body=dict(extra_body),
    )


def load_all_logiqa_examples(data_path: str | Path) -> list[LogiQAExample]:
    path = Path(data_path)
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
            example = _parse_record(payload, line_number)
            if not _valid_id(example.question_id):
                raise ValueError(f"LogiQA line {line_number} has invalid id")
            examples.append(example)
    if not examples:
        raise ValueError(f"LogiQA dev file is empty: {path}")
    return examples


def select_validation_examples(
    examples: list[LogiQAExample],
    excluded_ids: set[str],
    sample_count: int = VALIDATION_SAMPLES,
    seed: int = VALIDATION_SEED,
) -> tuple[list[LogiQAExample], list[str | int]]:
    """Select unique held-out IDs; later official records with duplicate IDs are ignored."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    unique: list[LogiQAExample] = []
    seen: set[str] = set()
    duplicate_ids: list[str | int] = []
    for example in examples:
        key = _id_key(example.question_id)
        if key in seen:
            if example.question_id not in duplicate_ids:
                duplicate_ids.append(example.question_id)
            continue
        seen.add(key)
        if key not in excluded_ids:
            unique.append(example)
    if len(unique) < sample_count:
        raise ValueError(
            f"Only {len(unique)} unique held-out samples remain; need {sample_count}"
        )
    indices = random.Random(seed).sample(range(len(unique)), sample_count)
    selected = [unique[index] for index in indices]
    if len({_id_key(example.question_id) for example in selected}) != sample_count:
        raise AssertionError("Validation selection did not produce unique IDs")
    return selected, duplicate_ids


def _excluded_ids(pilot_predictions: Path) -> tuple[set[str], list[str | int]]:
    rows = read_jsonl(pilot_predictions)
    if not rows:
        raise ValueError(f"Pilot predictions are empty: {pilot_predictions}")
    keys: set[str] = set()
    ids: list[str | int] = []
    for position, row in enumerate(rows, 1):
        question_id = row.get("question_id")
        if not _valid_id(question_id):
            raise ValueError(f"Pilot prediction {position} has invalid question_id")
        key = _id_key(question_id)
        if key in keys:
            raise ValueError(f"Pilot predictions contain duplicate ID {question_id!r}")
        keys.add(key)
        ids.append(question_id)
    return keys, ids


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
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


def _manifest_payload(
    settings: ValidationSettings,
    pilot_predictions: Path,
    all_examples: list[LogiQAExample],
    excluded_ids: list[str | int],
    selected: list[LogiQAExample],
    duplicate_ids: list[str | int],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "policy_selection_validation": True,
        "final_test": False,
        "mock_only": False,
        "seed": seed,
        "sample_count": sample_count,
        "data_path": str(settings.data_path),
        "pilot_predictions": str(pilot_predictions),
        "source_summary": str(settings.source_summary_path),
        "dataset_records": len(all_examples),
        "unique_dataset_ids": len({_id_key(item.question_id) for item in all_examples}),
        "duplicate_dataset_ids_ignored": duplicate_ids,
        "excluded_count": len(excluded_ids),
        "excluded_ids": excluded_ids,
        "selected_ids": [item.question_id for item in selected],
    }


def prepare_validation_split(
    pilot_predictions: str | Path,
    output_dir: str | Path,
    sample_count: int = VALIDATION_SAMPLES,
    seed: int = VALIDATION_SEED,
) -> tuple[ValidationSettings, list[LogiQAExample], dict[str, Any]]:
    pilot_path = Path(pilot_predictions).resolve()
    if not pilot_path.is_file():
        raise FileNotFoundError(f"Pilot predictions not found: {pilot_path}")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    target_predictions = (target / "predictions.jsonl").resolve()
    if target_predictions == pilot_path:
        raise ValueError("Validation output cannot overwrite historical Pilot predictions")
    settings = load_validation_settings(pilot_path)
    all_examples = load_all_logiqa_examples(settings.data_path)
    excluded_keys, excluded_ids = _excluded_ids(pilot_path)
    selected, duplicate_ids = select_validation_examples(
        all_examples, excluded_keys, sample_count=sample_count, seed=seed
    )
    manifest = _manifest_payload(
        settings,
        pilot_path,
        all_examples,
        excluded_ids,
        selected,
        duplicate_ids,
        sample_count,
        seed,
    )
    manifest_path = target / "split_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing split manifest belongs to a different validation run")
    else:
        if any((target / name).exists() for name in ("predictions.jsonl", "summary.json", "report.md")):
            raise ValueError("Validation artifacts exist without a split manifest; refusing overwrite")
        _write_json_atomic(manifest_path, manifest)
    return settings, selected, manifest


def _usage_dict(payload: Any, purpose: str) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Missing real usage for {purpose}")
    usage: dict[str, int] = {}
    for field in USAGE_FIELDS:
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Missing or invalid real {field} for {purpose}")
        usage[field] = value
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise RuntimeError(f"Inconsistent real token usage for {purpose}")
    return usage


def _completion_usage(result: CompletionResult, purpose: str) -> dict[str, int]:
    if not result.usage_reported or result.prompt_tokens is None or result.total_tokens is None:
        raise RuntimeError(
            f"OpenAI-compatible backend did not report real token usage for {purpose}; "
            "policy validation refuses estimated usage"
        )
    return _usage_dict(
        {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
        purpose,
    )


def _add_usage(*parts: dict[str, int]) -> dict[str, int]:
    return {field: sum(part[field] for part in parts) for field in USAGE_FIELDS}


def _latency(payload: Any, purpose: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)) or payload < 0:
        raise RuntimeError(f"Missing or invalid real latency for {purpose}")
    return float(payload)


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


def _checkpoint_path(directory: Path, question_id: str | int) -> Path:
    digest = hashlib.sha256(_id_key(question_id).encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def _load_checkpoint(
    path: Path,
    question_id: str | int,
    manifest_path: Path,
) -> dict[str, Any]:
    identity = {
        "question_id": question_id,
        "split_manifest": str(manifest_path.resolve()),
    }
    if not path.exists():
        return identity
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Validation checkpoint does not match current run: {path}")
    return checkpoint


def _stage_call(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    stage: str,
    backend: LLMBackend,
    messages: list[dict[str, str]],
    max_tokens: int,
    question_id: str | int,
) -> None:
    if stage in checkpoint:
        return
    result, latency = _timed_complete(
        backend, messages, max_tokens, stage, question_id
    )
    checkpoint[stage] = {
        "raw_output": result.content,
        "usage": _completion_usage(result, f"{stage} question {question_id!r}"),
        "latency_seconds": latency,
    }
    _write_json_atomic(checkpoint_path, checkpoint)


def _tolerant_payload(output: str) -> dict[str, Any]:
    parsed = tolerant_final_answer(output)
    return parsed.to_dict()


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


def _problem_payload(example: LogiQAExample) -> dict[str, Any]:
    return {
        "passage": example.passage,
        "question": example.question,
        "options": {
            letter: option for letter, option in zip(ANSWER_LETTERS, example.options)
        },
    }


def _component(stage: dict[str, Any], context: str) -> dict[str, Any]:
    return {
        "raw_output": stage["raw_output"],
        "usage": _usage_dict(stage["usage"], context),
        "calls": 1,
        "latency_seconds": _latency(stage["latency_seconds"], context),
    }


def _branch_payload(
    prompt_version: str,
    solver: dict[str, Any],
    critic_stage: dict[str, Any],
    refiner_stage: dict[str, Any],
    gold: str,
) -> dict[str, Any]:
    critic = _component(critic_stage, f"{prompt_version} Critic")
    refiner = _component(refiner_stage, f"{prompt_version} Refiner")
    refiner["strict_answer"] = extract_final_answer(refiner["raw_output"])
    refiner["tolerant"] = _tolerant_payload(refiner["raw_output"])
    continuation_usage = _add_usage(critic["usage"], refiner["usage"])
    full_usage = _add_usage(solver["usage"], continuation_usage)
    continuation_latency = critic["latency_seconds"] + refiner["latency_seconds"]
    full_latency = solver["latency_seconds"] + continuation_latency
    payload: dict[str, Any] = {
        "prompt_version": prompt_version,
        "critic": critic,
        "refiner": refiner,
        "strict_answer": refiner["strict_answer"],
        "tolerant_answer": refiner["tolerant"]["answer"],
        "transition": _transition(
            solver["tolerant"]["answer"], refiner["tolerant"]["answer"], gold
        ),
        "usage": {
            "continuation": continuation_usage,
            "complete_workflow": full_usage,
        },
        "calls": {"continuation": 2, "complete_workflow": 3},
        "latency_seconds": {
            "continuation": continuation_latency,
            "complete_workflow": full_latency,
        },
    }
    if prompt_version == STRUCTURED_V2:
        protocol = parse_critic_protocol(critic["raw_output"])
        classification = classify_structured_critic(
            critic["raw_output"], solver["tolerant"]["answer"]
        )
        refiner_protocol = parse_refiner_protocol(refiner["raw_output"])
        critic.update(
            {
                "parsed_verdict": protocol.parsed_verdict,
                "proposed_answer": protocol.proposed_answer,
                "effective_verdict": protocol.effective_verdict,
                "effective_proposed_answer": protocol.effective_proposed_answer,
                "review_sent_to_refiner": protocol.review_for_refiner,
                "protocol_classification": classification,
                "raw_revise_intent": classification["verdict"] == "REVISE",
                "actionable_revise": classification["category"]
                == "actionable_revise",
            }
        )
        refiner.update(refiner_protocol)
        payload["critic_parse_failure"] = protocol.parse_failure
        payload["refiner_protocol_violation"] = (
            protocol.effective_verdict == "KEEP"
            and solver["tolerant"]["answer"] is not None
            and refiner["tolerant"]["answer"] is not None
            and solver["tolerant"]["answer"] != refiner["tolerant"]["answer"]
        )
    return payload


def _build_prediction(
    example: LogiQAExample,
    manifest_path: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    solver_stage = checkpoint["solver"]
    solver = _component(solver_stage, "Solver")
    solver["strict_answer"] = extract_final_answer(solver["raw_output"])
    solver["tolerant"] = _tolerant_payload(solver["raw_output"])
    minimal = _branch_payload(
        MINIMAL_V1,
        solver,
        checkpoint["minimal_v1_critic"],
        checkpoint["minimal_v1_refiner"],
        example.gold,
    )
    structured = _branch_payload(
        STRUCTURED_V2,
        solver,
        checkpoint["structured_v2_critic"],
        checkpoint["structured_v2_refiner"],
        example.gold,
    )
    return {
        "question_id": example.question_id,
        "gold": example.gold,
        "policy_selection_validation": True,
        "final_test": False,
        "mock_only": False,
        "split_manifest": str(manifest_path.resolve()),
        "solver_called_once": True,
        "same_solver_state_for_both_policies": True,
        "problem": _problem_payload(example),
        "problem_and_choices": example.problem_text(),
        "solver": solver,
        "minimal_v1": minimal,
        "structured_v2": structured,
        "calls": {
            "solver": 1,
            "minimal_v1_continuation": 2,
            "structured_v2_continuation": 2,
            "actual_total": 5,
        },
    }


def _cost(usage: Any, calls: Any, latency: Any, context: str) -> dict[str, Any]:
    if isinstance(calls, bool) or not isinstance(calls, (int, float)) or calls < 0:
        raise ValueError(f"Invalid calls for {context}")
    return {
        "usage": _usage_dict(usage, context),
        "calls": float(calls),
        "latency_seconds": _latency(latency, context),
    }


def _aggregate_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(costs)
    totals = {
        field: sum(cost["usage"][field] for cost in costs) for field in USAGE_FIELDS
    }
    calls = sum(cost["calls"] for cost in costs)
    latency = sum(cost["latency_seconds"] for cost in costs)
    return {
        "samples": count,
        "total_usage": totals,
        "average_usage": {field: totals[field] / count for field in USAGE_FIELDS},
        "total_calls": calls,
        "average_calls": calls / count,
        "total_latency_seconds": latency,
        "average_latency_seconds": latency / count,
    }


def _answer_metrics(
    predictions: list[dict[str, Any]],
    strategy: str,
    parser: str,
) -> dict[str, Any]:
    answer_key = "strict_answer" if parser == "strict" else "tolerant_answer"
    correct_ids: list[str | int] = []
    failure_ids: list[str | int] = []
    for row in predictions:
        if strategy == "solver":
            answer = (
                row["solver"]["strict_answer"]
                if parser == "strict"
                else row["solver"]["tolerant"]["answer"]
            )
        else:
            answer = row[strategy][answer_key]
        if answer == row["gold"]:
            correct_ids.append(row["question_id"])
        if answer is None:
            failure_ids.append(row["question_id"])
    return {
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(predictions),
        "correct_ids": correct_ids,
        "parse_failures": len(failure_ids),
        "parse_failure_ids": failure_ids,
    }


def _policy_metrics(
    predictions: list[dict[str, Any]],
    policy: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    solver_tolerant = _answer_metrics(predictions, "solver", "tolerant")
    full_tolerant = _answer_metrics(predictions, policy, "tolerant")
    transition_ids = {name: [] for name in TRANSITIONS}
    for row in predictions:
        transition_ids[row[policy]["transition"]].append(row["question_id"])
    corrected_ids = transition_ids["wrong_to_correct"]
    degraded_ids = transition_ids["correct_to_wrong"]
    solver_errors = len(predictions) - solver_tolerant["correct"]
    full_costs = [case[policy]["cost"] for case in cases]
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    return {
        "samples": len(predictions),
        "strict": _answer_metrics(predictions, policy, "strict"),
        "tolerant": full_tolerant,
        "transitions": {
            name: {"count": len(ids), "sample_ids": ids}
            for name, ids in transition_ids.items()
        },
        "corrected": len(corrected_ids),
        "corrected_ids": corrected_ids,
        "degraded": len(degraded_ids),
        "degraded_ids": degraded_ids,
        "unchanged": len(predictions) - len(corrected_ids) - len(degraded_ids),
        "normalized": {
            "corrected_per_n": len(corrected_ids) / len(predictions),
            "degraded_per_n": len(degraded_ids) / len(predictions),
            "corrected_per_solver_errors": ratio(len(corrected_ids), solver_errors),
            "degraded_per_solver_correct": ratio(
                len(degraded_ids), solver_tolerant["correct"]
            ),
            "corrected_degraded_benefit_risk_ratio": ratio(
                len(corrected_ids), len(degraded_ids)
            ),
        },
        "full_recorded_cost": _aggregate_costs(full_costs),
        "minimum_cost_posthoc_oracle": minimum_cost_posthoc_oracle(cases, policy),
    }


def structured_critic_detection_metrics(
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = [row["question_id"] for row in predictions]
    actual = [
        row["solver"]["tolerant"]["answer"] != row["gold"] for row in predictions
    ]
    raw = [
        row["structured_v2"]["critic"]["raw_revise_intent"]
        for row in predictions
    ]
    actionable = [
        row["structured_v2"]["critic"]["actionable_revise"]
        for row in predictions
    ]
    result = {
        "actual_error_definition": "Solver tolerant answer differs from gold",
        "raw_revise_intent": binary_confusion_matrix(actual, raw, ids),
        "actionable_revise": binary_confusion_matrix(actual, actionable, ids),
    }
    result["raw_revise_intent"]["detection_rule"] = (
        "The raw Critic output contains one recognized VERDICT: REVISE, including "
        "incomplete, no-op, actionable, or otherwise malformed proposed answers."
    )
    result["actionable_revise"]["detection_rule"] = (
        "VERDICT: REVISE with one A-D proposed answer different from the Solver answer."
    )
    return result


def _case_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in predictions:
        solver_cost = _cost(
            row["solver"]["usage"], 1, row["solver"]["latency_seconds"], "Solver"
        )
        case: dict[str, Any] = {
            "question_id": row["question_id"],
            "gold": row["gold"],
            "solver": {
                "tolerant_answer": row["solver"]["tolerant"]["answer"],
                "cost": solver_cost,
            },
        }
        for policy in (MINIMAL_V1, STRUCTURED_V2):
            branch = row[policy]
            case[policy] = {
                "tolerant_answer": branch["tolerant_answer"],
                "transition": branch["transition"],
                "cost": _cost(
                    branch["usage"]["complete_workflow"],
                    branch["calls"]["complete_workflow"],
                    branch["latency_seconds"]["complete_workflow"],
                    f"{policy} Full",
                ),
            }
        cases.append(case)
    return cases


def build_validation_summary(
    predictions: list[dict[str, Any]],
    manifest: dict[str, Any],
    settings: ValidationSettings,
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("Cannot summarize empty validation predictions")
    cases = _case_rows(predictions)
    minimal = _policy_metrics(predictions, MINIMAL_V1, cases)
    structured = _policy_metrics(predictions, STRUCTURED_V2, cases)
    solver = {
        "strict": _answer_metrics(predictions, "solver", "strict"),
        "tolerant": _answer_metrics(predictions, "solver", "tolerant"),
        "recorded_cost": _aggregate_costs([case["solver"]["cost"] for case in cases]),
    }
    corrected = compare_id_sets(minimal["corrected_ids"], structured["corrected_ids"])
    degraded = compare_id_sets(minimal["degraded_ids"], structured["degraded_ids"])
    v1_correct_v2_wrong = [
        row["question_id"]
        for row in predictions
        if row[MINIMAL_V1]["tolerant_answer"] == row["gold"]
        and row[STRUCTURED_V2]["tolerant_answer"] != row["gold"]
    ]
    v1_wrong_v2_correct = [
        row["question_id"]
        for row in predictions
        if row[MINIMAL_V1]["tolerant_answer"] != row["gold"]
        and row[STRUCTURED_V2]["tolerant_answer"] == row["gold"]
    ]
    mcnemar = exact_mcnemar(len(v1_correct_v2_wrong), len(v1_wrong_v2_correct))
    mcnemar.update(
        {
            "comparison": "minimal_v1 vs structured_v2 tolerant correctness",
            "minimal_v1_correct_structured_v2_wrong_ids": v1_correct_v2_wrong,
            "minimal_v1_wrong_structured_v2_correct_ids": v1_wrong_v2_correct,
        }
    )
    classifications = Counter(
        row[STRUCTURED_V2]["critic"]["protocol_classification"]["category"]
        for row in predictions
    )
    actual_calls = sum(row["calls"]["actual_total"] for row in predictions)
    return {
        "policy_selection_validation": True,
        "final_test": False,
        "mock_only": False,
        "samples": len(predictions),
        "seed": manifest["seed"],
        "split_manifest": str((Path(predictions[0]["split_manifest"])).resolve()),
        "source_summary": str(settings.source_summary_path),
        "backend": {
            "base_url": settings.base_url,
            "model": settings.model,
            "temperature": settings.temperature,
            "extra_body": settings.extra_body,
        },
        "generation_caps": {
            "solver": settings.solver_max_tokens,
            "critic": settings.critic_max_tokens,
            "refiner": settings.refiner_max_tokens,
        },
        "inference": {
            "solver_calls": len(predictions),
            "minimal_v1_critic_refiner_calls": 2 * len(predictions),
            "structured_v2_critic_refiner_calls": 2 * len(predictions),
            "actual_total_calls": actual_calls,
            "same_solver_state_for_both_policies": True,
            "full_continuations_always_completed": True,
            "usage_estimated": False,
        },
        "solver": solver,
        "policies": {
            MINIMAL_V1: minimal,
            STRUCTURED_V2: structured,
        },
        "structured_v2_critic": {
            "protocol_classification": {
                name: classifications.get(name, 0) for name in CRITIC_PROTOCOL_CLASSES
            },
            "parse_failures": sum(
                row[STRUCTURED_V2]["critic_parse_failure"] for row in predictions
            ),
            "error_detection": structured_critic_detection_metrics(predictions),
        },
        "label_stability": {"corrected": corrected, "degraded": degraded},
        "paired_mcnemar_exact": mcnemar,
        "policy_selection": {
            "selected": None,
            "automatic_selection": False,
            "note": "Validation results are reported without automatically selecting a prompt.",
        },
    }


def _format_ratio(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _format_ids(values: list[str | int]) -> str:
    return ", ".join(map(str, values)) if values else "None"


def build_validation_report(summary: dict[str, Any]) -> str:
    solver = summary["solver"]
    lines = [
        "# LogiQA 2.0 Prompt Policy Validation",
        "",
        "**policy_selection_validation=true; final_test=false; mock_only=false.**",
        "",
        "This held-out validation compares two fixed FULL continuation policies. "
        "It does not automatically select or modify either prompt.",
        "",
        f"Samples: {summary['samples']} (seed={summary['seed']})",
        f"Actual calls: {summary['inference']['actual_total_calls']}",
        "",
        "## Accuracy, transition, and recorded full-workflow cost",
        "",
        "| Strategy | Strict acc. | Strict failures | Tolerant acc. | Tolerant failures | Corrected | Degraded | Corrected/errors | Degraded/correct | Benefit-risk | Avg tokens | Avg calls | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    solver_cost = solver["recorded_cost"]
    lines.append(
        f"| Solver Only | {solver['strict']['accuracy']:.4f} | "
        f"{solver['strict']['parse_failures']} | {solver['tolerant']['accuracy']:.4f} | "
        f"{solver['tolerant']['parse_failures']} | 0 | 0 | 0.0000 | 0.0000 | "
        f"undefined | {solver_cost['average_usage']['total_tokens']:.2f} | "
        f"{solver_cost['average_calls']:.2f} | {solver_cost['average_latency_seconds']:.4f} |"
    )
    for policy in (MINIMAL_V1, STRUCTURED_V2):
        metric = summary["policies"][policy]
        normalized = metric["normalized"]
        cost = metric["full_recorded_cost"]
        lines.append(
            f"| Full {policy} | {metric['strict']['accuracy']:.4f} | "
            f"{metric['strict']['parse_failures']} | {metric['tolerant']['accuracy']:.4f} | "
            f"{metric['tolerant']['parse_failures']} | {metric['corrected']} | "
            f"{metric['degraded']} | "
            f"{_format_ratio(normalized['corrected_per_solver_errors'])} | "
            f"{_format_ratio(normalized['degraded_per_solver_correct'])} | "
            f"{_format_ratio(normalized['corrected_degraded_benefit_risk_ratio'])} | "
            f"{cost['average_usage']['total_tokens']:.2f} | {cost['average_calls']:.2f} | "
            f"{cost['average_latency_seconds']:.4f} |"
        )
    lines.extend(["", "## Tolerant transitions", ""])
    for policy in (MINIMAL_V1, STRUCTURED_V2):
        transition = summary["policies"][policy]["transitions"]
        lines.extend(
            [
                f"### {policy}",
                "",
                f"- correct→correct: {transition['correct_to_correct']['count']}",
                f"- correct→wrong: {transition['correct_to_wrong']['count']}",
                f"- wrong→correct: {transition['wrong_to_correct']['count']}",
                f"- wrong→wrong: {transition['wrong_to_wrong']['count']}",
                "",
            ]
        )
    lines.extend(["## structured_v2 Critic error detection", ""])
    for name in ("raw_revise_intent", "actionable_revise"):
        metric = summary["structured_v2_critic"]["error_detection"][name]
        matrix = metric["matrix"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- TP={matrix['true_positive']}, FP={matrix['false_positive']}, "
                f"TN={matrix['true_negative']}, FN={matrix['false_negative']}",
                f"- Precision={metric['precision']:.4f}; Recall={metric['recall']:.4f}; "
                f"F1={metric['f1']:.4f}; Specificity={metric['specificity']:.4f}",
                "",
            ]
        )
    mcnemar = summary["paired_mcnemar_exact"]
    lines.extend(
        [
            "## Paired policy comparison",
            "",
            f"- Exact McNemar discordant pairs: {mcnemar['discordant_pairs']}",
            f"- minimal_v1 correct / structured_v2 wrong: {mcnemar['correct_to_wrong']}",
            f"- minimal_v1 wrong / structured_v2 correct: {mcnemar['wrong_to_correct']}",
            f"- Two-sided p-value: {mcnemar['p_value']:.6f}",
            "",
        ]
    )
    for label in ("corrected", "degraded"):
        overlap = summary["label_stability"][label]
        lines.extend(
            [
                f"### {label} ID overlap",
                "",
                f"- Intersection: {_format_ids(overlap['intersection'])}",
                f"- Union: {_format_ids(overlap['union'])}",
                f"- Jaccard: {overlap['jaccard']:.4f}",
                "",
            ]
        )
    lines.extend(["## Minimum-cost posthoc oracles", ""])
    for policy in (MINIMAL_V1, STRUCTURED_V2):
        oracle = summary["policies"][policy]["minimum_cost_posthoc_oracle"]
        cost = oracle["recorded_costs"]
        lines.extend(
            [
                f"- {policy}: accuracy={oracle['accuracy']:.4f}, "
                f"Full usage={oracle['full_usage_rate']:.4f}, "
                f"avg tokens={cost['average_usage']['total_tokens']:.2f}, "
                f"avg calls={cost['average_calls']:.2f}, "
                f"avg latency={cost['average_latency_seconds']:.4f}s; "
                "posthoc_oracle=true, deployable=false.",
            ]
        )
    lines.extend(
        [
            "",
            "All token usage is backend-reported. Missing usage aborts the run; no usage is estimated.",
            "No policy was automatically selected.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_completed(
    predictions_path: Path,
    manifest_path: Path,
    selected: list[LogiQAExample],
) -> dict[str, dict[str, Any]]:
    selected_keys = {_id_key(item.question_id) for item in selected}
    completed: dict[str, dict[str, Any]] = {}
    if not predictions_path.exists():
        return completed
    for position, row in enumerate(read_jsonl(predictions_path), 1):
        question_id = row.get("question_id")
        key = _id_key(question_id)
        if key not in selected_keys:
            raise ValueError(f"Existing validation row {position} is not in the split")
        if key in completed:
            raise ValueError(f"Duplicate completed validation ID {question_id!r}")
        if (
            row.get("policy_selection_validation") is not True
            or row.get("final_test") is not False
            or row.get("mock_only") is not False
            or row.get("split_manifest") != str(manifest_path.resolve())
        ):
            raise ValueError("Existing predictions belong to a different validation run")
        completed[key] = row
    return completed


def run_logiqa_policy_validation(
    pilot_predictions: str | Path,
    output_dir: str | Path,
    backend: LLMBackend,
    sample_count: int = VALIDATION_SAMPLES,
    seed: int = VALIDATION_SEED,
) -> dict[str, Any]:
    if bool(getattr(backend, "mock_only", True)):
        raise ValueError(
            "validate-logiqa-policies requires a real-compatible backend; Mock is forbidden"
        )
    target = Path(output_dir)
    settings, selected, manifest = prepare_validation_split(
        pilot_predictions, target, sample_count=sample_count, seed=seed
    )
    manifest_path = target / "split_manifest.json"
    predictions_path = target / "predictions.jsonl"
    completed = _prepare_completed(predictions_path, manifest_path, selected)
    checkpoint_dir = target / ".validation_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for example in selected:
        key = _id_key(example.question_id)
        if key in completed:
            continue
        checkpoint_path = _checkpoint_path(checkpoint_dir, example.question_id)
        checkpoint = _load_checkpoint(checkpoint_path, example.question_id, manifest_path)
        problem = example.problem_text()
        _stage_call(
            checkpoint,
            checkpoint_path,
            "solver",
            backend,
            build_solver_messages(problem),
            settings.solver_max_tokens,
            example.question_id,
        )
        solver_output = checkpoint["solver"]["raw_output"]
        _stage_call(
            checkpoint,
            checkpoint_path,
            "minimal_v1_critic",
            backend,
            build_versioned_critic_messages(problem, solver_output, MINIMAL_V1),
            settings.critic_max_tokens,
            example.question_id,
        )
        _stage_call(
            checkpoint,
            checkpoint_path,
            "minimal_v1_refiner",
            backend,
            build_versioned_refiner_messages(
                problem,
                solver_output,
                checkpoint["minimal_v1_critic"]["raw_output"],
                MINIMAL_V1,
            ),
            settings.refiner_max_tokens,
            example.question_id,
        )
        _stage_call(
            checkpoint,
            checkpoint_path,
            "structured_v2_critic",
            backend,
            build_versioned_critic_messages(problem, solver_output, STRUCTURED_V2),
            settings.critic_max_tokens,
            example.question_id,
        )
        critic_protocol = parse_critic_protocol(
            checkpoint["structured_v2_critic"]["raw_output"]
        )
        _stage_call(
            checkpoint,
            checkpoint_path,
            "structured_v2_refiner",
            backend,
            build_versioned_refiner_messages(
                problem,
                solver_output,
                critic_protocol.review_for_refiner,
                STRUCTURED_V2,
            ),
            settings.refiner_max_tokens,
            example.question_id,
        )
        prediction = _build_prediction(example, manifest_path, checkpoint)
        _append_jsonl(predictions_path, prediction)
        completed[key] = prediction
        checkpoint_path.unlink()
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    ordered = [completed[_id_key(example.question_id)] for example in selected]
    if len(ordered) != sample_count:
        raise RuntimeError("Policy validation did not complete every selected sample")
    summary = build_validation_summary(ordered, manifest, settings)
    _write_json_atomic(target / "summary.json", summary)
    _write_text_atomic(target / "report.md", build_validation_report(summary))
    return summary
