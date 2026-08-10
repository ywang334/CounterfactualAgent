from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .backend import LLMBackend
from .io_utils import read_jsonl
from .logiqa_audit import tolerant_final_answer
from .logiqa_pilot import (
    ANSWER_LETTERS,
    LogiQAExample,
    build_solver_messages,
    extract_final_answer,
    load_logiqa_dev,
)
from .logiqa_policy_validation import (
    _add_usage,
    _append_jsonl,
    _completion_usage,
    _latency,
    _timed_complete,
    _usage_dict,
    _write_json_atomic,
    _write_text_atomic,
    load_all_logiqa_examples,
)
from .logiqa_prompts import (
    MINIMAL_V1,
    STRUCTURED_V2,
    build_versioned_critic_messages,
    build_versioned_refiner_messages,
)
from .logiqa_replay import parse_critic_protocol, parse_refiner_protocol
from .prompt_stability_audit import compare_id_sets
from .types import CompletionResult


ACTION_COLLECTION_SEED = 20260812
ACTION_COLLECTION_SAMPLES = 200
DEFAULT_PILOT_PREDICTIONS = Path("artifacts/pilot_logiqa/predictions.jsonl")
DEFAULT_VALIDATION_PREDICTIONS = Path(
    "artifacts/logiqa_policy_validation_100/predictions.jsonl"
)
DEFAULT_VALIDATION_SUMMARY = Path(
    "artifacts/logiqa_policy_validation_100/summary.json"
)
ACTIONS = ("STOP", "SHORT", "FULL")
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True)
class ActionCollectionSettings:
    source_validation_summary: Path
    base_url: str
    model: str
    temperature: float
    solver_max_tokens: int
    critic_max_tokens: int
    refiner_max_tokens: int
    extra_body: dict[str, Any]


def load_action_collection_settings(
    validation_summary_path: str | Path = DEFAULT_VALIDATION_SUMMARY,
) -> ActionCollectionSettings:
    path = Path(validation_summary_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Saved policy-validation summary is required for exact config reuse: {path}"
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (
        summary.get("policy_selection_validation") is not True
        or summary.get("mock_only") is not False
    ):
        raise ValueError("Source summary is not a real LogiQA policy validation")
    backend = summary.get("backend")
    caps = summary.get("generation_caps")
    if not isinstance(backend, dict) or not isinstance(caps, dict):
        raise ValueError("Validation summary is missing backend or generation caps")
    base_url = backend.get("base_url")
    model = backend.get("model")
    temperature = backend.get("temperature")
    extra_body = backend.get("extra_body")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("Validation summary has invalid base_url")
    if not isinstance(model, str) or not model:
        raise ValueError("Validation summary has invalid model")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("Validation summary has invalid temperature")
    if not isinstance(extra_body, dict):
        raise ValueError("Validation summary has invalid extra_body")
    checked_caps: dict[str, int] = {}
    for role in ("solver", "critic", "refiner"):
        value = caps.get(role)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Validation summary has invalid {role} cap")
        checked_caps[role] = value
    return ActionCollectionSettings(
        source_validation_summary=path,
        base_url=base_url,
        model=model,
        temperature=float(temperature),
        solver_max_tokens=checked_caps["solver"],
        critic_max_tokens=checked_caps["critic"],
        refiner_max_tokens=checked_caps["refiner"],
        extra_body=dict(extra_body),
    )


def normalize_logiqa_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def content_sha256(
    passage: str,
    question: str,
    options: list[str] | tuple[str, ...],
) -> str:
    if len(options) != 4:
        raise ValueError("Content hash requires exactly four options")
    canonical = {
        "passage": normalize_logiqa_text(passage),
        "question": normalize_logiqa_text(question),
        "options": [normalize_logiqa_text(option) for option in options],
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def example_content_sha256(example: LogiQAExample) -> str:
    return content_sha256(
        example.passage,
        example.question,
        list(example.options),
    )


def _problem_content_sha256(problem: dict[str, Any]) -> str:
    passage = problem.get("passage")
    question = problem.get("question")
    options = problem.get("options")
    if not isinstance(passage, str) or not isinstance(question, str):
        raise ValueError("Saved problem lacks passage or question")
    if isinstance(options, dict):
        option_list = [options.get(letter) for letter in ANSWER_LETTERS]
    else:
        option_list = options
    if (
        not isinstance(option_list, (list, tuple))
        or len(option_list) != 4
        or not all(isinstance(option, str) for option in option_list)
    ):
        raise ValueError("Saved problem lacks complete A-D options")
    return content_sha256(passage, question, list(option_list))


def _pilot_content_hashes(predictions_path: Path) -> set[str]:
    rows = read_jsonl(predictions_path)
    summary_path = predictions_path.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Pilot summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    data_path = summary.get("data_path")
    limit = summary.get("requested_limit", summary.get("samples"))
    seed = summary.get("seed")
    if (
        not isinstance(data_path, str)
        or isinstance(limit, bool)
        or not isinstance(limit, int)
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("Pilot summary lacks deterministic source sampling settings")
    examples = load_logiqa_dev(data_path, limit=limit, seed=seed)
    if len(rows) != len(examples):
        raise ValueError("Pilot predictions and reconstructed examples have different sizes")
    hashes: set[str] = set()
    for row, example in zip(rows, examples):
        if row.get("question_id") != example.question_id:
            raise ValueError("Pilot prediction order does not match reconstructed source")
        hashes.add(example_content_sha256(example))
    return hashes


def _saved_problem_hashes(predictions_path: Path) -> set[str]:
    hashes: set[str] = set()
    for position, row in enumerate(read_jsonl(predictions_path), 1):
        problem = row.get("problem")
        if not isinstance(problem, dict):
            raise ValueError(
                f"Saved prediction {predictions_path}:{position} lacks structured problem"
            )
        hashes.add(_problem_content_sha256(problem))
    return hashes


def load_excluded_content_hashes(
    pilot_predictions: str | Path = DEFAULT_PILOT_PREDICTIONS,
    validation_predictions: str | Path = DEFAULT_VALIDATION_PREDICTIONS,
) -> tuple[set[str], dict[str, Any]]:
    pilot_path = Path(pilot_predictions).resolve()
    validation_path = Path(validation_predictions).resolve()
    if not pilot_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Both Pilot and policy-validation predictions are required")
    pilot_hashes = _pilot_content_hashes(pilot_path)
    validation_hashes = _saved_problem_hashes(validation_path)
    combined = pilot_hashes | validation_hashes
    return combined, {
        "pilot_predictions": str(pilot_path),
        "pilot_unique_content": len(pilot_hashes),
        "validation_predictions": str(validation_path),
        "validation_unique_content": len(validation_hashes),
        "combined_unique_content": len(combined),
    }


def select_action_collection_examples(
    examples: list[LogiQAExample],
    excluded_hashes: set[str],
    sample_count: int = ACTION_COLLECTION_SAMPLES,
    seed: int = ACTION_COLLECTION_SEED,
) -> tuple[list[LogiQAExample], dict[str, Any]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    unique: list[LogiQAExample] = []
    seen: set[str] = set()
    duplicate_count = 0
    excluded_count = 0
    for example in examples:
        digest = example_content_sha256(example)
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        if digest in excluded_hashes:
            excluded_count += 1
            continue
        unique.append(example)
    if len(unique) < sample_count:
        raise ValueError(
            f"Only {len(unique)} unique, non-overlapping train samples remain; "
            f"need {sample_count}"
        )
    indices = random.Random(seed).sample(range(len(unique)), sample_count)
    selected = [unique[index] for index in indices]
    selected_hashes = [example_content_sha256(example) for example in selected]
    if len(set(selected_hashes)) != sample_count:
        raise AssertionError("Selected action-collection samples are not content-unique")
    return selected, {
        "dataset_records": len(examples),
        "unique_content_records": len(seen),
        "within_train_duplicate_records_ignored": duplicate_count,
        "historical_content_records_excluded": excluded_count,
        "eligible_unique_records": len(unique),
    }


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_prompt_hashes(prompt_version: str) -> dict[str, str]:
    critic_messages = build_versioned_critic_messages(
        "{problem_and_choices}",
        "{solver_response}",
        prompt_version,
    )
    refiner_messages = build_versioned_refiner_messages(
        "{problem_and_choices}",
        "{solver_response}",
        "{critic_review}",
        prompt_version,
    )
    return {
        "critic_prompt_sha256": _sha256_payload(critic_messages),
        "refiner_prompt_sha256": _sha256_payload(refiner_messages),
        "continuation_policy_sha256": _sha256_payload(
            {"critic": critic_messages, "refiner": refiner_messages}
        ),
    }


def _git_commit(repository: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot record current git commit for policy manifest") from exc
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError("git rev-parse returned an invalid commit")
    return commit


def build_continuation_policy_manifest(
    settings: ActionCollectionSettings,
    repository: str | Path,
) -> dict[str, Any]:
    return {
        "continuation_policy_manifest_version": 1,
        "actions": ["STOP", "SHORT", "FULL"],
        "disabled_actions": ["SKIP", "MEDIUM"],
        "stop": {"calls": 0, "policy": "return Solver output"},
        "short": {
            "prompt_version": MINIMAL_V1,
            **_policy_prompt_hashes(MINIMAL_V1),
        },
        "full": {
            "prompt_version": STRUCTURED_V2,
            **_policy_prompt_hashes(STRUCTURED_V2),
        },
        "model_config": {
            "base_url": settings.base_url,
            "model": settings.model,
            "temperature": settings.temperature,
            "generation_caps": {
                "solver": settings.solver_max_tokens,
                "critic": settings.critic_max_tokens,
                "refiner": settings.refiner_max_tokens,
            },
            "extra_body": settings.extra_body,
        },
        "source_validation_summary": str(settings.source_validation_summary),
        "git_commit": _git_commit(Path(repository).resolve()),
    }


def _manifest_payload(
    data_path: Path,
    selected: list[LogiQAExample],
    selection_stats: dict[str, Any],
    exclusion_stats: dict[str, Any],
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "logiqa_action_collection": True,
        "mock_only": False,
        "budget_semantics_version": 2,
        "hard_budget": "completion_tokens+calls",
        "optimization_cost": "total_tokens+calls",
        "data_split": "train",
        "data_path": str(data_path),
        "seed": seed,
        "sample_count": sample_count,
        "content_normalization": "Unicode NFKC + casefold + collapsed whitespace",
        "content_hash": "SHA256(canonical passage, question, ordered options)",
        "selection_stats": selection_stats,
        "exclusion_sources": exclusion_stats,
        "selected_samples": [
            {
                "question_id": example.question_id,
                "content_sha256": example_content_sha256(example),
            }
            for example in selected
        ],
    }


def prepare_action_collection_split(
    data_path: str | Path,
    output_dir: str | Path,
    settings: ActionCollectionSettings,
    pilot_predictions: str | Path = DEFAULT_PILOT_PREDICTIONS,
    validation_predictions: str | Path = DEFAULT_VALIDATION_PREDICTIONS,
    sample_count: int = ACTION_COLLECTION_SAMPLES,
    seed: int = ACTION_COLLECTION_SEED,
) -> tuple[list[LogiQAExample], dict[str, Any], dict[str, Any]]:
    source = Path(data_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"Official LogiQA train.txt not found: {source}; automatic download is forbidden"
        )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    examples = load_all_logiqa_examples(source)
    excluded, exclusion_stats = load_excluded_content_hashes(
        pilot_predictions,
        validation_predictions,
    )
    selected, selection_stats = select_action_collection_examples(
        examples,
        excluded,
        sample_count=sample_count,
        seed=seed,
    )
    split_manifest = _manifest_payload(
        source,
        selected,
        selection_stats,
        exclusion_stats,
        sample_count,
        seed,
    )
    policy_manifest = build_continuation_policy_manifest(settings, Path.cwd())
    split_path = target / "split_manifest.json"
    policy_path = target / "continuation_policy_manifest.json"
    if split_path.exists():
        if json.loads(split_path.read_text(encoding="utf-8")) != split_manifest:
            raise ValueError("Existing split manifest belongs to a different collection")
    else:
        if any(
            (target / name).exists()
            for name in ("rollouts.jsonl", "summary.json", "report.md")
        ):
            raise ValueError("Collection artifacts exist without a split manifest")
        # The split is deliberately persisted before the policy manifest and calls.
        _write_json_atomic(split_path, split_manifest)
    if policy_path.exists():
        if json.loads(policy_path.read_text(encoding="utf-8")) != policy_manifest:
            raise ValueError(
                "Continuation policy or model config changed; refusing silent resume"
            )
    else:
        _write_json_atomic(policy_path, policy_manifest)
    return selected, split_manifest, policy_manifest


def _checkpoint_path(directory: Path, digest: str) -> Path:
    return directory / f"{digest}.json"


def _load_checkpoint(
    path: Path,
    digest: str,
    split_manifest: Path,
    policy_manifest: Path,
) -> dict[str, Any]:
    identity = {
        "content_sha256": digest,
        "split_manifest": str(split_manifest.resolve()),
        "continuation_policy_manifest": str(policy_manifest.resolve()),
    }
    if not path.exists():
        return identity
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if any(checkpoint.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Action collection checkpoint identity mismatch: {path}")
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
        backend,
        messages,
        max_tokens,
        stage,
        question_id,
    )
    checkpoint[stage] = {
        "raw_output": result.content,
        "usage": _completion_usage(
            result,
            f"{stage} question {question_id!r}",
        ),
        "latency_seconds": latency,
    }
    _write_json_atomic(checkpoint_path, checkpoint)


def _tolerant_payload(output: str) -> dict[str, Any]:
    return tolerant_final_answer(output).to_dict()


def _cost(
    usage: dict[str, int],
    calls: int,
    latency_seconds: float,
) -> dict[str, Any]:
    checked = _usage_dict(usage, "action collection cost")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise ValueError("Action collection cost has invalid calls")
    checked_latency = _latency(latency_seconds, "action collection cost")
    return {
        **checked,
        "calls": calls,
        "latency_seconds": checked_latency,
    }


def _zero_cost() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "latency_seconds": 0.0,
    }


def _stage_cost(stage: dict[str, Any]) -> dict[str, Any]:
    return _cost(
        stage["usage"],
        1,
        stage["latency_seconds"],
    )


def _combine_costs(*costs: dict[str, Any]) -> dict[str, Any]:
    return _cost(
        {
            field: sum(int(cost[field]) for cost in costs)
            for field in USAGE_FIELDS
        },
        sum(int(cost["calls"]) for cost in costs),
        sum(float(cost["latency_seconds"]) for cost in costs),
    )


def _outcome(solver_correct: bool, action_correct: bool) -> str:
    if not solver_correct and action_correct:
        return "helpful"
    if solver_correct and not action_correct:
        return "harmful"
    if solver_correct:
        return "neutral_correct"
    return "neutral_wrong"


def _problem_payload(example: LogiQAExample) -> dict[str, Any]:
    return {
        "passage": example.passage,
        "question": example.question,
        "options": {
            letter: option for letter, option in zip(ANSWER_LETTERS, example.options)
        },
    }


def _action_payload(
    action: str,
    solver_output: str,
    solver_cost: dict[str, Any],
    critic_stage: dict[str, Any] | None,
    refiner_stage: dict[str, Any] | None,
    gold: str,
) -> dict[str, Any]:
    if action == "STOP":
        output = solver_output
        incremental = _zero_cost()
        raw_outputs = {"solver": solver_output, "critic": None, "refiner": None}
    else:
        if critic_stage is None or refiner_stage is None:
            raise ValueError(f"{action} requires Critic and Refiner stages")
        output = refiner_stage["raw_output"]
        incremental = _combine_costs(
            _stage_cost(critic_stage),
            _stage_cost(refiner_stage),
        )
        raw_outputs = {
            "solver": solver_output,
            "critic": critic_stage["raw_output"],
            "refiner": refiner_stage["raw_output"],
        }
    tolerant = _tolerant_payload(output)
    strict = extract_final_answer(output)
    solver_correct = _tolerant_payload(solver_output)["answer"] == gold
    action_correct = tolerant["answer"] == gold
    return {
        "action": action,
        "strict_answer": strict,
        "tolerant": tolerant,
        "correct": action_correct,
        "parse_failure": tolerant["answer"] is None,
        "outcome": _outcome(solver_correct, action_correct),
        "raw_outputs": raw_outputs,
        "complete_cost": _combine_costs(solver_cost, incremental),
        "incremental_cost": incremental,
    }


def _build_rollout(
    example: LogiQAExample,
    split_manifest: Path,
    policy_manifest: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    solver_stage = checkpoint["solver"]
    solver_output = solver_stage["raw_output"]
    solver_cost = _stage_cost(solver_stage)
    solver_tolerant = _tolerant_payload(solver_output)
    solver_strict = extract_final_answer(solver_output)
    problem = example.problem_text()
    short = _action_payload(
        "SHORT",
        solver_output,
        solver_cost,
        checkpoint["short_critic"],
        checkpoint["short_refiner"],
        example.gold,
    )
    full = _action_payload(
        "FULL",
        solver_output,
        solver_cost,
        checkpoint["full_critic"],
        checkpoint["full_refiner"],
        example.gold,
    )
    full_protocol = parse_critic_protocol(checkpoint["full_critic"]["raw_output"])
    full["critic_protocol"] = full_protocol.to_dict()
    full["refiner_protocol"] = parse_refiner_protocol(
        checkpoint["full_refiner"]["raw_output"]
    )
    return {
        "sample_id": example_content_sha256(example),
        "question_id": example.question_id,
        "gold": example.gold,
        "logiqa_action_collection": True,
        "mock_only": False,
        "budget_semantics_version": 2,
        "hard_budget": "completion_tokens+calls",
        "optimization_cost": "total_tokens+calls",
        "split_manifest": str(split_manifest.resolve()),
        "continuation_policy_manifest": str(policy_manifest.resolve()),
        "solver_called_once": True,
        "same_solver_state_for_short_and_full": True,
        "state_for_controller": {
            "sample_id": example_content_sha256(example),
            "question_id": example.question_id,
            "problem": _problem_payload(example),
            "problem_and_choices": problem,
            "solver_raw_output": solver_output,
            "solver_strict_answer": solver_strict,
            "solver_tolerant": solver_tolerant,
            "generation_pre_action_state": {
                "query": problem,
                "current_answer": solver_output,
                "history": build_solver_messages(problem)
                + [
                    {
                        "role": "assistant",
                        "name": "solver",
                        "content": solver_output,
                    }
                ],
                "role": "critic",
                "round_index": 0,
                "collaboration_steps": 0,
                "usage": {
                    "extra_prompt_tokens": 0,
                    "extra_completion_tokens": 0,
                    "extra_total_tokens": 0,
                    "extra_calls": 0,
                },
                "terminated": False,
                "termination_reason": None,
                "metadata": {},
            },
        },
        "solver": {
            "raw_output": solver_output,
            "strict_answer": solver_strict,
            "tolerant": solver_tolerant,
            "correct": solver_tolerant["answer"] == example.gold,
            "cost": solver_cost,
        },
        "actions": {
            "STOP": _action_payload(
                "STOP",
                solver_output,
                solver_cost,
                None,
                None,
                example.gold,
            ),
            "SHORT": short,
            "FULL": full,
        },
        "actual_calls": 5,
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def _action_metrics(rollouts: list[dict[str, Any]], action: str) -> dict[str, Any]:
    strict_correct_ids = [
        row["question_id"]
        for row in rollouts
        if row["actions"][action]["strict_answer"] == row["gold"]
    ]
    tolerant_correct_ids = [
        row["question_id"]
        for row in rollouts
        if row["actions"][action]["tolerant"]["answer"] == row["gold"]
    ]
    solver_correct = [
        row["solver"]["tolerant"]["answer"] == row["gold"] for row in rollouts
    ]
    action_correct = [
        row["actions"][action]["tolerant"]["answer"] == row["gold"]
        for row in rollouts
    ]
    corrected_ids = [
        row["question_id"]
        for row, before, after in zip(rollouts, solver_correct, action_correct)
        if not before and after
    ]
    degraded_ids = [
        row["question_id"]
        for row, before, after in zip(rollouts, solver_correct, action_correct)
        if before and not after
    ]
    outcomes = Counter(row["actions"][action]["outcome"] for row in rollouts)
    costs = [row["actions"][action]["incremental_cost"] for row in rollouts]
    return {
        "strict": {
            "correct": len(strict_correct_ids),
            "accuracy": len(strict_correct_ids) / len(rollouts),
            "parse_failures": sum(
                row["actions"][action]["strict_answer"] is None for row in rollouts
            ),
            "correct_ids": strict_correct_ids,
        },
        "tolerant": {
            "correct": len(tolerant_correct_ids),
            "accuracy": len(tolerant_correct_ids) / len(rollouts),
            "parse_failures": sum(
                row["actions"][action]["tolerant"]["answer"] is None
                for row in rollouts
            ),
            "correct_ids": tolerant_correct_ids,
        },
        "corrected": len(corrected_ids),
        "corrected_ids": corrected_ids,
        "degraded": len(degraded_ids),
        "degraded_ids": degraded_ids,
        "outcomes": {
            name: outcomes.get(name, 0)
            for name in ("helpful", "harmful", "neutral_correct", "neutral_wrong")
        },
        "incremental_cost_distribution": {
            field: _distribution([float(cost[field]) for cost in costs])
            for field in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "calls",
                "latency_seconds",
            )
        },
    }


def _posthoc_oracle(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    selection_counts = {action: 0 for action in ACTIONS}
    correct_ids: list[str | int] = []
    selected_costs: list[dict[str, Any]] = []
    selections: dict[str, str] = {}
    action_order = {action: index for index, action in enumerate(ACTIONS)}
    for row in rollouts:
        candidates = [
            (
                action,
                row["actions"][action]["correct"],
                row["actions"][action]["complete_cost"],
            )
            for action in ACTIONS
        ]
        correct = [candidate for candidate in candidates if candidate[1]]
        eligible = correct or candidates
        action, is_correct, cost = min(
            eligible,
            key=lambda item: (
                item[2]["total_tokens"],
                item[2]["calls"],
                item[2]["latency_seconds"],
                action_order[item[0]],
            ),
        )
        selection_counts[action] += 1
        selections[row["sample_id"]] = action
        selected_costs.append(cost)
        if is_correct:
            correct_ids.append(row["question_id"])
    aggregate = _combine_costs(*selected_costs)
    return {
        "posthoc_oracle": True,
        "deployable": False,
        "warning": "Uses gold outcomes after generation and is not deployable.",
        "cost_order": "total_tokens, calls, latency, STOP/SHORT/FULL tie-break",
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(rollouts),
        "correct_ids": correct_ids,
        "selection_counts": selection_counts,
        "recorded_cost": {
            "total": aggregate,
            "mean": {
                field: aggregate[field] / len(rollouts)
                for field in (*USAGE_FIELDS, "calls", "latency_seconds")
            },
        },
        "selection_by_sample_id": selections,
    }


def build_action_collection_summary(
    rollouts: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    policy_manifest: dict[str, Any],
) -> dict[str, Any]:
    if not rollouts:
        raise ValueError("Cannot summarize empty action rollouts")
    metrics = {action: _action_metrics(rollouts, action) for action in ACTIONS}
    actual_costs = [
        _combine_costs(
            row["solver"]["cost"],
            row["actions"]["SHORT"]["incremental_cost"],
            row["actions"]["FULL"]["incremental_cost"],
        )
        for row in rollouts
    ]
    actual_total = _combine_costs(*actual_costs)
    return {
        "logiqa_action_collection": True,
        "mock_only": False,
        "budget_semantics_version": 2,
        "hard_budget": "completion_tokens+calls",
        "optimization_cost": "total_tokens+calls",
        "data_split": "train",
        "samples": len(rollouts),
        "seed": split_manifest["seed"],
        "split_manifest": split_manifest,
        "continuation_policy_manifest": policy_manifest,
        "actions": metrics,
        "corrected_degraded_overlap": {
            "corrected": compare_id_sets(
                metrics["SHORT"]["corrected_ids"],
                metrics["FULL"]["corrected_ids"],
            ),
            "degraded": compare_id_sets(
                metrics["SHORT"]["degraded_ids"],
                metrics["FULL"]["degraded_ids"],
            ),
        },
        "minimum_cost_posthoc_oracle": _posthoc_oracle(rollouts),
        "actual_run": {
            "solver_calls": len(rollouts),
            "short_critic_refiner_calls": 2 * len(rollouts),
            "full_critic_refiner_calls": 2 * len(rollouts),
            "actual_calls": sum(row["actual_calls"] for row in rollouts),
            "total_cost": actual_total,
            "mean_cost_per_sample": {
                field: actual_total[field] / len(rollouts)
                for field in (*USAGE_FIELDS, "calls", "latency_seconds")
            },
            "usage_estimated": False,
        },
        "controller_training": False,
        "budget_labels_generated": False,
        "enabled_actions": list(ACTIONS),
        "disabled_actions": ["SKIP", "MEDIUM"],
    }


def _format_ids(values: list[str | int]) -> str:
    return ", ".join(map(str, values)) if values else "None"


def build_action_collection_report(summary: dict[str, Any]) -> str:
    lines = [
        "# LogiQA 2.0 train three-action paired rollout collection",
        "",
        "Real paired collection only; no controller was trained and no budget-bound labels were generated.",
        "",
        f"Samples: {summary['samples']} (seed={summary['seed']})",
        f"Actual calls: {summary['actual_run']['actual_calls']}",
        "",
        "## Accuracy and outcomes",
        "",
        "| Action | Strict acc. | Strict failures | Tolerant acc. | Tolerant failures | Corrected | Degraded | Helpful | Harmful | Neutral correct | Neutral wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for action in ACTIONS:
        metric = summary["actions"][action]
        outcome = metric["outcomes"]
        lines.append(
            f"| {action} | {metric['strict']['accuracy']:.4f} | "
            f"{metric['strict']['parse_failures']} | {metric['tolerant']['accuracy']:.4f} | "
            f"{metric['tolerant']['parse_failures']} | {metric['corrected']} | "
            f"{metric['degraded']} | {outcome['helpful']} | {outcome['harmful']} | "
            f"{outcome['neutral_correct']} | {outcome['neutral_wrong']} |"
        )
    lines.extend(["", "## Incremental cost distributions", ""])
    for action in ACTIONS:
        lines.extend(
            [
                f"### {action}",
                "",
                "| Cost | Mean | P50 | P90 | P95 | Max |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        distributions = summary["actions"][action]["incremental_cost_distribution"]
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "calls",
            "latency_seconds",
        ):
            stat = distributions[field]
            lines.append(
                f"| {field} | {stat['mean']:.4f} | {stat['p50']:.4f} | "
                f"{stat['p90']:.4f} | {stat['p95']:.4f} | {stat['max']:.4f} |"
            )
        lines.append("")
    overlap = summary["corrected_degraded_overlap"]
    lines.extend(
        [
            "## Corrected/degraded overlap",
            "",
            f"- Corrected intersection: {_format_ids(overlap['corrected']['intersection'])}",
            f"- Corrected Jaccard: {overlap['corrected']['jaccard']:.4f}",
            f"- Degraded intersection: {_format_ids(overlap['degraded']['intersection'])}",
            f"- Degraded Jaccard: {overlap['degraded']['jaccard']:.4f}",
            "",
            "## Minimum-cost posthoc oracle",
            "",
        ]
    )
    oracle = summary["minimum_cost_posthoc_oracle"]
    lines.extend(
        [
            f"- Accuracy: {oracle['accuracy']:.4f}",
            f"- STOP/SHORT/FULL selections: {oracle['selection_counts']['STOP']}/"
            f"{oracle['selection_counts']['SHORT']}/{oracle['selection_counts']['FULL']}",
            "- posthoc_oracle=true; deployable=false.",
            "",
            "## Actual service cost",
            "",
        ]
    )
    actual = summary["actual_run"]["total_cost"]
    lines.extend(
        [
            f"- Prompt tokens: {actual['prompt_tokens']}",
            f"- Completion tokens: {actual['completion_tokens']}",
            f"- Total tokens: {actual['total_tokens']}",
            f"- Calls: {actual['calls']}",
            f"- Summed request latency: {actual['latency_seconds']:.4f}s",
            "",
            "All usage is service-reported; no missing usage is estimated.",
            "",
        ]
    )
    return "\n".join(lines)


def _prepare_completed(
    rollouts_path: Path,
    selected_hashes: set[str],
    split_manifest: Path,
    policy_manifest: Path,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not rollouts_path.exists():
        return completed
    for position, row in enumerate(read_jsonl(rollouts_path), 1):
        digest = row.get("sample_id")
        if not isinstance(digest, str) or digest not in selected_hashes:
            raise ValueError(f"Existing rollout {position} is not in current split")
        if digest in completed:
            raise ValueError(f"Duplicate completed rollout sample {digest}")
        if (
            row.get("logiqa_action_collection") is not True
            or row.get("mock_only") is not False
            or row.get("split_manifest") != str(split_manifest.resolve())
            or row.get("continuation_policy_manifest")
            != str(policy_manifest.resolve())
        ):
            raise ValueError("Existing rollout belongs to a different collection")
        completed[digest] = row
    return completed


def run_logiqa_action_collection(
    data_path: str | Path,
    output_dir: str | Path,
    backend: LLMBackend,
    settings: ActionCollectionSettings,
    pilot_predictions: str | Path = DEFAULT_PILOT_PREDICTIONS,
    validation_predictions: str | Path = DEFAULT_VALIDATION_PREDICTIONS,
    sample_count: int = ACTION_COLLECTION_SAMPLES,
    seed: int = ACTION_COLLECTION_SEED,
) -> dict[str, Any]:
    if bool(getattr(backend, "mock_only", True)):
        raise ValueError(
            "collect-logiqa-action-rollouts requires a real-compatible backend; Mock is forbidden"
        )
    target = Path(output_dir)
    selected, split_payload, policy_payload = prepare_action_collection_split(
        data_path,
        target,
        settings,
        pilot_predictions,
        validation_predictions,
        sample_count,
        seed,
    )
    split_path = target / "split_manifest.json"
    policy_path = target / "continuation_policy_manifest.json"
    rollouts_path = target / "rollouts.jsonl"
    selected_hashes = {example_content_sha256(example) for example in selected}
    completed = _prepare_completed(
        rollouts_path,
        selected_hashes,
        split_path,
        policy_path,
    )
    checkpoint_dir = target / ".action_collection_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for example in selected:
        digest = example_content_sha256(example)
        if digest in completed:
            continue
        checkpoint_path = _checkpoint_path(checkpoint_dir, digest)
        checkpoint = _load_checkpoint(
            checkpoint_path,
            digest,
            split_path,
            policy_path,
        )
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
            "short_critic",
            backend,
            build_versioned_critic_messages(problem, solver_output, MINIMAL_V1),
            settings.critic_max_tokens,
            example.question_id,
        )
        _stage_call(
            checkpoint,
            checkpoint_path,
            "short_refiner",
            backend,
            build_versioned_refiner_messages(
                problem,
                solver_output,
                checkpoint["short_critic"]["raw_output"],
                MINIMAL_V1,
            ),
            settings.refiner_max_tokens,
            example.question_id,
        )
        _stage_call(
            checkpoint,
            checkpoint_path,
            "full_critic",
            backend,
            build_versioned_critic_messages(problem, solver_output, STRUCTURED_V2),
            settings.critic_max_tokens,
            example.question_id,
        )
        full_protocol = parse_critic_protocol(checkpoint["full_critic"]["raw_output"])
        _stage_call(
            checkpoint,
            checkpoint_path,
            "full_refiner",
            backend,
            build_versioned_refiner_messages(
                problem,
                solver_output,
                full_protocol.review_for_refiner,
                STRUCTURED_V2,
            ),
            settings.refiner_max_tokens,
            example.question_id,
        )
        rollout = _build_rollout(
            example,
            split_path,
            policy_path,
            checkpoint,
        )
        _append_jsonl(rollouts_path, rollout)
        completed[digest] = rollout
        checkpoint_path.unlink()
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    ordered = [completed[example_content_sha256(example)] for example in selected]
    if len(ordered) != sample_count:
        raise RuntimeError("Action collection did not complete every selected sample")
    summary = build_action_collection_summary(
        ordered,
        split_payload,
        policy_payload,
    )
    _write_json_atomic(target / "summary.json", summary)
    _write_text_atomic(target / "report.md", build_action_collection_report(summary))
    return summary
