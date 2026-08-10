from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .backend import LLMBackend
from .io_utils import read_jsonl
from .logiqa_action_collection import (
    ACTION_COLLECTION_SAMPLES,
    DEFAULT_PILOT_PREDICTIONS,
    DEFAULT_VALIDATION_PREDICTIONS,
    ActionCollectionSettings,
    _git_commit,
    _policy_prompt_hashes,
    _problem_content_sha256,
    _sha256_payload,
    content_sha256,
    example_content_sha256,
    load_excluded_content_hashes,
    select_action_collection_examples,
)
from .logiqa_audit import tolerant_final_answer
from .logiqa_pilot import (
    ANSWER_LETTERS,
    LogiQAExample,
    build_solver_messages,
    extract_final_answer,
)
from .logiqa_policy_validation import (
    _append_jsonl,
    _completion_usage,
    _latency,
    _timed_complete,
    _usage_dict,
    _write_json_atomic,
    _write_text_atomic,
    load_all_logiqa_examples,
)
from .logiqa_prompts import STRUCTURED_V2, build_versioned_critic_messages
from .logiqa_replay import parse_critic_protocol


PRECRITIC_SEED = 20260814
PRECRITIC_SAMPLES = 800
DEFAULT_EXISTING_COLLECTION = Path(
    "artifacts/logiqa_action_collection_200/rollouts.jsonl"
)
TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _saved_collection_hashes(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Existing Collection 200 rollouts not found: {path}")
    hashes: set[str] = set()
    for position, row in enumerate(read_jsonl(path), 1):
        state = row.get("state_for_controller")
        problem = state.get("problem") if isinstance(state, dict) else None
        if not isinstance(problem, dict):
            raise ValueError(f"Existing collection row {position} lacks saved problem")
        digest = _problem_content_sha256(problem)
        sample_id = row.get("sample_id")
        if sample_id is not None and sample_id != digest:
            raise ValueError(
                f"Existing collection row {position} content hash does not match sample_id"
            )
        hashes.add(digest)
    return hashes


def load_precritic_excluded_hashes(
    pilot_predictions: str | Path = DEFAULT_PILOT_PREDICTIONS,
    validation_predictions: str | Path = DEFAULT_VALIDATION_PREDICTIONS,
    existing_collection: str | Path = DEFAULT_EXISTING_COLLECTION,
) -> tuple[set[str], dict[str, Any]]:
    previous, stats = load_excluded_content_hashes(
        pilot_predictions, validation_predictions
    )
    collection_path = Path(existing_collection).resolve()
    collection_hashes = _saved_collection_hashes(collection_path)
    combined = previous | collection_hashes
    return combined, {
        **stats,
        "existing_collection": str(collection_path),
        "existing_collection_unique_content": len(collection_hashes),
        "combined_unique_content": len(combined),
    }


def _split_identity(
    *,
    source: Path,
    source_sha256: str,
    seed: int,
    sample_count: int,
    selected: list[LogiQAExample],
) -> dict[str, Any]:
    return {
        "data_path": str(source),
        "data_sha256": source_sha256,
        "seed": seed,
        "sample_count": sample_count,
        "selected_samples": [
            {
                "question_id": example.question_id,
                "content_sha256": example_content_sha256(example),
            }
            for example in selected
        ],
    }


def _manifest_payload(
    *,
    source: Path,
    source_sha256: str,
    selected: list[LogiQAExample],
    selection_stats: dict[str, Any],
    exclusion_stats: dict[str, Any],
    settings: ActionCollectionSettings,
    repository: Path,
    sample_count: int,
    seed: int,
    mock_only: bool,
) -> dict[str, Any]:
    split_identity = _split_identity(
        source=source,
        source_sha256=source_sha256,
        seed=seed,
        sample_count=sample_count,
        selected=selected,
    )
    critic_sha = _policy_prompt_hashes(STRUCTURED_V2)["critic_prompt_sha256"]
    return {
        "precritic_collection": True,
        "prepared_only": True,
        "mock_only": bool(mock_only),
        "backend_initialized": False,
        "model_calls": 0,
        "data_split": "train",
        "seed": seed,
        "sample_count": sample_count,
        "data_path": str(source),
        "data_sha256": source_sha256,
        "content_normalization": "Unicode NFKC + casefold + collapsed whitespace",
        "content_hash": "SHA256(canonical passage, question, ordered options)",
        "selection_stats": selection_stats,
        "exclusion_sources": exclusion_stats,
        "selected_samples": split_identity["selected_samples"],
        "split_sha256": _sha256_payload(split_identity),
        "split_sha256_scope": (
            "data_path,data_sha256,seed,sample_count,ordered selected_samples"
        ),
        "workflow": ["solver", "structured_v2_critic"],
        "calls_per_sample": 2,
        "critic_policy": {
            "prompt_version": STRUCTURED_V2,
            "critic_prompt_sha256": critic_sha,
            "keep": "return saved Solver answer",
            "revise": "return effective proposed A-D answer",
            "malformed_fallback": "effective KEEP",
        },
        "model_config": {
            "base_url": settings.base_url,
            "model": settings.model,
            "temperature": settings.temperature,
            "generation_caps": {
                "solver": settings.solver_max_tokens,
                "critic": settings.critic_max_tokens,
            },
            "extra_body": settings.extra_body,
        },
        "source_validation_summary": str(settings.source_validation_summary),
        "git_commit": _git_commit(repository),
    }


def prepare_precritic_collection(
    data_path: str | Path,
    output_dir: str | Path,
    settings: ActionCollectionSettings,
    *,
    pilot_predictions: str | Path = DEFAULT_PILOT_PREDICTIONS,
    validation_predictions: str | Path = DEFAULT_VALIDATION_PREDICTIONS,
    existing_collection: str | Path = DEFAULT_EXISTING_COLLECTION,
    sample_count: int = PRECRITIC_SAMPLES,
    seed: int = PRECRITIC_SEED,
    mock_only: bool = False,
) -> dict[str, Any]:
    source = Path(data_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Official LogiQA train.txt not found: {source}")
    target = Path(output_dir)
    manifest_path = target / "split_manifest.json"
    forbidden = (
        target / "rollouts.jsonl",
        target / "summary.json",
        target / "report.md",
    )
    if not manifest_path.exists() and any(path.exists() for path in forbidden):
        raise ValueError("Collection artifacts exist without a split manifest")
    examples = load_all_logiqa_examples(source)
    excluded, exclusion_stats = load_precritic_excluded_hashes(
        pilot_predictions,
        validation_predictions,
        existing_collection,
    )
    selected, selection_stats = select_action_collection_examples(
        examples,
        excluded,
        sample_count=sample_count,
        seed=seed,
    )
    payload = _manifest_payload(
        source=source,
        source_sha256=_file_sha256(source),
        selected=selected,
        selection_stats=selection_stats,
        exclusion_stats=exclusion_stats,
        settings=settings,
        repository=Path.cwd().resolve(),
        sample_count=sample_count,
        seed=seed,
        mock_only=mock_only,
    )
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("Existing Pre-Critic split manifest does not match")
    else:
        target.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(manifest_path, payload)
    return payload


def _selected_from_manifest(
    source: Path, manifest: dict[str, Any]
) -> list[LogiQAExample]:
    if _file_sha256(source) != manifest.get("data_sha256"):
        raise ValueError("LogiQA train file changed after split preparation")
    all_examples = load_all_logiqa_examples(source)
    by_hash: dict[str, LogiQAExample] = {}
    for example in all_examples:
        by_hash.setdefault(example_content_sha256(example), example)
    selected: list[LogiQAExample] = []
    selected_rows = manifest.get("selected_samples")
    if not isinstance(selected_rows, list):
        raise ValueError("Split manifest lacks selected_samples")
    for row in selected_rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid selected sample in manifest")
        digest = row.get("content_sha256")
        example = by_hash.get(digest)
        if example is None or example.question_id != row.get("question_id"):
            raise ValueError("Cannot reconstruct a selected sample from train.txt")
        selected.append(example)
    identity = _split_identity(
        source=source,
        source_sha256=manifest["data_sha256"],
        seed=manifest["seed"],
        sample_count=manifest["sample_count"],
        selected=selected,
    )
    if _sha256_payload(identity) != manifest.get("split_sha256"):
        raise ValueError("Split manifest SHA256 verification failed")
    return selected


def _validate_policy_manifest(
    manifest: dict[str, Any], settings: ActionCollectionSettings
) -> None:
    expected_sha = _policy_prompt_hashes(STRUCTURED_V2)["critic_prompt_sha256"]
    policy = manifest.get("critic_policy")
    config = manifest.get("model_config")
    expected_config = {
        "base_url": settings.base_url,
        "model": settings.model,
        "temperature": settings.temperature,
        "generation_caps": {
            "solver": settings.solver_max_tokens,
            "critic": settings.critic_max_tokens,
        },
        "extra_body": settings.extra_body,
    }
    if (
        not isinstance(policy, dict)
        or policy.get("prompt_version") != STRUCTURED_V2
        or policy.get("critic_prompt_sha256") != expected_sha
    ):
        raise ValueError("structured_v2 Critic prompt changed after split preparation")
    if config != expected_config:
        raise ValueError("Model configuration changed after split preparation")


def _checkpoint_path(directory: Path, digest: str) -> Path:
    return directory / f"{digest}.json"


def _load_checkpoint(
    path: Path,
    digest: str,
    split_path: Path,
    split_sha256: str,
    mock_only: bool,
) -> dict[str, Any]:
    identity = {
        "content_sha256": digest,
        "split_manifest": str(split_path.resolve()),
        "split_sha256": split_sha256,
        "mock_only": bool(mock_only),
    }
    if not path.exists():
        return identity
    payload = json.loads(path.read_text(encoding="utf-8"))
    if any(payload.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Pre-Critic checkpoint identity mismatch: {path}")
    return payload


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
        "calls": 1,
        "latency_seconds": latency,
    }
    _write_json_atomic(checkpoint_path, checkpoint)


def _stage_cost(stage: dict[str, Any]) -> dict[str, Any]:
    usage = _usage_dict(stage.get("usage"), "Pre-Critic stage usage")
    calls = stage.get("calls")
    if calls != 1:
        raise ValueError("Every saved Pre-Critic stage must contain exactly one call")
    return {
        **usage,
        "calls": 1,
        "latency_seconds": _latency(
            stage.get("latency_seconds"), "Pre-Critic stage latency"
        ),
    }


def _combine_costs(*costs: dict[str, Any]) -> dict[str, Any]:
    payload = {
        field: sum(int(cost[field]) for cost in costs) for field in USAGE_FIELDS
    }
    checked = _usage_dict(payload, "combined Pre-Critic cost")
    return {
        **checked,
        "calls": sum(int(cost["calls"]) for cost in costs),
        "latency_seconds": sum(float(cost["latency_seconds"]) for cost in costs),
    }


def _problem_payload(example: LogiQAExample) -> dict[str, Any]:
    return {
        "passage": example.passage,
        "question": example.question,
        "options": {
            letter: option for letter, option in zip(ANSWER_LETTERS, example.options)
        },
    }


def _transition(solver_answer: str | None, critic_answer: str | None, gold: str) -> str:
    solver_correct = solver_answer == gold
    critic_correct = critic_answer == gold
    if solver_correct and critic_correct:
        return "correct_to_correct"
    if solver_correct:
        return "correct_to_wrong"
    if critic_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _build_rollout(
    example: LogiQAExample,
    manifest: dict[str, Any],
    split_path: Path,
    checkpoint: dict[str, Any],
    mock_only: bool,
) -> dict[str, Any]:
    solver_stage = checkpoint["solver"]
    critic_stage = checkpoint["structured_v2_critic"]
    solver_raw = solver_stage["raw_output"]
    critic_raw = critic_stage["raw_output"]
    solver_tolerant = tolerant_final_answer(solver_raw).to_dict()
    solver_answer = solver_tolerant["answer"]
    protocol = parse_critic_protocol(critic_raw)
    critic_answer = (
        solver_answer
        if protocol.effective_verdict == "KEEP"
        else protocol.effective_proposed_answer
    )
    if protocol.effective_verdict == "REVISE" and critic_answer not in ANSWER_LETTERS:
        raise ValueError("Effective REVISE lacks a valid proposed answer")
    solver_cost = _stage_cost(solver_stage)
    critic_cost = _stage_cost(critic_stage)
    problem_text = example.problem_text()
    label = _transition(solver_answer, critic_answer, example.gold)
    return {
        "precritic_collection": True,
        "prepared_only": False,
        "mock_only": bool(mock_only),
        "sample_id": example_content_sha256(example),
        "question_id": example.question_id,
        "gold": example.gold,
        "split_manifest": str(split_path.resolve()),
        "split_sha256": manifest["split_sha256"],
        "prompt_version": STRUCTURED_V2,
        "critic_prompt_sha256": manifest["critic_policy"][
            "critic_prompt_sha256"
        ],
        "model_config": manifest["model_config"],
        "git_commit": manifest["git_commit"],
        "solver_called_once": True,
        "actual_calls": 2,
        "state_for_controller": {
            "sample_id": example_content_sha256(example),
            "question_id": example.question_id,
            "problem": _problem_payload(example),
            "problem_and_choices": problem_text,
            "solver_raw_output": solver_raw,
            "solver_strict_answer": extract_final_answer(solver_raw),
            "solver_tolerant": solver_tolerant,
            "solver_usage": solver_cost,
            "generation_pre_critic_state": {
                "query": problem_text,
                "current_answer": solver_raw,
                "history": build_solver_messages(problem_text)
                + [
                    {
                        "role": "assistant",
                        "name": "solver",
                        "content": solver_raw,
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
            "raw_output": solver_raw,
            "strict_answer": extract_final_answer(solver_raw),
            "tolerant": solver_tolerant,
            "answer": solver_answer,
            "correct": solver_answer == example.gold,
            "cost": solver_cost,
        },
        "critic": {
            "raw_output": critic_raw,
            **protocol.to_dict(),
            "cost": critic_cost,
        },
        "actions": {
            "STOP": {
                "answer": solver_answer,
                "correct": solver_answer == example.gold,
                "cost": solver_cost,
            },
            "CRITIC_ONLY": {
                "answer": critic_answer,
                "correct": critic_answer == example.gold,
                "transition": label,
                "complete_cost": _combine_costs(solver_cost, critic_cost),
                "incremental_cost": critic_cost,
            },
        },
        "label": label,
    }


def _prepare_completed(
    path: Path, selected_hashes: set[str], split_path: Path, split_sha256: str
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for position, row in enumerate(read_jsonl(path), 1):
        digest = row.get("sample_id")
        if digest not in selected_hashes:
            raise ValueError(f"Rollout {position} is not part of the prepared split")
        if digest in completed:
            raise ValueError(f"Duplicate completed Pre-Critic rollout: {digest}")
        if (
            row.get("split_manifest") != str(split_path.resolve())
            or row.get("split_sha256") != split_sha256
        ):
            raise ValueError(f"Rollout {position} has a different split identity")
        completed[digest] = row
    return completed


def _summary(
    rollouts: list[dict[str, Any]], manifest: dict[str, Any], mock_only: bool
) -> dict[str, Any]:
    transitions = Counter(row["label"] for row in rollouts)
    solver_correct = sum(row["actions"]["STOP"]["correct"] for row in rollouts)
    critic_correct = sum(
        row["actions"]["CRITIC_ONLY"]["correct"] for row in rollouts
    )
    total_cost = _combine_costs(
        *[
            _combine_costs(row["solver"]["cost"], row["critic"]["cost"])
            for row in rollouts
        ]
    )
    count = len(rollouts)
    return {
        "precritic_collection": True,
        "prepared_only": False,
        "mock_only": bool(mock_only),
        "samples": count,
        "split_sha256": manifest["split_sha256"],
        "prompt_version": STRUCTURED_V2,
        "critic_prompt_sha256": manifest["critic_policy"][
            "critic_prompt_sha256"
        ],
        "solver": {"correct": solver_correct, "accuracy": solver_correct / count},
        "critic_only": {
            "correct": critic_correct,
            "accuracy": critic_correct / count,
        },
        "transitions": {
            label: transitions.get(label, 0) for label in TRANSITIONS
        },
        "actual_calls": sum(row["actual_calls"] for row in rollouts),
        "total_cost": total_cost,
        "mean_cost": {
            field: total_cost[field] / count
            for field in (*USAGE_FIELDS, "calls", "latency_seconds")
        },
        "usage_estimated": False,
    }


def _report(summary: dict[str, Any]) -> str:
    transitions = summary["transitions"]
    return "\n".join(
        [
            "# LogiQA Pre-Critic paired collection",
            "",
            f"mock_only={str(summary['mock_only']).lower()}",
            f"Samples: {summary['samples']}",
            f"Actual calls: {summary['actual_calls']}",
            "",
            f"Solver accuracy: {summary['solver']['accuracy']:.4f}",
            f"Critic-only accuracy: {summary['critic_only']['accuracy']:.4f}",
            "",
            f"- correct→correct: {transitions['correct_to_correct']}",
            f"- correct→wrong: {transitions['correct_to_wrong']}",
            f"- wrong→correct: {transitions['wrong_to_correct']}",
            f"- wrong→wrong: {transitions['wrong_to_wrong']}",
            "",
            "All usage is backend-reported; no token cost is estimated.",
            "",
        ]
    )


def collect_precritic_rollouts(
    data_path: str | Path,
    output_dir: str | Path,
    backend: LLMBackend,
    settings: ActionCollectionSettings,
) -> dict[str, Any]:
    target = Path(output_dir)
    split_path = target / "split_manifest.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            "Run prepare-precritic-collection before collecting rollouts"
        )
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if manifest.get("precritic_collection") is not True or manifest.get(
        "prepared_only"
    ) is not True:
        raise ValueError("Invalid prepared Pre-Critic split manifest")
    backend_mock = bool(getattr(backend, "mock_only", True))
    if manifest.get("mock_only") is not backend_mock:
        raise ValueError("Prepared split mock_only does not match backend mode")
    _validate_policy_manifest(manifest, settings)
    source = Path(data_path).resolve()
    if str(source) != manifest.get("data_path"):
        raise ValueError("collect data_path differs from the prepared split")
    selected = _selected_from_manifest(source, manifest)
    selected_hashes = {example_content_sha256(example) for example in selected}
    rollouts_path = target / "rollouts.jsonl"
    completed = _prepare_completed(
        rollouts_path,
        selected_hashes,
        split_path,
        manifest["split_sha256"],
    )
    checkpoint_dir = target / ".precritic_checkpoints"
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
            manifest["split_sha256"],
            backend_mock,
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
        _stage_call(
            checkpoint,
            checkpoint_path,
            "structured_v2_critic",
            backend,
            build_versioned_critic_messages(
                problem,
                checkpoint["solver"]["raw_output"],
                STRUCTURED_V2,
            ),
            settings.critic_max_tokens,
            example.question_id,
        )
        rollout = _build_rollout(
            example,
            manifest,
            split_path,
            checkpoint,
            backend_mock,
        )
        _append_jsonl(rollouts_path, rollout)
        completed[digest] = rollout
        checkpoint_path.unlink()
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass
    ordered = [completed[example_content_sha256(example)] for example in selected]
    if len(ordered) != manifest["sample_count"]:
        raise RuntimeError("Pre-Critic collection did not finish every sample")
    summary = _summary(ordered, manifest, backend_mock)
    _write_json_atomic(target / "summary.json", summary)
    _write_text_atomic(target / "report.md", _report(summary))
    return summary
