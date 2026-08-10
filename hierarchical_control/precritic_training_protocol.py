"""Pure-offline preparation of the formal Pre-Critic training protocol."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io_utils import read_jsonl
from .logiqa_action_collection import (
    _pilot_content_hashes,
    _problem_content_sha256,
    _saved_problem_hashes,
    _sha256_payload,
    example_content_sha256,
)
from .logiqa_pilot import ANSWER_LETTERS, LogiQAExample
from .logiqa_policy_validation import load_all_logiqa_examples


LABELS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)
EXPECTED_LABEL_COUNTS = {
    "correct_to_correct": 635,
    "correct_to_wrong": 79,
    "wrong_to_correct": 64,
    "wrong_to_wrong": 222,
}
EXPECTED_SOURCE_COUNTS = {"collection_200": 200, "collection_800": 800}
FINAL_TEST_SEED = 20260815
FINAL_TEST_SAMPLES = 500

DEFAULT_COLLECTION_200 = Path("artifacts/logiqa_action_collection_200/rollouts.jsonl")
DEFAULT_COLLECTION_800 = Path("artifacts/logiqa_precritic_collection_800/rollouts.jsonl")
DEFAULT_PILOT = Path("artifacts/pilot_logiqa/predictions.jsonl")
DEFAULT_VALIDATION = Path("artifacts/logiqa_policy_validation_100/predictions.jsonl")
DEFAULT_DEV_DATA = Path("/tmp/logiqa2-dev.txt")
DEFAULT_TRAINING_OUTPUT = Path("artifacts/precritic_training_1000")
DEFAULT_FINAL_TEST_OUTPUT = Path("artifacts/logiqa_final_test_500")

# Frozen from the successfully audited formal inputs. Tests provide an explicit
# checksum contract for their temporary fixtures.
EXPECTED_SOURCE_SHA256 = {
    "collection_200": "3ab3e27ee40450be84e394cb947d9ac5659e664609c326c9233f9cd836b6ee3f",
    "collection_800": "ee00a652b696f95556b46fac82da3ab417d16335a0f6614f7ce6ed8448e1f1ae",
    "pilot_predictions": "13183369908f147441715c082847cfe6fa7a3f5687f7ed3cbccedfc7277f5786",
    "pilot_summary": "685c773d0c7f1ff96e729506b89b6c457dfff6df5f559cac1d0d4b9ca0af97d2",
    "validation_predictions": "39db4c6152c51371658205addfa35cb4841168758b3953016ad3f0823337187c",
    "dev_data": "bbefb563b7ddc02640ccdc314c1315d5727dba48539d0ecdd126fa351e511b09",
}

COLLECTION_200_COST_REASON = (
    "Collection 200 stores combined structured_v2 Critic+Refiner continuation "
    "cost, not a separate Critic stage; Critic cost is unavailable and is not "
    "estimated."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _jsonl_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _answer(value: Any, context: str, *, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    if value not in ANSWER_LETTERS:
        suffix = " or null" if allow_none else ""
        raise ValueError(f"{context} must be A-D{suffix}")
    return str(value)


def _question_id(value: Any, context: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{context} has invalid question_id")
    return value


def _problem(value: Any, context: str) -> dict[str, Any]:
    source = _mapping(value, context)
    passage = source.get("passage")
    question = source.get("question")
    options = source.get("options")
    if not isinstance(passage, str) or not isinstance(question, str):
        raise ValueError(f"{context} lacks passage or question")
    option_map = _mapping(options, f"{context} options")
    if set(option_map) != set(ANSWER_LETTERS) or not all(
        isinstance(option_map[letter], str) for letter in ANSWER_LETTERS
    ):
        raise ValueError(f"{context} must contain exactly four string A-D options")
    return {
        "passage": passage,
        "question": question,
        "options": {letter: option_map[letter] for letter in ANSWER_LETTERS},
    }


def _usage(value: Any, context: str) -> dict[str, Any]:
    source = _mapping(value, context)
    result: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        amount = source.get(field)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f"{context} has invalid {field}")
        result[field] = amount
    if result["prompt_tokens"] + result["completion_tokens"] != result["total_tokens"]:
        raise ValueError(f"{context} has inconsistent token totals")
    latency = source.get("latency_seconds")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ValueError(f"{context} has invalid latency_seconds")
    result["latency_seconds"] = float(latency)
    return result


def _parse_status(solver: dict[str, Any], context: str) -> dict[str, Any]:
    strict = _answer(solver.get("strict_answer"), f"{context} strict answer")
    tolerant = _mapping(solver.get("tolerant"), f"{context} tolerant parse")
    tolerant_answer = _answer(
        tolerant.get("answer"), f"{context} tolerant answer"
    )
    match_count = tolerant.get("match_count")
    conflict = tolerant.get("conflict")
    if isinstance(match_count, bool) or not isinstance(match_count, int) or match_count < 0:
        raise ValueError(f"{context} has invalid tolerant match_count")
    if not isinstance(conflict, bool):
        raise ValueError(f"{context} has invalid tolerant conflict")
    return {
        "strict_answer": strict,
        "strict_parse_failure": strict is None,
        "tolerant_answer": tolerant_answer,
        "tolerant_parse_failure": tolerant_answer is None,
        "tolerant_match_count": match_count,
        "tolerant_conflict": conflict,
    }


def _effective_critic_answer(
    protocol_value: Any, solver_answer: str | None, context: str
) -> str | None:
    protocol = _mapping(protocol_value, context)
    verdict = protocol.get("effective_verdict")
    proposed = _answer(
        protocol.get("effective_proposed_answer"),
        f"{context} effective proposed answer",
    )
    parse_failure = protocol.get("parse_failure")
    if not isinstance(parse_failure, bool):
        raise ValueError(f"{context} parse_failure must be boolean")
    if verdict == "KEEP":
        if proposed is not None:
            raise ValueError(f"{context} effective KEEP must not propose an answer")
        return solver_answer
    if verdict == "REVISE":
        if proposed is None:
            raise ValueError(f"{context} effective REVISE requires A-D")
        return proposed
    raise ValueError(f"{context} has invalid effective verdict")


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


def build_training_example(
    row: dict[str, Any],
    *,
    dataset: str,
    source_path: Path,
    source_sha256: str,
    source_row: int,
) -> dict[str, Any]:
    """Build one whitelist-only record from a saved real rollout."""
    if dataset not in {"collection_200", "collection_800"}:
        raise ValueError(f"Unknown training source: {dataset}")
    if row.get("mock_only") is not False:
        raise ValueError(f"{dataset} row {source_row} is not a real rollout")
    state = _mapping(row.get("state_for_controller"), f"{dataset} row {source_row} state")
    problem = _problem(state.get("problem"), f"{dataset} row {source_row} problem")
    digest = row.get("sample_id")
    if not isinstance(digest, str) or digest != _problem_content_sha256(problem):
        raise ValueError(f"{dataset} row {source_row} sample_id is not its content SHA256")
    question_id = _question_id(row.get("question_id"), f"{dataset} row {source_row}")
    if state.get("sample_id") != digest or state.get("question_id") != question_id:
        raise ValueError(f"{dataset} row {source_row} controller identity mismatch")
    solver = _mapping(row.get("solver"), f"{dataset} row {source_row} Solver")
    raw_output = solver.get("raw_output")
    if not isinstance(raw_output, str) or state.get("solver_raw_output") != raw_output:
        raise ValueError(f"{dataset} row {source_row} Solver state mismatch")
    parse_status = _parse_status(solver, f"{dataset} row {source_row} Solver")
    solver_answer = parse_status["tolerant_answer"]
    solver_usage = _usage(solver.get("cost"), f"{dataset} row {source_row} Solver cost")
    if solver_usage["calls"] != 1:
        raise ValueError(f"{dataset} row {source_row} Solver must have one call")

    if dataset == "collection_200":
        actions = _mapping(row.get("actions"), f"{dataset} row {source_row} actions")
        full = _mapping(actions.get("FULL"), f"{dataset} row {source_row} FULL")
        protocol = full.get("critic_protocol")
        cost_available = False
        cost_target = None
    else:
        protocol = row.get("critic")
        critic = _mapping(protocol, f"{dataset} row {source_row} Critic")
        cost_target = _usage(
            critic.get("cost"), f"{dataset} row {source_row} Critic cost"
        )
        if cost_target["calls"] != 1:
            raise ValueError(f"{dataset} row {source_row} Critic must have one call")
        cost_available = True

    critic_answer = _effective_critic_answer(
        protocol, solver_answer, f"{dataset} row {source_row} Critic protocol"
    )
    gold = _answer(row.get("gold"), f"{dataset} row {source_row} gold", allow_none=False)
    label = _transition(solver_answer, critic_answer, gold)
    return {
        "sample_id": digest,
        "content_sha256": digest,
        "source": {
            "dataset": dataset,
            "path": str(source_path.resolve()),
            "sha256": source_sha256,
            "row": source_row,
            "question_id": question_id,
        },
        "model_input": {
            "problem": problem,
            "solver": {
                "raw_output": raw_output,
                "parse_status": parse_status,
                "usage": solver_usage,
            },
        },
        "label": label,
        "cost_available": cost_available,
        "critic_cost_target": cost_target,
    }


def merge_training_rollouts(
    collection_200: Sequence[dict[str, Any]],
    collection_800: Sequence[dict[str, Any]],
    *,
    source_paths: Mapping[str, Path],
    source_sha256: Mapping[str, str],
    expected_source_counts: Mapping[str, int] = EXPECTED_SOURCE_COUNTS,
    expected_label_counts: Mapping[str, int] = EXPECTED_LABEL_COUNTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = {
        "collection_200": collection_200,
        "collection_800": collection_800,
    }
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_counts: dict[str, int] = {}
    for dataset, rows in sources.items():
        expected = expected_source_counts.get(dataset)
        if expected is None or len(rows) != expected:
            raise ValueError(
                f"Expected exactly {expected} rows from {dataset}; found {len(rows)}"
            )
        source_counts[dataset] = len(rows)
        for index, row in enumerate(rows, 1):
            record = build_training_example(
                row,
                dataset=dataset,
                source_path=source_paths[dataset],
                source_sha256=source_sha256[dataset],
                source_row=index,
            )
            digest = record["sample_id"]
            if digest in seen:
                raise ValueError(f"Duplicate training content SHA256: {digest}")
            seen.add(digest)
            records.append(record)
    labels = Counter(record["label"] for record in records)
    actual_labels = {label: labels.get(label, 0) for label in LABELS}
    required_labels = {label: expected_label_counts.get(label) for label in LABELS}
    if actual_labels != required_labels:
        raise ValueError(
            "Training label counts differ from the frozen contract: "
            f"expected {required_labels}, got {actual_labels}"
        )
    available = sum(record["cost_available"] for record in records)
    if available != source_counts["collection_800"]:
        raise ValueError("Only Collection 800 may provide Critic cost targets")
    typed_ids = {
        json.dumps(
            [
                type(record["source"]["question_id"]).__name__,
                record["source"]["question_id"],
            ],
            ensure_ascii=False,
        )
        for record in records
    }
    return records, {
        "samples": len(records),
        "unique_content_sha256": len(seen),
        "unique_question_id": len(typed_ids),
        "source_counts": source_counts,
        "label_counts": actual_labels,
        "cost_available": available,
        "cost_unavailable": len(records) - available,
    }


def select_final_test_split(
    examples: Sequence[LogiQAExample],
    excluded_hashes: set[str],
    *,
    data_path: Path,
    data_sha256: str,
    seed: int = FINAL_TEST_SEED,
    sample_count: int = FINAL_TEST_SAMPLES,
) -> tuple[dict[str, Any], set[str]]:
    if sample_count <= 0:
        raise ValueError("Final-test sample_count must be positive")
    unique: list[tuple[LogiQAExample, str]] = []
    seen: set[str] = set()
    duplicates = 0
    excluded = 0
    for example in examples:
        digest = example_content_sha256(example)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        if digest in excluded_hashes:
            excluded += 1
            continue
        unique.append((example, digest))
    if len(unique) < sample_count:
        raise ValueError(
            f"Only {len(unique)} unused, content-unique dev samples remain; "
            f"need {sample_count}"
        )
    indices = random.Random(seed).sample(range(len(unique)), sample_count)
    selected = [unique[index] for index in indices]
    selected_rows = [
        {"question_id": example.question_id, "content_sha256": digest}
        for example, digest in selected
    ]
    selected_hashes = {row["content_sha256"] for row in selected_rows}
    if len(selected_hashes) != sample_count or selected_hashes & excluded_hashes:
        raise AssertionError("Final-test content isolation failed")
    split_identity = {
        "data_sha256": data_sha256,
        "seed": seed,
        "sample_count": sample_count,
        "selected_samples": selected_rows,
    }
    manifest = {
        "final_test": True,
        "sealed": True,
        "never_evaluated": True,
        "mock_only": False,
        "backend_initialized": False,
        "model_calls": 0,
        "seed": seed,
        "sample_count": sample_count,
        "data_path": str(data_path.resolve()),
        "data_sha256": data_sha256,
        "content_identity": "SHA256(normalized passage, question, ordered A-D options)",
        "selection_stats": {
            "data_records": len(examples),
            "unique_content_records": len(seen),
            "within_dev_duplicates_excluded": duplicates,
            "previously_used_content_excluded": excluded,
            "eligible_content_records": len(unique),
        },
        "selected_samples": selected_rows,
        "split_sha256": _sha256_payload(split_identity),
        "split_sha256_scope": (
            "data_sha256,seed,sample_count,ordered selected_samples"
        ),
        "use_restrictions": {
            "prompt_development": False,
            "feature_selection": False,
            "threshold_selection": False,
            "hyperparameter_selection": False,
            "evaluation_allowed_only_after_protocol_freeze": True,
        },
    }
    return manifest, selected_hashes


def _source_contract(
    paths: Mapping[str, Path], expected: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    if set(paths) != set(expected):
        raise ValueError("Expected SHA256 contract does not cover every source file")
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Required source file is missing: {resolved}")
        digest = _file_sha256(resolved)
        if digest != expected[name]:
            raise ValueError(
                f"Source SHA256 mismatch for {name}: expected {expected[name]}, got {digest}"
            )
        result[name] = {"path": str(resolved), "sha256": digest}
    return result


def _ensure_new_outputs(training_dir: Path, final_dir: Path) -> None:
    targets = (
        training_dir / "training_examples.jsonl",
        training_dir / "manifest.json",
        training_dir / "report.md",
        final_dir / "split_manifest.json",
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing formal training protocol: "
            + ", ".join(existing)
        )


def _report(manifest: dict[str, Any]) -> str:
    labels = manifest["label_counts"]
    overlap = manifest["content_overlap_checks"]
    return "\n".join(
        [
            "# Formal Pre-Critic training protocol",
            "",
            "Pure offline preparation: model_calls=0; controller_trained=false.",
            "",
            f"Training samples: {manifest['samples']}",
            f"Unique content SHA256: {manifest['unique_content_sha256']}",
            f"Unique source question IDs: {manifest['unique_question_id']} (not an identity key)",
            "",
            "## Labels",
            "",
            *[f"- {label}: {labels[label]}" for label in LABELS],
            "",
            "## Critic cost targets",
            "",
            f"- available, service-reported: {manifest['cost_targets']['available_samples']}",
            f"- unavailable and not estimated: {manifest['cost_targets']['unavailable_samples']}",
            "",
            "## Content isolation",
            "",
            *[f"- {key}: {value}" for key, value in overlap.items()],
            "",
            "## Sealed final test",
            "",
            f"- samples: {manifest['final_test']['sample_count']}",
            f"- split_sha256: {manifest['final_test']['split_sha256']}",
            "- final_test=true; sealed=true; never_evaluated=true; model_calls=0",
            "- It is prohibited for prompt, feature, threshold, or hyperparameter selection.",
            "",
        ]
    )


def prepare_precritic_training_protocol(
    *,
    collection_200_path: str | Path = DEFAULT_COLLECTION_200,
    collection_800_path: str | Path = DEFAULT_COLLECTION_800,
    pilot_predictions_path: str | Path = DEFAULT_PILOT,
    validation_predictions_path: str | Path = DEFAULT_VALIDATION,
    dev_data_path: str | Path = DEFAULT_DEV_DATA,
    training_output_dir: str | Path = DEFAULT_TRAINING_OUTPUT,
    final_test_output_dir: str | Path = DEFAULT_FINAL_TEST_OUTPUT,
    expected_source_sha256: Mapping[str, str] | None = None,
    expected_source_counts: Mapping[str, int] = EXPECTED_SOURCE_COUNTS,
    expected_label_counts: Mapping[str, int] = EXPECTED_LABEL_COUNTS,
    final_test_seed: int = FINAL_TEST_SEED,
    final_test_samples: int = FINAL_TEST_SAMPLES,
) -> dict[str, Any]:
    """Create the formal training dataset and sealed final split, entirely offline."""
    collection_200_path = Path(collection_200_path)
    collection_800_path = Path(collection_800_path)
    pilot_predictions_path = Path(pilot_predictions_path)
    validation_predictions_path = Path(validation_predictions_path)
    dev_data_path = Path(dev_data_path)
    pilot_summary_path = pilot_predictions_path.parent / "summary.json"
    training_dir = Path(training_output_dir)
    final_dir = Path(final_test_output_dir)
    _ensure_new_outputs(training_dir, final_dir)

    source_paths = {
        "collection_200": collection_200_path,
        "collection_800": collection_800_path,
        "pilot_predictions": pilot_predictions_path,
        "pilot_summary": pilot_summary_path,
        "validation_predictions": validation_predictions_path,
        "dev_data": dev_data_path,
    }
    source_files = _source_contract(
        source_paths,
        EXPECTED_SOURCE_SHA256
        if expected_source_sha256 is None
        else expected_source_sha256,
    )
    collection_200 = read_jsonl(collection_200_path)
    collection_800 = read_jsonl(collection_800_path)
    records, merge_stats = merge_training_rollouts(
        collection_200,
        collection_800,
        source_paths={
            "collection_200": collection_200_path,
            "collection_800": collection_800_path,
        },
        source_sha256={
            "collection_200": source_files["collection_200"]["sha256"],
            "collection_800": source_files["collection_800"]["sha256"],
        },
        expected_source_counts=expected_source_counts,
        expected_label_counts=expected_label_counts,
    )
    training_hashes = {record["sample_id"] for record in records}

    pilot_hashes = _pilot_content_hashes(pilot_predictions_path.resolve())
    validation_hashes = _saved_problem_hashes(validation_predictions_path.resolve())
    if training_hashes & pilot_hashes or training_hashes & validation_hashes:
        raise ValueError("Training content overlaps Pilot or Validation")
    dev_examples = load_all_logiqa_examples(dev_data_path)
    final_manifest, final_hashes = select_final_test_split(
        dev_examples,
        training_hashes | pilot_hashes | validation_hashes,
        data_path=dev_data_path,
        data_sha256=source_files["dev_data"]["sha256"],
        seed=final_test_seed,
        sample_count=final_test_samples,
    )
    overlap_checks = {
        "collection_200_vs_collection_800": 0,
        "training_vs_pilot": len(training_hashes & pilot_hashes),
        "training_vs_validation": len(training_hashes & validation_hashes),
        "final_test_vs_training": len(final_hashes & training_hashes),
        "final_test_vs_pilot": len(final_hashes & pilot_hashes),
        "final_test_vs_validation": len(final_hashes & validation_hashes),
        "final_test_internal_duplicates": final_test_samples - len(final_hashes),
    }
    if any(overlap_checks.values()):
        raise AssertionError(f"Content isolation failed: {overlap_checks}")

    training_bytes = _jsonl_bytes(records)
    final_bytes = _json_bytes(final_manifest)
    manifest = {
        "precritic_training_protocol": True,
        "offline_prepare": True,
        "mock_only": False,
        "backend_initialized": False,
        "model_calls": 0,
        "controller_trained": False,
        "identity_key": "sample_id == content_sha256",
        "question_id_used_for_identity_or_join": False,
        **merge_stats,
        "source_files": source_files,
        "source_sha256_contract_enforced": True,
        "training_examples_sha256": hashlib.sha256(training_bytes).hexdigest(),
        "label_semantics": {
            "policy": "effective structured_v2 Critic-only",
            "KEEP": "use Solver tolerant answer",
            "REVISE": "use effective proposed A-D answer",
            "classes": list(LABELS),
            "gold_used_only_for_offline_label": True,
        },
        "feature_schema": {
            "allowed": [
                "problem.passage",
                "problem.question",
                "problem.options.A-D",
                "solver.raw_output",
                "solver.parse_status",
                "solver.usage",
            ],
            "forbidden": [
                "gold",
                "Critic output",
                "Refiner output",
                "continuation answer",
                "action outcome",
            ],
        },
        "cost_targets": {
            "available_samples": merge_stats["cost_available"],
            "unavailable_samples": merge_stats["cost_unavailable"],
            "regression_eligible_samples": merge_stats["cost_available"],
            "collection_800_service_reported": True,
            "collection_200_estimated": False,
            "collection_200_unavailable_reason": COLLECTION_200_COST_REASON,
        },
        "content_overlap_checks": overlap_checks,
        "final_test": {
            "path": str((final_dir / "split_manifest.json").resolve()),
            "manifest_sha256": hashlib.sha256(final_bytes).hexdigest(),
            "split_sha256": final_manifest["split_sha256"],
            "sample_count": final_test_samples,
            "final_test": True,
            "sealed": True,
            "never_evaluated": True,
            "model_calls": 0,
        },
    }
    report = _report(manifest).encode("utf-8")
    _atomic_write(training_dir / "training_examples.jsonl", training_bytes)
    _atomic_write(training_dir / "manifest.json", _json_bytes(manifest))
    _atomic_write(training_dir / "report.md", report)
    _atomic_write(final_dir / "split_manifest.json", final_bytes)
    return {
        "precritic_training_protocol": True,
        "offline_prepare": True,
        "model_calls": 0,
        "controller_trained": False,
        "training_samples": merge_stats["samples"],
        "label_counts": merge_stats["label_counts"],
        "cost_regression_samples": merge_stats["cost_available"],
        "training_output_dir": str(training_dir.resolve()),
        "final_test_samples": final_test_samples,
        "final_test_split_sha256": final_manifest["split_sha256"],
        "final_test_output_dir": str(final_dir.resolve()),
    }
