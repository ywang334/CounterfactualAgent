"""Pure offline audit for the completed Pre-Critic Controller v1 run.

The audit is intentionally read-only with respect to the formal training inputs and
outputs.  It never constructs an LLM backend, never trains a model, and inspects only
the sealed Final Test manifest (never the referenced examples).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .io_utils import read_jsonl, write_jsonl
from .logiqa_audit import exact_mcnemar
from .precritic_controller_v1 import (
    ALL_SEEDS,
    BUDGET_RATES,
    EXPECTED_TRAINING_SAMPLES,
    EXPECTED_VALIDATION_SAMPLES,
    HIDDEN_DIM,
    LABELS,
    MODEL_NAME,
    PRIMARY_SEED,
    STABILITY_SEEDS,
    TRAINING_SHA256,
    PreCriticControllerV1,
    TrainingExample,
    deterministic_stratified_folds,
    gated_decisions,
    load_old_probe_decisions,
    load_training_examples,
    load_validation_examples,
    oof_budget_thresholds_v1,
    select_oof_threshold_v1,
    verify_sealed_final_manifest,
)
from .precritic_probe import (
    NUMERIC_FEATURES,
    OfflineMiniLMEncoder,
    ProbeExample,
    _hash_model_input,
)


ARTIFACT_NAMES = (
    "primary_model.pt",
    "seed_metrics.json",
    "oof_predictions.jsonl",
    "validation_predictions.jsonl",
    "summary.json",
    "report.md",
)
DEFAULT_CONTROLLER_DIR = Path("artifacts/precritic_controller_v1")
DEFAULT_TRAINING = Path("artifacts/precritic_training_1000/training_examples.jsonl")
DEFAULT_TRAINING_MANIFEST = Path("artifacts/precritic_training_1000/manifest.json")
DEFAULT_VALIDATION = Path("artifacts/logiqa_policy_validation_100/predictions.jsonl")
DEFAULT_OLD_PROBE = Path("artifacts/precritic_gate_probe/predictions.jsonl")
DEFAULT_OLD_PROBE_SUMMARY = Path("artifacts/precritic_gate_probe/summary.json")
DEFAULT_FINAL_MANIFEST = Path("artifacts/logiqa_final_test_500/split_manifest.json")
DEFAULT_RUN_METADATA_PREFIX = Path("/tmp/counterfactualagent_precritic_controller_v1")
DEFAULT_AUDIT_DIR = DEFAULT_CONTROLLER_DIR / "audit"
FLOAT_ATOL = 1e-6
TOKEN_ATOL = 1e-3


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _typed_id(value: str | int) -> str:
    return json.dumps([type(value).__name__, value], ensure_ascii=False)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _assert_close(actual: Any, expected: Any, context: str, atol: float = FLOAT_ATOL) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError(f"{context} differs: {actual!r} != {expected!r}")
        return
    if isinstance(expected, int):
        if actual != expected:
            raise ValueError(f"{context} differs: {actual!r} != {expected!r}")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=atol
        ):
            raise ValueError(f"{context} differs: {actual!r} != {expected!r}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{context} list shape differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_close(actual_item, expected_item, f"{context}[{index}]", atol)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"{context} is not an object")
        for key, expected_value in expected.items():
            if key not in actual:
                raise ValueError(f"{context} lacks {key!r}")
            _assert_close(actual[key], expected_value, f"{context}.{key}", atol)
        return
    if actual != expected:
        raise ValueError(f"{context} differs")


def verify_run_status(prefix: str | Path) -> dict[str, Any]:
    prefix = Path(prefix)
    pid_path = Path(f"{prefix}.pid")
    exitcode_path = Path(f"{prefix}.exitcode")
    log_path = Path(f"{prefix}.log")
    for path in (pid_path, exitcode_path, log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty training run metadata: {path}")
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        exitcode = int(exitcode_path.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError("Training PID or exit code is malformed") from exc
    if pid <= 0:
        raise ValueError("Training PID must be positive")
    process_alive = Path(f"/proc/{pid}").exists()
    if process_alive:
        raise RuntimeError(f"Training process {pid} is still running")
    if exitcode != 0:
        raise RuntimeError(f"Training failed with exit code {exitcode}")
    started_ns = pid_path.stat().st_mtime_ns
    finished_ns = exitcode_path.stat().st_mtime_ns
    if finished_ns < started_ns:
        raise ValueError("Training run timestamps are inconsistent")
    return {
        "pid": pid,
        "process_alive": False,
        "process_exited": True,
        "exitcode": exitcode,
        "successful": True,
        "started_at_epoch_seconds": started_ns / 1_000_000_000,
        "finished_at_epoch_seconds": finished_ns / 1_000_000_000,
        "wall_clock_seconds": (finished_ns - started_ns) / 1_000_000_000,
        "duration_evidence": "run.pid mtime to run.exitcode mtime",
        "metadata": {
            "pid": {"path": str(pid_path), "sha256": _file_sha256(pid_path)},
            "exitcode": {
                "path": str(exitcode_path),
                "sha256": _file_sha256(exitcode_path),
            },
            "log": {
                "path": str(log_path),
                "sha256": _file_sha256(log_path),
                "bytes": log_path.stat().st_size,
            },
        },
    }


def verify_artifacts(controller_dir: str | Path) -> dict[str, dict[str, Any]]:
    controller_dir = Path(controller_dir)
    result: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_NAMES:
        path = controller_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty Controller v1 artifact: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
            "nonempty": True,
        }
    return result


def _average_precision(binary_targets: Sequence[bool], scores: Sequence[float]) -> float:
    if not binary_targets or len(binary_targets) != len(scores):
        raise ValueError("PR-AUC inputs must be non-empty and aligned")
    positives = sum(binary_targets)
    if positives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    cursor = 0
    while cursor < len(order):
        score = float(scores[order[cursor]])
        end = cursor
        while end < len(order) and float(scores[order[end]]) == score:
            if binary_targets[order[end]]:
                true_positive += 1
            else:
                false_positive += 1
            end += 1
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        cursor = end
    return area


def _macro_f1(labels: Sequence[str], probabilities: Sequence[Sequence[float]]) -> float:
    if len(labels) != len(probabilities):
        raise ValueError("Macro-F1 inputs are not aligned")
    actual = [LABELS.index(label) for label in labels]
    predicted = [max(range(len(LABELS)), key=lambda index: row[index]) for row in probabilities]
    class_scores = []
    for class_index in range(len(LABELS)):
        true_positive = sum(
            prediction == class_index and target == class_index
            for prediction, target in zip(predicted, actual)
        )
        false_positive = sum(
            prediction == class_index and target != class_index
            for prediction, target in zip(predicted, actual)
        )
        false_negative = sum(
            prediction != class_index and target == class_index
            for prediction, target in zip(predicted, actual)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(class_scores) / len(class_scores)


def _head_metrics(
    labels: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    predicted_cost_tokens: Sequence[float],
    actual_cost_tokens: Sequence[float],
    cost_available: Sequence[bool],
) -> dict[str, Any]:
    if not (
        len(labels)
        == len(probabilities)
        == len(predicted_cost_tokens)
        == len(actual_cost_tokens)
        == len(cost_available)
    ):
        raise ValueError("Head metric inputs are not aligned")
    cost_errors = [
        abs(float(prediction) - float(actual))
        for prediction, actual, available in zip(
            predicted_cost_tokens, actual_cost_tokens, cost_available
        )
        if available
    ]
    return {
        "helpful_pr_auc": _average_precision(
            [label == "wrong_to_correct" for label in labels],
            [row[LABELS.index("wrong_to_correct")] for row in probabilities],
        ),
        "harmful_pr_auc": _average_precision(
            [label == "correct_to_wrong" for label in labels],
            [row[LABELS.index("correct_to_wrong")] for row in probabilities],
        ),
        "four_class_macro_f1": _macro_f1(labels, probabilities),
        "critic_incremental_total_tokens_mae": (
            statistics.fmean(cost_errors) if cost_errors else None
        ),
        "cost_mae_samples": len(cost_errors),
    }


def _label_policy_metrics(
    labels: Sequence[str], decisions: Sequence[bool]
) -> dict[str, Any]:
    if len(labels) != len(decisions):
        raise ValueError("Policy inputs are not aligned")
    transitions: Counter[str] = Counter()
    for label, decision in zip(labels, decisions):
        if decision:
            transition = label
        elif label.startswith("correct_to_"):
            transition = "correct_to_correct"
        else:
            transition = "wrong_to_wrong"
        transitions[transition] += 1
    corrected = transitions["wrong_to_correct"]
    degraded = transitions["correct_to_wrong"]
    correct = transitions["correct_to_correct"] + corrected
    return {
        "samples": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels),
        "corrected": corrected,
        "degraded": degraded,
        "net_benefit": corrected - degraded,
        "critic_calls": sum(decisions),
        "critic_call_rate": sum(decisions) / len(labels),
        "transitions": {label: transitions.get(label, 0) for label in LABELS},
    }


def _validation_policy_metrics(
    examples: Sequence[ProbeExample], decisions: Sequence[bool]
) -> dict[str, Any]:
    if len(examples) != len(decisions):
        raise ValueError("Validation policy inputs are not aligned")
    transitions: dict[str, list[str | int]] = {label: [] for label in LABELS}
    correct_ids: list[str | int] = []
    selected_costs = []
    for example, decision in zip(examples, decisions):
        solver_correct = example.solver_answer == example.gold
        selected_answer = example.critic_only_answer if decision else example.solver_answer
        selected_correct = selected_answer == example.gold
        if solver_correct and selected_correct:
            transition = "correct_to_correct"
        elif solver_correct:
            transition = "correct_to_wrong"
        elif selected_correct:
            transition = "wrong_to_correct"
        else:
            transition = "wrong_to_wrong"
        transitions[transition].append(example.question_id)
        if selected_correct:
            correct_ids.append(example.question_id)
        key = "CRITIC_ONLY" if decision else "STOP"
        cost = _require_mapping(
            example.audit_case["strategy_costs"].get(key), f"{key} saved cost"
        )
        if cost.get("available") is not True:
            raise ValueError("Validation stage cost is unavailable")
        if cost["prompt_tokens"] + cost["completion_tokens"] != cost["total_tokens"]:
            raise ValueError("Validation token identity failed")
        selected_costs.append(cost)
    total = {
        field: sum(cost[field] for cost in selected_costs)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "calls",
            "latency_seconds",
        )
    }
    count = len(examples)
    corrected_ids = transitions["wrong_to_correct"]
    degraded_ids = transitions["correct_to_wrong"]
    return {
        "samples": count,
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / count,
        "correct_ids": correct_ids,
        "corrected": len(corrected_ids),
        "corrected_ids": corrected_ids,
        "degraded": len(degraded_ids),
        "degraded_ids": degraded_ids,
        "net_benefit": len(corrected_ids) - len(degraded_ids),
        "critic_calls": sum(decisions),
        "critic_call_rate": sum(decisions) / count,
        "transitions": {
            label: {"count": len(ids), "sample_ids": ids}
            for label, ids in transitions.items()
        },
        "cost": {
            "service_reported_usage": True,
            "estimated": False,
            "total": total,
            "mean": {field: total[field] / count for field in total},
        },
    }


def _mcnemar(
    examples: Sequence[ProbeExample],
    first: Sequence[bool],
    second: Sequence[bool],
    first_name: str,
    second_name: str,
) -> dict[str, Any]:
    first_correct = [
        (example.critic_only_answer if decision else example.solver_answer)
        == example.gold
        for example, decision in zip(examples, first)
    ]
    second_correct = [
        (example.critic_only_answer if decision else example.solver_answer)
        == example.gold
        for example, decision in zip(examples, second)
    ]
    first_only = sum(a and not b for a, b in zip(first_correct, second_correct))
    second_only = sum(not a and b for a, b in zip(first_correct, second_correct))
    return {
        "first_policy": first_name,
        "second_policy": second_name,
        "first_correct_second_wrong": first_only,
        "first_wrong_second_correct": second_only,
        **exact_mcnemar(first_only, second_only),
    }


def _probability_vector(row: Mapping[str, Any], context: str) -> list[float]:
    payload = _require_mapping(row.get("probabilities"), f"{context} probabilities")
    if set(payload) != set(LABELS):
        raise ValueError(f"{context} has invalid probability classes")
    result = []
    for label in LABELS:
        value = payload[label]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context} has invalid probability")
        result.append(float(value))
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{context} probabilities do not sum to one")
    return result


def _fold_manifest_audit(
    labels: Sequence[str], seed: int, rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, list[int]]], dict[str, int]]:
    expected = deterministic_stratified_folds(labels, 5, seed)
    fold_by_index = {
        index: fold["fold"] for fold in expected for index in fold["validation"]
    }
    if len(fold_by_index) != len(labels):
        raise ValueError(f"Seed {seed} OOF folds do not cover Training 1000")
    for index, row in enumerate(rows):
        if row.get("oof_fold") != fold_by_index[index]:
            raise ValueError(f"Seed {seed} row {index} has the wrong OOF fold")
    seen: set[int] = set()
    sizes: dict[str, int] = {}
    for fold in expected:
        training = set(fold["train"])
        validation = set(fold["validation"])
        if training & validation or training | validation != set(range(len(labels))):
            raise ValueError(f"Seed {seed} fold {fold['fold']} overlaps")
        if seen & validation:
            raise ValueError(f"Seed {seed} held-out folds overlap")
        seen |= validation
        sizes[str(fold["fold"])] = len(validation)
    if seen != set(range(len(labels))):
        raise ValueError(f"Seed {seed} held-out folds are incomplete")
    return expected, sizes


def _audit_oof(
    rows: Sequence[dict[str, Any]],
    training_examples: Sequence[TrainingExample],
    saved_seed_metrics: Mapping[int, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_ids = [example.sample_id for example in training_examples]
    labels = [example.label for example in training_examples]
    actual_costs = [math.expm1(example.cost_log_target) for example in training_examples]
    cost_available = [example.cost_available for example in training_examples]
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in ALL_SEEDS}
    for row in rows:
        seed = row.get("seed")
        if seed not in by_seed:
            raise ValueError(f"Unexpected OOF seed: {seed!r}")
        by_seed[seed].append(row)
    recomputed: dict[str, Any] = {}
    audit_cases: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        seed_rows = by_seed[seed]
        if len(seed_rows) != EXPECTED_TRAINING_SAMPLES:
            raise ValueError(f"Seed {seed} must have exactly 1000 OOF rows")
        row_ids = [row.get("sample_id") for row in seed_rows]
        if row_ids != expected_ids or len(set(row_ids)) != len(row_ids):
            raise ValueError(f"Seed {seed} OOF identity/order differs from Training 1000")
        folds, fold_sizes = _fold_manifest_audit(labels, seed, seed_rows)
        probabilities: list[list[float]] = []
        scores: list[float] = []
        predicted_costs: list[float] = []
        for index, (row, example) in enumerate(zip(seed_rows, training_examples)):
            context = f"OOF seed {seed} row {index}"
            if (
                row.get("label") != example.label
                or row.get("source_dataset") != example.source_dataset
                or row.get("cost_available") is not example.cost_available
            ):
                raise ValueError(f"{context} differs from the frozen training example")
            vector = _probability_vector(row, context)
            score = vector[LABELS.index("wrong_to_correct")] - vector[
                LABELS.index("correct_to_wrong")
            ]
            _assert_close(row.get("gate_score"), score, f"{context} gate score")
            predicted_cost = row.get("predicted_critic_incremental_total_tokens")
            if not isinstance(predicted_cost, (int, float)) or predicted_cost < 0:
                raise ValueError(f"{context} has invalid predicted cost")
            probabilities.append(vector)
            scores.append(score)
            predicted_costs.append(float(predicted_cost))
        threshold = select_oof_threshold_v1(scores, labels)
        decisions = gated_decisions(scores, threshold["threshold"])
        budget_thresholds = oof_budget_thresholds_v1(scores)
        development_policy = _label_policy_metrics(labels, decisions)
        budget_curve = []
        for point in budget_thresholds:
            point_decisions = gated_decisions(scores, point["threshold"])
            budget_curve.append(
                {**point, "oof_policy": _label_policy_metrics(labels, point_decisions)}
            )
        head_metrics = _head_metrics(
            labels, probabilities, predicted_costs, actual_costs, cost_available
        )
        saved = saved_seed_metrics[seed]
        _assert_close(saved["oof"]["fold_manifest"], folds, f"seed {seed} fold manifest")
        _assert_close(
            saved["oof"]["development_threshold"], threshold, f"seed {seed} threshold"
        )
        _assert_close(
            saved["oof"]["development_policy"],
            development_policy,
            f"seed {seed} OOF policy",
        )
        _assert_close(saved["oof"]["budget_curve"], budget_curve, f"seed {seed} budget curve")
        _assert_close(saved["oof"]["head_metrics"], head_metrics, f"seed {seed} OOF heads", TOKEN_ATOL)
        for index, (row, decision) in enumerate(zip(seed_rows, decisions)):
            if row.get("critic_called") is not decision:
                raise ValueError(f"Seed {seed} OOF row {index} action differs")
            _assert_close(
                row.get("development_threshold"),
                threshold["threshold"],
                f"Seed {seed} OOF row {index} threshold",
            )
            policy_transition = (
                row["label"]
                if decision
                else (
                    "correct_to_correct"
                    if row["label"].startswith("correct_to_")
                    else "wrong_to_wrong"
                )
            )
            audit_cases.append(
                {
                    "offline_audit": True,
                    "controller_retrained": False,
                    "model_calls": 0,
                    "record_type": "oof_prediction",
                    "seed": seed,
                    "primary_seed": seed == PRIMARY_SEED,
                    "sample_id": row["sample_id"],
                    "label": row["label"],
                    "fold": row["oof_fold"],
                    "probabilities": row["probabilities"],
                    "gate_score": scores[index],
                    "threshold": threshold["threshold"],
                    "critic_called": decision,
                    "policy_transition": policy_transition,
                    "cost_available": row["cost_available"],
                    "predicted_critic_incremental_total_tokens": predicted_costs[index],
                    "saved_values_match_recomputation": True,
                }
            )
        recomputed[str(seed)] = {
            "role": "primary" if seed == PRIMARY_SEED else "stability_only",
            "samples": len(seed_rows),
            "unique_samples": len(set(row_ids)),
            "label_counts": {
                label: sum(item == label for item in labels) for label in LABELS
            },
            "folds_partition_once": True,
            "fold_sizes": fold_sizes,
            "development_threshold": threshold,
            "development_policy": development_policy,
            "budget_curve": budget_curve,
            "head_metrics": head_metrics,
            "saved_metrics_match": True,
        }
    return recomputed, audit_cases


def _validation_actual_costs(examples: Sequence[ProbeExample]) -> list[float]:
    result = []
    for example in examples:
        stop = example.audit_case["strategy_costs"]["STOP"]
        critic = example.audit_case["strategy_costs"]["CRITIC_ONLY"]
        incremental = critic["total_tokens"] - stop["total_tokens"]
        if incremental < 0 or critic["calls"] - stop["calls"] != 1:
            raise ValueError("Validation Critic incremental cost is inconsistent")
        result.append(float(incremental))
    return result


def _stability(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    accessors = {
        "accuracy": lambda item: item["controller_policy"]["accuracy"],
        "corrected": lambda item: item["controller_policy"]["corrected"],
        "degraded": lambda item: item["controller_policy"]["degraded"],
        "net_benefit": lambda item: item["controller_policy"]["net_benefit"],
        "critic_call_rate": lambda item: item["controller_policy"]["critic_call_rate"],
        "mean_total_tokens": lambda item: item["controller_policy"]["cost"]["mean"]["total_tokens"],
        "mean_calls": lambda item: item["controller_policy"]["cost"]["mean"]["calls"],
        "mean_latency_seconds": lambda item: item["controller_policy"]["cost"]["mean"]["latency_seconds"],
        "helpful_pr_auc": lambda item: item["head_metrics"]["helpful_pr_auc"],
        "harmful_pr_auc": lambda item: item["head_metrics"]["harmful_pr_auc"],
        "four_class_macro_f1": lambda item: item["head_metrics"]["four_class_macro_f1"],
        "critic_cost_mae": lambda item: item["head_metrics"]["critic_incremental_total_tokens_mae"],
    }
    return {
        "seeds": [item["seed"] for item in seed_results],
        "population_standard_deviation": True,
        "metrics": {
            name: {
                "mean": statistics.fmean(float(accessor(item)) for item in seed_results),
                "std": statistics.pstdev(float(accessor(item)) for item in seed_results),
            }
            for name, accessor in accessors.items()
        },
    }


def _audit_validation(
    rows: Sequence[dict[str, Any]],
    examples: Sequence[ProbeExample],
    saved_seed_metrics: Mapping[int, dict[str, Any]],
    oof_results: Mapping[str, dict[str, Any]],
    old_probe_decisions: Sequence[bool],
    saved_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    labels = [example.label for example in examples]
    expected_ids = [_typed_id(example.question_id) for example in examples]
    actual_costs = _validation_actual_costs(examples)
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in ALL_SEEDS}
    for row in rows:
        seed = row.get("seed")
        if seed not in by_seed:
            raise ValueError(f"Unexpected Validation seed: {seed!r}")
        by_seed[seed].append(row)
    seed_results = []
    audit_cases: list[dict[str, Any]] = []
    for seed in ALL_SEEDS:
        seed_rows = by_seed[seed]
        if len(seed_rows) != EXPECTED_VALIDATION_SAMPLES:
            raise ValueError(f"Seed {seed} must have exactly 100 Validation rows")
        row_ids = [_typed_id(row.get("question_id")) for row in seed_rows]
        if row_ids != expected_ids or len(set(row_ids)) != len(row_ids):
            raise ValueError(f"Seed {seed} Validation identity/order differs")
        probabilities: list[list[float]] = []
        scores: list[float] = []
        predicted_costs: list[float] = []
        threshold = oof_results[str(seed)]["development_threshold"]["threshold"]
        for index, (row, example) in enumerate(zip(seed_rows, examples)):
            context = f"Validation seed {seed} row {index}"
            if row.get("label") != example.label:
                raise ValueError(f"{context} label differs")
            if row.get("model_input_sha256") != _hash_model_input(example):
                raise ValueError(f"{context} model input hash differs")
            if row.get("validation_used_for_threshold") is not False:
                raise ValueError(f"{context} improperly uses Validation for thresholding")
            vector = _probability_vector(row, context)
            score = vector[LABELS.index("wrong_to_correct")] - vector[
                LABELS.index("correct_to_wrong")
            ]
            _assert_close(row.get("gate_score"), score, f"{context} score")
            _assert_close(row.get("development_threshold"), threshold, f"{context} threshold")
            decision = score >= threshold
            if row.get("critic_called") is not decision:
                raise ValueError(f"{context} action differs")
            selected = example.critic_only_answer if decision else example.solver_answer
            if row.get("selected_answer") != selected or row.get("correct") is not (
                selected == example.gold
            ):
                raise ValueError(f"{context} selected result differs")
            predicted_cost = row.get("predicted_critic_incremental_total_tokens")
            if not isinstance(predicted_cost, (int, float)) or predicted_cost < 0:
                raise ValueError(f"{context} has invalid cost prediction")
            _assert_close(
                row.get("actual_critic_incremental_total_tokens"),
                actual_costs[index],
                f"{context} actual cost",
                TOKEN_ATOL,
            )
            probabilities.append(vector)
            scores.append(score)
            predicted_costs.append(float(predicted_cost))
        decisions = gated_decisions(scores, threshold)
        controller_policy = _validation_policy_metrics(examples, decisions)
        head_metrics = _head_metrics(
            labels,
            probabilities,
            predicted_costs,
            actual_costs,
            [True] * len(examples),
        )
        budget_curve = []
        for point in oof_results[str(seed)]["budget_curve"]:
            point_decisions = gated_decisions(scores, point["threshold"])
            budget_curve.append(
                {
                    "target_budget_rate": point["target_budget_rate"],
                    "threshold": point["threshold"],
                    "threshold_source": point["threshold_source"],
                    "validation_used_for_threshold": False,
                    "deployment_operating_point_selected": False,
                    "validation_policy": _validation_policy_metrics(
                        examples, point_decisions
                    ),
                }
            )
        saved = saved_seed_metrics[seed]["validation"]
        _assert_close(saved["controller_policy"], controller_policy, f"seed {seed} Validation policy")
        _assert_close(saved["head_metrics"], head_metrics, f"seed {seed} Validation heads", TOKEN_ATOL)
        _assert_close(saved["budget_curve"], budget_curve, f"seed {seed} Validation curve")
        seed_result = {
            "seed": seed,
            "role": "primary" if seed == PRIMARY_SEED else "stability_only",
            "controller_policy": controller_policy,
            "head_metrics": head_metrics,
            "budget_curve": budget_curve,
            "saved_metrics_match": True,
        }
        seed_results.append(seed_result)
        for index, (row, example, decision) in enumerate(
            zip(seed_rows, examples, decisions)
        ):
            selected_cost = example.audit_case["strategy_costs"][
                "CRITIC_ONLY" if decision else "STOP"
            ]
            audit_cases.append(
                {
                    "offline_audit": True,
                    "controller_retrained": False,
                    "model_calls": 0,
                    "record_type": "validation_prediction",
                    "seed": seed,
                    "primary_seed": seed == PRIMARY_SEED,
                    "question_id": example.question_id,
                    "label": example.label,
                    "probabilities": row["probabilities"],
                    "gate_score": scores[index],
                    "threshold": threshold,
                    "critic_called": decision,
                    "selected_answer": row["selected_answer"],
                    "correct": row["correct"],
                    "selected_saved_cost": {
                        key: selected_cost[key]
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "calls",
                            "latency_seconds",
                        )
                    },
                    "saved_values_match_recomputation": True,
                }
            )

    primary = next(item for item in seed_results if item["seed"] == PRIMARY_SEED)
    primary_decisions = [bool(row["critic_called"]) for row in by_seed[PRIMARY_SEED]]
    stop_decisions = [False] * len(examples)
    always_decisions = [True] * len(examples)
    oracle_decisions = [example.label == "wrong_to_correct" for example in examples]
    comparison = {
        "STOP": _validation_policy_metrics(examples, stop_decisions),
        "ALWAYS_CRITIC_ONLY": _validation_policy_metrics(examples, always_decisions),
        "OLD_PROBE": {
            **_validation_policy_metrics(examples, old_probe_decisions),
            "frozen_historical_policy": True,
        },
        "CONTROLLER_V1_PRIMARY": {
            **primary["controller_policy"],
            "seed": PRIMARY_SEED,
            "threshold_source": "training_1000_stratified_5fold_oof_only",
        },
        "POSTHOC_ORACLE": {
            **_validation_policy_metrics(examples, oracle_decisions),
            "posthoc_oracle": True,
            "deployable": False,
        },
    }
    mcnemar = {
        "controller_vs_stop": _mcnemar(
            examples,
            primary_decisions,
            stop_decisions,
            "CONTROLLER_V1_PRIMARY",
            "STOP",
        ),
        "controller_vs_always": _mcnemar(
            examples,
            primary_decisions,
            always_decisions,
            "CONTROLLER_V1_PRIMARY",
            "ALWAYS_CRITIC_ONLY",
        ),
    }
    stability = _stability(seed_results)
    stored_validation = saved_summary["development_validation"]
    _assert_close(stored_validation["policy_comparison"], comparison, "saved policy comparison")
    _assert_close(stored_validation["mcnemar"], mcnemar, "saved McNemar")
    _assert_close(saved_summary["stability"], stability, "saved stability", TOKEN_ATOL)
    return (
        {
            "samples": len(examples),
            "seeds": seed_results,
            "policy_comparison": comparison,
            "stability": stability,
            "mcnemar": mcnemar,
            "saved_summary_matches": True,
        },
        audit_cases,
        by_seed,
    )


def _cpu_checkpoint_replay(
    checkpoint_path: Path,
    examples: Sequence[ProbeExample],
    primary_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("primary_seed") != PRIMARY_SEED
        or checkpoint.get("training_sha256") != TRAINING_SHA256
        or checkpoint.get("validation_used_for_training_or_selection") is not False
        or checkpoint.get("final_test_evaluated") is not False
    ):
        raise ValueError("Primary checkpoint violates the frozen training contract")
    state = _require_mapping(checkpoint.get("model_state_dict"), "checkpoint state")
    state_devices = sorted({str(tensor.device) for tensor in state.values()})
    if state_devices != ["cpu"]:
        raise ValueError(f"Checkpoint tensors are not all on CPU: {state_devices}")
    model = PreCriticControllerV1(
        embedding_dim=int(checkpoint["embedding_dim"]),
        numeric_dim=int(checkpoint["numeric_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(state)
    model.eval()
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise ValueError("Replayed Controller model is not on CPU")
    encoder = OfflineMiniLMEncoder(model_name=MODEL_NAME, device="cpu")
    encoder_parameters = list(encoder.model.parameters())
    if (
        {parameter.device.type for parameter in encoder_parameters} != {"cpu"}
        or any(parameter.requires_grad for parameter in encoder_parameters)
    ):
        raise ValueError("Frozen replay encoder is not entirely frozen on CPU")
    feature_texts = [example.feature_text for example in examples]
    embeddings = encoder.encode(feature_texts).to(dtype=torch.float32, device="cpu")
    numeric = torch.tensor([example.numeric for example in examples], dtype=torch.float32)
    numeric_mean = checkpoint["numeric_mean"].to(dtype=torch.float32, device="cpu")
    numeric_std = checkpoint["numeric_std"].to(dtype=torch.float32, device="cpu")
    standardized = (numeric - numeric_mean) / numeric_std
    with torch.no_grad():
        logits, cost_logs = model(embeddings, standardized)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_costs = torch.expm1(torch.clamp(cost_logs, min=0.0, max=20.0))
    tensors = {
        "embeddings": embeddings,
        "numeric": numeric,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "standardized_numeric": standardized,
        "logits": logits,
        "probabilities": probabilities,
        "predicted_costs": predicted_costs,
    }
    tensor_devices = {name: str(tensor.device) for name, tensor in tensors.items()}
    if set(tensor_devices.values()) != {"cpu"}:
        raise ValueError("Not all replay tensors are on CPU")
    threshold = float(checkpoint["development_threshold"]["threshold"])
    max_probability_difference = 0.0
    max_score_difference = 0.0
    max_cost_difference = 0.0
    replay_cases = []
    for index, (example, saved) in enumerate(zip(examples, primary_rows)):
        replay_vector = [float(value) for value in probabilities[index]]
        saved_vector = [float(saved["probabilities"][label]) for label in LABELS]
        probability_difference = max(
            abs(actual - expected)
            for actual, expected in zip(replay_vector, saved_vector)
        )
        replay_score = replay_vector[LABELS.index("wrong_to_correct")] - replay_vector[
            LABELS.index("correct_to_wrong")
        ]
        score_difference = abs(replay_score - float(saved["gate_score"]))
        cost_difference = abs(
            float(predicted_costs[index])
            - float(saved["predicted_critic_incremental_total_tokens"])
        )
        replay_decision = replay_score >= threshold
        if probability_difference > FLOAT_ATOL or score_difference > FLOAT_ATOL:
            raise ValueError(f"Checkpoint replay score differs at Validation row {index}")
        if cost_difference > TOKEN_ATOL:
            raise ValueError(f"Checkpoint replay cost differs at Validation row {index}")
        if replay_decision is not saved["critic_called"]:
            raise ValueError(f"Checkpoint replay action differs at Validation row {index}")
        max_probability_difference = max(max_probability_difference, probability_difference)
        max_score_difference = max(max_score_difference, score_difference)
        max_cost_difference = max(max_cost_difference, cost_difference)
        replay_cases.append(
            {
                "question_id": example.question_id,
                "saved_gate_score": float(saved["gate_score"]),
                "replayed_gate_score": replay_score,
                "saved_critic_called": bool(saved["critic_called"]),
                "replayed_critic_called": replay_decision,
                "max_probability_abs_difference": probability_difference,
                "cost_abs_difference": cost_difference,
            }
        )
    return (
        {
            "checkpoint_loaded_map_location": "cpu",
            "checkpoint_state_tensor_devices": state_devices,
            "model_parameter_devices": sorted(
                {parameter.device.type for parameter in model.parameters()}
            ),
            "encoder_parameter_devices": sorted(
                {parameter.device.type for parameter in encoder_parameters}
            ),
            "encoder_frozen": not any(
                parameter.requires_grad for parameter in encoder_parameters
            ),
            "replay_tensor_devices": tensor_devices,
            "samples": len(examples),
            "scores_within_tolerance": True,
            "actions_exact_match": True,
            "probability_atol": FLOAT_ATOL,
            "cost_token_atol": TOKEN_ATOL,
            "max_probability_abs_difference": max_probability_difference,
            "max_gate_score_abs_difference": max_score_difference,
            "max_predicted_cost_abs_difference": max_cost_difference,
        },
        replay_cases,
    )


def _training_device_evidence() -> dict[str, Any]:
    source_path = Path(__file__).with_name("precritic_controller_v1.py")
    source = source_path.read_text(encoding="utf-8")
    required_fragments = ('device="cpu"', 'device="cpu")')
    if not all(fragment in source for fragment in required_fragments):
        raise ValueError("Training source no longer contains the fixed CPU device contract")
    forbidden_fragments = (".cuda(", 'device="cuda', ".to(\"cuda")
    if any(fragment in source for fragment in forbidden_fragments):
        raise ValueError("Training source contains an unexpected CUDA path")
    return {
        "training_device": "cpu",
        "gpu_used": False,
        "source_path": str(source_path.resolve()),
        "source_sha256": _file_sha256(source_path),
        "evidence": [
            "OfflineMiniLMEncoder is constructed with device=cpu",
            "encoded embeddings are explicitly transferred to CPU",
            "numeric tensors and Controller parameters are created on CPU",
            "no CUDA transfer path exists in the frozen training implementation",
        ],
    }


def _report(summary: Mapping[str, Any]) -> str:
    comparison = summary["validation_recomputed"]["policy_comparison"]
    primary_oof = summary["oof_recomputed"][str(PRIMARY_SEED)]
    primary_validation = next(
        item
        for item in summary["validation_recomputed"]["seeds"]
        if item["seed"] == PRIMARY_SEED
    )
    lines = [
        "# Pre-Critic Controller v1 Offline Audit",
        "",
        "`offline_audit=true`, `controller_retrained=false`, `model_calls=0`, "
        "`final_test_evaluated=false`, and `deployable=false`.",
        "",
        "The audit recomputes metrics from saved JSONL predictions and replays only "
        "the frozen Controller checkpoint plus local MiniLM on CPU. It does not train, "
        "initialize an LLM/backend, alter thresholds, or read Final Test examples.",
        "",
        "## Run and artifacts",
        "",
        f"- Exit code: {summary['run_status']['exitcode']}",
        f"- Process exited: {summary['run_status']['process_exited']}",
        f"- Observed wall clock: {summary['run_status']['wall_clock_seconds']:.6f} s",
        f"- Training device: {summary['device_audit']['training_device']} "
        f"(GPU used: {summary['device_audit']['gpu_used']})",
        "",
        "| Artifact | Bytes | SHA256 |",
        "|---|---:|---|",
    ]
    for name, item in summary["artifact_integrity"].items():
        lines.append(f"| {name} | {item['bytes']} | `{item['sha256']}` |")
    lines.extend(
        [
            "",
            "## Frozen protocol",
            "",
            f"- Training examples: {summary['training_protocol_audit']['samples']}",
            f"- Training SHA256: `{summary['training_protocol_audit']['training_sha256']}`",
            f"- Primary seed: {summary['training_protocol_audit']['primary_seed']}",
            "- Stability seeds: "
            + ", ".join(
                str(seed)
                for seed in summary["training_protocol_audit"]["stability_seeds"]
            ),
            f"- Cost targets available/masked: "
            f"{summary['leakage_and_cost_audit']['cost_available_samples']}/"
            f"{summary['leakage_and_cost_audit']['cost_masked_samples']}",
            "- Validation used for training/OOF/thresholds: false",
            "- Hyperparameter search / best-seed selection: false / false",
            "- Final Test: manifest-only verification; sealed and never evaluated",
            "",
            "## Validation 100 recomputation",
            "",
            "| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean total tokens | Mean calls | Mean latency (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in (
        "STOP",
        "ALWAYS_CRITIC_ONLY",
        "OLD_PROBE",
        "CONTROLLER_V1_PRIMARY",
        "POSTHOC_ORACLE",
    ):
        metric = comparison[name]
        lines.append(
            f"| {name} | {metric['accuracy']:.4f} | {metric['corrected']} | "
            f"{metric['degraded']} | {metric['net_benefit']} | "
            f"{metric['critic_call_rate']:.4f} | "
            f"{metric['cost']['mean']['total_tokens']:.2f} | "
            f"{metric['cost']['mean']['calls']:.2f} | "
            f"{metric['cost']['mean']['latency_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`POSTHOC_ORACLE` is a minimum-cost retrospective oracle and remains "
            "`deployable=false`.",
            "",
            "## Primary-seed head metrics",
            "",
            "| Split | Helpful PR-AUC | Harmful PR-AUC | Macro-F1 | Cost MAE |",
            "|---|---:|---:|---:|---:|",
            f"| OOF | {primary_oof['head_metrics']['helpful_pr_auc']:.6f} | "
            f"{primary_oof['head_metrics']['harmful_pr_auc']:.6f} | "
            f"{primary_oof['head_metrics']['four_class_macro_f1']:.6f} | "
            f"{primary_oof['head_metrics']['critic_incremental_total_tokens_mae']:.3f} |",
            f"| Validation | {primary_validation['head_metrics']['helpful_pr_auc']:.6f} | "
            f"{primary_validation['head_metrics']['harmful_pr_auc']:.6f} | "
            f"{primary_validation['head_metrics']['four_class_macro_f1']:.6f} | "
            f"{primary_validation['head_metrics']['critic_incremental_total_tokens_mae']:.3f} |",
            "",
            "## Primary OOF-derived budget curve",
            "",
            "All points are audit-only; no operating point is selected.",
            "",
            "| Budget | Threshold | OOF calls | OOF acc. | OOF corr./degr./net | Val calls | Val acc. | Val corr./degr./net | Mean val tokens |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for oof_point, validation_point in zip(
        primary_oof["budget_curve"], primary_validation["budget_curve"]
    ):
        oof_policy = oof_point["oof_policy"]
        validation_policy = validation_point["validation_policy"]
        lines.append(
            f"| {oof_point['target_budget_rate']:.0%} | "
            f"{oof_point['threshold']:.8f} | {oof_policy['critic_calls']} | "
            f"{oof_policy['accuracy']:.4f} | "
            f"{oof_policy['corrected']}/{oof_policy['degraded']}/{oof_policy['net_benefit']} | "
            f"{validation_policy['critic_calls']} | {validation_policy['accuracy']:.4f} | "
            f"{validation_policy['corrected']}/{validation_policy['degraded']}/"
            f"{validation_policy['net_benefit']} | "
            f"{validation_policy['cost']['mean']['total_tokens']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Stability across five fixed seeds",
            "",
            "Population mean ± standard deviation; no best seed was selected.",
            "",
        ]
    )
    for name, metric in summary["validation_recomputed"]["stability"]["metrics"].items():
        lines.append(f"- {name}: {metric['mean']:.6f} ± {metric['std']:.6f}")
    mcnemar = summary["validation_recomputed"]["mcnemar"]
    lines.extend(
        [
            "",
            "## Paired exact McNemar",
            "",
            f"- Controller vs STOP: p={mcnemar['controller_vs_stop']['p_value']:.8f}",
            f"- Controller vs Always Critic-only: "
            f"p={mcnemar['controller_vs_always']['p_value']:.8f}",
            "",
            "## CPU checkpoint replay",
            "",
            f"- Validation rows replayed: {summary['checkpoint_cpu_replay']['samples']}",
            f"- Actions exact match: {summary['checkpoint_cpu_replay']['actions_exact_match']}",
            f"- Maximum score difference: "
            f"{summary['checkpoint_cpu_replay']['max_gate_score_abs_difference']:.3e}",
            f"- Maximum probability difference: "
            f"{summary['checkpoint_cpu_replay']['max_probability_abs_difference']:.3e}",
            "",
            "Audit complete. No threshold or final operating point was selected.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run_precritic_controller_v1_audit(
    *,
    controller_dir: str | Path = DEFAULT_CONTROLLER_DIR,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    validation_path: str | Path = DEFAULT_VALIDATION,
    old_probe_predictions_path: str | Path = DEFAULT_OLD_PROBE,
    old_probe_summary_path: str | Path = DEFAULT_OLD_PROBE_SUMMARY,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    run_metadata_prefix: str | Path = DEFAULT_RUN_METADATA_PREFIX,
    output_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict[str, Any]:
    """Audit the frozen run without training or any backend/model API calls."""
    controller_dir = Path(controller_dir)
    output_dir = Path(output_dir)
    targets = (
        output_dir / "audit_summary.json",
        output_dir / "audit_cases.jsonl",
        output_dir / "audit_report.md",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError("Controller v1 audit artifacts already exist; refusing to overwrite")

    run_status = verify_run_status(run_metadata_prefix)
    artifact_integrity = verify_artifacts(controller_dir)
    before_hashes = {name: item["sha256"] for name, item in artifact_integrity.items()}
    saved_summary = json.loads((controller_dir / "summary.json").read_text(encoding="utf-8"))
    seed_payload = json.loads(
        (controller_dir / "seed_metrics.json").read_text(encoding="utf-8")
    )
    if (
        saved_summary.get("model_calls") != 0
        or saved_summary.get("final_test_evaluated") is not False
        or saved_summary.get("deployable") is not False
        or seed_payload.get("best_seed_selected") is not False
        or seed_payload.get("primary_seed") != PRIMARY_SEED
        or seed_payload.get("stability_seeds") != list(STABILITY_SEEDS)
    ):
        raise ValueError("Saved Controller summary violates the frozen protocol")
    saved_seed_list = seed_payload.get("seeds")
    if not isinstance(saved_seed_list, list) or [item.get("seed") for item in saved_seed_list] != list(ALL_SEEDS):
        raise ValueError("Saved seed metrics do not contain the five fixed seeds in order")
    saved_seed_metrics = {int(item["seed"]): item for item in saved_seed_list}
    if any(item.get("used_for_model_selection") is not False for item in saved_seed_list):
        raise ValueError("A stability seed was used for model selection")

    training_examples, training_manifest = load_training_examples(
        training_path, training_manifest_path
    )
    if len(training_examples) != EXPECTED_TRAINING_SAMPLES:
        raise ValueError("Training sample count differs from the frozen contract")
    training_sha = _file_sha256(Path(training_path))
    if training_sha != TRAINING_SHA256:
        raise ValueError("Training JSONL SHA256 differs from the frozen contract")
    final_guard = verify_sealed_final_manifest(
        final_test_manifest_path, training_manifest
    )
    training_ids = {example.sample_id for example in training_examples}
    validation_examples, validation_sha = load_validation_examples(
        validation_path, training_manifest, training_ids
    )
    old_probe_decisions, old_probe_sources = load_old_probe_decisions(
        old_probe_predictions_path,
        old_probe_summary_path,
        validation_examples,
        validation_sha,
    )

    raw_training = read_jsonl(training_path)
    cost_available = sum(row.get("cost_available") is True for row in raw_training)
    cost_masked = len(raw_training) - cost_available
    if cost_available != 800 or cost_masked != 200:
        raise ValueError("Cost-loss mask must contain exactly 800 available and 200 masked rows")
    for index, row in enumerate(raw_training):
        model_input = _require_mapping(row.get("model_input"), f"Training row {index} input")
        if set(model_input) != {"problem", "solver"}:
            raise ValueError("Training features violate the strict whitelist")
        if row["cost_available"] is False and row.get("critic_cost_target") is not None:
            raise ValueError("A masked cost target was estimated")

    oof_rows = read_jsonl(controller_dir / "oof_predictions.jsonl")
    validation_rows = read_jsonl(controller_dir / "validation_predictions.jsonl")
    if len(oof_rows) != len(ALL_SEEDS) * EXPECTED_TRAINING_SAMPLES:
        raise ValueError("OOF predictions must contain 5000 rows")
    if len(validation_rows) != len(ALL_SEEDS) * EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("Validation predictions must contain 500 rows")
    oof_recomputed, oof_cases = _audit_oof(
        oof_rows, training_examples, saved_seed_metrics
    )
    validation_recomputed, validation_cases, validation_by_seed = _audit_validation(
        validation_rows,
        validation_examples,
        saved_seed_metrics,
        oof_recomputed,
        old_probe_decisions,
        saved_summary,
    )

    checkpoint_replay, replay_cases = _cpu_checkpoint_replay(
        controller_dir / "primary_model.pt",
        validation_examples,
        validation_by_seed[PRIMARY_SEED],
    )
    replay_by_id = {
        _typed_id(item["question_id"]): item for item in replay_cases
    }
    for item in validation_cases:
        if item["seed"] == PRIMARY_SEED:
            item["checkpoint_cpu_replay"] = replay_by_id[_typed_id(item["question_id"])]

    final_manifest_sha = _file_sha256(Path(final_test_manifest_path))
    if final_manifest_sha != final_guard["manifest_sha256"]:
        raise ValueError("Final Test manifest changed during the audit")
    device_audit = _training_device_evidence()
    training_protocol_audit = {
        "samples": len(training_examples),
        "training_sha256": training_sha,
        "training_sha256_contract": TRAINING_SHA256,
        "training_sha256_matches": True,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "fixed_seed_order": list(ALL_SEEDS),
        "oof_folds_per_seed": 5,
        "each_oof_sample_appears_once_per_seed": True,
        "folds_nonoverlapping": True,
        "hyperparameter_search": False,
        "best_seed_selected": False,
        "primary_always_20260816": True,
        "budget_threshold_source": "Training 1000 stratified 5-fold OOF only",
        "threshold_adjusted_during_audit": False,
    }
    leakage_and_cost_audit = {
        "feature_whitelist": [
            "problem and A-D options",
            "Solver raw output",
            "Solver parse status",
            "Solver usage",
        ],
        "gold_in_model_features": False,
        "critic_or_refiner_output_in_model_features": False,
        "action_result_in_model_features": False,
        "cost_available_samples": cost_available,
        "cost_masked_samples": cost_masked,
        "masked_cost_estimated": False,
        "cost_loss_available_source": "Collection 800 only",
        "validation_samples": len(validation_examples),
        "validation_used_for_training": False,
        "validation_used_for_oof": False,
        "validation_used_for_thresholds": False,
        "validation_used_for_seed_selection": False,
        "final_test": {
            **final_guard,
            "manifest_sha256_unchanged": True,
            "manifest_only_read": True,
            "examples_read": False,
            "model_calls": 0,
        },
    }
    summary: dict[str, Any] = {
        "offline_audit": True,
        "controller_retrained": False,
        "thresholds_adjusted": False,
        "model_backend_initialized": False,
        "model_calls": 0,
        "final_test_evaluated": False,
        "deployable": False,
        "run_status": run_status,
        "artifact_integrity": artifact_integrity,
        "training_protocol_audit": training_protocol_audit,
        "device_audit": device_audit,
        "leakage_and_cost_audit": leakage_and_cost_audit,
        "sources": {
            "training_manifest": {
                "path": str(Path(training_manifest_path).resolve()),
                "sha256": _file_sha256(Path(training_manifest_path)),
            },
            "validation": {
                "path": str(Path(validation_path).resolve()),
                "sha256": validation_sha,
            },
            "old_probe": old_probe_sources,
            "final_test_manifest": {
                "path": str(Path(final_test_manifest_path).resolve()),
                "sha256": final_manifest_sha,
            },
        },
        "oof_recomputed": oof_recomputed,
        "validation_recomputed": validation_recomputed,
        "checkpoint_cpu_replay": checkpoint_replay,
        "saved_metrics_consistency": {
            "oof": True,
            "validation": True,
            "stability": True,
            "mcnemar": True,
            "checkpoint_scores": True,
            "checkpoint_actions": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "audit_summary.json", summary)
    write_jsonl(output_dir / "audit_cases.jsonl", [*oof_cases, *validation_cases])
    _write_text_atomic(output_dir / "audit_report.md", _report(summary))

    after_hashes = {
        name: _file_sha256(controller_dir / name) for name in ARTIFACT_NAMES
    }
    if after_hashes != before_hashes:
        raise RuntimeError("A historical Controller v1 artifact changed during the audit")
    if _file_sha256(Path(training_path)) != training_sha:
        raise RuntimeError("Frozen Training 1000 changed during the audit")
    if _file_sha256(Path(final_test_manifest_path)) != final_manifest_sha:
        raise RuntimeError("Sealed Final Test manifest changed during the audit")
    return summary

