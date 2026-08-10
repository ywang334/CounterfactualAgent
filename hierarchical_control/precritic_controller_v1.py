"""Formal offline supervised training for Pre-Critic Controller v1."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .io_utils import read_jsonl, write_jsonl
from .logiqa_action_collection import _problem_content_sha256
from .logiqa_audit import exact_mcnemar
from .logiqa_pilot import ANSWER_LETTERS
from .precritic_probe import (
    LABELS,
    NUMERIC_FEATURES,
    OfflineMiniLMEncoder,
    ProbeExample,
    _hash_model_input,
    _numeric_features,
    _policy_metrics,
    _render_feature_text,
    aggregate_validation_cost,
    build_probe_example,
    deterministic_stratified_folds,
)


TRAINING_SHA256 = "52d87713773308bc6085fa7d35f3c1be29f6de6e35633a286c68c0e241007303"
PRIMARY_SEED = 20260816
STABILITY_SEEDS = (20260817, 20260818, 20260819, 20260820)
ALL_SEEDS = (PRIMARY_SEED, *STABILITY_SEEDS)
N_SPLITS = 5
HIDDEN_DIM = 64
EPOCHS = 120
LEARNING_RATE = 1e-3
COST_LOSS_WEIGHT = 0.1
BUDGET_RATES = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
EXPECTED_TRAINING_SAMPLES = 1000
EXPECTED_VALIDATION_SAMPLES = 100
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TRAINING = Path("artifacts/precritic_training_1000/training_examples.jsonl")
DEFAULT_TRAINING_MANIFEST = Path("artifacts/precritic_training_1000/manifest.json")
DEFAULT_VALIDATION = Path("artifacts/logiqa_policy_validation_100/predictions.jsonl")
DEFAULT_OLD_PROBE = Path("artifacts/precritic_gate_probe/predictions.jsonl")
DEFAULT_OLD_PROBE_SUMMARY = Path("artifacts/precritic_gate_probe/summary.json")
DEFAULT_FINAL_MANIFEST = Path("artifacts/logiqa_final_test_500/split_manifest.json")
DEFAULT_OUTPUT = Path("artifacts/precritic_controller_v1")


class FrozenEncoder(Protocol):
    name: str
    dimension: int
    mock_only: bool

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class PreCriticControllerV1(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        numeric_dim: int = len(NUMERIC_FEATURES),
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.numeric_dim = numeric_dim
        self.hidden_dim = hidden_dim
        self.hidden = nn.Linear(embedding_dim + numeric_dim, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, len(LABELS))
        self.cost_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, embeddings: torch.Tensor, numeric: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.hidden(torch.cat([embeddings, numeric], dim=-1)))
        return self.class_head(hidden), self.cost_head(hidden).squeeze(-1)


@dataclass(frozen=True)
class TrainingExample:
    sample_id: str
    source_dataset: str
    question_id: str | int
    label: str
    model_input: dict[str, Any]
    feature_text: str
    numeric: tuple[float, ...]
    cost_available: bool
    cost_log_target: float


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _typed_id(value: str | int) -> str:
    return json.dumps([type(value).__name__, value], ensure_ascii=False)


def _valid_question_id(value: Any) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Training source has invalid question_id")
    return value


def _usage(value: Any, context: str) -> dict[str, Any]:
    usage = _mapping(value, context)
    result: dict[str, Any] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        amount = usage.get(field)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError(f"{context} has invalid {field}")
        result[field] = amount
    if result["prompt_tokens"] + result["completion_tokens"] != result["total_tokens"]:
        raise ValueError(f"{context} token identity failed")
    latency = usage.get("latency_seconds")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ValueError(f"{context} has invalid latency")
    result["latency_seconds"] = float(latency)
    return result


def _inspect_feature_keys(value: Any) -> None:
    forbidden = {"gold", "critic", "refiner", "actions", "outcome", "correct"}
    if isinstance(value, dict):
        keys = {str(key).casefold() for key in value}
        overlap = forbidden & keys
        if overlap:
            raise ValueError(f"Forbidden model-input keys: {sorted(overlap)}")
        for child in value.values():
            _inspect_feature_keys(child)
    elif isinstance(value, list):
        for child in value:
            _inspect_feature_keys(child)


def load_training_examples(
    training_path: str | Path,
    manifest_path: str | Path,
    *,
    expected_sha256: str = TRAINING_SHA256,
    expected_samples: int = EXPECTED_TRAINING_SAMPLES,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    training_path = Path(training_path)
    manifest_path = Path(manifest_path)
    if not training_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Training examples and protocol manifest are required")
    digest = _file_sha256(training_path)
    if digest != expected_sha256:
        raise ValueError(
            f"Training SHA256 mismatch: expected {expected_sha256}, got {digest}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("precritic_training_protocol") is not True
        or manifest.get("training_examples_sha256") != digest
        or manifest.get("model_calls") != 0
        or manifest.get("controller_trained") is not False
    ):
        raise ValueError("Invalid frozen Pre-Critic training protocol manifest")
    rows = read_jsonl(training_path)
    if len(rows) != expected_samples or manifest.get("samples") != expected_samples:
        raise ValueError(
            f"Expected exactly {expected_samples} training rows; found {len(rows)}"
        )
    expected_keys = {
        "sample_id",
        "content_sha256",
        "source",
        "model_input",
        "label",
        "cost_available",
        "critic_cost_target",
    }
    examples: list[TrainingExample] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if set(row) != expected_keys:
            raise ValueError(f"Training row {index} violates the whitelist schema")
        digest = row.get("sample_id")
        if (
            not isinstance(digest, str)
            or row.get("content_sha256") != digest
            or digest in seen
        ):
            raise ValueError(f"Training row {index} has invalid or duplicate content identity")
        seen.add(digest)
        source = _mapping(row.get("source"), f"Training row {index} source")
        dataset = source.get("dataset")
        if dataset not in {"collection_200", "collection_800"}:
            raise ValueError(f"Training row {index} has unknown source")
        question_id = _valid_question_id(source.get("question_id"))
        model_input = _mapping(row.get("model_input"), f"Training row {index} input")
        if set(model_input) != {"problem", "solver"}:
            raise ValueError(f"Training row {index} has non-whitelist model input")
        _inspect_feature_keys(model_input)
        problem = _mapping(model_input.get("problem"), f"Training row {index} problem")
        if _problem_content_sha256(problem) != digest:
            raise ValueError(f"Training row {index} content SHA256 mismatch")
        solver = _mapping(model_input.get("solver"), f"Training row {index} Solver")
        if set(solver) != {"raw_output", "parse_status", "usage"}:
            raise ValueError(f"Training row {index} Solver features are not whitelisted")
        if not isinstance(solver.get("raw_output"), str):
            raise ValueError(f"Training row {index} lacks Solver output")
        _usage(solver.get("usage"), f"Training row {index} Solver usage")
        parse = _mapping(solver.get("parse_status"), f"Training row {index} parse")
        if set(parse) != {
            "strict_answer",
            "strict_parse_failure",
            "tolerant_answer",
            "tolerant_parse_failure",
            "tolerant_match_count",
            "tolerant_conflict",
        }:
            raise ValueError(f"Training row {index} parse schema changed")
        label = row.get("label")
        if label not in LABELS:
            raise ValueError(f"Training row {index} has invalid label")
        cost_available = row.get("cost_available")
        if not isinstance(cost_available, bool):
            raise ValueError(f"Training row {index} has invalid cost mask")
        if cost_available:
            if dataset != "collection_800":
                raise ValueError("Only Collection 800 may have Critic cost targets")
            cost = _usage(
                row.get("critic_cost_target"),
                f"Training row {index} Critic cost target",
            )
            if cost["calls"] != 1:
                raise ValueError("Critic cost target must represent one call")
            cost_log_target = math.log1p(cost["total_tokens"])
        else:
            if dataset != "collection_200" or row.get("critic_cost_target") is not None:
                raise ValueError("Unavailable Collection 200 cost must remain null")
            cost_log_target = 0.0
        examples.append(
            TrainingExample(
                sample_id=digest,
                source_dataset=dataset,
                question_id=question_id,
                label=str(label),
                model_input=model_input,
                feature_text=_render_feature_text(model_input),
                numeric=_numeric_features(model_input),
                cost_available=cost_available,
                cost_log_target=cost_log_target,
            )
        )
    labels = Counter(example.label for example in examples)
    if {label: labels.get(label, 0) for label in LABELS} != manifest.get("label_counts"):
        raise ValueError("Training labels differ from the frozen manifest")
    cost_targets = _mapping(manifest.get("cost_targets"), "training cost targets")
    expected_available = cost_targets.get("available_samples")
    if (
        isinstance(expected_available, bool)
        or not isinstance(expected_available, int)
        or sum(example.cost_available for example in examples) != expected_available
    ):
        raise ValueError("Training cost-mask count differs from the frozen manifest")
    return examples, manifest


def verify_sealed_final_manifest(
    final_manifest_path: str | Path,
    training_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify only the sealed manifest; never open its referenced data file."""
    path = Path(final_manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sealed Final Test manifest not found: {path}")
    frozen = _mapping(training_manifest.get("final_test"), "training final-test guard")
    expected_sha = frozen.get("manifest_sha256")
    actual_sha = _file_sha256(path)
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ValueError("Sealed Final Test manifest SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("final_test") is not True
        or payload.get("sealed") is not True
        or payload.get("never_evaluated") is not True
        or payload.get("model_calls") != 0
        or payload.get("split_sha256") != frozen.get("split_sha256")
    ):
        raise ValueError("Final Test is not sealed and untouched")
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": actual_sha,
        "split_sha256": payload["split_sha256"],
        "sealed": True,
        "never_evaluated": True,
        "final_test_evaluated": False,
    }


def load_validation_examples(
    validation_path: str | Path,
    training_manifest: Mapping[str, Any],
    training_ids: set[str],
    *,
    expected_samples: int = EXPECTED_VALIDATION_SAMPLES,
) -> tuple[list[ProbeExample], str]:
    path = Path(validation_path)
    if not path.is_file():
        raise FileNotFoundError(f"Validation predictions not found: {path}")
    source_files = _mapping(training_manifest.get("source_files"), "source files")
    frozen = _mapping(source_files.get("validation_predictions"), "Validation source")
    digest = _file_sha256(path)
    if digest != frozen.get("sha256"):
        raise ValueError("Validation SHA256 differs from the training protocol")
    rows = read_jsonl(path)
    if len(rows) != expected_samples:
        raise ValueError(f"Expected {expected_samples} Validation rows; found {len(rows)}")
    examples = [build_probe_example(row, "validation_100") for row in rows]
    ids = {_typed_id(example.question_id) for example in examples}
    if len(ids) != len(examples):
        raise ValueError("Validation question IDs are not unique")
    content_ids = {
        _problem_content_sha256(example.model_input["problem"]) for example in examples
    }
    if len(content_ids) != len(examples) or content_ids & training_ids:
        raise ValueError("Validation content is duplicated or overlaps training")
    for example in examples:
        _inspect_feature_keys(example.model_input)
    return examples, digest


def load_old_probe_decisions(
    predictions_path: str | Path,
    summary_path: str | Path,
    validation_examples: Sequence[ProbeExample],
    validation_sha256: str,
) -> tuple[list[bool], dict[str, str]]:
    predictions_path = Path(predictions_path)
    summary_path = Path(summary_path)
    if not predictions_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Frozen old Probe predictions and summary are required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("sources", {}).get("validation_sha256") != validation_sha256:
        raise ValueError("Old Probe was not evaluated on this Validation 100")
    rows = [
        row
        for row in read_jsonl(predictions_path)
        if row.get("split") == "validation_100_once"
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _typed_id(_valid_question_id(row.get("question_id")))
        if key in by_id:
            raise ValueError("Old Probe has duplicate Validation predictions")
        by_id[key] = row
    decisions: list[bool] = []
    for example in validation_examples:
        row = by_id.get(_typed_id(example.question_id))
        if row is None or row.get("model_input_sha256") != _hash_model_input(example):
            raise ValueError("Old Probe Validation identity or model input changed")
        decision = row.get("critic_called")
        if not isinstance(decision, bool):
            raise ValueError("Old Probe decision is invalid")
        decisions.append(decision)
    if len(by_id) != len(validation_examples):
        raise ValueError("Old Probe Validation set is not one-to-one")
    return decisions, {
        "predictions_path": str(predictions_path.resolve()),
        "predictions_sha256": _file_sha256(predictions_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _file_sha256(summary_path),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _class_weights(targets: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=len(LABELS)).to(torch.float32)
    if torch.any(counts == 0):
        raise ValueError("Every OOF training fold must contain all four classes")
    return targets.numel() / (len(LABELS) * counts)


def _standardize(
    training: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = training.mean(dim=0)
    std = training.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (training - mean) / std, (other - mean) / std, mean, std


def masked_huber_cost_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    if predictions.shape != targets.shape or predictions.shape != available.shape:
        raise ValueError("Cost predictions, targets, and mask must align")
    mask = available.to(dtype=torch.bool)
    if not torch.any(mask):
        return predictions.sum() * 0.0
    return F.huber_loss(predictions[mask], targets[mask], reduction="mean", delta=1.0)


def train_controller_model(
    embeddings: torch.Tensor,
    numeric: torch.Tensor,
    targets: torch.Tensor,
    cost_targets: torch.Tensor,
    cost_available: torch.Tensor,
    *,
    seed: int,
    hidden_dim: int = HIDDEN_DIM,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    cost_loss_weight: float = COST_LOSS_WEIGHT,
) -> tuple[PreCriticControllerV1, dict[str, Any]]:
    _seed_everything(seed)
    model = PreCriticControllerV1(
        embedding_dim=embeddings.shape[1],
        numeric_dim=numeric.shape[1],
        hidden_dim=hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = _class_weights(targets)
    final_classification = 0.0
    final_cost = 0.0
    final_total = 0.0
    model.train()
    for _ in range(epochs):
        logits, cost_predictions = model(embeddings, numeric)
        classification_loss = F.cross_entropy(logits, targets, weight=weights)
        cost_loss = masked_huber_cost_loss(
            cost_predictions, cost_targets, cost_available
        )
        loss = classification_loss + cost_loss_weight * cost_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_classification = float(classification_loss.detach())
        final_cost = float(cost_loss.detach())
        final_total = float(loss.detach())
    model.eval()
    return model, {
        "final_total_loss": final_total,
        "final_classification_loss": final_classification,
        "final_masked_cost_loss": final_cost,
        "class_weights": weights.tolist(),
        "cost_available_samples": int(cost_available.sum()),
        "cost_masked_samples": int((~cost_available.to(torch.bool)).sum()),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "cost_loss_weight": cost_loss_weight,
    }


def oof_controller_predictions(
    embeddings: torch.Tensor,
    numeric: torch.Tensor,
    labels: Sequence[str],
    cost_targets: torch.Tensor,
    cost_available: torch.Tensor,
    *,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[dict[str, list[int]]],
    list[dict[str, Any]],
]:
    folds = deterministic_stratified_folds(labels, N_SPLITS, seed)
    targets = torch.tensor([LABELS.index(label) for label in labels], dtype=torch.long)
    probabilities = torch.zeros((len(labels), len(LABELS)), dtype=torch.float32)
    predicted_cost_logs = torch.zeros(len(labels), dtype=torch.float32)
    fold_metrics: list[dict[str, Any]] = []
    for fold in folds:
        train_indices = torch.tensor(fold["train"], dtype=torch.long)
        validation_indices = torch.tensor(fold["validation"], dtype=torch.long)
        train_numeric, validation_numeric, _, _ = _standardize(
            numeric[train_indices], numeric[validation_indices]
        )
        model, metrics = train_controller_model(
            embeddings[train_indices],
            train_numeric,
            targets[train_indices],
            cost_targets[train_indices],
            cost_available[train_indices],
            seed=seed + fold["fold"] + 1,
            epochs=epochs,
        )
        with torch.no_grad():
            logits, fold_costs = model(
                embeddings[validation_indices], validation_numeric
            )
            probabilities[validation_indices] = torch.softmax(logits, dim=-1)
            predicted_cost_logs[validation_indices] = fold_costs
        fold_metrics.append(
            {
                "fold": fold["fold"],
                "training_samples": len(fold["train"]),
                "oof_samples": len(fold["validation"]),
                **metrics,
            }
        )
    if torch.any(probabilities.sum(dim=-1) == 0):
        raise AssertionError("OOF predictions are incomplete")
    return probabilities, predicted_cost_logs, folds, fold_metrics


def gate_scores(probabilities: torch.Tensor) -> torch.Tensor:
    return (
        probabilities[:, LABELS.index("wrong_to_correct")]
        - probabilities[:, LABELS.index("correct_to_wrong")]
    )


def _threshold_above(scores: Sequence[float]) -> float:
    maximum = max(scores)
    return maximum + max(abs(maximum), 1.0) * 1e-7


def _threshold_below(scores: Sequence[float]) -> float:
    minimum = min(scores)
    return minimum - max(abs(minimum), 1.0) * 1e-7


def gated_decisions(scores: Sequence[float], threshold: float) -> list[bool]:
    return [float(score) >= threshold for score in scores]


def gate_outcome(labels: Sequence[str], decisions: Sequence[bool]) -> dict[str, Any]:
    if len(labels) != len(decisions):
        raise ValueError("Gate labels and decisions must align")
    corrected = sum(
        decision and label == "wrong_to_correct"
        for label, decision in zip(labels, decisions)
    )
    degraded = sum(
        decision and label == "correct_to_wrong"
        for label, decision in zip(labels, decisions)
    )
    calls = sum(decisions)
    return {
        "corrected": corrected,
        "degraded": degraded,
        "net_benefit": corrected - degraded,
        "critic_calls": calls,
        "critic_call_rate": calls / len(labels),
    }


def select_oof_threshold_v1(
    scores: Sequence[float], labels: Sequence[str]
) -> dict[str, Any]:
    if not scores or len(scores) != len(labels):
        raise ValueError("OOF scores and labels must be non-empty and aligned")
    candidates = []
    for threshold in [_threshold_above(scores), *sorted(set(scores), reverse=True)]:
        candidates.append(
            {
                "threshold": float(threshold),
                **gate_outcome(labels, gated_decisions(scores, threshold)),
            }
        )
    best = max(
        candidates,
        key=lambda item: (
            item["net_benefit"],
            -item["critic_calls"],
            item["threshold"],
        ),
    )
    safe_fallback = best["net_benefit"] <= 0
    if safe_fallback:
        best = candidates[0]
    return {
        **best,
        "selection_source": "training_1000_stratified_5fold_oof_only",
        "selection_objective": (
            "maximize corrected-degraded; ties use lower Critic call rate"
        ),
        "safe_fallback_always_stop": safe_fallback,
        "candidate_count": len(candidates),
        "validation_used_for_selection": False,
        "deployment_operating_point_selected": False,
    }


def oof_budget_thresholds_v1(
    scores: Sequence[float], rates: Sequence[float] = BUDGET_RATES
) -> list[dict[str, Any]]:
    if not scores:
        raise ValueError("OOF budget thresholds require scores")
    unique_desc = sorted(set(float(score) for score in scores), reverse=True)
    candidates = [
        (_threshold_above(scores), 0),
        *[
            (threshold, sum(float(score) >= threshold for score in scores))
            for threshold in unique_desc
        ],
    ]
    result = []
    for rate in rates:
        if not 0 < rate <= 1:
            raise ValueError("Budget rates must be in (0, 1]")
        maximum_calls = math.floor(len(scores) * rate + 1e-12)
        if rate == 1.0:
            threshold, calls = _threshold_below(scores), len(scores)
        else:
            eligible = [candidate for candidate in candidates if candidate[1] <= maximum_calls]
            threshold, calls = max(eligible, key=lambda item: (item[1], -item[0]))
        result.append(
            {
                "target_budget_rate": float(rate),
                "threshold": float(threshold),
                "oof_critic_calls": calls,
                "oof_critic_call_rate": calls / len(scores),
                "threshold_source": "training_1000_oof_score_distribution",
                "validation_used_for_threshold": False,
                "deployment_operating_point_selected": False,
            }
        )
    return result


def _average_precision(binary_targets: Sequence[bool], scores: Sequence[float]) -> float:
    if len(binary_targets) != len(scores) or not scores:
        raise ValueError("PR-AUC targets and scores must align")
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
        threshold_score = float(scores[order[cursor]])
        end = cursor
        while end < len(order) and float(scores[order[end]]) == threshold_score:
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


def _macro_f1(labels: Sequence[str], probabilities: torch.Tensor) -> float:
    if probabilities.shape != (len(labels), len(LABELS)):
        raise ValueError("Macro-F1 probabilities have the wrong shape")
    predicted = probabilities.argmax(dim=-1).tolist()
    actual = [LABELS.index(label) for label in labels]
    scores = []
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
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(scores) / len(scores)


def _token_predictions(cost_logs: torch.Tensor) -> torch.Tensor:
    return torch.expm1(torch.clamp(cost_logs, min=0.0, max=20.0))


def quality_head_metrics(
    probabilities: torch.Tensor,
    predicted_cost_logs: torch.Tensor,
    labels: Sequence[str],
    cost_log_targets: torch.Tensor,
    cost_available: torch.Tensor,
) -> dict[str, Any]:
    if probabilities.shape[0] != len(labels):
        raise ValueError("Quality metrics are not aligned")
    helpful_scores = probabilities[:, LABELS.index("wrong_to_correct")].tolist()
    harmful_scores = probabilities[:, LABELS.index("correct_to_wrong")].tolist()
    available = cost_available.to(torch.bool)
    if not torch.any(available):
        cost_mae = None
        cost_count = 0
    else:
        predicted_tokens = _token_predictions(predicted_cost_logs[available])
        actual_tokens = torch.expm1(cost_log_targets[available])
        cost_mae = float(torch.mean(torch.abs(predicted_tokens - actual_tokens)))
        cost_count = int(available.sum())
    return {
        "helpful_pr_auc": _average_precision(
            [label == "wrong_to_correct" for label in labels], helpful_scores
        ),
        "harmful_pr_auc": _average_precision(
            [label == "correct_to_wrong" for label in labels], harmful_scores
        ),
        "four_class_macro_f1": _macro_f1(labels, probabilities),
        "critic_incremental_total_tokens_mae": cost_mae,
        "cost_mae_samples": cost_count,
    }


def label_policy_metrics(
    labels: Sequence[str], decisions: Sequence[bool]
) -> dict[str, Any]:
    if len(labels) != len(decisions):
        raise ValueError("OOF policy labels and decisions must align")
    transitions = Counter()
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


def _validation_cost_targets(
    examples: Sequence[ProbeExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    logs = []
    for example in examples:
        stop = example.audit_case["strategy_costs"]["STOP"]
        critic = example.audit_case["strategy_costs"]["CRITIC_ONLY"]
        if stop.get("available") is not True or critic.get("available") is not True:
            raise ValueError("Validation stage usage is unavailable")
        incremental = critic["total_tokens"] - stop["total_tokens"]
        if incremental < 0 or critic["calls"] - stop["calls"] != 1:
            raise ValueError("Validation Critic incremental cost is inconsistent")
        logs.append(math.log1p(incremental))
    return torch.tensor(logs, dtype=torch.float32), torch.ones(
        len(logs), dtype=torch.bool
    )


def evaluate_validation_policy(
    examples: Sequence[ProbeExample], decisions: Sequence[bool]
) -> dict[str, Any]:
    return {
        **_policy_metrics(examples, decisions),
        "cost": aggregate_validation_cost(examples, decisions),
    }


def _policy_correctness(
    examples: Sequence[ProbeExample], decisions: Sequence[bool]
) -> list[bool]:
    return [
        (example.critic_only_answer if decision else example.solver_answer)
        == example.gold
        for example, decision in zip(examples, decisions)
    ]


def paired_mcnemar(
    examples: Sequence[ProbeExample],
    first_decisions: Sequence[bool],
    second_decisions: Sequence[bool],
    *,
    first_name: str,
    second_name: str,
) -> dict[str, Any]:
    first = _policy_correctness(examples, first_decisions)
    second = _policy_correctness(examples, second_decisions)
    first_correct_second_wrong = sum(a and not b for a, b in zip(first, second))
    first_wrong_second_correct = sum(not a and b for a, b in zip(first, second))
    result = exact_mcnemar(
        first_correct_second_wrong, first_wrong_second_correct
    )
    return {
        "first_policy": first_name,
        "second_policy": second_name,
        "first_correct_second_wrong": first_correct_second_wrong,
        "first_wrong_second_correct": first_wrong_second_correct,
        **result,
    }


def _probabilities_payload(row: torch.Tensor) -> dict[str, float]:
    return {label: float(row[index]) for index, label in enumerate(LABELS)}


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def run_seed_training(
    *,
    seed: int,
    training_examples: Sequence[TrainingExample],
    training_embeddings: torch.Tensor,
    training_numeric: torch.Tensor,
    validation_examples: Sequence[ProbeExample],
    validation_embeddings: torch.Tensor,
    validation_numeric: torch.Tensor,
    epochs: int = EPOCHS,
) -> tuple[
    PreCriticControllerV1,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    labels = [example.label for example in training_examples]
    target_indices = torch.tensor(
        [LABELS.index(label) for label in labels], dtype=torch.long
    )
    cost_targets = torch.tensor(
        [example.cost_log_target for example in training_examples],
        dtype=torch.float32,
    )
    cost_available = torch.tensor(
        [example.cost_available for example in training_examples], dtype=torch.bool
    )
    oof_probs, oof_cost_logs, folds, fold_training = oof_controller_predictions(
        training_embeddings,
        training_numeric,
        labels,
        cost_targets,
        cost_available,
        seed=seed,
        epochs=epochs,
    )
    oof_scores = [float(value) for value in gate_scores(oof_probs)]
    threshold = select_oof_threshold_v1(oof_scores, labels)
    budget_thresholds = oof_budget_thresholds_v1(oof_scores)
    development_decisions = gated_decisions(oof_scores, threshold["threshold"])
    oof_budget_curve = []
    for point in budget_thresholds:
        decisions = gated_decisions(oof_scores, point["threshold"])
        oof_budget_curve.append(
            {**point, "oof_policy": label_policy_metrics(labels, decisions)}
        )

    standardized_training, standardized_validation, numeric_mean, numeric_std = (
        _standardize(training_numeric, validation_numeric)
    )
    model, final_training = train_controller_model(
        training_embeddings,
        standardized_training,
        target_indices,
        cost_targets,
        cost_available,
        seed=seed,
        epochs=epochs,
    )
    with torch.no_grad():
        validation_logits, validation_cost_logs = model(
            validation_embeddings, standardized_validation
        )
        validation_probs = torch.softmax(validation_logits, dim=-1)
    validation_labels = [example.label for example in validation_examples]
    validation_cost_targets, validation_cost_available = _validation_cost_targets(
        validation_examples
    )
    validation_scores = [float(value) for value in gate_scores(validation_probs)]
    validation_decisions = gated_decisions(
        validation_scores, threshold["threshold"]
    )
    validation_budget_curve = []
    for point in budget_thresholds:
        decisions = gated_decisions(validation_scores, point["threshold"])
        validation_budget_curve.append(
            {
                **point,
                "validation_policy": evaluate_validation_policy(
                    validation_examples, decisions
                ),
            }
        )

    fold_by_index = {
        index: fold["fold"] for fold in folds for index in fold["validation"]
    }
    oof_rows = []
    for index, example in enumerate(training_examples):
        oof_rows.append(
            {
                "controller_v1": True,
                "development_validation": False,
                "final_test_evaluated": False,
                "deployable": False,
                "model_calls": 0,
                "seed": seed,
                "primary_seed": seed == PRIMARY_SEED,
                "sample_id": example.sample_id,
                "source_dataset": example.source_dataset,
                "label": example.label,
                "oof_fold": fold_by_index[index],
                "probabilities": _probabilities_payload(oof_probs[index]),
                "gate_score": oof_scores[index],
                "development_threshold": threshold["threshold"],
                "critic_called": development_decisions[index],
                "cost_available": example.cost_available,
                "predicted_critic_incremental_total_tokens": float(
                    _token_predictions(oof_cost_logs[index : index + 1])[0]
                ),
                "model_input_sha256": hashlib.sha256(
                    json.dumps(
                        example.model_input,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    validation_rows = []
    predicted_validation_costs = _token_predictions(validation_cost_logs)
    actual_validation_costs = torch.expm1(validation_cost_targets)
    for index, example in enumerate(validation_examples):
        decision = validation_decisions[index]
        selected_answer = (
            example.critic_only_answer if decision else example.solver_answer
        )
        validation_rows.append(
            {
                "controller_v1": True,
                "development_validation": True,
                "final_test_evaluated": False,
                "deployable": False,
                "model_calls": 0,
                "seed": seed,
                "primary_seed": seed == PRIMARY_SEED,
                "question_id": example.question_id,
                "label": example.label,
                "probabilities": _probabilities_payload(validation_probs[index]),
                "gate_score": validation_scores[index],
                "development_threshold": threshold["threshold"],
                "threshold_source": threshold["selection_source"],
                "validation_used_for_threshold": False,
                "critic_called": decision,
                "selected_answer": selected_answer,
                "correct": selected_answer == example.gold,
                "predicted_critic_incremental_total_tokens": float(
                    predicted_validation_costs[index]
                ),
                "actual_critic_incremental_total_tokens": float(
                    actual_validation_costs[index]
                ),
                "model_input_sha256": _hash_model_input(example),
            }
        )

    seed_metrics = {
        "seed": seed,
        "role": "primary" if seed == PRIMARY_SEED else "stability_only",
        "used_for_model_selection": False,
        "oof": {
            "folds": N_SPLITS,
            "stratified": True,
            "fold_manifest": folds,
            "fold_training": fold_training,
            "development_threshold": threshold,
            "development_policy": label_policy_metrics(
                labels, development_decisions
            ),
            "budget_curve": oof_budget_curve,
            "head_metrics": quality_head_metrics(
                oof_probs,
                oof_cost_logs,
                labels,
                cost_targets,
                cost_available,
            ),
        },
        "final_training": final_training,
        "validation": {
            "development_validation": True,
            "validation_used_for_training": False,
            "validation_used_for_threshold": False,
            "validation_used_for_model_selection": False,
            "controller_policy": evaluate_validation_policy(
                validation_examples, validation_decisions
            ),
            "head_metrics": quality_head_metrics(
                validation_probs,
                validation_cost_logs,
                validation_labels,
                validation_cost_targets,
                validation_cost_available,
            ),
            "budget_curve": validation_budget_curve,
        },
    }
    return (
        model,
        numeric_mean,
        numeric_std,
        seed_metrics,
        oof_rows,
        validation_rows,
    )


def _stability_summary(seed_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "accuracy": lambda metric: metric["validation"]["controller_policy"]["accuracy"],
        "corrected": lambda metric: metric["validation"]["controller_policy"]["corrected"],
        "degraded": lambda metric: metric["validation"]["controller_policy"]["degraded"],
        "net_benefit": lambda metric: metric["validation"]["controller_policy"]["net_benefit"],
        "critic_call_rate": lambda metric: metric["validation"]["controller_policy"]["critic_call_rate"],
        "mean_total_tokens": lambda metric: metric["validation"]["controller_policy"]["cost"]["mean"]["total_tokens"],
        "mean_calls": lambda metric: metric["validation"]["controller_policy"]["cost"]["mean"]["calls"],
        "mean_latency_seconds": lambda metric: metric["validation"]["controller_policy"]["cost"]["mean"]["latency_seconds"],
        "helpful_pr_auc": lambda metric: metric["validation"]["head_metrics"]["helpful_pr_auc"],
        "harmful_pr_auc": lambda metric: metric["validation"]["head_metrics"]["harmful_pr_auc"],
        "four_class_macro_f1": lambda metric: metric["validation"]["head_metrics"]["four_class_macro_f1"],
        "critic_cost_mae": lambda metric: metric["validation"]["head_metrics"]["critic_incremental_total_tokens_mae"],
    }
    return {
        "seeds": [metric["seed"] for metric in seed_metrics],
        "population_standard_deviation": True,
        "metrics": {
            name: _mean_std([float(accessor(metric)) for metric in seed_metrics])
            for name, accessor in paths.items()
        },
    }


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


def _save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _report(summary: dict[str, Any]) -> str:
    comparison = summary["development_validation"]["policy_comparison"]
    primary = summary["primary_seed_metrics"]
    lines = [
        "# Pre-Critic Controller v1",
        "",
        "Formal supervised training on Training 1000 with frozen MiniLM. "
        "`development_validation=true`, `final_test_evaluated=false`, "
        "`deployable=false`, and `model_calls=0`.",
        "",
        "Validation 100 is evaluation-only. It is not used for training, OOF "
        "thresholds, or model/seed selection. Final Test 500 remains sealed.",
        "",
        "## Development Validation 100",
        "",
        "| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean tokens | Mean calls | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
    head = primary["validation"]["head_metrics"]
    lines.extend(
        [
            "",
            "## Primary seed head metrics",
            "",
            f"- Seed: {PRIMARY_SEED}",
            f"- Helpful PR-AUC: {head['helpful_pr_auc']:.6f}",
            f"- Harmful PR-AUC: {head['harmful_pr_auc']:.6f}",
            f"- Four-class macro-F1: {head['four_class_macro_f1']:.6f}",
            f"- Critic incremental total-token MAE: "
            f"{head['critic_incremental_total_tokens_mae']:.3f}",
            "",
            "## Primary OOF budget points",
            "",
            "All points are shown; no deployment operating point is selected.",
            "",
            "| Budget | Threshold | OOF calls | OOF net | Validation calls | Validation accuracy | Validation net |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    validation_curve = primary["validation"]["budget_curve"]
    for oof_point, validation_point in zip(
        primary["oof"]["budget_curve"], validation_curve
    ):
        oof_policy = oof_point["oof_policy"]
        validation_policy = validation_point["validation_policy"]
        lines.append(
            f"| {oof_point['target_budget_rate']:.0%} | "
            f"{oof_point['threshold']:.8f} | {oof_policy['critic_calls']} | "
            f"{oof_policy['net_benefit']} | {validation_policy['critic_calls']} | "
            f"{validation_policy['accuracy']:.4f} | "
            f"{validation_policy['net_benefit']} |"
        )
    lines.extend(
        [
            "",
            "## Stability",
            "",
            "Seeds 20260817-20260820 are stability reports only; the primary model "
            "is always seed 20260816 and no best seed is selected.",
            "",
        ]
    )
    for name, metric in summary["stability"]["metrics"].items():
        lines.append(f"- {name}: {metric['mean']:.6f} ± {metric['std']:.6f}")
    lines.extend(
        [
            "",
            "## Paired tests",
            "",
            f"- Controller v1 vs STOP exact McNemar p="
            f"{summary['development_validation']['mcnemar']['controller_vs_stop']['p_value']:.8f}",
            f"- Controller v1 vs Always Critic-only exact McNemar p="
            f"{summary['development_validation']['mcnemar']['controller_vs_always']['p_value']:.8f}",
            "",
            "`POSTHOC_ORACLE` uses gold after generation and is deployable=false.",
            "No Final Test example or model inference was read or executed.",
            "",
        ]
    )
    return "\n".join(lines)


def train_precritic_controller_v1(
    *,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    validation_path: str | Path = DEFAULT_VALIDATION,
    old_probe_predictions_path: str | Path = DEFAULT_OLD_PROBE,
    old_probe_summary_path: str | Path = DEFAULT_OLD_PROBE_SUMMARY,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    encoder: FrozenEncoder | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    targets = (
        output / "primary_model.pt",
        output / "seed_metrics.json",
        output / "oof_predictions.jsonl",
        output / "validation_predictions.jsonl",
        output / "summary.json",
        output / "report.md",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError("Controller v1 artifacts already exist; refusing to overwrite")

    training_examples, training_manifest = load_training_examples(
        training_path, training_manifest_path
    )
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

    active_encoder = encoder or OfflineMiniLMEncoder(
        model_name=MODEL_NAME, device="cpu"
    )
    if getattr(active_encoder, "mock_only", True):
        raise ValueError("Formal Controller v1 training forbids mock/hash encoders")
    all_texts = [example.feature_text for example in training_examples] + [
        example.feature_text for example in validation_examples
    ]
    encoded = active_encoder.encode(all_texts).to(dtype=torch.float32, device="cpu")
    if encoded.shape != (len(all_texts), active_encoder.dimension):
        raise ValueError("Frozen encoder returned an unexpected tensor shape")
    training_embeddings = encoded[: len(training_examples)]
    validation_embeddings = encoded[len(training_examples) :]
    training_numeric = torch.tensor(
        [example.numeric for example in training_examples], dtype=torch.float32
    )
    validation_numeric = torch.tensor(
        [example.numeric for example in validation_examples], dtype=torch.float32
    )

    all_seed_metrics: list[dict[str, Any]] = []
    all_oof_rows: list[dict[str, Any]] = []
    all_validation_rows: list[dict[str, Any]] = []
    primary_model: PreCriticControllerV1 | None = None
    primary_mean: torch.Tensor | None = None
    primary_std: torch.Tensor | None = None
    for seed in ALL_SEEDS:
        model, mean, std, metrics, oof_rows, validation_rows = run_seed_training(
            seed=seed,
            training_examples=training_examples,
            training_embeddings=training_embeddings,
            training_numeric=training_numeric,
            validation_examples=validation_examples,
            validation_embeddings=validation_embeddings,
            validation_numeric=validation_numeric,
        )
        all_seed_metrics.append(metrics)
        all_oof_rows.extend(oof_rows)
        all_validation_rows.extend(validation_rows)
        if seed == PRIMARY_SEED:
            primary_model = model
            primary_mean = mean
            primary_std = std
    if primary_model is None or primary_mean is None or primary_std is None:
        raise AssertionError("Fixed primary seed was not trained")

    primary_metrics = next(
        metric for metric in all_seed_metrics if metric["seed"] == PRIMARY_SEED
    )
    primary_rows = [
        row for row in all_validation_rows if row["seed"] == PRIMARY_SEED
    ]
    primary_decisions = [bool(row["critic_called"]) for row in primary_rows]
    stop_decisions = [False] * len(validation_examples)
    always_decisions = [True] * len(validation_examples)
    oracle_decisions = [
        example.label == "wrong_to_correct" for example in validation_examples
    ]
    comparison = {
        "STOP": evaluate_validation_policy(validation_examples, stop_decisions),
        "ALWAYS_CRITIC_ONLY": evaluate_validation_policy(
            validation_examples, always_decisions
        ),
        "OLD_PROBE": {
            **evaluate_validation_policy(validation_examples, old_probe_decisions),
            "frozen_historical_policy": True,
        },
        "CONTROLLER_V1_PRIMARY": {
            **evaluate_validation_policy(validation_examples, primary_decisions),
            "seed": PRIMARY_SEED,
            "threshold_source": "training_1000_stratified_5fold_oof_only",
        },
        "POSTHOC_ORACLE": {
            **evaluate_validation_policy(validation_examples, oracle_decisions),
            "posthoc_oracle": True,
            "deployable": False,
        },
    }
    mcnemar = {
        "controller_vs_stop": paired_mcnemar(
            validation_examples,
            primary_decisions,
            stop_decisions,
            first_name="CONTROLLER_V1_PRIMARY",
            second_name="STOP",
        ),
        "controller_vs_always": paired_mcnemar(
            validation_examples,
            primary_decisions,
            always_decisions,
            first_name="CONTROLLER_V1_PRIMARY",
            second_name="ALWAYS_CRITIC_ONLY",
        ),
    }
    stability = _stability_summary(all_seed_metrics)

    training_sha = _file_sha256(Path(training_path))
    sources = {
        "training_examples": {
            "path": str(Path(training_path).resolve()),
            "sha256": training_sha,
            "samples": len(training_examples),
        },
        "training_manifest": {
            "path": str(Path(training_manifest_path).resolve()),
            "sha256": _file_sha256(Path(training_manifest_path)),
        },
        "development_validation": {
            "path": str(Path(validation_path).resolve()),
            "sha256": validation_sha,
            "samples": len(validation_examples),
        },
        "old_probe": old_probe_sources,
        "final_test_guard": final_guard,
    }
    seed_metrics_payload = {
        "controller_trained": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "deployable": False,
        "model_calls": 0,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "best_seed_selected": False,
        "embedding_encode_calls": 1,
        "sources": sources,
        "seeds": all_seed_metrics,
        "stability": stability,
    }
    summary = {
        "controller_v1": True,
        "controller_trained": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "deployable": False,
        "model_backend_initialized": False,
        "model_calls": 0,
        "prompt_modified": False,
        "parser_modified": False,
        "sources": sources,
        "data_boundary": {
            "training_only": "Training 1000",
            "validation_role": "development evaluation only",
            "validation_used_for_training": False,
            "validation_used_for_oof_or_thresholds": False,
            "validation_used_for_model_or_seed_selection": False,
            "final_test_manifest_verified_only": True,
            "final_test_examples_read": False,
        },
        "model": {
            "encoder": active_encoder.name,
            "encoder_frozen": True,
            "encoder_local_files_only": True,
            "embedding_dim": active_encoder.dimension,
            "embedding_encode_calls": 1,
            "architecture": "one_hidden_layer_shared_trunk_with_class_and_cost_heads",
            "hidden_dim": HIDDEN_DIM,
            "classes": list(LABELS),
            "numeric_features": list(NUMERIC_FEATURES),
            "cost_head_target": "log1p(Critic incremental total tokens)",
            "gate_score": "P(wrong_to_correct)-P(correct_to_wrong)",
        },
        "training_protocol": {
            "primary_seed": PRIMARY_SEED,
            "stability_seeds": list(STABILITY_SEEDS),
            "primary_model_always_uses_primary_seed": True,
            "best_seed_selected": False,
            "folds": N_SPLITS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_dim": HIDDEN_DIM,
            "class_weighted_cross_entropy": True,
            "cost_loss": "masked Huber",
            "cost_loss_weight": COST_LOSS_WEIGHT,
            "cost_available_samples": 800,
            "cost_masked_samples": 200,
            "cost_estimated_for_masked_samples": False,
            "hyperparameter_search": False,
            "budget_rates": list(BUDGET_RATES),
            "deployment_operating_point_selected": False,
        },
        "primary_seed_metrics": primary_metrics,
        "stability": stability,
        "development_validation": {
            "evaluated_after_oof_threshold_freeze": True,
            "policy_comparison": comparison,
            "mcnemar": mcnemar,
        },
    }

    checkpoint = {
        "controller_v1": True,
        "controller_trained": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "deployable": False,
        "model_calls": 0,
        "model_class": "PreCriticControllerV1",
        "model_state_dict": primary_model.state_dict(),
        "embedding_dim": active_encoder.dimension,
        "numeric_dim": len(NUMERIC_FEATURES),
        "hidden_dim": HIDDEN_DIM,
        "classes": list(LABELS),
        "encoder_name": active_encoder.name,
        "encoder_frozen": True,
        "encoder_local_files_only": True,
        "numeric_mean": primary_mean,
        "numeric_std": primary_std,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "development_threshold": primary_metrics["oof"]["development_threshold"],
        "budget_thresholds": [
            {
                key: point[key]
                for key in (
                    "target_budget_rate",
                    "threshold",
                    "oof_critic_calls",
                    "oof_critic_call_rate",
                    "threshold_source",
                )
            }
            for point in primary_metrics["oof"]["budget_curve"]
        ],
        "training_sha256": training_sha,
        "validation_used_for_training_or_selection": False,
        "final_test_guard": final_guard,
        "loss": "class-weighted cross-entropy + 0.1 * masked Huber cost loss",
    }

    output.mkdir(parents=True, exist_ok=True)
    _save_checkpoint_atomic(output / "primary_model.pt", checkpoint)
    _write_json_atomic(output / "seed_metrics.json", seed_metrics_payload)
    write_jsonl(output / "oof_predictions.jsonl", all_oof_rows)
    write_jsonl(output / "validation_predictions.jsonl", all_validation_rows)
    _write_json_atomic(output / "summary.json", summary)
    _write_text_atomic(output / "report.md", _report(summary))
    return {
        "controller_v1": True,
        "controller_trained": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "deployable": False,
        "model_calls": 0,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "output_dir": str(output.resolve()),
    }
