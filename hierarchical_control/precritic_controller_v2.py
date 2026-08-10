"""Factorized Pre-Critic Controller v2 supervised training.

The module is deliberately offline: it reuses frozen Training 1000 and
development Validation 100, never constructs an LLM backend, and verifies only
the sealed Final Test manifest metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .io_utils import read_jsonl, write_jsonl
from .precritic_controller_v1 import (
    ALL_SEEDS,
    BUDGET_RATES,
    COST_LOSS_WEIGHT,
    EPOCHS,
    EXPECTED_VALIDATION_SAMPLES,
    HIDDEN_DIM,
    LABELS,
    LEARNING_RATE,
    MODEL_NAME,
    N_SPLITS,
    PRIMARY_SEED,
    STABILITY_SEEDS,
    TRAINING_SHA256,
    TrainingExample,
    _average_precision,
    _file_sha256,
    _macro_f1,
    _mean_std,
    _standardize,
    _token_predictions,
    _validation_cost_targets,
    deterministic_stratified_folds,
    evaluate_validation_policy,
    gated_decisions,
    label_policy_metrics,
    load_old_probe_decisions,
    load_training_examples,
    load_validation_examples,
    masked_huber_cost_loss,
    oof_budget_thresholds_v1,
    paired_mcnemar,
    select_oof_threshold_v1,
    verify_sealed_final_manifest,
)
from .precritic_probe import (
    NUMERIC_FEATURES,
    OfflineMiniLMEncoder,
    ProbeExample,
    _hash_model_input,
)


DEFAULT_TRAINING = Path("artifacts/precritic_training_1000/training_examples.jsonl")
DEFAULT_TRAINING_MANIFEST = Path("artifacts/precritic_training_1000/manifest.json")
DEFAULT_VALIDATION = Path("artifacts/logiqa_policy_validation_100/predictions.jsonl")
DEFAULT_VALIDATION_SUMMARY = Path("artifacts/logiqa_policy_validation_100/summary.json")
DEFAULT_OLD_PROBE = Path("artifacts/precritic_gate_probe/predictions.jsonl")
DEFAULT_OLD_PROBE_SUMMARY = Path("artifacts/precritic_gate_probe/summary.json")
DEFAULT_CONTROLLER_V1_DIR = Path("artifacts/precritic_controller_v1")
DEFAULT_FINAL_MANIFEST = Path("artifacts/logiqa_final_test_500/split_manifest.json")
DEFAULT_OUTPUT = Path("artifacts/precritic_controller_v2_factorized")
MIN_COST_IMPROVEMENT = 0.05
HEAD_THRESHOLD = 0.5


class FrozenEncoder(Protocol):
    name: str
    dimension: int
    mock_only: bool

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class FactorizedPreCriticControllerV2(nn.Module):
    """Frozen features -> shared 64d trunk -> three conditional heads."""

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
        self.trunk = nn.Linear(embedding_dim + numeric_dim, hidden_dim)
        self.solver_error_head = nn.Linear(hidden_dim, 1)
        self.critic_fix_head = nn.Linear(hidden_dim, 1)
        self.critic_harm_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, embeddings: torch.Tensor, numeric: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.trunk(torch.cat([embeddings, numeric], dim=-1)))
        return (
            self.solver_error_head(hidden).squeeze(-1),
            self.critic_fix_head(hidden).squeeze(-1),
            self.critic_harm_head(hidden).squeeze(-1),
            self.cost_head(hidden).squeeze(-1),
        )


def factorized_targets(labels: Sequence[str]) -> dict[str, torch.Tensor]:
    if not labels or any(label not in LABELS for label in labels):
        raise ValueError("Factorized targets require valid frozen labels")
    solver_error = torch.tensor(
        [label.startswith("wrong_to_") for label in labels], dtype=torch.float32
    )
    critic_fix = torch.tensor(
        [label == "wrong_to_correct" for label in labels], dtype=torch.float32
    )
    critic_harm = torch.tensor(
        [label == "correct_to_wrong" for label in labels], dtype=torch.float32
    )
    error_mask = solver_error.to(torch.bool)
    return {
        "solver_error": solver_error,
        "critic_fix": critic_fix,
        "critic_harm": critic_harm,
        "solver_error_mask": torch.ones(len(labels), dtype=torch.bool),
        "critic_fix_mask": error_mask,
        "critic_harm_mask": ~error_mask,
    }


def masked_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Balanced binary BCE with exactly zero gradient outside the mask."""
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("Conditional BCE tensors must align")
    active = mask.to(torch.bool)
    if not torch.any(active):
        raise ValueError("Conditional BCE mask is empty")
    active_targets = targets[active]
    positives = int(active_targets.sum().item())
    negatives = int(active_targets.numel() - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Conditional BCE requires both classes")
    count = active_targets.numel()
    positive_weight = count / (2.0 * positives)
    negative_weight = count / (2.0 * negatives)
    weights = torch.where(
        active_targets > 0.5,
        torch.full_like(active_targets, positive_weight),
        torch.full_like(active_targets, negative_weight),
    )
    loss = F.binary_cross_entropy_with_logits(
        logits[active], active_targets, weight=weights, reduction="mean"
    )
    return loss, {
        "active_samples": count,
        "positive_samples": positives,
        "negative_samples": negatives,
        "positive_class_weight": positive_weight,
        "negative_class_weight": negative_weight,
        "strict_condition_mask": True,
    }


def factorized_probabilities(
    solver_error_logits: torch.Tensor,
    critic_fix_logits: torch.Tensor,
    critic_harm_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not (
        solver_error_logits.shape
        == critic_fix_logits.shape
        == critic_harm_logits.shape
    ):
        raise ValueError("Factorized logits must align")
    p_error = torch.sigmoid(solver_error_logits)
    p_fix = torch.sigmoid(critic_fix_logits)
    p_harm_given_correct = torch.sigmoid(critic_harm_logits)
    p_help = p_error * p_fix
    p_harm = (1.0 - p_error) * p_harm_given_correct
    four_class = torch.stack(
        (
            (1.0 - p_error) * (1.0 - p_harm_given_correct),
            p_harm,
            p_help,
            p_error * (1.0 - p_fix),
        ),
        dim=-1,
    )
    if not torch.allclose(
        four_class.sum(dim=-1),
        torch.ones_like(p_error),
        rtol=0.0,
        atol=1e-6,
    ):
        raise AssertionError("Factorized probabilities do not sum to one")
    return {
        "p_error": p_error,
        "p_fix_given_error": p_fix,
        "p_harm_given_correct": p_harm_given_correct,
        "p_help": p_help,
        "p_harm": p_harm,
        "gate_score": p_help - p_harm,
        "four_class": four_class,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_factorized_model(
    embeddings: torch.Tensor,
    numeric: torch.Tensor,
    labels: Sequence[str],
    cost_targets: torch.Tensor,
    cost_available: torch.Tensor,
    *,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[FactorizedPreCriticControllerV2, dict[str, Any]]:
    _seed_everything(seed)
    targets = factorized_targets(labels)
    model = FactorizedPreCriticControllerV2(
        embedding_dim=embeddings.shape[1],
        numeric_dim=numeric.shape[1],
        hidden_dim=HIDDEN_DIM,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    final: dict[str, Any] = {}
    model.train()
    for _ in range(epochs):
        error_logits, fix_logits, harm_logits, cost_logs = model(embeddings, numeric)
        error_loss, error_stats = masked_balanced_bce(
            error_logits, targets["solver_error"], targets["solver_error_mask"]
        )
        fix_loss, fix_stats = masked_balanced_bce(
            fix_logits, targets["critic_fix"], targets["critic_fix_mask"]
        )
        harm_loss, harm_stats = masked_balanced_bce(
            harm_logits, targets["critic_harm"], targets["critic_harm_mask"]
        )
        classification_loss = error_loss + fix_loss + harm_loss
        cost_loss = masked_huber_cost_loss(cost_logs, cost_targets, cost_available)
        loss = classification_loss + COST_LOSS_WEIGHT * cost_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = {
            "final_total_loss": float(loss.detach()),
            "final_factorized_classification_loss": float(
                classification_loss.detach()
            ),
            "final_solver_error_loss": float(error_loss.detach()),
            "final_critic_fix_loss": float(fix_loss.detach()),
            "final_critic_harm_loss": float(harm_loss.detach()),
            "final_masked_cost_loss": float(cost_loss.detach()),
            "solver_error_head": error_stats,
            "critic_fix_head": fix_stats,
            "critic_harm_head": harm_stats,
        }
    model.eval()
    return model, {
        **final,
        "cost_available_samples": int(cost_available.sum()),
        "cost_masked_samples": int((~cost_available.to(torch.bool)).sum()),
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "cost_loss_weight": COST_LOSS_WEIGHT,
        "classification_loss_combination": "sum_of_three_balanced_masked_BCE",
    }


def oof_factorized_predictions(
    embeddings: torch.Tensor,
    numeric: torch.Tensor,
    labels: Sequence[str],
    cost_targets: torch.Tensor,
    cost_available: torch.Tensor,
    *,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    folds = deterministic_stratified_folds(labels, N_SPLITS, seed)
    logits = {
        "solver_error": torch.zeros(len(labels), dtype=torch.float32),
        "critic_fix": torch.zeros(len(labels), dtype=torch.float32),
        "critic_harm": torch.zeros(len(labels), dtype=torch.float32),
    }
    predicted_cost_logs = torch.zeros(len(labels), dtype=torch.float32)
    fold_training = []
    for fold in folds:
        train_indices = torch.tensor(fold["train"], dtype=torch.long)
        validation_indices = torch.tensor(fold["validation"], dtype=torch.long)
        train_numeric, validation_numeric, _, _ = _standardize(
            numeric[train_indices], numeric[validation_indices]
        )
        model, metrics = train_factorized_model(
            embeddings[train_indices],
            train_numeric,
            [labels[index] for index in fold["train"]],
            cost_targets[train_indices],
            cost_available[train_indices],
            seed=seed + fold["fold"] + 1,
            epochs=epochs,
        )
        with torch.no_grad():
            error, fix, harm, costs = model(
                embeddings[validation_indices], validation_numeric
            )
        logits["solver_error"][validation_indices] = error
        logits["critic_fix"][validation_indices] = fix
        logits["critic_harm"][validation_indices] = harm
        predicted_cost_logs[validation_indices] = costs
        fold_training.append(
            {
                "fold": fold["fold"],
                "training_samples": len(fold["train"]),
                "oof_samples": len(fold["validation"]),
                **metrics,
            }
        )
    probabilities = factorized_probabilities(
        logits["solver_error"], logits["critic_fix"], logits["critic_harm"]
    )
    return probabilities, predicted_cost_logs, folds, fold_training


def _binary_f1(
    targets: Sequence[bool], scores: Sequence[float], threshold: float = HEAD_THRESHOLD
) -> float:
    if not targets or len(targets) != len(scores):
        raise ValueError("Binary F1 inputs must be non-empty and aligned")
    predicted = [float(score) >= threshold for score in scores]
    true_positive = sum(a and p for a, p in zip(targets, predicted))
    false_positive = sum(not a and p for a, p in zip(targets, predicted))
    false_negative = sum(a and not p for a, p in zip(targets, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def factorized_head_metrics(
    probabilities: Mapping[str, torch.Tensor],
    labels: Sequence[str],
) -> dict[str, Any]:
    error_targets = [label.startswith("wrong_to_") for label in labels]
    error_scores = probabilities["p_error"].tolist()
    error_indices = [index for index, target in enumerate(error_targets) if target]
    correct_indices = [index for index, target in enumerate(error_targets) if not target]
    fix_targets = [labels[index] == "wrong_to_correct" for index in error_indices]
    fix_scores = [
        float(probabilities["p_fix_given_error"][index]) for index in error_indices
    ]
    harm_targets = [labels[index] == "correct_to_wrong" for index in correct_indices]
    harm_scores = [
        float(probabilities["p_harm_given_correct"][index])
        for index in correct_indices
    ]
    return {
        "head_decision_threshold": HEAD_THRESHOLD,
        "solver_error": {
            "samples": len(labels),
            "positive_samples": sum(error_targets),
            "negative_samples": len(labels) - sum(error_targets),
            "pr_auc": _average_precision(error_targets, error_scores),
            "f1": _binary_f1(error_targets, error_scores),
        },
        "critic_fix_given_solver_error": {
            "strict_condition_mask": True,
            "samples": len(error_indices),
            "positive_samples": sum(fix_targets),
            "negative_samples": len(fix_targets) - sum(fix_targets),
            "pr_auc": _average_precision(fix_targets, fix_scores),
            "f1": _binary_f1(fix_targets, fix_scores),
        },
        "critic_harm_given_solver_correct": {
            "strict_condition_mask": True,
            "samples": len(correct_indices),
            "positive_samples": sum(harm_targets),
            "negative_samples": len(harm_targets) - sum(harm_targets),
            "pr_auc": _average_precision(harm_targets, harm_scores),
            "f1": _binary_f1(harm_targets, harm_scores),
        },
        "final_helpful_pr_auc": _average_precision(
            [label == "wrong_to_correct" for label in labels],
            probabilities["p_help"].tolist(),
        ),
        "final_harmful_pr_auc": _average_precision(
            [label == "correct_to_wrong" for label in labels],
            probabilities["p_harm"].tolist(),
        ),
        "four_class_macro_f1": _macro_f1(labels, probabilities["four_class"]),
    }


def cost_constant_baselines(
    cost_targets: torch.Tensor, cost_available: torch.Tensor
) -> dict[str, Any]:
    actual = _token_predictions(cost_targets[cost_available.to(torch.bool)])
    if actual.numel() == 0:
        raise ValueError("Cost baselines require available targets")
    values = [float(value) for value in actual]
    mean_value = statistics.fmean(values)
    median_value = statistics.median(values)
    constants = {
        "zero": 0.0,
        "training_mean": mean_value,
        "training_median": median_value,
    }
    metrics = {
        name: {
            "constant_total_tokens": value,
            "mae_total_tokens": statistics.fmean(
                abs(actual_value - value) for actual_value in values
            ),
        }
        for name, value in constants.items()
    }
    best_name = min(metrics, key=lambda name: metrics[name]["mae_total_tokens"])
    return {
        "samples": len(values),
        "target_transform": "log1p(Critic incremental total tokens)",
        "target_normalization": "none",
        "inverse_transform": "expm1(clamp(predicted_log, min=0, max=20))",
        "baselines": metrics,
        "best_baseline": best_name,
        "best_baseline_mae": metrics[best_name]["mae_total_tokens"],
        "fixed_training_median_total_tokens": median_value,
    }


def assess_cost_model(
    predicted_cost_logs: torch.Tensor,
    cost_targets: torch.Tensor,
    cost_available: torch.Tensor,
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    available = cost_available.to(torch.bool)
    predictions = _token_predictions(predicted_cost_logs[available])
    actual = _token_predictions(cost_targets[available])
    mae = float(torch.mean(torch.abs(predictions - actual)))
    best = float(baselines["best_baseline_mae"])
    relative_improvement = (best - mae) / best
    enabled = relative_improvement >= MIN_COST_IMPROVEMENT
    return {
        "oof_mae_total_tokens": mae,
        "best_constant_baseline": baselines["best_baseline"],
        "best_constant_baseline_mae": best,
        "required_relative_improvement": MIN_COST_IMPROVEMENT,
        "actual_relative_improvement": relative_improvement,
        "cost_model_enabled": enabled,
        "disabled_reason": (
            None
            if enabled
            else "OOF MAE does not beat the best constant baseline by at least 5%"
        ),
        "effective_cost_source": (
            "factorized_cost_head"
            if enabled
            else "fixed_training_median_total_tokens"
        ),
        "fixed_training_median_total_tokens": baselines[
            "fixed_training_median_total_tokens"
        ],
        "hard_budget_uses_cost_prediction": False,
        "hard_budget_guard": {
            "critic_completion_token_cap": 512,
            "critic_call_cap": 1,
        },
    }


def _cost_target_audit(
    training_examples: Sequence[TrainingExample],
    v1_dir: Path,
) -> dict[str, Any]:
    actual = [
        math.expm1(example.cost_log_target)
        for example in training_examples
        if example.cost_available
    ]
    roundtrip = [
        abs(math.expm1(math.log1p(value)) - value) for value in actual
    ]
    v1_files = {}
    for name in (
        "primary_model.pt",
        "seed_metrics.json",
        "oof_predictions.jsonl",
        "validation_predictions.jsonl",
        "summary.json",
        "report.md",
    ):
        path = v1_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Frozen v1 artifact is missing: {path}")
        v1_files[name] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
    checkpoint = torch.load(
        v1_dir / "primary_model.pt", map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("training_sha256") != TRAINING_SHA256
        or checkpoint.get("primary_seed") != PRIMARY_SEED
        or checkpoint.get("validation_used_for_training_or_selection") is not False
    ):
        raise ValueError("Frozen v1 checkpoint contract changed")
    return {
        "v1_artifacts": v1_files,
        "available_samples": len(actual),
        "masked_samples": len(training_examples) - len(actual),
        "stored_target": "log1p(Critic incremental total tokens)",
        "target_normalization": "none",
        "input_numeric_normalization": (
            "z-score fitted on OOF training fold or full Training 1000"
        ),
        "inverse_transform": "expm1(clamp(predicted_log, min=0, max=20))",
        "roundtrip_checked": True,
        "maximum_log1p_expm1_roundtrip_error": max(roundtrip),
        "v1_artifacts_modified": False,
    }


def _load_v1_decisions(
    v1_dir: Path,
    examples: Sequence[ProbeExample],
) -> tuple[list[bool], dict[str, Any]]:
    summary_path = v1_dir / "summary.json"
    rows_path = v1_dir / "validation_predictions.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("final_test_evaluated") is not False
        or summary.get("model_calls") != 0
        or summary["sources"]["training_examples"]["sha256"] != TRAINING_SHA256
    ):
        raise ValueError("Frozen Controller v1 summary changed")
    rows = [
        row
        for row in read_jsonl(rows_path)
        if row.get("seed") == PRIMARY_SEED and row.get("primary_seed") is True
    ]
    if len(rows) != len(examples):
        raise ValueError("Controller v1 must have 100 primary Validation rows")
    decisions = []
    for row, example in zip(rows, examples):
        if (
            row.get("question_id") != example.question_id
            or row.get("model_input_sha256") != _hash_model_input(example)
            or not isinstance(row.get("critic_called"), bool)
        ):
            raise ValueError("Controller v1 Validation identity changed")
        decisions.append(row["critic_called"])
    return decisions, {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _file_sha256(summary_path),
        "validation_predictions_path": str(rows_path.resolve()),
        "validation_predictions_sha256": _file_sha256(rows_path),
        "primary_seed": PRIMARY_SEED,
    }


def _probability_payload(
    probabilities: Mapping[str, torch.Tensor], index: int
) -> dict[str, Any]:
    return {
        "solver_error": float(probabilities["p_error"][index]),
        "critic_fix_given_solver_error": float(
            probabilities["p_fix_given_error"][index]
        ),
        "critic_harm_given_solver_correct": float(
            probabilities["p_harm_given_correct"][index]
        ),
        "helpful": float(probabilities["p_help"][index]),
        "harmful": float(probabilities["p_harm"][index]),
        "four_class": {
            label: float(probabilities["four_class"][index, class_index])
            for class_index, label in enumerate(LABELS)
        },
    }


def _validation_cost_mae(
    predicted_cost_logs: torch.Tensor, actual_cost_logs: torch.Tensor
) -> float:
    return float(
        torch.mean(
            torch.abs(
                _token_predictions(predicted_cost_logs)
                - _token_predictions(actual_cost_logs)
            )
        )
    )


def run_seed_training_v2(
    *,
    seed: int,
    training_examples: Sequence[TrainingExample],
    training_embeddings: torch.Tensor,
    training_numeric: torch.Tensor,
    validation_examples: Sequence[ProbeExample],
    validation_embeddings: torch.Tensor,
    validation_numeric: torch.Tensor,
    cost_baselines: Mapping[str, Any],
    epochs: int = EPOCHS,
) -> tuple[
    FactorizedPreCriticControllerV2,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    labels = [example.label for example in training_examples]
    cost_targets = torch.tensor(
        [example.cost_log_target for example in training_examples],
        dtype=torch.float32,
    )
    cost_available = torch.tensor(
        [example.cost_available for example in training_examples], dtype=torch.bool
    )
    oof_probs, oof_cost_logs, folds, fold_training = oof_factorized_predictions(
        training_embeddings,
        training_numeric,
        labels,
        cost_targets,
        cost_available,
        seed=seed,
        epochs=epochs,
    )
    oof_scores = oof_probs["gate_score"].tolist()
    threshold = select_oof_threshold_v1(oof_scores, labels)
    budget_thresholds = oof_budget_thresholds_v1(oof_scores)
    development_decisions = gated_decisions(oof_scores, threshold["threshold"])
    oof_budget_curve = [
        {
            **point,
            "oof_policy": label_policy_metrics(
                labels, gated_decisions(oof_scores, point["threshold"])
            ),
        }
        for point in budget_thresholds
    ]
    cost_protection = assess_cost_model(
        oof_cost_logs, cost_targets, cost_available, cost_baselines
    )
    oof_heads = {
        **factorized_head_metrics(oof_probs, labels),
        "cost_model": cost_protection,
    }

    standardized_training, standardized_validation, mean, std = _standardize(
        training_numeric, validation_numeric
    )
    model, final_training = train_factorized_model(
        training_embeddings,
        standardized_training,
        labels,
        cost_targets,
        cost_available,
        seed=seed,
        epochs=epochs,
    )
    with torch.no_grad():
        error, fix, harm, validation_cost_logs = model(
            validation_embeddings, standardized_validation
        )
    validation_probs = factorized_probabilities(error, fix, harm)
    validation_labels = [example.label for example in validation_examples]
    validation_scores = validation_probs["gate_score"].tolist()
    validation_decisions = gated_decisions(
        validation_scores, threshold["threshold"]
    )
    validation_cost_targets, _ = _validation_cost_targets(validation_examples)
    validation_heads = {
        **factorized_head_metrics(validation_probs, validation_labels),
        "cost_head_mae_total_tokens": _validation_cost_mae(
            validation_cost_logs, validation_cost_targets
        ),
        "cost_model_enabled_from_training_oof": cost_protection[
            "cost_model_enabled"
        ],
        "effective_cost_source": cost_protection["effective_cost_source"],
    }
    validation_budget_curve = [
        {
            **point,
            "validation_policy": evaluate_validation_policy(
                validation_examples,
                gated_decisions(validation_scores, point["threshold"]),
            ),
        }
        for point in budget_thresholds
    ]

    fold_by_index = {
        index: fold["fold"] for fold in folds for index in fold["validation"]
    }
    predicted_oof_tokens = _token_predictions(oof_cost_logs)
    effective_oof_tokens = (
        predicted_oof_tokens
        if cost_protection["cost_model_enabled"]
        else torch.full_like(
            predicted_oof_tokens,
            float(cost_baselines["fixed_training_median_total_tokens"]),
        )
    )
    oof_rows = []
    for index, example in enumerate(training_examples):
        model_input_sha = hashlib.sha256(
            json.dumps(
                example.model_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        oof_rows.append(
            {
                "controller_v2_factorized": True,
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
                "probabilities": _probability_payload(oof_probs, index),
                "gate_score": float(oof_probs["gate_score"][index]),
                "development_threshold": threshold["threshold"],
                "critic_called": development_decisions[index],
                "cost_available": example.cost_available,
                "raw_cost_head_prediction_total_tokens": float(
                    predicted_oof_tokens[index]
                ),
                "effective_cost_estimate_total_tokens": float(
                    effective_oof_tokens[index]
                ),
                "cost_model_enabled": cost_protection["cost_model_enabled"],
                "model_input_sha256": model_input_sha,
            }
        )

    predicted_validation_tokens = _token_predictions(validation_cost_logs)
    effective_validation_tokens = (
        predicted_validation_tokens
        if cost_protection["cost_model_enabled"]
        else torch.full_like(
            predicted_validation_tokens,
            float(cost_baselines["fixed_training_median_total_tokens"]),
        )
    )
    actual_validation_tokens = _token_predictions(validation_cost_targets)
    validation_rows = []
    for index, example in enumerate(validation_examples):
        decision = validation_decisions[index]
        selected = (
            example.critic_only_answer if decision else example.solver_answer
        )
        validation_rows.append(
            {
                "controller_v2_factorized": True,
                "development_validation": True,
                "final_test_evaluated": False,
                "deployable": False,
                "model_calls": 0,
                "seed": seed,
                "primary_seed": seed == PRIMARY_SEED,
                "question_id": example.question_id,
                "label": example.label,
                "probabilities": _probability_payload(validation_probs, index),
                "gate_score": float(validation_probs["gate_score"][index]),
                "development_threshold": threshold["threshold"],
                "threshold_source": threshold["selection_source"],
                "validation_used_for_threshold": False,
                "critic_called": decision,
                "selected_answer": selected,
                "correct": selected == example.gold,
                "raw_cost_head_prediction_total_tokens": float(
                    predicted_validation_tokens[index]
                ),
                "effective_cost_estimate_total_tokens": float(
                    effective_validation_tokens[index]
                ),
                "actual_critic_incremental_total_tokens": float(
                    actual_validation_tokens[index]
                ),
                "cost_model_enabled": cost_protection["cost_model_enabled"],
                "model_input_sha256": _hash_model_input(example),
            }
        )

    metrics = {
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
            "head_metrics": oof_heads,
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
            "head_metrics": validation_heads,
            "budget_curve": validation_budget_curve,
        },
    }
    return model, mean, std, metrics, oof_rows, validation_rows


def _stability_summary(seed_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "accuracy": lambda m: m["validation"]["controller_policy"]["accuracy"],
        "corrected": lambda m: m["validation"]["controller_policy"]["corrected"],
        "degraded": lambda m: m["validation"]["controller_policy"]["degraded"],
        "net_benefit": lambda m: m["validation"]["controller_policy"]["net_benefit"],
        "critic_call_rate": lambda m: m["validation"]["controller_policy"]["critic_call_rate"],
        "mean_total_tokens": lambda m: m["validation"]["controller_policy"]["cost"]["mean"]["total_tokens"],
        "mean_calls": lambda m: m["validation"]["controller_policy"]["cost"]["mean"]["calls"],
        "mean_latency_seconds": lambda m: m["validation"]["controller_policy"]["cost"]["mean"]["latency_seconds"],
        "solver_error_pr_auc": lambda m: m["validation"]["head_metrics"]["solver_error"]["pr_auc"],
        "solver_error_f1": lambda m: m["validation"]["head_metrics"]["solver_error"]["f1"],
        "critic_fix_pr_auc": lambda m: m["validation"]["head_metrics"]["critic_fix_given_solver_error"]["pr_auc"],
        "critic_fix_f1": lambda m: m["validation"]["head_metrics"]["critic_fix_given_solver_error"]["f1"],
        "critic_harm_pr_auc": lambda m: m["validation"]["head_metrics"]["critic_harm_given_solver_correct"]["pr_auc"],
        "critic_harm_f1": lambda m: m["validation"]["head_metrics"]["critic_harm_given_solver_correct"]["f1"],
        "final_helpful_pr_auc": lambda m: m["validation"]["head_metrics"]["final_helpful_pr_auc"],
        "final_harmful_pr_auc": lambda m: m["validation"]["head_metrics"]["final_harmful_pr_auc"],
        "four_class_macro_f1": lambda m: m["validation"]["head_metrics"]["four_class_macro_f1"],
        "cost_head_mae": lambda m: m["validation"]["head_metrics"]["cost_head_mae_total_tokens"],
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


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _report(summary: Mapping[str, Any]) -> str:
    comparison = summary["development_validation"]["policy_comparison"]
    primary = summary["primary_seed_metrics"]
    lines = [
        "# Factorized Pre-Critic Controller v2",
        "",
        "Offline supervised label-factorization experiment. "
        "final_test_evaluated=false, deployable=false, and model_calls=0.",
        "",
        "Training data, frozen MiniLM, feature schema, 64d trunk, folds, seeds, "
        "optimizer, and epochs match Controller v1. Validation 100 is evaluation-only.",
        "",
        "## Validation 100",
        "",
        "| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean tokens | Mean calls | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "STOP",
        "ALWAYS_CRITIC_ONLY",
        "OLD_PROBE",
        "CONTROLLER_V1_PRIMARY",
        "CONTROLLER_V2_FACTORIZED_PRIMARY",
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
    cost = summary["cost_model_protection"]
    lines.extend(
        [
            "",
            "## Cost protection",
            "",
            f"- v1 target normalization: {summary['v1_cost_target_audit']['target_normalization']}",
            f"- Best constant baseline: {cost['best_constant_baseline']} "
            f"(MAE={cost['best_constant_baseline_mae']:.3f})",
            f"- Primary OOF cost-head MAE: {cost['oof_mae_total_tokens']:.3f}",
            f"- Relative improvement: {cost['actual_relative_improvement']:.6f}",
            f"- cost_model_enabled: {cost['cost_model_enabled']}",
            f"- Effective source: {cost['effective_cost_source']}",
            "",
            "## Primary factorized heads",
            "",
            "| Split | Error PR/F1 | Fix PR/F1 | Harm PR/F1 | Helpful PR | Harmful PR | Macro-F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("oof", "validation"):
        head = primary[split]["head_metrics"]
        lines.append(
            f"| {split.upper()} | {head['solver_error']['pr_auc']:.6f}/"
            f"{head['solver_error']['f1']:.6f} | "
            f"{head['critic_fix_given_solver_error']['pr_auc']:.6f}/"
            f"{head['critic_fix_given_solver_error']['f1']:.6f} | "
            f"{head['critic_harm_given_solver_correct']['pr_auc']:.6f}/"
            f"{head['critic_harm_given_solver_correct']['f1']:.6f} | "
            f"{head['final_helpful_pr_auc']:.6f} | "
            f"{head['final_harmful_pr_auc']:.6f} | "
            f"{head['four_class_macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Primary OOF-derived budget curve",
            "",
            "All points are shown. No final operating point is selected.",
            "",
            "| Budget | Threshold | OOF calls | OOF C/D/Net | Val calls | Val accuracy | Val C/D/Net | Mean val tokens |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for oof_point, validation_point in zip(
        primary["oof"]["budget_curve"], primary["validation"]["budget_curve"]
    ):
        oof = oof_point["oof_policy"]
        validation = validation_point["validation_policy"]
        lines.append(
            f"| {oof_point['target_budget_rate']:.0%} | "
            f"{oof_point['threshold']:.8f} | {oof['critic_calls']} | "
            f"{oof['corrected']}/{oof['degraded']}/{oof['net_benefit']} | "
            f"{validation['critic_calls']} | {validation['accuracy']:.4f} | "
            f"{validation['corrected']}/{validation['degraded']}/"
            f"{validation['net_benefit']} | "
            f"{validation['cost']['mean']['total_tokens']:.2f} |"
        )
    lines.extend(["", "## Five-seed stability", ""])
    for name, metric in summary["stability"]["metrics"].items():
        lines.append(f"- {name}: {metric['mean']:.6f} ± {metric['std']:.6f}")
    mcnemar = summary["development_validation"]["mcnemar"]
    lines.extend(
        [
            "",
            "## Exact McNemar",
            "",
            f"- v2 vs STOP: p={mcnemar['v2_vs_stop']['p_value']:.8f}",
            f"- v2 vs Always Critic: p={mcnemar['v2_vs_always']['p_value']:.8f}",
            f"- v2 vs v1: p={mcnemar['v2_vs_v1']['p_value']:.8f}",
            "",
            "The posthoc oracle uses gold after generation and is deployable=false.",
            "Final Test 500 remains sealed and was not read or evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


def train_precritic_controller_v2(
    *,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    validation_path: str | Path = DEFAULT_VALIDATION,
    validation_summary_path: str | Path = DEFAULT_VALIDATION_SUMMARY,
    old_probe_predictions_path: str | Path = DEFAULT_OLD_PROBE,
    old_probe_summary_path: str | Path = DEFAULT_OLD_PROBE_SUMMARY,
    controller_v1_dir: str | Path = DEFAULT_CONTROLLER_V1_DIR,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    encoder: FrozenEncoder | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output_names = (
        "primary_model.pt",
        "seed_metrics.json",
        "oof_predictions.jsonl",
        "validation_predictions.jsonl",
        "summary.json",
        "report.md",
    )
    if any((output / name).exists() for name in output_names):
        raise FileExistsError("Controller v2 artifacts already exist; refusing to overwrite")

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
    v1_decisions, v1_sources = _load_v1_decisions(
        Path(controller_v1_dir), validation_examples
    )
    v1_cost_audit = _cost_target_audit(
        training_examples, Path(controller_v1_dir)
    )
    v1_hashes_before = {
        name: item["sha256"] for name, item in v1_cost_audit["v1_artifacts"].items()
    }

    labels = [example.label for example in training_examples]
    target_contract = factorized_targets(labels)
    target_counts = {
        "solver_error": {
            "positive": int(target_contract["solver_error"].sum()),
            "negative": len(labels) - int(target_contract["solver_error"].sum()),
        },
        "critic_fix_given_solver_error": {
            "condition_samples": int(target_contract["critic_fix_mask"].sum()),
            "positive": int(target_contract["critic_fix"].sum()),
            "negative": int(target_contract["critic_fix_mask"].sum())
            - int(target_contract["critic_fix"].sum()),
        },
        "critic_harm_given_solver_correct": {
            "condition_samples": int(target_contract["critic_harm_mask"].sum()),
            "positive": int(target_contract["critic_harm"].sum()),
            "negative": int(target_contract["critic_harm_mask"].sum())
            - int(target_contract["critic_harm"].sum()),
        },
    }
    expected_counts = {
        "solver_error": {"positive": 286, "negative": 714},
        "critic_fix_given_solver_error": {
            "condition_samples": 286,
            "positive": 64,
            "negative": 222,
        },
        "critic_harm_given_solver_correct": {
            "condition_samples": 714,
            "positive": 79,
            "negative": 635,
        },
    }
    if target_counts != expected_counts:
        raise ValueError("Frozen factorized target counts changed")

    cost_targets = torch.tensor(
        [example.cost_log_target for example in training_examples],
        dtype=torch.float32,
    )
    cost_available = torch.tensor(
        [example.cost_available for example in training_examples], dtype=torch.bool
    )
    baselines = cost_constant_baselines(cost_targets, cost_available)
    validation_summary = json.loads(
        Path(validation_summary_path).read_text(encoding="utf-8")
    )
    if (
        validation_summary.get("samples") != EXPECTED_VALIDATION_SAMPLES
        or validation_summary.get("generation_caps", {}).get("critic") != 512
        or validation_summary.get("mock_only") is not False
    ):
        raise ValueError("Frozen Validation summary or Critic cap changed")

    active_encoder = encoder or OfflineMiniLMEncoder(
        model_name=MODEL_NAME, device="cpu"
    )
    if getattr(active_encoder, "mock_only", True):
        raise ValueError("Formal Controller v2 forbids mock/hash encoders")
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

    seed_metrics = []
    oof_rows = []
    validation_rows = []
    primary_model = None
    primary_mean = None
    primary_std = None
    for seed in ALL_SEEDS:
        model, mean, std, metrics, seed_oof, seed_validation = run_seed_training_v2(
            seed=seed,
            training_examples=training_examples,
            training_embeddings=training_embeddings,
            training_numeric=training_numeric,
            validation_examples=validation_examples,
            validation_embeddings=validation_embeddings,
            validation_numeric=validation_numeric,
            cost_baselines=baselines,
        )
        seed_metrics.append(metrics)
        oof_rows.extend(seed_oof)
        validation_rows.extend(seed_validation)
        if seed == PRIMARY_SEED:
            primary_model = model
            primary_mean = mean
            primary_std = std
    if primary_model is None or primary_mean is None or primary_std is None:
        raise AssertionError("Fixed primary v2 model was not trained")

    primary_metrics = next(
        item for item in seed_metrics if item["seed"] == PRIMARY_SEED
    )
    primary_rows = [
        row for row in validation_rows if row["seed"] == PRIMARY_SEED
    ]
    v2_decisions = [bool(row["critic_called"]) for row in primary_rows]
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
            **evaluate_validation_policy(validation_examples, v1_decisions),
            "frozen_historical_policy": True,
            "seed": PRIMARY_SEED,
        },
        "CONTROLLER_V2_FACTORIZED_PRIMARY": {
            **evaluate_validation_policy(validation_examples, v2_decisions),
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
        "v2_vs_stop": paired_mcnemar(
            validation_examples,
            v2_decisions,
            stop_decisions,
            first_name="CONTROLLER_V2_FACTORIZED_PRIMARY",
            second_name="STOP",
        ),
        "v2_vs_always": paired_mcnemar(
            validation_examples,
            v2_decisions,
            always_decisions,
            first_name="CONTROLLER_V2_FACTORIZED_PRIMARY",
            second_name="ALWAYS_CRITIC_ONLY",
        ),
        "v2_vs_v1": paired_mcnemar(
            validation_examples,
            v2_decisions,
            v1_decisions,
            first_name="CONTROLLER_V2_FACTORIZED_PRIMARY",
            second_name="CONTROLLER_V1_PRIMARY",
        ),
    }
    stability = _stability_summary(seed_metrics)
    cost_protection = primary_metrics["oof"]["head_metrics"]["cost_model"]
    sources = {
        "training_examples": {
            "path": str(Path(training_path).resolve()),
            "sha256": _file_sha256(Path(training_path)),
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
        "validation_summary": {
            "path": str(Path(validation_summary_path).resolve()),
            "sha256": _file_sha256(Path(validation_summary_path)),
        },
        "old_probe": old_probe_sources,
        "controller_v1": v1_sources,
        "final_test_guard": final_guard,
    }
    seed_payload = {
        "controller_v2_factorized": True,
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
        "cost_constant_baselines": baselines,
        "seeds": seed_metrics,
        "stability": stability,
    }
    summary = {
        "controller_v2_factorized": True,
        "controller_trained": True,
        "label_factorization_experiment": True,
        "development_validation": {
            "evaluated_after_oof_threshold_freeze": True,
            "policy_comparison": comparison,
            "mcnemar": mcnemar,
        },
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
            "validation_used_for_cost_model_enablement": False,
            "validation_used_for_model_or_seed_selection": False,
            "final_test_manifest_verified_only": True,
            "final_test_examples_read": False,
        },
        "model": {
            "encoder": active_encoder.name,
            "encoder_frozen": True,
            "encoder_local_files_only": True,
            "device": "cpu",
            "embedding_dim": active_encoder.dimension,
            "embedding_encode_calls": 1,
            "shared_trunk": "single hidden layer",
            "hidden_dim": HIDDEN_DIM,
            "numeric_features": list(NUMERIC_FEATURES),
            "heads": [
                "solver_error",
                "critic_fix_given_solver_error",
                "critic_harm_given_solver_correct",
                "cost_log1p_total_tokens",
            ],
            "gate_score": (
                "p_error*p_fix - (1-p_error)*p_harm_given_correct"
            ),
        },
        "factorized_target_contract": target_counts,
        "training_protocol": {
            "primary_seed": PRIMARY_SEED,
            "stability_seeds": list(STABILITY_SEEDS),
            "primary_model_always_uses_primary_seed": True,
            "best_seed_selected": False,
            "folds": N_SPLITS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "hidden_dim": HIDDEN_DIM,
            "balanced_binary_cross_entropy": True,
            "strict_conditional_masks": True,
            "cost_loss": "masked Huber",
            "cost_loss_weight": COST_LOSS_WEIGHT,
            "cost_available_samples": 800,
            "cost_masked_samples": 200,
            "hyperparameter_search": False,
            "budget_rates": list(BUDGET_RATES),
            "deployment_operating_point_selected": False,
        },
        "v1_cost_target_audit": v1_cost_audit,
        "cost_constant_baselines": baselines,
        "cost_model_protection": cost_protection,
        "primary_seed_metrics": primary_metrics,
        "stability": stability,
    }
    checkpoint = {
        "controller_v2_factorized": True,
        "controller_trained": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "deployable": False,
        "model_calls": 0,
        "model_class": "FactorizedPreCriticControllerV2",
        "model_state_dict": primary_model.state_dict(),
        "embedding_dim": active_encoder.dimension,
        "numeric_dim": len(NUMERIC_FEATURES),
        "hidden_dim": HIDDEN_DIM,
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
        "training_sha256": TRAINING_SHA256,
        "validation_used_for_training_or_selection": False,
        "final_test_guard": final_guard,
        "factorized_target_contract": target_counts,
        "cost_model_protection": cost_protection,
        "cost_constant_baselines": baselines,
        "hard_budget_guard": {
            "critic_completion_token_cap": 512,
            "critic_call_cap": 1,
            "uses_predicted_cost": False,
        },
        "loss": (
            "sum of three class-balanced condition-masked BCE losses "
            "+ 0.1 * masked Huber cost loss"
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    _save_checkpoint_atomic(output / "primary_model.pt", checkpoint)
    _write_json_atomic(output / "seed_metrics.json", seed_payload)
    write_jsonl(output / "oof_predictions.jsonl", oof_rows)
    write_jsonl(output / "validation_predictions.jsonl", validation_rows)
    _write_json_atomic(output / "summary.json", summary)
    _write_text_atomic(output / "report.md", _report(summary))

    v1_hashes_after = {
        name: _file_sha256(Path(item["path"]))
        for name, item in v1_cost_audit["v1_artifacts"].items()
    }
    if v1_hashes_after != v1_hashes_before:
        raise RuntimeError("Frozen Controller v1 artifacts changed during v2 training")
    return {
        "controller_v2_factorized": True,
        "controller_trained": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "deployable": False,
        "model_calls": 0,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "cost_model_enabled": cost_protection["cost_model_enabled"],
        "output_dir": str(output.resolve()),
    }
