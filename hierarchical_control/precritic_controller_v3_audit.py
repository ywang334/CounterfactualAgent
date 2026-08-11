"""Pure-offline generalization and stability audit for Controller v3.

This module deliberately contains no encoder, optimizer, backward, backend, or
network path.  It replays the frozen primary controller from cached embeddings
and audits the already-saved out-of-fold predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .io_utils import read_jsonl, write_jsonl
from .precritic_controller_v1 import (
    ALL_SEEDS,
    DEFAULT_FINAL_MANIFEST,
    DEFAULT_TRAINING,
    DEFAULT_TRAINING_MANIFEST,
    DEFAULT_VALIDATION,
    LABELS,
    PRIMARY_SEED,
    TRAINING_SHA256,
    _average_precision,
    _file_sha256,
    _macro_f1,
    gated_decisions,
    label_policy_metrics,
    load_training_examples,
    load_validation_examples,
    verify_sealed_final_manifest,
)
from .precritic_controller_v3 import PreCriticControllerV3
from .precritic_controller_v3_training import (
    DEFAULT_OUTPUT as DEFAULT_CONTROLLER_DIR,
    NUMERIC_STATE_DIM,
    _cached_split,
    _predict,
    apply_state_normalization,
    StateNormalization,
)


DEFAULT_OUTPUT = DEFAULT_CONTROLLER_DIR / "generalization_audit"
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_REPLICATES = 2_000
CALIBRATION_BINS = 10
TOP_RATES = (0.01, 0.05, 0.10, 0.20, 0.50)
AUDIT_FILES = ("audit_summary.json", "audit_cases.jsonl", "audit_report.md")
REQUIRED_CONTROLLER_FILES = (
    "feature_cache.pt",
    "primary_model.pt",
    "seed_metrics.json",
    "oof_predictions.jsonl",
    "validation_predictions.jsonl",
    "summary.json",
    "report.md",
)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "mean": None, "std": None, "quantiles": {}}
    numeric = [float(value) for value in values]
    return {
        "samples": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.pstdev(numeric),
        "quantiles": {
            name: _quantile(numeric, probability)
            for name, probability in (
                ("p00", 0.0),
                ("p05", 0.05),
                ("p10", 0.10),
                ("p25", 0.25),
                ("p50", 0.50),
                ("p75", 0.75),
                ("p90", 0.90),
                ("p95", 0.95),
                ("p100", 1.0),
            )
        },
    }


def _binary_f1(targets: Sequence[bool], probabilities: Sequence[float]) -> float:
    predictions = [float(value) >= 0.5 for value in probabilities]
    true_positive = sum(prediction and target for prediction, target in zip(predictions, targets))
    false_positive = sum(prediction and not target for prediction, target in zip(predictions, targets))
    false_negative = sum(not prediction and target for prediction, target in zip(predictions, targets))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def binary_calibration(
    targets: Sequence[bool], probabilities: Sequence[float], bins: int = CALIBRATION_BINS
) -> dict[str, Any]:
    """Return Brier score and equal-width expected calibration error."""
    if not targets or len(targets) != len(probabilities):
        raise ValueError("Calibration targets and probabilities must be aligned and non-empty")
    if bins <= 0:
        raise ValueError("Calibration bins must be positive")
    numeric_targets = [1.0 if value else 0.0 for value in targets]
    numeric_probabilities = [float(value) for value in probabilities]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in numeric_probabilities):
        raise ValueError("Calibration probabilities must be finite values in [0, 1]")
    rows = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position
            for position, value in enumerate(numeric_probabilities)
            if value >= lower and (value < upper or (index == bins - 1 and value <= upper))
        ]
        if selected:
            confidence = statistics.fmean(numeric_probabilities[position] for position in selected)
            frequency = statistics.fmean(numeric_targets[position] for position in selected)
            contribution = len(selected) / len(targets) * abs(confidence - frequency)
            ece += contribution
        else:
            confidence = frequency = contribution = None
        rows.append(
            {
                "bin": index,
                "lower_inclusive": lower,
                "upper_inclusive_only_for_last": upper,
                "count": len(selected),
                "mean_probability": confidence,
                "positive_frequency": frequency,
                "ece_contribution": contribution,
            }
        )
    return {
        "samples": len(targets),
        "brier_score": statistics.fmean(
            (probability - target) ** 2
            for probability, target in zip(numeric_probabilities, numeric_targets)
        ),
        "ece_10_bin": ece,
        "bins": rows,
    }


def multiclass_calibration(
    labels: Sequence[str], probabilities: Sequence[Sequence[float]], bins: int = CALIBRATION_BINS
) -> dict[str, Any]:
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Multiclass labels and probabilities must be aligned and non-empty")
    confidences = []
    correctness = []
    brier_rows = []
    for label, row in zip(labels, probabilities):
        if len(row) != len(LABELS):
            raise ValueError("Multiclass probability width changed")
        numeric = [float(value) for value in row]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in numeric):
            raise ValueError("Multiclass probabilities are invalid")
        if not math.isclose(sum(numeric), 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Multiclass probabilities do not sum to one")
        predicted = max(range(len(numeric)), key=numeric.__getitem__)
        target = LABELS.index(label)
        confidences.append(numeric[predicted])
        correctness.append(predicted == target)
        brier_rows.append(
            sum((value - (1.0 if index == target else 0.0)) ** 2 for index, value in enumerate(numeric))
        )
    calibration = binary_calibration(correctness, confidences, bins)
    return {
        "samples": len(labels),
        "brier_score": statistics.fmean(brier_rows),
        "ece_10_bin": calibration["ece_10_bin"],
        "bins": calibration["bins"],
    }


def _extract_probabilities(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "solver_error": [float(row["probabilities"]["solver_error"]) for row in rows],
        "critic_fix": [
            float(row["probabilities"]["critic_fix_given_solver_error"]) for row in rows
        ],
        "critic_harm": [
            float(row["probabilities"]["critic_harm_given_solver_correct"]) for row in rows
        ],
        "helpful": [float(row["probabilities"]["helpful"]) for row in rows],
        "harmful": [float(row["probabilities"]["harmful"]) for row in rows],
        "factorized": [
            [float(row["probabilities"]["factorized_four_class"][label]) for label in LABELS]
            for row in rows
        ],
        "auxiliary": [
            [float(row["probabilities"]["auxiliary_four_class"][label]) for label in LABELS]
            for row in rows
        ],
        "gate_score": [float(row["gate_score"]) for row in rows],
    }


def diagnostic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute discrimination, classification, and calibration diagnostics."""
    if not rows:
        raise ValueError("Diagnostic metrics require rows")
    labels = [str(row["label"]) for row in rows]
    if any(label not in LABELS for label in labels):
        raise ValueError("Unknown transition label")
    probabilities = _extract_probabilities(rows)
    solver_error_targets = [label.startswith("wrong_to_") for label in labels]
    fix_indices = [index for index, target in enumerate(solver_error_targets) if target]
    harm_indices = [index for index, target in enumerate(solver_error_targets) if not target]

    def conditional(name: str, indices: Sequence[int], positive_label: str) -> dict[str, Any]:
        targets = [labels[index] == positive_label for index in indices]
        values = [probabilities[name][index] for index in indices]
        return {
            "samples": len(indices),
            "positive_samples": sum(targets),
            "pr_auc": _average_precision(targets, values),
            "f1": _binary_f1(targets, values),
            **binary_calibration(targets, values),
        }

    helpful_targets = [label == "wrong_to_correct" for label in labels]
    harmful_targets = [label == "correct_to_wrong" for label in labels]
    return {
        "samples": len(rows),
        "solver_error": {
            "positive_samples": sum(solver_error_targets),
            "pr_auc": _average_precision(solver_error_targets, probabilities["solver_error"]),
            "f1": _binary_f1(solver_error_targets, probabilities["solver_error"]),
            **binary_calibration(solver_error_targets, probabilities["solver_error"]),
        },
        "critic_fix_given_solver_error": conditional(
            "critic_fix", fix_indices, "wrong_to_correct"
        ),
        "critic_harm_given_solver_correct": conditional(
            "critic_harm", harm_indices, "correct_to_wrong"
        ),
        "helpful": {
            "positive_samples": sum(helpful_targets),
            "pr_auc": _average_precision(helpful_targets, probabilities["helpful"]),
            **binary_calibration(helpful_targets, probabilities["helpful"]),
        },
        "harmful": {
            "positive_samples": sum(harmful_targets),
            "pr_auc": _average_precision(harmful_targets, probabilities["harmful"]),
            **binary_calibration(harmful_targets, probabilities["harmful"]),
        },
        "factorized_four_class": {
            "macro_f1": _macro_f1(labels, torch.tensor(probabilities["factorized"])),
            **multiclass_calibration(labels, probabilities["factorized"]),
        },
        "auxiliary_four_class": {
            "macro_f1": _macro_f1(labels, torch.tensor(probabilities["auxiliary"])),
            **multiclass_calibration(labels, probabilities["auxiliary"]),
        },
    }


def _label_distribution(labels: Sequence[str]) -> dict[str, Any]:
    counts = Counter(labels)
    return {
        "samples": len(labels),
        "counts": {label: counts.get(label, 0) for label in LABELS},
        "rates": {label: counts.get(label, 0) / len(labels) for label in LABELS},
    }


def _group_diagnostic(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    labels = [str(row["label"]) for row in rows]
    scores = [float(row["gate_score"]) for row in rows]
    decisions = gated_decisions(scores, threshold)
    return {
        "label_distribution": _label_distribution(labels),
        "head_metrics": diagnostic_metrics(rows),
        "policy_at_frozen_primary_threshold": label_policy_metrics(labels, decisions),
        "gate_score_distribution": _distribution(scores),
        "p_help_distribution": _distribution(
            [float(row["probabilities"]["helpful"]) for row in rows]
        ),
        "p_harm_distribution": _distribution(
            [float(row["probabilities"]["harmful"]) for row in rows]
        ),
    }


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        value = float(values[order[start]])
        while end < len(order) and float(values[order[end]]) == value:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in order[start:end]:
            result[position] = rank
        start = end
    return result


def pearson_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second) or not first:
        raise ValueError("Correlation inputs must be aligned and non-empty")
    first_mean = statistics.fmean(first)
    second_mean = statistics.fmean(second)
    numerator = sum((x - first_mean) * (y - second_mean) for x, y in zip(first, second))
    first_scale = math.sqrt(sum((x - first_mean) ** 2 for x in first))
    second_scale = math.sqrt(sum((y - second_mean) ** 2 for y in second))
    if first_scale == 0.0 or second_scale == 0.0:
        return None
    return numerator / (first_scale * second_scale)


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    return pearson_correlation(_ranks(first), _ranks(second))


def set_overlap(first: set[str], second: set[str]) -> dict[str, Any]:
    intersection = first & second
    union = first | second
    if not union:
        jaccard = 1.0
    else:
        jaccard = len(intersection) / len(union)
    minimum = min(len(first), len(second))
    overlap_coefficient = 1.0 if not first and not second else (0.0 if minimum == 0 else len(intersection) / minimum)
    return {
        "first": len(first),
        "second": len(second),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": jaccard,
        "overlap_coefficient": overlap_coefficient,
    }


def _top_ids(rows: Sequence[Mapping[str, Any]], rate: float) -> set[str]:
    count = max(1, int(round(len(rows) * rate)))
    ranked = sorted(rows, key=lambda row: (-float(row["gate_score"]), str(row["sample_id"])))
    return {str(row["sample_id"]) for row in ranked[:count]}


def seed_stability(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    seeds = sorted(rows_by_seed)
    if seeds != list(ALL_SEEDS):
        raise ValueError("Seed stability requires the frozen five-seed protocol")
    ordered: dict[int, list[Mapping[str, Any]]] = {}
    identity: list[str] | None = None
    for seed in seeds:
        rows = sorted(rows_by_seed[seed], key=lambda row: str(row["sample_id"]))
        ids = [str(row["sample_id"]) for row in rows]
        if len(ids) != 1000 or len(set(ids)) != 1000:
            raise ValueError("Each seed must contain exactly 1000 unique OOF samples")
        if identity is None:
            identity = ids
        elif ids != identity:
            raise ValueError("OOF seed identities do not align")
        ordered[seed] = rows

    pairs = []
    for left_index, left_seed in enumerate(seeds):
        for right_seed in seeds[left_index + 1 :]:
            left = ordered[left_seed]
            right = ordered[right_seed]
            left_scores = [float(row["gate_score"]) for row in left]
            right_scores = [float(row["gate_score"]) for row in right]
            top = {}
            for rate in TOP_RATES:
                top[f"{int(rate * 100)}pct"] = set_overlap(
                    _top_ids(left, rate), _top_ids(right, rate)
                )
            selected_sets = {}
            for label, name in (
                ("wrong_to_correct", "corrected"),
                ("correct_to_wrong", "degraded"),
            ):
                first = {
                    str(row["sample_id"])
                    for row in left
                    if row["label"] == label and bool(row["critic_called"])
                }
                second = {
                    str(row["sample_id"])
                    for row in right
                    if row["label"] == label and bool(row["critic_called"])
                }
                selected_sets[name] = set_overlap(first, second)
            pairs.append(
                {
                    "left_seed": left_seed,
                    "right_seed": right_seed,
                    "pearson": pearson_correlation(left_scores, right_scores),
                    "spearman": spearman_correlation(left_scores, right_scores),
                    "top_sample_overlap": top,
                    "selected_outcome_overlap": selected_sets,
                }
            )

    per_seed = []
    for seed in seeds:
        rows = ordered[seed]
        labels = [str(row["label"]) for row in rows]
        decisions = [bool(row["critic_called"]) for row in rows]
        thresholds = {float(row["development_threshold"]) for row in rows}
        if len(thresholds) != 1:
            raise ValueError("A seed contains inconsistent thresholds")
        per_seed.append(
            {
                "seed": seed,
                "threshold": thresholds.pop(),
                **label_policy_metrics(labels, decisions),
            }
        )

    def pair_summary(path: str) -> dict[str, Any]:
        values = []
        for pair in pairs:
            value: Any = pair
            for key in path.split("."):
                value = value[key]
            if value is not None:
                values.append(float(value))
        return _distribution(values)

    return {
        "seeds": seeds,
        "pairs": pairs,
        "pairwise_summary": {
            "pearson": pair_summary("pearson"),
            "spearman": pair_summary("spearman"),
            "top_jaccard": {
                f"{int(rate * 100)}pct": pair_summary(
                    f"top_sample_overlap.{int(rate * 100)}pct.jaccard"
                )
                for rate in TOP_RATES
            },
            "corrected_jaccard": pair_summary("selected_outcome_overlap.corrected.jaccard"),
            "degraded_jaccard": pair_summary("selected_outcome_overlap.degraded.jaccard"),
        },
        "threshold_and_policy": per_seed,
        "threshold_distribution": _distribution([row["threshold"] for row in per_seed]),
        "critic_call_rate_distribution": _distribution([row["critic_call_rate"] for row in per_seed]),
        "net_benefit_distribution": _distribution([row["net_benefit"] for row in per_seed]),
    }


def _fast_threshold(scores: Sequence[float], labels: Sequence[str]) -> dict[str, Any]:
    """Exact equivalent of the frozen threshold objective in O(n log n)."""
    if not scores or len(scores) != len(labels):
        raise ValueError("Threshold scores and labels must align")
    ordered = sorted(zip(map(float, scores), labels), key=lambda item: item[0], reverse=True)
    above = max(scores) + max(abs(max(scores)), 1.0) * 1e-7
    best = {"threshold": above, "corrected": 0, "degraded": 0, "net_benefit": 0, "critic_calls": 0}
    corrected = degraded = calls = 0
    index = 0
    while index < len(ordered):
        threshold = ordered[index][0]
        end = index
        while end < len(ordered) and ordered[end][0] == threshold:
            label = ordered[end][1]
            corrected += label == "wrong_to_correct"
            degraded += label == "correct_to_wrong"
            calls += 1
            end += 1
        candidate = {
            "threshold": threshold,
            "corrected": corrected,
            "degraded": degraded,
            "net_benefit": corrected - degraded,
            "critic_calls": calls,
        }
        if (candidate["net_benefit"], -candidate["critic_calls"], candidate["threshold"]) > (
            best["net_benefit"], -best["critic_calls"], best["threshold"]
        ):
            best = candidate
        index = end
    safe = best["net_benefit"] <= 0
    if safe:
        best = {"threshold": above, "corrected": 0, "degraded": 0, "net_benefit": 0, "critic_calls": 0}
    return {
        **best,
        "critic_call_rate": best["critic_calls"] / len(labels),
        "safe_fallback_always_stop": safe,
    }


def stratified_bootstrap(
    scores: Sequence[float],
    labels: Sequence[str],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("Bootstrap scores and labels must align")
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    strata = {label: [index for index, value in enumerate(labels) if value == label] for label in LABELS}
    if any(not indices for indices in strata.values()):
        raise ValueError("Stratified bootstrap requires all four labels")
    generator = random.Random(seed)
    records = []
    for replicate in range(replicates):
        selected = []
        for label in LABELS:
            indices = strata[label]
            selected.extend(generator.choices(indices, k=len(indices)))
        sampled_scores = [float(scores[index]) for index in selected]
        sampled_labels = [labels[index] for index in selected]
        optimum = _fast_threshold(sampled_scores, sampled_labels)
        selected_identity = set(selected)
        oob = [index for index in range(len(labels)) if index not in selected_identity]
        if not oob:
            raise RuntimeError("Bootstrap replicate unexpectedly has no out-of-bag samples")
        oob_labels = [labels[index] for index in oob]
        oob_decisions = [float(scores[index]) >= optimum["threshold"] for index in oob]
        oob_policy = label_policy_metrics(oob_labels, oob_decisions)
        records.append(
            {
                "replicate": replicate,
                "threshold": optimum["threshold"],
                "critic_call_rate": optimum["critic_call_rate"],
                "net_benefit": optimum["net_benefit"],
                "safe_fallback_always_stop": optimum["safe_fallback_always_stop"],
                "oob_samples": len(oob),
                "oob_critic_call_rate": oob_policy["critic_call_rate"],
                "oob_corrected": oob_policy["corrected"],
                "oob_degraded": oob_policy["degraded"],
                "oob_net_benefit": oob_policy["net_benefit"],
                "oob_accuracy": oob_policy["accuracy"],
            }
        )

    def interval(field: str) -> dict[str, Any]:
        values = [float(record[field]) for record in records]
        return {
            "mean": statistics.fmean(values),
            "p2_5": _quantile(values, 0.025),
            "p50": _quantile(values, 0.5),
            "p97_5": _quantile(values, 0.975),
        }

    return {
        "diagnostic_only": True,
        "independent_test_performance": False,
        "seed": seed,
        "replicates": replicates,
        "stratified_by_four_class_label": True,
        "selection_objective_unchanged": "maximize corrected-degraded; ties use fewer calls",
        "optimal_threshold_95_interval": interval("threshold"),
        "selected_critic_call_rate_95_interval": interval("critic_call_rate"),
        "selected_net_benefit_95_interval": interval("net_benefit"),
        "safe_fallback_to_stop_count": sum(record["safe_fallback_always_stop"] for record in records),
        "safe_fallback_to_stop_rate": statistics.fmean(
            record["safe_fallback_always_stop"] for record in records
        ),
        "out_of_bag_threshold_transfer": {
            "samples_95_interval": interval("oob_samples"),
            "critic_call_rate_95_interval": interval("oob_critic_call_rate"),
            "corrected_95_interval": interval("oob_corrected"),
            "degraded_95_interval": interval("oob_degraded"),
            "net_benefit_95_interval": interval("oob_net_benefit"),
            "accuracy_95_interval": interval("oob_accuracy"),
        },
        "records_sha256": _json_sha256(records),
    }


def _histogram_overlap(first: Sequence[float], second: Sequence[float], bins: int = 100) -> float:
    if not first or not second:
        raise ValueError("Distribution overlap requires two non-empty samples")
    lower = min(min(first), min(second))
    upper = max(max(first), max(second))
    if lower == upper:
        return 1.0
    first_histogram = [0] * bins
    second_histogram = [0] * bins
    for values, histogram in ((first, first_histogram), (second, second_histogram)):
        for value in values:
            index = min(bins - 1, int((float(value) - lower) / (upper - lower) * bins))
            histogram[index] += 1
    return sum(
        min(left / len(first), right / len(second))
        for left, right in zip(first_histogram, second_histogram)
    )


def label_score_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_label": {}, "overlap": {}}
    fields = {
        "gate_score": lambda row: float(row["gate_score"]),
        "p_help": lambda row: float(row["probabilities"]["helpful"]),
        "p_harm": lambda row: float(row["probabilities"]["harmful"]),
    }
    by_label_values: dict[str, dict[str, list[float]]] = {}
    for label in LABELS:
        selected = [row for row in rows if row["label"] == label]
        by_label_values[label] = {
            name: [getter(row) for row in selected] for name, getter in fields.items()
        }
        result["by_label"][label] = {
            name: _distribution(values) for name, values in by_label_values[label].items()
        }
    for name in fields:
        pairs = {}
        for left_index, left in enumerate(LABELS):
            for right in LABELS[left_index + 1 :]:
                pairs[f"{left}__{right}"] = _histogram_overlap(
                    by_label_values[left][name], by_label_values[right][name]
                )
        result["overlap"][name] = {
            "definition": "100-bin histogram overlap coefficient; 0=disjoint, 1=identical",
            "pairs": pairs,
        }
    return result


def assert_replay_consistency(
    replay_rows: Sequence[Mapping[str, Any]],
    saved_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    if len(replay_rows) != len(saved_rows):
        raise ValueError("Replay and saved prediction counts differ")
    fields = (
        "solver_error",
        "critic_fix_given_solver_error",
        "critic_harm_given_solver_correct",
        "helpful",
        "harmful",
    )
    maximum = 0.0
    for replay, saved in zip(replay_rows, saved_rows):
        if replay["identity"] != saved["identity"]:
            raise ValueError("Replay and saved identities differ")
        for field in fields:
            difference = abs(float(replay["probabilities"][field]) - float(saved["probabilities"][field]))
            maximum = max(maximum, difference)
            if difference > tolerance:
                raise ValueError(f"Replay probability mismatch for {field}: {difference}")
        if abs(float(replay["gate_score"]) - float(saved["gate_score"])) > tolerance:
            raise ValueError("Replay gate score mismatch")
        if bool(replay["critic_called"]) != bool(saved["critic_called"]):
            raise ValueError("Replay action mismatch")
    return {
        "consistent": True,
        "samples": len(replay_rows),
        "absolute_tolerance": tolerance,
        "maximum_probability_difference": maximum,
    }


def _tensor_outputs_to_rows(
    outputs: Mapping[str, torch.Tensor],
    labels: Sequence[str],
    identities: Sequence[str],
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for index, (label, identity) in enumerate(zip(labels, identities)):
        factorized = outputs["factorized_transition_probabilities"][index]
        auxiliary = outputs["transition_aux_probabilities"][index]
        score = float(outputs["gate_score"][index])
        rows.append(
            {
                "identity": str(identity),
                "label": label,
                "probabilities": {
                    "solver_error": float(outputs["p_solver_error"][index]),
                    "critic_fix_given_solver_error": float(outputs["p_critic_fix_given_error"][index]),
                    "critic_harm_given_solver_correct": float(outputs["p_critic_harm_given_correct"][index]),
                    "helpful": float(outputs["p_help"][index]),
                    "harmful": float(outputs["p_harm"][index]),
                    "factorized_four_class": {
                        label_name: float(factorized[class_index])
                        for class_index, label_name in enumerate(LABELS)
                    },
                    "auxiliary_four_class": {
                        label_name: float(auxiliary[class_index])
                        for class_index, label_name in enumerate(LABELS)
                    },
                },
                "gate_score": score,
                "critic_called": score >= threshold,
            }
        )
    return rows


def _saved_rows_for_seed(path: Path, *, validation: bool = False) -> dict[int, list[dict[str, Any]]]:
    rows = read_jsonl(path)
    expected_per_seed = 100 if validation else 1000
    result = {seed: [] for seed in ALL_SEEDS}
    for row in rows:
        seed = int(row["seed"])
        if seed not in result:
            raise ValueError("Prediction artifact contains an unexpected seed")
        copied = dict(row)
        copied["identity"] = str(row["question_id"] if validation else row["sample_id"])
        result[seed].append(copied)
    if any(len(seed_rows) != expected_per_seed for seed_rows in result.values()):
        raise ValueError("Prediction artifact is incomplete for one or more seeds")
    return result


def _history_snapshot(
    *,
    controller_dir: Path,
    training_path: Path,
    training_manifest_path: Path,
    validation_path: Path,
    final_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "training_examples": training_path,
        "training_manifest": training_manifest_path,
        "development_validation": validation_path,
        "final_test_manifest": final_manifest_path,
        "prompt_source": Path(__file__).with_name("logiqa_prompts.py"),
        "parser_source": Path(__file__).with_name("logiqa_pilot.py"),
        "controller_v3_source": Path(__file__).with_name("precritic_controller_v3.py"),
        "controller_v3_training_source": Path(__file__).with_name("precritic_controller_v3_training.py"),
    }
    for name in REQUIRED_CONTROLLER_FILES:
        paths[f"controller_v3/{name}"] = controller_dir / name
    artifact_root = controller_dir.parent
    for directory_name in (
        "precritic_controller_v1",
        "precritic_controller_v2_factorized",
        "precritic_controller_v3_smoke",
    ):
        directory = artifact_root / directory_name
        if directory.is_dir():
            for path in sorted(item for item in directory.rglob("*") if item.is_file()):
                paths[f"{directory_name}/{path.relative_to(directory)}"] = path
    snapshot = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Required frozen input is missing: {path}")
        snapshot[name] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return snapshot


def _training_dynamics(seed_payload: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    primary_final = None
    for seed_metrics in seed_payload["seeds"]:
        seed = int(seed_metrics["seed"])
        for fold in seed_metrics["oof"]["fold_training"]:
            runs.append({"seed": seed, "scope": f"fold_{fold['fold']}", **fold})
        final = {"seed": seed, "scope": "full_training_1000", **seed_metrics["final_training"]}
        runs.append(final)
        if seed == PRIMARY_SEED:
            primary_final = final
    if primary_final is None:
        raise ValueError("Primary final training dynamics are missing")
    fields = (
        "final_total_loss",
        "final_solver_error_loss",
        "final_critic_fix_loss",
        "final_critic_harm_loss",
        "final_auxiliary_loss",
        "maximum_preclip_gradient_norm",
    )
    distributions = {
        field: _distribution([float(run[field]) for run in runs]) for field in fields
    }
    in_sample = comparison["training_full_in_sample"]["head_metrics"]
    oof = comparison["training_oof"]["head_metrics"]
    gaps = {
        "solver_error_pr_auc": in_sample["solver_error"]["pr_auc"] - oof["solver_error"]["pr_auc"],
        "critic_fix_pr_auc": in_sample["critic_fix_given_solver_error"]["pr_auc"] - oof["critic_fix_given_solver_error"]["pr_auc"],
        "critic_harm_pr_auc": in_sample["critic_harm_given_solver_correct"]["pr_auc"] - oof["critic_harm_given_solver_correct"]["pr_auc"],
        "helpful_pr_auc": in_sample["helpful"]["pr_auc"] - oof["helpful"]["pr_auc"],
        "harmful_pr_auc": in_sample["harmful"]["pr_auc"] - oof["harmful"]["pr_auc"],
        "factorized_macro_f1": in_sample["factorized_four_class"]["macro_f1"] - oof["factorized_four_class"]["macro_f1"],
        "auxiliary_macro_f1": in_sample["auxiliary_four_class"]["macro_f1"] - oof["auxiliary_four_class"]["macro_f1"],
    }
    return {
        "models_summarized": len(runs),
        "primary_full_training_terminal": {
            field: primary_final[field] for field in fields
        },
        "all_fold_and_full_model_terminal_distributions": distributions,
        "in_sample_minus_oof_metric_gaps": gaps,
        "minority_memory_diagnostic": {
            "minority_heads": ["critic_fix", "critic_harm", "helpful", "harmful"],
            "observed_gap_only": True,
            "automatic_causal_conclusion": False,
            "note": "Large positive in-sample minus OOF gaps are compatible with memorization but do not establish its cause.",
        },
    }


def _label_total_variation(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    return 0.5 * sum(abs(first["rates"][label] - second["rates"][label]) for label in LABELS)


def _report(summary: Mapping[str, Any]) -> str:
    comparison = summary["prediction_regimes"]
    source = summary["source_and_fold_diagnostics"]["sources"]
    bootstrap = summary["bootstrap"]
    stability = summary["seed_stability"]
    lines = [
        "# Controller v3 Generalization Audit",
        "",
        "This is a pure-offline stability diagnostic. It is not an independent test and selects no new threshold, seed, budget, or architecture.",
        "",
        "## Regime comparison",
        "",
        "| Regime | solver-error PR-AUC | fix PR-AUC | harm PR-AUC | helpful PR-AUC | harmful PR-AUC | factorized macro-F1 | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("training_full_in_sample", "Training full model (in-sample diagnostic)"),
        ("training_oof", "Training OOF"),
        ("validation", "Validation 100"),
    ):
        metrics = comparison[key]["head_metrics"]
        lines.append(
            f"| {label} | {metrics['solver_error']['pr_auc']:.4f} | "
            f"{metrics['critic_fix_given_solver_error']['pr_auc']:.4f} | "
            f"{metrics['critic_harm_given_solver_correct']['pr_auc']:.4f} | "
            f"{metrics['helpful']['pr_auc']:.4f} | {metrics['harmful']['pr_auc']:.4f} | "
            f"{metrics['factorized_four_class']['macro_f1']:.4f} | "
            f"{metrics['factorized_four_class']['ece_10_bin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Training-full values are explicitly in-sample diagnostics and must not be read as generalization performance.",
            "",
            "## Source shift",
            "",
            f"- Collection 200: net={source['collection_200']['policy_at_frozen_primary_threshold']['net_benefit']}, call rate={source['collection_200']['policy_at_frozen_primary_threshold']['critic_call_rate']:.3f}.",
            f"- Collection 800: net={source['collection_800']['policy_at_frozen_primary_threshold']['net_benefit']}, call rate={source['collection_800']['policy_at_frozen_primary_threshold']['critic_call_rate']:.3f}.",
            f"- Label-distribution total variation (200 vs 800): {summary['distribution_shift']['collection_200_vs_800_label_total_variation']:.4f}.",
            "",
            "## Seed and threshold stability",
            "",
            f"- Pairwise gate-score Pearson mean: {stability['pairwise_summary']['pearson']['mean']:.4f}.",
            f"- Pairwise gate-score Spearman mean: {stability['pairwise_summary']['spearman']['mean']:.4f}.",
            f"- Threshold range: {stability['threshold_distribution']['quantiles']['p00']:.6f} to {stability['threshold_distribution']['quantiles']['p100']:.6f}.",
            f"- Bootstrap STOP fallback rate: {bootstrap['safe_fallback_to_stop_rate']:.3f}.",
            f"- Bootstrap OOB net-gain 95% interval: [{bootstrap['out_of_bag_threshold_transfer']['net_benefit_95_interval']['p2_5']:.2f}, {bootstrap['out_of_bag_threshold_transfer']['net_benefit_95_interval']['p97_5']:.2f}].",
            "",
            "## Diagnostic conclusion",
            "",
            summary["diagnostic_conclusion"]["text"],
            "",
            "No causal attribution is made from these diagnostics, and no new operating point is selected.",
            "",
            "## Boundaries",
            "",
            "- No LLM/backend/API calls; no embedding forward.",
            "- No training, backward pass, or optimizer initialization.",
            "- Final Test examples were not read; only the sealed manifest was verified.",
            "- Existing models, thresholds, prompts, parsers, cache, and historical outputs were unchanged.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_precritic_controller_v3_generalization_audit(
    *,
    controller_dir: str | Path = DEFAULT_CONTROLLER_DIR,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    validation_path: str | Path = DEFAULT_VALIDATION,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    controller_dir = Path(controller_dir)
    training_path = Path(training_path)
    training_manifest_path = Path(training_manifest_path)
    validation_path = Path(validation_path)
    final_test_manifest_path = Path(final_test_manifest_path)
    output_dir = Path(output_dir)
    if any((output_dir / name).exists() for name in AUDIT_FILES):
        raise FileExistsError("Generalization audit artifacts already exist; refusing to overwrite")

    before = _history_snapshot(
        controller_dir=controller_dir,
        training_path=training_path,
        training_manifest_path=training_manifest_path,
        validation_path=validation_path,
        final_manifest_path=final_test_manifest_path,
    )
    training_examples, training_manifest = load_training_examples(training_path, training_manifest_path)
    if _file_sha256(training_path) != TRAINING_SHA256:
        raise ValueError("Frozen Training 1000 SHA256 changed")
    final_guard = verify_sealed_final_manifest(final_test_manifest_path, training_manifest)
    training_ids = {example.sample_id for example in training_examples}
    validation_examples, _ = load_validation_examples(validation_path, training_manifest, training_ids)
    training_labels = [example.label for example in training_examples]
    validation_labels = [example.label for example in validation_examples]

    checkpoint_path = controller_dir / "primary_model.pt"
    cache_path = controller_dir / "feature_cache.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if checkpoint.get("feature_cache_sha256") != _file_sha256(cache_path):
        raise ValueError("Primary checkpoint no longer matches the frozen feature cache")
    training_features = _cached_split(cache["training"], len(training_examples))
    validation_features = _cached_split(cache["validation"], len(validation_examples))
    if tuple(example.sample_id for example in training_examples) != training_features.content_sha256:
        raise ValueError("Training cache identity changed")

    normalization = StateNormalization(
        mean=checkpoint["numeric_mean_first_8"].float().cpu(),
        std=checkpoint["numeric_std_first_8"].float().cpu(),
        fit_indices_sha256=checkpoint["development_threshold"].get("selection_source", "frozen"),
    )
    normalized_training = apply_state_normalization(training_features.structured_state, normalization)
    normalized_validation = apply_state_normalization(validation_features.structured_state, normalization)
    model = PreCriticControllerV3().cpu()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    threshold = float(checkpoint["development_threshold"]["threshold"])
    training_full_outputs = _predict(
        model, training_features, normalized_training, list(range(len(training_examples))), torch.device("cpu")
    )
    validation_outputs = _predict(
        model, validation_features, normalized_validation, list(range(len(validation_examples))), torch.device("cpu")
    )
    training_full_rows = _tensor_outputs_to_rows(
        training_full_outputs,
        training_labels,
        [example.sample_id for example in training_examples],
        threshold,
    )
    validation_replay_rows = _tensor_outputs_to_rows(
        validation_outputs,
        validation_labels,
        [str(example.question_id) for example in validation_examples],
        threshold,
    )

    oof_by_seed = _saved_rows_for_seed(controller_dir / "oof_predictions.jsonl")
    validation_by_seed = _saved_rows_for_seed(
        controller_dir / "validation_predictions.jsonl", validation=True
    )
    primary_oof = oof_by_seed[PRIMARY_SEED]
    primary_validation_saved = validation_by_seed[PRIMARY_SEED]
    expected_training_ids = [example.sample_id for example in training_examples]
    if [row["sample_id"] for row in primary_oof] != expected_training_ids:
        raise ValueError("Primary OOF order no longer matches Training 1000")
    expected_validation_ids = [str(example.question_id) for example in validation_examples]
    if [row["identity"] for row in primary_validation_saved] != expected_validation_ids:
        raise ValueError("Saved Validation order no longer matches Validation 100")
    replay_consistency = assert_replay_consistency(
        validation_replay_rows, primary_validation_saved
    )

    regimes = {
        "training_full_in_sample": {
            "in_sample_diagnostic": True,
            "generalization_performance": False,
            "head_metrics": diagnostic_metrics(training_full_rows),
            "policy_at_frozen_primary_threshold": label_policy_metrics(
                training_labels, [row["critic_called"] for row in training_full_rows]
            ),
        },
        "training_oof": {
            "in_sample_diagnostic": False,
            "generalization_diagnostic": True,
            "head_metrics": diagnostic_metrics(primary_oof),
            "policy_at_frozen_primary_threshold": label_policy_metrics(
                training_labels, [row["critic_called"] for row in primary_oof]
            ),
        },
        "validation": {
            "development_validation": True,
            "independent_final_test": False,
            "head_metrics": diagnostic_metrics(validation_replay_rows),
            "policy_at_frozen_primary_threshold": label_policy_metrics(
                validation_labels, [row["critic_called"] for row in validation_replay_rows]
            ),
        },
    }

    sources: dict[str, Any] = {}
    for source_name in ("collection_200", "collection_800"):
        selected = [row for row in primary_oof if row["source_dataset"] == source_name]
        if len(selected) != (200 if source_name == "collection_200" else 800):
            raise ValueError(f"Frozen {source_name} membership changed")
        sources[source_name] = _group_diagnostic(selected, threshold)
    folds = {}
    fold_members: set[str] = set()
    for fold in range(5):
        selected = [row for row in primary_oof if int(row["oof_fold"]) == fold]
        if not selected:
            raise ValueError("Every primary OOF fold must be non-empty")
        selected_ids = {str(row["sample_id"]) for row in selected}
        if len(selected_ids) != len(selected) or fold_members & selected_ids:
            raise ValueError("Primary OOF folds overlap or contain duplicate samples")
        fold_members.update(selected_ids)
        folds[str(fold)] = _group_diagnostic(selected, threshold)
    if fold_members != set(expected_training_ids):
        raise ValueError("Primary OOF folds do not cover Training 1000 exactly once")

    seed_diagnostics = seed_stability(oof_by_seed)
    bootstrap = stratified_bootstrap(
        [float(row["gate_score"]) for row in primary_oof],
        training_labels,
    )
    label_scores = {
        "training_full_in_sample": label_score_diagnostics(training_full_rows),
        "training_oof": label_score_diagnostics(primary_oof),
        "validation": label_score_diagnostics(validation_replay_rows),
    }
    seed_payload = json.loads((controller_dir / "seed_metrics.json").read_text(encoding="utf-8"))
    dynamics = _training_dynamics(seed_payload, regimes)

    validation_distribution = _label_distribution(validation_labels)
    shift = {
        "collection_200_vs_800_label_total_variation": _label_total_variation(
            sources["collection_200"]["label_distribution"],
            sources["collection_800"]["label_distribution"],
        ),
        "collection_200_vs_validation_label_total_variation": _label_total_variation(
            sources["collection_200"]["label_distribution"], validation_distribution
        ),
        "collection_800_vs_validation_label_total_variation": _label_total_variation(
            sources["collection_800"]["label_distribution"], validation_distribution
        ),
        "interpretation": "Descriptive distribution diagnostics only; they do not identify a causal source of failure.",
    }
    oof_help = regimes["training_oof"]["head_metrics"]["helpful"]["pr_auc"]
    validation_help = regimes["validation"]["head_metrics"]["helpful"]["pr_auc"]
    in_sample_help = regimes["training_full_in_sample"]["head_metrics"]["helpful"]["pr_auc"]
    conclusion = {
        "causal_attribution": False,
        "automatic_selection_made": False,
        "observations": {
            "in_sample_minus_oof_helpful_pr_auc": in_sample_help - oof_help,
            "oof_minus_validation_helpful_pr_auc": oof_help - validation_help,
            "bootstrap_stop_fallback_rate": bootstrap["safe_fallback_to_stop_rate"],
            "mean_pairwise_seed_spearman": seed_diagnostics["pairwise_summary"]["spearman"]["mean"],
        },
        "text": (
            "The audit separates in-sample/OOF gaps, cross-source differences, and seed/threshold instability, "
            "but these observational results do not support a unique causal attribution. Training-full metrics "
            "are diagnostic only; OOF and Validation remain the relevant generalization views."
        ),
    }

    cases = []
    full_by_id = {row["identity"]: row for row in training_full_rows}
    seed_by_id = {
        seed: {str(row["sample_id"]): row for row in rows}
        for seed, rows in oof_by_seed.items()
    }
    for row in primary_oof:
        sample_id = str(row["sample_id"])
        cases.append(
            {
                "audit_split": "training_1000",
                "in_sample_prediction_is_diagnostic_only": True,
                "sample_id": sample_id,
                "model_input_sha256": row["model_input_sha256"],
                "source_dataset": row["source_dataset"],
                "label": row["label"],
                "primary_oof_fold": row["oof_fold"],
                "primary_oof": {
                    "probabilities": row["probabilities"],
                    "gate_score": row["gate_score"],
                    "critic_called": row["critic_called"],
                },
                "primary_full_in_sample": {
                    "probabilities": full_by_id[sample_id]["probabilities"],
                    "gate_score": full_by_id[sample_id]["gate_score"],
                    "critic_called": full_by_id[sample_id]["critic_called"],
                },
                "seed_oof": {
                    str(seed): {
                        "gate_score": seed_by_id[seed][sample_id]["gate_score"],
                        "critic_called": seed_by_id[seed][sample_id]["critic_called"],
                    }
                    for seed in ALL_SEEDS
                },
                "contains_gold": False,
                "contains_raw_problem_or_agent_output": False,
            }
        )
    for replay, saved in zip(validation_replay_rows, primary_validation_saved):
        cases.append(
            {
                "audit_split": "validation_100",
                "development_validation": True,
                "independent_final_test": False,
                "question_id": replay["identity"],
                "model_input_sha256": saved["model_input_sha256"],
                "label": replay["label"],
                "primary_replay": {
                    "probabilities": replay["probabilities"],
                    "gate_score": replay["gate_score"],
                    "critic_called": replay["critic_called"],
                },
                "contains_gold": False,
                "contains_raw_problem_or_agent_output": False,
            }
        )

    after = _history_snapshot(
        controller_dir=controller_dir,
        training_path=training_path,
        training_manifest_path=training_manifest_path,
        validation_path=validation_path,
        final_manifest_path=final_test_manifest_path,
    )
    if before != after:
        raise RuntimeError("Frozen inputs or historical artifacts changed during audit")
    summary = {
        "controller_v3_generalization_audit": True,
        "offline_audit": True,
        "diagnostic_only": True,
        "deployable": False,
        "independent_test_performance": False,
        "new_threshold_selected": False,
        "operating_point_selected": False,
        "architecture_selected": False,
        "seed_selected": False,
        "budget_selected": False,
        "controller_retrained": False,
        "optimizer_initialized": False,
        "backward_calls": 0,
        "embedding_forward_calls": 0,
        "model_calls": 0,
        "llm_calls": 0,
        "backend_initialized": False,
        "final_test_evaluated": False,
        "final_test_examples_read": False,
        "feature_cache_modified": False,
        "checkpoint_modified": False,
        "prompt_modified": False,
        "parser_modified": False,
        "primary_seed": PRIMARY_SEED,
        "frozen_primary_threshold": threshold,
        "replay_device": "cpu",
        "replay_consistency": replay_consistency,
        "prediction_regimes": regimes,
        "source_and_fold_diagnostics": {"sources": sources, "folds": folds},
        "distribution_shift": shift,
        "seed_stability": seed_diagnostics,
        "bootstrap": bootstrap,
        "label_score_diagnostics": label_scores,
        "training_dynamics": dynamics,
        "diagnostic_conclusion": conclusion,
        "final_test_guard": {**final_guard, "manifest_only_read": True, "examples_read": False},
        "integrity": {
            "before": before,
            "after": after,
            "all_inputs_and_historical_artifacts_unchanged": True,
        },
        "output_contract": {
            "cases": len(cases),
            "cases_contain_gold": False,
            "cases_contain_raw_agent_outputs": False,
            "bootstrap_records_saved": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "audit_summary.json", summary)
    write_jsonl(output_dir / "audit_cases.jsonl", cases)
    _atomic_text(output_dir / "audit_report.md", _report(summary))
    return summary

