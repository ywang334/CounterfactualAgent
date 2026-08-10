from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_jsonl, write_jsonl
from .logiqa_audit import tolerant_final_answer
from .logiqa_pilot import ANSWER_LETTERS, extract_final_answer


CRITIC_PROTOCOL_CLASSES = (
    "canonical_keep",
    "contradictory_keep",
    "incomplete_revise",
    "noop_revise",
    "actionable_revise",
    "malformed",
)
REVISE_CLASSES = {
    "incomplete_revise",
    "noop_revise",
    "actionable_revise",
}
CONTRACT_INCONSISTENCY_CLASSES = {
    "contradictory_keep",
    "incomplete_revise",
    "noop_revise",
}
TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)


def _field_values(output: str, field: str) -> list[str]:
    return re.findall(
        rf"^{re.escape(field)}:[ \t]*(.*?)[ \t]*$",
        output,
        flags=re.MULTILINE,
    )


def classify_structured_critic(
    raw_output: str,
    solver_answer: str | None,
) -> dict[str, Any]:
    verdicts = _field_values(raw_output, "VERDICT")
    proposed_answers = _field_values(raw_output, "PROPOSED_ANSWER")
    verdict = verdicts[0] if len(verdicts) == 1 else None
    proposed = proposed_answers[0] if len(proposed_answers) == 1 else None
    if verdict not in {"KEEP", "REVISE"} or proposed not in {
        "NONE",
        *ANSWER_LETTERS,
    }:
        category = "malformed"
    elif verdict == "KEEP":
        if proposed == "NONE" or (
            solver_answer in ANSWER_LETTERS and proposed == solver_answer
        ):
            category = "canonical_keep"
        elif proposed in ANSWER_LETTERS:
            category = "contradictory_keep"
        else:
            category = "malformed"
    elif proposed == "NONE":
        category = "incomplete_revise"
    elif solver_answer in ANSWER_LETTERS and proposed == solver_answer:
        category = "noop_revise"
    else:
        category = "actionable_revise"
    return {
        "category": category,
        "verdict": verdict,
        "proposed_answer": proposed,
        "verdict_match_count": len(verdicts),
        "proposed_answer_match_count": len(proposed_answers),
        "contract_inconsistency": category in CONTRACT_INCONSISTENCY_CLASSES,
        "detected_solver_error": category in REVISE_CLASSES,
    }


def _id_key(question_id: str | int) -> str:
    return json.dumps([type(question_id).__name__, question_id], ensure_ascii=False)


def pair_unique_samples(
    minimal_rows: list[dict[str, Any]],
    structured_rows: list[dict[str, Any]],
    expected_samples: int = 50,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(minimal_rows) != expected_samples or len(structured_rows) != expected_samples:
        raise ValueError(
            f"Expected exactly {expected_samples} samples in each policy file; "
            f"found minimal_v1={len(minimal_rows)}, structured_v2={len(structured_rows)}"
        )

    def index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for position, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{label} row {position} is not an object")
            question_id = row.get("question_id")
            if isinstance(question_id, bool) or not isinstance(question_id, (str, int)):
                raise ValueError(f"{label} row {position} has invalid question_id")
            key = _id_key(question_id)
            if key in indexed:
                raise ValueError(f"{label} has duplicate question_id {question_id!r}")
            indexed[key] = row
        return indexed

    minimal_index = index(minimal_rows, "minimal_v1")
    structured_index = index(structured_rows, "structured_v2")
    if set(minimal_index) != set(structured_index):
        missing_v2 = [
            minimal_index[key]["question_id"]
            for key in minimal_index.keys() - structured_index.keys()
        ]
        missing_v1 = [
            structured_index[key]["question_id"]
            for key in structured_index.keys() - minimal_index.keys()
        ]
        raise ValueError(
            "Policy sample IDs are not one-to-one; "
            f"missing_in_structured_v2={missing_v2}, missing_in_minimal_v1={missing_v1}"
        )
    return [
        (minimal_row, structured_index[_id_key(minimal_row["question_id"])])
        for minimal_row in minimal_rows
    ]


def _usage(payload: Any, context: str) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} is missing usage")
    result: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{context} has invalid {field}")
        result[field] = value
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        raise ValueError(f"{context} has inconsistent token totals")
    return result


def _number(payload: Any, context: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)) or payload < 0:
        raise ValueError(f"{context} is missing or invalid")
    return float(payload)


def _cost(
    usage: Any,
    calls: Any,
    latency: Any,
    context: str,
) -> dict[str, Any]:
    calls_value = _number(calls, f"{context} calls")
    return {
        "usage": _usage(usage, f"{context} usage"),
        "calls": calls_value,
        "latency_seconds": _number(latency, f"{context} latency"),
    }


def _transition(
    solver_answer: str | None,
    full_answer: str | None,
    gold: str,
) -> str:
    solver_correct = solver_answer == gold
    full_correct = full_answer == gold
    if solver_correct and full_correct:
        return "correct_to_correct"
    if solver_correct:
        return "correct_to_wrong"
    if full_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def binary_confusion_matrix(
    actual_error: Iterable[bool],
    predicted_error: Iterable[bool],
    sample_ids: Iterable[str | int] | None = None,
) -> dict[str, Any]:
    actual = list(actual_error)
    predicted = list(predicted_error)
    ids = list(range(len(actual))) if sample_ids is None else list(sample_ids)
    if len(actual) != len(predicted) or len(actual) != len(ids):
        raise ValueError("Confusion-matrix inputs must have equal lengths")
    buckets: dict[str, list[str | int]] = {
        "true_positive": [],
        "false_positive": [],
        "true_negative": [],
        "false_negative": [],
    }
    for question_id, truth, detection in zip(ids, actual, predicted):
        if truth and detection:
            buckets["true_positive"].append(question_id)
        elif detection:
            buckets["false_positive"].append(question_id)
        elif truth:
            buckets["false_negative"].append(question_id)
        else:
            buckets["true_negative"].append(question_id)
    tp = len(buckets["true_positive"])
    fp = len(buckets["false_positive"])
    tn = len(buckets["true_negative"])
    fn = len(buckets["false_negative"])
    precision = _safe_ratio(tp, tp + fp) or 0.0
    recall = _safe_ratio(tp, tp + fn) or 0.0
    specificity = _safe_ratio(tn, tn + fp) or 0.0
    f1 = _safe_ratio(2 * precision * recall, precision + recall) or 0.0
    return {
        "detection_rule": (
            "Predicted error means the raw Critic verdict is a recognizable REVISE "
            "(incomplete, no-op, or actionable); malformed output counts as no detection."
        ),
        "matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
        "sample_ids": buckets,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "actual_positive": tp + fn,
        "actual_negative": tn + fp,
        "predicted_positive": tp + fp,
        "predicted_negative": tn + fn,
    }


def compare_id_sets(
    minimal_ids: Iterable[str | int],
    structured_ids: Iterable[str | int],
) -> dict[str, Any]:
    minimal = set(minimal_ids)
    structured = set(structured_ids)
    intersection = minimal & structured
    union = minimal | structured
    sort_key = lambda value: (str(type(value)), str(value))
    return {
        "minimal_v1": sorted(minimal, key=sort_key),
        "structured_v2": sorted(structured, key=sort_key),
        "intersection": sorted(intersection, key=sort_key),
        "union": sorted(union, key=sort_key),
        "minimal_v1_only": sorted(minimal - structured, key=sort_key),
        "structured_v2_only": sorted(structured - minimal, key=sort_key),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": len(intersection) / len(union) if union else 1.0,
    }


def _aggregate_selected_costs(costs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(costs)
    total_usage = {
        field: sum(cost["usage"][field] for cost in costs)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    total_calls = sum(cost["calls"] for cost in costs)
    total_latency = sum(cost["latency_seconds"] for cost in costs)
    return {
        "samples": count,
        "total_usage": total_usage,
        "average_usage": {
            field: total_usage[field] / count for field in total_usage
        },
        "total_calls": total_calls,
        "average_calls": total_calls / count,
        "total_latency_seconds": total_latency,
        "average_latency_seconds": total_latency / count,
    }


def minimum_cost_posthoc_oracle(
    cases: list[dict[str, Any]],
    policy_key: str,
) -> dict[str, Any]:
    selected_costs: list[dict[str, Any]] = []
    selections: dict[str, str] = {}
    full_usage_ids: list[str | int] = []
    correct_ids: list[str | int] = []

    def cost_key(cost: dict[str, Any], strategy: str) -> tuple[float, float, float, int]:
        return (
            float(cost["usage"]["total_tokens"]),
            float(cost["calls"]),
            float(cost["latency_seconds"]),
            0 if strategy == "solver_only" else 1,
        )

    for case in cases:
        candidates = [
            (
                "solver_only",
                case["solver"]["tolerant_answer"],
                case["solver"]["cost"],
            ),
            (
                "full",
                case[policy_key]["tolerant_answer"],
                case[policy_key]["cost"],
            ),
        ]
        correct_candidates = [
            candidate for candidate in candidates if candidate[1] == case["gold"]
        ]
        eligible = correct_candidates or candidates
        strategy, answer, cost = min(
            eligible,
            key=lambda candidate: cost_key(candidate[2], candidate[0]),
        )
        id_key = _id_key(case["question_id"])
        selections[id_key] = strategy
        selected_costs.append(cost)
        if strategy == "full":
            full_usage_ids.append(case["question_id"])
        if answer == case["gold"]:
            correct_ids.append(case["question_id"])
    return {
        "posthoc_oracle": True,
        "deployable": False,
        "warning": (
            "Uses gold outcomes after generation to choose the lowest-cost correct policy; "
            "it is not deployable."
        ),
        "cost_order": "total_tokens, then calls, then latency, then Solver tie-break",
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(cases),
        "correct_ids": correct_ids,
        "full_usage_count": len(full_usage_ids),
        "full_usage_rate": len(full_usage_ids) / len(cases),
        "full_usage_ids": full_usage_ids,
        "recorded_costs": _aggregate_selected_costs(selected_costs),
        "selection_by_id_key": selections,
    }


def _policy_metrics(
    cases: list[dict[str, Any]],
    policy_key: str,
) -> dict[str, Any]:
    count = len(cases)
    solver_correct_ids = [
        case["question_id"]
        for case in cases
        if case["solver"]["tolerant_answer"] == case["gold"]
    ]
    full_correct_ids = [
        case["question_id"]
        for case in cases
        if case[policy_key]["tolerant_answer"] == case["gold"]
    ]
    transition_ids = {name: [] for name in TRANSITIONS}
    for case in cases:
        transition_ids[case[policy_key]["transition"]].append(case["question_id"])
    corrected_ids = transition_ids["wrong_to_correct"]
    degraded_ids = transition_ids["correct_to_wrong"]
    unchanged_ids = (
        transition_ids["correct_to_correct"] + transition_ids["wrong_to_wrong"]
    )
    solver_error_count = count - len(solver_correct_ids)
    full_cost = _aggregate_selected_costs(
        [case[policy_key]["cost"] for case in cases]
    )
    oracle = minimum_cost_posthoc_oracle(cases, policy_key)
    return {
        "answer_basis": "tolerant FINAL_ANSWER parser",
        "samples": count,
        "solver": {
            "correct": len(solver_correct_ids),
            "accuracy": len(solver_correct_ids) / count,
            "errors": solver_error_count,
            "correct_ids": solver_correct_ids,
        },
        "full": {
            "correct": len(full_correct_ids),
            "accuracy": len(full_correct_ids) / count,
            "correct_ids": full_correct_ids,
        },
        "transitions": {
            name: {"count": len(transition_ids[name]), "sample_ids": transition_ids[name]}
            for name in TRANSITIONS
        },
        "corrected": len(corrected_ids),
        "corrected_ids": corrected_ids,
        "degraded": len(degraded_ids),
        "degraded_ids": degraded_ids,
        "unchanged": len(unchanged_ids),
        "unchanged_ids": unchanged_ids,
        "normalized": {
            "corrected_per_n": len(corrected_ids) / count,
            "degraded_per_n": len(degraded_ids) / count,
            "corrected_per_solver_errors": _safe_ratio(
                len(corrected_ids),
                solver_error_count,
            ),
            "degraded_per_solver_correct": _safe_ratio(
                len(degraded_ids),
                len(solver_correct_ids),
            ),
            "corrected_degraded_benefit_risk_ratio": _safe_ratio(
                len(corrected_ids),
                len(degraded_ids),
            ),
        },
        "full_recorded_cost": full_cost,
        "minimum_cost_posthoc_oracle": oracle,
    }


def _build_case(
    minimal: dict[str, Any],
    structured: dict[str, Any],
) -> dict[str, Any]:
    question_id = minimal["question_id"]
    if structured.get("question_id") != question_id:
        raise ValueError("Internal ID pairing error")
    gold = minimal.get("gold")
    if gold not in ANSWER_LETTERS or structured.get("gold") != gold:
        raise ValueError(f"Gold mismatch for question {question_id!r}")
    if (
        structured.get("prompt_version") != "structured_v2"
        or structured.get("prompt_development") is not True
        or structured.get("solver_reused") is not True
        or structured.get("solver_called") is not False
    ):
        raise ValueError(
            f"Question {question_id!r} is not a saved structured_v2 replay result"
        )
    minimal_raw = minimal.get("raw_outputs")
    structured_solver = structured.get("solver")
    structured_refiner = structured.get("refiner")
    structured_critic = structured.get("critic")
    if (
        not isinstance(minimal_raw, dict)
        or not isinstance(structured_solver, dict)
        or not isinstance(structured_refiner, dict)
        or not isinstance(structured_critic, dict)
    ):
        raise ValueError(f"Missing saved outputs for question {question_id!r}")
    solver_output = minimal_raw.get("solver")
    if not isinstance(solver_output, str) or structured_solver.get("raw_output") != solver_output:
        raise ValueError(f"Saved Solver state mismatch for question {question_id!r}")
    minimal_full_output = minimal_raw.get("refiner")
    structured_full_output = structured_refiner.get("raw_output")
    critic_output = structured_critic.get("raw_output")
    if not all(
        isinstance(output, str)
        for output in (minimal_full_output, structured_full_output, critic_output)
    ):
        raise ValueError(f"Missing raw continuation output for question {question_id!r}")
    solver_strict = extract_final_answer(solver_output)
    solver_tolerant = tolerant_final_answer(solver_output).answer
    minimal_strict = extract_final_answer(minimal_full_output)
    minimal_tolerant = tolerant_final_answer(minimal_full_output).answer
    structured_strict = extract_final_answer(structured_full_output)
    structured_tolerant = tolerant_final_answer(structured_full_output).answer
    saved_solver_tolerant = structured.get("tolerant", {}).get("solver_answer")
    saved_full_tolerant = structured.get("tolerant", {}).get("full_answer")
    if saved_solver_tolerant != solver_tolerant or saved_full_tolerant != structured_tolerant:
        raise ValueError(f"Historical tolerant result mismatch for question {question_id!r}")
    minimal_usage = minimal.get("usage")
    minimal_latency = minimal.get("latency_seconds")
    structured_usage = structured.get("usage")
    structured_calls = structured.get("calls")
    structured_latency = structured.get("latency_seconds")
    if not all(
        isinstance(payload, dict)
        for payload in (
            minimal_usage,
            minimal_latency,
            structured_usage,
            structured_calls,
            structured_latency,
        )
    ):
        raise ValueError(f"Missing recorded cost for question {question_id!r}")
    minimal_calls = minimal_usage.get("calls")
    if not isinstance(minimal_calls, dict):
        raise ValueError(f"Missing minimal_v1 calls for question {question_id!r}")
    solver_cost = _cost(
        minimal_usage.get("solver_only"),
        minimal_calls.get("solver_only"),
        minimal_latency.get("solver_only"),
        f"Solver {question_id!r}",
    )
    structured_solver_cost = _cost(
        structured_usage.get("solver_reused"),
        structured_calls.get("complete_workflow_equivalent", 0) - 2,
        structured_latency.get("solver_recorded"),
        f"structured_v2 saved Solver {question_id!r}",
    )
    if solver_cost != structured_solver_cost:
        raise ValueError(f"Saved Solver cost mismatch for question {question_id!r}")
    minimal_cost = _cost(
        minimal_usage.get("solver_critic_refiner"),
        minimal_calls.get("solver_critic_refiner"),
        minimal_latency.get("solver_critic_refiner"),
        f"minimal_v1 Full {question_id!r}",
    )
    structured_cost = _cost(
        structured_usage.get("complete_v2"),
        structured_calls.get("complete_workflow_equivalent"),
        structured_latency.get("complete_v2"),
        f"structured_v2 Full {question_id!r}",
    )
    critic_classification = classify_structured_critic(
        critic_output,
        solver_tolerant,
    )
    historical_parse_failure = structured.get("critic_parse_failure")
    if not isinstance(historical_parse_failure, bool):
        raise ValueError(
            f"Missing historical critic_parse_failure for question {question_id!r}"
        )
    minimal_transition = _transition(solver_tolerant, minimal_tolerant, gold)
    structured_transition = _transition(solver_tolerant, structured_tolerant, gold)
    return {
        "question_id": question_id,
        "gold": gold,
        "solver": {
            "raw_output": solver_output,
            "strict_answer": solver_strict,
            "tolerant_answer": solver_tolerant,
            "correct": solver_tolerant == gold,
            "cost": solver_cost,
        },
        "minimal_v1": {
            "critic_raw_output": minimal_raw.get("critic"),
            "refiner_raw_output": minimal_full_output,
            "strict_answer": minimal_strict,
            "tolerant_answer": minimal_tolerant,
            "correct": minimal_tolerant == gold,
            "transition": minimal_transition,
            "cost": minimal_cost,
        },
        "structured_v2": {
            "problem": structured.get("problem"),
            "problem_and_choices": structured.get("problem_and_choices"),
            "critic_raw_output": critic_output,
            "critic_protocol_classification": critic_classification,
            "historical_critic_parse_failure": historical_parse_failure,
            "contract_inconsistency": critic_classification[
                "contract_inconsistency"
            ],
            "refiner_raw_output": structured_full_output,
            "strict_answer": structured_strict,
            "tolerant_answer": structured_tolerant,
            "correct": structured_tolerant == gold,
            "transition": structured_transition,
            "cost": structured_cost,
        },
        "label_change": {
            "transition_changed": minimal_transition != structured_transition,
            "minimal_v1_corrected": minimal_transition == "wrong_to_correct",
            "structured_v2_corrected": structured_transition == "wrong_to_correct",
            "minimal_v1_degraded": minimal_transition == "correct_to_wrong",
            "structured_v2_degraded": structured_transition == "correct_to_wrong",
        },
    }


def _critic_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = [
        case["structured_v2"]["critic_protocol_classification"]["category"]
        for case in cases
    ]
    counts = Counter(classifications)
    classification_summary = {
        category: {
            "count": counts.get(category, 0),
            "sample_ids": [
                case["question_id"]
                for case in cases
                if case["structured_v2"]["critic_protocol_classification"][
                    "category"
                ]
                == category
            ],
        }
        for category in CRITIC_PROTOCOL_CLASSES
    }
    historical_parse_failure_ids = [
        case["question_id"]
        for case in cases
        if case["structured_v2"]["historical_critic_parse_failure"]
    ]
    contract_inconsistency_ids = [
        case["question_id"]
        for case in cases
        if case["structured_v2"]["contract_inconsistency"]
    ]
    canonical_historical_failure_ids = [
        case["question_id"]
        for case in cases
        if case["structured_v2"]["historical_critic_parse_failure"]
        and case["structured_v2"]["critic_protocol_classification"]["category"]
        == "canonical_keep"
    ]
    confusion = binary_confusion_matrix(
        [
            case["solver"]["tolerant_answer"] != case["gold"]
            for case in cases
        ],
        [
            case["structured_v2"]["critic_protocol_classification"][
                "detected_solver_error"
            ]
            for case in cases
        ],
        [case["question_id"] for case in cases],
    )
    return {
        "classification": classification_summary,
        "historical_parse_failure": {
            "count": len(historical_parse_failure_ids),
            "sample_ids": historical_parse_failure_ids,
        },
        "contract_inconsistency": {
            "definition": (
                "contradictory_keep, incomplete_revise, or noop_revise; "
                "canonical KEEP+Solver-answer remains canonical even if the historical "
                "strict parser marked it as a failure."
            ),
            "count": len(contract_inconsistency_ids),
            "sample_ids": contract_inconsistency_ids,
        },
        "canonical_keep_with_historical_parse_failure": {
            "count": len(canonical_historical_failure_ids),
            "sample_ids": canonical_historical_failure_ids,
        },
        "error_detection": confusion,
    }


def _build_summary(
    cases: list[dict[str, Any]],
    minimal_path: Path,
    structured_path: Path,
) -> dict[str, Any]:
    minimal_metrics = _policy_metrics(cases, "minimal_v1")
    structured_metrics = _policy_metrics(cases, "structured_v2")
    corrected_stability = compare_id_sets(
        minimal_metrics["corrected_ids"],
        structured_metrics["corrected_ids"],
    )
    degraded_stability = compare_id_sets(
        minimal_metrics["degraded_ids"],
        structured_metrics["degraded_ids"],
    )
    changed_transition_ids = [
        case["question_id"]
        for case in cases
        if case["label_change"]["transition_changed"]
    ]
    return {
        "offline_audit": True,
        "model_backend_initialized": False,
        "model_calls": 0,
        "controller_training": False,
        "samples": len(cases),
        "answer_basis": "tolerant explicit FINAL_ANSWER parser",
        "sources": {
            "minimal_v1": str(minimal_path.resolve()),
            "structured_v2": str(structured_path.resolve()),
        },
        "policies": {
            "minimal_v1": minimal_metrics,
            "structured_v2": structured_metrics,
        },
        "structured_v2_critic": _critic_summary(cases),
        "label_stability": {
            "corrected": corrected_stability,
            "degraded": degraded_stability,
            "transition_changed_count": len(changed_transition_ids),
            "transition_changed_ids": changed_transition_ids,
        },
        "policy_selection": {
            "selected": None,
            "automatic_selection": False,
            "note": (
                "This audit compares saved development-set continuations only and does "
                "not select, freeze, or modify a prompt policy."
            ),
        },
    }


def _format_ids(values: list[str | int]) -> str:
    return ", ".join(map(str, values)) if values else "None"


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _build_report(summary: dict[str, Any]) -> str:
    minimal = summary["policies"]["minimal_v1"]
    structured = summary["policies"]["structured_v2"]
    lines = [
        "# Prompt Policy Stability Audit",
        "",
        "Pure offline audit: no model backend was initialized, no model was called, "
        "and no controller was trained.",
        "",
        "The 50 samples are prompt-development data, not a final test set. "
        "No prompt policy is automatically selected or modified.",
        "",
        "## Policy comparison",
        "",
        "| Policy | Solver acc. | Full acc. | Corrected | Degraded | Corrected/N | Degraded/N | Corrected/Solver errors | Degraded/Solver correct | Benefit-risk | Avg tokens | Avg calls | Avg latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, metrics in (
        ("minimal_v1", minimal),
        ("structured_v2", structured),
    ):
        normalized = metrics["normalized"]
        cost = metrics["full_recorded_cost"]
        ratio = normalized["corrected_degraded_benefit_risk_ratio"]
        lines.append(
            f"| {label} | {metrics['solver']['accuracy']:.4f} | "
            f"{metrics['full']['accuracy']:.4f} | {metrics['corrected']} | "
            f"{metrics['degraded']} | {normalized['corrected_per_n']:.4f} | "
            f"{normalized['degraded_per_n']:.4f} | "
            f"{_format_metric(normalized['corrected_per_solver_errors'])} | "
            f"{_format_metric(normalized['degraded_per_solver_correct'])} | "
            f"{_format_metric(ratio)} | {cost['average_usage']['total_tokens']:.2f} | "
            f"{cost['average_calls']:.2f} | "
            f"{cost['average_latency_seconds']:.4f} |"
        )
    lines.extend(["", "## Tolerant transition matrices", ""])
    for label, metrics in (
        ("minimal_v1", minimal),
        ("structured_v2", structured),
    ):
        counts = metrics["transitions"]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- correct→correct: {counts['correct_to_correct']['count']}",
                f"- correct→wrong: {counts['correct_to_wrong']['count']}",
                f"- wrong→correct: {counts['wrong_to_correct']['count']}",
                f"- wrong→wrong: {counts['wrong_to_wrong']['count']}",
                "",
            ]
        )
    critic = summary["structured_v2_critic"]
    lines.extend(
        [
            "## structured_v2 Critic protocol",
            "",
            "| Classification | Count | IDs |",
            "|---|---:|---|",
        ]
    )
    for category in CRITIC_PROTOCOL_CLASSES:
        payload = critic["classification"][category]
        lines.append(
            f"| {category} | {payload['count']} | "
            f"{_format_ids(payload['sample_ids'])} |"
        )
    confusion = critic["error_detection"]
    matrix = confusion["matrix"]
    lines.extend(
        [
            "",
            f"- Historical critic_parse_failure: "
            f"{critic['historical_parse_failure']['count']}",
            f"- contract_inconsistency: "
            f"{critic['contract_inconsistency']['count']}",
            f"- Canonical KEEP recovered from historical parse failures: "
            f"{critic['canonical_keep_with_historical_parse_failure']['count']}",
            "",
            "## Critic error detection",
            "",
            f"- TP={matrix['true_positive']}, FP={matrix['false_positive']}, "
            f"TN={matrix['true_negative']}, FN={matrix['false_negative']}",
            f"- Precision={confusion['precision']:.4f}",
            f"- Recall={confusion['recall']:.4f}",
            f"- F1={confusion['f1']:.4f}",
            f"- Specificity={confusion['specificity']:.4f}",
            "",
            "## Corrected/degraded label stability",
            "",
        ]
    )
    for label in ("corrected", "degraded"):
        comparison = summary["label_stability"][label]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Intersection: {_format_ids(comparison['intersection'])}",
                f"- Union: {_format_ids(comparison['union'])}",
                f"- minimal_v1 only: {_format_ids(comparison['minimal_v1_only'])}",
                f"- structured_v2 only: {_format_ids(comparison['structured_v2_only'])}",
                f"- Jaccard: {comparison['jaccard']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Minimum-cost posthoc oracles",
            "",
            "**Both oracles use gold after generation and are deployable=false.**",
            "",
            "| Policy | Oracle acc. | Full usage | Avg tokens | Avg calls | Avg latency (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, metrics in (
        ("minimal_v1", minimal),
        ("structured_v2", structured),
    ):
        oracle = metrics["minimum_cost_posthoc_oracle"]
        costs = oracle["recorded_costs"]
        lines.append(
            f"| {label} | {oracle['accuracy']:.4f} | "
            f"{oracle['full_usage_rate']:.4f} | "
            f"{costs['average_usage']['total_tokens']:.2f} | "
            f"{costs['average_calls']:.2f} | "
            f"{costs['average_latency_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Stop condition",
            "",
            "No policy was selected, frozen, or modified. No further inference or "
            "prompt adjustment was performed.",
            "",
        ]
    )
    return "\n".join(lines)


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


def run_prompt_stability_audit(
    minimal_predictions: str | Path,
    structured_predictions: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    minimal_path = Path(minimal_predictions)
    structured_path = Path(structured_predictions)
    output = Path(output_dir)
    for source in (minimal_path, structured_path):
        if not source.is_file():
            raise FileNotFoundError(f"Prompt policy predictions not found: {source}")
    if output.resolve() in {minimal_path.parent.resolve(), structured_path.parent.resolve()}:
        raise ValueError("Audit output directory cannot overwrite a historical policy directory")
    minimal_rows = read_jsonl(minimal_path)
    structured_rows = read_jsonl(structured_path)
    pairs = pair_unique_samples(minimal_rows, structured_rows, expected_samples=50)
    cases = [_build_case(minimal, structured) for minimal, structured in pairs]
    minimal_oracle = minimum_cost_posthoc_oracle(cases, "minimal_v1")
    structured_oracle = minimum_cost_posthoc_oracle(cases, "structured_v2")
    for case in cases:
        key = _id_key(case["question_id"])
        case["minimal_v1"]["posthoc_oracle_selected"] = minimal_oracle[
            "selection_by_id_key"
        ][key]
        case["structured_v2"]["posthoc_oracle_selected"] = structured_oracle[
            "selection_by_id_key"
        ][key]
    summary = _build_summary(cases, minimal_path, structured_path)
    target_files = (
        output / "summary.json",
        output / "cases.jsonl",
        output / "report.md",
    )
    if any(path.exists() for path in target_files):
        raise FileExistsError(
            "Prompt stability audit artifacts already exist; refusing to overwrite them"
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "summary.json", summary)
    write_jsonl(output / "cases.jsonl", cases)
    _write_text_atomic(output / "report.md", _build_report(summary))
    return summary
