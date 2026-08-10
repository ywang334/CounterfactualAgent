from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_jsonl, write_jsonl
from .logiqa_pilot import ANSWER_LETTERS


STRATEGIES = (
    "STOP",
    "MINIMAL_V1_ABLATION",
    "CRITIC_ONLY",
    "CONDITIONAL_REFINE",
    "ALWAYS_FULL",
)
TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)
COLLECTION_STAGE_COST_REASON = (
    "The collection stores aggregate SHORT/FULL continuation cost but not separate "
    "structured Critic and Refiner usage. Stage-dependent token and latency costs "
    "are unavailable and are not estimated."
)


def _required_mapping(payload: Any, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    return payload


def _answer(value: Any, context: str, *, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    if value not in ANSWER_LETTERS:
        raise ValueError(f"{context} must be A-D" + (" or null" if allow_none else ""))
    return str(value)


def _question_id(row: dict[str, Any], context: str) -> str | int:
    value = row.get("question_id")
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{context} has invalid question_id")
    return value


def _id_key(value: str | int) -> str:
    return json.dumps([type(value).__name__, value], ensure_ascii=False)


def _check_unique_rows(
    rows: list[dict[str, Any]],
    label: str,
    expected_samples: int,
) -> None:
    if len(rows) != expected_samples:
        raise ValueError(
            f"Expected exactly {expected_samples} {label} rows; found {len(rows)}"
        )
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {index} is not an object")
        question_id = _question_id(row, f"{label} row {index}")
        key = _id_key(question_id)
        if key in seen:
            raise ValueError(f"{label} has duplicate question_id {question_id!r}")
        seen.add(key)


def _transition(solver_answer: str | None, answer: str | None, gold: str) -> str:
    solver_correct = solver_answer == gold
    policy_correct = answer == gold
    if solver_correct and policy_correct:
        return "correct_to_correct"
    if solver_correct:
        return "correct_to_wrong"
    if policy_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _stage_cost(payload: Any, context: str) -> dict[str, Any]:
    stage = _required_mapping(payload, context)
    usage = _required_mapping(stage.get("usage"), f"{context} usage")
    result: dict[str, Any] = {"available": True}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{context} has invalid {field}")
        result[field] = value
    if result["prompt_tokens"] + result["completion_tokens"] != result["total_tokens"]:
        raise ValueError(f"{context} has inconsistent token totals")
    calls = stage.get("calls")
    latency = stage.get("latency_seconds")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise ValueError(f"{context} has invalid calls")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ValueError(f"{context} has invalid latency")
    result["calls"] = calls
    result["latency_seconds"] = float(latency)
    return result


def _combine_costs(*costs: dict[str, Any]) -> dict[str, Any]:
    if not all(cost.get("available") is True for cost in costs):
        raise ValueError("Cannot combine unavailable stage costs")
    result: dict[str, Any] = {"available": True}
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "calls",
        "latency_seconds",
    ):
        result[field] = sum(cost[field] for cost in costs)
    if result["prompt_tokens"] + result["completion_tokens"] != result["total_tokens"]:
        raise ValueError("Combined cost has inconsistent token totals")
    return result


def _unavailable_collection_cost(calls: int) -> dict[str, Any]:
    return {
        "available": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "calls": calls,
        "latency_seconds": None,
        "unavailable_reason": COLLECTION_STAGE_COST_REASON,
    }


def _critic_fields(
    critic: dict[str, Any],
    parse_failure: Any,
    context: str,
) -> dict[str, Any]:
    effective_verdict = critic.get("effective_verdict")
    if effective_verdict not in {"KEEP", "REVISE"}:
        raise ValueError(f"{context} has invalid effective_verdict")
    effective_proposed = _answer(
        critic.get("effective_proposed_answer"),
        f"{context} effective_proposed_answer",
    )
    if effective_verdict == "KEEP" and effective_proposed is not None:
        raise ValueError(f"{context} KEEP must not have an effective proposed answer")
    if effective_verdict == "REVISE" and effective_proposed is None:
        raise ValueError(f"{context} REVISE requires an effective proposed answer")
    if not isinstance(parse_failure, bool):
        raise ValueError(f"{context} parse_failure must be boolean")
    parsed_verdict = critic.get("parsed_verdict")
    if parsed_verdict not in {None, "KEEP", "REVISE"}:
        raise ValueError(f"{context} has invalid parsed_verdict")
    proposed_field = critic.get("proposed_answer")
    proposed_answer = (
        None
        if proposed_field == "NONE"
        else _answer(proposed_field, f"{context} proposed_answer")
    )
    return {
        "raw_output": critic.get("raw_output"),
        "parsed_verdict": parsed_verdict,
        "proposed_answer_field": proposed_field,
        "proposed_answer": proposed_answer,
        "effective_verdict": effective_verdict,
        "effective_proposed_answer": effective_proposed,
        "parse_failure": parse_failure,
    }


def _answers_and_effects(
    *,
    solver_answer: str | None,
    minimal_answer: str | None,
    full_answer: str | None,
    critic: dict[str, Any],
    gold: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    effective_verdict = critic["effective_verdict"]
    critic_only = (
        solver_answer
        if effective_verdict == "KEEP"
        else critic["effective_proposed_answer"]
    )
    conditional = solver_answer if effective_verdict == "KEEP" else full_answer
    strategy_answers = {
        "STOP": solver_answer,
        "MINIMAL_V1_ABLATION": minimal_answer,
        "CRITIC_ONLY": critic_only,
        "CONDITIONAL_REFINE": conditional,
        "ALWAYS_FULL": full_answer,
    }
    policies = {
        strategy: {
            "answer": answer,
            "correct": answer == gold,
            "transition": _transition(solver_answer, answer, gold),
        }
        for strategy, answer in strategy_answers.items()
    }
    effective_proposed = critic["effective_proposed_answer"]
    raw_proposed = critic["proposed_answer"]
    agreement = (
        effective_proposed == full_answer
        if effective_verdict == "REVISE" and effective_proposed in ANSWER_LETTERS
        else None
    )
    raw_agreement = (
        raw_proposed == full_answer if raw_proposed in ANSWER_LETTERS else None
    )
    critic_effect = {
        **critic,
        "proposed_refiner_agreement": agreement,
        "raw_explicit_proposed_refiner_agreement": raw_agreement,
    }
    changed = full_answer != solver_answer
    beneficial = solver_answer != gold and full_answer == gold
    harmful = solver_answer == gold and full_answer != gold
    refiner_effect = {
        "final_answer": full_answer,
        "changed_solver_answer": changed,
        "changed_on_effective_keep": changed and effective_verdict == "KEEP",
        "changed_on_effective_revise": changed and effective_verdict == "REVISE",
        "beneficial": beneficial,
        "harmful": harmful,
        "neutral_change": changed and not beneficial and not harmful,
    }
    return policies, critic_effect, refiner_effect


def build_collection_case(row: dict[str, Any]) -> dict[str, Any]:
    question_id = _question_id(row, "collection row")
    if row.get("mock_only") is not False:
        raise ValueError(f"Collection question {question_id!r} is not a real rollout")
    gold = _answer(row.get("gold"), f"Collection question {question_id!r} gold", allow_none=False)
    solver = _required_mapping(row.get("solver"), f"Collection question {question_id!r} solver")
    actions = _required_mapping(row.get("actions"), f"Collection question {question_id!r} actions")
    stop = _required_mapping(actions.get("STOP"), f"Collection question {question_id!r} STOP")
    short = _required_mapping(actions.get("SHORT"), f"Collection question {question_id!r} SHORT")
    full = _required_mapping(actions.get("FULL"), f"Collection question {question_id!r} FULL")
    solver_answer = _answer(
        _required_mapping(solver.get("tolerant"), "collection solver tolerant").get("answer"),
        f"Collection question {question_id!r} Solver answer",
    )
    stop_answer = _answer(
        _required_mapping(stop.get("tolerant"), "collection STOP tolerant").get("answer"),
        f"Collection question {question_id!r} STOP answer",
    )
    if stop_answer != solver_answer:
        raise ValueError(f"Collection question {question_id!r} STOP changed Solver answer")
    minimal_answer = _answer(
        _required_mapping(short.get("tolerant"), "collection SHORT tolerant").get("answer"),
        f"Collection question {question_id!r} SHORT answer",
    )
    full_answer = _answer(
        _required_mapping(full.get("tolerant"), "collection FULL tolerant").get("answer"),
        f"Collection question {question_id!r} FULL answer",
    )
    protocol = _required_mapping(
        full.get("critic_protocol"),
        f"Collection question {question_id!r} critic protocol",
    )
    raw_outputs = _required_mapping(
        full.get("raw_outputs"), f"Collection question {question_id!r} FULL outputs"
    )
    critic = _critic_fields(
        {**protocol, "raw_output": raw_outputs.get("critic")},
        protocol.get("parse_failure"),
        f"Collection question {question_id!r} Critic",
    )
    policies, critic_effect, refiner_effect = _answers_and_effects(
        solver_answer=solver_answer,
        minimal_answer=minimal_answer,
        full_answer=full_answer,
        critic=critic,
        gold=gold,
    )
    solver_cost = _required_mapping(solver.get("cost"), "collection solver cost")
    short_increment = _required_mapping(short.get("incremental_cost"), "collection SHORT cost")
    full_increment = _required_mapping(full.get("incremental_cost"), "collection FULL cost")
    if (
        solver_cost.get("calls") != 1
        or short_increment.get("calls") != 2
        or full_increment.get("calls") != 2
        or row.get("actual_calls") != 5
    ):
        raise ValueError(f"Collection question {question_id!r} does not have the fixed five-call trace")
    conditional_calls = 3 if critic["effective_verdict"] == "REVISE" else 2
    costs = {
        "STOP": _unavailable_collection_cost(1),
        "MINIMAL_V1_ABLATION": _unavailable_collection_cost(3),
        "CRITIC_ONLY": _unavailable_collection_cost(2),
        "CONDITIONAL_REFINE": _unavailable_collection_cost(conditional_calls),
        "ALWAYS_FULL": _unavailable_collection_cost(3),
    }
    return {
        "offline_audit": True,
        "deployable": False,
        "dataset": "collection_200",
        "question_id": question_id,
        "sample_id": row.get("sample_id"),
        "gold": gold,
        "solver": {
            "answer": solver_answer,
            "correct": solver_answer == gold,
            "raw_output": solver.get("raw_output"),
        },
        "policies": policies,
        "critic": critic_effect,
        "refiner": {
            **refiner_effect,
            "raw_output": raw_outputs.get("refiner"),
        },
        "source_outputs": {
            "minimal_v1_refiner": _required_mapping(
                short.get("raw_outputs"), "collection SHORT outputs"
            ).get("refiner"),
            "structured_v2_critic": raw_outputs.get("critic"),
            "structured_v2_refiner": raw_outputs.get("refiner"),
        },
        "strategy_costs": costs,
    }


def build_validation_case(row: dict[str, Any]) -> dict[str, Any]:
    question_id = _question_id(row, "validation row")
    if (
        row.get("mock_only") is not False
        or row.get("policy_selection_validation") is not True
        or row.get("solver_called_once") is not True
        or row.get("same_solver_state_for_both_policies") is not True
    ):
        raise ValueError(f"Validation question {question_id!r} is not a paired real trace")
    gold = _answer(row.get("gold"), f"Validation question {question_id!r} gold", allow_none=False)
    solver = _required_mapping(row.get("solver"), f"Validation question {question_id!r} solver")
    minimal = _required_mapping(row.get("minimal_v1"), f"Validation question {question_id!r} minimal_v1")
    full = _required_mapping(row.get("structured_v2"), f"Validation question {question_id!r} structured_v2")
    solver_answer = _answer(
        _required_mapping(solver.get("tolerant"), "validation solver tolerant").get("answer"),
        f"Validation question {question_id!r} Solver answer",
    )
    minimal_answer = _answer(
        minimal.get("tolerant_answer"), f"Validation question {question_id!r} minimal answer"
    )
    full_answer = _answer(
        full.get("tolerant_answer"), f"Validation question {question_id!r} full answer"
    )
    critic_payload = _required_mapping(
        full.get("critic"), f"Validation question {question_id!r} Critic"
    )
    critic = _critic_fields(
        critic_payload,
        full.get("critic_parse_failure"),
        f"Validation question {question_id!r} Critic",
    )
    policies, critic_effect, refiner_effect = _answers_and_effects(
        solver_answer=solver_answer,
        minimal_answer=minimal_answer,
        full_answer=full_answer,
        critic=critic,
        gold=gold,
    )
    solver_cost = _stage_cost(solver, f"Validation question {question_id!r} Solver")
    minimal_critic = _stage_cost(
        minimal.get("critic"), f"Validation question {question_id!r} minimal Critic"
    )
    minimal_refiner = _stage_cost(
        minimal.get("refiner"), f"Validation question {question_id!r} minimal Refiner"
    )
    structured_critic = _stage_cost(
        critic_payload, f"Validation question {question_id!r} structured Critic"
    )
    refiner_payload = _required_mapping(
        full.get("refiner"), f"Validation question {question_id!r} structured Refiner"
    )
    structured_refiner = _stage_cost(
        refiner_payload, f"Validation question {question_id!r} structured Refiner"
    )
    conditional_stages = [solver_cost, structured_critic]
    if critic["effective_verdict"] == "REVISE":
        conditional_stages.append(structured_refiner)
    costs = {
        "STOP": solver_cost,
        "MINIMAL_V1_ABLATION": _combine_costs(
            solver_cost, minimal_critic, minimal_refiner
        ),
        "CRITIC_ONLY": _combine_costs(solver_cost, structured_critic),
        "CONDITIONAL_REFINE": _combine_costs(*conditional_stages),
        "ALWAYS_FULL": _combine_costs(
            solver_cost, structured_critic, structured_refiner
        ),
    }
    return {
        "offline_audit": True,
        "deployable": False,
        "dataset": "validation_100",
        "question_id": question_id,
        "gold": gold,
        "solver": {
            "answer": solver_answer,
            "correct": solver_answer == gold,
            "raw_output": solver.get("raw_output"),
        },
        "policies": policies,
        "critic": critic_effect,
        "refiner": {
            **refiner_effect,
            "raw_output": refiner_payload.get("raw_output"),
        },
        "source_outputs": {
            "minimal_v1_refiner": _required_mapping(
                minimal.get("refiner"), "validation minimal refiner"
            ).get("raw_output"),
            "structured_v2_critic": critic_payload.get("raw_output"),
            "structured_v2_refiner": refiner_payload.get("raw_output"),
        },
        "strategy_costs": costs,
    }


def strategy_metrics(cases: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    transitions: dict[str, list[str | int]] = {name: [] for name in TRANSITIONS}
    correct_ids: list[str | int] = []
    for case in cases:
        payload = case["policies"][strategy]
        transitions[payload["transition"]].append(case["question_id"])
        if payload["correct"]:
            correct_ids.append(case["question_id"])
    count = len(cases)
    corrected_ids = transitions["wrong_to_correct"]
    degraded_ids = transitions["correct_to_wrong"]
    return {
        "samples": count,
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / count if count else 0.0,
        "correct_ids": correct_ids,
        "corrected": len(corrected_ids),
        "corrected_ids": corrected_ids,
        "degraded": len(degraded_ids),
        "degraded_ids": degraded_ids,
        "net_benefit": len(corrected_ids) - len(degraded_ids),
        "transitions": {
            name: {"count": len(ids), "sample_ids": ids}
            for name, ids in transitions.items()
        },
    }


def _aggregate_costs(
    cases: list[dict[str, Any]], strategy: str, *, stage_usage_available: bool
) -> dict[str, Any]:
    costs = [case["strategy_costs"][strategy] for case in cases]
    total_calls = sum(cost["calls"] for cost in costs)
    call_metrics = {
        "available": True,
        "total": total_calls,
        "mean": total_calls / len(costs) if costs else 0.0,
    }
    if not stage_usage_available:
        return {
            "usage": {
                "available": False,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "reason": COLLECTION_STAGE_COST_REASON,
            },
            "calls": call_metrics,
            "latency": {
                "available": False,
                "total_seconds": None,
                "mean_seconds": None,
                "reason": COLLECTION_STAGE_COST_REASON,
            },
            "estimated": False,
        }
    if not all(cost.get("available") is True for cost in costs):
        raise ValueError("Validation stage usage unexpectedly unavailable")
    usage_total = {
        field: sum(cost[field] for cost in costs)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if usage_total["prompt_tokens"] + usage_total["completion_tokens"] != usage_total["total_tokens"]:
        raise ValueError("Aggregated validation cost has inconsistent token totals")
    latency = sum(cost["latency_seconds"] for cost in costs)
    count = len(costs)
    return {
        "usage": {
            "available": True,
            "total": usage_total,
            "mean": {
                field: usage_total[field] / count if count else 0.0
                for field in usage_total
            },
        },
        "calls": call_metrics,
        "latency": {
            "available": True,
            "total_seconds": latency,
            "mean_seconds": latency / count if count else 0.0,
        },
        "estimated": False,
    }


def _critic_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    effective_counts = Counter(case["critic"]["effective_verdict"] for case in cases)
    parsed_counts = Counter(case["critic"]["parsed_verdict"] for case in cases)
    parse_failure_ids = [
        case["question_id"] for case in cases if case["critic"]["parse_failure"]
    ]
    effective_revise = [
        case for case in cases if case["critic"]["effective_verdict"] == "REVISE"
    ]
    effective_agree = [
        case for case in effective_revise
        if case["critic"]["proposed_refiner_agreement"] is True
    ]
    explicit_raw = [
        case for case in cases if case["critic"]["proposed_answer"] in ANSWER_LETTERS
    ]
    explicit_agree = [
        case for case in explicit_raw
        if case["critic"]["raw_explicit_proposed_refiner_agreement"] is True
    ]
    return {
        "verdict_basis": (
            "Historical effective_verdict; malformed or contract-invalid Critic output "
            "uses its saved safe KEEP fallback."
        ),
        "effective_verdict": {
            verdict: {
                "count": effective_counts.get(verdict, 0),
                "sample_ids": [
                    case["question_id"]
                    for case in cases
                    if case["critic"]["effective_verdict"] == verdict
                ],
            }
            for verdict in ("KEEP", "REVISE")
        },
        "parsed_raw_verdict": {
            "KEEP": parsed_counts.get("KEEP", 0),
            "REVISE": parsed_counts.get("REVISE", 0),
            "MISSING": parsed_counts.get(None, 0),
        },
        "parse_failure": {
            "count": len(parse_failure_ids),
            "sample_ids": parse_failure_ids,
        },
        "proposed_answer_refiner_agreement": {
            "basis": "effective REVISE with a valid effective proposed A-D answer",
            "agree": len(effective_agree),
            "eligible": len(effective_revise),
            "rate": len(effective_agree) / len(effective_revise)
            if effective_revise
            else None,
            "agree_ids": [case["question_id"] for case in effective_agree],
        },
        "raw_explicit_proposal_refiner_agreement": {
            "basis": "any parsed raw A-D proposal, including contract-invalid output",
            "agree": len(explicit_agree),
            "eligible": len(explicit_raw),
            "rate": len(explicit_agree) / len(explicit_raw) if explicit_raw else None,
        },
    }


def _refiner_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [case for case in cases if case["refiner"]["changed_solver_answer"]]
    beneficial = [case for case in cases if case["refiner"]["beneficial"]]
    harmful = [case for case in cases if case["refiner"]["harmful"]]
    neutral = [case for case in cases if case["refiner"]["neutral_change"]]
    keep_changes = [case for case in cases if case["refiner"]["changed_on_effective_keep"]]
    revise_changes = [case for case in cases if case["refiner"]["changed_on_effective_revise"]]
    conditional = strategy_metrics(cases, "CONDITIONAL_REFINE")
    return {
        "actual_change_vs_solver": {
            "count": len(changed),
            "sample_ids": [case["question_id"] for case in changed],
        },
        "changed_after_effective_keep": {
            "count": len(keep_changes),
            "sample_ids": [case["question_id"] for case in keep_changes],
        },
        "changed_after_effective_revise": {
            "count": len(revise_changes),
            "sample_ids": [case["question_id"] for case in revise_changes],
        },
        "beneficial_changes": {
            "count": len(beneficial),
            "sample_ids": [case["question_id"] for case in beneficial],
        },
        "harmful_changes": {
            "count": len(harmful),
            "sample_ids": [case["question_id"] for case in harmful],
        },
        "neutral_wrong_to_wrong_changes": {
            "count": len(neutral),
            "sample_ids": [case["question_id"] for case in neutral],
        },
        "always_full_net_benefit": len(beneficial) - len(harmful),
        "conditional_refine_net_benefit": conditional["net_benefit"],
    }


def _dataset_summary(
    cases: list[dict[str, Any]], *, stage_usage_available: bool
) -> dict[str, Any]:
    return {
        "samples": len(cases),
        "answer_basis": "saved tolerant explicit FINAL_ANSWER result",
        "strategies": {
            strategy: {
                **strategy_metrics(cases, strategy),
                "cost": _aggregate_costs(
                    cases, strategy, stage_usage_available=stage_usage_available
                ),
            }
            for strategy in STRATEGIES
        },
        "critic": _critic_summary(cases),
        "refiner": _refiner_summary(cases),
        "cost_scope": (
            "Exact saved per-stage service usage, calls, and latency."
            if stage_usage_available
            else COLLECTION_STAGE_COST_REASON
        ),
    }


def _format_rate(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.4f}"


def _build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Critic Gating Offline Audit",
        "",
        "This is a saved-output counterfactual audit. `offline_audit=true`, "
        "`deployable=false`; no backend was initialized, no model was called, and no "
        "controller was trained.",
        "",
        "Gating uses the historical `effective_verdict`: KEEP preserves the Solver; "
        "REVISE uses the valid effective proposed answer or the saved structured_v2 "
        "Refiner output, depending on policy.",
        "",
    ]
    for dataset_key, title in (
        ("validation_100", "Validation 100"),
        ("collection_200", "Collection 200"),
    ):
        dataset = summary["datasets"][dataset_key]
        lines.extend(
            [
                f"## {title}",
                "",
                "| Strategy | Accuracy | Corrected | Degraded | c→c | c→w | w→c | w→w | Calls | Mean total tokens | Mean latency (s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for strategy in STRATEGIES:
            metric = dataset["strategies"][strategy]
            transitions = metric["transitions"]
            cost = metric["cost"]
            mean_tokens = (
                f"{cost['usage']['mean']['total_tokens']:.2f}"
                if cost["usage"]["available"]
                else "unavailable"
            )
            mean_latency = (
                f"{cost['latency']['mean_seconds']:.4f}"
                if cost["latency"]["available"]
                else "unavailable"
            )
            lines.append(
                f"| {strategy} | {metric['accuracy']:.4f} | {metric['corrected']} | "
                f"{metric['degraded']} | "
                f"{transitions['correct_to_correct']['count']} | "
                f"{transitions['correct_to_wrong']['count']} | "
                f"{transitions['wrong_to_correct']['count']} | "
                f"{transitions['wrong_to_wrong']['count']} | "
                f"{cost['calls']['total']} | {mean_tokens} | {mean_latency} |"
            )
        critic = dataset["critic"]
        agreement = critic["proposed_answer_refiner_agreement"]
        refiner = dataset["refiner"]
        lines.extend(
            [
                "",
                f"- Effective Critic KEEP/REVISE: "
                f"{critic['effective_verdict']['KEEP']['count']}/"
                f"{critic['effective_verdict']['REVISE']['count']}",
                f"- Critic parse failures: {critic['parse_failure']['count']}",
                f"- Proposed/Refiner agreement: {agreement['agree']}/"
                f"{agreement['eligible']} ({_format_rate(agreement['rate'])})",
                f"- Refiner changed Solver answer: "
                f"{refiner['actual_change_vs_solver']['count']}",
                f"- Refiner beneficial/harmful changes: "
                f"{refiner['beneficial_changes']['count']}/"
                f"{refiner['harmful_changes']['count']}",
                f"- ALWAYS_FULL net benefit: {refiner['always_full_net_benefit']}",
                f"- CONDITIONAL_REFINE net benefit: "
                f"{refiner['conditional_refine_net_benefit']}",
                "",
            ]
        )
        if dataset_key == "collection_200":
            lines.extend(
                [
                    "Collection stage token/latency cost is **unavailable** because "
                    "Critic and Refiner usage was not saved separately. No estimate or "
                    "subtraction was used.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "These policies are posthoc simulations over already generated outputs. "
            "They are not deployable policy results and do not select or modify a "
            "production prompt or ActionController.",
            "",
        ]
    )
    return "\n".join(lines)


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


def run_critic_gating_audit(
    collection_rollouts: str | Path,
    validation_predictions: str | Path,
    output_dir: str | Path,
    *,
    expected_collection_samples: int = 200,
    expected_validation_samples: int = 100,
) -> dict[str, Any]:
    collection_path = Path(collection_rollouts)
    validation_path = Path(validation_predictions)
    output = Path(output_dir)
    for source in (collection_path, validation_path):
        if not source.is_file():
            raise FileNotFoundError(f"Critic gating audit input not found: {source}")
    if output.resolve() in {collection_path.parent.resolve(), validation_path.parent.resolve()}:
        raise ValueError("Audit output cannot overwrite either historical input directory")
    targets = (output / "summary.json", output / "cases.jsonl", output / "report.md")
    if any(target.exists() for target in targets):
        raise FileExistsError("Critic gating audit artifacts already exist; refusing to overwrite")

    collection_rows = read_jsonl(collection_path)
    validation_rows = read_jsonl(validation_path)
    _check_unique_rows(collection_rows, "collection", expected_collection_samples)
    _check_unique_rows(validation_rows, "validation", expected_validation_samples)
    collection_cases = [build_collection_case(row) for row in collection_rows]
    validation_cases = [build_validation_case(row) for row in validation_rows]
    cases = collection_cases + validation_cases
    summary = {
        "offline_audit": True,
        "deployable": False,
        "model_backend_initialized": False,
        "model_calls": 0,
        "controller_training": False,
        "historical_results_modified": False,
        "strategy_definitions": {
            "STOP": "Solver Only",
            "MINIMAL_V1_ABLATION": "Saved minimal_v1 SHORT Critic→Refiner",
            "CRITIC_ONLY": "KEEP uses Solver; REVISE uses effective Critic proposed answer",
            "CONDITIONAL_REFINE": "KEEP stops after Critic; REVISE uses saved Refiner answer",
            "ALWAYS_FULL": "Saved structured_v2 Critic→Refiner",
        },
        "sources": {
            "collection_200": str(collection_path.resolve()),
            "validation_100": str(validation_path.resolve()),
        },
        "datasets": {
            "collection_200": _dataset_summary(
                collection_cases, stage_usage_available=False
            ),
            "validation_100": _dataset_summary(
                validation_cases, stage_usage_available=True
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "summary.json", summary)
    write_jsonl(output / "cases.jsonl", cases)
    _write_text_atomic(output / "report.md", _build_report(summary))
    return summary
