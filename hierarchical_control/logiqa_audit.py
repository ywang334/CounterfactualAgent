from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_jsonl, write_jsonl


ANSWER_MARKER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])FINAL_ANSWER:[ \t]*([A-D])(?=$|[^A-Za-z0-9_/])"
)
TRANSITIONS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True)
class TolerantParseResult:
    answer: str | None
    match_count: int
    conflict: bool
    matches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matches"] = list(self.matches)
        return payload


def tolerant_final_answer(output: str) -> TolerantParseResult:
    """Parse explicit answer markers only, choosing the final marker."""
    matches = tuple(match.group(1) for match in ANSWER_MARKER_PATTERN.finditer(output))
    return TolerantParseResult(
        answer=matches[-1] if matches else None,
        match_count=len(matches),
        conflict=len(set(matches)) > 1,
        matches=matches,
    )


def _transition(solver_correct: bool, full_correct: bool) -> str:
    if solver_correct and full_correct:
        return "correct_to_correct"
    if solver_correct:
        return "correct_to_wrong"
    if full_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def exact_mcnemar(correct_to_wrong: int, wrong_to_correct: int) -> dict[str, Any]:
    """Two-sided exact McNemar test via the conditional Binomial(n, 0.5)."""
    discordant = correct_to_wrong + wrong_to_correct
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(correct_to_wrong, wrong_to_correct)
        tail_probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail_probability)
    return {
        "method": "two-sided exact McNemar (conditional binomial)",
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def _require_prediction(row: dict[str, Any], index: int) -> None:
    required = {
        "question_id",
        "gold",
        "solver_answer",
        "refiner_answer",
        "solver_correct",
        "refiner_correct",
        "solver_parse_failure",
        "refiner_parse_failure",
        "raw_outputs",
        "usage",
        "latency_seconds",
        "mock_only",
    }
    missing = required - set(row)
    if missing:
        raise ValueError(f"Prediction row {index} is missing fields: {sorted(missing)}")
    gold = row["gold"]
    if not isinstance(gold, str) or len(gold) != 1 or gold not in "ABCD":
        raise ValueError(f"Prediction row {index} has invalid gold: {gold!r}")
    for role, answer_key, correct_key, failure_key in (
        ("solver", "solver_answer", "solver_correct", "solver_parse_failure"),
        ("refiner", "refiner_answer", "refiner_correct", "refiner_parse_failure"),
    ):
        answer = row[answer_key]
        if answer is not None and (
            not isinstance(answer, str) or len(answer) != 1 or answer not in "ABCD"
        ):
            raise ValueError(f"Prediction row {index} has invalid {role} strict answer: {answer!r}")
        if bool(row[correct_key]) != (answer == gold):
            raise ValueError(f"Prediction row {index} has inconsistent {role} correctness")
        if bool(row[failure_key]) != (answer is None):
            raise ValueError(f"Prediction row {index} has inconsistent {role} parse failure")
    raw = row["raw_outputs"]
    if not isinstance(raw, dict) or not all(isinstance(raw.get(name), str) for name in ("solver", "critic", "refiner")):
        raise ValueError(f"Prediction row {index} has invalid raw_outputs")
    for strategy in ("solver_only", "solver_critic_refiner"):
        usage = row["usage"].get(strategy)
        if not isinstance(usage, dict) or not all(isinstance(usage.get(name), int) for name in USAGE_FIELDS):
            raise ValueError(f"Prediction row {index} has invalid {strategy} usage")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError(f"Prediction row {index} has inconsistent {strategy} token totals")
        if not isinstance(row["usage"].get("calls", {}).get(strategy), int):
            raise ValueError(f"Prediction row {index} has invalid {strategy} call count")
        if not isinstance(row["latency_seconds"].get(strategy), (int, float)):
            raise ValueError(f"Prediction row {index} has invalid {strategy} latency")


def _strategy_cost(row: dict[str, Any], strategy: str) -> dict[str, Any]:
    return {
        "usage": {name: int(row["usage"][strategy][name]) for name in USAGE_FIELDS},
        "calls": int(row["usage"]["calls"][strategy]),
        "latency_seconds": float(row["latency_seconds"][strategy]),
    }


def aggregate_costs(selections: Iterable[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    totals = {name: 0 for name in USAGE_FIELDS}
    total_calls = 0
    total_latency = 0.0
    count = 0
    for row, strategy in selections:
        cost = _strategy_cost(row, strategy)
        for name in USAGE_FIELDS:
            totals[name] += cost["usage"][name]
        total_calls += cost["calls"]
        total_latency += cost["latency_seconds"]
        count += 1
    divisor = max(count, 1)
    return {
        "samples": count,
        "total_usage": totals,
        "average_usage": {name: totals[name] / divisor for name in USAGE_FIELDS},
        "total_calls": total_calls,
        "average_calls": total_calls / divisor,
        "total_latency_seconds": total_latency,
        "average_latency_seconds": total_latency / divisor,
    }


def _mode_summary(cases: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    sample_count = len(cases)
    solver_correct_ids: list[Any] = []
    full_correct_ids: list[Any] = []
    solver_failure_ids: list[Any] = []
    full_failure_ids: list[Any] = []
    transition_ids = {name: [] for name in TRANSITIONS}
    for case in cases:
        question_id = case["question_id"]
        entry = case[mode]
        solver_answer = entry["solver_answer"]
        full_answer = entry["full_answer"]
        solver_correct = solver_answer == case["gold"]
        full_correct = full_answer == case["gold"]
        if solver_correct:
            solver_correct_ids.append(question_id)
        if full_correct:
            full_correct_ids.append(question_id)
        if solver_answer is None:
            solver_failure_ids.append(question_id)
        if full_answer is None:
            full_failure_ids.append(question_id)
        transition_ids[_transition(solver_correct, full_correct)].append(question_id)

    def strategy_metrics(correct_ids: list[Any], failure_ids: list[Any]) -> dict[str, Any]:
        parsed = sample_count - len(failure_ids)
        return {
            "correct": len(correct_ids),
            "accuracy": len(correct_ids) / sample_count,
            "parse_failures": len(failure_ids),
            "parse_failure_ids": failure_ids,
            "format_compliant": parsed,
            "format_compliance_rate": parsed / sample_count,
        }

    corrected_ids = transition_ids["wrong_to_correct"]
    degraded_ids = transition_ids["correct_to_wrong"]
    unchanged_ids = transition_ids["correct_to_correct"] + transition_ids["wrong_to_wrong"]
    return {
        "solver_only": strategy_metrics(solver_correct_ids, solver_failure_ids),
        "full": strategy_metrics(full_correct_ids, full_failure_ids),
        "transitions": {
            name: {"count": len(ids), "sample_ids": ids} for name, ids in transition_ids.items()
        },
        "corrected_ids": corrected_ids,
        "degraded_ids": degraded_ids,
        "unchanged_ids": unchanged_ids,
    }


def audit_logiqa_predictions(
    predictions: list[dict[str, Any]],
    source_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not predictions:
        raise ValueError("Predictions file is empty")
    seen_ids: set[Any] = set()
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(predictions, 1):
        _require_prediction(row, index)
        question_id = row["question_id"]
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question_id in predictions: {question_id!r}")
        seen_ids.add(question_id)
        solver_tolerant = tolerant_final_answer(row["raw_outputs"]["solver"])
        full_tolerant = tolerant_final_answer(row["raw_outputs"]["refiner"])
        solver_recovered = row["solver_answer"] is None and solver_tolerant.answer is not None
        full_recovered = row["refiner_answer"] is None and full_tolerant.answer is not None
        strict_transition = _transition(bool(row["solver_correct"]), bool(row["refiner_correct"]))
        tolerant_solver_correct = solver_tolerant.answer == row["gold"]
        tolerant_full_correct = full_tolerant.answer == row["gold"]
        tolerant_transition = _transition(tolerant_solver_correct, tolerant_full_correct)
        oracle_strategy = (
            "solver_only" if tolerant_solver_correct or not tolerant_full_correct else "solver_critic_refiner"
        )
        cases.append(
            {
                "question_id": question_id,
                "gold": row["gold"],
                "strict": {
                    "solver_answer": row["solver_answer"],
                    "full_answer": row["refiner_answer"],
                    "solver_correct": bool(row["solver_correct"]),
                    "full_correct": bool(row["refiner_correct"]),
                    "transition": strict_transition,
                },
                "tolerant": {
                    "solver_answer": solver_tolerant.answer,
                    "full_answer": full_tolerant.answer,
                    "solver_correct": tolerant_solver_correct,
                    "full_correct": tolerant_full_correct,
                    "solver_parse": solver_tolerant.to_dict(),
                    "full_parse": full_tolerant.to_dict(),
                    "transition": tolerant_transition,
                },
                "format_recovered": {
                    "solver_only": solver_recovered,
                    "full": full_recovered,
                    "any": solver_recovered or full_recovered,
                },
                "costs": {
                    "solver_only": _strategy_cost(row, "solver_only"),
                    "full": _strategy_cost(row, "solver_critic_refiner"),
                },
                "posthoc_oracle": {
                    "posthoc_oracle": True,
                    "selected_strategy": oracle_strategy,
                    "correct": tolerant_solver_correct or tolerant_full_correct,
                },
                "raw_outputs": dict(row["raw_outputs"]),
                "source_mock_only": bool(row["mock_only"]),
            }
        )

    strict = _mode_summary(cases, "strict")
    tolerant = _mode_summary(cases, "tolerant")
    solver_recovered_ids = [
        case["question_id"] for case in cases if case["format_recovered"]["solver_only"]
    ]
    full_recovered_ids = [case["question_id"] for case in cases if case["format_recovered"]["full"]]
    any_recovered_ids = [case["question_id"] for case in cases if case["format_recovered"]["any"]]
    solver_conflict_ids = [
        case["question_id"] for case in cases if case["tolerant"]["solver_parse"]["conflict"]
    ]
    full_conflict_ids = [
        case["question_id"] for case in cases if case["tolerant"]["full_parse"]["conflict"]
    ]
    oracle_selections = [
        (row, case["posthoc_oracle"]["selected_strategy"])
        for row, case in zip(predictions, cases)
    ]
    oracle_full_ids = [
        case["question_id"]
        for case in cases
        if case["posthoc_oracle"]["selected_strategy"] == "solver_critic_refiner"
    ]
    oracle_correct_ids = [case["question_id"] for case in cases if case["posthoc_oracle"]["correct"]]
    summary = {
        "offline_audit": True,
        "source_predictions": str(Path(source_path).resolve()),
        "samples": len(cases),
        "source_mock_only": any(case["source_mock_only"] for case in cases),
        "strict": strict,
        "tolerant": tolerant,
        "format_recovery": {
            "solver_only_ids": solver_recovered_ids,
            "full_ids": full_recovered_ids,
            "any_ids": any_recovered_ids,
            "solver_conflict_ids": solver_conflict_ids,
            "full_conflict_ids": full_conflict_ids,
        },
        "mcnemar_exact_tolerant": exact_mcnemar(
            tolerant["transitions"]["correct_to_wrong"]["count"],
            tolerant["transitions"]["wrong_to_correct"]["count"],
        ),
        "recorded_costs": {
            "solver_only": aggregate_costs((row, "solver_only") for row in predictions),
            "full": aggregate_costs((row, "solver_critic_refiner") for row in predictions),
        },
        "posthoc_oracle": {
            "posthoc_oracle": True,
            "deployable": False,
            "warning": "Uses gold outcomes after inference and is not a deployable policy.",
            "correct": len(oracle_correct_ids),
            "accuracy": len(oracle_correct_ids) / len(cases),
            "correct_ids": oracle_correct_ids,
            "full_usage_count": len(oracle_full_ids),
            "full_usage_rate": len(oracle_full_ids) / len(cases),
            "full_usage_ids": oracle_full_ids,
            "recorded_costs": aggregate_costs(oracle_selections),
        },
    }
    return summary, cases


def _format_ids(ids: list[Any]) -> str:
    return ", ".join(str(item) for item in ids) if ids else "None"


def build_audit_report(summary: dict[str, Any]) -> str:
    strict = summary["strict"]
    tolerant = summary["tolerant"]
    mcnemar = summary["mcnemar_exact_tolerant"]
    oracle = summary["posthoc_oracle"]
    costs = summary["recorded_costs"]
    lines = [
        "# LogiQA Pilot Offline Audit",
        "",
        f"Source: `{summary['source_predictions']}`",
        f"Samples: {summary['samples']}",
        "",
        "No LLM or semantic-error classifier was used. Strict results are preserved; tolerant results only use explicit `FINAL_ANSWER:` markers.",
        "",
        "## Accuracy and formatting",
        "",
        "| Parser | Strategy | Accuracy | Parse failures | Format compliance |",
        "|---|---|---:|---:|---:|",
    ]
    for name, mode in (("Strict", strict), ("Tolerant", tolerant)):
        for strategy, label in (("solver_only", "Solver Only"), ("full", "Full")):
            metric = mode[strategy]
            lines.append(
                f"| {name} | {label} | {metric['accuracy']:.4f} | {metric['parse_failures']} | {metric['format_compliance_rate']:.4f} |"
            )
    lines.extend(["", "## Tolerant transition matrix", ""])
    for name in TRANSITIONS:
        entry = tolerant["transitions"][name]
        lines.append(f"- `{name}`: {entry['count']} — {_format_ids(entry['sample_ids'])}")
    recovery = summary["format_recovery"]
    lines.extend(
        [
            "",
            "## Format recovery",
            "",
            f"- Solver Only: {_format_ids(recovery['solver_only_ids'])}",
            f"- Full: {_format_ids(recovery['full_ids'])}",
            f"- Conflicting Solver markers: {_format_ids(recovery['solver_conflict_ids'])}",
            f"- Conflicting Full markers: {_format_ids(recovery['full_conflict_ids'])}",
            "",
            "## Exact McNemar test (tolerant)",
            "",
            f"- Correct→wrong: {mcnemar['correct_to_wrong']}",
            f"- Wrong→correct: {mcnemar['wrong_to_correct']}",
            f"- Two-sided exact p-value: {mcnemar['p_value']:.8f}",
            "",
            "## Recorded strategy costs",
            "",
            "| Strategy | Avg total tokens | Avg calls | Avg latency (s) |",
            "|---|---:|---:|---:|",
            f"| Solver Only | {costs['solver_only']['average_usage']['total_tokens']:.2f} | {costs['solver_only']['average_calls']:.2f} | {costs['solver_only']['average_latency_seconds']:.4f} |",
            f"| Full | {costs['full']['average_usage']['total_tokens']:.2f} | {costs['full']['average_calls']:.2f} | {costs['full']['average_latency_seconds']:.4f} |",
            "",
            "## Post-hoc Oracle",
            "",
            "**Warning: `posthoc_oracle=true`. This uses gold outcomes after inference and is not deployable.**",
            "",
            f"- Accuracy: {oracle['accuracy']:.4f} ({oracle['correct']}/{summary['samples']})",
            f"- Full usage: {oracle['full_usage_rate']:.4f} ({oracle['full_usage_count']}/{summary['samples']})",
            f"- Average total tokens: {oracle['recorded_costs']['average_usage']['total_tokens']:.2f}",
            f"- Average calls: {oracle['recorded_costs']['average_calls']:.2f}",
            f"- Average latency: {oracle['recorded_costs']['average_latency_seconds']:.4f} seconds",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run_logiqa_audit(predictions_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    predictions = read_jsonl(predictions_path)
    summary, cases = audit_logiqa_predictions(predictions, predictions_path)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    write_jsonl(target / "audit_cases.jsonl", cases)
    _write_text_atomic(
        target / "audit_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text_atomic(target / "audit_report.md", build_audit_report(summary))
    return summary
