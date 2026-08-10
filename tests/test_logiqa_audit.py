from __future__ import annotations

import json

import pytest

import hierarchical_control.cli as cli
from hierarchical_control.io_utils import write_jsonl
from hierarchical_control.logiqa_audit import (
    audit_logiqa_predictions,
    tolerant_final_answer,
)


def _prediction(
    question_id: int,
    gold: str,
    solver_answer: str | None,
    full_answer: str | None,
    solver_raw: str | None = None,
    full_raw: str | None = None,
) -> dict:
    solver_raw = solver_raw if solver_raw is not None else f"Reasoning\nFINAL_ANSWER: {solver_answer}"
    full_raw = full_raw if full_raw is not None else f"Refined\nFINAL_ANSWER: {full_answer}"
    return {
        "question_id": question_id,
        "gold": gold,
        "solver_answer": solver_answer,
        "refiner_answer": full_answer,
        "solver_correct": solver_answer == gold,
        "refiner_correct": full_answer == gold,
        "solver_parse_failure": solver_answer is None,
        "refiner_parse_failure": full_answer is None,
        "raw_outputs": {"solver": solver_raw, "critic": "Recorded critique", "refiner": full_raw},
        "usage": {
            "solver_only": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
            "solver_critic_refiner": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
            "calls": {"solver_only": 1, "solver_critic_refiner": 3},
        },
        "latency_seconds": {"solver_only": 1.0, "solver_critic_refiner": 3.0},
        "mock_only": False,
    }


def test_tolerant_parser_recovers_inline_marker_without_guessing_letters():
    recovered = tolerant_final_answer("Analysis ends here; FINAL_ANSWER: C")
    assert recovered.answer == "C"
    assert recovered.match_count == 1
    assert recovered.conflict is False

    for ordinary in (
        "Option A is stronger than B, while C and D are weaker.",
        "The final answer might be D.",
        "NOT_FINAL_ANSWER: A",
        "FINAL_ANSWER: A/B",
    ):
        assert tolerant_final_answer(ordinary).answer is None


def test_tolerant_parser_uses_last_explicit_marker_and_records_conflicts():
    result = tolerant_final_answer(
        "First pass FINAL_ANSWER: A. Reconsidered.\nFINAL_ANSWER: C"
    )
    assert result.answer == "C"
    assert result.matches == ("A", "C")
    assert result.match_count == 2
    assert result.conflict is True

    repeated = tolerant_final_answer("FINAL_ANSWER: B\nAgain FINAL_ANSWER: B")
    assert repeated.answer == "B"
    assert repeated.match_count == 2
    assert repeated.conflict is False


def test_transition_matrix_and_posthoc_oracle_recorded_costs():
    rows = [
        _prediction(1, "A", "A", "A"),
        _prediction(2, "A", "A", "B"),
        _prediction(3, "A", "B", "A"),
        _prediction(4, "A", "B", "B"),
    ]
    summary, cases = audit_logiqa_predictions(rows, "predictions.jsonl")
    transitions = summary["tolerant"]["transitions"]
    assert {name: entry["count"] for name, entry in transitions.items()} == {
        "correct_to_correct": 1,
        "correct_to_wrong": 1,
        "wrong_to_correct": 1,
        "wrong_to_wrong": 1,
    }
    oracle = summary["posthoc_oracle"]
    assert oracle["posthoc_oracle"] is True
    assert oracle["deployable"] is False
    assert oracle["accuracy"] == 0.75
    assert oracle["full_usage_count"] == 1
    assert oracle["full_usage_rate"] == 0.25
    assert oracle["full_usage_ids"] == [3]
    assert oracle["recorded_costs"]["total_usage"] == {
        "prompt_tokens": 38,
        "completion_tokens": 22,
        "total_tokens": 60,
    }
    assert oracle["recorded_costs"]["total_calls"] == 6
    assert oracle["recorded_costs"]["average_calls"] == 1.5
    assert oracle["recorded_costs"]["total_latency_seconds"] == 6.0
    assert oracle["recorded_costs"]["average_latency_seconds"] == 1.5
    assert [case["posthoc_oracle"]["selected_strategy"] for case in cases] == [
        "solver_only",
        "solver_only",
        "solver_critic_refiner",
        "solver_only",
    ]


def test_format_recovery_is_reported_in_cases():
    row = _prediction(
        9,
        "C",
        None,
        None,
        solver_raw="Reasoning on same line. FINAL_ANSWER: C",
        full_raw="Refined on same line. FINAL_ANSWER: B",
    )
    summary, cases = audit_logiqa_predictions([row], "predictions.jsonl")
    assert summary["format_recovery"]["solver_only_ids"] == [9]
    assert summary["format_recovery"]["full_ids"] == [9]
    assert cases[0]["strict"]["solver_answer"] is None
    assert cases[0]["tolerant"]["solver_answer"] == "C"
    assert cases[0]["tolerant"]["full_answer"] == "B"
    assert cases[0]["format_recovered"] == {
        "solver_only": True,
        "full": True,
        "any": True,
    }


def test_audit_cli_never_initializes_a_backend(tmp_path, monkeypatch, capsys):
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [_prediction(1, "A", "A", "A")])

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("audit must not initialize a backend")

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden_backend)
    monkeypatch.setattr(cli, "MockBackend", forbidden_backend)
    output_dir = tmp_path / "audit"
    assert (
        cli.main(
            [
                "audit-logiqa-pilot",
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    summary = json.loads((output_dir / "audit_summary.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (output_dir / "audit_cases.jsonl").read_text().splitlines()]
    report = (output_dir / "audit_report.md").read_text(encoding="utf-8")
    assert summary["offline_audit"] is True
    assert summary["source_mock_only"] is False
    assert len(cases) == 1
    assert "posthoc_oracle=true" in report
