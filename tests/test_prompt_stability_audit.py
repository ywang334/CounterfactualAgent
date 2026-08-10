from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierarchical_control import cli
from hierarchical_control.prompt_stability_audit import (
    binary_confusion_matrix,
    classify_structured_critic,
    compare_id_sets,
    minimum_cost_posthoc_oracle,
    pair_unique_samples,
)


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _cost(total: int, calls: float, latency: float) -> dict:
    prompt = total - 1
    return {
        "usage": _usage(prompt, 1),
        "calls": calls,
        "latency_seconds": latency,
    }


@pytest.mark.parametrize(
    ("output", "solver", "category", "inconsistent", "detected"),
    [
        (
            "VERDICT: KEEP\nPROPOSED_ANSWER: NONE",
            "A",
            "canonical_keep",
            False,
            False,
        ),
        (
            "VERDICT: KEEP\nPROPOSED_ANSWER: A",
            "A",
            "canonical_keep",
            False,
            False,
        ),
        (
            "VERDICT: KEEP\nPROPOSED_ANSWER: B",
            "A",
            "contradictory_keep",
            True,
            False,
        ),
        (
            "VERDICT: REVISE\nPROPOSED_ANSWER: NONE",
            "A",
            "incomplete_revise",
            True,
            True,
        ),
        (
            "VERDICT: REVISE\nPROPOSED_ANSWER: A",
            "A",
            "noop_revise",
            True,
            True,
        ),
        (
            "VERDICT: REVISE\nPROPOSED_ANSWER: C",
            "A",
            "actionable_revise",
            False,
            True,
        ),
        ("No structured fields", "A", "malformed", False, False),
        (
            "VERDICT: KEEP\nVERDICT: REVISE\nPROPOSED_ANSWER: B",
            "A",
            "malformed",
            False,
            False,
        ),
    ],
)
def test_structured_critic_protocol_classification(
    output,
    solver,
    category,
    inconsistent,
    detected,
):
    result = classify_structured_critic(output, solver)
    assert result["category"] == category
    assert result["contract_inconsistency"] is inconsistent
    assert result["detected_solver_error"] is detected


def test_confusion_matrix_metrics():
    result = binary_confusion_matrix(
        [True, True, False, False],
        [True, False, True, False],
        [10, 11, 12, 13],
    )
    assert result["matrix"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    assert result["sample_ids"]["true_positive"] == [10]
    assert result["sample_ids"]["false_negative"] == [11]
    assert result["sample_ids"]["false_positive"] == [12]
    assert result["sample_ids"]["true_negative"] == [13]
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["specificity"] == 0.5


def test_jaccard_and_changed_sets():
    result = compare_id_sets([1, 2], [2, 3])
    assert result["intersection"] == [2]
    assert result["union"] == [1, 2, 3]
    assert result["minimal_v1_only"] == [1]
    assert result["structured_v2_only"] == [3]
    assert result["jaccard"] == pytest.approx(1 / 3)


def test_minimum_cost_posthoc_oracle_uses_gold_and_actual_costs():
    cases = [
        {
            "question_id": 1,
            "gold": "A",
            "solver": {"tolerant_answer": "A", "cost": _cost(10, 1, 1.0)},
            "minimal_v1": {"tolerant_answer": "B", "cost": _cost(30, 3, 3.0)},
        },
        {
            "question_id": 2,
            "gold": "B",
            "solver": {"tolerant_answer": "A", "cost": _cost(10, 1, 1.0)},
            "minimal_v1": {"tolerant_answer": "B", "cost": _cost(30, 3, 3.0)},
        },
        {
            "question_id": 3,
            "gold": "B",
            "solver": {"tolerant_answer": "A", "cost": _cost(10, 1, 1.0)},
            "minimal_v1": {"tolerant_answer": "C", "cost": _cost(30, 3, 3.0)},
        },
        {
            "question_id": 4,
            "gold": "A",
            "solver": {"tolerant_answer": "A", "cost": _cost(10, 1, 1.0)},
            "minimal_v1": {"tolerant_answer": "A", "cost": _cost(5, 1, 0.5)},
        },
    ]
    result = minimum_cost_posthoc_oracle(cases, "minimal_v1")
    assert result["posthoc_oracle"] is True
    assert result["deployable"] is False
    assert result["correct"] == 3
    assert result["accuracy"] == 0.75
    assert result["full_usage_count"] == 2
    assert result["full_usage_ids"] == [2, 4]
    assert result["recorded_costs"]["total_usage"]["total_tokens"] == 55
    assert result["recorded_costs"]["total_calls"] == 6
    assert result["recorded_costs"]["total_latency_seconds"] == 5.5


def test_exactly_50_unique_one_to_one_ids_are_required():
    minimal = [{"question_id": index} for index in range(50)]
    structured = [{"question_id": index} for index in reversed(range(50))]
    pairs = pair_unique_samples(minimal, structured)
    assert len(pairs) == 50
    assert all(left["question_id"] == right["question_id"] for left, right in pairs)

    with pytest.raises(ValueError, match="exactly 50"):
        pair_unique_samples(minimal[:-1], structured)

    duplicate = list(structured)
    duplicate[-1] = {"question_id": duplicate[0]["question_id"]}
    with pytest.raises(ValueError, match="duplicate"):
        pair_unique_samples(minimal, duplicate)

    mismatch = list(structured)
    mismatch[-1] = {"question_id": 999}
    with pytest.raises(ValueError, match="not one-to-one"):
        pair_unique_samples(minimal, mismatch)


def _write_policy_inputs(root: Path) -> tuple[Path, Path]:
    minimal_path = root / "minimal.jsonl"
    structured_path = root / "structured.jsonl"
    minimal_rows = []
    structured_rows = []
    for question_id in range(50):
        solver_output = "Saved solver.\nFINAL_ANSWER: A"
        full_output = "Saved full.\nFINAL_ANSWER: A"
        minimal_rows.append(
            {
                "question_id": question_id,
                "gold": "A",
                "raw_outputs": {
                    "solver": solver_output,
                    "critic": "Saved minimal critic.",
                    "refiner": full_output,
                },
                "usage": {
                    "solver_only": _usage(10, 5),
                    "solver_critic_refiner": _usage(30, 10),
                    "calls": {
                        "solver_only": 1,
                        "solver_critic_refiner": 3,
                    },
                },
                "latency_seconds": {
                    "solver_only": 1.0,
                    "solver_critic_refiner": 3.0,
                },
                "mock_only": False,
            }
        )
        structured_rows.append(
            {
                "question_id": question_id,
                "gold": "A",
                "prompt_version": "structured_v2",
                "prompt_development": True,
                "solver_reused": True,
                "solver_called": False,
                "mock_only": False,
                "problem": {
                    "passage": "P",
                    "question": "Q",
                    "options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                },
                "problem_and_choices": "P\nQ\nA. a\nB. b\nC. c\nD. d",
                "solver": {
                    "raw_output": solver_output,
                    "tolerant": {"answer": "A"},
                },
                "critic": {
                    "raw_output": "VERDICT: KEEP\nPROPOSED_ANSWER: NONE"
                },
                "critic_parse_failure": False,
                "refiner": {
                    "raw_output": full_output,
                    "tolerant": {"answer": "A"},
                },
                "tolerant": {
                    "solver_answer": "A",
                    "full_answer": "A",
                },
                "usage": {
                    "solver_reused": _usage(10, 5),
                    "complete_v2": _usage(40, 10),
                },
                "calls": {
                    "complete_workflow_equivalent": 3,
                },
                "latency_seconds": {
                    "solver_recorded": 1.0,
                    "complete_v2": 4.0,
                },
            }
        )
    minimal_path.write_text(
        "".join(json.dumps(row) + "\n" for row in minimal_rows),
        encoding="utf-8",
    )
    structured_path.write_text(
        "".join(json.dumps(row) + "\n" for row in structured_rows),
        encoding="utf-8",
    )
    return minimal_path, structured_path


def test_audit_cli_never_initializes_backend_and_preserves_inputs(
    tmp_path,
    monkeypatch,
    capsys,
):
    minimal_path, structured_path = _write_policy_inputs(tmp_path)
    minimal_before = minimal_path.read_bytes()
    structured_before = structured_path.read_bytes()

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("offline audit initialized a backend")

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden_backend)
    monkeypatch.setattr(cli, "MockBackend", forbidden_backend)
    output_dir = tmp_path / "audit"
    assert (
        cli.main(
            [
                "audit-prompt-stability",
                "--minimal-predictions",
                str(minimal_path),
                "--structured-predictions",
                str(structured_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert minimal_path.read_bytes() == minimal_before
    assert structured_path.read_bytes() == structured_before
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "cases.jsonl",
        "report.md",
        "summary.json",
    ]
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["offline_audit"] is True
    assert summary["model_backend_initialized"] is False
    assert summary["model_calls"] == 0
    assert summary["controller_training"] is False
    assert summary["samples"] == 50
    assert len((output_dir / "cases.jsonl").read_text().splitlines()) == 50
