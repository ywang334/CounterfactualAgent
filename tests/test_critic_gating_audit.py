from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hierarchical_control.cli import main
from hierarchical_control.critic_gating_audit import (
    _dataset_summary,
    build_collection_case,
    build_validation_case,
    run_critic_gating_audit,
    strategy_metrics,
)
from hierarchical_control.io_utils import write_jsonl


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _stage(prompt: int, completion: int, latency: float, raw: str) -> dict:
    return {
        "raw_output": raw,
        "usage": _usage(prompt, completion),
        "calls": 1,
        "latency_seconds": latency,
    }


def _critic(
    verdict: str,
    proposed: str | None,
    *,
    prompt: int = 20,
    completion: int = 2,
    latency: float = 2.0,
) -> dict:
    proposed_field = proposed or "NONE"
    return {
        **_stage(
            prompt,
            completion,
            latency,
            f"VERDICT: {verdict}\nPROPOSED_ANSWER: {proposed_field}",
        ),
        "parsed_verdict": verdict,
        "proposed_answer": proposed_field,
        "effective_verdict": verdict,
        "effective_proposed_answer": proposed,
    }


def _validation_row(
    question_id: int,
    *,
    gold: str = "A",
    solver: str = "A",
    minimal: str = "A",
    full: str = "A",
    verdict: str = "KEEP",
    proposed: str | None = None,
) -> dict:
    structured_critic = _critic(verdict, proposed)
    return {
        "question_id": question_id,
        "gold": gold,
        "mock_only": False,
        "policy_selection_validation": True,
        "solver_called_once": True,
        "same_solver_state_for_both_policies": True,
        "solver": {
            **_stage(10, 1, 1.0, f"solver\nFINAL_ANSWER: {solver}"),
            "tolerant": {"answer": solver},
        },
        "minimal_v1": {
            "tolerant_answer": minimal,
            "critic": _stage(40, 4, 4.0, "minimal critic"),
            "refiner": _stage(
                50, 5, 5.0, f"minimal refiner\nFINAL_ANSWER: {minimal}"
            ),
        },
        "structured_v2": {
            "tolerant_answer": full,
            "critic_parse_failure": False,
            "critic": structured_critic,
            "refiner": _stage(
                30, 3, 3.0, f"structured refiner\nFINAL_ANSWER: {full}"
            ),
        },
    }


def _collection_cost(calls: int) -> dict:
    return {
        "prompt_tokens": calls * 10,
        "completion_tokens": calls,
        "total_tokens": calls * 11,
        "calls": calls,
        "latency_seconds": float(calls),
    }


def _collection_row(
    question_id: int,
    *,
    gold: str = "A",
    solver: str = "A",
    minimal: str = "A",
    full: str = "A",
    verdict: str = "KEEP",
    proposed: str | None = None,
) -> dict:
    proposed_field = proposed or "NONE"
    solver_raw = f"solver\nFINAL_ANSWER: {solver}"
    return {
        "question_id": question_id,
        "sample_id": f"sample-{question_id}",
        "gold": gold,
        "mock_only": False,
        "actual_calls": 5,
        "solver": {
            "raw_output": solver_raw,
            "tolerant": {"answer": solver},
            "cost": _collection_cost(1),
        },
        "actions": {
            "STOP": {
                "tolerant": {"answer": solver},
                "raw_outputs": {"solver": solver_raw},
            },
            "SHORT": {
                "tolerant": {"answer": minimal},
                "incremental_cost": _collection_cost(2),
                "raw_outputs": {
                    "critic": "minimal critic",
                    "refiner": f"minimal refiner\nFINAL_ANSWER: {minimal}",
                },
            },
            "FULL": {
                "tolerant": {"answer": full},
                "incremental_cost": _collection_cost(2),
                "critic_protocol": {
                    "parsed_verdict": verdict,
                    "proposed_answer": proposed_field,
                    "parse_failure": False,
                    "effective_verdict": verdict,
                    "effective_proposed_answer": proposed,
                },
                "raw_outputs": {
                    "critic": (
                        f"VERDICT: {verdict}\nPROPOSED_ANSWER: {proposed_field}"
                    ),
                    "refiner": f"structured refiner\nFINAL_ANSWER: {full}",
                },
            },
        },
    }


def test_keep_and_revise_simulate_all_five_strategies_and_transitions():
    keep = build_validation_case(
        _validation_row(
            1, gold="A", solver="A", minimal="B", full="B", verdict="KEEP"
        )
    )
    revise = build_validation_case(
        _validation_row(
            2,
            gold="B",
            solver="A",
            minimal="A",
            full="B",
            verdict="REVISE",
            proposed="B",
        )
    )
    disagree = build_validation_case(
        _validation_row(
            3,
            gold="C",
            solver="A",
            minimal="A",
            full="C",
            verdict="REVISE",
            proposed="B",
        )
    )

    assert keep["policies"]["CRITIC_ONLY"]["answer"] == "A"
    assert keep["policies"]["CONDITIONAL_REFINE"]["answer"] == "A"
    assert keep["policies"]["ALWAYS_FULL"]["answer"] == "B"
    assert keep["refiner"]["changed_on_effective_keep"] is True
    assert revise["policies"]["CRITIC_ONLY"]["answer"] == "B"
    assert revise["policies"]["CONDITIONAL_REFINE"]["answer"] == "B"
    assert revise["critic"]["proposed_refiner_agreement"] is True
    assert disagree["policies"]["CRITIC_ONLY"]["answer"] == "B"
    assert disagree["policies"]["CONDITIONAL_REFINE"]["answer"] == "C"
    assert disagree["critic"]["proposed_refiner_agreement"] is False

    conditional = strategy_metrics([keep, revise, disagree], "CONDITIONAL_REFINE")
    always = strategy_metrics([keep, revise, disagree], "ALWAYS_FULL")
    assert conditional["transitions"]["correct_to_correct"]["count"] == 1
    assert conditional["transitions"]["wrong_to_correct"]["count"] == 2
    assert conditional["corrected"] == 2
    assert conditional["degraded"] == 0
    assert always["degraded"] == 1


def test_validation_uses_exact_saved_stage_usage_for_each_gating_policy():
    keep = build_validation_case(_validation_row(1, verdict="KEEP"))
    revise = build_validation_case(
        _validation_row(2, gold="B", solver="A", full="B", verdict="REVISE", proposed="B")
    )
    assert keep["strategy_costs"]["STOP"]["total_tokens"] == 11
    assert keep["strategy_costs"]["MINIMAL_V1_ABLATION"]["total_tokens"] == 110
    assert keep["strategy_costs"]["CRITIC_ONLY"]["total_tokens"] == 33
    assert keep["strategy_costs"]["CONDITIONAL_REFINE"]["total_tokens"] == 33
    assert keep["strategy_costs"]["ALWAYS_FULL"]["total_tokens"] == 66
    assert revise["strategy_costs"]["CONDITIONAL_REFINE"]["total_tokens"] == 66

    summary = _dataset_summary([keep, revise], stage_usage_available=True)
    conditional = summary["strategies"]["CONDITIONAL_REFINE"]["cost"]
    assert conditional["usage"]["total"] == {
        "prompt_tokens": 90,
        "completion_tokens": 9,
        "total_tokens": 99,
    }
    assert conditional["calls"] == {"available": True, "total": 5, "mean": 2.5}
    assert conditional["latency"]["total_seconds"] == 9.0
    assert conditional["estimated"] is False


def test_collection_reports_exact_calls_but_stage_tokens_unavailable():
    keep = build_collection_case(_collection_row(1, verdict="KEEP"))
    revise = build_collection_case(
        _collection_row(2, gold="B", solver="A", full="B", verdict="REVISE", proposed="B")
    )
    summary = _dataset_summary([keep, revise], stage_usage_available=False)
    assert summary["strategies"]["STOP"]["cost"]["calls"]["total"] == 2
    assert summary["strategies"]["MINIMAL_V1_ABLATION"]["cost"]["calls"]["total"] == 6
    assert summary["strategies"]["CRITIC_ONLY"]["cost"]["calls"]["total"] == 4
    assert summary["strategies"]["CONDITIONAL_REFINE"]["cost"]["calls"]["total"] == 5
    assert summary["strategies"]["ALWAYS_FULL"]["cost"]["calls"]["total"] == 6
    for strategy in summary["strategies"].values():
        assert strategy["cost"]["usage"]["available"] is False
        assert strategy["cost"]["usage"]["total_tokens"] is None
        assert strategy["cost"]["latency"]["available"] is False
        assert strategy["cost"]["estimated"] is False


def test_runner_preserves_inputs_and_cli_does_not_initialize_backend(tmp_path, monkeypatch):
    collection_path = tmp_path / "collection.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_jsonl(collection_path, [_collection_row(1), _collection_row(2, verdict="REVISE", proposed="B")])
    write_jsonl(validation_path, [_validation_row(11), _validation_row(12, verdict="REVISE", proposed="B")])
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (collection_path, validation_path)
    }
    output = tmp_path / "audit"
    summary = run_critic_gating_audit(
        collection_path,
        validation_path,
        output,
        expected_collection_samples=2,
        expected_validation_samples=2,
    )
    assert summary["offline_audit"] is True
    assert summary["deployable"] is False
    assert summary["model_backend_initialized"] is False
    assert summary["model_calls"] == 0
    assert {path.name for path in output.iterdir()} == {
        "summary.json",
        "cases.jsonl",
        "report.md",
    }
    assert len((output / "cases.jsonl").read_text().splitlines()) == 4
    after = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (collection_path, validation_path)
    }
    assert before == after

    import hierarchical_control.cli as cli

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("offline audit must not initialize OpenAIBackend")

    called = {}

    def fake_audit(**kwargs):
        called.update(kwargs)
        return {"offline_audit": True}

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden_backend)
    monkeypatch.setattr(cli, "run_critic_gating_audit", fake_audit)
    assert main(
        [
            "audit-critic-gating",
            "--collection",
            str(collection_path),
            "--validation",
            str(validation_path),
            "--output-dir",
            str(tmp_path / "cli-audit"),
        ]
    ) == 0
    assert called == {
        "collection_rollouts": str(collection_path),
        "validation_predictions": str(validation_path),
        "output_dir": str(tmp_path / "cli-audit"),
    }
