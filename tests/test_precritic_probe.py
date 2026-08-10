from __future__ import annotations

import json

import torch

from hierarchical_control.cli import main
from hierarchical_control.precritic_probe import (
    LABELS,
    ProbeExample,
    aggregate_validation_cost,
    build_probe_example,
    deterministic_stratified_folds,
    oof_budget_thresholds,
    select_oof_threshold,
)


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


def _validation_row(
    question_id: int,
    *,
    gold: str,
    solver_answer: str = "A",
    critic_answer: str | None = None,
    sentinel: str = "CONTINUATION_SENTINEL",
) -> dict:
    verdict = "REVISE" if critic_answer is not None else "KEEP"
    proposed = critic_answer or "NONE"
    final = critic_answer or solver_answer
    return {
        "question_id": question_id,
        "gold": gold,
        "mock_only": False,
        "policy_selection_validation": True,
        "solver_called_once": True,
        "same_solver_state_for_both_policies": True,
        "problem": {
            "passage": f"Passage {question_id}",
            "question": "Which option must be true?",
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
        },
        "solver": {
            **_stage(10, 1, 1.0, f"solver reasoning\nFINAL_ANSWER: {solver_answer}"),
            "strict_answer": solver_answer,
            "tolerant": {
                "answer": solver_answer,
                "match_count": 1,
                "conflict": False,
            },
        },
        "minimal_v1": {
            "tolerant_answer": solver_answer,
            "critic": _stage(12, 2, 2.0, sentinel + " minimal critic"),
            "refiner": _stage(13, 3, 3.0, sentinel + " minimal refiner"),
        },
        "structured_v2": {
            "tolerant_answer": final,
            "critic_parse_failure": False,
            "critic": {
                **_stage(
                    20,
                    2,
                    2.0,
                    sentinel + f"\nVERDICT: {verdict}\nPROPOSED_ANSWER: {proposed}",
                ),
                "parsed_verdict": verdict,
                "proposed_answer": proposed,
                "effective_verdict": verdict,
                "effective_proposed_answer": critic_answer,
            },
            "refiner": _stage(
                30,
                3,
                3.0,
                sentinel + f" refiner\nFINAL_ANSWER: {final}",
            ),
        },
    }


def test_label_mapping_covers_all_four_stop_to_critic_transitions():
    rows = [
        _validation_row(1, gold="A"),
        _validation_row(2, gold="A", critic_answer="B"),
        _validation_row(3, gold="B", critic_answer="B"),
        _validation_row(4, gold="C", critic_answer="B"),
    ]
    labels = [build_probe_example(row, "validation_100").label for row in rows]
    assert labels == list(LABELS)


def test_model_input_has_no_gold_or_continuation_leakage():
    example = build_probe_example(
        _validation_row(1, gold="B", critic_answer="B"), "validation_100"
    )
    assert set(example.model_input) == {"problem", "solver"}
    assert set(example.model_input["solver"]) == {
        "raw_output",
        "parse_status",
        "usage",
    }
    serialized = json.dumps(example.model_input, ensure_ascii=False)
    assert "CONTINUATION_SENTINEL" not in serialized
    forbidden_keys = {"gold", "critic", "refiner", "actions", "outcome"}

    def inspect_keys(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint({str(key).casefold() for key in value})
            for child in value.values():
                inspect_keys(child)
        elif isinstance(value, list):
            for child in value:
                inspect_keys(child)

    inspect_keys(example.model_input)


def test_stratified_folds_are_deterministic_and_partition_oof_once():
    labels = list(LABELS) * 5
    first = deterministic_stratified_folds(labels, n_splits=5, seed=20260813)
    second = deterministic_stratified_folds(labels, n_splits=5, seed=20260813)
    assert first == second
    held_out = [index for fold in first for index in fold["validation"]]
    assert sorted(held_out) == list(range(20))
    assert len(held_out) == len(set(held_out))
    for fold in first:
        fold_labels = [labels[index] for index in fold["validation"]]
        assert sorted(fold_labels) == sorted(LABELS)
        assert set(fold["train"]).isdisjoint(fold["validation"])


def test_threshold_selection_and_budget_curve_use_oof_only_with_safe_stop():
    scores = [0.8, 0.7, 0.6, 0.5]
    labels = [
        "wrong_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    ]
    selected = select_oof_threshold(scores, labels)
    assert selected["selection_source"].endswith("oof_only")
    assert selected["validation_used_for_selection"] is False
    assert selected["net_benefit"] == 1
    assert selected["critic_calls"] == 1
    assert selected["threshold"] == 0.8

    harmful_only = select_oof_threshold(
        [0.9, 0.8], ["correct_to_wrong", "wrong_to_wrong"]
    )
    assert harmful_only["safe_fallback_always_stop"] is True
    assert harmful_only["critic_calls"] == 0
    assert harmful_only["net_benefit"] == 0
    curve = oof_budget_thresholds(list(range(100)))
    assert [point["oof_critic_calls"] for point in curve] == [10, 20, 30, 50, 100]
    assert all(
        point["threshold_source"] == "collection_200_oof_score_distribution"
        for point in curve
    )


def test_validation_cost_uses_exact_saved_solver_and_critic_stages():
    stop_example = build_probe_example(
        _validation_row(1, gold="A"), "validation_100"
    )
    critic_example = build_probe_example(
        _validation_row(2, gold="B", critic_answer="B"), "validation_100"
    )
    cost = aggregate_validation_cost([stop_example, critic_example], [False, True])
    assert cost["total"] == {
        "prompt_tokens": 40,
        "completion_tokens": 4,
        "total_tokens": 44,
        "calls": 3,
        "latency_seconds": 4.0,
    }
    assert cost["mean"]["total_tokens"] == 22.0
    assert cost["estimated"] is False


def test_probe_cli_never_initializes_backend(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("probe CLI must not initialize a model backend")

    received = {}

    def fake_probe(**kwargs):
        received.update(kwargs)
        return {"controller_probe": True, "model_calls": 0}

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden_backend)
    monkeypatch.setattr(cli, "run_precritic_gate_probe", fake_probe)
    assert main(
        [
            "probe-precritic-gate",
            "--collection",
            "collection.jsonl",
            "--validation",
            "validation.jsonl",
            "--output-dir",
            str(tmp_path / "probe"),
        ]
    ) == 0
    assert received == {
        "collection_rollouts": "collection.jsonl",
        "validation_predictions": "validation.jsonl",
        "output_dir": str(tmp_path / "probe"),
    }
