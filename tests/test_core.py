from __future__ import annotations

import copy

import pytest

from hierarchical_control.backend import MockBackend
from hierarchical_control.collection import (
    collect_action_labels,
    collect_budget_labels,
    collect_counterfactual_outcomes,
    select_action_label,
    select_budget_label,
)
from hierarchical_control.config import Action, BudgetLimit, ExperimentConfig
from hierarchical_control.engine import CollaborationEngine
from hierarchical_control.evaluator import MockEvaluator
from hierarchical_control.graph import build_workflow
from hierarchical_control.types import AgentState, CompletionResult, Usage


@pytest.fixture
def setup():
    config = ExperimentConfig()
    backend = MockBackend()
    return config, backend, CollaborationEngine(backend, config)


def test_skip_and_stop_do_not_call_model(setup):
    config, backend, engine = setup
    state = engine.solve_once("probe")
    calls = len(backend.calls)
    state = engine.execute_action(state, Action.SKIP, config.collection_limit)
    state = engine.execute_action(state, Action.STOP, config.collection_limit)
    assert len(backend.calls) == calls
    assert state.terminated


def test_action_mask_and_budget_are_hard_limits(setup):
    config, _, engine = setup
    state = engine.solve_once("probe")
    low = config.tier("LOW")
    mask = engine.action_mask(state, low)
    assert mask == {"SKIP": True, "SHORT": True, "MEDIUM": False, "FULL": False, "STOP": True}
    with pytest.raises(ValueError, match="Illegal action"):
        engine.execute_action(state, Action.FULL, low)
    state = engine.execute_action(state, Action.SHORT, low)
    state = engine.execute_action(state, Action.SHORT, low)
    assert state.usage.extra_completion_tokens == low.extra_tokens
    assert state.usage.extra_tokens == low.extra_tokens
    assert state.usage.extra_total_tokens == (
        state.usage.extra_prompt_tokens + state.usage.extra_completion_tokens
    )
    assert state.usage.extra_calls == low.extra_calls
    assert state.terminated


def test_budget_collection_reuses_solver_and_labels_minimum(setup):
    _, backend, engine = setup
    examples = [
        {"id": str(difficulty), "query": f"q{difficulty}", "difficulty": difficulty}
        for difficulty in range(5)
    ]
    result = collect_budget_labels(examples, engine, MockEvaluator())
    assert [row["budget_label"] for row in result.rollouts] == [
        "ZERO",
        "LOW",
        "MEDIUM",
        "HIGH",
        None,
    ]
    assert sum(call["purpose"] == "solver" for call in backend.calls) == len(examples)
    assert len(result.training) == 4
    assert all(
        row["budget_semantics_version"] == 2
        and row["hard_budget"] == "completion_tokens+calls"
        and row["optimization_cost"] == "total_tokens+calls"
        for row in result.rollouts
    )
    assert {
        "extra_prompt_tokens",
        "extra_completion_tokens",
        "extra_total_tokens",
        "extra_calls",
    } == set(result.rollouts[0]["candidates"][0]["actual_cost"])


def test_counterfactual_branches_are_isolated(setup):
    _, _, engine = setup
    state = engine.solve_once("branch query", {"difficulty": 2})
    pristine = copy.deepcopy(state.to_dict())
    outcomes = collect_counterfactual_outcomes(
        state, {"query": state.query, "difficulty": 2}, engine, MockEvaluator()
    )
    assert state.to_dict() == pristine
    assert [outcome["action"] for outcome in outcomes] == [
        "SKIP",
        "SHORT",
        "MEDIUM",
        "FULL",
        "STOP",
    ]
    outcomes[0]["final_state"]["history"].append({"role": "system", "content": "mutation"})
    assert outcomes[1]["final_state"]["history"][-1].get("content") != "mutation"


def test_label_selectors_use_quality_then_cost():
    candidates = [
        {"tier": "ZERO", "quality": 0.0, "success": False},
        {"tier": "LOW", "quality": 1.0, "success": True},
        {"tier": "MEDIUM", "quality": 1.0, "success": True},
        {"tier": "HIGH", "quality": 1.0, "success": True},
    ]
    assert select_budget_label(candidates) == "LOW"
    outcomes = [
        {"action": "SHORT", "quality": 0.8, "future_cost": {"extra_tokens": 64, "extra_calls": 1}},
        {"action": "MEDIUM", "quality": 0.8, "future_cost": {"extra_tokens": 192, "extra_calls": 1}},
        {"action": "STOP", "quality": 0.1, "future_cost": {"extra_tokens": 0, "extra_calls": 0}},
    ]
    selected = select_action_label(
        outcomes,
        {"SHORT": True, "MEDIUM": True, "STOP": True},
        BudgetLimit(300, 2),
        call_cost_weight=1024,
    )
    assert selected["action"] == "SHORT"


def test_quality_tie_cost_order_uses_total_tokens_not_completion_tokens():
    outcomes = [
        {
            "action": "SHORT",
            "quality": 1.0,
            "future_cost": {
                "extra_prompt_tokens": 1000,
                "extra_completion_tokens": 1,
                "extra_total_tokens": 1001,
                "extra_calls": 1,
            },
        },
        {
            "action": "MEDIUM",
            "quality": 1.0,
            "future_cost": {
                "extra_prompt_tokens": 0,
                "extra_completion_tokens": 100,
                "extra_total_tokens": 100,
                "extra_calls": 1,
            },
        },
    ]
    selected = select_action_label(
        outcomes,
        {"SHORT": True, "MEDIUM": True},
        BudgetLimit(128, 1),
        call_cost_weight=1024.0,
    )
    assert selected["action"] == "MEDIUM"


def test_action_mask_uses_completion_cap_not_posthoc_total_tokens(setup):
    config, _, engine = setup
    state = AgentState(
        query="q",
        current_answer="a",
        history=[],
        usage=Usage(
            extra_prompt_tokens=10_000,
            extra_completion_tokens=60,
            extra_total_tokens=10_060,
            extra_calls=1,
        ),
    )
    mask = engine.action_mask(state, BudgetLimit(128, 2))
    assert mask["SHORT"] is True
    assert mask["MEDIUM"] is False
    assert engine.remaining(state, BudgetLimit(128, 2)).extra_tokens == 68


class _SequenceBackend:
    mock_only = False

    def __init__(self, results):
        self.results = list(results)

    def complete(self, messages, max_tokens, purpose):
        return self.results.pop(0)


def _completion(prompt: int, completion: int, total: int | None = None):
    return CompletionResult(
        content="answer",
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion if total is None else total,
        usage_reported=True,
    )


def test_engine_records_all_real_usage_and_checks_token_identity():
    config = ExperimentConfig(max_collaboration_steps=2)
    backend = _SequenceBackend([_completion(10, 2), _completion(100, 3)])
    engine = CollaborationEngine(backend, config)
    state = engine.solve_once("q")
    state = engine.execute_action(state, Action.SHORT, config.collection_limit)
    assert state.usage.to_dict() == {
        "extra_prompt_tokens": 100,
        "extra_completion_tokens": 3,
        "extra_total_tokens": 103,
        "extra_calls": 1,
    }

    inconsistent = CollaborationEngine(
        _SequenceBackend([_completion(10, 2, total=99)]),
        config,
    )
    with pytest.raises(RuntimeError, match="inconsistent token usage"):
        inconsistent.solve_once("q")


def test_real_backend_missing_usage_fails_without_estimation():
    backend = _SequenceBackend(
        [CompletionResult(content="answer", completion_tokens=2)]
    )
    engine = CollaborationEngine(backend, ExperimentConfig())
    with pytest.raises(RuntimeError, match="usage estimation is forbidden"):
        engine.solve_once("q")


def test_legacy_extra_tokens_usage_and_state_are_readable():
    usage = Usage.from_dict({"extra_tokens": 7, "extra_calls": 2})
    assert usage.extra_tokens == 7
    assert usage.extra_completion_tokens == 7
    assert usage.extra_prompt_tokens == 0
    assert usage.extra_total_tokens == 7
    assert usage.to_dict() == {
        "extra_prompt_tokens": 0,
        "extra_completion_tokens": 7,
        "extra_total_tokens": 7,
        "extra_calls": 2,
    }
    state = AgentState.from_dict(
        {
            "query": "legacy",
            "current_answer": "answer",
            "history": [],
            "usage": {"extra_tokens": 9, "extra_calls": 1},
        }
    )
    assert state.usage.extra_completion_tokens == 9
    assert state.usage.extra_total_tokens == 9


def test_action_rollouts_emit_v2_cost_semantics(setup):
    _, _, engine = setup
    result = collect_action_labels(
        [{"id": "q", "query": "q", "difficulty": 1}],
        engine,
        MockEvaluator(),
    )
    assert result.rollouts
    assert all(
        row["budget_semantics_version"] == 2
        and row["hard_budget"] == "completion_tokens+calls"
        and row["optimization_cost"] == "total_tokens+calls"
        for row in result.rollouts
    )
    assert all(
        set(outcome["future_cost"])
        == {
            "extra_prompt_tokens",
            "extra_completion_tokens",
            "extra_total_tokens",
            "extra_calls",
        }
        for row in result.rollouts
        for outcome in row["outcomes"]
    )


def test_langgraph_workflow_executes_masked_cycle(setup):
    config, _, engine = setup

    def budget_predictor(query, max_budget):
        return "LOW"

    def action_predictor(state, allocated, mask):
        return "SHORT" if mask["SHORT"] else "STOP"

    graph = build_workflow(engine, budget_predictor, action_predictor)
    result = graph.invoke(
        {"query": "graph query", "metadata": {"difficulty": 1}, "max_budget": config.tier("HIGH")}
    )
    assert result["allocated_tier"] == "LOW"
    assert result["agent_state"].terminated
    usage = result["agent_state"].usage
    assert usage.extra_completion_tokens == 128
    assert usage.extra_calls == 2
    assert usage.extra_total_tokens == usage.extra_prompt_tokens + 128
