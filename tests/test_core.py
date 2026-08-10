from __future__ import annotations

import copy

import pytest

from hierarchical_control.backend import MockBackend
from hierarchical_control.collection import (
    collect_budget_labels,
    collect_counterfactual_outcomes,
    select_action_label,
    select_budget_label,
)
from hierarchical_control.config import Action, BudgetLimit, ExperimentConfig
from hierarchical_control.engine import CollaborationEngine
from hierarchical_control.evaluator import MockEvaluator
from hierarchical_control.graph import build_workflow


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
    assert state.usage.extra_tokens == low.extra_tokens
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
    assert result["agent_state"].usage.to_dict() == {"extra_tokens": 128, "extra_calls": 2}
