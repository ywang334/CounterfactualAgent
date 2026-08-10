from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ACTION_LABELS, BUDGET_LABELS, Action, BudgetLimit, BudgetTier, ExperimentConfig
from .engine import CollaborationEngine, StaticReferencePolicy
from .evaluator import Evaluator
from .types import AgentState


TIER_REFERENCE_ACTION = {
    BudgetTier.ZERO.value: Action.STOP.value,
    BudgetTier.LOW.value: Action.SHORT.value,
    BudgetTier.MEDIUM.value: Action.MEDIUM.value,
    BudgetTier.HIGH.value: Action.FULL.value,
}

COST_SEMANTICS_V2 = {
    "budget_semantics_version": 2,
    "hard_budget": "completion_tokens+calls",
    "optimization_cost": "total_tokens+calls",
}


@dataclass
class BudgetCollectionResult:
    rollouts: list[dict[str, Any]]
    training: list[dict[str, Any]]
    metrics: dict[str, Any]


def _example_id(example: dict[str, Any], index: int) -> str:
    return str(example.get("id", index))


def _max_budget(example: dict[str, Any], config: ExperimentConfig) -> BudgetLimit:
    value = example.get("max_budget")
    if value is None:
        return config.tier(BudgetTier.HIGH)
    if isinstance(value, str):
        return config.tier(value)
    if isinstance(value, dict):
        return BudgetLimit(**value)
    raise ValueError("max_budget must be a tier name or a budget object")


def select_budget_label(
    candidates: list[dict[str, Any]], quality_tolerance: float = 1e-8
) -> str | None:
    if not any(bool(candidate["success"]) for candidate in candidates):
        return None
    best_quality = max(float(candidate["quality"]) for candidate in candidates)
    for tier in BUDGET_LABELS:
        candidate = next((item for item in candidates if item["tier"] == tier), None)
        if candidate is not None and float(candidate["quality"]) >= best_quality - quality_tolerance:
            return tier
    raise AssertionError("At least one candidate must match the best quality")


def collect_budget_labels(
    examples: list[dict[str, Any]],
    engine: CollaborationEngine,
    evaluator: Evaluator,
    include_unsolved: bool = False,
) -> BudgetCollectionResult:
    rollouts: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    solved_count = 0
    for index, example in enumerate(examples):
        query = str(example["query"])
        max_budget = _max_budget(example, engine.config)
        # This is the only Solver call for all budget branches of this query.
        solver_state = engine.solve_once(query, example)
        candidates: list[dict[str, Any]] = []
        for tier in BUDGET_LABELS:
            budget = engine.config.tier(tier)
            policy = StaticReferencePolicy(engine, TIER_REFERENCE_ACTION[tier])
            final_state = policy.run(solver_state, budget)
            quality, success = evaluator.evaluate(query, final_state.current_answer, example)
            candidates.append(
                {
                    "tier": tier,
                    "within_max_budget": (
                        budget.extra_tokens <= max_budget.extra_tokens
                        and budget.extra_calls <= max_budget.extra_calls
                    ),
                    "quality": quality,
                    "success": success,
                    "actual_cost": final_state.usage.to_dict(),
                    "final_answer": final_state.current_answer,
                    "termination_reason": final_state.termination_reason,
                }
            )
        label = select_budget_label(
            [candidate for candidate in candidates if candidate["within_max_budget"]],
            engine.config.quality_tolerance,
        )
        unsolved = label is None
        if not unsolved:
            solved_count += 1
        rollout = {
            **COST_SEMANTICS_V2,
            "id": _example_id(example, index),
            "query": query,
            "max_budget": max_budget.to_dict(),
            "solver_state": solver_state.to_dict(),
            "candidates": candidates,
            "budget_label": label,
            "unsolved": unsolved,
            "mock_only": bool(getattr(engine.backend, "mock_only", False)),
        }
        rollouts.append(rollout)
        if not unsolved or include_unsolved:
            training.append(
                {
                    **COST_SEMANTICS_V2,
                    "id": rollout["id"],
                    "query": query,
                    "max_budget": max_budget.to_dict(),
                    "budget_label": label,
                    "unsolved": unsolved,
                    "mock_only": rollout["mock_only"],
                }
            )
    return BudgetCollectionResult(
        rollouts=rollouts,
        training=training,
        metrics={
            "queries": len(examples),
            "solved": solved_count,
            "unsolved": len(examples) - solved_count,
            "training_examples": len(training),
            "solver_calls_expected": len(examples),
        },
    )


@dataclass
class ActionCollectionResult:
    rollouts: list[dict[str, Any]]
    training: list[dict[str, Any]]
    metrics: dict[str, Any]


def _future_cost(start: AgentState, final: AgentState) -> dict[str, int]:
    return {
        "extra_prompt_tokens": (
            final.usage.extra_prompt_tokens - start.usage.extra_prompt_tokens
        ),
        "extra_completion_tokens": (
            final.usage.extra_completion_tokens
            - start.usage.extra_completion_tokens
        ),
        "extra_total_tokens": (
            final.usage.extra_total_tokens - start.usage.extra_total_tokens
        ),
        "extra_calls": final.usage.extra_calls - start.usage.extra_calls,
    }


def _cost_completion_tokens(cost: dict[str, Any]) -> int:
    """Read v2 completion cost or the legacy extra_tokens completion field."""
    value = cost.get("extra_completion_tokens", cost.get("extra_tokens"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Future cost is missing non-negative completion tokens")
    return value


def _cost_total_tokens(cost: dict[str, Any]) -> int:
    completion = _cost_completion_tokens(cost)
    prompt = cost.get("extra_prompt_tokens", 0)
    total = cost.get("extra_total_tokens", prompt + completion)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (prompt, total)
    ):
        raise ValueError("Future cost has invalid prompt or total tokens")
    if total != prompt + completion:
        raise ValueError(
            "Future cost total tokens must equal prompt tokens plus completion tokens"
        )
    return total


def collect_counterfactual_outcomes(
    start: AgentState,
    example: dict[str, Any],
    engine: CollaborationEngine,
    evaluator: Evaluator,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    policy = StaticReferencePolicy(engine, engine.config.reference_action)
    pristine = start.to_dict()
    for action_name in ACTION_LABELS:
        branch = start.clone()
        branch = engine.execute_action(branch, action_name, engine.config.collection_limit)
        if not branch.terminated:
            branch = policy.run(branch, engine.config.collection_limit)
        quality, success = evaluator.evaluate(start.query, branch.current_answer, example)
        outcomes.append(
            {
                "action": action_name,
                "quality": quality,
                "success": success,
                "future_cost": _future_cost(start, branch),
                "final_answer": branch.current_answer,
                "final_state": branch.to_dict(),
            }
        )
        if start.to_dict() != pristine:
            raise RuntimeError("Counterfactual branch mutated the shared start state")
    return outcomes


def select_action_label(
    outcomes: list[dict[str, Any]],
    action_mask: dict[str, bool],
    remaining: BudgetLimit,
    call_cost_weight: float,
    quality_tolerance: float = 1e-8,
) -> dict[str, Any]:
    feasible: list[dict[str, Any]] = []
    for outcome in outcomes:
        cost = outcome["future_cost"]
        if not action_mask.get(outcome["action"], False):
            continue
        if (
            _cost_completion_tokens(cost) > remaining.extra_tokens
            or cost["extra_calls"] > remaining.extra_calls
        ):
            continue
        feasible.append(outcome)
    if not feasible:
        raise RuntimeError("STOP should always provide at least one feasible action")
    best_quality = max(float(item["quality"]) for item in feasible)
    quality_tied = [
        item for item in feasible if float(item["quality"]) >= best_quality - quality_tolerance
    ]
    return min(
        quality_tied,
        key=lambda item: (
            item["future_cost"]["extra_calls"] * call_cost_weight
            + _cost_total_tokens(item["future_cost"]),
            ACTION_LABELS.index(item["action"]),
        ),
    )


def collect_action_labels(
    examples: list[dict[str, Any]],
    engine: CollaborationEngine,
    evaluator: Evaluator,
) -> ActionCollectionResult:
    rollouts: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    state_count = 0
    branch_count = 0
    for index, example in enumerate(examples):
        query = str(example["query"])
        state = engine.solve_once(query, example)
        trajectory_policy = StaticReferencePolicy(engine, engine.config.reference_action)
        trajectory_step = 0
        while not state.terminated:
            snapshot = state.clone()
            outcomes = collect_counterfactual_outcomes(snapshot, example, engine, evaluator)
            state_id = f"{_example_id(example, index)}:{trajectory_step}"
            rollout = {
                **COST_SEMANTICS_V2,
                "state_id": state_id,
                "query_id": _example_id(example, index),
                "state": snapshot.to_dict(),
                "collection_limit": engine.config.collection_limit.to_dict(),
                "outcomes": outcomes,
                "mock_only": bool(getattr(engine.backend, "mock_only", False)),
            }
            rollouts.append(rollout)
            state_count += 1
            branch_count += len(outcomes)

            for tier in BUDGET_LABELS:
                allocated = engine.config.tier(tier)
                if (
                    snapshot.usage.extra_completion_tokens > allocated.extra_tokens
                    or snapshot.usage.extra_calls > allocated.extra_calls
                ):
                    continue
                remaining = engine.remaining(snapshot, allocated)
                mask = engine.action_mask(snapshot, allocated)
                selected = select_action_label(
                    outcomes,
                    mask,
                    remaining,
                    engine.config.call_cost_weight,
                    engine.config.quality_tolerance,
                )
                training.append(
                    {
                        **COST_SEMANTICS_V2,
                        "state_id": state_id,
                        "query": query,
                        "state": snapshot.to_dict(),
                        "allocated_tier": tier,
                        "allocated_budget": allocated.to_dict(),
                        "remaining_budget": remaining.to_dict(),
                        "action_mask": mask,
                        "action_label": selected["action"],
                        "target_quality": selected["quality"],
                        "target_future_prompt_tokens": selected["future_cost"][
                            "extra_prompt_tokens"
                        ],
                        "target_future_completion_tokens": selected["future_cost"][
                            "extra_completion_tokens"
                        ],
                        "target_future_total_tokens": selected["future_cost"][
                            "extra_total_tokens"
                        ],
                        "target_future_calls": selected["future_cost"]["extra_calls"],
                        "mock_only": rollout["mock_only"],
                    }
                )
            next_action = trajectory_policy.choose(state, engine.config.collection_limit)
            state = engine.execute_action(state, next_action, engine.config.collection_limit)
            trajectory_step += 1
    return ActionCollectionResult(
        rollouts=rollouts,
        training=training,
        metrics={
            "queries": len(examples),
            "reference_states": state_count,
            "counterfactual_branches": branch_count,
            "training_examples": len(training),
        },
    )
