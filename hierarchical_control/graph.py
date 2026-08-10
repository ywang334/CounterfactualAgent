from __future__ import annotations

from typing import Any, Callable, TypedDict

from .config import Action, BudgetLimit, BudgetTier, ExperimentConfig
from .engine import CollaborationEngine
from .types import AgentState


BudgetPredictor = Callable[[str, BudgetLimit], str]
ActionPredictor = Callable[[AgentState, BudgetLimit, dict[str, bool]], str]


class GraphState(TypedDict, total=False):
    query: str
    metadata: dict[str, Any]
    max_budget: BudgetLimit
    allocated_tier: str
    allocated_budget: BudgetLimit
    agent_state: AgentState
    selected_action: str


def build_workflow(
    engine: CollaborationEngine,
    budget_predictor: BudgetPredictor,
    action_predictor: ActionPredictor,
):
    """Build the required cyclic workflow with LangGraph."""
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - environment error
        raise RuntimeError("LangGraph is required. Install the project dependencies first.") from exc

    config: ExperimentConfig = engine.config

    def allocate(state: GraphState) -> dict[str, Any]:
        max_budget = state.get("max_budget", config.tier(BudgetTier.HIGH))
        tier = BudgetTier(budget_predictor(state["query"], max_budget))
        selected = config.tier(tier)
        # The allocator may not exceed a caller-supplied deployment ceiling.
        selected = BudgetLimit(
            min(selected.extra_tokens, max_budget.extra_tokens),
            min(selected.extra_calls, max_budget.extra_calls),
        )
        return {"allocated_tier": tier.value, "allocated_budget": selected}

    def solver(state: GraphState) -> dict[str, Any]:
        solved = engine.solve_once(state["query"], state.get("metadata", {}))
        return {"agent_state": solved}

    def controller(state: GraphState) -> dict[str, Any]:
        agent_state = state["agent_state"]
        budget = state["allocated_budget"]
        mask = engine.action_mask(agent_state, budget)
        action = Action(action_predictor(agent_state, budget, mask))
        if not mask[action.value]:
            raise ValueError(f"Action predictor selected masked action {action.value}")
        return {"selected_action": action.value}

    def collaborate(state: GraphState) -> dict[str, Any]:
        next_state = engine.execute_action(
            state["agent_state"], state["selected_action"], state["allocated_budget"]
        )
        return {"agent_state": next_state}

    def after_controller(state: GraphState) -> str:
        return "end" if state["selected_action"] == Action.STOP.value else "collaborate"

    def after_collaboration(state: GraphState) -> str:
        return "end" if state["agent_state"].terminated else "controller"

    graph = StateGraph(GraphState)
    graph.add_node("budget_allocator", allocate)
    graph.add_node("solver", solver)
    graph.add_node("action_controller", controller)
    graph.add_node("critic_or_refiner", collaborate)
    graph.add_edge(START, "budget_allocator")
    graph.add_edge("budget_allocator", "solver")
    graph.add_edge("solver", "action_controller")
    graph.add_conditional_edges(
        "action_controller", after_controller, {"collaborate": "critic_or_refiner", "end": END}
    )
    graph.add_conditional_edges(
        "critic_or_refiner", after_collaboration, {"controller": "action_controller", "end": END}
    )
    return graph.compile()
