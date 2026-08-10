from __future__ import annotations

from dataclasses import dataclass

from .backend import LLMBackend
from .config import ACTION_LABELS, Action, BudgetLimit, ExperimentConfig
from .types import AgentState, CompletionResult


@dataclass
class CollaborationEngine:
    backend: LLMBackend
    config: ExperimentConfig

    @staticmethod
    def _validated_usage(
        completion: CompletionResult,
        purpose: str,
    ) -> tuple[int, int, int]:
        if (
            not completion.usage_reported
            or completion.prompt_tokens is None
            or completion.total_tokens is None
        ):
            raise RuntimeError(
                f"Backend did not report real prompt/completion/total token usage for {purpose}; "
                "usage estimation is forbidden"
            )
        values = (
            completion.prompt_tokens,
            completion.completion_tokens,
            completion.total_tokens,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise RuntimeError(f"Backend reported invalid token usage for {purpose}")
        prompt_tokens, completion_tokens, total_tokens = values
        if total_tokens != prompt_tokens + completion_tokens:
            raise RuntimeError(
                f"Backend reported inconsistent token usage for {purpose}: "
                f"total_tokens={total_tokens}, prompt_tokens={prompt_tokens}, "
                f"completion_tokens={completion_tokens}"
            )
        return prompt_tokens, completion_tokens, total_tokens

    def solve_once(self, query: str, metadata: dict | None = None) -> AgentState:
        messages = [
            {"role": "system", "content": "You are the frozen Solver. Produce the best direct answer."},
            {"role": "user", "content": query},
        ]
        result = self.backend.complete(messages, self.config.solver_max_tokens, "solver")
        # The Solver is outside the additional collaboration budget, but a real
        # backend must still provide internally consistent usage.
        self._validated_usage(result, "solver")
        history = messages + [{"role": "assistant", "name": "solver", "content": result.content}]
        # Solver usage is intentionally excluded from additional collaboration usage.
        return AgentState(
            query=query,
            current_answer=result.content,
            history=history,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def remaining(state: AgentState, budget: BudgetLimit) -> BudgetLimit:
        return BudgetLimit(
            max(0, budget.extra_tokens - state.usage.extra_completion_tokens),
            max(0, budget.extra_calls - state.usage.extra_calls),
        )

    def action_mask(self, state: AgentState, budget: BudgetLimit) -> dict[str, bool]:
        if state.terminated or state.collaboration_steps >= self.config.max_collaboration_steps:
            return {name: name == Action.STOP.value for name in ACTION_LABELS}
        remaining = self.remaining(state, budget)
        mask = {name: False for name in ACTION_LABELS}
        mask[Action.SKIP.value] = True
        mask[Action.STOP.value] = True
        for action in (Action.SHORT, Action.MEDIUM, Action.FULL):
            cap = self.config.action_token_caps[action.value]
            mask[action.value] = remaining.extra_calls >= 1 and remaining.extra_tokens >= cap
        return mask

    def execute_action(self, state: AgentState, action: str | Action, budget: BudgetLimit) -> AgentState:
        action = Action(action)
        mask = self.action_mask(state, budget)
        if not mask[action.value]:
            raise ValueError(f"Illegal action {action.value}; mask={mask}")
        result_state = state.clone()
        if action is Action.STOP:
            result_state.terminated = True
            result_state.termination_reason = "controller_stop"
            return result_state
        if action is Action.SKIP:
            result_state.history.append(
                {"role": "system", "name": "controller", "content": f"SKIP {result_state.role}"}
            )
            self._advance(result_state)
            return result_state

        cap = self.config.action_token_caps[action.value]
        role = result_state.role
        messages = list(result_state.history)
        if role == "critic":
            messages.append(
                {
                    "role": "system",
                    "content": "You are the frozen Critic. Analyze the current answer and identify concrete defects.",
                }
            )
            messages.append({"role": "user", "content": f"Current answer:\n{result_state.current_answer}"})
        elif role == "refiner":
            messages.append(
                {
                    "role": "system",
                    "content": "You are the frozen Refiner. Return a corrected final answer using the critique.",
                }
            )
        else:
            raise ValueError(f"Unknown role: {role}")
        completion = self.backend.complete(messages, cap, role)
        prompt_tokens, completion_tokens, total_tokens = self._validated_usage(
            completion, role
        )
        if completion_tokens > cap:
            raise RuntimeError(
                f"Backend reported {completion_tokens} completion tokens above cap {cap}"
            )
        result_state.usage.add(
            completion_tokens,
            1,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        if (
            result_state.usage.extra_completion_tokens > budget.extra_tokens
            or result_state.usage.extra_calls > budget.extra_calls
        ):
            raise RuntimeError("Budget invariant violated after backend call")
        result_state.history.append({"role": "assistant", "name": role, "content": completion.content})
        if role == "refiner":
            result_state.current_answer = completion.content
        self._advance(result_state)
        return result_state

    def _advance(self, state: AgentState) -> None:
        state.collaboration_steps += 1
        if state.role == "critic":
            state.role = "refiner"
        else:
            state.role = "critic"
            state.round_index += 1
        if state.collaboration_steps >= self.config.max_collaboration_steps:
            state.terminated = True
            state.termination_reason = "max_collaboration_steps"


class StaticReferencePolicy:
    def __init__(self, engine: CollaborationEngine, preferred_action: str) -> None:
        self.engine = engine
        self.preferred_action = Action(preferred_action)

    def choose(self, state: AgentState, budget: BudgetLimit) -> Action:
        mask = self.engine.action_mask(state, budget)
        if mask[self.preferred_action.value]:
            return self.preferred_action
        for candidate in (Action.MEDIUM, Action.SHORT):
            if mask[candidate.value]:
                return candidate
        return Action.STOP

    def run(self, initial: AgentState, budget: BudgetLimit) -> AgentState:
        state = initial.clone()
        while not state.terminated:
            state = self.engine.execute_action(state, self.choose(state, budget), budget)
        return state
