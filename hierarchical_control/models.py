from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import BudgetLimit, ExperimentConfig
from .types import AgentState


class BudgetAllocator(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 96, num_tiers: int = 4) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tiers),
        )

    def forward(self, query_embedding: torch.Tensor, max_budget_features: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([query_embedding, max_budget_features], dim=-1))


class ActionController(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        numeric_dim: int = 10,
        hidden_dim: int = 128,
        num_actions: int = 5,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.numeric_dim = numeric_dim
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim * 2 + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.auxiliary_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        query_embedding: torch.Tensor,
        state_embedding: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.trunk(torch.cat([query_embedding, state_embedding, numeric_features], dim=-1))
        auxiliary = self.auxiliary_head(hidden)
        return {
            "logits": self.action_head(hidden),
            "quality": auxiliary[:, 0],
            "token_cost": auxiliary[:, 1],
            "call_cost": auxiliary[:, 2],
        }


def _denominators(config: ExperimentConfig) -> tuple[float, float]:
    high = config.tier("HIGH")
    return float(max(high.extra_tokens, 1)), float(max(high.extra_calls, 1))


def budget_features(budget: BudgetLimit, config: ExperimentConfig) -> list[float]:
    token_denominator, call_denominator = _denominators(config)
    return [budget.extra_tokens / token_denominator, budget.extra_calls / call_denominator]


def state_text(state: AgentState) -> str:
    history = "\n".join(
        f"{message.get('name', message.get('role', 'unknown'))}: {message.get('content', '')}"
        for message in state.history
    )
    return f"role={state.role}\nround={state.round_index}\nanswer={state.current_answer}\nhistory:\n{history}"


def action_numeric_features(
    state: AgentState,
    allocated: BudgetLimit,
    remaining: BudgetLimit,
    config: ExperimentConfig,
) -> list[float]:
    token_denominator, call_denominator = _denominators(config)
    return [
        allocated.extra_tokens / token_denominator,
        allocated.extra_calls / call_denominator,
        remaining.extra_tokens / token_denominator,
        remaining.extra_calls / call_denominator,
        state.usage.extra_tokens / token_denominator,
        state.usage.extra_calls / call_denominator,
        state.round_index / max(config.max_collaboration_steps / 2.0, 1.0),
        state.collaboration_steps / max(float(config.max_collaboration_steps), 1.0),
        min(len(state.history), 32) / 32.0,
        float(state.role == "refiner"),
    ]


def checkpoint_payload(model: nn.Module, encoder_name: str, config: ExperimentConfig) -> dict[str, Any]:
    return {
        "model_class": type(model).__name__,
        "model_state_dict": model.state_dict(),
        "embedding_dim": getattr(model, "embedding_dim"),
        "hidden_dim": getattr(model, "hidden_dim"),
        "encoder_name": encoder_name,
        "experiment_config": config.to_dict(),
    }
