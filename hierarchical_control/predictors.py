from __future__ import annotations

from pathlib import Path

import torch

from .config import ACTION_LABELS, BUDGET_LABELS, BudgetLimit, ExperimentConfig
from .encoders import TextEncoder
from .models import ActionController, BudgetAllocator, action_numeric_features, budget_features, state_text
from .types import AgentState


def _load(path: str | Path) -> dict:
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class BudgetModelPredictor:
    def __init__(self, checkpoint: str | Path, encoder: TextEncoder, config: ExperimentConfig) -> None:
        payload = _load(checkpoint)
        if payload["embedding_dim"] != encoder.dimension:
            raise ValueError("Checkpoint and encoder embedding dimensions differ")
        if payload["encoder_name"] != encoder.name:
            raise ValueError("Checkpoint and encoder names differ")
        self.model = BudgetAllocator(payload["embedding_dim"], payload["hidden_dim"])
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.encoder = encoder
        self.config = config

    def __call__(self, query: str, max_budget: BudgetLimit) -> str:
        query_embedding = self.encoder.encode([query])
        numeric = torch.tensor([budget_features(max_budget, self.config)], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(query_embedding, numeric)[0]
        for index, tier in enumerate(BUDGET_LABELS):
            limit = self.config.tier(tier)
            if limit.extra_tokens > max_budget.extra_tokens or limit.extra_calls > max_budget.extra_calls:
                logits[index] = -torch.inf
        if not torch.isfinite(logits).any():
            return "ZERO"
        return BUDGET_LABELS[int(logits.argmax())]


class ActionModelPredictor:
    def __init__(self, checkpoint: str | Path, encoder: TextEncoder, config: ExperimentConfig) -> None:
        payload = _load(checkpoint)
        if payload["embedding_dim"] != encoder.dimension:
            raise ValueError("Checkpoint and encoder embedding dimensions differ")
        if payload["encoder_name"] != encoder.name:
            raise ValueError("Checkpoint and encoder names differ")
        self.model = ActionController(
            payload["embedding_dim"],
            numeric_dim=payload.get("numeric_dim", 10),
            hidden_dim=payload["hidden_dim"],
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()
        self.encoder = encoder
        self.config = config

    def __call__(
        self, state: AgentState, allocated: BudgetLimit, action_mask: dict[str, bool]
    ) -> str:
        remaining = BudgetLimit(
            max(0, allocated.extra_tokens - state.usage.extra_tokens),
            max(0, allocated.extra_calls - state.usage.extra_calls),
        )
        query_embedding = self.encoder.encode([state.query])
        state_embedding = self.encoder.encode([state_text(state)])
        numeric = torch.tensor(
            [action_numeric_features(state, allocated, remaining, self.config)], dtype=torch.float32
        )
        with torch.no_grad():
            logits = self.model(query_embedding, state_embedding, numeric)["logits"][0]
        for index, action in enumerate(ACTION_LABELS):
            if not action_mask[action]:
                logits[index] = -torch.inf
        return ACTION_LABELS[int(logits.argmax())]
