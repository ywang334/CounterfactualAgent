from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import ACTION_LABELS, BUDGET_LABELS, BudgetLimit, ExperimentConfig
from .encoders import TextEncoder
from .models import (
    ActionController,
    BudgetAllocator,
    action_numeric_features,
    budget_features,
    checkpoint_payload,
    state_text,
)
from .types import AgentState


@dataclass
class TrainResult:
    model: torch.nn.Module
    metrics: dict[str, Any]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _batches(size: int, batch_size: int, generator: torch.Generator):
    indices = torch.randperm(size, generator=generator)
    for start in range(0, size, batch_size):
        yield indices[start : start + batch_size]


def train_budget_allocator(
    records: list[dict[str, Any]],
    encoder: TextEncoder,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    epochs: int = 40,
    learning_rate: float = 3e-3,
    batch_size: int = 16,
    hidden_dim: int = 96,
) -> TrainResult:
    records = [record for record in records if not record.get("unsolved") and record.get("budget_label")]
    if not records:
        raise ValueError("No solved budget training examples")
    _seed_everything(config.seed)
    queries = [str(record["query"]) for record in records]
    query_embeddings = encoder.encode(queries)
    max_budgets = torch.tensor(
        [budget_features(BudgetLimit(**record["max_budget"]), config) for record in records],
        dtype=torch.float32,
    )
    targets = torch.tensor([BUDGET_LABELS.index(record["budget_label"]) for record in records])
    model = BudgetAllocator(encoder.dimension, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    final_loss = 0.0
    model.train()
    for _ in range(epochs):
        for indices in _batches(len(records), batch_size, generator):
            logits = model(query_embeddings[indices], max_budgets[indices])
            loss = F.cross_entropy(logits, targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    model.eval()
    with torch.no_grad():
        logits = model(query_embeddings, max_budgets)
        accuracy = float((logits.argmax(dim=-1) == targets).float().mean())
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, encoder.name, config)
    payload["mock_only"] = bool(encoder.mock_only)
    torch.save(payload, path)
    return TrainResult(
        model,
        {
            "examples": len(records),
            "epochs": epochs,
            "final_loss": final_loss,
            "training_accuracy": accuracy,
            "encoder": encoder.name,
            "mock_only": bool(encoder.mock_only),
            "checkpoint": str(path),
        },
    )


def train_action_controller(
    records: list[dict[str, Any]],
    encoder: TextEncoder,
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    epochs: int = 50,
    learning_rate: float = 3e-3,
    batch_size: int = 16,
    hidden_dim: int = 128,
    auxiliary_weight: float = 0.2,
) -> TrainResult:
    if not records:
        raise ValueError("No action training examples")
    _seed_everything(config.seed + 1)
    states = [AgentState.from_dict(record["state"]) for record in records]
    query_embeddings = encoder.encode([str(record["query"]) for record in records])
    state_embeddings = encoder.encode([state_text(state) for state in states])
    numeric = torch.tensor(
        [
            action_numeric_features(
                state,
                BudgetLimit(**record["allocated_budget"]),
                BudgetLimit(**record["remaining_budget"]),
                config,
            )
            for state, record in zip(states, records)
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([ACTION_LABELS.index(record["action_label"]) for record in records])
    token_denominator = max(config.tier("HIGH").extra_tokens, 1)
    call_denominator = max(config.tier("HIGH").extra_calls, 1)
    auxiliary_targets = torch.tensor(
        [
            [
                float(record["target_quality"]),
                float(record["target_future_tokens"]) / token_denominator,
                float(record["target_future_calls"]) / call_denominator,
            ]
            for record in records
        ],
        dtype=torch.float32,
    )
    model = ActionController(encoder.dimension, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 1)
    final_loss = 0.0
    final_classification_loss = 0.0
    final_auxiliary_loss = 0.0
    model.train()
    for _ in range(epochs):
        for indices in _batches(len(records), batch_size, generator):
            output = model(query_embeddings[indices], state_embeddings[indices], numeric[indices])
            classification_loss = F.cross_entropy(output["logits"], targets[indices])
            predictions = torch.stack(
                [output["quality"], output["token_cost"], output["call_cost"]], dim=-1
            )
            auxiliary_loss = F.mse_loss(predictions, auxiliary_targets[indices])
            loss = classification_loss + auxiliary_weight * auxiliary_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
            final_classification_loss = float(classification_loss.detach())
            final_auxiliary_loss = float(auxiliary_loss.detach())
    model.eval()
    with torch.no_grad():
        output = model(query_embeddings, state_embeddings, numeric)
        accuracy = float((output["logits"].argmax(dim=-1) == targets).float().mean())
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, encoder.name, config)
    payload["numeric_dim"] = model.numeric_dim
    payload["auxiliary_weight"] = auxiliary_weight
    payload["mock_only"] = bool(encoder.mock_only)
    torch.save(payload, path)
    return TrainResult(
        model,
        {
            "examples": len(records),
            "epochs": epochs,
            "final_loss": final_loss,
            "classification_loss": final_classification_loss,
            "auxiliary_loss": final_auxiliary_loss,
            "training_accuracy": accuracy,
            "encoder": encoder.name,
            "mock_only": bool(encoder.mock_only),
            "checkpoint": str(path),
        },
    )
