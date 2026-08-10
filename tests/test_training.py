from __future__ import annotations

import torch

from hierarchical_control.backend import MockBackend
from hierarchical_control.collection import collect_action_labels, collect_budget_labels
from hierarchical_control.config import ExperimentConfig
from hierarchical_control.encoders import HashingEncoder
from hierarchical_control.engine import CollaborationEngine
from hierarchical_control.evaluator import MockEvaluator
from hierarchical_control.training import train_action_controller, train_budget_allocator


def test_both_controllers_train_backward_and_save(tmp_path):
    config = ExperimentConfig()
    examples = [
        {"id": f"q{i}", "query": f"toy query {i}", "difficulty": i} for i in range(4)
    ]
    budget_engine = CollaborationEngine(MockBackend(), config)
    budget_data = collect_budget_labels(examples, budget_engine, MockEvaluator()).training
    action_engine = CollaborationEngine(MockBackend(), config)
    action_data = collect_action_labels(examples, action_engine, MockEvaluator()).training
    encoder = HashingEncoder(32)
    budget_path = tmp_path / "budget_allocator.pt"
    action_path = tmp_path / "action_controller.pt"
    budget = train_budget_allocator(
        budget_data, encoder, config, budget_path, epochs=2, batch_size=4, hidden_dim=16
    )
    action = train_action_controller(
        action_data, encoder, config, action_path, epochs=2, batch_size=8, hidden_dim=24
    )
    assert budget_path.is_file() and action_path.is_file()
    assert any(parameter.grad is not None for parameter in budget.model.parameters())
    assert any(parameter.grad is not None for parameter in action.model.parameters())
    assert torch.load(budget_path, map_location="cpu", weights_only=False)["encoder_name"] == encoder.name
    assert torch.load(action_path, map_location="cpu", weights_only=False)["encoder_name"] == encoder.name
