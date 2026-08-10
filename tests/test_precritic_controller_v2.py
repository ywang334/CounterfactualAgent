from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from hierarchical_control.cli import build_parser, main
from hierarchical_control.precritic_controller_v1 import (
    ALL_SEEDS,
    PRIMARY_SEED,
    STABILITY_SEEDS,
    verify_sealed_final_manifest,
)
from hierarchical_control.precritic_controller_v2 import (
    MIN_COST_IMPROVEMENT,
    assess_cost_model,
    cost_constant_baselines,
    factorized_probabilities,
    factorized_targets,
    masked_balanced_bce,
)


def test_factorized_targets_have_strict_frozen_condition_counts():
    rows = [
        json.loads(line)
        for line in Path(
            "artifacts/precritic_training_1000/training_examples.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = [row["label"] for row in rows]
    targets = factorized_targets(labels)
    assert len(labels) == 1000
    assert int(targets["solver_error"].sum()) == 286
    assert int(targets["critic_fix_mask"].sum()) == 286
    assert int(targets["critic_fix"].sum()) == 64
    assert int(targets["critic_harm_mask"].sum()) == 714
    assert int(targets["critic_harm"].sum()) == 79
    assert not torch.any(
        targets["critic_fix_mask"] & targets["critic_harm_mask"]
    )
    assert torch.all(
        targets["critic_fix_mask"] | targets["critic_harm_mask"]
    )


def test_masked_balanced_bce_has_no_gradient_outside_condition():
    logits = torch.tensor([0.1, -0.2, 100.0, -100.0], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.tensor([True, True, False, False])
    loss, metrics = masked_balanced_bce(logits, targets, mask)
    loss.backward()
    assert metrics["active_samples"] == 2
    assert metrics["positive_samples"] == 1
    assert metrics["negative_samples"] == 1
    assert logits.grad is not None
    assert logits.grad[0] != 0 and logits.grad[1] != 0
    assert logits.grad[2] == 0 and logits.grad[3] == 0


def test_probability_factorization_and_gate_score_are_exact():
    logits = torch.zeros(3)
    result = factorized_probabilities(logits, logits, logits)
    assert torch.allclose(result["p_error"], torch.full((3,), 0.5))
    assert torch.allclose(result["p_help"], torch.full((3,), 0.25))
    assert torch.allclose(result["p_harm"], torch.full((3,), 0.25))
    assert torch.equal(result["gate_score"], torch.zeros(3))
    assert torch.allclose(result["four_class"], torch.full((3, 4), 0.25))
    assert torch.allclose(result["four_class"].sum(dim=-1), torch.ones(3))


def test_cost_baselines_and_safe_disable_use_training_median():
    actual = torch.tensor([100.0, 200.0, 300.0])
    targets = torch.log1p(actual)
    available = torch.ones(3, dtype=torch.bool)
    baselines = cost_constant_baselines(targets, available)
    assert set(baselines["baselines"]) == {
        "zero",
        "training_mean",
        "training_median",
    }
    assert baselines["fixed_training_median_total_tokens"] == pytest.approx(200.0)
    disabled = assess_cost_model(
        torch.zeros_like(targets), targets, available, baselines
    )
    assert disabled["cost_model_enabled"] is False
    assert disabled["effective_cost_source"] == "fixed_training_median_total_tokens"
    assert disabled["hard_budget_uses_cost_prediction"] is False
    assert disabled["hard_budget_guard"] == {
        "critic_completion_token_cap": 512,
        "critic_call_cap": 1,
    }
    enabled = assess_cost_model(targets, targets, available, baselines)
    assert enabled["cost_model_enabled"] is True
    assert enabled["actual_relative_improvement"] >= MIN_COST_IMPROVEMENT


def test_v2_final_test_guard_reads_only_sealed_manifest(tmp_path):
    manifest = tmp_path / "split_manifest.json"
    payload = {
        "final_test": True,
        "sealed": True,
        "never_evaluated": True,
        "model_calls": 0,
        "split_sha256": "frozen-split",
        "data_path": "/must/not/be/read/final-test.txt",
        "selected_samples": [{"opaque": "must not be inspected"}],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    guard = verify_sealed_final_manifest(
        manifest,
        {
            "final_test": {
                "manifest_sha256": digest,
                "split_sha256": "frozen-split",
            }
        },
    )
    assert guard["sealed"] is True
    assert guard["never_evaluated"] is True
    assert "data_path" not in guard and "selected_samples" not in guard


def test_v2_cli_has_fixed_protocol_and_never_initializes_backend(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Factorized v2 must not initialize a backend")

    received = {}

    def fake_train(**kwargs):
        received.update(kwargs)
        return {
            "controller_v2_factorized": True,
            "final_test_evaluated": False,
            "deployable": False,
            "model_calls": 0,
        }

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden)
    monkeypatch.setattr(cli, "MockBackend", forbidden)
    monkeypatch.setattr(cli, "train_precritic_controller_v2", fake_train)
    output = tmp_path / "v2"
    assert main(
        ["train-precritic-controller-v2", "--output-dir", str(output)]
    ) == 0
    assert received["output_dir"] == str(output)
    assert "seed" not in received and "epochs" not in received
    args = build_parser().parse_args(["train-precritic-controller-v2"])
    assert not hasattr(args, "seed")
    assert not hasattr(args, "learning_rate")
    assert PRIMARY_SEED == 20260816
    assert STABILITY_SEEDS == (20260817, 20260818, 20260819, 20260820)
    assert ALL_SEEDS == (20260816, 20260817, 20260818, 20260819, 20260820)

