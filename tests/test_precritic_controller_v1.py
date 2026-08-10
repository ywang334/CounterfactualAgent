from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from hierarchical_control.cli import build_parser, main
from hierarchical_control.precritic_controller_v1 import (
    ALL_SEEDS,
    BUDGET_RATES,
    LABELS,
    PRIMARY_SEED,
    STABILITY_SEEDS,
    gated_decisions,
    load_training_examples,
    masked_huber_cost_loss,
    oof_budget_thresholds_v1,
    oof_controller_predictions,
    select_oof_threshold_v1,
    verify_sealed_final_manifest,
)


def _write_training_fixture(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, str]:
    training = tmp_path / "training.jsonl"
    training.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(training.read_bytes()).hexdigest()
    labels = Counter(row["label"] for row in rows)
    available = sum(row["cost_available"] for row in rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "precritic_training_protocol": True,
                "training_examples_sha256": digest,
                "model_calls": 0,
                "controller_trained": False,
                "samples": len(rows),
                "label_counts": {label: labels.get(label, 0) for label in LABELS},
                "cost_targets": {"available_samples": available},
            }
        ),
        encoding="utf-8",
    )
    return training, manifest, digest


def test_training_loader_enforces_whitelist_and_blocks_gold_continuation_leak(tmp_path):
    source_rows = [
        json.loads(line)
        for line in Path(
            "artifacts/precritic_training_1000/training_examples.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    rows = [source_rows[0], source_rows[200]]
    training, manifest, digest = _write_training_fixture(tmp_path, rows)
    examples, _ = load_training_examples(
        training, manifest, expected_sha256=digest, expected_samples=2
    )
    assert len(examples) == 2
    assert {example.cost_available for example in examples} == {False, True}

    leaked = json.loads(json.dumps(rows))
    leaked[0]["model_input"]["gold"] = "A"
    leaked[0]["model_input"]["critic"] = "CONTINUATION_SENTINEL"
    training, manifest, digest = _write_training_fixture(tmp_path, leaked)
    with pytest.raises(ValueError, match="non-whitelist|Forbidden"):
        load_training_examples(
            training, manifest, expected_sha256=digest, expected_samples=2
        )


def test_masked_cost_loss_ignores_all_unavailable_targets():
    predictions = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    available = torch.tensor([True, False, True])
    first = masked_huber_cost_loss(
        predictions, torch.tensor([1.5, -9999.0, 2.5]), available
    )
    second = masked_huber_cost_loss(
        predictions, torch.tensor([1.5, 999999.0, 2.5]), available
    )
    assert torch.equal(first, second)
    first.backward()
    assert predictions.grad is not None
    assert predictions.grad[1] == 0


def test_oof_is_deterministic_and_partitions_each_sample_once():
    labels = list(LABELS) * 5
    embeddings = torch.arange(20 * 6, dtype=torch.float32).reshape(20, 6) / 100
    numeric = torch.arange(20 * 3, dtype=torch.float32).reshape(20, 3) / 10
    cost_targets = torch.linspace(4.0, 6.0, 20)
    cost_available = torch.tensor([index % 5 != 0 for index in range(20)])
    first = oof_controller_predictions(
        embeddings,
        numeric,
        labels,
        cost_targets,
        cost_available,
        seed=PRIMARY_SEED,
        epochs=2,
    )
    second = oof_controller_predictions(
        embeddings,
        numeric,
        labels,
        cost_targets,
        cost_available,
        seed=PRIMARY_SEED,
        epochs=2,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[2:] == second[2:]
    held_out = [index for fold in first[2] for index in fold["validation"]]
    assert sorted(held_out) == list(range(20))
    assert len(held_out) == len(set(held_out))


def test_fixed_primary_seed_and_oof_only_budget_thresholds():
    assert PRIMARY_SEED == 20260816
    assert STABILITY_SEEDS == (20260817, 20260818, 20260819, 20260820)
    assert ALL_SEEDS == (20260816, 20260817, 20260818, 20260819, 20260820)
    args = build_parser().parse_args(["train-precritic-controller-v1"])
    assert not hasattr(args, "seed")
    assert not hasattr(args, "epochs")

    scores = [float(index) / 100 for index in range(100)]
    labels = list(LABELS) * 25
    threshold = select_oof_threshold_v1(scores, labels)
    assert threshold["validation_used_for_selection"] is False
    assert threshold["selection_source"].startswith("training_1000_")
    points = oof_budget_thresholds_v1(scores)
    assert tuple(point["target_budget_rate"] for point in points) == BUDGET_RATES
    assert [point["oof_critic_calls"] for point in points] == [5, 10, 20, 30, 50, 100]
    assert all(point["validation_used_for_threshold"] is False for point in points)
    # Validation labels are not accepted by either threshold API; applying a
    # frozen threshold is a pure score comparison.
    assert gated_decisions([0.9, -0.1], threshold["threshold"]) == [
        0.9 >= threshold["threshold"],
        -0.1 >= threshold["threshold"],
    ]


def test_final_test_guard_checks_sha_and_seal_without_opening_data(tmp_path):
    final = tmp_path / "split_manifest.json"
    final.write_text(
        json.dumps(
            {
                "final_test": True,
                "sealed": True,
                "never_evaluated": True,
                "model_calls": 0,
                "split_sha256": "split-sha",
                "data_path": "/must/not/be/opened/dev.txt",
                "selected_samples": [{"opaque": "not inspected"}],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(final.read_bytes()).hexdigest()
    guard = verify_sealed_final_manifest(
        final,
        {"final_test": {"manifest_sha256": digest, "split_sha256": "split-sha"}},
    )
    assert guard["sealed"] is True and guard["final_test_evaluated"] is False
    assert "selected_samples" not in guard and "data_path" not in guard

    payload = json.loads(final.read_text())
    payload["sealed"] = False
    final.write_text(json.dumps(payload), encoding="utf-8")
    new_digest = hashlib.sha256(final.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not sealed"):
        verify_sealed_final_manifest(
            final,
            {"final_test": {"manifest_sha256": new_digest, "split_sha256": "split-sha"}},
        )


def test_controller_cli_never_initializes_backend(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("Controller v1 training must not initialize backend")

    received = {}

    def fake_train(**kwargs):
        received.update(kwargs)
        return {
            "controller_trained": True,
            "development_validation": True,
            "final_test_evaluated": False,
            "model_calls": 0,
        }

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden_backend)
    monkeypatch.setattr(cli, "MockBackend", forbidden_backend)
    monkeypatch.setattr(cli, "train_precritic_controller_v1", fake_train)
    assert main(
        [
            "train-precritic-controller-v1",
            "--output-dir",
            str(tmp_path / "controller"),
        ]
    ) == 0
    assert received["output_dir"] == str(tmp_path / "controller")
    assert "seed" not in received and "epochs" not in received
