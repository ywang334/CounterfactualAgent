from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from hierarchical_control.cli import build_parser, main
from hierarchical_control.precritic_controller_v1 import (
    load_training_examples,
    load_validation_examples,
)
from hierarchical_control.precritic_controller_v2 import factorized_targets
from hierarchical_control.precritic_controller_v3 import (
    EMBEDDING_DIM,
    STRUCTURED_STATE_FEATURES,
    build_v3_feature_batch,
)
from hierarchical_control.precritic_controller_v3_training import (
    CachedFeatureSplit,
    _cached_split,
    apply_state_normalization,
    build_feature_cache_payload,
    fit_state_normalization,
    loss_balance,
    train_v3_model,
    v3_training_loss,
)
from tests.test_precritic_controller_v3 import TinyFieldEncoder, _model_input


def _cached_features(count: int = 8) -> CachedFeatureSplit:
    encoder = TinyFieldEncoder(max_seq_length=5)
    batch = build_v3_feature_batch(
        [_model_input(2 + index % 5) for index in range(count)], encoder
    )
    return CachedFeatureSplit(
        sample_hashes=tuple(f"input-{index}" for index in range(count)),
        content_sha256=tuple(f"content-{index}" for index in range(count)),
        text_embeddings=batch.text_embeddings,
        type_ids=batch.type_ids,
        position_ids=batch.position_ids,
        padding_mask=batch.padding_mask,
        structured_state=batch.structured_state,
        state_positions=batch.state_positions,
    )


def test_feature_cache_encodes_combined_splits_once_and_contains_no_leakage(tmp_path):
    encoder = TinyFieldEncoder(max_seq_length=5)
    inputs = [_model_input(2), _model_input(7)]
    batch = build_v3_feature_batch(inputs, encoder)
    assert encoder.embedding_forward_calls == 1
    training = SimpleNamespace(sample_id="content-train", model_input=inputs[0])
    validation = SimpleNamespace(model_input=inputs[1])
    payload = build_feature_cache_payload(
        batch=batch,
        training_examples=[training],
        validation_examples=[validation],
        metadata={
            "feature_cache": True,
            "data_sha256": "data",
            "encoder_sha256": "encoder",
            "schema_sha256": "schema",
            "code_sha256": "code",
        },
    )
    assert set(payload["training"]) == {
        "sample_hashes",
        "content_sha256",
        "text_embeddings",
        "type_ids",
        "position_ids",
        "padding_mask",
        "structured_state",
        "state_positions",
    }
    forbidden = {
        "gold", "label", "critic", "refiner", "outcome", "correct",
        "continuation", "raw_output",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert not ({str(key).casefold() for key in value} & forbidden)
            for child in value.values():
                inspect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                inspect(child)

    inspect(payload)
    path = tmp_path / "feature_cache.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    cached_train = _cached_split(loaded["training"], 1)
    cached_validation = _cached_split(loaded["validation"], 1)
    assert cached_train.text_embeddings.shape[0] == 1
    assert cached_validation.position_ids.shape == cached_validation.padding_mask.shape


def test_frozen_training_and_validation_whitelist_schemas_build_together_once():
    training, manifest = load_training_examples(
        "artifacts/precritic_training_1000/training_examples.jsonl",
        "artifacts/precritic_training_1000/manifest.json",
    )
    validation, _ = load_validation_examples(
        "artifacts/logiqa_policy_validation_100/predictions.jsonl",
        manifest,
        {example.sample_id for example in training},
    )
    encoder = TinyFieldEncoder(max_seq_length=10_000)
    batch = build_v3_feature_batch(
        [example.model_input for example in training]
        + [example.model_input for example in validation],
        encoder,
    )
    assert batch.text_embeddings.shape[0] == 1100
    assert encoder.embedding_forward_calls == 1
    assert max(batch.solver_chunk_counts) == 1


def test_fold_normalization_uses_only_training_indices_and_preserves_categories():
    state = torch.zeros((6, len(STRUCTURED_STATE_FEATURES)), dtype=torch.float32)
    state[:4, :8] = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    state[4:, :8] = 10_000.0
    state[:, 8] = 1.0
    state[:, 13] = 1.0
    normalization = fit_state_normalization(state, [0, 1, 2, 3])
    normalized = apply_state_normalization(state, normalization)
    assert torch.allclose(normalized[:4, :8].mean(0), torch.zeros(8), atol=1e-6)
    assert torch.allclose(
        normalized[:4, :8].std(0, unbiased=False), torch.ones(8), atol=1e-6
    )
    assert torch.equal(normalized[:, 8:], state[:, 8:])
    assert torch.all(normalization.mean < 100.0)  # held-out large values were excluded


def test_factorized_loss_masks_and_auxiliary_weight_are_strict():
    labels = [
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    ] * 2
    balance = loss_balance(labels)
    error = torch.zeros(8, requires_grad=True)
    fix = torch.zeros(8, requires_grad=True)
    harm = torch.zeros(8, requires_grad=True)
    auxiliary = torch.zeros((8, 4), requires_grad=True)
    losses = v3_training_loss(
        {
            "solver_error_logits": error,
            "critic_fix_logits": fix,
            "critic_harm_logits": harm,
            "transition_logits": auxiliary,
        },
        labels,
        balance,
    )
    assert torch.allclose(
        losses["total"],
        losses["solver_error"]
        + losses["critic_fix"]
        + losses["critic_harm"]
        + 0.25 * losses["auxiliary"],
    )
    losses["total"].backward()
    targets = factorized_targets(labels)
    assert torch.all(fix.grad[~targets["critic_fix_mask"]] == 0)
    assert torch.all(harm.grad[~targets["critic_harm_mask"]] == 0)
    assert torch.all(error.grad != 0)
    assert auxiliary.grad is not None and torch.any(auxiliary.grad != 0)


def test_fixed_seed_training_replay_is_deterministic_on_cpu():
    features = _cached_features()
    labels = [
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    ] * 2
    normalization = fit_state_normalization(features.structured_state, range(8))
    state = apply_state_normalization(features.structured_state, normalization)
    first, first_metrics = train_v3_model(
        features=features,
        normalized_state=state,
        labels=labels,
        training_indices=range(8),
        seed=1234,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
    )
    second, second_metrics = train_v3_model(
        features=features,
        normalized_state=state,
        labels=labels,
        training_indices=range(8),
        seed=1234,
        device=torch.device("cpu"),
        epochs=2,
        batch_size=4,
    )
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), name
    assert first_metrics == second_metrics


def test_train_v3_cli_has_fixed_protocol_and_never_initializes_backend(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline v3 training must not initialize a backend")

    received = {}

    def fake_train(**kwargs):
        received.update(kwargs)
        return {
            "controller_v3": True,
            "controller_trained": True,
            "model_calls": 0,
            "final_test_evaluated": False,
            "deployable": False,
        }

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden)
    monkeypatch.setattr(cli, "MockBackend", forbidden)
    monkeypatch.setattr(cli, "train_precritic_controller_v3", fake_train)
    output = tmp_path / "v3"
    assert main(["train-precritic-controller-v3", "--output-dir", str(output)]) == 0
    assert received["output_dir"] == str(output)
    assert "backend" not in received
    args = build_parser().parse_args(["train-precritic-controller-v3"])
    for forbidden_option in ("seed", "epochs", "learning_rate", "batch_size"):
        assert not hasattr(args, forbidden_option)
