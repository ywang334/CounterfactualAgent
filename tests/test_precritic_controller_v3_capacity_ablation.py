from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import hierarchical_control.cli as cli
import hierarchical_control.precritic_controller_v3_capacity_ablation as ablation
from hierarchical_control.precritic_controller_v1 import ALL_SEEDS, LABELS
from hierarchical_control.precritic_controller_v3 import (
    EMBEDDING_DIM,
    FIELD_TYPES,
    STRUCTURED_STATE_FEATURES,
    PreCriticV3Batch,
    controller_parameter_counts,
)
from hierarchical_control.precritic_controller_v3_capacity_ablation import (
    CapacityAblationController,
    EXPECTED_PARAMETER_COUNTS,
    VARIANT_ORDER,
    _cache_splits,
    _load_stage,
    _stage_contract,
    _atomic_torch,
    budget_curve_from_oof,
    capacity_contract,
    capacity_spec,
    validate_frozen_fold_manifest,
)
from hierarchical_control.precritic_controller_v3_training import (
    loss_balance,
    v3_training_loss,
)


def _batch(samples: int = 4, sequence: int = 10) -> PreCriticV3Batch:
    return PreCriticV3Batch(
        text_embeddings=torch.randn(samples, sequence, EMBEDDING_DIM),
        type_ids=torch.tensor(
            [[min(index, len(FIELD_TYPES) - 1) for index in range(sequence)]] * samples,
            dtype=torch.long,
        ),
        position_ids=torch.arange(sequence).repeat(samples, 1),
        padding_mask=torch.zeros(samples, sequence, dtype=torch.bool),
        structured_state=torch.zeros(samples, len(STRUCTURED_STATE_FEATURES)),
        state_positions=torch.full((samples,), sequence - 1, dtype=torch.long),
        field_order=tuple(tuple() for _ in range(samples)),
        solver_chunk_counts=(1,) * samples,
        solver_source_token_counts=(1,) * samples,
        solver_chunk_token_counts=((1,),) * samples,
        selected_answers=("A",) * samples,
        parse_statuses=("both_parsed_agree",) * samples,
    )


def test_capacity_parameter_counts_are_exact_and_only_capacity_changes() -> None:
    contract = capacity_contract()
    assert contract["only_capacity_changes"] is True
    assert contract["shared_invariants"]["cost_head"] is False
    for key in VARIANT_ORDER:
        spec = capacity_spec(key)
        counts = controller_parameter_counts(CapacityAblationController(spec))
        assert counts == {
            "total": EXPECTED_PARAMETER_COUNTS[key],
            "trainable": EXPECTED_PARAMETER_COUNTS[key],
            "frozen": 0,
        }
    tiny = contract["variants"]["tiny"]
    maas = contract["variants"]["maas"]
    assert tiny["d_model"] == maas["d_model"] == 64
    assert tiny["nhead"] == maas["nhead"] == 4
    assert tiny["dim_feedforward"] == maas["dim_feedforward"] == 256
    assert (tiny["num_layers"], maas["num_layers"]) == (1, 2)


def test_capacity_forward_uses_identical_factorized_loss_contract() -> None:
    labels = list(LABELS)
    balance = loss_balance(labels)
    for key in VARIANT_ORDER:
        outputs = CapacityAblationController(capacity_spec(key))(_batch())
        losses = v3_training_loss(outputs, labels, balance)
        assert torch.allclose(
            losses["total"],
            losses["solver_error"]
            + losses["critic_fix"]
            + losses["critic_harm"]
            + 0.25 * losses["auxiliary"],
        )
        assert set(outputs) >= {
            "p_help", "p_harm", "gate_score", "transition_aux_probabilities"
        }
        assert torch.allclose(outputs["gate_score"], outputs["p_help"] - outputs["p_harm"])


def test_frozen_primary_fold_manifest_is_reused_exactly() -> None:
    summary = json.load(open("artifacts/precritic_controller_v3/summary.json"))
    historical = summary["primary_seed_metrics"]["oof"]["fold_manifest"]
    labels = [json.loads(line)["label"] for line in open(
        "artifacts/precritic_training_1000/training_examples.jsonl"
    )]
    validated = validate_frozen_fold_manifest(historical, labels)
    assert validated == [
        {
            "fold": int(fold["fold"]),
            "train": list(fold["train"]),
            "validation": list(fold["validation"]),
        }
        for fold in historical
    ]
    assert sorted(index for fold in validated for index in fold["validation"]) == list(range(1000))


def test_feature_cache_is_loaded_without_encoder_or_embedding_forward(monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("encoder or embedding forward initialized")

    monkeypatch.setattr(ablation, "torch", ablation.torch)
    path = ablation.DEFAULT_CONTROLLER_V3_DIR / "feature_cache.pt"
    before = path.stat().st_mtime_ns
    training, validation, metadata = _cache_splits(path, 1000, 100)
    assert training.text_embeddings.shape[0] == 1000
    assert validation.text_embeddings.shape[0] == 100
    assert metadata["embedding_encode_calls"] == 1
    assert path.stat().st_mtime_ns == before
    assert "LocalMiniLM" not in ablation.__dict__


def test_budget_curves_are_oof_derived_and_diagnostic(monkeypatch) -> None:
    labels = list(LABELS) * 2
    scores = [0.8, 0.7, 0.9, 0.1, 0.6, 0.5, 0.4, 0.3]
    costs = [
        {
            "solver": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "calls": 1, "latency_seconds": 0.1},
            "critic": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "calls": 1, "latency_seconds": 0.1},
        }
        for _ in labels
    ]
    monkeypatch.setattr(
        ablation,
        "evaluate_validation_policy",
        lambda examples, decisions: {"samples": len(examples), "critic_calls": sum(decisions)},
    )
    oof, validation = budget_curve_from_oof(
        scores, labels, scores, [object()] * len(labels), costs
    )
    assert len(oof) == len(validation) == 6
    assert [row["target_budget_rate"] for row in oof] == [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    assert all(row["diagnostic_only"] and not row["operating_point_selected"] for row in oof + validation)


def test_atomic_stage_checkpoint_requires_exact_resume_contract(tmp_path) -> None:
    spec = capacity_spec("tiny")
    contract = _stage_contract(spec=spec, cache_sha="cache", fold_sha="fold", stage="fold_0")
    path = tmp_path / "fold_0.pt"
    _atomic_torch(path, {"complete": True, "contract": contract, "value": 1})
    assert _load_stage(path, contract)["value"] == 1
    changed = dict(contract)
    changed["feature_cache_sha256"] = "changed"
    with pytest.raises(ValueError, match="contract changed"):
        _load_stage(path, changed)


def test_capacity_cli_is_fixed_primary_only_and_never_constructs_backend(monkeypatch, tmp_path) -> None:
    received = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("offline capacity ablation initialized backend/encoder")

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden)
    monkeypatch.setattr(cli, "MockBackend", forbidden)
    monkeypatch.setattr(cli, "MiniLMEncoder", forbidden)
    monkeypatch.setattr(cli, "run_capacity_ablation", lambda **kwargs: received.update(kwargs) or {"pilot": True})
    output = tmp_path / "capacity"
    assert cli.main([
        "ablate-precritic-controller-v3-capacity",
        "--output-dir",
        str(output),
    ]) == 0
    assert received["resume"] is False
    assert received["output_dir"] == str(output)
    args = cli.build_parser().parse_args(["ablate-precritic-controller-v3-capacity"])
    for forbidden_name in ("seed", "epochs", "learning_rate", "batch_size", "backend"):
        assert not hasattr(args, forbidden_name)
    assert ALL_SEEDS[0] == 20260816


def test_final_test_boundary_is_manifest_only() -> None:
    source = open(ablation.__file__, encoding="utf-8").read()
    assert "verify_sealed_final_manifest" in source
    assert "final_test_examples_read\": False" in source
    assert "load_logiqa" not in source

