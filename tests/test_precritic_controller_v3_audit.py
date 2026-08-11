from __future__ import annotations

import math

import pytest

from hierarchical_control import cli
from hierarchical_control.precritic_controller_v1 import ALL_SEEDS, LABELS
from hierarchical_control.precritic_controller_v3_audit import (
    assert_replay_consistency,
    binary_calibration,
    diagnostic_metrics,
    multiclass_calibration,
    pearson_correlation,
    seed_stability,
    set_overlap,
    spearman_correlation,
    stratified_bootstrap,
)


def _probabilities(label: str, score: float) -> dict:
    factorized = {name: 0.05 for name in LABELS}
    factorized[label] = 0.85
    return {
        "solver_error": 0.8 if label.startswith("wrong_to_") else 0.2,
        "critic_fix_given_solver_error": 0.8 if label == "wrong_to_correct" else 0.2,
        "critic_harm_given_solver_correct": 0.8 if label == "correct_to_wrong" else 0.2,
        "helpful": 0.8 if label == "wrong_to_correct" else 0.1,
        "harmful": 0.8 if label == "correct_to_wrong" else 0.1,
        "factorized_four_class": factorized,
        "auxiliary_four_class": dict(factorized),
    }


def _row(index: int, label: str, score: float, seed: int = ALL_SEEDS[0]) -> dict:
    threshold = 0.5
    return {
        "sample_id": f"sample-{index:04d}",
        "identity": f"sample-{index:04d}",
        "seed": seed,
        "label": label,
        "gate_score": score,
        "development_threshold": threshold,
        "critic_called": score >= threshold,
        "source_dataset": "collection_200" if index < 200 else "collection_800",
        "oof_fold": index % 5,
        "probabilities": _probabilities(label, score),
    }


def test_binary_and_multiclass_calibration() -> None:
    binary = binary_calibration([False, True], [0.0, 1.0])
    assert binary["brier_score"] == 0.0
    assert binary["ece_10_bin"] == 0.0
    multiclass = multiclass_calibration(
        [LABELS[0], LABELS[1]],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    assert multiclass["brier_score"] == 0.0
    assert multiclass["ece_10_bin"] == 0.0
    with pytest.raises(ValueError, match="sum to one"):
        multiclass_calibration([LABELS[0]], [[0.2, 0.2, 0.2, 0.2]])


def test_diagnostic_metrics_has_condition_masks_and_calibration() -> None:
    labels = list(LABELS) * 3
    rows = [_row(index, label, index / 20.0) for index, label in enumerate(labels)]
    metrics = diagnostic_metrics(rows)
    assert metrics["critic_fix_given_solver_error"]["samples"] == 6
    assert metrics["critic_harm_given_solver_correct"]["samples"] == 6
    assert metrics["helpful"]["positive_samples"] == 3
    assert 0.0 <= metrics["factorized_four_class"]["ece_10_bin"] <= 1.0
    assert math.isclose(metrics["factorized_four_class"]["macro_f1"], 1.0)


def test_correlation_and_jaccard_helpers() -> None:
    assert pearson_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    overlap = set_overlap({"a", "b"}, {"b", "c"})
    assert overlap["jaccard"] == pytest.approx(1 / 3)
    assert overlap["overlap_coefficient"] == pytest.approx(0.5)
    assert set_overlap(set(), set())["jaccard"] == 1.0


def test_seed_stability_checks_alignment_and_top_sets() -> None:
    rows_by_seed = {}
    for seed_index, seed in enumerate(ALL_SEEDS):
        rows = []
        for index in range(1000):
            label = LABELS[index % len(LABELS)]
            score = (index + seed_index * 0.01) / 1001.0
            rows.append(_row(index, label, score, seed))
        rows_by_seed[seed] = rows
    result = seed_stability(rows_by_seed)
    assert len(result["pairs"]) == 10
    assert result["pairwise_summary"]["pearson"]["mean"] == pytest.approx(1.0)
    assert result["pairwise_summary"]["top_jaccard"]["10pct"]["mean"] == 1.0
    rows_by_seed[ALL_SEEDS[-1]] = rows_by_seed[ALL_SEEDS[-1]][:-1]
    with pytest.raises(ValueError, match="1000 unique"):
        seed_stability(rows_by_seed)


def test_stratified_bootstrap_is_deterministic_and_reports_oob() -> None:
    labels = list(LABELS) * 20
    scores = [
        0.9 if label == "wrong_to_correct" else 0.8 if label == "correct_to_wrong" else index / 1000
        for index, label in enumerate(labels)
    ]
    first = stratified_bootstrap(scores, labels, replicates=50, seed=20260822)
    second = stratified_bootstrap(scores, labels, replicates=50, seed=20260822)
    assert first == second
    assert first["records_sha256"] == second["records_sha256"]
    assert first["out_of_bag_threshold_transfer"]["samples_95_interval"]["p2_5"] > 0
    assert first["independent_test_performance"] is False


def test_replay_consistency_covers_probabilities_scores_and_actions() -> None:
    row = _row(1, "wrong_to_correct", 0.7)
    result = assert_replay_consistency([row], [dict(row)])
    assert result["consistent"] is True
    cross_device = dict(row)
    cross_device["probabilities"] = dict(row["probabilities"])
    cross_device["probabilities"]["solver_error"] += 5e-5
    result = assert_replay_consistency([row], [cross_device])
    assert result["maximum_probability_difference"] == pytest.approx(5e-5)
    changed = dict(row)
    changed["gate_score"] = 0.1
    with pytest.raises(ValueError, match="gate score"):
        assert_replay_consistency([row], [changed])


def test_audit_cli_does_not_construct_backend_optimizer_or_encoder(monkeypatch, tmp_path) -> None:
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden runtime component initialized")

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden)
    monkeypatch.setattr(cli, "MockBackend", forbidden)
    monkeypatch.setattr(cli, "MiniLMEncoder", forbidden)
    monkeypatch.setattr(cli, "run_precritic_controller_v3_generalization_audit", lambda **kwargs: calls.append(kwargs) or {"offline_audit": True})
    assert cli.main([
        "audit-precritic-controller-v3-generalization",
        "--output-dir",
        str(tmp_path / "audit"),
    ]) == 0
    assert len(calls) == 1

