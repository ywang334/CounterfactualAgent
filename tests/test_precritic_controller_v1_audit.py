from __future__ import annotations

import os
from pathlib import Path

import pytest

from hierarchical_control.cli import main
from hierarchical_control.precritic_controller_v1 import LABELS
from hierarchical_control.precritic_controller_v1_audit import (
    ARTIFACT_NAMES,
    _average_precision,
    _fold_manifest_audit,
    _head_metrics,
    _label_policy_metrics,
    _validation_policy_metrics,
    verify_artifacts,
    verify_run_status,
)
from hierarchical_control.precritic_probe import ProbeExample


def test_run_status_requires_successful_exited_process_and_reports_duration(tmp_path):
    prefix = tmp_path / "formal_run"
    pid = Path(f"{prefix}.pid")
    exitcode = Path(f"{prefix}.exitcode")
    log = Path(f"{prefix}.log")
    pid.write_text("99999999\n", encoding="utf-8")
    exitcode.write_text("0\n", encoding="utf-8")
    log.write_text("completed\n", encoding="utf-8")
    os.utime(pid, ns=(1_000_000_000, 1_000_000_000))
    os.utime(exitcode, ns=(4_500_000_000, 4_500_000_000))
    result = verify_run_status(prefix)
    assert result["successful"] is True
    assert result["process_exited"] is True
    assert result["wall_clock_seconds"] == 3.5

    exitcode.write_text("2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exit code 2"):
        verify_run_status(prefix)


def test_artifact_integrity_requires_six_nonempty_files_and_hashes(tmp_path):
    for index, name in enumerate(ARTIFACT_NAMES):
        (tmp_path / name).write_bytes(f"artifact-{index}".encode())
    result = verify_artifacts(tmp_path)
    assert set(result) == set(ARTIFACT_NAMES)
    assert all(item["nonempty"] and len(item["sha256"]) == 64 for item in result.values())
    (tmp_path / ARTIFACT_NAMES[0]).write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="Missing or empty"):
        verify_artifacts(tmp_path)


def test_oof_fold_audit_partitions_each_sample_once():
    labels = list(LABELS) * 5
    from hierarchical_control.precritic_controller_v1 import (
        PRIMARY_SEED,
        deterministic_stratified_folds,
    )

    folds = deterministic_stratified_folds(labels, 5, PRIMARY_SEED)
    fold_by_index = {
        index: fold["fold"] for fold in folds for index in fold["validation"]
    }
    rows = [{"oof_fold": fold_by_index[index]} for index in range(len(labels))]
    manifest, sizes = _fold_manifest_audit(labels, PRIMARY_SEED, rows)
    assert manifest == folds
    assert sum(sizes.values()) == len(labels)
    assert set(fold_by_index) == set(range(len(labels)))

    rows[0]["oof_fold"] = (rows[0]["oof_fold"] + 1) % 5
    with pytest.raises(ValueError, match="wrong OOF fold"):
        _fold_manifest_audit(labels, PRIMARY_SEED, rows)


def _probe_example(
    question_id: int,
    label: str,
    solver_answer: str,
    critic_answer: str,
    gold: str,
    stop_tokens: int,
    critic_tokens: int,
) -> ProbeExample:
    stop_prompt = stop_tokens - 1
    critic_prompt = critic_tokens - 2
    audit_case = {
        "strategy_costs": {
            "STOP": {
                "available": True,
                "prompt_tokens": stop_prompt,
                "completion_tokens": 1,
                "total_tokens": stop_tokens,
                "calls": 1,
                "latency_seconds": 1.0,
            },
            "CRITIC_ONLY": {
                "available": True,
                "prompt_tokens": critic_prompt,
                "completion_tokens": 2,
                "total_tokens": critic_tokens,
                "calls": 2,
                "latency_seconds": 3.0,
            },
        }
    }
    return ProbeExample(
        dataset="validation_100",
        question_id=question_id,
        gold=gold,
        label=label,
        model_input={},
        feature_text="",
        numeric=(),
        solver_answer=solver_answer,
        critic_only_answer=critic_answer,
        audit_case=audit_case,
    )


def test_offline_metric_recomputation_uses_exact_saved_costs():
    examples = [
        _probe_example(1, "wrong_to_correct", "A", "B", "B", 10, 30),
        _probe_example(2, "correct_to_wrong", "C", "D", "C", 20, 50),
    ]
    metric = _validation_policy_metrics(examples, [True, False])
    assert metric["accuracy"] == 1.0
    assert metric["corrected"] == 1 and metric["degraded"] == 0
    assert metric["critic_calls"] == 1
    assert metric["cost"]["total"]["total_tokens"] == 50
    assert metric["cost"]["total"]["calls"] == 3
    assert metric["cost"]["total"]["latency_seconds"] == 4.0
    assert metric["cost"]["estimated"] is False

    labels = ["wrong_to_correct", "correct_to_wrong"]
    assert _label_policy_metrics(labels, [True, False])["net_benefit"] == 1
    probabilities = [[0.0, 0.1, 0.9, 0.0], [0.0, 0.8, 0.2, 0.0]]
    heads = _head_metrics(labels, probabilities, [20.0, 30.0], [20.0, 50.0], [True, False])
    assert heads["critic_incremental_total_tokens_mae"] == 0.0
    assert heads["cost_mae_samples"] == 1
    assert _average_precision([True, False], [0.9, 0.2]) == 1.0


def test_audit_cli_never_initializes_backend_or_trains(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline audit must not initialize a backend or train")

    received = {}

    def fake_audit(**kwargs):
        received.update(kwargs)
        return {
            "offline_audit": True,
            "controller_retrained": False,
            "model_calls": 0,
            "final_test_evaluated": False,
            "deployable": False,
        }

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden)
    monkeypatch.setattr(cli, "MockBackend", forbidden)
    monkeypatch.setattr(cli, "train_precritic_controller_v1", forbidden)
    monkeypatch.setattr(cli, "run_precritic_controller_v1_audit", fake_audit)
    output = tmp_path / "audit"
    assert main(
        ["audit-precritic-controller-v1", "--output-dir", str(output)]
    ) == 0
    assert received["output_dir"] == str(output)
    assert not any(
        key in received for key in ("backend", "seed", "epochs", "threshold")
    )

