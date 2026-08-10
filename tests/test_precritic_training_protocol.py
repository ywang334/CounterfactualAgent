from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hierarchical_control.cli import main
from hierarchical_control.logiqa_action_collection import content_sha256
from hierarchical_control.logiqa_pilot import LogiQAExample, load_logiqa_dev
from hierarchical_control.precritic_training_protocol import (
    LABELS,
    _source_contract,
    merge_training_rollouts,
    prepare_precritic_training_protocol,
    select_final_test_split,
)


OPTIONS = {"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"}


def _cost(prompt: int, completion: int, latency: float = 0.5) -> dict:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "calls": 1,
        "latency_seconds": latency,
    }


def _rollout(
    dataset: str,
    question_id: int,
    passage: str,
    *,
    gold: str,
    solver_answer: str,
    critic_answer: str | None,
) -> dict:
    problem = {
        "passage": passage,
        "question": "Which option must be true?",
        "options": dict(OPTIONS),
    }
    digest = content_sha256(passage, problem["question"], list(OPTIONS.values()))
    raw_solver = f"Solver reasoning for {passage}\nFINAL_ANSWER: {solver_answer}"
    solver = {
        "raw_output": raw_solver,
        "strict_answer": solver_answer,
        "tolerant": {
            "answer": solver_answer,
            "match_count": 1,
            "conflict": False,
        },
        "cost": _cost(10, 2),
    }
    state = {
        "sample_id": digest,
        "question_id": question_id,
        "problem": problem,
        "solver_raw_output": raw_solver,
    }
    verdict = "REVISE" if critic_answer is not None else "KEEP"
    protocol = {
        "parsed_verdict": verdict,
        "proposed_answer": critic_answer,
        "parse_failure": False,
        "effective_verdict": verdict,
        "effective_proposed_answer": critic_answer,
    }
    row = {
        "sample_id": digest,
        "question_id": question_id,
        "mock_only": False,
        "gold": gold,
        "state_for_controller": state,
        "solver": solver,
    }
    if dataset == "collection_200":
        row["actions"] = {
            "FULL": {
                "critic_protocol": protocol,
                "raw_outputs": {
                    "critic": "CRITIC_CONTINUATION_SENTINEL",
                    "refiner": "REFINER_CONTINUATION_SENTINEL",
                },
            }
        }
    else:
        row["critic"] = {
            **protocol,
            "raw_output": "CRITIC_CONTINUATION_SENTINEL",
            "cost": _cost(20, 3, 0.75),
        }
    return row


def _four_rows() -> tuple[list[dict], list[dict]]:
    # Repeated question IDs are deliberate; identity is content SHA256.
    old = [
        _rollout(
            "collection_200", 7, "old c2c", gold="A", solver_answer="A", critic_answer=None
        ),
        _rollout(
            "collection_200", 7, "old c2w", gold="A", solver_answer="A", critic_answer="B"
        ),
    ]
    new = [
        _rollout(
            "collection_800", 7, "new w2c", gold="B", solver_answer="A", critic_answer="B"
        ),
        _rollout(
            "collection_800", 7, "new w2w", gold="C", solver_answer="A", critic_answer="B"
        ),
    ]
    return old, new


def _merge(old: list[dict], new: list[dict]):
    paths = {
        "collection_200": Path("old.jsonl"),
        "collection_800": Path("new.jsonl"),
    }
    return merge_training_rollouts(
        old,
        new,
        source_paths=paths,
        source_sha256={"collection_200": "old-sha", "collection_800": "new-sha"},
        expected_source_counts={"collection_200": 2, "collection_800": 2},
        expected_label_counts={label: 1 for label in LABELS},
    )


def test_merge_is_deterministic_content_keyed_and_tolerates_duplicate_question_id():
    old, new = _four_rows()
    first, stats = _merge(old, new)
    second, second_stats = _merge(old, new)
    assert first == second
    assert stats == second_stats
    assert stats["samples"] == stats["unique_content_sha256"] == 4
    assert stats["unique_question_id"] == 1
    assert stats["label_counts"] == {label: 1 for label in LABELS}
    assert [row["label"] for row in first] == list(LABELS)

    duplicate = _rollout(
        "collection_800",
        999,
        "old c2c",
        gold="A",
        solver_answer="A",
        critic_answer=None,
    )
    with pytest.raises(ValueError, match="Duplicate training content SHA256"):
        merge_training_rollouts(
            old,
            [duplicate, new[1]],
            source_paths={
                "collection_200": Path("old.jsonl"),
                "collection_800": Path("new.jsonl"),
            },
            source_sha256={"collection_200": "old", "collection_800": "new"},
            expected_source_counts={"collection_200": 2, "collection_800": 2},
            expected_label_counts={label: 1 for label in LABELS},
        )


def test_training_whitelist_has_no_gold_or_continuation_leak_and_never_estimates_cost():
    old, new = _four_rows()
    records, stats = _merge(old, new)
    assert stats["cost_available"] == 2
    assert stats["cost_unavailable"] == 2
    for record in records:
        assert set(record["model_input"]) == {"problem", "solver"}
        serialized = json.dumps(record, ensure_ascii=False)
        assert "CRITIC_CONTINUATION_SENTINEL" not in serialized
        assert "REFINER_CONTINUATION_SENTINEL" not in serialized
        assert '"gold"' not in serialized.casefold()
    for record in records[:2]:
        assert record["cost_available"] is False
        assert record["critic_cost_target"] is None
    for record in records[2:]:
        assert record["cost_available"] is True
        assert record["critic_cost_target"] == _cost(20, 3, 0.75)


def test_final_split_is_deterministic_content_unique_and_fully_disjoint(tmp_path):
    examples = [
        LogiQAExample(i, f"passage {i}", "question", tuple(OPTIONS.values()), "A")
        for i in range(8)
    ]
    # Duplicate content under a different source question ID.
    examples.append(LogiQAExample(999, "passage 2", "question", tuple(OPTIONS.values()), "B"))
    excluded = {content_sha256("passage 0", "question", list(OPTIONS.values()))}
    first, first_hashes = select_final_test_split(
        examples,
        excluded,
        data_path=tmp_path / "dev.txt",
        data_sha256="dev-sha",
        seed=20260815,
        sample_count=4,
    )
    second, second_hashes = select_final_test_split(
        examples,
        excluded,
        data_path=tmp_path / "dev.txt",
        data_sha256="dev-sha",
        seed=20260815,
        sample_count=4,
    )
    assert first == second and first_hashes == second_hashes
    assert len(first_hashes) == 4 and first_hashes.isdisjoint(excluded)
    assert first["selection_stats"]["within_dev_duplicates_excluded"] == 1
    assert first["final_test"] and first["sealed"] and first["never_evaluated"]
    assert first["model_calls"] == 0
    assert "gold" not in json.dumps(first, ensure_ascii=False).casefold()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dev_record(question_id: int, passage: str) -> dict:
    return {
        "id": question_id,
        "answer": 0,
        "text": passage,
        "question": "dev question",
        "options": list(OPTIONS.values()),
    }


def test_full_offline_prepare_writes_leakage_safe_outputs_and_freezes_split(tmp_path):
    old, new = _four_rows()
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    _write_jsonl(old_path, old)
    _write_jsonl(new_path, new)

    dev_path = tmp_path / "dev.txt"
    _write_jsonl(dev_path, [_dev_record(i, f"dev passage {i}") for i in range(12)])
    pilot_dir = tmp_path / "pilot"
    pilot_path = pilot_dir / "predictions.jsonl"
    pilot_example = load_logiqa_dev(dev_path, limit=1, seed=7)[0]
    _write_jsonl(pilot_path, [{"question_id": pilot_example.question_id, "gold": "A"}])
    pilot_summary = pilot_dir / "summary.json"
    pilot_summary.write_text(
        json.dumps(
            {
                "data_path": str(dev_path),
                "requested_limit": 1,
                "samples": 1,
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.jsonl"
    validation_problem = {
        "passage": "dev passage 11",
        "question": "dev question",
        "options": dict(OPTIONS),
    }
    _write_jsonl(validation_path, [{"question_id": 11, "problem": validation_problem}])

    expected = {
        "collection_200": _sha(old_path),
        "collection_800": _sha(new_path),
        "pilot_predictions": _sha(pilot_path),
        "pilot_summary": _sha(pilot_summary),
        "validation_predictions": _sha(validation_path),
        "dev_data": _sha(dev_path),
    }
    training_dir = tmp_path / "training"
    final_dir = tmp_path / "final"
    result = prepare_precritic_training_protocol(
        collection_200_path=old_path,
        collection_800_path=new_path,
        pilot_predictions_path=pilot_path,
        validation_predictions_path=validation_path,
        dev_data_path=dev_path,
        training_output_dir=training_dir,
        final_test_output_dir=final_dir,
        expected_source_sha256=expected,
        expected_source_counts={"collection_200": 2, "collection_800": 2},
        expected_label_counts={label: 1 for label in LABELS},
        final_test_samples=4,
    )
    assert result["model_calls"] == 0 and result["controller_trained"] is False
    records = [json.loads(line) for line in (training_dir / "training_examples.jsonl").read_text().splitlines()]
    manifest = json.loads((training_dir / "manifest.json").read_text())
    final = json.loads((final_dir / "split_manifest.json").read_text())
    assert len(records) == 4 and len(final["selected_samples"]) == 4
    assert manifest["content_overlap_checks"] == {
        "collection_200_vs_collection_800": 0,
        "training_vs_pilot": 0,
        "training_vs_validation": 0,
        "final_test_vs_training": 0,
        "final_test_vs_pilot": 0,
        "final_test_vs_validation": 0,
        "final_test_internal_duplicates": 0,
    }
    assert final["final_test"] and final["sealed"] and final["never_evaluated"]
    assert "gold" not in json.dumps(records, ensure_ascii=False).casefold()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_precritic_training_protocol(
            collection_200_path=old_path,
            collection_800_path=new_path,
            pilot_predictions_path=pilot_path,
            validation_predictions_path=validation_path,
            dev_data_path=dev_path,
            training_output_dir=training_dir,
            final_test_output_dir=final_dir,
            expected_source_sha256=expected,
            expected_source_counts={"collection_200": 2, "collection_800": 2},
            expected_label_counts={label: 1 for label in LABELS},
            final_test_samples=4,
        )


def test_source_sha_mismatch_fails_before_processing(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source SHA256 mismatch"):
        _source_contract({"source": source}, {"source": "0" * 64})


def test_prepare_cli_never_initializes_backend(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("offline protocol preparation must not initialize backend")

    received = {}

    def fake_prepare(**kwargs):
        received.update(kwargs)
        return {"offline_prepare": True, "model_calls": 0}

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden_backend)
    monkeypatch.setattr(cli, "MockBackend", forbidden_backend)
    monkeypatch.setattr(cli, "prepare_precritic_training_protocol", fake_prepare)
    assert main(
        [
            "prepare-precritic-training-protocol",
            "--dev-data-path",
            str(tmp_path / "dev.txt"),
            "--training-output-dir",
            str(tmp_path / "training"),
            "--final-test-output-dir",
            str(tmp_path / "final"),
        ]
    ) == 0
    assert received["dev_data_path"] == str(tmp_path / "dev.txt")
    assert received["training_output_dir"] == str(tmp_path / "training")
    assert received["final_test_output_dir"] == str(tmp_path / "final")
