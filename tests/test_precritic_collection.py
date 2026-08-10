from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierarchical_control.cli import main
from hierarchical_control.logiqa_action_collection import (
    ActionCollectionSettings,
    content_sha256,
)
from hierarchical_control.precritic_collection import (
    collect_precritic_rollouts,
    prepare_precritic_collection,
)
from hierarchical_control.types import CompletionResult


VALID_REVISE = """QUESTION_POLARITY: MUST
CONSTRAINT_AUDIT: The Solver violates the decisive constraint.
DECISIVE_ERROR: The selected option cannot satisfy the ordering rule.
ALTERNATIVE_VERIFICATION: B satisfies every stated constraint.
VERDICT: REVISE
PROPOSED_ANSWER: B"""


def _record(
    question_id: int,
    answer: int,
    passage: str,
    question: str,
    options: list[str] | None = None,
) -> dict:
    return {
        "id": question_id,
        "answer": answer,
        "text": passage,
        "question": question,
        "options": options or ["Alpha", "Beta", "Gamma", "Delta"],
        "gold_secret": "DO_NOT_LEAK_GOLD_SENTINEL",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _inputs(root: Path) -> tuple[Path, Path, Path, Path, str]:
    pilot_row = _record(1, 0, "Pilot passage", "Pilot question?")
    pilot_dev = root / "pilot" / "dev.txt"
    _write_jsonl(pilot_dev, [pilot_row])
    pilot = root / "pilot" / "predictions.jsonl"
    _write_jsonl(pilot, [{"question_id": 1, "gold": "A"}])
    (pilot.parent / "summary.json").write_text(
        json.dumps(
            {
                "data_path": str(pilot_dev),
                "requested_limit": 1,
                "samples": 1,
                "seed": 7,
            }
        ),
        encoding="utf-8",
    )

    validation_problem = {
        "passage": "Validation passage",
        "question": "Validation question?",
        "options": {"A": "One", "B": "Two", "C": "Three", "D": "Four"},
    }
    validation = root / "validation" / "predictions.jsonl"
    _write_jsonl(validation, [{"question_id": 2, "problem": validation_problem}])

    collection_problem = {
        "passage": "Existing collection passage",
        "question": "Existing collection question?",
        "options": {"A": "I", "B": "II", "C": "III", "D": "IV"},
    }
    collection_digest = content_sha256(
        collection_problem["passage"],
        collection_problem["question"],
        list(collection_problem["options"].values()),
    )
    collection = root / "collection" / "rollouts.jsonl"
    _write_jsonl(
        collection,
        [
            {
                "sample_id": collection_digest,
                "state_for_controller": {"problem": collection_problem},
            }
        ],
    )

    eligible = _record(
        4,
        1,
        "Eligible passage where W precedes X.",
        "Which option MUST be true?",
        ["First", "Second", "Third", "Fourth"],
    )
    eligible_digest = content_sha256(
        eligible["text"], eligible["question"], eligible["options"]
    )
    train = root / "train.txt"
    _write_jsonl(
        train,
        [
            pilot_row,
            _record(
                2,
                1,
                validation_problem["passage"],
                validation_problem["question"],
                list(validation_problem["options"].values()),
            ),
            _record(
                3,
                2,
                collection_problem["passage"],
                collection_problem["question"],
                list(collection_problem["options"].values()),
            ),
            eligible,
            _record(
                5,
                1,
                " eligible PASSAGE where w precedes x. ",
                "which option must be TRUE?",
                [" first ", "SECOND", "third", "fourth"],
            ),
        ],
    )
    return train, pilot, validation, collection, eligible_digest


def _settings(root: Path) -> ActionCollectionSettings:
    return ActionCollectionSettings(
        source_validation_summary=(root / "validation_summary.json").resolve(),
        base_url="http://recording.invalid/v1",
        model="recording-model",
        temperature=0.0,
        solver_max_tokens=111,
        critic_max_tokens=222,
        refiner_max_tokens=333,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


class FakeMockBackend:
    mock_only = True

    def __init__(
        self,
        output_dir: Path | None = None,
        fail_purpose: str | None = None,
        usage_failure: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.fail_purpose = fail_purpose
        self.usage_failure = usage_failure
        self.calls: list[dict] = []

    def complete(self, messages, max_tokens, purpose):
        if not self.calls and self.output_dir is not None:
            assert (self.output_dir / "split_manifest.json").is_file()
            assert not (self.output_dir / "summary.json").exists()
        self.calls.append(
            {
                "purpose": purpose,
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        if purpose == self.fail_purpose:
            raise RuntimeError(f"intentional interruption at {purpose}")
        content = (
            "One saved Solver state.\nFINAL_ANSWER: A"
            if purpose == "solver"
            else VALID_REVISE
        )
        prompt, completion = (
            (10, 2) if purpose == "solver" else (30, 3)
        )
        if self.usage_failure == "missing":
            return CompletionResult(content=content, completion_tokens=completion)
        total = prompt + completion
        if self.usage_failure == "inconsistent":
            total += 1
        return CompletionResult(
            content=content,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            usage_reported=True,
        )


def _prepare_mock(root: Path) -> tuple[Path, Path, ActionCollectionSettings]:
    train, pilot, validation, collection, _ = _inputs(root / "inputs")
    output = root / "output"
    settings = _settings(root)
    prepare_precritic_collection(
        train,
        output,
        settings,
        pilot_predictions=pilot,
        validation_predictions=validation,
        existing_collection=collection,
        sample_count=1,
        seed=20260814,
        mock_only=True,
    )
    return train, output, settings


def test_prepare_excludes_all_history_deduplicates_and_writes_only_manifest(tmp_path):
    train, pilot, validation, collection, eligible_digest = _inputs(
        tmp_path / "inputs"
    )
    output = tmp_path / "output"
    manifest = prepare_precritic_collection(
        train,
        output,
        _settings(tmp_path),
        pilot_predictions=pilot,
        validation_predictions=validation,
        existing_collection=collection,
        sample_count=1,
        seed=20260814,
        mock_only=True,
    )
    assert {path.name for path in output.iterdir()} == {"split_manifest.json"}
    assert manifest["prepared_only"] is True
    assert manifest["mock_only"] is True
    assert manifest["backend_initialized"] is False
    assert manifest["model_calls"] == 0
    assert manifest["seed"] == 20260814
    assert manifest["selected_samples"] == [
        {"question_id": 4, "content_sha256": eligible_digest}
    ]
    assert manifest["selection_stats"]["historical_content_records_excluded"] == 3
    assert manifest["selection_stats"]["within_train_duplicate_records_ignored"] == 1
    assert manifest["exclusion_sources"]["combined_unique_content"] == 3
    assert len(manifest["split_sha256"]) == 64
    assert len(manifest["critic_policy"]["critic_prompt_sha256"]) == 64
    assert len(manifest["git_commit"]) == 40
    assert not (output / "summary.json").exists()


def test_two_call_mock_flow_gold_isolated_and_stage_usage_exact(tmp_path):
    train, output, settings = _prepare_mock(tmp_path)
    backend = FakeMockBackend(output)
    summary = collect_precritic_rollouts(train, output, backend, settings)
    assert [call["purpose"] for call in backend.calls] == [
        "solver",
        "structured_v2_critic",
    ]
    assert [call["max_tokens"] for call in backend.calls] == [111, 222]
    prompts = [
        "\n".join(message["content"] for message in call["messages"])
        for call in backend.calls
    ]
    assert all("DO_NOT_LEAK_GOLD_SENTINEL" not in prompt for prompt in prompts)
    assert all("gold" not in prompt.casefold() for prompt in prompts)
    assert "A. First" in prompts[1] and "D. Fourth" in prompts[1]
    assert "One saved Solver state." in prompts[1]

    row = json.loads((output / "rollouts.jsonl").read_text())
    assert row["mock_only"] is True
    assert row["actual_calls"] == 2
    assert row["actions"]["STOP"]["answer"] == "A"
    assert row["actions"]["CRITIC_ONLY"]["answer"] == "B"
    assert row["label"] == "wrong_to_correct"
    assert row["solver"]["cost"]["total_tokens"] == 12
    assert row["critic"]["cost"]["total_tokens"] == 33
    assert row["actions"]["CRITIC_ONLY"]["complete_cost"]["total_tokens"] == 45
    assert row["solver"]["cost"]["calls"] == 1
    assert row["critic"]["cost"]["calls"] == 1
    assert row["actions"]["CRITIC_ONLY"]["complete_cost"]["calls"] == 2
    assert set(row["actions"]) == {"STOP", "CRITIC_ONLY"}

    def no_gold_key(value):
        if isinstance(value, dict):
            assert all("gold" not in str(key).casefold() for key in value)
            for child in value.values():
                no_gold_key(child)
        elif isinstance(value, list):
            for child in value:
                no_gold_key(child)

    no_gold_key(row["state_for_controller"])
    assert summary["mock_only"] is True
    assert summary["actual_calls"] == 2
    assert summary["usage_estimated"] is False
    assert json.loads((output / "split_manifest.json").read_text())["mock_only"] is True
    assert "mock_only=true" in (output / "report.md").read_text()


def test_stage_and_sample_resume_do_not_repeat_calls(tmp_path):
    train, output, settings = _prepare_mock(tmp_path)
    interrupted = FakeMockBackend(fail_purpose="structured_v2_critic")
    with pytest.raises(RuntimeError, match="structured_v2_critic"):
        collect_precritic_rollouts(train, output, interrupted, settings)
    assert [call["purpose"] for call in interrupted.calls] == [
        "solver",
        "structured_v2_critic",
    ]
    checkpoint_files = list((output / ".precritic_checkpoints").glob("*.json"))
    assert len(checkpoint_files) == 1
    assert json.loads(checkpoint_files[0].read_text())["mock_only"] is True
    resumed = FakeMockBackend()
    collect_precritic_rollouts(train, output, resumed, settings)
    assert [call["purpose"] for call in resumed.calls] == [
        "structured_v2_critic"
    ]
    assert len((output / "rollouts.jsonl").read_text().splitlines()) == 1

    already_complete = FakeMockBackend()
    collect_precritic_rollouts(train, output, already_complete, settings)
    assert already_complete.calls == []
    assert len((output / "rollouts.jsonl").read_text().splitlines()) == 1


@pytest.mark.parametrize("failure", ["missing", "inconsistent"])
def test_missing_or_inconsistent_stage_usage_fails_without_summary(tmp_path, failure):
    train, output, settings = _prepare_mock(tmp_path)
    backend = FakeMockBackend(usage_failure=failure)
    with pytest.raises(RuntimeError, match="usage"):
        collect_precritic_rollouts(train, output, backend, settings)
    assert not (output / "summary.json").exists()


def test_prepare_cli_does_not_initialize_backend(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("prepare must not initialize any backend")

    received = {}

    def fake_prepare(**kwargs):
        received.update(kwargs)
        return {"prepared_only": True, "backend_initialized": False}

    settings = _settings(tmp_path)
    monkeypatch.setattr(cli, "OpenAIBackend", forbidden_backend)
    monkeypatch.setattr(cli, "MockBackend", forbidden_backend)
    monkeypatch.setattr(cli, "load_action_collection_settings", lambda: settings)
    monkeypatch.setattr(cli, "prepare_precritic_collection", fake_prepare)
    assert main(
        [
            "prepare-precritic-collection",
            "--data-path",
            "train.txt",
            "--output-dir",
            str(tmp_path / "prepared"),
        ]
    ) == 0
    assert received == {
        "data_path": "train.txt",
        "output_dir": str(tmp_path / "prepared"),
        "settings": settings,
        "mock_only": False,
    }
