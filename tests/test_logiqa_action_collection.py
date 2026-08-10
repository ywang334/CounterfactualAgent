from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierarchical_control.backend import MockBackend
from hierarchical_control.logiqa_action_collection import (
    ActionCollectionSettings,
    content_sha256,
    run_logiqa_action_collection,
    select_action_collection_examples,
)
from hierarchical_control.logiqa_pilot import LogiQAExample
from hierarchical_control.types import CompletionResult


VALID_KEEP = """QUESTION_POLARITY: MUST
CONSTRAINT_AUDIT: No decisive error is demonstrated.
DECISIVE_ERROR: NONE
ALTERNATIVE_VERIFICATION: NONE
VERDICT: KEEP
PROPOSED_ANSWER: NONE"""


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
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    pilot_dev = root / "pilot" / "dev.txt"
    pilot_row = _record(1, 0, "Pilot passage", "Pilot question?")
    _write_jsonl(pilot_dev, [pilot_row])
    pilot_predictions = root / "pilot" / "predictions.jsonl"
    _write_jsonl(
        pilot_predictions,
        [{"question_id": 1, "gold": "A", "gold_secret": "DO_NOT_LEAK_GOLD"}],
    )
    (pilot_predictions.parent / "summary.json").write_text(
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

    validation_predictions = root / "validation" / "predictions.jsonl"
    validation_problem = {
        "passage": "Validation passage",
        "question": "Validation question?",
        "options": {
            "A": "One",
            "B": "Two",
            "C": "Three",
            "D": "Four",
        },
    }
    _write_jsonl(
        validation_predictions,
        [{"question_id": 2, "problem": validation_problem}],
    )

    train_path = root / "train.txt"
    _write_jsonl(
        train_path,
        [
            pilot_row,
            _record(
                2,
                1,
                "Validation passage",
                "Validation question?",
                ["One", "Two", "Three", "Four"],
            ),
            _record(
                3,
                1,
                "Eligible passage where X precedes Y.",
                "Which option MUST be true?",
                ["First", "Second", "Third", "Fourth"],
            ),
            # Same normalized content as id=3 and therefore not independently eligible.
            _record(
                4,
                1,
                "  ELIGIBLE   passage where x precedes y. ",
                "which option must be TRUE?",
                [" first ", "SECOND", "third", "fourth"],
            ),
        ],
    )
    return train_path, pilot_predictions, validation_predictions


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


class FakeActionBackend:
    mock_only = False

    def __init__(
        self,
        output_dir: Path | None = None,
        fail_purpose: str | None = None,
        missing_usage_purpose: str | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.fail_purpose = fail_purpose
        self.missing_usage_purpose = missing_usage_purpose
        self.calls: list[dict] = []

    def complete(self, messages, max_tokens, purpose):
        if not self.calls and self.output_dir is not None:
            assert (self.output_dir / "split_manifest.json").is_file()
            assert (self.output_dir / "continuation_policy_manifest.json").is_file()
        self.calls.append(
            {
                "purpose": purpose,
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        if purpose == self.fail_purpose:
            raise RuntimeError(f"intentional interruption at {purpose}")
        if purpose == self.missing_usage_purpose:
            return CompletionResult(content="missing", completion_tokens=1)
        content = {
            "solver": "One shared Solver state.\nFINAL_ANSWER: A",
            "short_critic": "The answer should be revised to B.",
            "short_refiner": "Apply the correction.\nFINAL_ANSWER: B",
            "full_critic": VALID_KEEP,
            "full_refiner": (
                "CRITIQUE_VALIDATION: NOT_APPLICABLE\n"
                "REFINEMENT_DECISION: KEEP_ORIGINAL\n"
                "JUSTIFICATION: Keep the original answer.\n"
                "FINAL_ANSWER: A"
            ),
        }[purpose]
        prompt, completion = {
            "solver": (10, 2),
            "short_critic": (100, 3),
            "short_refiner": (200, 4),
            "full_critic": (300, 5),
            "full_refiner": (400, 6),
        }[purpose]
        return CompletionResult(
            content=content,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            usage_reported=True,
        )


def _prompt_text(call: dict) -> str:
    return "\n".join(message["content"] for message in call["messages"])


def test_content_hash_normalization_deduplicates_and_excludes():
    examples = [
        LogiQAExample(1, " A  passage ", "Question?", ("A", "B", "C", "D"), "A"),
        LogiQAExample(2, "a PASSAGE", " question? ", ("a", "b", "c", "d"), "B"),
        LogiQAExample(3, "Excluded", "Q", ("a", "b", "c", "d"), "A"),
        LogiQAExample(4, "Eligible", "Q", ("a", "b", "c", "d"), "A"),
    ]
    excluded = {content_sha256("Excluded", "Q", ["a", "b", "c", "d"])}
    selected, stats = select_action_collection_examples(
        examples,
        excluded,
        sample_count=2,
        seed=20260812,
    )
    hashes = {
        content_sha256(item.passage, item.question, list(item.options))
        for item in selected
    }
    assert len(hashes) == 2
    assert excluded.isdisjoint(hashes)
    assert stats["within_train_duplicate_records_ignored"] == 1
    assert stats["historical_content_records_excluded"] == 1


def test_paired_collection_same_solver_gold_isolation_costs_and_order(tmp_path):
    train, pilot, validation = _write_inputs(tmp_path / "inputs")
    output = tmp_path / "output"
    backend = FakeActionBackend(output)
    summary = run_logiqa_action_collection(
        train,
        output,
        backend,
        _settings(tmp_path),
        pilot_predictions=pilot,
        validation_predictions=validation,
        sample_count=1,
        seed=20260812,
    )

    assert [call["purpose"] for call in backend.calls] == [
        "solver",
        "short_critic",
        "short_refiner",
        "full_critic",
        "full_refiner",
    ]
    assert [call["max_tokens"] for call in backend.calls] == [111, 222, 333, 222, 333]
    for call in backend.calls:
        prompt = _prompt_text(call)
        assert "DO_NOT_LEAK_GOLD" not in prompt
        assert "gold" not in prompt.casefold()
        if call["purpose"] != "solver":
            assert "One shared Solver state." in prompt

    row = json.loads((output / "rollouts.jsonl").read_text(encoding="utf-8"))
    assert "gold" not in json.dumps(row["state_for_controller"]).casefold()
    assert row["same_solver_state_for_short_and_full"] is True
    stop = row["actions"]["STOP"]
    assert stop["incremental_cost"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "latency_seconds": 0.0,
    }
    assert stop["outcome"] == "neutral_wrong"
    assert row["actions"]["SHORT"]["outcome"] == "helpful"
    assert row["actions"]["FULL"]["outcome"] == "neutral_wrong"
    short_cost = row["actions"]["SHORT"]["incremental_cost"]
    assert short_cost["prompt_tokens"] == 300
    assert short_cost["completion_tokens"] == 7
    assert short_cost["total_tokens"] == 307
    assert short_cost["calls"] == 2
    for action in ("STOP", "SHORT", "FULL"):
        for name in ("complete_cost", "incremental_cost"):
            cost = row["actions"][action][name]
            assert cost["total_tokens"] == cost["prompt_tokens"] + cost["completion_tokens"]
    assert row["budget_semantics_version"] == 2
    assert summary["actual_run"]["actual_calls"] == 5
    assert summary["actual_run"]["total_cost"]["total_tokens"] == 1030
    assert summary["minimum_cost_posthoc_oracle"]["selection_counts"] == {
        "STOP": 0,
        "SHORT": 1,
        "FULL": 0,
    }
    policy = json.loads(
        (output / "continuation_policy_manifest.json").read_text(encoding="utf-8")
    )
    assert len(policy["short"]["continuation_policy_sha256"]) == 64
    assert len(policy["full"]["continuation_policy_sha256"]) == 64
    assert len(policy["git_commit"]) == 40


def test_stage_checkpoint_resume_does_not_repeat_completed_calls(tmp_path):
    train, pilot, validation = _write_inputs(tmp_path / "inputs")
    output = tmp_path / "output"
    interrupted = FakeActionBackend(fail_purpose="short_refiner")
    with pytest.raises(RuntimeError, match="no Mock fallback"):
        run_logiqa_action_collection(
            train,
            output,
            interrupted,
            _settings(tmp_path),
            pilot_predictions=pilot,
            validation_predictions=validation,
            sample_count=1,
        )
    assert [call["purpose"] for call in interrupted.calls] == [
        "solver",
        "short_critic",
        "short_refiner",
    ]

    resumed = FakeActionBackend()
    run_logiqa_action_collection(
        train,
        output,
        resumed,
        _settings(tmp_path),
        pilot_predictions=pilot,
        validation_predictions=validation,
        sample_count=1,
    )
    assert [call["purpose"] for call in resumed.calls] == [
        "short_refiner",
        "full_critic",
        "full_refiner",
    ]
    assert len((output / "rollouts.jsonl").read_text().splitlines()) == 1


def test_mock_and_missing_usage_are_rejected(tmp_path):
    train, pilot, validation = _write_inputs(tmp_path / "inputs")
    mock_output = tmp_path / "mock_output"
    mock = MockBackend()
    with pytest.raises(ValueError, match="Mock is forbidden"):
        run_logiqa_action_collection(
            train,
            mock_output,
            mock,
            _settings(tmp_path),
            pilot_predictions=pilot,
            validation_predictions=validation,
            sample_count=1,
        )
    assert mock.calls == []
    assert not mock_output.exists()

    missing_output = tmp_path / "missing_output"
    missing = FakeActionBackend(missing_usage_purpose="solver")
    with pytest.raises(RuntimeError, match="refuses estimated usage"):
        run_logiqa_action_collection(
            train,
            missing_output,
            missing,
            _settings(tmp_path),
            pilot_predictions=pilot,
            validation_predictions=validation,
            sample_count=1,
        )
    assert [call["purpose"] for call in missing.calls] == ["solver"]
