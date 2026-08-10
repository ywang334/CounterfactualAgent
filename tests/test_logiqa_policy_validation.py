from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierarchical_control.backend import MockBackend
from hierarchical_control.logiqa_pilot import LogiQAExample
from hierarchical_control.logiqa_policy_validation import (
    run_logiqa_policy_validation,
    select_validation_examples,
    structured_critic_detection_metrics,
)
from hierarchical_control.types import CompletionResult


VALID_KEEP = """QUESTION_POLARITY: MUST
CONSTRAINT_AUDIT: The selected option satisfies the stated constraints.
DECISIVE_ERROR: NONE
ALTERNATIVE_VERIFICATION: NONE
VERDICT: KEEP
PROPOSED_ANSWER: NONE"""


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _write_source(root: Path, records: int = 3) -> Path:
    root.mkdir(parents=True)
    dev_path = root / "dev.txt"
    dev_rows = []
    for index in range(records):
        dev_rows.append(
            {
                "id": 100 + index,
                "answer": index % 4,
                "text": f"Passage {index}: X must precede Y.",
                "question": f"Question {index}: Which option MUST be true?",
                "options": [
                    f"Alpha {index}",
                    f"Beta {index}",
                    f"Gamma {index}",
                    f"Delta {index}",
                ],
            }
        )
    dev_path.write_text(
        "".join(json.dumps(row) + "\n" for row in dev_rows),
        encoding="utf-8",
    )
    pilot_prediction = {
        "question_id": 100,
        "gold": "A",
        "gold_secret": "DO_NOT_LEAK_GOLD_SENTINEL",
    }
    predictions = root / "predictions.jsonl"
    predictions.write_text(json.dumps(pilot_prediction) + "\n", encoding="utf-8")
    summary = {
        "mock_only": False,
        "data_path": str(dev_path),
        "backend": {
            "base_url": "http://recording.invalid/v1",
            "model": "recording-model",
        },
        "temperature": 0.0,
        "generation_caps": {"solver": 111, "critic": 222, "refiner": 333},
        "request_extra_body": {
            "chat_template_kwargs": {"enable_thinking": False}
        },
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return predictions


class FakeValidationBackend:
    mock_only = False

    def __init__(
        self,
        manifest_path: Path | None = None,
        fail_purpose: str | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.fail_purpose = fail_purpose
        self.calls: list[dict] = []

    def complete(self, messages, max_tokens, purpose):
        if not self.calls and self.manifest_path is not None:
            assert self.manifest_path.is_file()
        call = {
            "purpose": purpose,
            "messages": [dict(message) for message in messages],
            "max_tokens": max_tokens,
        }
        self.calls.append(call)
        if purpose == self.fail_purpose:
            raise RuntimeError(f"intentional interruption at {purpose}")
        if purpose == "solver":
            content = "Saved solver reasoning from the single shared state.\nFINAL_ANSWER: A"
        elif purpose.endswith("critic"):
            content = VALID_KEEP if purpose.startswith("structured") else "Keep the answer."
        else:
            if purpose.startswith("structured"):
                content = (
                    "CRITIQUE_VALIDATION: NOT_APPLICABLE\n"
                    "REFINEMENT_DECISION: KEEP_ORIGINAL\n"
                    "JUSTIFICATION: No decisive error was shown.\n"
                    "FINAL_ANSWER: A"
                )
            else:
                content = "Minimal refinement.\nFINAL_ANSWER: A"
        return CompletionResult(
            content=content,
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
            usage_reported=True,
        )


def _all_prompt_text(backend: FakeValidationBackend) -> list[str]:
    return [
        "\n".join(message["content"] for message in call["messages"])
        for call in backend.calls
    ]


def test_held_out_selection_excludes_ids_and_is_deterministic():
    examples = [
        LogiQAExample(index, "p", "q", ("a", "b", "c", "d"), "A")
        for index in range(10)
    ]
    examples.append(
        LogiQAExample(2, "duplicate", "q", ("a", "b", "c", "d"), "B")
    )
    excluded = {json.dumps(["int", 1]), json.dumps(["int", 4])}
    first, duplicates = select_validation_examples(
        examples, excluded, sample_count=5, seed=20260811
    )
    second, _ = select_validation_examples(
        examples, excluded, sample_count=5, seed=20260811
    )
    ids = [item.question_id for item in first]
    assert ids == [item.question_id for item in second]
    assert len(ids) == len(set(ids)) == 5
    assert set(ids).isdisjoint({1, 4})
    assert duplicates == [2]


def test_validation_uses_one_solver_state_gold_isolated_and_manifest_precedes_calls(
    tmp_path,
):
    source = _write_source(tmp_path / "pilot")
    output = tmp_path / "validation"
    backend = FakeValidationBackend(output / "split_manifest.json")
    summary = run_logiqa_policy_validation(
        source, output, backend, sample_count=1, seed=20260811
    )

    assert [call["purpose"] for call in backend.calls] == [
        "solver",
        "minimal_v1_critic",
        "minimal_v1_refiner",
        "structured_v2_critic",
        "structured_v2_refiner",
    ]
    assert [call["max_tokens"] for call in backend.calls] == [111, 222, 333, 222, 333]
    prompts = _all_prompt_text(backend)
    for prompt in prompts:
        assert "DO_NOT_LEAK_GOLD_SENTINEL" not in prompt
        assert "gold" not in prompt.casefold()
    shared_solver = "Saved solver reasoning from the single shared state."
    for call, prompt in zip(backend.calls, prompts):
        if call["purpose"] != "solver":
            assert shared_solver in prompt
    row = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert row["policy_selection_validation"] is True
    assert row["final_test"] is False
    assert row["mock_only"] is False
    assert row["solver_called_once"] is True
    assert row["same_solver_state_for_both_policies"] is True
    assert row["calls"]["actual_total"] == 5
    assert summary["inference"]["actual_total_calls"] == 5
    assert summary["policy_selection"]["selected"] is None
    assert summary["policies"]["minimal_v1"]["minimum_cost_posthoc_oracle"][
        "deployable"
    ] is False


def test_structured_critic_reports_raw_and_actionable_metrics_separately():
    actual = [True, True, False, False]
    raw = [True, False, True, False]
    actionable = [False, False, True, False]
    rows = []
    for index, (truth, raw_intent, action) in enumerate(
        zip(actual, raw, actionable)
    ):
        rows.append(
            {
                "question_id": index,
                "gold": "A",
                "solver": {"tolerant": {"answer": "B" if truth else "A"}},
                "structured_v2": {
                    "critic": {
                        "raw_revise_intent": raw_intent,
                        "actionable_revise": action,
                    }
                },
            }
        )
    metrics = structured_critic_detection_metrics(rows)
    assert metrics["raw_revise_intent"]["matrix"] == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }
    assert metrics["actionable_revise"]["matrix"] == {
        "true_positive": 0,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 2,
    }


def test_stage_checkpoint_resume_never_repeats_completed_calls(tmp_path):
    source = _write_source(tmp_path / "pilot")
    output = tmp_path / "validation"
    interrupted = FakeValidationBackend(fail_purpose="minimal_v1_refiner")
    with pytest.raises(RuntimeError, match="no Mock fallback"):
        run_logiqa_policy_validation(
            source, output, interrupted, sample_count=1, seed=20260811
        )
    assert [call["purpose"] for call in interrupted.calls] == [
        "solver",
        "minimal_v1_critic",
        "minimal_v1_refiner",
    ]

    resumed = FakeValidationBackend()
    run_logiqa_policy_validation(
        source, output, resumed, sample_count=1, seed=20260811
    )
    assert [call["purpose"] for call in resumed.calls] == [
        "minimal_v1_refiner",
        "structured_v2_critic",
        "structured_v2_refiner",
    ]
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 1


def test_mock_backend_is_forbidden_before_split_or_calls(tmp_path):
    source = _write_source(tmp_path / "pilot")
    output = tmp_path / "validation"
    backend = MockBackend()
    with pytest.raises(ValueError, match="Mock is forbidden"):
        run_logiqa_policy_validation(
            source, output, backend, sample_count=1, seed=20260811
        )
    assert backend.calls == []
    assert not output.exists()
