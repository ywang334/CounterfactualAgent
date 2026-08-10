from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierarchical_control import cli
from hierarchical_control.logiqa_pilot import build_critic_messages, build_refiner_messages
from hierarchical_control.logiqa_prompts import (
    MINIMAL_V1,
    STRUCTURED_V2,
    build_versioned_critic_messages,
    build_versioned_refiner_messages,
)
from hierarchical_control.logiqa_replay import (
    SAFE_KEEP_REVIEW,
    parse_critic_protocol,
    run_logiqa_prompt_replay,
)
from hierarchical_control.types import CompletionResult


VALID_KEEP = """QUESTION_POLARITY: MUST
CONSTRAINT_AUDIT: The selected answer satisfies every stated constraint.
DECISIVE_ERROR: NONE
ALTERNATIVE_VERIFICATION: NONE
VERDICT: KEEP
PROPOSED_ANSWER: NONE"""

VALID_REVISE = """QUESTION_POLARITY: COULD
CONSTRAINT_AUDIT: Option A violates the adjacency constraint.
DECISIVE_ERROR: A places X and Y apart.
ALTERNATIVE_VERIFICATION: C places X and Y together and satisfies all constraints.
VERDICT: REVISE
PROPOSED_ANSWER: C"""


def _usage(prompt: int, completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _write_source(
    root: Path,
    solver_answer: str = "A",
    gold_index: int = 0,
) -> Path:
    root.mkdir(parents=True)
    dev_path = root / "dev.txt"
    dev_row = {
        "id": 101,
        "answer": gold_index,
        "text": "Passage with constraint X before Y.",
        "question": "Which option MUST be true?",
        "options": [
            "Alpha option",
            "Beta option",
            "Gamma option",
            "Delta option",
        ],
        "type": {"toy": True},
    }
    dev_path.write_text(json.dumps(dev_row) + "\n", encoding="utf-8")
    solver_output = f"Saved solver reasoning.\nFINAL_ANSWER: {solver_answer}"
    prediction = {
        "question_id": 101,
        "gold": "ABCD"[gold_index],
        "gold_secret": "DO_NOT_LEAK_GOLD_SENTINEL",
        "solver_answer": solver_answer,
        "refiner_answer": solver_answer,
        "solver_correct": solver_answer == "ABCD"[gold_index],
        "refiner_correct": solver_answer == "ABCD"[gold_index],
        "solver_parse_failure": False,
        "refiner_parse_failure": False,
        "raw_outputs": {
            "solver": solver_output,
            "critic": "Existing minimal_v1 critic output.",
            "refiner": f"Existing minimal_v1 refiner.\nFINAL_ANSWER: {solver_answer}",
        },
        "usage": {
            "solver": _usage(10, 5),
            "solver_critic_refiner": _usage(40, 15),
        },
        "latency_seconds": {
            "solver": 0.1,
            "solver_critic_refiner": 0.4,
        },
        "mock_only": False,
    }
    predictions_path = root / "predictions.jsonl"
    predictions_path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    summary = {
        "mock_only": False,
        "data_path": str(dev_path),
        "requested_limit": 1,
        "samples": 1,
        "seed": 17,
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
    (root / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return predictions_path


class FakeReplayBackend:
    mock_only = False

    def __init__(
        self,
        critic_output: str = VALID_KEEP,
        refiner_answer: str = "A",
        refiner_decision: str = "KEEP_ORIGINAL",
        fail_purpose: str | None = None,
    ) -> None:
        self.critic_output = critic_output
        self.refiner_answer = refiner_answer
        self.refiner_decision = refiner_decision
        self.fail_purpose = fail_purpose
        self.calls: list[dict] = []

    def complete(self, messages, max_tokens, purpose):
        assert purpose in {"critic", "refiner"}
        self.calls.append(
            {
                "purpose": purpose,
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        if purpose == self.fail_purpose:
            raise RuntimeError(f"intentional {purpose} interruption")
        if purpose == "critic":
            content = self.critic_output
            prompt_tokens, completion_tokens = 20, 6
        else:
            content = (
                "CRITIQUE_VALIDATION: NOT_APPLICABLE\n"
                f"REFINEMENT_DECISION: {self.refiner_decision}\n"
                "JUSTIFICATION: Preserve or revise according to the review.\n"
                f"FINAL_ANSWER: {self.refiner_answer}"
            )
            prompt_tokens, completion_tokens = 30, 7
        return CompletionResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            usage_reported=True,
        )


def _prompt_text(call: dict) -> str:
    return "\n".join(message["content"] for message in call["messages"])


def test_versioned_minimal_v1_messages_are_unchanged():
    problem = "Original problem\nA. one\nB. two\nC. three\nD. four"
    solver = "Reasoning\nFINAL_ANSWER: A"
    critic = "Existing review"
    assert build_versioned_critic_messages(problem, solver, MINIMAL_V1) == (
        build_critic_messages(problem, solver)
    )
    assert build_versioned_refiner_messages(problem, solver, critic, MINIMAL_V1) == (
        build_refiner_messages(problem, solver, critic)
    )


@pytest.mark.parametrize(
    ("output", "failure", "verdict", "answer"),
    [
        (VALID_KEEP, False, "KEEP", None),
        (VALID_REVISE, False, "REVISE", "C"),
        ("VERDICT: KEEP\nPROPOSED_ANSWER: A", True, "KEEP", None),
        ("VERDICT: REVISE\nPROPOSED_ANSWER: NONE", True, "KEEP", None),
        ("VERDICT: revise\nPROPOSED_ANSWER: C", True, "KEEP", None),
        ("VERDICT: KEEP", True, "KEEP", None),
        (
            "VERDICT: KEEP\nVERDICT: REVISE\nPROPOSED_ANSWER: C",
            True,
            "KEEP",
            None,
        ),
    ],
)
def test_critic_protocol_keep_revise_and_safe_fallback(
    output,
    failure,
    verdict,
    answer,
):
    parsed = parse_critic_protocol(output)
    assert parsed.parse_failure is failure
    assert parsed.effective_verdict == verdict
    assert parsed.effective_proposed_answer == answer
    if failure:
        assert parsed.review_for_refiner == SAFE_KEEP_REVIEW


def test_replay_messages_include_full_problem_and_never_include_gold(tmp_path):
    source = _write_source(tmp_path / "pilot")
    backend = FakeReplayBackend()
    output_dir = tmp_path / "structured"
    summary = run_logiqa_prompt_replay(
        source,
        STRUCTURED_V2,
        output_dir,
        backend,
    )

    assert [call["purpose"] for call in backend.calls] == ["critic", "refiner"]
    assert [call["max_tokens"] for call in backend.calls] == [222, 333]
    critic_prompt = _prompt_text(backend.calls[0])
    refiner_prompt = _prompt_text(backend.calls[1])
    for prompt in (critic_prompt, refiner_prompt):
        assert "Passage with constraint X before Y." in prompt
        assert "Which option MUST be true?" in prompt
        for letter, option in zip(
            "ABCD",
            ("Alpha option", "Beta option", "Gamma option", "Delta option"),
        ):
            assert f"{letter}. {option}" in prompt
        assert "Saved solver reasoning." in prompt
        assert "DO_NOT_LEAK_GOLD_SENTINEL" not in prompt
        assert "gold" not in prompt.casefold()
    assert VALID_KEEP in refiner_prompt
    assert summary["solver_called"] is False
    assert summary["prompt_version"] == STRUCTURED_V2
    assert summary["mock_only"] is False

    row = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert row["prompt_development"] is True
    assert row["deployable_result"] is False
    assert row["solver_reused"] is True
    assert row["solver_called"] is False
    assert row["critic"]["raw_output"] == VALID_KEEP
    assert row["critic"]["effective_verdict"] == "KEEP"
    assert row["tolerant"]["full_answer"] == "A"
    assert row["refiner_protocol_violation"] is False
    assert row["usage"]["critic"] == _usage(20, 6)
    assert row["usage"]["refiner"] == _usage(30, 7)


def test_invalid_critic_is_saved_and_refiner_receives_safe_keep(tmp_path):
    source = _write_source(tmp_path / "pilot")
    malformed = "Analysis without required protocol fields."
    backend = FakeReplayBackend(critic_output=malformed)
    output_dir = tmp_path / "structured"
    run_logiqa_prompt_replay(source, STRUCTURED_V2, output_dir, backend)

    row = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert row["critic"]["raw_output"] == malformed
    assert row["critic_parse_failure"] is True
    assert row["critic"]["effective_verdict"] == "KEEP"
    refiner_prompt = _prompt_text(backend.calls[1])
    assert SAFE_KEEP_REVIEW in refiner_prompt
    assert malformed not in refiner_prompt


def test_keep_change_is_detected_as_refiner_protocol_violation(tmp_path):
    source = _write_source(tmp_path / "pilot", solver_answer="A", gold_index=0)
    backend = FakeReplayBackend(
        critic_output=VALID_KEEP,
        refiner_answer="B",
        refiner_decision="APPLY_REVISION",
    )
    output_dir = tmp_path / "structured"
    summary = run_logiqa_prompt_replay(
        source,
        STRUCTURED_V2,
        output_dir,
        backend,
    )
    row = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    assert row["refiner_protocol_violation"] is True
    assert row["transition"] == "correct_to_wrong"
    assert summary["refiner_protocol_violation"]["count"] == 1
    assert summary["strategies"]["structured_v2_full"]["degraded"] == 1


def test_replay_resume_skips_completed_sample_and_never_calls_solver(tmp_path):
    source = _write_source(tmp_path / "pilot")
    output_dir = tmp_path / "structured"
    first = FakeReplayBackend()
    run_logiqa_prompt_replay(source, STRUCTURED_V2, output_dir, first)
    assert [call["purpose"] for call in first.calls] == ["critic", "refiner"]

    second = FakeReplayBackend(fail_purpose="critic")
    run_logiqa_prompt_replay(source, STRUCTURED_V2, output_dir, second)
    assert second.calls == []
    assert len((output_dir / "predictions.jsonl").read_text().splitlines()) == 1


def test_stage_checkpoint_avoids_repeating_completed_critic(tmp_path):
    source = _write_source(tmp_path / "pilot")
    output_dir = tmp_path / "structured"
    interrupted = FakeReplayBackend(fail_purpose="refiner")
    with pytest.raises(RuntimeError, match="during refiner"):
        run_logiqa_prompt_replay(source, STRUCTURED_V2, output_dir, interrupted)
    assert [call["purpose"] for call in interrupted.calls] == ["critic", "refiner"]

    resumed = FakeReplayBackend()
    run_logiqa_prompt_replay(source, STRUCTURED_V2, output_dir, resumed)
    assert [call["purpose"] for call in resumed.calls] == ["refiner"]


def test_replay_refuses_to_overwrite_minimal_v1_source(tmp_path):
    source = _write_source(tmp_path / "pilot")
    backend = FakeReplayBackend()
    original = source.read_bytes()
    with pytest.raises(ValueError, match="cannot overwrite"):
        run_logiqa_prompt_replay(
            source,
            STRUCTURED_V2,
            source.parent,
            backend,
        )
    assert source.read_bytes() == original
    assert backend.calls == []


def test_replay_cli_uses_saved_backend_settings_and_never_calls_solver(
    tmp_path,
    monkeypatch,
    capsys,
):
    source = _write_source(tmp_path / "pilot")
    backend = FakeReplayBackend()
    constructor_args = {}

    def fake_openai_backend(**kwargs):
        constructor_args.update(kwargs)
        return backend

    monkeypatch.setattr(cli, "OpenAIBackend", fake_openai_backend)
    output_dir = tmp_path / "structured"
    assert (
        cli.main(
            [
                "replay-logiqa-prompts",
                "--predictions",
                str(source),
                "--prompt-version",
                STRUCTURED_V2,
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert [call["purpose"] for call in backend.calls] == ["critic", "refiner"]
    assert constructor_args["base_url"] == "http://recording.invalid/v1"
    assert constructor_args["model"] == "recording-model"
    assert constructor_args["temperature"] == 0.0
    assert constructor_args["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_replay_cli_does_not_default_to_structured_v2():
    args = cli.build_parser().parse_args(
        [
            "replay-logiqa-prompts",
            "--predictions",
            "predictions.jsonl",
            "--output-dir",
            "out",
        ]
    )
    assert args.prompt_version == MINIMAL_V1
