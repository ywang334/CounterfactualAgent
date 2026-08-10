from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierarchical_control.backend import MockBackend
from hierarchical_control.cli import build_parser
from hierarchical_control.config import ExperimentConfig
from hierarchical_control.logiqa_pilot import (
    extract_final_answer,
    load_logiqa_dev,
    run_logiqa_pilot,
)
from hierarchical_control.types import CompletionResult


def _write_dev(path: Path, answers: list[int]) -> None:
    rows = [
        {
            "id": 100 + index,
            "answer": answer,
            "text": f"Passage {index}",
            "question": f"Question {index}?",
            "options": [f"option {letter} {index}" for letter in "ABCD"],
            "type": {"toy": True},
        }
        for index, answer in enumerate(answers)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_logiqa_fields_sampling_and_answer_mapping(tmp_path):
    data_path = tmp_path / "dev.txt"
    _write_dev(data_path, [0, 1, 2, 3])
    first = load_logiqa_dev(data_path, limit=4, seed=17)
    second = load_logiqa_dev(data_path, limit=4, seed=17)
    assert first == second
    by_id = {example.question_id: example for example in first}
    assert {question_id: example.gold for question_id, example in by_id.items()} == {
        100: "A",
        101: "B",
        102: "C",
        103: "D",
    }
    assert by_id[100].passage == "Passage 0"
    assert by_id[100].question == "Question 0?"
    assert by_id[100].options == (
        "option A 0",
        "option B 0",
        "option C 0",
        "option D 0",
    )
    assert len(load_logiqa_dev(data_path, limit=2, seed=3)) == 2


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Reasoning\nFINAL_ANSWER: A", "A"),
        ("FINAL_ANSWER: D\n", "D"),
        ("Reasoning FINAL_ANSWER: A", None),
        ("FINAL_ANSWER:A", None),
        ("FINAL_ANSWER: a", None),
        ("FINAL_ANSWER: A.", None),
        ("FINAL_ANSWER: B\nextra", None),
        ("", None),
    ],
)
def test_strict_final_answer_parser(output, expected):
    assert extract_final_answer(output) == expected


class RecordingRealBackend:
    mock_only = False
    base_url = "http://recording.invalid/v1"
    model = "recording-model"
    temperature = 0.0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, messages, max_tokens, purpose):
        self.calls.append(
            {"messages": [dict(message) for message in messages], "max_tokens": max_tokens, "purpose": purpose}
        )
        contents = {
            "solver": "Initial reasoning.\nFINAL_ANSWER: A",
            "critic": "The proposed inference overlooks a constraint; re-check every option.",
            "refiner": "Revised reasoning.\nFINAL_ANSWER: D",
        }
        completion_tokens = {"solver": 5, "critic": 7, "refiner": 6}[purpose]
        prompt_tokens = {"solver": 20, "critic": 30, "refiner": 40}[purpose]
        return CompletionResult(
            content=contents[purpose],
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            usage_reported=True,
        )


def test_gold_is_not_prompted_and_real_outputs_are_marked(tmp_path):
    data_path = tmp_path / "dev.txt"
    _write_dev(data_path, [3])
    backend = RecordingRealBackend()
    config = ExperimentConfig(
        pilot_generation_caps={"solver": 111, "critic": 222, "refiner": 333}
    )
    output_dir = tmp_path / "pilot"
    summary = run_logiqa_pilot(data_path, output_dir, backend, config, limit=1, seed=9)

    assert [call["purpose"] for call in backend.calls] == ["solver", "critic", "refiner"]
    assert [call["max_tokens"] for call in backend.calls] == [111, 222, 333]
    all_prompts = [
        "\n".join(message["content"] for message in call["messages"]) for call in backend.calls
    ]
    assert all("FINAL_ANSWER: D" not in prompt for prompt in all_prompts)
    assert all("gold" not in prompt.casefold() for prompt in all_prompts)
    critic_prompt = all_prompts[1]
    assert "Passage 0" in critic_prompt
    assert "Question 0?" in critic_prompt
    assert "option A 0" in critic_prompt
    assert "Initial reasoning.\nFINAL_ANSWER: A" in critic_prompt

    prediction = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8"))
    written_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert prediction["mock_only"] is False
    assert written_summary["mock_only"] is False
    assert summary["mock_only"] is False
    assert summary["solver_only"]["accuracy"] == 0.0
    assert summary["solver_critic_refiner"]["accuracy"] == 1.0
    assert summary["transitions"] == {"corrected": 1, "degraded": 0, "unchanged": 0}
    assert summary["solver_only"]["average_total_tokens"] == 25.0
    assert summary["solver_critic_refiner"]["average_total_tokens"] == 108.0
    assert summary["solver_critic_refiner"]["average_extra_collaboration_tokens"] == 83.0
    assert prediction["usage"]["calls"] == {
        "solver_only": 1,
        "extra_collaboration": 2,
        "solver_critic_refiner": 3,
    }


def test_pilot_rejects_mock_backend(tmp_path):
    data_path = tmp_path / "dev.txt"
    _write_dev(data_path, [0])
    with pytest.raises(ValueError, match="Mock is forbidden"):
        run_logiqa_pilot(data_path, tmp_path / "out", MockBackend(), ExperimentConfig(), limit=1)


def test_pilot_cli_defaults():
    args = build_parser().parse_args(
        ["pilot-logiqa", "--data-path", "dev.txt", "--output-dir", "artifacts/pilot"]
    )
    assert args.limit == 50
    assert args.seed == 20260810
    assert args.temperature == 0.0
    assert args.base_url == "http://localhost:8000/v1"
    assert args.model == "localmodel"
