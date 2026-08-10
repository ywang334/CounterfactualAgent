from __future__ import annotations

import re

import pytest

from hierarchical_control.cli import main
from hierarchical_control.precritic_representation_audit import (
    FIELD_NAMES,
    analyze_rendered_input,
    render_feature_text_with_spans,
    summarize_cases,
)
from hierarchical_control.precritic_probe import _render_feature_text


class TinyOffsetTokenizer:
    """Whitespace tokenizer with BERT-like two-special-token truncation."""

    is_fast = True
    model_max_length = 10_000

    def __call__(self, text: str, **kwargs):
        matches = list(re.finditer(r"\S+", text))
        content_ids = list(range(100, 100 + len(matches)))
        content_offsets = [(match.start(), match.end()) for match in matches]
        if kwargs.get("truncation"):
            maximum = int(kwargs["max_length"])
            content_limit = max(maximum - 2, 0)
            content_ids = content_ids[:content_limit]
            content_offsets = content_offsets[:content_limit]
        if kwargs.get("add_special_tokens", True):
            return {
                "input_ids": [1, *content_ids, 2],
                "offset_mapping": [(0, 0), *content_offsets, (0, 0)],
                "special_tokens_mask": [1, *([0] * len(content_ids)), 1],
            }
        return {
            "input_ids": content_ids,
            "offset_mapping": content_offsets,
            "special_tokens_mask": [0] * len(content_ids),
        }


def _model_input(solver_words: int = 4):
    return {
        "problem": {
            "passage": "passage one two",
            "question": "question one",
            "options": {
                "A": "alpha choice",
                "B": "beta choice",
                "C": "gamma choice",
                "D": "delta choice",
            },
        },
        "solver": {
            "raw_output": " ".join(
                f"solver{index}" for index in range(solver_words)
            ),
            "parse_status": {
                "strict_answer": "A",
                "tolerant_answer": "A",
                "tolerant_match_count": 1,
                "tolerant_conflict": False,
            },
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "calls": 1,
                "latency_seconds": 0.0,
            },
        },
    }


def test_field_boundaries_exactly_reproduce_frozen_renderer():
    model_input = _model_input()
    text, fields = render_feature_text_with_spans(model_input)
    assert text == _render_feature_text(model_input)
    assert tuple(fields) == FIELD_NAMES
    assert text[fields["passage"].start : fields["passage"].end] == (
        model_input["problem"]["passage"]
    )
    assert text[fields["option_D"].start : fields["option_D"].end] == (
        model_input["problem"]["options"]["D"]
    )
    assert text[
        fields["solver_raw_output"].start : fields["solver_raw_output"].end
    ] == model_input["solver"]["raw_output"]
    parse_text = text[fields["parse_status"].start : fields["parse_status"].end]
    assert parse_text.startswith("strict_answer=A\n")
    assert parse_text.endswith("tolerant_conflict=false\n")


def test_no_truncation_fully_retains_every_field_and_special_tokens():
    result = analyze_rendered_input(_model_input(), TinyOffsetTokenizer(), 200)
    assert result["exceeds_limit"] is False
    assert result["untruncated_tokens"] == result["actual_retained_tokens"]
    assert result["untruncated_special_tokens"] == 2
    assert result["retained_special_tokens"] == 2
    assert all(
        field["status"] == "fully_retained"
        for field in result["fields"].values()
    )
    assert result["solver_output_retention_ratio"] == 1.0
    assert result["parse_status_retention_ratio"] == 1.0


def test_solver_and_parse_fields_can_be_partially_truncated():
    tokenizer = TinyOffsetTokenizer()
    solver_partial = None
    parse_partial = None
    for maximum in range(3, 80):
        result = analyze_rendered_input(_model_input(12), tokenizer, maximum)
        if result["fields"]["solver_raw_output"]["status"] == "partially_retained":
            solver_partial = result
        if result["fields"]["parse_status"]["status"] == "partially_retained":
            parse_partial = result
    assert solver_partial is not None
    assert 0 < solver_partial["solver_output_retention_ratio"] < 1
    assert parse_partial is not None
    assert 0 < parse_partial["parse_status_retention_ratio"] < 1


def test_late_fields_can_be_fully_dropped():
    result = analyze_rendered_input(_model_input(8), TinyOffsetTokenizer(), 8)
    assert result["exceeds_limit"] is True
    assert result["fields"]["solver_raw_output"]["status"] == "fully_dropped"
    assert result["fields"]["parse_status"]["status"] == "fully_dropped"
    assert result["solver_output_retention_ratio"] == 0.0
    assert result["parse_status_retention_ratio"] == 0.0


def test_individual_field_limit_and_special_tokens_are_handled_separately():
    result = analyze_rendered_input(_model_input(20), TinyOffsetTokenizer(), 10)
    solver = result["fields"]["solver_raw_output"]
    assert solver["individual_encoded_tokens"] == 22
    assert solver["individual_exceeds_limit"] is True
    # Special tokens count toward sequence length but cannot be attributed to a field.
    attributed_full = sum(
        field["concatenated_full_tokens"]
        for field in result["fields"].values()
    )
    assert attributed_full <= result["untruncated_content_tokens"]
    assert result["untruncated_tokens"] == (
        result["untruncated_content_tokens"] + 2
    )
    assert result["actual_retained_tokens"] == 10


def test_summary_reports_retention_statuses():
    tokenizer = TinyOffsetTokenizer()
    first = analyze_rendered_input(_model_input(), tokenizer, 200)
    second = analyze_rendered_input(_model_input(), tokenizer, 8)
    summary = summarize_cases([first, second])
    assert summary["samples"] == 2
    assert summary["exceeds_limit_count"] == 1
    assert summary["fields"]["solver_raw_output"]["status_counts"] == {
        "fully_retained": 1,
        "partially_retained": 0,
        "fully_dropped": 1,
    }


def test_representation_audit_cli_never_initializes_backend(monkeypatch, tmp_path):
    import hierarchical_control.cli as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("Representation audit must not initialize backend")

    received = {}

    def fake_audit(**kwargs):
        received.update(kwargs)
        return {
            "offline_audit": True,
            "controller_v3_implemented": False,
            "controller_trained": False,
            "model_calls": 0,
            "embedding_forward_calls": 0,
            "final_test_evaluated": False,
            "deployable": False,
        }

    monkeypatch.setattr(cli, "OpenAIBackend", forbidden)
    monkeypatch.setattr(cli, "MockBackend", forbidden)
    monkeypatch.setattr(cli, "run_precritic_representation_audit", fake_audit)
    output = tmp_path / "audit"
    assert main(
        ["audit-precritic-representation", "--output-dir", str(output)]
    ) == 0
    assert received["output_dir"] == str(output)
    assert "backend" not in received and "encoder" not in received


def test_slow_tokenizer_is_rejected():
    tokenizer = TinyOffsetTokenizer()
    tokenizer.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        analyze_rendered_input(_model_input(), tokenizer, 32)

