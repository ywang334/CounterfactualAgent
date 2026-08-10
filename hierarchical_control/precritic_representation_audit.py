"""Offline tokenizer-only audit of the frozen Pre-Critic input representation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .io_utils import write_jsonl
from .logiqa_action_collection import _problem_content_sha256
from .logiqa_pilot import ANSWER_LETTERS
from .precritic_controller_v1 import (
    DEFAULT_TRAINING,
    DEFAULT_TRAINING_MANIFEST,
    DEFAULT_VALIDATION,
    DEFAULT_FINAL_MANIFEST,
    LABELS,
    MODEL_NAME,
    _file_sha256,
    load_training_examples,
    load_validation_examples,
    verify_sealed_final_manifest,
)
from .precritic_probe import ProbeExample, _hash_model_input, _render_feature_text


DEFAULT_OUTPUT = Path("artifacts/precritic_representation_audit")
HISTORICAL_DIRS = (
    Path("artifacts/precritic_controller_v1"),
    Path("artifacts/precritic_controller_v2_factorized"),
)
HISTORICAL_NAMES = (
    "primary_model.pt",
    "seed_metrics.json",
    "oof_predictions.jsonl",
    "validation_predictions.jsonl",
    "summary.json",
    "report.md",
)
FIELD_NAMES = (
    "passage",
    "question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "solver_raw_output",
    "parse_status",
)
RETENTION_STATUSES = (
    "fully_retained",
    "partially_retained",
    "fully_dropped",
)


class OffsetTokenizer(Protocol):
    is_fast: bool
    model_max_length: int

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class LocalTokenizerBundle:
    tokenizer: OffsetTokenizer
    model_name: str
    max_seq_length: int
    snapshot_path: str
    sentence_transformer_config_path: str
    sentence_transformer_config_sha256: str
    tokenizer_class: str
    tokenizer_model_max_length: int
    local_files_only: bool = True
    embedding_forward_calls: int = 0


@dataclass(frozen=True)
class RenderedField:
    name: str
    start: int
    end: int
    text: str


def load_local_tokenizer_bundle(
    model_name: str = MODEL_NAME,
) -> LocalTokenizerBundle:
    """Load only a local tokenizer and SentenceTransformer sequence config."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError(
            "huggingface_hub and transformers are required for the offline audit"
        ) from exc
    try:
        snapshot = Path(
            snapshot_download(model_name, local_files_only=True)
        ).resolve()
    except Exception as exc:
        raise RuntimeError(
            f"Local MiniLM snapshot is unavailable for {model_name}; "
            "network download is forbidden"
        ) from exc
    config_path = snapshot / "sentence_bert_config.json"
    if not config_path.is_file():
        raise RuntimeError(
            "Local SentenceTransformer sentence_bert_config.json is missing"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        max_seq_length = int(config["max_seq_length"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Local SentenceTransformer max_seq_length is invalid"
        ) from exc
    if max_seq_length <= 2:
        raise RuntimeError("SentenceTransformer max_seq_length must exceed 2")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            use_fast=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Local MiniLM tokenizer is unavailable; network download is forbidden"
        ) from exc
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast local tokenizer with offset mapping is required"
        )
    return LocalTokenizerBundle(
        tokenizer=tokenizer,
        model_name=model_name,
        max_seq_length=max_seq_length,
        snapshot_path=str(snapshot),
        sentence_transformer_config_path=str(config_path),
        sentence_transformer_config_sha256=_file_sha256(config_path),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_model_max_length=int(tokenizer.model_max_length),
    )


def render_feature_text_with_spans(
    model_input: Mapping[str, Any],
) -> tuple[str, dict[str, RenderedField]]:
    """Rebuild span boundaries, then assert exact equality with the frozen renderer."""
    problem = model_input["problem"]
    solver = model_input["solver"]
    parse = solver["parse_status"]
    parts: list[str] = []
    fields: dict[str, RenderedField] = {}
    length = 0

    def append(text: str) -> None:
        nonlocal length
        parts.append(text)
        length += len(text)

    def append_field(name: str, value: Any) -> None:
        nonlocal length
        text = str(value)
        start = length
        append(text)
        fields[name] = RenderedField(name=name, start=start, end=length, text=text)

    append("<problem>\nPASSAGE: ")
    append_field("passage", problem["passage"])
    append("\nQUESTION: ")
    append_field("question", problem["question"])
    append("\n")
    for index, letter in enumerate(ANSWER_LETTERS):
        append(f"{letter}. ")
        append_field(f"option_{letter}", problem["options"][letter])
        if index + 1 < len(ANSWER_LETTERS):
            append("\n")
    append("\n</problem>\n<solver_response>\n")
    append_field("solver_raw_output", solver["raw_output"])
    append("\n</solver_response>\n<parse_status>\n")
    parse_text = (
        f"strict_answer={parse['strict_answer']}\n"
        f"tolerant_answer={parse['tolerant_answer']}\n"
        f"tolerant_match_count={parse['tolerant_match_count']}\n"
        f"tolerant_conflict={str(parse['tolerant_conflict']).lower()}\n"
    )
    append_field("parse_status", parse_text)
    append("</parse_status>")
    rendered = "".join(parts)
    frozen = _render_feature_text(dict(model_input))
    if rendered != frozen:
        raise AssertionError(
            "Span-aware reconstruction differs from frozen _render_feature_text"
        )
    if tuple(fields) != FIELD_NAMES:
        raise AssertionError("Field span order differs from the frozen representation")
    return frozen, fields


def _encoded(
    tokenizer: OffsetTokenizer,
    text: str,
    *,
    truncation: bool,
    max_length: int | None = None,
) -> dict[str, list[Any]]:
    kwargs: dict[str, Any] = {
        "add_special_tokens": True,
        "truncation": truncation,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    encoded = tokenizer(text, **kwargs)
    result = {
        key: list(encoded[key])
        for key in ("input_ids", "offset_mapping", "special_tokens_mask")
    }
    count = len(result["input_ids"])
    if not (
        len(result["offset_mapping"])
        == len(result["special_tokens_mask"])
        == count
    ):
        raise ValueError("Tokenizer IDs, offsets, and special-token mask do not align")
    normalized_offsets = []
    for offset in result["offset_mapping"]:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("Tokenizer returned an invalid offset")
        normalized_offsets.append((int(offset[0]), int(offset[1])))
    result["offset_mapping"] = normalized_offsets
    result["special_tokens_mask"] = [
        int(value) for value in result["special_tokens_mask"]
    ]
    return result


def _overlaps(offset: tuple[int, int], field: RenderedField) -> bool:
    start, end = offset
    return end > field.start and start < field.end


def _field_retention(
    field: RenderedField,
    full_encoding: Mapping[str, Sequence[Any]],
    retained_encoding: Mapping[str, Sequence[Any]],
) -> dict[str, Any]:
    full_offsets = [
        tuple(offset)
        for offset, special in zip(
            full_encoding["offset_mapping"],
            full_encoding["special_tokens_mask"],
        )
        if not special and _overlaps(tuple(offset), field)
    ]
    retained_offsets = Counter(
        tuple(offset)
        for offset, special in zip(
            retained_encoding["offset_mapping"],
            retained_encoding["special_tokens_mask"],
        )
        if not special and _overlaps(tuple(offset), field)
    )
    consumed: Counter[tuple[int, int]] = Counter()
    retained_count = 0
    for offset in full_offsets:
        if consumed[offset] < retained_offsets[offset]:
            retained_count += 1
            consumed[offset] += 1
    full_count = len(full_offsets)
    if full_count == 0 or retained_count == full_count:
        status = "fully_retained"
        ratio = 1.0
    elif retained_count == 0:
        status = "fully_dropped"
        ratio = 0.0
    else:
        status = "partially_retained"
        ratio = retained_count / full_count
    return {
        "status": status,
        "concatenated_full_tokens": full_count,
        "concatenated_retained_tokens": retained_count,
        "retention_ratio": ratio,
        "character_length": len(field.text),
    }


def analyze_rendered_input(
    model_input: Mapping[str, Any],
    tokenizer: OffsetTokenizer,
    max_seq_length: int,
) -> dict[str, Any]:
    if max_seq_length <= 2:
        raise ValueError("max_seq_length must exceed two special-token slots")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Offset analysis requires a fast tokenizer")
    text, fields = render_feature_text_with_spans(model_input)
    full_encoding = _encoded(tokenizer, text, truncation=False)
    retained_encoding = _encoded(
        tokenizer,
        text,
        truncation=True,
        max_length=max_seq_length,
    )
    full_length = len(full_encoding["input_ids"])
    retained_length = len(retained_encoding["input_ids"])
    exceeds = full_length > max_seq_length
    if exceeds and retained_length != max_seq_length:
        raise ValueError("Tokenizer did not enforce the configured truncation length")
    if not exceeds and retained_length != full_length:
        raise ValueError("Tokenizer changed an input that did not require truncation")
    field_results: dict[str, Any] = {}
    for name, field in fields.items():
        result = _field_retention(field, full_encoding, retained_encoding)
        individual = _encoded(tokenizer, field.text, truncation=False)
        individual_length = len(individual["input_ids"])
        field_results[name] = {
            **result,
            "individual_encoded_tokens": individual_length,
            "individual_exceeds_limit": individual_length > max_seq_length,
        }
    full_special = sum(full_encoding["special_tokens_mask"])
    retained_special = sum(retained_encoding["special_tokens_mask"])
    return {
        "max_seq_length": max_seq_length,
        "untruncated_tokens": full_length,
        "actual_retained_tokens": retained_length,
        "tokens_removed": full_length - retained_length,
        "exceeds_limit": exceeds,
        "truncated": exceeds,
        "untruncated_special_tokens": full_special,
        "retained_special_tokens": retained_special,
        "untruncated_content_tokens": full_length - full_special,
        "retained_content_tokens": retained_length - retained_special,
        "fields": field_results,
        "solver_output_retention_ratio": field_results[
            "solver_raw_output"
        ]["retention_ratio"],
        "parse_status_retention_ratio": field_results[
            "parse_status"
        ]["retention_ratio"],
        "rendered_text_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Percentiles require values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            key: 0.0
            for key in ("min", "mean", "p50", "p75", "p90", "p95", "p99", "max")
        }
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    if count == 0:
        return {
            "samples": 0,
            "exceeds_limit_count": 0,
            "exceeds_limit_rate": 0.0,
            "untruncated_token_length": _distribution([]),
            "actual_retained_token_length": _distribution([]),
            "tokens_removed": _distribution([]),
            "fields": {},
        }
    fields: dict[str, Any] = {}
    for name in FIELD_NAMES:
        status_counts = Counter(case["fields"][name]["status"] for case in cases)
        individually_exceeds = sum(
            bool(case["fields"][name]["individual_exceeds_limit"])
            for case in cases
        )
        fields[name] = {
            "status_counts": {
                status: status_counts.get(status, 0)
                for status in RETENTION_STATUSES
            },
            "status_rates": {
                status: status_counts.get(status, 0) / count
                for status in RETENTION_STATUSES
            },
            "retention_ratio": _distribution(
                [case["fields"][name]["retention_ratio"] for case in cases]
            ),
            "concatenated_full_tokens": _distribution(
                [case["fields"][name]["concatenated_full_tokens"] for case in cases]
            ),
            "concatenated_retained_tokens": _distribution(
                [
                    case["fields"][name]["concatenated_retained_tokens"]
                    for case in cases
                ]
            ),
            "individual_encoded_tokens": _distribution(
                [case["fields"][name]["individual_encoded_tokens"] for case in cases]
            ),
            "individual_exceeds_limit_count": individually_exceeds,
            "individual_exceeds_limit_rate": individually_exceeds / count,
        }
    exceeds = sum(bool(case["exceeds_limit"]) for case in cases)
    return {
        "samples": count,
        "exceeds_limit_count": exceeds,
        "exceeds_limit_rate": exceeds / count,
        "untruncated_token_length": _distribution(
            [case["untruncated_tokens"] for case in cases]
        ),
        "actual_retained_token_length": _distribution(
            [case["actual_retained_tokens"] for case in cases]
        ),
        "tokens_removed": _distribution(
            [case["tokens_removed"] for case in cases]
        ),
        "solver_output_retention_ratio": _distribution(
            [case["solver_output_retention_ratio"] for case in cases]
        ),
        "parse_status_retention_ratio": _distribution(
            [case["parse_status_retention_ratio"] for case in cases]
        ),
        "fields": fields,
    }


def _by_label(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        label: summarize_cases(
            [case for case in cases if case["label"] == label]
        )
        for label in LABELS
    }


def _severity_view(summary: Mapping[str, Any]) -> dict[str, Any]:
    solver = summary["fields"]["solver_raw_output"]
    parse = summary["fields"]["parse_status"]
    return {
        "samples": summary["samples"],
        "exceeds_limit_rate": summary["exceeds_limit_rate"],
        "untruncated_tokens_mean": summary["untruncated_token_length"]["mean"],
        "untruncated_tokens_p90": summary["untruncated_token_length"]["p90"],
        "solver_partial_or_dropped_rate": (
            solver["status_rates"]["partially_retained"]
            + solver["status_rates"]["fully_dropped"]
        ),
        "solver_fully_dropped_rate": solver["status_rates"]["fully_dropped"],
        "solver_mean_retention_ratio": solver["retention_ratio"]["mean"],
        "parse_partial_or_dropped_rate": (
            parse["status_rates"]["partially_retained"]
            + parse["status_rates"]["fully_dropped"]
        ),
        "parse_fully_dropped_rate": parse["status_rates"]["fully_dropped"],
        "parse_mean_retention_ratio": parse["retention_ratio"]["mean"],
    }


def corrected_degraded_analysis(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    corrected = summarize_cases(
        [case for case in cases if case["label"] == "wrong_to_correct"]
    )
    degraded = summarize_cases(
        [case for case in cases if case["label"] == "correct_to_wrong"]
    )
    other = summarize_cases(
        [
            case
            for case in cases
            if case["label"]
            not in {"wrong_to_correct", "correct_to_wrong"}
        ]
    )
    corrected_view = _severity_view(corrected)
    degraded_view = _severity_view(degraded)
    other_view = _severity_view(other)

    def more_severe(target: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
        return (
            target["exceeds_limit_rate"] > reference["exceeds_limit_rate"]
            or target["solver_partial_or_dropped_rate"]
            > reference["solver_partial_or_dropped_rate"]
            or target["parse_partial_or_dropped_rate"]
            > reference["parse_partial_or_dropped_rate"]
            or target["solver_mean_retention_ratio"]
            < reference["solver_mean_retention_ratio"]
            or target["parse_mean_retention_ratio"]
            < reference["parse_mean_retention_ratio"]
        )

    return {
        "corrected": corrected_view,
        "degraded": degraded_view,
        "other_labels": other_view,
        "corrected_more_severe_than_other_labels": more_severe(
            corrected_view, other_view
        ),
        "degraded_more_severe_than_other_labels": more_severe(
            degraded_view, other_view
        ),
        "comparison_is_descriptive_not_causal": True,
    }


def _sha_snapshot(
    training_path: Path,
    training_manifest_path: Path,
    validation_path: Path,
    final_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    paths = {
        "training_examples": training_path,
        "training_manifest": training_manifest_path,
        "validation_predictions": validation_path,
        "final_test_manifest": final_manifest_path,
        "representation_source": Path(__file__).with_name("precritic_probe.py"),
    }
    for directory in HISTORICAL_DIRS:
        for name in HISTORICAL_NAMES:
            paths[f"{directory.name}/{name}"] = directory / name
    result = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Critical input/history artifact missing: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return result


def _build_case(
    *,
    split: str,
    sample_id: str,
    question_id: str | int,
    label: str,
    model_input_sha256: str,
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    if label not in LABELS:
        raise ValueError("Representation case has an invalid transition label")
    return {
        "precritic_representation_audit": True,
        "offline_audit": True,
        "model_calls": 0,
        "embedding_forward_calls": 0,
        "split": split,
        "sample_id": sample_id,
        "question_id": question_id,
        "label": label,
        "model_input_sha256": model_input_sha256,
        **analysis,
    }


def _field_loss_counts(summary: Mapping[str, Any], field: str) -> dict[str, int]:
    counts = summary["fields"][field]["status_counts"]
    return {
        "fully_retained": counts["fully_retained"],
        "partially_retained": counts["partially_retained"],
        "fully_dropped": counts["fully_dropped"],
    }


def _report(summary: Mapping[str, Any]) -> str:
    overall = summary["statistics"]["overall"]
    training = summary["statistics"]["training_1000"]
    validation = summary["statistics"]["validation_100"]
    solver = _field_loss_counts(overall, "solver_raw_output")
    parse = _field_loss_counts(overall, "parse_status")
    comparison = summary["corrected_degraded_analysis"]["combined"]
    lines = [
        "# Pre-Critic Input Representation Audit",
        "",
        "Pure offline tokenizer analysis. No embedding forward, Controller training, "
        "LLM/backend call, prompt/parser change, or Final Test example read occurred.",
        "",
        "## Direct answers",
        "",
        f"1. {overall['exceeds_limit_count']}/{overall['samples']} combined samples "
        f"({overall['exceeds_limit_rate']:.2%}) exceed the locally configured "
        f"MiniLM limit of {summary['tokenizer']['max_seq_length']} tokens. "
        f"Training: {training['exceeds_limit_count']}/{training['samples']}; "
        f"Validation: {validation['exceeds_limit_count']}/{validation['samples']}.",
        f"2. Solver output retained/partial/dropped: "
        f"{solver['fully_retained']}/{solver['partially_retained']}/"
        f"{solver['fully_dropped']}. Parse status retained/partial/dropped: "
        f"{parse['fully_retained']}/{parse['partially_retained']}/"
        f"{parse['fully_dropped']}.",
        f"3. Corrected truncation is descriptively more severe than other labels: "
        f"{comparison['corrected_more_severe_than_other_labels']}. "
        f"Degraded truncation is more severe: "
        f"{comparison['degraded_more_severe_than_other_labels']}. "
        "These comparisons are descriptive, not causal.",
        f"4. Field-independent encoding recommended: "
        f"{summary['field_independent_encoding_assessment']['recommended']}. "
        f"{summary['field_independent_encoding_assessment']['reason']}",
        "",
        "## Tokenizer",
        "",
        f"- Model: {summary['tokenizer']['model_name']}",
        f"- SentenceTransformer max_seq_length: "
        f"{summary['tokenizer']['max_seq_length']}",
        f"- Tokenizer: {summary['tokenizer']['tokenizer_class']}",
        f"- local_files_only: {summary['tokenizer']['local_files_only']}",
        f"- embedding_forward_calls: {summary['tokenizer']['embedding_forward_calls']}",
        "",
        "## Split truncation",
        "",
        "| Split | N | Exceeded | Rate | Mean tokens | P50 | P90 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in (
        ("Training 1000", training),
        ("Validation 100", validation),
        ("Combined", overall),
    ):
        length = item["untruncated_token_length"]
        lines.append(
            f"| {name} | {item['samples']} | {item['exceeds_limit_count']} | "
            f"{item['exceeds_limit_rate']:.2%} | {length['mean']:.2f} | "
            f"{length['p50']:.1f} | {length['p90']:.1f} | "
            f"{length['p95']:.1f} | {length['p99']:.1f} | "
            f"{length['max']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Field retention, combined",
            "",
            "| Field | Full | Partial | Dropped | Mean retention | Individually exceeds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for field in FIELD_NAMES:
        metric = overall["fields"][field]
        counts = metric["status_counts"]
        lines.append(
            f"| {field} | {counts['fully_retained']} | "
            f"{counts['partially_retained']} | {counts['fully_dropped']} | "
            f"{metric['retention_ratio']['mean']:.4f} | "
            f"{metric['individual_exceeds_limit_count']} "
            f"({metric['individual_exceeds_limit_rate']:.2%}) |"
        )
    lines.extend(
        [
            "",
            "## Transition-label truncation",
            "",
            "| Split/label | N | Exceed rate | Mean tokens | Solver loss | Solver retained | Parse loss | Parse retained |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in ("training_1000", "validation_100"):
        for label in LABELS:
            metric = summary["statistics"][f"{split_name}_by_label"][label]
            if metric["samples"] == 0:
                continue
            solver_metric = metric["fields"]["solver_raw_output"]
            parse_metric = metric["fields"]["parse_status"]
            solver_loss = (
                solver_metric["status_counts"]["partially_retained"]
                + solver_metric["status_counts"]["fully_dropped"]
            )
            parse_loss = (
                parse_metric["status_counts"]["partially_retained"]
                + parse_metric["status_counts"]["fully_dropped"]
            )
            lines.append(
                f"| {split_name}/{label} | {metric['samples']} | "
                f"{metric['exceeds_limit_rate']:.2%} | "
                f"{metric['untruncated_token_length']['mean']:.2f} | "
                f"{solver_loss} | "
                f"{solver_metric['retention_ratio']['mean']:.4f} | "
                f"{parse_loss} | "
                f"{parse_metric['retention_ratio']['mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- offline_audit=true",
            "- controller_v3_implemented=false",
            "- controller_trained=false",
            "- model_calls=0",
            "- embedding_forward_calls=0",
            "- final_test_evaluated=false",
            "- deployable=false",
            "- critical inputs and v1/v2 artifacts unchanged=true",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run_precritic_representation_audit(
    *,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    validation_path: str | Path = DEFAULT_VALIDATION,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    tokenizer_bundle: LocalTokenizerBundle | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    targets = (
        output / "summary.json",
        output / "cases.jsonl",
        output / "report.md",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "Representation audit artifacts already exist; refusing to overwrite"
        )
    training_path = Path(training_path)
    training_manifest_path = Path(training_manifest_path)
    validation_path = Path(validation_path)
    final_manifest_path = Path(final_test_manifest_path)
    before_sha = _sha_snapshot(
        training_path,
        training_manifest_path,
        validation_path,
        final_manifest_path,
    )
    training_examples, training_manifest = load_training_examples(
        training_path, training_manifest_path
    )
    final_guard = verify_sealed_final_manifest(
        final_manifest_path, training_manifest
    )
    validation_examples, validation_sha = load_validation_examples(
        validation_path,
        training_manifest,
        {example.sample_id for example in training_examples},
    )
    bundle = tokenizer_bundle or load_local_tokenizer_bundle()

    cases = []
    for example in training_examples:
        if example.feature_text != _render_feature_text(example.model_input):
            raise ValueError("Training feature text differs from the frozen renderer")
        analysis = analyze_rendered_input(
            example.model_input, bundle.tokenizer, bundle.max_seq_length
        )
        model_input_sha = hashlib.sha256(
            json.dumps(
                example.model_input,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        cases.append(
            _build_case(
                split="training_1000",
                sample_id=example.sample_id,
                question_id=example.question_id,
                label=example.label,
                model_input_sha256=model_input_sha,
                analysis=analysis,
            )
        )
    for example in validation_examples:
        if example.feature_text != _render_feature_text(example.model_input):
            raise ValueError("Validation feature text differs from the frozen renderer")
        analysis = analyze_rendered_input(
            example.model_input, bundle.tokenizer, bundle.max_seq_length
        )
        cases.append(
            _build_case(
                split="validation_100",
                sample_id=_problem_content_sha256(example.model_input["problem"]),
                question_id=example.question_id,
                label=example.label,
                model_input_sha256=_hash_model_input(example),
                analysis=analysis,
            )
        )
    if len(cases) != 1100:
        raise ValueError("Representation audit requires exactly 1100 cases")
    if any(
        forbidden in case
        for case in cases
        for forbidden in ("gold", "critic", "critic_output", "refiner")
    ):
        raise AssertionError("Representation audit cases contain forbidden fields")

    training_cases = [
        case for case in cases if case["split"] == "training_1000"
    ]
    validation_cases = [
        case for case in cases if case["split"] == "validation_100"
    ]
    overall = summarize_cases(cases)
    statistics_payload = {
        "overall": overall,
        "training_1000": summarize_cases(training_cases),
        "validation_100": summarize_cases(validation_cases),
        "combined_by_label": _by_label(cases),
        "training_1000_by_label": _by_label(training_cases),
        "validation_100_by_label": _by_label(validation_cases),
    }
    corrected_degraded = {
        "combined": corrected_degraded_analysis(cases),
        "training_1000": corrected_degraded_analysis(training_cases),
        "validation_100": corrected_degraded_analysis(validation_cases),
    }
    solver_loss = (
        overall["fields"]["solver_raw_output"]["status_counts"][
            "partially_retained"
        ]
        + overall["fields"]["solver_raw_output"]["status_counts"]["fully_dropped"]
    )
    parse_loss = (
        overall["fields"]["parse_status"]["status_counts"]["partially_retained"]
        + overall["fields"]["parse_status"]["status_counts"]["fully_dropped"]
    )
    field_independent_recommended = solver_loss > 0 or parse_loss > 0
    after_sha = _sha_snapshot(
        training_path,
        training_manifest_path,
        validation_path,
        final_manifest_path,
    )
    if after_sha != before_sha:
        raise RuntimeError("Critical input or historical artifact changed during audit")
    tokenizer_payload = {
        "model_name": bundle.model_name,
        "max_seq_length": bundle.max_seq_length,
        "max_seq_length_source": "local sentence_bert_config.json",
        "snapshot_path": bundle.snapshot_path,
        "sentence_transformer_config_path": (
            bundle.sentence_transformer_config_path
        ),
        "sentence_transformer_config_sha256": (
            bundle.sentence_transformer_config_sha256
        ),
        "tokenizer_class": bundle.tokenizer_class,
        "tokenizer_model_max_length": bundle.tokenizer_model_max_length,
        "local_files_only": bundle.local_files_only,
        "embedding_forward_calls": bundle.embedding_forward_calls,
        "special_tokens_included_in_sequence_length": True,
        "field_attribution_excludes_special_tokens": True,
    }
    summary = {
        "precritic_representation_audit": True,
        "offline_audit": True,
        "controller_v3_implemented": False,
        "controller_trained": False,
        "model_backend_initialized": False,
        "model_calls": 0,
        "embedding_forward_calls": 0,
        "final_test_evaluated": False,
        "deployable": False,
        "prompt_modified": False,
        "parser_modified": False,
        "feature_renderer_modified": False,
        "tokenizer": tokenizer_payload,
        "data": {
            "training_samples": len(training_cases),
            "validation_samples": len(validation_cases),
            "total_samples": len(cases),
            "validation_sha256": validation_sha,
            "gold_written_to_cases": False,
            "critic_output_written_to_cases": False,
        },
        "final_test_guard": {
            **final_guard,
            "manifest_sha256_unchanged": True,
            "manifest_only_read": True,
            "examples_read": False,
            "model_calls": 0,
        },
        "statistics": statistics_payload,
        "corrected_degraded_analysis": corrected_degraded,
        "field_independent_encoding_assessment": {
            "recommended": field_independent_recommended,
            "reason": (
                "Late fields lose tokens under concatenation; independent field "
                "encoding would prevent earlier fields from consuming their token budget."
                if field_independent_recommended
                else "Solver output and parse status are fully retained in all audited samples."
            ),
            "evidence_only_for_future_controller_v3": True,
            "controller_v3_implemented": False,
        },
        "integrity": {
            "before": before_sha,
            "after": after_sha,
            "all_critical_inputs_and_history_unchanged": True,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "cases.jsonl", cases)
    _write_json_atomic(output / "summary.json", summary)
    _write_text_atomic(output / "report.md", _report(summary))
    return summary
