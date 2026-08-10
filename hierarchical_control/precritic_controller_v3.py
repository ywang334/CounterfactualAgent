"""Field-aware Pre-Critic Controller v3 architecture and offline smoke test.

This module deliberately contains no training or policy-selection routine.  It
only builds the field-independent feature pipeline, executes frozen local
MiniLM embeddings, and verifies a randomly initialized controller forward pass.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn

from .io_utils import write_jsonl
from .logiqa_pilot import ANSWER_LETTERS
from .precritic_controller_v1 import (
    DEFAULT_FINAL_MANIFEST,
    DEFAULT_TRAINING,
    DEFAULT_TRAINING_MANIFEST,
    MODEL_NAME,
    _file_sha256,
    load_training_examples,
    verify_sealed_final_manifest,
)
from .precritic_probe import NUMERIC_FEATURES, _numeric_features
from .precritic_representation_audit import load_local_tokenizer_bundle


DEFAULT_OUTPUT = Path("artifacts/precritic_controller_v3_smoke")
SMOKE_SEED = 20260821
DEFAULT_SMOKE_SAMPLES = 8
EMBEDDING_DIM = 384
MODEL_DIM = 128
NUM_HEADS = 4
FEEDFORWARD_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.1

FIELD_TYPES = (
    "cls",
    "passage",
    "question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "solver_output",
    "state",
)
FIELD_TYPE_TO_ID = {name: index for index, name in enumerate(FIELD_TYPES)}
FIXED_TEXT_FIELDS = (
    "passage",
    "question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
)
PARSE_STATUS_CATEGORIES = (
    "both_parsed_agree",
    "strict_only",
    "tolerant_only",
    "parsed_conflict",
    "unparsed",
)
ANSWER_CATEGORIES = (*ANSWER_LETTERS, "NONE")
STRUCTURED_STATE_FEATURES = (
    *NUMERIC_FEATURES,
    *(f"parse_status_{name}" for name in PARSE_STATUS_CATEGORIES),
    *(f"solver_answer_{name}" for name in ANSWER_CATEGORIES),
)
EXPECTED_TRAINABLE_PARAMETER_MIN = 400_000
EXPECTED_TRAINABLE_PARAMETER_MAX = 600_000

HISTORICAL_DIRS = (
    Path("artifacts/precritic_controller_v1"),
    Path("artifacts/precritic_controller_v2_factorized"),
)
HISTORICAL_FILES = (
    "primary_model.pt",
    "seed_metrics.json",
    "oof_predictions.jsonl",
    "validation_predictions.jsonl",
    "summary.json",
    "report.md",
)


class FieldTextEncoder(Protocol):
    """Injectable interface used by the field pipeline and unit tests."""

    name: str
    dimension: int
    max_seq_length: int
    local_files_only: bool
    mock_only: bool
    embedding_forward_calls: int

    def content_token_ids(self, text: str) -> list[int]: ...

    def decode_content_ids(self, token_ids: Sequence[int]) -> str: ...

    def sequence_token_length(self, text: str) -> int: ...

    def special_token_count(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class LocalMiniLMFieldEncoder:
    """Frozen, local-only all-MiniLM-L6-v2 field encoder."""

    mock_only = False
    local_files_only = True

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
        *,
        model_factory: Any | None = None,
        tokenizer_bundle: Any | None = None,
    ) -> None:
        bundle = tokenizer_bundle or load_local_tokenizer_bundle(model_name)
        if model_factory is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment error
                raise RuntimeError(
                    "sentence-transformers is required for the local MiniLM smoke"
                ) from exc
            model_factory = SentenceTransformer
        try:
            self.model = model_factory(
                bundle.snapshot_path,
                device=device,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Local all-MiniLM-L6-v2 is unavailable; downloading is forbidden"
            ) from exc
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = self.model.tokenizer
        self.name = model_name
        self.snapshot_path = str(bundle.snapshot_path)
        self.max_seq_length = int(bundle.max_seq_length)
        model_limit = int(self.model.max_seq_length)
        if model_limit != self.max_seq_length:
            raise RuntimeError(
                "SentenceTransformer model and local config max_seq_length differ"
            )
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        if self.dimension != EMBEDDING_DIM:
            raise RuntimeError(
                f"Expected MiniLM dimension {EMBEDDING_DIM}, got {self.dimension}"
            )
        self.embedding_forward_calls = 0

    def content_token_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )
        return [int(value) for value in encoded["input_ids"]]

    def decode_content_ids(self, token_ids: Sequence[int]) -> str:
        return str(
            self.tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def sequence_token_length(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(encoded["input_ids"])

    def special_token_count(self) -> int:
        count = int(self.tokenizer.num_special_tokens_to_add(pair=False))
        if count < 0 or count >= self.max_seq_length:
            raise RuntimeError("Tokenizer returned an invalid special-token count")
        return count

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        values = list(texts)
        if not values:
            return torch.empty((0, self.dimension), dtype=torch.float32)
        lengths = [self.sequence_token_length(text) for text in values]
        if any(length > self.max_seq_length for length in lengths):
            raise ValueError("Field encoder refused an input requiring truncation")
        self.embedding_forward_calls += 1
        with torch.inference_mode():
            result = self.model.encode(
                values,
                batch_size=32,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return result.detach().to(dtype=torch.float32, device="cpu")

    def parameter_counts(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        return {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(total - trainable),
        }


@dataclass(frozen=True)
class SolverChunks:
    texts: tuple[str, ...]
    source_content_tokens: int
    chunk_content_tokens: tuple[int, ...]
    max_content_tokens_per_chunk: int


@dataclass(frozen=True)
class PreCriticV3Batch:
    text_embeddings: torch.Tensor
    type_ids: torch.Tensor
    padding_mask: torch.Tensor
    structured_state: torch.Tensor
    state_positions: torch.Tensor
    field_order: tuple[tuple[str, ...], ...]
    solver_chunk_counts: tuple[int, ...]
    solver_source_token_counts: tuple[int, ...]
    solver_chunk_token_counts: tuple[tuple[int, ...], ...]
    selected_answers: tuple[str, ...]
    parse_statuses: tuple[str, ...]


def _one_hot(value: str, categories: Sequence[str]) -> tuple[float, ...]:
    if value not in categories:
        raise ValueError(f"Unknown structured category: {value}")
    return tuple(float(value == candidate) for candidate in categories)


def _validate_model_input(model_input: Mapping[str, Any]) -> None:
    if set(model_input) != {"problem", "solver"}:
        raise ValueError("Controller v3 input violates the top-level whitelist")
    problem = model_input.get("problem")
    solver = model_input.get("solver")
    if not isinstance(problem, dict) or not isinstance(solver, dict):
        raise ValueError("Controller v3 problem and Solver inputs must be objects")
    if set(problem) != {"passage", "question", "options"}:
        raise ValueError("Controller v3 problem input violates the whitelist")
    if not isinstance(problem["passage"], str) or not isinstance(
        problem["question"], str
    ):
        raise ValueError("Passage and question must be strings")
    options = problem.get("options")
    if not isinstance(options, dict) or set(options) != set(ANSWER_LETTERS):
        raise ValueError("Controller v3 requires exactly A-D options")
    if any(not isinstance(options[letter], str) for letter in ANSWER_LETTERS):
        raise ValueError("Every option must be a string")
    if set(solver) != {"raw_output", "parse_status", "usage"}:
        raise ValueError("Controller v3 Solver input violates the whitelist")
    if not isinstance(solver["raw_output"], str):
        raise ValueError("Solver raw output must be a string")
    parse = solver.get("parse_status")
    expected_parse = {
        "strict_answer",
        "strict_parse_failure",
        "tolerant_answer",
        "tolerant_parse_failure",
        "tolerant_match_count",
        "tolerant_conflict",
    }
    if not isinstance(parse, dict) or set(parse) != expected_parse:
        raise ValueError("Controller v3 parse status violates the whitelist")
    if not isinstance(parse["strict_parse_failure"], bool) or not isinstance(
        parse["tolerant_parse_failure"], bool
    ) or not isinstance(parse["tolerant_conflict"], bool):
        raise ValueError("Controller v3 parse booleans are invalid")
    if isinstance(parse["tolerant_match_count"], bool) or not isinstance(
        parse["tolerant_match_count"], int
    ) or parse["tolerant_match_count"] < 0:
        raise ValueError("Controller v3 tolerant match count is invalid")
    for name in ("strict_answer", "tolerant_answer"):
        if parse[name] is not None and parse[name] not in ANSWER_LETTERS:
            raise ValueError(f"Controller v3 {name} is invalid")
    usage = solver.get("usage")
    expected_usage = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "calls",
        "latency_seconds",
    }
    if not isinstance(usage, dict) or set(usage) != expected_usage:
        raise ValueError("Controller v3 Solver usage violates the whitelist")
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        if isinstance(usage[name], bool) or not isinstance(usage[name], int) or usage[name] < 0:
            raise ValueError(f"Controller v3 Solver usage {name} is invalid")
    if usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]:
        raise ValueError("Controller v3 Solver token identity failed")
    latency = usage["latency_seconds"]
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or latency < 0:
        raise ValueError("Controller v3 Solver latency is invalid")


def _parse_status_and_answer(
    model_input: Mapping[str, Any],
) -> tuple[str, str]:
    parse = model_input["solver"]["parse_status"]
    strict = parse["strict_answer"]
    tolerant = parse["tolerant_answer"]
    strict_valid = strict in ANSWER_LETTERS
    tolerant_valid = tolerant in ANSWER_LETTERS
    conflict = bool(parse["tolerant_conflict"]) or (
        strict_valid and tolerant_valid and strict != tolerant
    )
    if conflict:
        status = "parsed_conflict"
    elif strict_valid and tolerant_valid:
        status = "both_parsed_agree"
    elif strict_valid:
        status = "strict_only"
    elif tolerant_valid:
        status = "tolerant_only"
    else:
        status = "unparsed"
    selected = tolerant if tolerant_valid else strict if strict_valid else "NONE"
    return status, str(selected)


def structured_state_vector(
    model_input: Mapping[str, Any],
) -> tuple[torch.Tensor, str, str]:
    """Return the frozen eight numeric features plus explicit categorical state."""
    _validate_model_input(model_input)
    parse_status, answer = _parse_status_and_answer(model_input)
    values = (
        *_numeric_features(dict(model_input)),
        *_one_hot(parse_status, PARSE_STATUS_CATEGORIES),
        *_one_hot(answer, ANSWER_CATEGORIES),
    )
    if len(values) != len(STRUCTURED_STATE_FEATURES) or not all(
        math.isfinite(value) for value in values
    ):
        raise AssertionError("Controller v3 structured state is invalid")
    return torch.tensor(values, dtype=torch.float32), parse_status, answer


def chunk_solver_output(text: str, encoder: FieldTextEncoder) -> SolverChunks:
    """Deterministically cover every Solver tokenizer ID without truncation."""
    source_ids = encoder.content_token_ids(text)
    capacity = encoder.max_seq_length - encoder.special_token_count()
    if capacity <= 0:
        raise ValueError("Tokenizer leaves no capacity for Solver content")
    id_chunks = [
        source_ids[start : start + capacity]
        for start in range(0, len(source_ids), capacity)
    ]
    if not id_chunks:
        id_chunks = [[]]
    texts = tuple(encoder.decode_content_ids(chunk) for chunk in id_chunks)
    lengths = tuple(len(chunk) for chunk in id_chunks)
    if sum(lengths) != len(source_ids):
        raise AssertionError("Solver token chunks do not cover the source")
    sequence_lengths = [encoder.sequence_token_length(chunk) for chunk in texts]
    if any(length > encoder.max_seq_length for length in sequence_lengths):
        raise ValueError(
            "Decoded Solver chunk still exceeds max_seq_length; refusing truncation"
        )
    return SolverChunks(
        texts=texts,
        source_content_tokens=len(source_ids),
        chunk_content_tokens=lengths,
        max_content_tokens_per_chunk=capacity,
    )


def _fixed_texts(model_input: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    problem = model_input["problem"]
    return (
        ("passage", problem["passage"]),
        ("question", problem["question"]),
        ("option_A", problem["options"]["A"]),
        ("option_B", problem["options"]["B"]),
        ("option_C", problem["options"]["C"]),
        ("option_D", problem["options"]["D"]),
    )


def build_v3_feature_batch(
    model_inputs: Sequence[Mapping[str, Any]],
    encoder: FieldTextEncoder,
) -> PreCriticV3Batch:
    if not model_inputs:
        raise ValueError("Controller v3 batch cannot be empty")
    if encoder.dimension != EMBEDDING_DIM:
        raise ValueError(
            f"Controller v3 requires {EMBEDDING_DIM}d embeddings, got {encoder.dimension}"
        )

    flat_texts: list[str] = []
    records: list[dict[str, Any]] = []
    for model_input in model_inputs:
        _validate_model_input(model_input)
        texts = list(_fixed_texts(model_input))
        for name, text in texts:
            if encoder.sequence_token_length(text) > encoder.max_seq_length:
                raise ValueError(
                    f"Independent field {name} exceeds local MiniLM max_seq_length; "
                    "silent truncation is forbidden"
                )
        chunks = chunk_solver_output(model_input["solver"]["raw_output"], encoder)
        texts.extend(
            (f"solver_chunk_{index}", text)
            for index, text in enumerate(chunks.texts)
        )
        state, parse_status, answer = structured_state_vector(model_input)
        start = len(flat_texts)
        flat_texts.extend(text for _, text in texts)
        records.append(
            {
                "texts": texts,
                "flat_start": start,
                "state": state,
                "parse_status": parse_status,
                "answer": answer,
                "chunks": chunks,
            }
        )

    encoded = encoder.encode(flat_texts)
    if encoded.shape != (len(flat_texts), EMBEDDING_DIM):
        raise ValueError(
            "Field encoder returned an unexpected embedding batch shape"
        )
    sequence_lengths = [1 + len(record["texts"]) + 1 for record in records]
    batch_size = len(records)
    max_tokens = max(sequence_lengths)
    embeddings = torch.zeros(
        (batch_size, max_tokens, EMBEDDING_DIM), dtype=torch.float32
    )
    type_ids = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    padding_mask = torch.ones((batch_size, max_tokens), dtype=torch.bool)
    states = torch.stack([record["state"] for record in records])
    state_positions = torch.empty(batch_size, dtype=torch.long)
    orders: list[tuple[str, ...]] = []

    for row_index, record in enumerate(records):
        texts = record["texts"]
        order = ["cls", *(name for name, _ in texts), "state"]
        orders.append(tuple(order))
        padding_mask[row_index, : len(order)] = False
        type_ids[row_index, 0] = FIELD_TYPE_TO_ID["cls"]
        for local_index, (name, _) in enumerate(texts, 1):
            flat_index = record["flat_start"] + local_index - 1
            embeddings[row_index, local_index] = encoded[flat_index]
            field_type = "solver_output" if name.startswith("solver_chunk_") else name
            type_ids[row_index, local_index] = FIELD_TYPE_TO_ID[field_type]
        state_position = len(order) - 1
        state_positions[row_index] = state_position
        type_ids[row_index, state_position] = FIELD_TYPE_TO_ID["state"]

    return PreCriticV3Batch(
        text_embeddings=embeddings,
        type_ids=type_ids,
        padding_mask=padding_mask,
        structured_state=states,
        state_positions=state_positions,
        field_order=tuple(orders),
        solver_chunk_counts=tuple(len(record["chunks"].texts) for record in records),
        solver_source_token_counts=tuple(
            record["chunks"].source_content_tokens for record in records
        ),
        solver_chunk_token_counts=tuple(
            record["chunks"].chunk_content_tokens for record in records
        ),
        selected_answers=tuple(record["answer"] for record in records),
        parse_statuses=tuple(record["parse_status"] for record in records),
    )


class PreCriticControllerV3(nn.Module):
    """Field tokens -> 2-layer Transformer -> factorized and auxiliary heads."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(EMBEDDING_DIM, MODEL_DIM)
        self.field_type_embedding = nn.Embedding(len(FIELD_TYPES), MODEL_DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, MODEL_DIM))
        self.state_projection = nn.Linear(len(STRUCTURED_STATE_FEATURES), MODEL_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS,
            norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.solver_error_head = nn.Linear(MODEL_DIM, 1)
        self.critic_fix_head = nn.Linear(MODEL_DIM, 1)
        self.critic_harm_head = nn.Linear(MODEL_DIM, 1)
        self.transition_aux_head = nn.Linear(MODEL_DIM, 4)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, batch: PreCriticV3Batch) -> dict[str, torch.Tensor]:
        embeddings = batch.text_embeddings
        if embeddings.ndim != 3 or embeddings.shape[-1] != EMBEDDING_DIM:
            raise ValueError("Controller v3 text embeddings have invalid shape")
        batch_size, sequence_length, _ = embeddings.shape
        expected_matrix = (batch_size, sequence_length)
        if batch.type_ids.shape != expected_matrix or batch.padding_mask.shape != expected_matrix:
            raise ValueError("Controller v3 type IDs or padding mask do not align")
        if batch.structured_state.shape != (
            batch_size,
            len(STRUCTURED_STATE_FEATURES),
        ) or batch.state_positions.shape != (batch_size,):
            raise ValueError("Controller v3 structured state does not align")
        if torch.any(batch.padding_mask[:, 0]) or torch.any(
            batch.padding_mask[
                torch.arange(batch_size), batch.state_positions
            ]
        ):
            raise ValueError("CLS and state tokens cannot be padded")

        hidden = self.input_projection(embeddings)
        hidden = hidden.clone()
        hidden[:, 0, :] = self.cls_token.expand(batch_size, -1, -1)[:, 0, :]
        hidden[
            torch.arange(batch_size), batch.state_positions
        ] = self.state_projection(batch.structured_state)
        hidden = hidden + self.field_type_embedding(batch.type_ids)
        encoded = self.transformer(
            hidden,
            src_key_padding_mask=batch.padding_mask,
        )
        pooled = encoded[:, 0]
        solver_error_logits = self.solver_error_head(pooled).squeeze(-1)
        critic_fix_logits = self.critic_fix_head(pooled).squeeze(-1)
        critic_harm_logits = self.critic_harm_head(pooled).squeeze(-1)
        transition_logits = self.transition_aux_head(pooled)
        p_error = torch.sigmoid(solver_error_logits)
        p_fix = torch.sigmoid(critic_fix_logits)
        p_harm_given_correct = torch.sigmoid(critic_harm_logits)
        p_help = p_error * p_fix
        p_harm = (1.0 - p_error) * p_harm_given_correct
        factorized_transition = torch.stack(
            (
                (1.0 - p_error) * (1.0 - p_harm_given_correct),
                p_harm,
                p_help,
                p_error * (1.0 - p_fix),
            ),
            dim=-1,
        )
        return {
            "solver_error_logits": solver_error_logits,
            "critic_fix_logits": critic_fix_logits,
            "critic_harm_logits": critic_harm_logits,
            "transition_logits": transition_logits,
            "p_solver_error": p_error,
            "p_critic_fix_given_error": p_fix,
            "p_critic_harm_given_correct": p_harm_given_correct,
            "p_help": p_help,
            "p_harm": p_harm,
            "gate_score": p_help - p_harm,
            "factorized_transition_probabilities": factorized_transition,
            "transition_aux_probabilities": torch.softmax(transition_logits, dim=-1),
        }


def controller_parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


def _snapshot_critical_files(
    training_path: Path,
    training_manifest_path: Path,
    final_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {
        "training_examples": training_path,
        "training_manifest": training_manifest_path,
        "final_test_manifest": final_manifest_path,
        "controller_v1_source": Path(__file__).with_name("precritic_controller_v1.py"),
        "controller_v2_source": Path(__file__).with_name("precritic_controller_v2.py"),
    }
    for directory in HISTORICAL_DIRS:
        for name in HISTORICAL_FILES:
            path = directory / name
            if path.is_file():
                paths[f"{directory.name}/{name}"] = path
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Critical frozen input is missing: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return result


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _render_smoke_report(summary: Mapping[str, Any]) -> str:
    architecture = summary["architecture"]
    smoke = summary["smoke"]
    controller = summary["parameter_counts"]["controller"]
    encoder = summary["parameter_counts"]["minilm_encoder"]
    return "\n".join(
        (
            "# Pre-Critic Controller v3 Feature-Pipeline Smoke",
            "",
            "This is an offline architecture and feature-pipeline validation only. It is not training or policy evaluation.",
            "",
            "## Architecture",
            "",
            f"- Field order: `{architecture['base_field_order']}` plus one token per Solver chunk and a structured state token.",
            f"- Projection: {architecture['embedding_dim']} -> {architecture['d_model']}.",
            f"- Transformer: {architecture['layers']} layers, {architecture['nhead']} heads, FFN={architecture['feedforward_dim']}, dropout={architecture['dropout']}.",
            "- Heads: solver_error, critic_fix, critic_harm, and four-class transition auxiliary head.",
            "- Cost head: absent. Existing hard-budget and fixed cost-fallback semantics are unchanged.",
            "",
            "## Smoke result",
            "",
            f"- Samples: {smoke['samples']}.",
            f"- Batch text embedding shape: `{smoke['text_embedding_shape']}`.",
            f"- Solver chunks per sample: `{smoke['solver_chunk_counts']}`.",
            f"- Controller parameters: {controller['trainable']:,} trainable / {controller['total']:,} total.",
            f"- MiniLM parameters: {encoder['frozen']:,} frozen / {encoder['total']:,} total.",
            "- All factorized and four-class probabilities passed finite/range/sum checks.",
            "",
            "## Boundaries",
            "",
            "- local_files_only=true; mock_only=false",
            "- llm_calls=0; model_calls=0; backend_initialized=false",
            "- controller_trained=false; training_steps=0; backward_calls=0",
            "- oof_generated=false; threshold_selected=false; policy_evaluated=false",
            "- final_test_evaluated=false; final_test_examples_read=false",
            "- v1_v2_modified=false; prompt_modified=false; parser_modified=false",
            "- deployable=false",
            "",
        )
    )


def run_precritic_controller_v3_smoke(
    *,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    sample_count: int = DEFAULT_SMOKE_SAMPLES,
    encoder: FieldTextEncoder | None = None,
) -> dict[str, Any]:
    """Run the sole local MiniLM v3 feature-pipeline smoke; never train."""
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("Smoke sample_count must be a positive integer")
    training_path = Path(training_path)
    training_manifest_path = Path(training_manifest_path)
    final_test_manifest_path = Path(final_test_manifest_path)
    output_dir = Path(output_dir)
    targets = (
        output_dir / "summary.json",
        output_dir / "cases.jsonl",
        output_dir / "report.md",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError(
            "Controller v3 smoke outputs already exist; refusing to overwrite or rerun"
        )

    before = _snapshot_critical_files(
        training_path, training_manifest_path, final_test_manifest_path
    )
    examples, training_manifest = load_training_examples(
        training_path, training_manifest_path
    )
    final_guard = verify_sealed_final_manifest(
        final_test_manifest_path, training_manifest
    )
    if sample_count > len(examples):
        raise ValueError("Smoke sample_count exceeds frozen Training 1000")

    active_encoder = encoder or LocalMiniLMFieldEncoder()
    if active_encoder.mock_only:
        raise ValueError("Formal Controller v3 smoke forbids mock encoders")
    if not active_encoder.local_files_only:
        raise ValueError("Formal Controller v3 smoke requires local_files_only=True")
    ranked = sorted(
        enumerate(examples),
        key=lambda item: (
            -len(
                active_encoder.content_token_ids(
                    item[1].model_input["solver"]["raw_output"]
                )
            ),
            item[0],
        ),
    )
    selected = [example for _, example in ranked[:sample_count]]
    batch = build_v3_feature_batch(
        [example.model_input for example in selected], active_encoder
    )
    torch.manual_seed(SMOKE_SEED)
    controller = PreCriticControllerV3().to("cpu")
    controller.eval()
    with torch.inference_mode():
        outputs = controller(batch)

    controller_counts = controller_parameter_counts(controller)
    if not (
        EXPECTED_TRAINABLE_PARAMETER_MIN
        <= controller_counts["trainable"]
        <= EXPECTED_TRAINABLE_PARAMETER_MAX
    ):
        raise AssertionError("Controller v3 trainable parameter count is out of range")
    encoder_counts = (
        active_encoder.parameter_counts()
        if hasattr(active_encoder, "parameter_counts")
        else {"total": 0, "trainable": 0, "frozen": 0}
    )
    if encoder_counts["trainable"] != 0:
        raise AssertionError("MiniLM encoder is not fully frozen")
    factorized = outputs["factorized_transition_probabilities"]
    auxiliary = outputs["transition_aux_probabilities"]
    factor_probabilities = torch.stack(
        (
            outputs["p_solver_error"],
            outputs["p_critic_fix_given_error"],
            outputs["p_critic_harm_given_correct"],
        ),
        dim=-1,
    )
    probability_valid = bool(
        torch.isfinite(factor_probabilities).all()
        and torch.all((factor_probabilities >= 0) & (factor_probabilities <= 1))
        and torch.isfinite(factorized).all()
        and torch.all((factorized >= 0) & (factorized <= 1))
        and torch.allclose(
            factorized.sum(dim=-1),
            torch.ones(factorized.shape[0]),
            rtol=0.0,
            atol=1e-6,
        )
        and torch.isfinite(auxiliary).all()
        and torch.all((auxiliary >= 0) & (auxiliary <= 1))
        and torch.allclose(
            auxiliary.sum(dim=-1),
            torch.ones(auxiliary.shape[0]),
            rtol=0.0,
            atol=1e-6,
        )
    )
    if not probability_valid:
        raise AssertionError("Controller v3 probability validation failed")

    cases = []
    for index, example in enumerate(selected):
        cases.append(
            {
                "sample_id": example.sample_id,
                "source_dataset": example.source_dataset,
                "question_id": example.question_id,
                "field_order": list(batch.field_order[index]),
                "solver_source_content_tokens": batch.solver_source_token_counts[index],
                "solver_chunk_count": batch.solver_chunk_counts[index],
                "solver_chunk_content_tokens": list(
                    batch.solver_chunk_token_counts[index]
                ),
                "selected_answer": batch.selected_answers[index],
                "parse_status": batch.parse_statuses[index],
                "state_position": int(batch.state_positions[index]),
                "unpadded_sequence_tokens": int(
                    (~batch.padding_mask[index]).sum().item()
                ),
            }
        )

    after = _snapshot_critical_files(
        training_path, training_manifest_path, final_test_manifest_path
    )
    unchanged = before == after
    if not unchanged:
        raise AssertionError("A frozen input or v1/v2 historical artifact changed")
    architecture = {
        "embedding_dim": EMBEDDING_DIM,
        "d_model": MODEL_DIM,
        "input_projection": f"{EMBEDDING_DIM}->{MODEL_DIM}",
        "field_type_embeddings": len(FIELD_TYPES),
        "field_types": list(FIELD_TYPES),
        "base_field_order": [
            "cls",
            *FIXED_TEXT_FIELDS,
            "solver_chunk_0..N",
            "state",
        ],
        "structured_state_features": list(STRUCTURED_STATE_FEATURES),
        "structured_state_dim": len(STRUCTURED_STATE_FEATURES),
        "layers": NUM_LAYERS,
        "nhead": NUM_HEADS,
        "feedforward_dim": FEEDFORWARD_DIM,
        "dropout": DROPOUT,
        "factorized_heads": ["solver_error", "critic_fix", "critic_harm"],
        "transition_auxiliary_classes": 4,
        "cost_head": False,
        "hard_budget_logic_modified": False,
        "cost_fallback_logic_modified": False,
    }
    summary: dict[str, Any] = {
        "precritic_controller_v3_smoke": True,
        "architecture_only": True,
        "feature_pipeline_validated": True,
        "mock_only": False,
        "local_files_only": True,
        "deployable": False,
        "controller_trained": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_initialized": False,
        "oof_generated": False,
        "threshold_selected": False,
        "operating_point_selected": False,
        "policy_evaluated": False,
        "model_calls": 0,
        "llm_calls": 0,
        "backend_initialized": False,
        "final_test_evaluated": False,
        "final_test_examples_read": False,
        "v1_v2_modified": False,
        "prompt_modified": False,
        "parser_modified": False,
        "architecture": architecture,
        "encoder": {
            "name": active_encoder.name,
            "dimension": active_encoder.dimension,
            "max_seq_length": active_encoder.max_seq_length,
            "max_seq_length_source": "local SentenceTransformer configuration",
            "solver_chunking": "deterministic tokenizer content-token chunks",
            "silent_truncation": False,
            "embedding_forward_calls": active_encoder.embedding_forward_calls,
            "device": "cpu",
            "frozen": True,
        },
        "parameter_counts": {
            "controller": controller_counts,
            "minilm_encoder": encoder_counts,
            "controller_expected_trainable_range": [
                EXPECTED_TRAINABLE_PARAMETER_MIN,
                EXPECTED_TRAINABLE_PARAMETER_MAX,
            ],
        },
        "smoke": {
            "seed": SMOKE_SEED,
            "selection": "largest Solver tokenizer lengths; stable source-order tie break",
            "samples": len(selected),
            "text_embedding_shape": list(batch.text_embeddings.shape),
            "type_id_shape": list(batch.type_ids.shape),
            "padding_mask_shape": list(batch.padding_mask.shape),
            "structured_state_shape": list(batch.structured_state.shape),
            "state_positions_shape": list(batch.state_positions.shape),
            "solver_chunk_counts": list(batch.solver_chunk_counts),
            "solver_source_token_counts": list(batch.solver_source_token_counts),
            "factorized_logits_shape": list(outputs["solver_error_logits"].shape),
            "transition_logits_shape": list(outputs["transition_logits"].shape),
            "factorized_transition_probability_shape": list(factorized.shape),
            "transition_aux_probability_shape": list(auxiliary.shape),
            "probability_checks_passed": probability_valid,
            "controller_forward_calls": 1,
        },
        "final_test_guard": {
            **final_guard,
            "manifest_only_read": True,
            "examples_read": False,
            "model_calls": 0,
        },
        "integrity": {
            "before": before,
            "after": after,
            "all_frozen_inputs_and_history_unchanged": unchanged,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "cases.jsonl", cases)
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_write_text(output_dir / "report.md", _render_smoke_report(summary))
    return summary
