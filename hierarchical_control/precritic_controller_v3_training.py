"""Formal offline training protocol for field-aware Pre-Critic Controller v3."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .io_utils import read_jsonl, write_jsonl
from .logiqa_action_collection import _problem_content_sha256
from .precritic_controller_v1 import (
    ALL_SEEDS,
    BUDGET_RATES,
    DEFAULT_FINAL_MANIFEST,
    DEFAULT_TRAINING,
    DEFAULT_TRAINING_MANIFEST,
    DEFAULT_VALIDATION,
    EXPECTED_VALIDATION_SAMPLES,
    LABELS,
    N_SPLITS,
    PRIMARY_SEED,
    STABILITY_SEEDS,
    TRAINING_SHA256,
    TrainingExample,
    _average_precision,
    _file_sha256,
    _macro_f1,
    _mean_std,
    evaluate_validation_policy,
    gated_decisions,
    label_policy_metrics,
    load_training_examples,
    load_validation_examples,
    oof_budget_thresholds_v1,
    paired_mcnemar,
    select_oof_threshold_v1,
    verify_sealed_final_manifest,
)
from .precritic_controller_v2 import (
    factorized_head_metrics,
    factorized_targets,
)
from .precritic_probe import ProbeExample, _hash_model_input, deterministic_stratified_folds
from .precritic_controller_v3 import (
    ANSWER_CATEGORIES,
    EMBEDDING_DIM,
    EXPECTED_TRAINABLE_PARAMETER_MAX,
    EXPECTED_TRAINABLE_PARAMETER_MIN,
    FIELD_TYPES,
    MAX_CONTROLLER_SEQUENCE_LENGTH,
    MODEL_DIM,
    PARSE_STATUS_CATEGORIES,
    STRUCTURED_STATE_FEATURES,
    LocalMiniLMFieldEncoder,
    PreCriticControllerV3,
    PreCriticV3Batch,
    build_v3_feature_batch,
    controller_parameter_counts,
)
from .precritic_representation_audit import load_local_tokenizer_bundle


DEFAULT_OUTPUT = Path("artifacts/precritic_controller_v3")
DEFAULT_V1_DIR = Path("artifacts/precritic_controller_v1")
DEFAULT_V2_DIR = Path("artifacts/precritic_controller_v2_factorized")
CACHE_SCHEMA_VERSION = 1
TRAINING_SEED_PROTOCOL = ALL_SEEDS
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 50
GRADIENT_CLIP_NORM = 1.0
AUXILIARY_LOSS_WEIGHT = 0.25
NUMERIC_STATE_DIM = 8
CACHE_FILE = "feature_cache.pt"
OUTPUT_FILES = (
    CACHE_FILE,
    "primary_model.pt",
    "seed_metrics.json",
    "oof_predictions.jsonl",
    "validation_predictions.jsonl",
    "summary.json",
    "report.md",
)
HISTORICAL_OUTPUT_DIRS = (
    DEFAULT_V1_DIR,
    DEFAULT_V2_DIR,
    Path("artifacts/precritic_controller_v3_smoke"),
)
HISTORICAL_OUTPUT_NAMES = (
    "primary_model.pt",
    "seed_metrics.json",
    "oof_predictions.jsonl",
    "validation_predictions.jsonl",
    "summary.json",
    "report.md",
    "cases.jsonl",
)


@dataclass(frozen=True)
class CachedFeatureSplit:
    sample_hashes: tuple[str, ...]
    content_sha256: tuple[str, ...]
    text_embeddings: torch.Tensor
    type_ids: torch.Tensor
    position_ids: torch.Tensor
    padding_mask: torch.Tensor
    structured_state: torch.Tensor
    state_positions: torch.Tensor


@dataclass(frozen=True)
class StateNormalization:
    mean: torch.Tensor
    std: torch.Tensor
    fit_indices_sha256: str


@dataclass(frozen=True)
class LossBalance:
    positive_weights: Mapping[str, float]
    negative_weights: Mapping[str, float]
    auxiliary_class_weights: torch.Tensor
    counts: Mapping[str, Any]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _combined_sha256(paths: Sequence[Path]) -> dict[str, Any]:
    files = {str(path.resolve()): _file_sha256(path) for path in paths}
    return {"files": files, "combined_sha256": _canonical_sha256(files)}


def _directory_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Local encoder snapshot is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError("Local encoder snapshot contains no files")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        content = item.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _feature_schema() -> dict[str, Any]:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "embedding_dim": EMBEDDING_DIM,
        "controller_model_dim": MODEL_DIM,
        "max_controller_sequence_length": MAX_CONTROLLER_SEQUENCE_LENGTH,
        "field_types": list(FIELD_TYPES),
        "structured_state_features": list(STRUCTURED_STATE_FEATURES),
        "numeric_state_dimensions": NUMERIC_STATE_DIM,
        "categorical_state_dimensions": len(STRUCTURED_STATE_FEATURES) - NUMERIC_STATE_DIM,
        "parse_status_categories": list(PARSE_STATUS_CATEGORIES),
        "solver_answer_categories": list(ANSWER_CATEGORIES),
        "cache_tensor_fields": [
            "text_embeddings",
            "type_ids",
            "position_ids",
            "padding_mask",
            "structured_state",
            "state_positions",
        ],
        "cache_contains_labels": False,
        "cache_contains_gold": False,
        "cache_contains_continuation_outputs": False,
    }


def _critical_snapshot(
    training_path: Path,
    training_manifest_path: Path,
    validation_path: Path,
    final_manifest_path: Path,
    v1_dir: Path,
    v2_dir: Path,
) -> dict[str, dict[str, Any]]:
    paths: dict[str, Path] = {
        "training_examples": training_path,
        "training_manifest": training_manifest_path,
        "validation_predictions": validation_path,
        "final_test_manifest": final_manifest_path,
        "controller_v1_source": Path(__file__).with_name("precritic_controller_v1.py"),
        "controller_v2_source": Path(__file__).with_name("precritic_controller_v2.py"),
        "prompt_source": Path(__file__).with_name("logiqa_prompts.py"),
        "parser_source": Path(__file__).with_name("logiqa_pilot.py"),
    }
    for directory in (*HISTORICAL_OUTPUT_DIRS, v1_dir, v2_dir):
        for name in HISTORICAL_OUTPUT_NAMES:
            candidate = directory / name
            if candidate.is_file():
                paths[f"{directory.name}/{name}"] = candidate
    result = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Frozen input or historical artifact missing: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return result


def _model_input_sha(model_input: Mapping[str, Any]) -> str:
    return _canonical_sha256(model_input)


def _split_from_batch(
    batch: PreCriticV3Batch,
    start: int,
    end: int,
    sample_hashes: Sequence[str],
    content_hashes: Sequence[str],
) -> dict[str, Any]:
    if end - start != len(sample_hashes) or len(sample_hashes) != len(content_hashes):
        raise ValueError("Feature cache identity and tensor ranges do not align")
    return {
        "sample_hashes": list(sample_hashes),
        "content_sha256": list(content_hashes),
        "text_embeddings": batch.text_embeddings[start:end].clone().cpu(),
        "type_ids": batch.type_ids[start:end].clone().cpu(),
        "position_ids": batch.position_ids[start:end].clone().cpu(),
        "padding_mask": batch.padding_mask[start:end].clone().cpu(),
        "structured_state": batch.structured_state[start:end].clone().cpu(),
        "state_positions": batch.state_positions[start:end].clone().cpu(),
    }


def _forbidden_cache_key(value: Any) -> None:
    forbidden = {
        "gold",
        "label",
        "critic",
        "refiner",
        "outcome",
        "correct",
        "continuation",
        "raw_output",
    }
    if isinstance(value, dict):
        overlap = {str(key).casefold() for key in value} & forbidden
        if overlap:
            raise ValueError(f"Feature cache contains forbidden keys: {sorted(overlap)}")
        for child in value.values():
            _forbidden_cache_key(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _forbidden_cache_key(child)


def build_feature_cache_payload(
    *,
    batch: PreCriticV3Batch,
    training_examples: Sequence[TrainingExample],
    validation_examples: Sequence[ProbeExample],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    training_count = len(training_examples)
    validation_count = len(validation_examples)
    if batch.text_embeddings.shape[0] != training_count + validation_count:
        raise ValueError("Combined feature batch has the wrong sample count")
    training_input_hashes = [
        _model_input_sha(example.model_input) for example in training_examples
    ]
    validation_input_hashes = [
        _model_input_sha(example.model_input) for example in validation_examples
    ]
    training_content = [example.sample_id for example in training_examples]
    validation_content = [
        _problem_content_sha256(example.model_input["problem"])
        for example in validation_examples
    ]
    payload = {
        "metadata": dict(metadata),
        "training": _split_from_batch(
            batch,
            0,
            training_count,
            training_input_hashes,
            training_content,
        ),
        "validation": _split_from_batch(
            batch,
            training_count,
            training_count + validation_count,
            validation_input_hashes,
            validation_content,
        ),
    }
    _forbidden_cache_key(payload)
    return payload


def _cached_split(value: Mapping[str, Any], expected_samples: int) -> CachedFeatureSplit:
    required = {
        "sample_hashes",
        "content_sha256",
        "text_embeddings",
        "type_ids",
        "position_ids",
        "padding_mask",
        "structured_state",
        "state_positions",
    }
    if set(value) != required:
        raise ValueError("Feature cache split schema changed")
    hashes = tuple(value["sample_hashes"])
    contents = tuple(value["content_sha256"])
    embeddings = value["text_embeddings"]
    type_ids = value["type_ids"]
    position_ids = value["position_ids"]
    padding_mask = value["padding_mask"]
    state = value["structured_state"]
    state_positions = value["state_positions"]
    if len(hashes) != expected_samples or len(set(hashes)) != expected_samples:
        raise ValueError("Feature cache sample hashes are incomplete or duplicated")
    if len(contents) != expected_samples or len(set(contents)) != expected_samples:
        raise ValueError("Feature cache content hashes are incomplete or duplicated")
    if not isinstance(embeddings, torch.Tensor) or embeddings.shape[0] != expected_samples:
        raise ValueError("Feature cache embeddings are invalid")
    if embeddings.ndim != 3 or embeddings.shape[2] != EMBEDDING_DIM:
        raise ValueError("Feature cache embedding shape changed")
    matrix_shape = embeddings.shape[:2]
    if matrix_shape[1] > MAX_CONTROLLER_SEQUENCE_LENGTH:
        raise ValueError("Feature cache exceeds position capacity")
    if (
        not isinstance(type_ids, torch.Tensor)
        or type_ids.shape != matrix_shape
        or not isinstance(position_ids, torch.Tensor)
        or position_ids.shape != matrix_shape
        or not isinstance(padding_mask, torch.Tensor)
        or padding_mask.shape != matrix_shape
    ):
        raise ValueError("Feature cache mask/type/position tensors do not align")
    if state.shape != (expected_samples, len(STRUCTURED_STATE_FEATURES)):
        raise ValueError("Feature cache structured state shape changed")
    if state_positions.shape != (expected_samples,):
        raise ValueError("Feature cache state positions changed")
    if type_ids.dtype != torch.long or position_ids.dtype != torch.long:
        raise ValueError("Feature cache type and position IDs must be int64")
    if padding_mask.dtype != torch.bool or state.dtype != torch.float32:
        raise ValueError("Feature cache mask or state dtype changed")
    if embeddings.dtype != torch.float32 or not torch.isfinite(embeddings).all():
        raise ValueError("Feature cache embeddings are non-finite or wrong dtype")
    if not torch.isfinite(state).all():
        raise ValueError("Feature cache structured state is non-finite")
    if torch.any(type_ids < 0) or torch.any(type_ids >= len(FIELD_TYPES)):
        raise ValueError("Feature cache type IDs are out of range")
    if torch.any(position_ids < 0) or torch.any(
        position_ids >= MAX_CONTROLLER_SEQUENCE_LENGTH
    ):
        raise ValueError("Feature cache position IDs are out of range")
    return CachedFeatureSplit(
        sample_hashes=hashes,
        content_sha256=contents,
        text_embeddings=embeddings.cpu(),
        type_ids=type_ids.cpu(),
        position_ids=position_ids.cpu(),
        padding_mask=padding_mask.cpu(),
        structured_state=state.cpu(),
        state_positions=state_positions.cpu(),
    )


def _save_torch_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _cache_contract(
    *,
    training_path: Path,
    training_manifest_path: Path,
    validation_path: Path,
    final_manifest_path: Path,
    encoder_snapshot_path: Path,
    encoder_snapshot_sha256: str,
    encoder_config_sha256: str,
) -> dict[str, Any]:
    code = _combined_sha256(
        [Path(__file__), Path(__file__).with_name("precritic_controller_v3.py")]
    )
    schema = _feature_schema()
    return {
        "feature_cache": True,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "training_samples": 1000,
        "validation_samples": EXPECTED_VALIDATION_SAMPLES,
        "model_inputs_cached": False,
        "labels_cached": False,
        "gold_cached": False,
        "continuation_outputs_cached": False,
        "data_sha256": {
            "training_examples": _file_sha256(training_path),
            "training_manifest": _file_sha256(training_manifest_path),
            "validation_predictions": _file_sha256(validation_path),
            "final_test_manifest": _file_sha256(final_manifest_path),
        },
        "encoder": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "snapshot_path": str(encoder_snapshot_path.resolve()),
            "snapshot_sha256": encoder_snapshot_sha256,
            "sentence_transformer_config_sha256": encoder_config_sha256,
            "local_files_only": True,
            "frozen": True,
            "dimension": EMBEDDING_DIM,
        },
        "schema": schema,
        "schema_sha256": _canonical_sha256(schema),
        "code": code,
        "embedding_encode_calls": 1,
        "final_test_examples_read": False,
        "model_calls": 0,
        "backend_initialized": False,
    }


def build_or_load_feature_cache(
    *,
    cache_path: Path,
    training_examples: Sequence[TrainingExample],
    validation_examples: Sequence[ProbeExample],
    training_path: Path,
    training_manifest_path: Path,
    validation_path: Path,
    final_manifest_path: Path,
    device: str,
    encoder: Any | None = None,
    allow_mock_encoder: bool = False,
) -> tuple[CachedFeatureSplit, CachedFeatureSplit, dict[str, Any], bool, dict[str, int]]:
    bundle = load_local_tokenizer_bundle()
    snapshot = Path(bundle.snapshot_path)
    snapshot_sha = _directory_sha256(snapshot)
    contract = _cache_contract(
        training_path=training_path,
        training_manifest_path=training_manifest_path,
        validation_path=validation_path,
        final_manifest_path=final_manifest_path,
        encoder_snapshot_path=snapshot,
        encoder_snapshot_sha256=snapshot_sha,
        encoder_config_sha256=bundle.sentence_transformer_config_sha256,
    )
    expected_training_hashes = tuple(
        _model_input_sha(example.model_input) for example in training_examples
    )
    expected_validation_hashes = tuple(
        _model_input_sha(example.model_input) for example in validation_examples
    )
    expected_training_content = tuple(example.sample_id for example in training_examples)
    expected_validation_content = tuple(
        _problem_content_sha256(example.model_input["problem"])
        for example in validation_examples
    )
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        _forbidden_cache_key(payload)
        stored_metadata = dict(payload.get("metadata", {}))
        encoder_counts = stored_metadata.pop("encoder_parameter_counts", None)
        if stored_metadata != contract or not isinstance(encoder_counts, dict):
            raise ValueError("Feature cache metadata/code/data/encoder contract changed")
        training = _cached_split(payload["training"], len(training_examples))
        validation = _cached_split(payload["validation"], len(validation_examples))
        if (
            training.sample_hashes != expected_training_hashes
            or validation.sample_hashes != expected_validation_hashes
            or training.content_sha256 != expected_training_content
            or validation.content_sha256 != expected_validation_content
        ):
            raise ValueError("Feature cache sample identities changed")
        return training, validation, payload["metadata"], True, encoder_counts

    active_encoder = encoder or LocalMiniLMFieldEncoder(
        device=device,
        tokenizer_bundle=bundle,
    )
    if getattr(active_encoder, "mock_only", True) and not allow_mock_encoder:
        raise ValueError("Formal Controller v3 training forbids mock encoders")
    if not getattr(active_encoder, "local_files_only", False):
        raise ValueError("Controller v3 feature cache requires local_files_only=True")
    if active_encoder.embedding_forward_calls != 0:
        raise ValueError("Feature encoder must be fresh before one-time cache encoding")
    combined_inputs = [example.model_input for example in training_examples] + [
        example.model_input for example in validation_examples
    ]
    batch = build_v3_feature_batch(combined_inputs, active_encoder)
    if active_encoder.embedding_forward_calls != 1:
        raise AssertionError("Training and Validation fields must be encoded exactly once")
    encoder_counts = active_encoder.parameter_counts()
    if encoder_counts["trainable"] != 0:
        raise AssertionError("MiniLM must remain fully frozen")
    payload = build_feature_cache_payload(
        batch=batch,
        training_examples=training_examples,
        validation_examples=validation_examples,
        metadata={**contract, "encoder_parameter_counts": encoder_counts},
    )
    _save_torch_atomic(cache_path, payload)
    training = _cached_split(payload["training"], len(training_examples))
    validation = _cached_split(payload["validation"], len(validation_examples))
    return training, validation, payload["metadata"], False, encoder_counts


def fit_state_normalization(
    structured_state: torch.Tensor,
    training_indices: Sequence[int],
) -> StateNormalization:
    indices = torch.tensor(list(training_indices), dtype=torch.long)
    if indices.numel() == 0:
        raise ValueError("State normalization requires training indices")
    numeric = structured_state[indices, :NUMERIC_STATE_DIM]
    mean = numeric.mean(dim=0)
    std = numeric.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return StateNormalization(
        mean=mean,
        std=std,
        fit_indices_sha256=_canonical_sha256(indices.tolist()),
    )


def apply_state_normalization(
    structured_state: torch.Tensor,
    normalization: StateNormalization,
) -> torch.Tensor:
    result = structured_state.clone()
    result[:, :NUMERIC_STATE_DIM] = (
        result[:, :NUMERIC_STATE_DIM] - normalization.mean
    ) / normalization.std
    if not torch.equal(
        result[:, NUMERIC_STATE_DIM:],
        structured_state[:, NUMERIC_STATE_DIM:],
    ):
        raise AssertionError("Categorical state features must not be standardized")
    return result


def loss_balance(labels: Sequence[str]) -> LossBalance:
    targets = factorized_targets(labels)
    definitions = {
        "solver_error": (targets["solver_error"], targets["solver_error_mask"]),
        "critic_fix": (targets["critic_fix"], targets["critic_fix_mask"]),
        "critic_harm": (targets["critic_harm"], targets["critic_harm_mask"]),
    }
    positive = {}
    negative = {}
    counts: dict[str, Any] = {}
    for name, (target, mask) in definitions.items():
        active = target[mask]
        positives = int(active.sum())
        negatives = int(active.numel() - positives)
        if positives <= 0 or negatives <= 0:
            raise ValueError(f"Balanced {name} loss requires both classes")
        positive[name] = active.numel() / (2.0 * positives)
        negative[name] = active.numel() / (2.0 * negatives)
        counts[name] = {
            "active": int(active.numel()),
            "positive": positives,
            "negative": negatives,
        }
    class_indices = torch.tensor([LABELS.index(label) for label in labels])
    class_counts = torch.bincount(class_indices, minlength=len(LABELS)).float()
    if torch.any(class_counts == 0):
        raise ValueError("Auxiliary loss requires all four classes")
    auxiliary = len(labels) / (len(LABELS) * class_counts)
    counts["auxiliary"] = {
        label: int(class_counts[index]) for index, label in enumerate(LABELS)
    }
    return LossBalance(positive, negative, auxiliary, counts)


def _balanced_masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    active = mask.to(torch.bool)
    if not torch.any(active):
        return logits.sum() * 0.0
    selected_targets = targets[active]
    weights = torch.where(
        selected_targets > 0.5,
        torch.full_like(selected_targets, positive_weight),
        torch.full_like(selected_targets, negative_weight),
    )
    losses = F.binary_cross_entropy_with_logits(
        logits[active], selected_targets, reduction="none"
    )
    return torch.mean(losses * weights)


def v3_training_loss(
    outputs: Mapping[str, torch.Tensor],
    labels: Sequence[str],
    balance: LossBalance,
) -> dict[str, torch.Tensor]:
    targets = factorized_targets(labels)
    device = outputs["solver_error_logits"].device
    error = _balanced_masked_bce(
        outputs["solver_error_logits"],
        targets["solver_error"].to(device),
        targets["solver_error_mask"].to(device),
        balance.positive_weights["solver_error"],
        balance.negative_weights["solver_error"],
    )
    fix = _balanced_masked_bce(
        outputs["critic_fix_logits"],
        targets["critic_fix"].to(device),
        targets["critic_fix_mask"].to(device),
        balance.positive_weights["critic_fix"],
        balance.negative_weights["critic_fix"],
    )
    harm = _balanced_masked_bce(
        outputs["critic_harm_logits"],
        targets["critic_harm"].to(device),
        targets["critic_harm_mask"].to(device),
        balance.positive_weights["critic_harm"],
        balance.negative_weights["critic_harm"],
    )
    class_indices = torch.tensor(
        [LABELS.index(label) for label in labels],
        dtype=torch.long,
        device=device,
    )
    auxiliary = F.cross_entropy(
        outputs["transition_logits"],
        class_indices,
        weight=balance.auxiliary_class_weights.to(device),
    )
    total = error + fix + harm + AUXILIARY_LOSS_WEIGHT * auxiliary
    return {
        "total": total,
        "solver_error": error,
        "critic_fix": fix,
        "critic_harm": harm,
        "auxiliary": auxiliary,
    }


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _indexed_batch(
    features: CachedFeatureSplit,
    indices: Sequence[int] | torch.Tensor,
    normalized_state: torch.Tensor,
    device: torch.device,
) -> PreCriticV3Batch:
    index = torch.as_tensor(indices, dtype=torch.long)
    count = int(index.numel())
    return PreCriticV3Batch(
        text_embeddings=features.text_embeddings[index].to(device),
        type_ids=features.type_ids[index].to(device),
        position_ids=features.position_ids[index].to(device),
        padding_mask=features.padding_mask[index].to(device),
        structured_state=normalized_state[index].to(device),
        state_positions=features.state_positions[index].to(device),
        field_order=tuple(() for _ in range(count)),
        solver_chunk_counts=tuple(0 for _ in range(count)),
        solver_source_token_counts=tuple(0 for _ in range(count)),
        solver_chunk_token_counts=tuple(() for _ in range(count)),
        selected_answers=tuple("" for _ in range(count)),
        parse_statuses=tuple("" for _ in range(count)),
    )


def train_v3_model(
    *,
    features: CachedFeatureSplit,
    normalized_state: torch.Tensor,
    labels: Sequence[str],
    training_indices: Sequence[int],
    seed: int,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[PreCriticControllerV3, dict[str, Any]]:
    indices = list(training_indices)
    subset_labels = [labels[index] for index in indices]
    balance = loss_balance(subset_labels)
    _seed_everything(seed, device)
    model = PreCriticControllerV3().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    final_epoch: dict[str, float] = {}
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(indices), generator=generator).tolist()
        totals = Counter()
        samples = 0
        gradient_norm_max = 0.0
        for start in range(0, len(order), batch_size):
            local = order[start : start + batch_size]
            global_indices = [indices[position] for position in local]
            batch_labels = [labels[index] for index in global_indices]
            batch = _indexed_batch(
                features, global_indices, normalized_state, device
            )
            outputs = model(batch)
            losses = v3_training_loss(outputs, batch_labels, balance)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP_NORM
            )
            optimizer.step()
            size = len(global_indices)
            samples += size
            gradient_norm_max = max(gradient_norm_max, float(gradient_norm))
            for name, value in losses.items():
                totals[name] += float(value.detach()) * size
        final_epoch = {
            f"final_{name}_loss": totals[name] / samples for name in losses
        }
        final_epoch["final_epoch"] = epoch + 1
        final_epoch["maximum_preclip_gradient_norm"] = gradient_norm_max
    model.eval()
    return model, {
        **final_epoch,
        "training_samples": len(indices),
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "auxiliary_loss_weight": AUXILIARY_LOSS_WEIGHT,
        "loss_balance": {
            "positive_weights": dict(balance.positive_weights),
            "negative_weights": dict(balance.negative_weights),
            "auxiliary_class_weights": balance.auxiliary_class_weights.tolist(),
            "counts": dict(balance.counts),
        },
    }


def _predict(
    model: PreCriticControllerV3,
    features: CachedFeatureSplit,
    normalized_state: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    names = (
        "p_solver_error",
        "p_critic_fix_given_error",
        "p_critic_harm_given_correct",
        "p_help",
        "p_harm",
        "gate_score",
        "factorized_transition_probabilities",
        "transition_aux_probabilities",
    )
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            outputs = model(
                _indexed_batch(features, selected, normalized_state, device)
            )
            for name in names:
                pieces[name].append(outputs[name].detach().cpu())
    return {name: torch.cat(values, dim=0) for name, values in pieces.items()}


def _factorized_mapping(outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "p_error": outputs["p_solver_error"],
        "p_fix_given_error": outputs["p_critic_fix_given_error"],
        "p_harm_given_correct": outputs["p_critic_harm_given_correct"],
        "p_help": outputs["p_help"],
        "p_harm": outputs["p_harm"],
        "gate_score": outputs["gate_score"],
        "four_class": outputs["factorized_transition_probabilities"],
    }


def _prediction_disagreement(
    factorized: torch.Tensor,
    auxiliary: torch.Tensor,
) -> dict[str, Any]:
    first = factorized.argmax(dim=-1)
    second = auxiliary.argmax(dim=-1)
    different = first != second
    matrix = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.long)
    for left, right in zip(first.tolist(), second.tolist()):
        matrix[left, right] += 1
    return {
        "samples": int(first.numel()),
        "disagreements": int(different.sum()),
        "disagreement_rate": float(different.float().mean()),
        "factorized_rows_auxiliary_columns": {
            LABELS[row]: {
                LABELS[column]: int(matrix[row, column])
                for column in range(len(LABELS))
            }
            for row in range(len(LABELS))
        },
    }


def _head_metrics(
    outputs: Mapping[str, torch.Tensor], labels: Sequence[str]
) -> dict[str, Any]:
    factorized = _factorized_mapping(outputs)
    metrics = factorized_head_metrics(factorized, labels)
    metrics["factorized_four_class_macro_f1"] = metrics.pop(
        "four_class_macro_f1"
    )
    metrics["auxiliary_four_class_macro_f1"] = _macro_f1(
        labels, outputs["transition_aux_probabilities"]
    )
    metrics["factorized_auxiliary_disagreement"] = _prediction_disagreement(
        outputs["factorized_transition_probabilities"],
        outputs["transition_aux_probabilities"],
    )
    return metrics


def _probability_payload(outputs: Mapping[str, torch.Tensor], index: int) -> dict[str, Any]:
    return {
        "solver_error": float(outputs["p_solver_error"][index]),
        "critic_fix_given_solver_error": float(
            outputs["p_critic_fix_given_error"][index]
        ),
        "critic_harm_given_solver_correct": float(
            outputs["p_critic_harm_given_correct"][index]
        ),
        "helpful": float(outputs["p_help"][index]),
        "harmful": float(outputs["p_harm"][index]),
        "factorized_four_class": {
            label: float(outputs["factorized_transition_probabilities"][index, class_index])
            for class_index, label in enumerate(LABELS)
        },
        "auxiliary_four_class": {
            label: float(outputs["transition_aux_probabilities"][index, class_index])
            for class_index, label in enumerate(LABELS)
        },
    }


def _load_training_cost_records(
    training_path: Path,
    training_examples: Sequence[TrainingExample],
) -> list[dict[str, Any]]:
    rows = read_jsonl(training_path)
    if len(rows) != len(training_examples):
        raise ValueError("Training cost records do not align with Training 1000")
    records = []
    required = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "calls",
        "latency_seconds",
    }
    for row, example in zip(rows, training_examples):
        if row.get("sample_id") != example.sample_id:
            raise ValueError("Training cost record identity changed")
        solver = row["model_input"]["solver"]["usage"]
        critic = row.get("critic_cost_target")
        for name, value in (("solver", solver), ("critic", critic)):
            if value is None:
                if name == "solver" or example.cost_available:
                    raise ValueError("Training cost availability contract changed")
                continue
            if set(value) != required:
                raise ValueError(f"Training {name} cost schema changed")
            if value["prompt_tokens"] + value["completion_tokens"] != value["total_tokens"]:
                raise ValueError(f"Training {name} token identity failed")
        if (critic is not None) != example.cost_available:
            raise ValueError("Training Critic cost mask changed")
        records.append({"solver": dict(solver), "critic": None if critic is None else dict(critic)})
    return records


def _oof_policy_metrics(
    labels: Sequence[str],
    decisions: Sequence[bool],
    training_costs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = label_policy_metrics(labels, decisions)
    if len(training_costs) != len(labels):
        raise ValueError("OOF costs do not align with labels")
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
        "latency_seconds": 0.0,
    }
    unavailable_calls = 0
    for decision, record in zip(decisions, training_costs):
        stages = [record["solver"]]
        if decision:
            if record["critic"] is None:
                unavailable_calls += 1
            else:
                stages.append(record["critic"])
        for stage in stages:
            for field in totals:
                totals[field] += stage[field]
    exact = unavailable_calls == 0
    exact_total = totals if exact else None
    exact_mean = (
        {field: value / len(labels) for field, value in totals.items()}
        if exact
        else None
    )
    return {
        **policy,
        "cost": {
            "service_reported_usage": exact,
            "estimated": False,
            "available": exact,
            "unavailable_called_collection_200_samples": unavailable_calls,
            "total": exact_total,
            "mean": exact_mean,
            "calls": {
                "available": True,
                "solver_calls": len(labels),
                "incremental_critic_calls": sum(decisions),
                "total_calls": len(labels) + sum(decisions),
            },
        },
    }


def run_seed_training_v3(
    *,
    seed: int,
    training_examples: Sequence[TrainingExample],
    training_features: CachedFeatureSplit,
    training_costs: Sequence[Mapping[str, Any]],
    validation_examples: Sequence[ProbeExample],
    validation_features: CachedFeatureSplit,
    device: torch.device,
) -> tuple[PreCriticControllerV3, StateNormalization, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [example.label for example in training_examples]
    folds = deterministic_stratified_folds(labels, n_splits=N_SPLITS, seed=seed)
    output_shapes = {
        "p_solver_error": (len(labels),),
        "p_critic_fix_given_error": (len(labels),),
        "p_critic_harm_given_correct": (len(labels),),
        "p_help": (len(labels),),
        "p_harm": (len(labels),),
        "gate_score": (len(labels),),
        "factorized_transition_probabilities": (len(labels), len(LABELS)),
        "transition_aux_probabilities": (len(labels), len(LABELS)),
    }
    oof = {name: torch.empty(shape) for name, shape in output_shapes.items()}
    fold_training = []
    for fold in folds:
        normalization = fit_state_normalization(
            training_features.structured_state, fold["train"]
        )
        normalized = apply_state_normalization(
            training_features.structured_state, normalization
        )
        model, training_metrics = train_v3_model(
            features=training_features,
            normalized_state=normalized,
            labels=labels,
            training_indices=fold["train"],
            seed=seed + fold["fold"] + 1,
            device=device,
        )
        predictions = _predict(
            model,
            training_features,
            normalized,
            fold["validation"],
            device,
        )
        validation_index = torch.tensor(fold["validation"], dtype=torch.long)
        for name in oof:
            oof[name][validation_index] = predictions[name]
        fold_training.append(
            {
                "fold": fold["fold"],
                "training_samples": len(fold["train"]),
                "oof_samples": len(fold["validation"]),
                "normalization": {
                    "fit_scope": "fold_training_only",
                    "mean_first_8": normalization.mean.tolist(),
                    "std_first_8": normalization.std.tolist(),
                    "fit_indices_sha256": normalization.fit_indices_sha256,
                    "categorical_dimensions_standardized": False,
                },
                **training_metrics,
            }
        )
        print(
            json.dumps(
                {"seed": seed, "fold": fold["fold"], "status": "oof_complete"}
            ),
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scores = oof["gate_score"].tolist()
    threshold = select_oof_threshold_v1(scores, labels)
    budget_thresholds = oof_budget_thresholds_v1(scores, BUDGET_RATES)
    development_decisions = gated_decisions(scores, threshold["threshold"])
    oof_curve = [
        {
            **point,
            "oof_policy": _oof_policy_metrics(
                labels,
                gated_decisions(scores, point["threshold"]),
                training_costs,
            ),
        }
        for point in budget_thresholds
    ]

    full_indices = list(range(len(labels)))
    full_normalization = fit_state_normalization(
        training_features.structured_state, full_indices
    )
    normalized_training = apply_state_normalization(
        training_features.structured_state, full_normalization
    )
    normalized_validation = apply_state_normalization(
        validation_features.structured_state, full_normalization
    )
    final_model, final_training = train_v3_model(
        features=training_features,
        normalized_state=normalized_training,
        labels=labels,
        training_indices=full_indices,
        seed=seed,
        device=device,
    )
    validation_indices = list(range(len(validation_examples)))
    validation_outputs = _predict(
        final_model,
        validation_features,
        normalized_validation,
        validation_indices,
        device,
    )
    validation_labels = [example.label for example in validation_examples]
    validation_scores = validation_outputs["gate_score"].tolist()
    validation_decisions = gated_decisions(
        validation_scores, threshold["threshold"]
    )
    validation_curve = [
        {
            **point,
            "validation_policy": evaluate_validation_policy(
                validation_examples,
                gated_decisions(validation_scores, point["threshold"]),
            ),
        }
        for point in budget_thresholds
    ]

    fold_by_index = {
        index: fold["fold"] for fold in folds for index in fold["validation"]
    }
    oof_rows = []
    for index, example in enumerate(training_examples):
        oof_rows.append(
            {
                "controller_v3": True,
                "development_validation": False,
                "final_test_evaluated": False,
                "deployable": False,
                "model_calls": 0,
                "seed": seed,
                "primary_seed": seed == PRIMARY_SEED,
                "sample_id": example.sample_id,
                "source_dataset": example.source_dataset,
                "label": example.label,
                "oof_fold": fold_by_index[index],
                "probabilities": _probability_payload(oof, index),
                "gate_score": float(oof["gate_score"][index]),
                "development_threshold": threshold["threshold"],
                "critic_called": development_decisions[index],
                "cost_available": example.cost_available,
                "model_input_sha256": training_features.sample_hashes[index],
            }
        )
    validation_rows = []
    for index, example in enumerate(validation_examples):
        decision = validation_decisions[index]
        selected = example.critic_only_answer if decision else example.solver_answer
        validation_rows.append(
            {
                "controller_v3": True,
                "development_validation": True,
                "final_test_evaluated": False,
                "deployable": False,
                "model_calls": 0,
                "seed": seed,
                "primary_seed": seed == PRIMARY_SEED,
                "question_id": example.question_id,
                "label": example.label,
                "probabilities": _probability_payload(validation_outputs, index),
                "gate_score": float(validation_outputs["gate_score"][index]),
                "development_threshold": threshold["threshold"],
                "threshold_source": threshold["selection_source"],
                "validation_used_for_threshold": False,
                "critic_called": decision,
                "selected_answer": selected,
                "correct": selected == example.gold,
                "model_input_sha256": validation_features.sample_hashes[index],
            }
        )
    metrics = {
        "seed": seed,
        "role": "primary" if seed == PRIMARY_SEED else "stability_only",
        "used_for_model_selection": False,
        "oof": {
            "folds": N_SPLITS,
            "stratified": True,
            "fold_manifest": folds,
            "fold_training": fold_training,
            "development_threshold": threshold,
            "development_policy": _oof_policy_metrics(
                labels, development_decisions, training_costs
            ),
            "budget_curve": oof_curve,
            "head_metrics": _head_metrics(oof, labels),
        },
        "final_training": {
            **final_training,
            "state_normalization": {
                "fit_scope": "full_training_1000",
                "mean_first_8": full_normalization.mean.tolist(),
                "std_first_8": full_normalization.std.tolist(),
                "fit_indices_sha256": full_normalization.fit_indices_sha256,
                "categorical_dimensions_standardized": False,
            },
        },
        "validation": {
            "development_validation": True,
            "validation_used_for_training": False,
            "validation_used_for_threshold": False,
            "validation_used_for_model_selection": False,
            "controller_policy": evaluate_validation_policy(
                validation_examples, validation_decisions
            ),
            "head_metrics": _head_metrics(validation_outputs, validation_labels),
            "budget_curve": validation_curve,
        },
    }
    return final_model.cpu(), full_normalization, metrics, oof_rows, validation_rows


def _load_historical_primary_decisions(
    directory: Path,
    validation_examples: Sequence[ProbeExample],
    training_examples: Sequence[TrainingExample],
    *,
    version: str,
) -> tuple[list[bool], list[bool], dict[str, Any]]:
    validation_path = directory / "validation_predictions.jsonl"
    oof_path = directory / "oof_predictions.jsonl"
    summary_path = directory / "summary.json"
    for path in (validation_path, oof_path, summary_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Frozen {version} artifact missing: {path}")
    validation_rows = [
        row
        for row in read_jsonl(validation_path)
        if row.get("seed") == PRIMARY_SEED and row.get("primary_seed") is True
    ]
    oof_rows = [
        row
        for row in read_jsonl(oof_path)
        if row.get("seed") == PRIMARY_SEED and row.get("primary_seed") is True
    ]
    if len(validation_rows) != len(validation_examples) or len(oof_rows) != len(training_examples):
        raise ValueError(f"Frozen {version} primary predictions are incomplete")
    validation_decisions = []
    for row, example in zip(validation_rows, validation_examples):
        if (
            row.get("question_id") != example.question_id
            or row.get("model_input_sha256") != _hash_model_input(example)
            or not isinstance(row.get("critic_called"), bool)
        ):
            raise ValueError(f"Frozen {version} Validation identity changed")
        validation_decisions.append(row["critic_called"])
    oof_decisions = []
    for row, example in zip(oof_rows, training_examples):
        if row.get("sample_id") != example.sample_id or not isinstance(
            row.get("critic_called"), bool
        ):
            raise ValueError(f"Frozen {version} OOF identity changed")
        oof_decisions.append(row["critic_called"])
    return validation_decisions, oof_decisions, {
        "summary_sha256": _file_sha256(summary_path),
        "validation_predictions_sha256": _file_sha256(validation_path),
        "oof_predictions_sha256": _file_sha256(oof_path),
        "primary_seed": PRIMARY_SEED,
        "frozen_historical_policy": True,
    }


def _stability_summary(seed_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = {
        "accuracy": lambda item: item["validation"]["controller_policy"]["accuracy"],
        "corrected": lambda item: item["validation"]["controller_policy"]["corrected"],
        "degraded": lambda item: item["validation"]["controller_policy"]["degraded"],
        "net_benefit": lambda item: item["validation"]["controller_policy"]["net_benefit"],
        "critic_call_rate": lambda item: item["validation"]["controller_policy"]["critic_call_rate"],
        "mean_total_tokens": lambda item: item["validation"]["controller_policy"]["cost"]["mean"]["total_tokens"],
        "mean_calls": lambda item: item["validation"]["controller_policy"]["cost"]["mean"]["calls"],
        "mean_latency_seconds": lambda item: item["validation"]["controller_policy"]["cost"]["mean"]["latency_seconds"],
        "solver_error_pr_auc": lambda item: item["validation"]["head_metrics"]["solver_error"]["pr_auc"],
        "critic_fix_pr_auc": lambda item: item["validation"]["head_metrics"]["critic_fix_given_solver_error"]["pr_auc"],
        "critic_harm_pr_auc": lambda item: item["validation"]["head_metrics"]["critic_harm_given_solver_correct"]["pr_auc"],
        "helpful_pr_auc": lambda item: item["validation"]["head_metrics"]["final_helpful_pr_auc"],
        "harmful_pr_auc": lambda item: item["validation"]["head_metrics"]["final_harmful_pr_auc"],
        "factorized_macro_f1": lambda item: item["validation"]["head_metrics"]["factorized_four_class_macro_f1"],
        "auxiliary_macro_f1": lambda item: item["validation"]["head_metrics"]["auxiliary_four_class_macro_f1"],
        "head_disagreement_rate": lambda item: item["validation"]["head_metrics"]["factorized_auxiliary_disagreement"]["disagreement_rate"],
    }
    return {
        "seeds": [item["seed"] for item in seed_metrics],
        "primary_seed_fixed": PRIMARY_SEED,
        "best_seed_selected": False,
        "metrics": {
            name: _mean_std([float(accessor(item)) for item in seed_metrics])
            for name, accessor in paths.items()
        },
    }


def _report(summary: Mapping[str, Any]) -> str:
    primary = summary["primary_seed_metrics"]
    comparison = summary["development_validation"]["policy_comparison"]
    lines = [
        "# Pre-Critic Controller v3 Offline Training",
        "",
        "This is development validation only. Final Test 500 was not read or evaluated, and no deployment operating point was selected.",
        "",
        "## Fixed architecture and training",
        "",
        f"- Device: `{summary['training_protocol']['device']}`",
        f"- Trainable parameters: {summary['model']['parameter_counts']['trainable']:,}",
        "- Learned absolute position embeddings: 32 positions",
        "- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4, batch=32, epochs=50, gradient_clip=1.0",
        "- Loss: error + fix + harm + 0.25 * auxiliary transition CE",
        "- Gate score uses only factorized heads; the auxiliary head is diagnostic.",
        "- Cost head is absent; hard-budget and fixed cost-fallback semantics are unchanged.",
        "",
        "## Primary-seed OOF",
        "",
        f"- Corrected/degraded/net: {primary['oof']['development_policy']['corrected']}/{primary['oof']['development_policy']['degraded']}/{primary['oof']['development_policy']['net_benefit']}",
        f"- Critic call rate: {primary['oof']['development_policy']['critic_call_rate']:.4f}",
        "- OOF aggregate token and latency costs remain unavailable whenever selected samples come from Collection 200; no estimates are used.",
        "",
        "## Validation 100 policy comparison",
        "",
        "| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean total tokens | Mean calls | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("STOP", "ALWAYS_CRITIC_ONLY", "CONTROLLER_V1_PRIMARY", "CONTROLLER_V2_PRIMARY", "CONTROLLER_V3_PRIMARY"):
        item = comparison[name]
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item['corrected']} | {item['degraded']} | {item['net_benefit']} | {item['critic_call_rate']:.4f} | {item['cost']['mean']['total_tokens']:.2f} | {item['cost']['mean']['calls']:.4f} | {item['cost']['mean']['latency_seconds']:.4f} |"
        )
    heads = primary["validation"]["head_metrics"]
    lines.extend(
        [
            "",
            "## Validation diagnostics",
            "",
            f"- Solver-error PR-AUC/F1: {heads['solver_error']['pr_auc']:.4f}/{heads['solver_error']['f1']:.4f}",
            f"- Critic-fix PR-AUC/F1: {heads['critic_fix_given_solver_error']['pr_auc']:.4f}/{heads['critic_fix_given_solver_error']['f1']:.4f}",
            f"- Critic-harm PR-AUC/F1: {heads['critic_harm_given_solver_correct']['pr_auc']:.4f}/{heads['critic_harm_given_solver_correct']['f1']:.4f}",
            f"- Helpful/harmful PR-AUC: {heads['final_helpful_pr_auc']:.4f}/{heads['final_harmful_pr_auc']:.4f}",
            f"- Factorized/auxiliary macro-F1: {heads['factorized_four_class_macro_f1']:.4f}/{heads['auxiliary_four_class_macro_f1']:.4f}",
            f"- Factorized/auxiliary disagreement: {heads['factorized_auxiliary_disagreement']['disagreements']}/{heads['factorized_auxiliary_disagreement']['samples']}",
            "",
            "No claim that v3 is superior to v1 or v2 is made from this development validation.",
            "",
            "## Boundaries",
            "",
            "- model_calls=0; llm_calls=0; backend_initialized=false",
            "- final_test_evaluated=false; final_test_examples_read=false",
            "- prompt_modified=false; parser_modified=false; v1_v2_modified=false",
            "- rollout_collected=false; cost_head=false; deployable=false",
            "",
        ]
    )
    return "\n".join(lines)


def train_precritic_controller_v3(
    *,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    validation_path: str | Path = DEFAULT_VALIDATION,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    controller_v1_dir: str | Path = DEFAULT_V1_DIR,
    controller_v2_dir: str | Path = DEFAULT_V2_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    started = time.perf_counter()
    training_path = Path(training_path)
    training_manifest_path = Path(training_manifest_path)
    validation_path = Path(validation_path)
    final_test_manifest_path = Path(final_test_manifest_path)
    v1_dir = Path(controller_v1_dir)
    v2_dir = Path(controller_v2_dir)
    output = Path(output_dir)
    cache_path = output / CACHE_FILE
    completed_names = OUTPUT_FILES[1:]
    if any((output / name).exists() for name in completed_names):
        raise FileExistsError("Controller v3 artifacts already exist; refusing to overwrite")
    if cache_path.exists() and not cache_path.is_file():
        raise FileExistsError("Controller v3 feature cache path is not a file")

    before = _critical_snapshot(
        training_path,
        training_manifest_path,
        validation_path,
        final_test_manifest_path,
        v1_dir,
        v2_dir,
    )
    training_examples, training_manifest = load_training_examples(
        training_path, training_manifest_path
    )
    if _file_sha256(training_path) != TRAINING_SHA256:
        raise ValueError("Frozen Training 1000 SHA256 changed")
    final_guard = verify_sealed_final_manifest(
        final_test_manifest_path, training_manifest
    )
    training_ids = {example.sample_id for example in training_examples}
    validation_examples, validation_sha = load_validation_examples(
        validation_path, training_manifest, training_ids
    )
    labels = [example.label for example in training_examples]
    training_costs = _load_training_cost_records(training_path, training_examples)
    expected_counts = {
        "correct_to_correct": 635,
        "correct_to_wrong": 79,
        "wrong_to_correct": 64,
        "wrong_to_wrong": 222,
    }
    if dict(Counter(labels)) != expected_counts:
        raise ValueError("Frozen Training 1000 label contract changed")

    v1_validation, v1_oof, v1_sources = _load_historical_primary_decisions(
        v1_dir, validation_examples, training_examples, version="v1"
    )
    v2_validation, v2_oof, v2_sources = _load_historical_primary_decisions(
        v2_dir, validation_examples, training_examples, version="v2"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_details = {
        "type": device.type,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    cache_started = time.perf_counter()
    training_features, validation_features, cache_contract, cache_reused, encoder_counts = build_or_load_feature_cache(
        cache_path=cache_path,
        training_examples=training_examples,
        validation_examples=validation_examples,
        training_path=training_path,
        training_manifest_path=training_manifest_path,
        validation_path=validation_path,
        final_manifest_path=final_test_manifest_path,
        device=str(device),
    )
    cache_seconds = time.perf_counter() - cache_started
    if encoder_counts.get("trainable") != 0:
        raise AssertionError("Cached MiniLM parameter contract is not fully frozen")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    seed_metrics = []
    oof_rows = []
    validation_rows = []
    primary_model = None
    primary_normalization = None
    training_started = time.perf_counter()
    for seed in TRAINING_SEED_PROTOCOL:
        model, normalization, metrics, seed_oof, seed_validation = run_seed_training_v3(
            seed=seed,
            training_examples=training_examples,
            training_features=training_features,
            training_costs=training_costs,
            validation_examples=validation_examples,
            validation_features=validation_features,
            device=device,
        )
        seed_metrics.append(metrics)
        oof_rows.extend(seed_oof)
        validation_rows.extend(seed_validation)
        if seed == PRIMARY_SEED:
            primary_model = model
            primary_normalization = normalization
        print(json.dumps({"seed": seed, "status": "seed_complete"}), flush=True)
    training_seconds = time.perf_counter() - training_started
    if primary_model is None or primary_normalization is None:
        raise AssertionError("Fixed primary seed model was not trained")
    primary_metrics = next(item for item in seed_metrics if item["seed"] == PRIMARY_SEED)
    primary_validation_rows = [
        row for row in validation_rows if row["primary_seed"] is True
    ]
    v3_validation = [row["critic_called"] for row in primary_validation_rows]
    primary_oof_rows = [row for row in oof_rows if row["primary_seed"] is True]
    v3_oof = [row["critic_called"] for row in primary_oof_rows]
    stop_validation = [False] * len(validation_examples)
    always_validation = [True] * len(validation_examples)
    stop_oof = [False] * len(training_examples)
    always_oof = [True] * len(training_examples)
    comparison = {
        "STOP": evaluate_validation_policy(validation_examples, stop_validation),
        "ALWAYS_CRITIC_ONLY": evaluate_validation_policy(validation_examples, always_validation),
        "CONTROLLER_V1_PRIMARY": {
            **evaluate_validation_policy(validation_examples, v1_validation),
            "frozen_historical_policy": True,
        },
        "CONTROLLER_V2_PRIMARY": {
            **evaluate_validation_policy(validation_examples, v2_validation),
            "frozen_historical_policy": True,
        },
        "CONTROLLER_V3_PRIMARY": {
            **evaluate_validation_policy(validation_examples, v3_validation),
            "threshold_source": "training_1000_stratified_5fold_oof_only",
        },
    }
    oof_comparison = {
        "STOP": _oof_policy_metrics(labels, stop_oof, training_costs),
        "ALWAYS_CRITIC_ONLY": _oof_policy_metrics(labels, always_oof, training_costs),
        "CONTROLLER_V1_PRIMARY": _oof_policy_metrics(labels, v1_oof, training_costs),
        "CONTROLLER_V2_PRIMARY": _oof_policy_metrics(labels, v2_oof, training_costs),
        "CONTROLLER_V3_PRIMARY": _oof_policy_metrics(labels, v3_oof, training_costs),
    }
    mcnemar = {
        "v3_vs_stop": paired_mcnemar(
            validation_examples, v3_validation, stop_validation,
            first_name="CONTROLLER_V3_PRIMARY", second_name="STOP",
        ),
        "v3_vs_always": paired_mcnemar(
            validation_examples, v3_validation, always_validation,
            first_name="CONTROLLER_V3_PRIMARY", second_name="ALWAYS_CRITIC_ONLY",
        ),
        "v3_vs_v1": paired_mcnemar(
            validation_examples, v3_validation, v1_validation,
            first_name="CONTROLLER_V3_PRIMARY", second_name="CONTROLLER_V1_PRIMARY",
        ),
        "v3_vs_v2": paired_mcnemar(
            validation_examples, v3_validation, v2_validation,
            first_name="CONTROLLER_V3_PRIMARY", second_name="CONTROLLER_V2_PRIMARY",
        ),
    }
    stability = _stability_summary(seed_metrics)
    parameter_counts = controller_parameter_counts(primary_model)
    if not (
        EXPECTED_TRAINABLE_PARAMETER_MIN
        <= parameter_counts["trainable"]
        <= EXPECTED_TRAINABLE_PARAMETER_MAX
    ):
        raise AssertionError("Controller v3 parameter count is outside contract")
    elapsed = time.perf_counter() - started
    after = _critical_snapshot(
        training_path,
        training_manifest_path,
        validation_path,
        final_test_manifest_path,
        v1_dir,
        v2_dir,
    )
    if before != after:
        raise RuntimeError("A frozen input or historical artifact changed during v3 training")
    sources = {
        "training_examples": {
            "path": str(training_path.resolve()),
            "sha256": _file_sha256(training_path),
            "samples": len(training_examples),
        },
        "training_manifest": {
            "path": str(training_manifest_path.resolve()),
            "sha256": _file_sha256(training_manifest_path),
        },
        "development_validation": {
            "path": str(validation_path.resolve()),
            "sha256": validation_sha,
            "samples": len(validation_examples),
        },
        "feature_cache": {
            "path": str(cache_path.resolve()),
            "sha256": _file_sha256(cache_path),
            "reused": cache_reused,
            "embedding_encode_calls": cache_contract["embedding_encode_calls"],
        },
        "controller_v1": v1_sources,
        "controller_v2": v2_sources,
        "final_test_guard": {
            **final_guard,
            "manifest_only_read": True,
            "examples_read": False,
        },
    }
    summary = {
        "controller_v3": True,
        "controller_trained": True,
        "formal_offline_training": True,
        "development_validation": {
            "evaluated_after_oof_threshold_freeze": True,
            "validation_used_for_threshold": False,
            "policy_comparison": comparison,
            "oof_policy_comparison": oof_comparison,
            "mcnemar": mcnemar,
        },
        "final_test_evaluated": False,
        "final_test_examples_read": False,
        "deployable": False,
        "model_calls": 0,
        "llm_calls": 0,
        "backend_initialized": False,
        "rollout_collected": False,
        "prompt_modified": False,
        "parser_modified": False,
        "v1_v2_modified": False,
        "sources": sources,
        "feature_cache_contract": cache_contract,
        "data_boundary": {
            "training": "Training 1000 only",
            "validation": "development evaluation only",
            "validation_used_for_training": False,
            "validation_used_for_oof": False,
            "validation_used_for_thresholds": False,
            "validation_used_for_seed_or_model_selection": False,
            "final_test_manifest_verified_only": True,
            "final_test_examples_read": False,
        },
        "model": {
            "class": "PreCriticControllerV3",
            "parameter_counts": parameter_counts,
            "encoder_parameter_counts": encoder_counts,
            "learned_absolute_position_embedding": True,
            "max_sequence_length": MAX_CONTROLLER_SEQUENCE_LENGTH,
            "field_type_embedding": True,
            "cost_head": False,
            "gate_score": "p_help - p_harm from factorized heads only",
            "auxiliary_head_used_for_gate": False,
            "hard_budget_logic_modified": False,
            "cost_fallback_logic_modified": False,
        },
        "training_protocol": {
            "primary_seed": PRIMARY_SEED,
            "stability_seeds": list(STABILITY_SEEDS),
            "best_seed_selected": False,
            "folds": N_SPLITS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "loss": "L_error + L_fix + L_harm + 0.25 * L_aux",
            "strict_conditional_masks": True,
            "fold_numeric_normalization": "first 8 dimensions, fold training only",
            "categorical_state_normalized": False,
            "validation_normalization": "full Training 1000 statistics",
            "hyperparameter_search": False,
            "deployment_operating_point_selected": False,
            "device": device_details,
            "feature_cache_seconds": cache_seconds,
            "training_seconds": training_seconds,
            "total_seconds": elapsed,
        },
        "primary_seed_metrics": primary_metrics,
        "stability": stability,
        "integrity": {
            "before": before,
            "after": after,
            "all_frozen_inputs_and_history_unchanged": True,
        },
    }
    checkpoint = {
        "controller_v3": True,
        "controller_trained": True,
        "formal_offline_training": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "deployable": False,
        "model_calls": 0,
        "model_class": "PreCriticControllerV3",
        "model_state_dict": primary_model.state_dict(),
        "parameter_counts": parameter_counts,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "feature_cache_sha256": _file_sha256(cache_path),
        "numeric_mean_first_8": primary_normalization.mean,
        "numeric_std_first_8": primary_normalization.std,
        "development_threshold": primary_metrics["oof"]["development_threshold"],
        "budget_thresholds": [
            {
                key: point[key]
                for key in (
                    "target_budget_rate",
                    "threshold",
                    "oof_critic_calls",
                    "oof_critic_call_rate",
                    "threshold_source",
                )
            }
            for point in primary_metrics["oof"]["budget_curve"]
        ],
        "training_sha256": TRAINING_SHA256,
        "validation_used_for_training_or_selection": False,
        "final_test_guard": final_guard,
        "cost_head": False,
        "hard_budget_guard": {
            "critic_completion_token_cap": 512,
            "critic_call_cap": 1,
            "uses_predicted_cost": False,
        },
    }
    seed_payload = {
        "controller_v3": True,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds": list(STABILITY_SEEDS),
        "best_seed_selected": False,
        "feature_cache_sha256": _file_sha256(cache_path),
        "embedding_encode_calls": 1,
        "seeds": seed_metrics,
        "stability": stability,
        "model_calls": 0,
        "final_test_evaluated": False,
        "deployable": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _save_torch_atomic(output / "primary_model.pt", checkpoint)
    _write_json_atomic(output / "seed_metrics.json", seed_payload)
    write_jsonl(output / "oof_predictions.jsonl", oof_rows)
    write_jsonl(output / "validation_predictions.jsonl", validation_rows)
    _write_json_atomic(output / "summary.json", summary)
    _write_text_atomic(output / "report.md", _report(summary))
    return {
        "controller_v3": True,
        "controller_trained": True,
        "formal_offline_training": True,
        "device": device_details,
        "parameter_counts": parameter_counts,
        "training_seconds": training_seconds,
        "total_seconds": elapsed,
        "feature_cache_reused": cache_reused,
        "embedding_encode_calls": 1,
        "final_test_evaluated": False,
        "deployable": False,
        "model_calls": 0,
        "output_dir": str(output.resolve()),
    }
