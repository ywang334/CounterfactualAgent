"""Offline Controller v3 capacity ablation using the frozen feature cache.

Only the Transformer capacity changes.  This module has no encoder, backend,
data-collection, or Final-Test-example path.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from .io_utils import read_jsonl
from .precritic_controller_v1 import (
    BUDGET_RATES,
    DEFAULT_FINAL_MANIFEST,
    DEFAULT_TRAINING,
    DEFAULT_TRAINING_MANIFEST,
    DEFAULT_VALIDATION,
    LABELS,
    N_SPLITS,
    PRIMARY_SEED,
    TRAINING_SHA256,
    evaluate_validation_policy,
    gated_decisions,
    load_training_examples,
    load_validation_examples,
    oof_budget_thresholds_v1,
    select_oof_threshold_v1,
    verify_sealed_final_manifest,
    _file_sha256,
)
from .precritic_controller_v3 import (
    DROPOUT,
    EMBEDDING_DIM,
    FIELD_TYPES,
    MAX_CONTROLLER_SEQUENCE_LENGTH,
    NUM_HEADS,
    STRUCTURED_STATE_FEATURES,
    PreCriticV3Batch,
    controller_parameter_counts,
)
from .precritic_controller_v3_training import (
    AUXILIARY_LOSS_WEIGHT,
    BATCH_SIZE,
    EPOCHS,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CachedFeatureSplit,
    StateNormalization,
    _cached_split,
    _head_metrics,
    _indexed_batch,
    _load_training_cost_records,
    _oof_policy_metrics,
    _predict,
    _probability_payload,
    apply_state_normalization,
    fit_state_normalization,
    loss_balance,
    v3_training_loss,
)
from .precritic_probe import deterministic_stratified_folds


DEFAULT_CONTROLLER_V3_DIR = Path("artifacts/precritic_controller_v3")
DEFAULT_OUTPUT = Path("artifacts/precritic_controller_v3_capacity_ablation")
VARIANT_ORDER = ("tiny", "maas")
VARIANT_CONFIGS = {
    "tiny": {"display_name": "V3-Tiny", "d_model": 64, "num_layers": 1, "nhead": 4, "dim_feedforward": 256},
    "maas": {"display_name": "V3-MaAS", "d_model": 64, "num_layers": 2, "nhead": 4, "dim_feedforward": 256},
}
EXPECTED_PARAMETER_COUNTS = {"tiny": 79_111, "maas": 129_095}
ALLOWED_PREEXISTING_CONTROL_FILES = {"run.log", "run.pid"}
FINAL_OUTPUT_FILES = ("summary.json", "report.md")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CapacitySpec:
    key: str
    display_name: str
    d_model: int
    num_layers: int
    nhead: int
    dim_feedforward: int


def capacity_spec(key: str) -> CapacitySpec:
    if key not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown capacity variant: {key}")
    return CapacitySpec(key=key, **VARIANT_CONFIGS[key])


class CapacityAblationController(nn.Module):
    """Same v3 relation model with capacity supplied by a frozen spec."""

    def __init__(self, spec: CapacitySpec) -> None:
        super().__init__()
        if spec.nhead != NUM_HEADS or spec.d_model % spec.nhead:
            raise ValueError("Capacity spec violates the fixed four-head contract")
        self.spec = spec
        d_model = spec.d_model
        self.input_projection = nn.Linear(EMBEDDING_DIM, d_model)
        self.field_type_embedding = nn.Embedding(len(FIELD_TYPES), d_model)
        self.position_embedding = nn.Embedding(MAX_CONTROLLER_SEQUENCE_LENGTH, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.state_projection = nn.Linear(len(STRUCTURED_STATE_FEATURES), d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=spec.nhead,
            dim_feedforward=spec.dim_feedforward,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=spec.num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.solver_error_head = nn.Linear(d_model, 1)
        self.critic_fix_head = nn.Linear(d_model, 1)
        self.critic_harm_head = nn.Linear(d_model, 1)
        self.transition_aux_head = nn.Linear(d_model, len(LABELS))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, batch: PreCriticV3Batch) -> dict[str, torch.Tensor]:
        embeddings = batch.text_embeddings
        if embeddings.ndim != 3 or embeddings.shape[-1] != EMBEDDING_DIM:
            raise ValueError("Capacity ablation embeddings have invalid shape")
        batch_size, sequence_length, _ = embeddings.shape
        expected = (batch_size, sequence_length)
        if (
            batch.type_ids.shape != expected
            or batch.position_ids.shape != expected
            or batch.padding_mask.shape != expected
        ):
            raise ValueError("Capacity ablation masks/type/position IDs do not align")
        if sequence_length > MAX_CONTROLLER_SEQUENCE_LENGTH:
            raise ValueError("Capacity ablation position capacity exceeded")
        if torch.any(batch.position_ids < 0) or torch.any(
            batch.position_ids >= MAX_CONTROLLER_SEQUENCE_LENGTH
        ):
            raise ValueError("Capacity ablation position IDs are out of range")
        if batch.structured_state.shape != (
            batch_size,
            len(STRUCTURED_STATE_FEATURES),
        ) or batch.state_positions.shape != (batch_size,):
            raise ValueError("Capacity ablation structured state does not align")
        if torch.any(batch.padding_mask[:, 0]) or torch.any(
            batch.padding_mask[torch.arange(batch_size), batch.state_positions]
        ):
            raise ValueError("CLS and state tokens cannot be padded")

        hidden = self.input_projection(embeddings).clone()
        hidden[:, 0, :] = self.cls_token.expand(batch_size, -1, -1)[:, 0, :]
        hidden[torch.arange(batch_size), batch.state_positions] = self.state_projection(
            batch.structured_state
        )
        hidden = (
            hidden
            + self.field_type_embedding(batch.type_ids)
            + self.position_embedding(batch.position_ids)
        )
        encoded = self.transformer(hidden, src_key_padding_mask=batch.padding_mask)
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
        factorized = torch.stack(
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
            "factorized_transition_probabilities": factorized,
            "transition_aux_probabilities": torch.softmax(transition_logits, dim=-1),
        }


def capacity_contract() -> dict[str, Any]:
    variants = {}
    for key in VARIANT_ORDER:
        spec = capacity_spec(key)
        counts = controller_parameter_counts(CapacityAblationController(spec))
        if counts["trainable"] != EXPECTED_PARAMETER_COUNTS[key] or counts["frozen"] != 0:
            raise AssertionError(f"{spec.display_name} parameter count changed")
        variants[key] = {**spec.__dict__, "parameter_counts": counts}
    return {
        "only_capacity_changes": True,
        "shared_invariants": {
            "embedding_dim": EMBEDDING_DIM,
            "max_sequence_length": MAX_CONTROLLER_SEQUENCE_LENGTH,
            "field_type_embedding": True,
            "learned_absolute_position_embedding": True,
            "structured_state_features": len(STRUCTURED_STATE_FEATURES),
            "factorized_heads": ["solver_error", "critic_fix", "critic_harm"],
            "auxiliary_four_class_head": True,
            "dropout": DROPOUT,
            "loss": "L_error + L_fix + L_harm + 0.25 * L_aux",
            "cost_head": False,
        },
        "variants": variants,
    }


def validate_frozen_fold_manifest(
    fold_manifest: Sequence[Mapping[str, Any]], labels: Sequence[str]
) -> list[dict[str, Any]]:
    if len(fold_manifest) != N_SPLITS:
        raise ValueError("Historical primary fold count changed")
    validation_members = []
    normalized = []
    all_indices = set(range(len(labels)))
    for expected_fold, fold in enumerate(fold_manifest):
        if int(fold["fold"]) != expected_fold:
            raise ValueError("Historical primary fold order changed")
        training = [int(index) for index in fold["train"]]
        validation = [int(index) for index in fold["validation"]]
        if set(training) & set(validation) or set(training) | set(validation) != all_indices:
            raise ValueError("Historical primary fold train/validation partition is invalid")
        if len(training) != len(set(training)) or len(validation) != len(set(validation)):
            raise ValueError("Historical primary fold contains duplicate indices")
        validation_members.extend(validation)
        normalized.append({"fold": expected_fold, "train": training, "validation": validation})
    if sorted(validation_members) != list(range(len(labels))):
        raise ValueError("Historical OOF folds do not cover Training 1000 exactly once")
    return normalized


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def train_capacity_model(
    *,
    spec: CapacitySpec,
    features: CachedFeatureSplit,
    normalized_state: torch.Tensor,
    labels: Sequence[str],
    training_indices: Sequence[int],
    seed: int,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tuple[CapacityAblationController, dict[str, Any]]:
    """Exact v3 optimizer/loss loop with only the model factory changed."""
    indices = list(training_indices)
    subset_labels = [labels[index] for index in indices]
    balance = loss_balance(subset_labels)
    _seed_everything(seed, device)
    model = CapacityAblationController(spec).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
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
            outputs = model(
                _indexed_batch(features, global_indices, normalized_state, device)
            )
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


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _cache_splits(
    cache_path: Path, training_count: int, validation_count: int
) -> tuple[CachedFeatureSplit, CachedFeatureSplit, Mapping[str, Any]]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    if set(payload) != {"metadata", "training", "validation"}:
        raise ValueError("Frozen feature cache schema changed")
    metadata = payload["metadata"]
    if metadata.get("embedding_encode_calls") != 1:
        raise ValueError("Frozen feature cache encoding contract changed")
    if not metadata.get("feature_cache") or metadata.get("gold_cached"):
        raise ValueError("Frozen feature cache boundary changed")
    return (
        _cached_split(payload["training"], training_count),
        _cached_split(payload["validation"], validation_count),
        metadata,
    )


def budget_curve_from_oof(
    scores: Sequence[float],
    labels: Sequence[str],
    validation_scores: Sequence[float],
    validation_examples: Sequence[Any],
    training_costs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points = oof_budget_thresholds_v1(scores, BUDGET_RATES)
    oof = []
    validation = []
    for point in points:
        threshold = float(point["threshold"])
        oof.append(
            {
                **point,
                "diagnostic_only": True,
                "operating_point_selected": False,
                "oof_policy": _oof_policy_metrics(
                    labels, gated_decisions(scores, threshold), training_costs
                ),
            }
        )
        validation.append(
            {
                **point,
                "diagnostic_only": True,
                "operating_point_selected": False,
                "validation_policy": evaluate_validation_policy(
                    validation_examples,
                    gated_decisions(validation_scores, threshold),
                ),
            }
        )
    return oof, validation


def _stage_contract(
    *, spec: CapacitySpec, cache_sha: str, fold_sha: str, stage: str
) -> dict[str, Any]:
    return {
        "variant": spec.key,
        "capacity_spec": spec.__dict__,
        "feature_cache_sha256": cache_sha,
        "fold_manifest_sha256": fold_sha,
        "primary_seed": PRIMARY_SEED,
        "stage": stage,
        "training_protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "auxiliary_loss_weight": AUXILIARY_LOSS_WEIGHT,
        },
    }


def _load_stage(path: Path, expected_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("contract") != expected_contract or not payload.get("complete"):
        raise ValueError(f"Resume checkpoint contract changed: {path}")
    return payload


def _variant_run(
    *,
    spec: CapacitySpec,
    output_dir: Path,
    resume: bool,
    folds: Sequence[Mapping[str, Any]],
    fold_sha: str,
    training_examples: Sequence[Any],
    validation_examples: Sequence[Any],
    training_features: CachedFeatureSplit,
    validation_features: CachedFeatureSplit,
    training_costs: Sequence[Mapping[str, Any]],
    cache_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    labels = [example.label for example in training_examples]
    variant_dir = output_dir / spec.key
    checkpoint_dir = output_dir / "checkpoints" / spec.key
    output_names = ("primary_model.pt", "oof_predictions.jsonl", "validation_predictions.jsonl")
    if not resume and any((variant_dir / name).exists() for name in output_names):
        raise FileExistsError(f"{spec.display_name} output already exists")
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
    fold_by_index = {}
    for fold in folds:
        fold_number = int(fold["fold"])
        for index in fold["validation"]:
            fold_by_index[index] = fold_number
        stage = f"fold_{fold_number}"
        contract = _stage_contract(
            spec=spec, cache_sha=cache_sha, fold_sha=fold_sha, stage=stage
        )
        checkpoint_path = checkpoint_dir / f"{stage}.pt"
        if checkpoint_path.exists():
            if not resume:
                raise FileExistsError("First capacity-ablation launch forbids resume checkpoints")
            payload = _load_stage(checkpoint_path, contract)
            predictions = payload["predictions"]
            training_metrics = payload["training_metrics"]
            normalization_payload = payload["normalization"]
        else:
            normalization = fit_state_normalization(
                training_features.structured_state, fold["train"]
            )
            normalized = apply_state_normalization(
                training_features.structured_state, normalization
            )
            model, training_metrics = train_capacity_model(
                spec=spec,
                features=training_features,
                normalized_state=normalized,
                labels=labels,
                training_indices=fold["train"],
                seed=PRIMARY_SEED + fold_number + 1,
                device=device,
            )
            predictions = _predict(
                model,
                training_features,
                normalized,
                fold["validation"],
                device,
            )
            normalization_payload = {
                "fit_scope": "fold_training_only",
                "mean_first_8": normalization.mean.tolist(),
                "std_first_8": normalization.std.tolist(),
                "fit_indices_sha256": normalization.fit_indices_sha256,
                "categorical_dimensions_standardized": False,
            }
            _atomic_torch(
                checkpoint_path,
                {
                    "complete": True,
                    "contract": contract,
                    "predictions": predictions,
                    "training_metrics": training_metrics,
                    "normalization": normalization_payload,
                },
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        validation_index = torch.tensor(fold["validation"], dtype=torch.long)
        for name in oof:
            oof[name][validation_index] = predictions[name]
        fold_training.append(
            {
                "fold": fold_number,
                "training_samples": len(fold["train"]),
                "oof_samples": len(fold["validation"]),
                "normalization": normalization_payload,
                **training_metrics,
                "checkpoint": str(checkpoint_path.resolve()),
                "resumed": checkpoint_path.exists() and resume,
            }
        )
        print(json.dumps({"variant": spec.key, "stage": stage, "status": "complete"}), flush=True)

    full_indices = list(range(len(labels)))
    full_contract = _stage_contract(
        spec=spec, cache_sha=cache_sha, fold_sha=fold_sha, stage="full_training_1000"
    )
    full_checkpoint = checkpoint_dir / "full_training_1000.pt"
    if full_checkpoint.exists():
        if not resume:
            raise FileExistsError("First capacity-ablation launch forbids full checkpoint reuse")
        payload = _load_stage(full_checkpoint, full_contract)
        model = CapacityAblationController(spec)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        normalization = StateNormalization(
            mean=payload["normalization_mean"],
            std=payload["normalization_std"],
            fit_indices_sha256=payload["normalization_sha256"],
        )
        final_training = payload["training_metrics"]
    else:
        normalization = fit_state_normalization(
            training_features.structured_state, full_indices
        )
        normalized_training = apply_state_normalization(
            training_features.structured_state, normalization
        )
        model, final_training = train_capacity_model(
            spec=spec,
            features=training_features,
            normalized_state=normalized_training,
            labels=labels,
            training_indices=full_indices,
            seed=PRIMARY_SEED,
            device=device,
        )
        model = model.cpu()
        _atomic_torch(
            full_checkpoint,
            {
                "complete": True,
                "contract": full_contract,
                "model_state_dict": model.state_dict(),
                "normalization_mean": normalization.mean,
                "normalization_std": normalization.std,
                "normalization_sha256": normalization.fit_indices_sha256,
                "training_metrics": final_training,
            },
        )
    model = model.cpu().eval()
    normalized_training = apply_state_normalization(
        training_features.structured_state, normalization
    )
    normalized_validation = apply_state_normalization(
        validation_features.structured_state, normalization
    )
    full_training_outputs = _predict(
        model, training_features, normalized_training, full_indices, torch.device("cpu")
    )
    validation_outputs = _predict(
        model,
        validation_features,
        normalized_validation,
        list(range(len(validation_examples))),
        torch.device("cpu"),
    )
    scores = oof["gate_score"].tolist()
    validation_scores = validation_outputs["gate_score"].tolist()
    diagnostic_threshold = select_oof_threshold_v1(scores, labels)
    diagnostic_threshold["diagnostic_only"] = True
    diagnostic_threshold["operating_point_selected"] = False
    oof_curve, validation_curve = budget_curve_from_oof(
        scores, labels, validation_scores, validation_examples, training_costs
    )
    decisions = gated_decisions(scores, diagnostic_threshold["threshold"])
    validation_decisions = gated_decisions(
        validation_scores, diagnostic_threshold["threshold"]
    )
    in_sample_scores = full_training_outputs["gate_score"].tolist()
    in_sample_decisions = gated_decisions(
        in_sample_scores, diagnostic_threshold["threshold"]
    )
    oof_rows = []
    for index, example in enumerate(training_examples):
        oof_rows.append(
            {
                "capacity_ablation": True,
                "variant": spec.key,
                "primary_seed": PRIMARY_SEED,
                "sample_id": example.sample_id,
                "source_dataset": example.source_dataset,
                "label": example.label,
                "oof_fold": fold_by_index[index],
                "probabilities": _probability_payload(oof, index),
                "gate_score": float(oof["gate_score"][index]),
                "diagnostic_threshold": diagnostic_threshold["threshold"],
                "critic_called": decisions[index],
                "model_input_sha256": training_features.sample_hashes[index],
                "deployable": False,
                "model_calls": 0,
            }
        )
    validation_rows = []
    for index, example in enumerate(validation_examples):
        validation_rows.append(
            {
                "capacity_ablation": True,
                "variant": spec.key,
                "primary_seed": PRIMARY_SEED,
                "question_id": example.question_id,
                "label": example.label,
                "probabilities": _probability_payload(validation_outputs, index),
                "gate_score": float(validation_outputs["gate_score"][index]),
                "diagnostic_threshold": diagnostic_threshold["threshold"],
                "critic_called": validation_decisions[index],
                "model_input_sha256": validation_features.sample_hashes[index],
                "development_validation": True,
                "deployable": False,
                "model_calls": 0,
            }
        )
    checkpoint = {
        "capacity_ablation": True,
        "variant": spec.key,
        "capacity_spec": spec.__dict__,
        "parameter_counts": controller_parameter_counts(model),
        "model_state_dict": model.state_dict(),
        "normalization_mean_first_8": normalization.mean,
        "normalization_std_first_8": normalization.std,
        "diagnostic_threshold": diagnostic_threshold,
        "feature_cache_sha256": cache_sha,
        "fold_manifest_sha256": fold_sha,
        "primary_seed": PRIMARY_SEED,
        "deployable": False,
        "model_calls": 0,
    }
    variant_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch(variant_dir / "primary_model.pt", checkpoint)
    _atomic_jsonl(variant_dir / "oof_predictions.jsonl", oof_rows)
    _atomic_jsonl(variant_dir / "validation_predictions.jsonl", validation_rows)
    full_metrics = _head_metrics(full_training_outputs, labels)
    oof_metrics = _head_metrics(oof, labels)
    validation_labels = [example.label for example in validation_examples]
    validation_metrics = _head_metrics(validation_outputs, validation_labels)
    return {
        "variant": spec.key,
        "display_name": spec.display_name,
        "capacity_spec": spec.__dict__,
        "parameter_counts": controller_parameter_counts(model),
        "fold_manifest_sha256": fold_sha,
        "fold_training": fold_training,
        "final_training": final_training,
        "diagnostic_threshold": diagnostic_threshold,
        "training_full_in_sample": {
            "in_sample_diagnostic": True,
            "head_metrics": full_metrics,
            "policy": _oof_policy_metrics(labels, in_sample_decisions, training_costs),
        },
        "training_oof": {
            "head_metrics": oof_metrics,
            "policy": _oof_policy_metrics(labels, decisions, training_costs),
            "budget_curve": oof_curve,
        },
        "validation": {
            "development_validation": True,
            "head_metrics": validation_metrics,
            "policy": evaluate_validation_policy(validation_examples, validation_decisions),
            "budget_curve": validation_curve,
        },
        "overfitting_gaps": {
            "solver_error_pr_auc": full_metrics["solver_error"]["pr_auc"] - oof_metrics["solver_error"]["pr_auc"],
            "critic_fix_pr_auc": full_metrics["critic_fix_given_solver_error"]["pr_auc"] - oof_metrics["critic_fix_given_solver_error"]["pr_auc"],
            "critic_harm_pr_auc": full_metrics["critic_harm_given_solver_correct"]["pr_auc"] - oof_metrics["critic_harm_given_solver_correct"]["pr_auc"],
            "helpful_pr_auc": full_metrics["final_helpful_pr_auc"] - oof_metrics["final_helpful_pr_auc"],
            "harmful_pr_auc": full_metrics["final_harmful_pr_auc"] - oof_metrics["final_harmful_pr_auc"],
            "factorized_macro_f1": full_metrics["factorized_four_class_macro_f1"] - oof_metrics["factorized_four_class_macro_f1"],
        },
        "resume_supported": True,
        "resumed": resume,
    }


def _historical_snapshot(
    *, controller_v3_dir: Path, training_path: Path, training_manifest_path: Path,
    validation_path: Path, final_manifest_path: Path
) -> dict[str, Any]:
    paths = {
        "training": training_path,
        "training_manifest": training_manifest_path,
        "validation": validation_path,
        "final_manifest": final_manifest_path,
        "feature_cache": controller_v3_dir / "feature_cache.pt",
        "v3_primary_model": controller_v3_dir / "primary_model.pt",
        "v3_seed_metrics": controller_v3_dir / "seed_metrics.json",
        "v3_oof": controller_v3_dir / "oof_predictions.jsonl",
        "v3_validation": controller_v3_dir / "validation_predictions.jsonl",
        "v3_summary": controller_v3_dir / "summary.json",
        "v3_report": controller_v3_dir / "report.md",
        "v3_generalization_audit": controller_v3_dir / "generalization_audit" / "audit_summary.json",
        "prompt": Path(__file__).with_name("logiqa_prompts.py"),
        "parser": Path(__file__).with_name("logiqa_pilot.py"),
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Frozen capacity-ablation input missing: {path}")
        result[name] = {"path": str(path.resolve()), "sha256": _file_sha256(path), "bytes": path.stat().st_size}
    return result


def _historical_v3(controller_dir: Path) -> dict[str, Any]:
    summary = json.loads((controller_dir / "summary.json").read_text(encoding="utf-8"))
    audit = json.loads(
        (controller_dir / "generalization_audit" / "audit_summary.json").read_text(encoding="utf-8")
    )
    primary = summary["primary_seed_metrics"]
    regimes = audit["prediction_regimes"]
    return {
        "variant": "existing_v3_454k",
        "display_name": "V3-454K",
        "historical_read_only": True,
        "retrained": False,
        "parameter_counts": summary["model"]["parameter_counts"],
        "diagnostic_threshold": primary["oof"]["development_threshold"],
        "training_full_in_sample": regimes["training_full_in_sample"],
        "training_oof": {
            "head_metrics": primary["oof"]["head_metrics"],
            "policy": primary["oof"]["development_policy"],
            "budget_curve": primary["oof"]["budget_curve"],
        },
        "validation": {
            "head_metrics": primary["validation"]["head_metrics"],
            "policy": primary["validation"]["controller_policy"],
            "budget_curve": primary["validation"]["budget_curve"],
        },
        "overfitting_gaps": audit["training_dynamics"]["in_sample_minus_oof_metric_gaps"],
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Controller v3 Capacity Ablation Pilot",
        "",
        "Development-only capacity comparison. No model, seed, threshold, budget, or operating point is selected.",
        "",
        "| Variant | Parameters | OOF accuracy | OOF corrected/degraded | Validation accuracy | Validation corrected/degraded |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("tiny", "maas", "existing_v3_454k"):
        value = summary["variants"][key]
        oof = value["training_oof"]["policy"]
        validation = value["validation"]["policy"]
        lines.append(
            f"| {value['display_name']} | {value['parameter_counts']['trainable']:,} | "
            f"{oof['accuracy']:.3f} | {oof['corrected']}/{oof['degraded']} | "
            f"{validation['accuracy']:.3f} | {validation['corrected']}/{validation['degraded']} |"
        )
    lines.extend(
        [
            "",
            "All max-net thresholds and budget curves are diagnostic only.",
            "Existing V3-454K values were read from frozen historical artifacts and were not retrained.",
            "",
            "## Boundaries",
            "",
            "- Feature cache reused; embedding forward calls added: 0.",
            "- No LLM/backend/API calls or data collection.",
            "- Final Test examples not read; sealed manifest only.",
            "- No cost head and no budget-logic changes.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_capacity_ablation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT,
    controller_v3_dir: str | Path = DEFAULT_CONTROLLER_V3_DIR,
    training_path: str | Path = DEFAULT_TRAINING,
    training_manifest_path: str | Path = DEFAULT_TRAINING_MANIFEST,
    validation_path: str | Path = DEFAULT_VALIDATION,
    final_test_manifest_path: str | Path = DEFAULT_FINAL_MANIFEST,
    resume: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    controller_v3_dir = Path(controller_v3_dir)
    training_path = Path(training_path)
    training_manifest_path = Path(training_manifest_path)
    validation_path = Path(validation_path)
    final_test_manifest_path = Path(final_test_manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        unexpected = {
            path.name for path in output_dir.iterdir()
            if path.name not in ALLOWED_PREEXISTING_CONTROL_FILES
        }
        if unexpected:
            raise FileExistsError(
                f"First capacity-ablation launch found existing experiment state: {sorted(unexpected)}"
            )
    elif not (output_dir / "checkpoints").is_dir():
        raise FileNotFoundError("--resume requires atomic stage checkpoints")

    before = _historical_snapshot(
        controller_v3_dir=controller_v3_dir,
        training_path=training_path,
        training_manifest_path=training_manifest_path,
        validation_path=validation_path,
        final_manifest_path=final_test_manifest_path,
    )
    training_examples, training_manifest = load_training_examples(
        training_path, training_manifest_path
    )
    if _file_sha256(training_path) != TRAINING_SHA256:
        raise ValueError("Frozen Training 1000 SHA256 changed")
    final_guard = verify_sealed_final_manifest(final_test_manifest_path, training_manifest)
    training_ids = {example.sample_id for example in training_examples}
    validation_examples, _ = load_validation_examples(
        validation_path, training_manifest, training_ids
    )
    labels = [example.label for example in training_examples]
    historical_summary = json.loads(
        (controller_v3_dir / "summary.json").read_text(encoding="utf-8")
    )
    historical_folds = historical_summary["primary_seed_metrics"]["oof"]["fold_manifest"]
    folds = validate_frozen_fold_manifest(historical_folds, labels)
    fold_sha = _canonical_sha256(folds)
    historical_fold_sha = _canonical_sha256(
        [{"fold": int(fold["fold"]), "train": list(fold["train"]), "validation": list(fold["validation"])} for fold in historical_folds]
    )
    if fold_sha != historical_fold_sha:
        raise AssertionError("Capacity ablation fold manifest differs from historical v3")
    cache_path = controller_v3_dir / "feature_cache.pt"
    cache_sha = _file_sha256(cache_path)
    training_features, validation_features, cache_metadata = _cache_splits(
        cache_path, len(training_examples), len(validation_examples)
    )
    if tuple(example.sample_id for example in training_examples) != training_features.content_sha256:
        raise ValueError("Frozen feature-cache Training identities changed")
    training_costs = _load_training_cost_records(training_path, training_examples)
    contract = capacity_contract()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_details = {
        "device": str(device),
        "type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    variants = {}
    for key in VARIANT_ORDER:
        variants[key] = _variant_run(
            spec=capacity_spec(key),
            output_dir=output_dir,
            resume=resume,
            folds=folds,
            fold_sha=fold_sha,
            training_examples=training_examples,
            validation_examples=validation_examples,
            training_features=training_features,
            validation_features=validation_features,
            training_costs=training_costs,
            cache_sha=cache_sha,
            device=device,
        )
    variants["existing_v3_454k"] = _historical_v3(controller_v3_dir)
    after = _historical_snapshot(
        controller_v3_dir=controller_v3_dir,
        training_path=training_path,
        training_manifest_path=training_manifest_path,
        validation_path=validation_path,
        final_manifest_path=final_test_manifest_path,
    )
    if before != after:
        raise RuntimeError("Frozen inputs or historical artifacts changed during capacity ablation")
    summary = {
        "controller_v3_capacity_ablation": True,
        "pilot": True,
        "development_validation": True,
        "final_test_evaluated": False,
        "final_test_examples_read": False,
        "deployable": False,
        "model_selected": False,
        "operating_point_selected": False,
        "seed_selected": False,
        "budget_selected": False,
        "hyperparameter_search": False,
        "primary_seed": PRIMARY_SEED,
        "stability_seeds_run": False,
        "feature_cache_reused": True,
        "feature_cache_sha256": cache_sha,
        "embedding_forward_calls": 0,
        "historical_cache_embedding_encode_calls": cache_metadata["embedding_encode_calls"],
        "model_calls": 0,
        "llm_calls": 0,
        "backend_initialized": False,
        "rollout_collected": False,
        "cost_head": False,
        "budget_logic_modified": False,
        "prompt_modified": False,
        "parser_modified": False,
        "existing_v3_retrained": False,
        "capacity_contract": contract,
        "training_protocol": {
            "fold_manifest_source": "frozen existing V3 primary seed",
            "fold_manifest_sha256": fold_sha,
            "folds": N_SPLITS,
            "seed": PRIMARY_SEED,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "loss": "L_error + L_fix + L_harm + 0.25 * L_aux",
            "resume_requested": resume,
            "first_launch_resume_forbidden": True,
            "device": device_details,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "variants": variants,
        "final_test_guard": {**final_guard, "manifest_only_read": True, "examples_read": False},
        "integrity": {
            "before": before,
            "after": after,
            "all_inputs_and_historical_artifacts_unchanged": True,
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_text(output_dir / "report.md", _report(summary))
    return summary

