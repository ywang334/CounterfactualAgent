from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .critic_gating_audit import (
    build_collection_case,
    build_validation_case,
)
from .io_utils import read_jsonl, write_jsonl
from .logiqa_pilot import ANSWER_LETTERS


LABELS = (
    "correct_to_correct",
    "correct_to_wrong",
    "wrong_to_correct",
    "wrong_to_wrong",
)
SEED = 20260813
N_SPLITS = 5
HIDDEN_DIM = 64
EPOCHS = 120
LEARNING_RATE = 1e-3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BUDGET_RATES = (0.10, 0.20, 0.30, 0.50, 1.00)
NUMERIC_FEATURES = (
    "log1p_prompt_tokens_over_10",
    "log1p_completion_tokens_over_10",
    "log1p_total_tokens_over_10",
    "solver_calls",
    "strict_parse_success",
    "tolerant_parse_success",
    "tolerant_match_count_capped_over_3",
    "tolerant_conflict",
)


class FrozenTextEncoder(Protocol):
    name: str
    dimension: int
    mock_only: bool

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class OfflineMiniLMEncoder:
    """Frozen MiniLM loaded strictly from local files; it never downloads."""

    mock_only = False

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError("sentence-transformers is required for the formal probe") from exc
        try:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "all-MiniLM-L6-v2 is not available locally; prepare the frozen "
                "encoder cache before running the offline probe"
            ) from exc
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.name = model_name
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            encoded = self.model.encode(
                list(texts),
                batch_size=32,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return encoded.detach().to(dtype=torch.float32, device="cpu")


class PreCriticGateProbe(nn.Module):
    """One-hidden-layer classifier used only by this learnability probe."""

    def __init__(
        self,
        embedding_dim: int,
        numeric_dim: int = len(NUMERIC_FEATURES),
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = len(LABELS),
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.numeric_dim = numeric_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, embeddings: torch.Tensor, numeric: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([embeddings, numeric], dim=-1))


@dataclass(frozen=True)
class ProbeExample:
    dataset: str
    question_id: str | int
    gold: str
    label: str
    model_input: dict[str, Any]
    feature_text: str
    numeric: tuple[float, ...]
    solver_answer: str | None
    critic_only_answer: str | None
    audit_case: dict[str, Any]


def _mapping(payload: Any, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be an object")
    return payload


def _usage(payload: Any, calls: Any, context: str) -> dict[str, int]:
    value = _mapping(payload, f"{context} usage")
    result: dict[str, int] = {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        token_count = value.get(field)
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
            raise ValueError(f"{context} has invalid {field}")
        result[field] = token_count
    if result["prompt_tokens"] + result["completion_tokens"] != result["total_tokens"]:
        raise ValueError(f"{context} has inconsistent token totals")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
        raise ValueError(f"{context} has invalid calls")
    result["calls"] = calls
    return result


def _render_feature_text(model_input: dict[str, Any]) -> str:
    problem = model_input["problem"]
    parse = model_input["solver"]["parse_status"]
    options = problem["options"]
    return (
        "<problem>\n"
        f"PASSAGE: {problem['passage']}\n"
        f"QUESTION: {problem['question']}\n"
        + "\n".join(f"{letter}. {options[letter]}" for letter in ANSWER_LETTERS)
        + "\n</problem>\n<solver_response>\n"
        + model_input["solver"]["raw_output"]
        + "\n</solver_response>\n<parse_status>\n"
        + f"strict_answer={parse['strict_answer']}\n"
        + f"tolerant_answer={parse['tolerant_answer']}\n"
        + f"tolerant_match_count={parse['tolerant_match_count']}\n"
        + f"tolerant_conflict={str(parse['tolerant_conflict']).lower()}\n"
        + "</parse_status>"
    )


def _numeric_features(model_input: dict[str, Any]) -> tuple[float, ...]:
    usage = model_input["solver"]["usage"]
    parse = model_input["solver"]["parse_status"]
    return (
        math.log1p(usage["prompt_tokens"]) / 10.0,
        math.log1p(usage["completion_tokens"]) / 10.0,
        math.log1p(usage["total_tokens"]) / 10.0,
        float(usage["calls"]),
        float(parse["strict_answer"] in ANSWER_LETTERS),
        float(parse["tolerant_answer"] in ANSWER_LETTERS),
        min(float(parse["tolerant_match_count"]), 3.0) / 3.0,
        float(parse["tolerant_conflict"]),
    )


def build_probe_example(row: dict[str, Any], dataset: str) -> ProbeExample:
    if dataset == "collection_200":
        state = _mapping(row.get("state_for_controller"), "collection state")
        problem = _mapping(state.get("problem"), "collection problem")
        solver = _mapping(row.get("solver"), "collection solver")
        tolerant = _mapping(solver.get("tolerant"), "collection Solver tolerant")
        solver_usage = _mapping(solver.get("cost"), "collection Solver cost")
        usage = _usage(solver_usage, solver_usage.get("calls"), "collection Solver")
        audit_case = build_collection_case(row)
    elif dataset == "validation_100":
        problem = _mapping(row.get("problem"), "validation problem")
        solver = _mapping(row.get("solver"), "validation solver")
        tolerant = _mapping(solver.get("tolerant"), "validation Solver tolerant")
        usage = _usage(solver.get("usage"), solver.get("calls"), "validation Solver")
        audit_case = build_validation_case(row)
    else:
        raise ValueError(f"Unknown probe dataset: {dataset}")
    options = _mapping(problem.get("options"), f"{dataset} options")
    if set(options) != set(ANSWER_LETTERS):
        raise ValueError(f"{dataset} problem must contain exactly A-D options")
    raw_output = solver.get("raw_output")
    if not isinstance(raw_output, str):
        raise ValueError(f"{dataset} Solver raw output is missing")
    match_count = tolerant.get("match_count")
    conflict = tolerant.get("conflict")
    if isinstance(match_count, bool) or not isinstance(match_count, int) or match_count < 0:
        raise ValueError(f"{dataset} tolerant match_count is invalid")
    if not isinstance(conflict, bool):
        raise ValueError(f"{dataset} tolerant conflict is invalid")
    model_input = {
        "problem": {
            "passage": str(problem.get("passage", "")),
            "question": str(problem.get("question", "")),
            "options": {letter: str(options[letter]) for letter in ANSWER_LETTERS},
        },
        "solver": {
            "raw_output": raw_output,
            "parse_status": {
                "strict_answer": solver.get("strict_answer"),
                "tolerant_answer": tolerant.get("answer"),
                "tolerant_match_count": match_count,
                "tolerant_conflict": conflict,
            },
            "usage": usage,
        },
    }
    label = audit_case["policies"]["CRITIC_ONLY"]["transition"]
    if label not in LABELS:
        raise ValueError(f"Unexpected Critic-only label: {label}")
    gold = audit_case["gold"]
    return ProbeExample(
        dataset=dataset,
        question_id=audit_case["question_id"],
        gold=gold,
        label=label,
        model_input=model_input,
        feature_text=_render_feature_text(model_input),
        numeric=_numeric_features(model_input),
        solver_answer=audit_case["solver"]["answer"],
        critic_only_answer=audit_case["policies"]["CRITIC_ONLY"]["answer"],
        audit_case=audit_case,
    )


def deterministic_stratified_folds(
    labels: Sequence[str], n_splits: int = N_SPLITS, seed: int = SEED
) -> list[dict[str, list[int]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    grouped: dict[str, list[int]] = {label: [] for label in LABELS}
    for index, label in enumerate(labels):
        if label not in grouped:
            raise ValueError(f"Unknown class label: {label}")
        grouped[label].append(index)
    if any(len(indices) < n_splits for indices in grouped.values()):
        raise ValueError("Every class must have at least n_splits examples")
    validation_folds: list[list[int]] = [[] for _ in range(n_splits)]
    for class_index, label in enumerate(LABELS):
        indices = list(grouped[label])
        random.Random(seed + class_index).shuffle(indices)
        for rank, index in enumerate(indices):
            validation_folds[rank % n_splits].append(index)
    all_indices = set(range(len(labels)))
    result = []
    for fold_index, validation in enumerate(validation_folds):
        validation = sorted(validation)
        training = sorted(all_indices - set(validation))
        result.append({"fold": fold_index, "train": training, "validation": validation})
    flattened = [index for fold in result for index in fold["validation"]]
    if sorted(flattened) != list(range(len(labels))) or len(flattened) != len(set(flattened)):
        raise AssertionError("OOF folds do not partition the training samples")
    return result


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _class_weights(targets: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=len(LABELS)).to(torch.float32)
    if torch.any(counts == 0):
        raise ValueError("Class-weighted training requires every label in the split")
    return targets.numel() / (len(LABELS) * counts)


def _standardize(
    training: torch.Tensor, other: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = training.mean(dim=0)
    std = training.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (training - mean) / std, (other - mean) / std, mean, std


def train_probe_model(
    embeddings: torch.Tensor,
    numeric: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int,
    hidden_dim: int = HIDDEN_DIM,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[PreCriticGateProbe, dict[str, Any]]:
    _seed_everything(seed)
    model = PreCriticGateProbe(
        embeddings.shape[1], numeric.shape[1], hidden_dim=hidden_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    weights = _class_weights(targets)
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        logits = model(embeddings, numeric)
        loss = F.cross_entropy(logits, targets, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    return model, {
        "final_loss": final_loss,
        "class_weights": weights.tolist(),
        "epochs": epochs,
        "learning_rate": learning_rate,
    }


def oof_predictions(
    embeddings: torch.Tensor,
    numeric: torch.Tensor,
    labels: Sequence[str],
    *,
    seed: int = SEED,
) -> tuple[torch.Tensor, list[dict[str, list[int]]], list[dict[str, Any]]]:
    folds = deterministic_stratified_folds(labels, N_SPLITS, seed)
    targets = torch.tensor([LABELS.index(label) for label in labels], dtype=torch.long)
    probabilities = torch.zeros((len(labels), len(LABELS)), dtype=torch.float32)
    training_metrics: list[dict[str, Any]] = []
    for fold in folds:
        train_indices = torch.tensor(fold["train"], dtype=torch.long)
        validation_indices = torch.tensor(fold["validation"], dtype=torch.long)
        train_numeric, validation_numeric, _, _ = _standardize(
            numeric[train_indices], numeric[validation_indices]
        )
        model, metrics = train_probe_model(
            embeddings[train_indices],
            train_numeric,
            targets[train_indices],
            seed=seed + fold["fold"] + 1,
        )
        with torch.no_grad():
            probabilities[validation_indices] = torch.softmax(
                model(embeddings[validation_indices], validation_numeric), dim=-1
            )
        training_metrics.append({"fold": fold["fold"], **metrics})
    if torch.any(probabilities.sum(dim=-1) == 0):
        raise AssertionError("Missing OOF predictions")
    return probabilities, folds, training_metrics


def gate_scores(probabilities: torch.Tensor) -> torch.Tensor:
    return (
        probabilities[:, LABELS.index("wrong_to_correct")]
        - probabilities[:, LABELS.index("correct_to_wrong")]
    )


def _threshold_above(scores: Sequence[float]) -> float:
    maximum = max(scores)
    return maximum + max(abs(maximum), 1.0) * 1e-7


def _threshold_below(scores: Sequence[float]) -> float:
    minimum = min(scores)
    return minimum - max(abs(minimum), 1.0) * 1e-7


def _gated(scores: Sequence[float], threshold: float) -> list[bool]:
    return [float(score) >= threshold for score in scores]


def _gate_outcome(labels: Sequence[str], decisions: Sequence[bool]) -> dict[str, Any]:
    if len(labels) != len(decisions):
        raise ValueError("Gate labels and decisions must have equal lengths")
    corrected = sum(
        decision and label == "wrong_to_correct"
        for label, decision in zip(labels, decisions)
    )
    degraded = sum(
        decision and label == "correct_to_wrong"
        for label, decision in zip(labels, decisions)
    )
    calls = sum(decisions)
    return {
        "corrected": corrected,
        "degraded": degraded,
        "net_benefit": corrected - degraded,
        "critic_calls": calls,
        "critic_call_rate": calls / len(labels) if labels else 0.0,
    }


def select_oof_threshold(
    scores: Sequence[float], labels: Sequence[str]
) -> dict[str, Any]:
    if not scores or len(scores) != len(labels):
        raise ValueError("OOF scores and labels must be non-empty and aligned")
    candidates: list[dict[str, Any]] = []
    for threshold in [_threshold_above(scores), *sorted(set(scores), reverse=True)]:
        outcome = _gate_outcome(labels, _gated(scores, threshold))
        candidates.append({"threshold": float(threshold), **outcome})
    best = max(
        candidates,
        key=lambda item: (
            item["net_benefit"],
            -item["critic_calls"],
            item["threshold"],
        ),
    )
    safe_fallback = best["net_benefit"] <= 0
    if safe_fallback:
        best = candidates[0]
    return {
        **best,
        "selection_source": "collection_200_stratified_5fold_oof_only",
        "selection_objective": (
            "maximize corrected-degraded; ties use lower Critic call rate"
        ),
        "safe_fallback_always_stop": safe_fallback,
        "candidate_count": len(candidates),
        "validation_used_for_selection": False,
    }


def oof_budget_thresholds(
    scores: Sequence[float], rates: Sequence[float] = BUDGET_RATES
) -> list[dict[str, Any]]:
    if not scores:
        raise ValueError("Budget thresholds require OOF scores")
    unique_desc = sorted(set(float(score) for score in scores), reverse=True)
    candidates = [
        (_threshold_above(scores), 0),
        *[
            (threshold, sum(score >= threshold for score in scores))
            for threshold in unique_desc
        ],
    ]
    result = []
    for rate in rates:
        if not 0 < rate <= 1:
            raise ValueError("Budget rates must be in (0, 1]")
        maximum_calls = math.floor(len(scores) * rate + 1e-12)
        eligible = [item for item in candidates if item[1] <= maximum_calls]
        if rate == 1.0:
            threshold, calls = _threshold_below(scores), len(scores)
        else:
            threshold, calls = max(eligible, key=lambda item: (item[1], -item[0]))
        result.append(
            {
                "target_budget_rate": float(rate),
                "threshold": float(threshold),
                "oof_critic_calls": calls,
                "oof_critic_call_rate": calls / len(scores),
                "threshold_source": "collection_200_oof_score_distribution",
            }
        )
    return result


def _policy_metrics(
    examples: Sequence[ProbeExample], decisions: Sequence[bool]
) -> dict[str, Any]:
    if len(examples) != len(decisions):
        raise ValueError("Examples and gate decisions must align")
    transitions = {label: [] for label in LABELS}
    correct_ids: list[str | int] = []
    for example, decision in zip(examples, decisions):
        answer = example.critic_only_answer if decision else example.solver_answer
        solver_correct = example.solver_answer == example.gold
        policy_correct = answer == example.gold
        if solver_correct and policy_correct:
            transition = "correct_to_correct"
        elif solver_correct:
            transition = "correct_to_wrong"
        elif policy_correct:
            transition = "wrong_to_correct"
        else:
            transition = "wrong_to_wrong"
        transitions[transition].append(example.question_id)
        if policy_correct:
            correct_ids.append(example.question_id)
    corrected = transitions["wrong_to_correct"]
    degraded = transitions["correct_to_wrong"]
    return {
        "samples": len(examples),
        "correct": len(correct_ids),
        "accuracy": len(correct_ids) / len(examples) if examples else 0.0,
        "correct_ids": correct_ids,
        "corrected": len(corrected),
        "corrected_ids": corrected,
        "degraded": len(degraded),
        "degraded_ids": degraded,
        "net_benefit": len(corrected) - len(degraded),
        "critic_calls": sum(decisions),
        "critic_call_rate": sum(decisions) / len(examples) if examples else 0.0,
        "transitions": {
            label: {"count": len(ids), "sample_ids": ids}
            for label, ids in transitions.items()
        },
    }


def aggregate_validation_cost(
    examples: Sequence[ProbeExample], decisions: Sequence[bool]
) -> dict[str, Any]:
    if len(examples) != len(decisions):
        raise ValueError("Examples and cost decisions must align")
    selected = []
    for example, decision in zip(examples, decisions):
        if example.dataset != "validation_100":
            raise ValueError("Exact stage cost is only available for validation_100")
        key = "CRITIC_ONLY" if decision else "STOP"
        cost = example.audit_case["strategy_costs"][key]
        if cost.get("available") is not True:
            raise ValueError("Saved validation stage cost is unavailable")
        selected.append(cost)
    totals = {
        field: sum(cost[field] for cost in selected)
        for field in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "calls",
            "latency_seconds",
        )
    }
    if totals["prompt_tokens"] + totals["completion_tokens"] != totals["total_tokens"]:
        raise ValueError("Aggregated validation token identity failed")
    count = len(selected)
    return {
        "service_reported_usage": True,
        "estimated": False,
        "total": totals,
        "mean": {field: totals[field] / count for field in totals},
    }


def _evaluate_validation_policy(
    examples: Sequence[ProbeExample], decisions: Sequence[bool]
) -> dict[str, Any]:
    return {
        **_policy_metrics(examples, decisions),
        "cost": aggregate_validation_cost(examples, decisions),
    }


def _hash_model_input(example: ProbeExample) -> str:
    canonical = json.dumps(
        example.model_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _input_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability_payload(row: torch.Tensor) -> dict[str, float]:
    return {label: float(row[index]) for index, label in enumerate(LABELS)}


def _report(summary: dict[str, Any]) -> str:
    validation = summary["validation_evaluation"]
    lines = [
        "# Pre-Critic Gate Learnability Probe",
        "",
        "Pure offline supervised probe. `controller_probe=true`, `deployable=false`, "
        "`final_test=false`. No LLM/backend was initialized or called.",
        "",
        "Collection 200 is used only for stratified OOF threshold selection and final "
        "Probe training. Validation 100 is evaluated once after the threshold is fixed.",
        "",
        "## OOF threshold",
        "",
        f"- Threshold: {summary['oof']['selected_threshold']['threshold']:.8f}",
        f"- OOF corrected/degraded/net: "
        f"{summary['oof']['selected_threshold']['corrected']}/"
        f"{summary['oof']['selected_threshold']['degraded']}/"
        f"{summary['oof']['selected_threshold']['net_benefit']}",
        f"- OOF Critic call rate: "
        f"{summary['oof']['selected_threshold']['critic_call_rate']:.4f}",
        f"- Safe always-STOP fallback: "
        f"{summary['oof']['selected_threshold']['safe_fallback_always_stop']}",
        "",
        "## Independent Validation 100",
        "",
        "| Policy | Accuracy | Corrected | Degraded | Net | Critic rate | Mean total tokens | Mean calls | Mean latency (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("STOP", "ALWAYS_CRITIC_ONLY", "LEARNED_GATE", "POSTHOC_ORACLE"):
        metric = validation["policies"][name]
        lines.append(
            f"| {name} | {metric['accuracy']:.4f} | {metric['corrected']} | "
            f"{metric['degraded']} | {metric['net_benefit']} | "
            f"{metric['critic_call_rate']:.4f} | "
            f"{metric['cost']['mean']['total_tokens']:.2f} | "
            f"{metric['cost']['mean']['calls']:.2f} | "
            f"{metric['cost']['mean']['latency_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`POSTHOC_ORACLE` uses gold after generation and is deployable=false.",
            "",
            "## OOF-derived budget curve on Validation 100",
            "",
            "Validation labels were not used to choose any threshold.",
            "",
            "| OOF budget | Threshold | Validation Critic rate | Accuracy | Corrected | Degraded | Mean tokens |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for point in validation["budget_curve"]:
        metric = point["validation"]
        lines.append(
            f"| {point['target_budget_rate']:.0%} | {point['threshold']:.8f} | "
            f"{metric['critic_call_rate']:.4f} | {metric['accuracy']:.4f} | "
            f"{metric['corrected']} | {metric['degraded']} | "
            f"{metric['cost']['mean']['total_tokens']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Learnability assessment",
            "",
            f"- Continue collection/formal controller training: "
            f"{summary['learnability_assessment']['worth_continuing']}",
            f"- Basis: {summary['learnability_assessment']['basis']}",
            f"- Caveat: {summary['learnability_assessment']['caveat']}",
            "",
            "## Boundary",
            "",
            "This is a learnability probe, not a deployable controller and not a final "
            "test. No hyperparameter search, prompt change, extra collection, or follow-up "
            "tuning was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_precritic_gate_probe(
    collection_rollouts: str | Path,
    validation_predictions: str | Path,
    output_dir: str | Path,
    *,
    encoder: FrozenTextEncoder | None = None,
    seed: int = SEED,
    expected_collection_samples: int = 200,
    expected_validation_samples: int = 100,
) -> dict[str, Any]:
    collection_path = Path(collection_rollouts)
    validation_path = Path(validation_predictions)
    output = Path(output_dir)
    for source in (collection_path, validation_path):
        if not source.is_file():
            raise FileNotFoundError(f"Probe input not found: {source}")
    if output.resolve() in {collection_path.parent.resolve(), validation_path.parent.resolve()}:
        raise ValueError("Probe output cannot overwrite a historical input directory")
    targets = (
        output / "summary.json",
        output / "predictions.jsonl",
        output / "report.md",
        output / "probe_model.pt",
    )
    if any(path.exists() for path in targets):
        raise FileExistsError("Probe artifacts already exist; refusing to overwrite")

    collection_rows = read_jsonl(collection_path)
    if len(collection_rows) != expected_collection_samples:
        raise ValueError(
            f"Expected {expected_collection_samples} collection rows; "
            f"found {len(collection_rows)}"
        )
    collection_examples = [
        build_probe_example(row, "collection_200") for row in collection_rows
    ]
    if len({_id_key(example.question_id) for example in collection_examples}) != len(
        collection_examples
    ):
        raise ValueError("Collection question IDs are not unique")
    active_encoder = encoder or OfflineMiniLMEncoder()
    if getattr(active_encoder, "mock_only", True):
        raise ValueError("Formal controller probe forbids mock/hash encoders")

    collection_embeddings = active_encoder.encode(
        [example.feature_text for example in collection_examples]
    )
    collection_numeric = torch.tensor(
        [example.numeric for example in collection_examples], dtype=torch.float32
    )
    labels = [example.label for example in collection_examples]
    oof_probs, folds, fold_training = oof_predictions(
        collection_embeddings, collection_numeric, labels, seed=seed
    )
    oof_scores_tensor = gate_scores(oof_probs)
    oof_scores_list = [float(value) for value in oof_scores_tensor]
    threshold = select_oof_threshold(oof_scores_list, labels)
    budget_thresholds = oof_budget_thresholds(oof_scores_list)
    learned_oof_decisions = _gated(oof_scores_list, threshold["threshold"])

    targets_tensor = torch.tensor(
        [LABELS.index(label) for label in labels], dtype=torch.long
    )
    standardized_collection, _, numeric_mean, numeric_std = _standardize(
        collection_numeric, collection_numeric
    )
    final_model, final_training = train_probe_model(
        collection_embeddings,
        standardized_collection,
        targets_tensor,
        seed=seed + 100,
    )

    # The independent validation split is only encoded and evaluated after all OOF
    # threshold choices and final training are complete.
    validation_rows = read_jsonl(validation_path)
    if len(validation_rows) != expected_validation_samples:
        raise ValueError(
            f"Expected {expected_validation_samples} validation rows; "
            f"found {len(validation_rows)}"
        )
    validation_examples = [
        build_probe_example(row, "validation_100") for row in validation_rows
    ]
    if len({_id_key(example.question_id) for example in validation_examples}) != len(
        validation_examples
    ):
        raise ValueError("Validation question IDs are not unique")
    validation_embeddings = active_encoder.encode(
        [example.feature_text for example in validation_examples]
    )
    validation_numeric = torch.tensor(
        [example.numeric for example in validation_examples], dtype=torch.float32
    )
    standardized_validation = (validation_numeric - numeric_mean) / numeric_std
    with torch.no_grad():
        validation_probs = torch.softmax(
            final_model(validation_embeddings, standardized_validation), dim=-1
        )
    validation_scores = [float(value) for value in gate_scores(validation_probs)]
    learned_decisions = _gated(validation_scores, threshold["threshold"])
    stop_decisions = [False] * len(validation_examples)
    always_decisions = [True] * len(validation_examples)
    oracle_decisions = [
        example.solver_answer != example.gold
        and example.critic_only_answer == example.gold
        for example in validation_examples
    ]
    validation_policies = {
        "STOP": _evaluate_validation_policy(validation_examples, stop_decisions),
        "ALWAYS_CRITIC_ONLY": _evaluate_validation_policy(
            validation_examples, always_decisions
        ),
        "LEARNED_GATE": _evaluate_validation_policy(
            validation_examples, learned_decisions
        ),
        "POSTHOC_ORACLE": {
            **_evaluate_validation_policy(validation_examples, oracle_decisions),
            "posthoc_oracle": True,
            "deployable": False,
        },
    }
    budget_curve = []
    for budget in budget_thresholds:
        decisions = _gated(validation_scores, budget["threshold"])
        budget_curve.append(
            {
                **budget,
                "validation": _evaluate_validation_policy(
                    validation_examples, decisions
                ),
                "validation_used_for_threshold": False,
            }
        )

    label_counts = Counter(labels)
    summary = {
        "controller_probe": True,
        "deployable": False,
        "final_test": False,
        "offline": True,
        "probe_training": True,
        "model_backend_initialized": False,
        "model_calls": 0,
        "controller_training": False,
        "existing_controllers_modified": False,
        "sources": {
            "collection_training": str(collection_path.resolve()),
            "collection_sha256": _input_sha256(collection_path),
            "validation_once": str(validation_path.resolve()),
            "validation_sha256": _input_sha256(validation_path),
        },
        "model": {
            "encoder": active_encoder.name,
            "encoder_frozen": True,
            "encoder_local_files_only": True,
            "embedding_dim": active_encoder.dimension,
            "architecture": "single_hidden_layer_mlp",
            "hidden_dim": HIDDEN_DIM,
            "numeric_features": list(NUMERIC_FEATURES),
            "classes": list(LABELS),
            "gate_score": "P(wrong_to_correct)-P(correct_to_wrong)",
            "seed": seed,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "class_weighted_cross_entropy": True,
            "hyperparameter_search": False,
        },
        "leakage_boundary": {
            "allowed": [
                "problem",
                "A-D options",
                "Solver raw output",
                "Solver parse status",
                "Solver usage",
            ],
            "excluded": [
                "gold",
                "Critic output",
                "Refiner output",
                "STOP/SHORT/FULL or other action outcomes",
            ],
        },
        "training_collection_200": {
            "samples": len(collection_examples),
            "usage_scope": (
                "No collection stage token or latency cost is estimated; only labels "
                "and exact fixed Critic call decisions are used."
            ),
            "label_counts": {label: label_counts.get(label, 0) for label in LABELS},
        },
        "oof": {
            "folds": N_SPLITS,
            "stratified": True,
            "fold_manifest": folds,
            "fold_training": fold_training,
            "selected_threshold": threshold,
            "learned_gate_metrics": _policy_metrics(
                collection_examples, learned_oof_decisions
            ),
            "budget_thresholds": budget_thresholds,
        },
        "final_training": final_training,
        "validation_evaluation": {
            "evaluated_once": True,
            "threshold_frozen_before_validation": True,
            "validation_used_for_threshold_selection": False,
            "policies": validation_policies,
            "budget_curve": budget_curve,
        },
        "learnability_assessment": {
            "worth_continuing": (
                threshold["net_benefit"] > 0
                and validation_policies["LEARNED_GATE"]["net_benefit"] > 0
            ),
            "basis": (
                "OOF threshold has positive corrected-degraded and the independently "
                "evaluated Learned Gate also has positive net benefit."
            ),
            "caveat": (
                "The signal is based on 200 training and 100 policy-selection validation "
                "samples; this Probe remains deployable=false and final_test=false."
            ),
        },
    }

    oof_fold_by_index = {
        index: fold["fold"] for fold in folds for index in fold["validation"]
    }
    prediction_rows: list[dict[str, Any]] = []
    for index, example in enumerate(collection_examples):
        decision = learned_oof_decisions[index]
        answer = example.critic_only_answer if decision else example.solver_answer
        prediction_rows.append(
            {
                "controller_probe": True,
                "deployable": False,
                "final_test": False,
                "split": "collection_200_oof",
                "question_id": example.question_id,
                "gold": example.gold,
                "label": example.label,
                "oof_fold": oof_fold_by_index[index],
                "probabilities": _probability_payload(oof_probs[index]),
                "gate_score": oof_scores_list[index],
                "threshold": threshold["threshold"],
                "critic_called": decision,
                "selected_answer": answer,
                "correct": answer == example.gold,
                "model_input_sha256": _hash_model_input(example),
            }
        )
    for index, example in enumerate(validation_examples):
        decision = learned_decisions[index]
        answer = example.critic_only_answer if decision else example.solver_answer
        prediction_rows.append(
            {
                "controller_probe": True,
                "deployable": False,
                "final_test": False,
                "split": "validation_100_once",
                "question_id": example.question_id,
                "gold": example.gold,
                "label": example.label,
                "probabilities": _probability_payload(validation_probs[index]),
                "gate_score": validation_scores[index],
                "threshold": threshold["threshold"],
                "critic_called": decision,
                "selected_answer": answer,
                "correct": answer == example.gold,
                "model_input_sha256": _hash_model_input(example),
                "selected_cost": example.audit_case["strategy_costs"][
                    "CRITIC_ONLY" if decision else "STOP"
                ],
            }
        )

    checkpoint = {
        "controller_probe": True,
        "deployable": False,
        "final_test": False,
        "model_class": "PreCriticGateProbe",
        "model_state_dict": final_model.state_dict(),
        "embedding_dim": active_encoder.dimension,
        "numeric_dim": len(NUMERIC_FEATURES),
        "hidden_dim": HIDDEN_DIM,
        "classes": list(LABELS),
        "encoder_name": active_encoder.name,
        "encoder_frozen": True,
        "numeric_mean": numeric_mean,
        "numeric_std": numeric_std,
        "threshold": threshold,
        "seed": seed,
        "training_config": {
            "folds": N_SPLITS,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "class_weighted_cross_entropy": True,
            "hyperparameter_search": False,
        },
        "input_schema": {
            "allowed": summary["leakage_boundary"]["allowed"],
            "excluded": summary["leakage_boundary"]["excluded"],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "summary.json", summary)
    write_jsonl(output / "predictions.jsonl", prediction_rows)
    _write_text_atomic(output / "report.md", _report(summary))
    _save_checkpoint_atomic(output / "probe_model.pt", checkpoint)
    return summary


def _id_key(value: str | int) -> str:
    return json.dumps([type(value).__name__, value], ensure_ascii=False)
