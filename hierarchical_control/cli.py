from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .backend import MockBackend, OpenAIBackend
from .collection import collect_action_labels, collect_budget_labels
from .config import Action, BudgetTier, ExperimentConfig
from .critic_gating_audit import run_critic_gating_audit
from .encoders import HashingEncoder, MiniLMEncoder
from .engine import CollaborationEngine
from .evaluator import ExactMatchEvaluator, MockEvaluator
from .graph import build_workflow
from .io_utils import read_jsonl, update_metrics, write_jsonl
from .logiqa_audit import run_logiqa_audit
from .logiqa_action_collection import (
    load_action_collection_settings,
    run_logiqa_action_collection,
)
from .logiqa_pilot import run_logiqa_pilot
from .logiqa_policy_validation import (
    load_validation_settings,
    run_logiqa_policy_validation,
)
from .logiqa_prompts import MINIMAL_V1, PROMPT_VERSIONS
from .logiqa_replay import load_logiqa_replay_settings, run_logiqa_prompt_replay
from .precritic_probe import run_precritic_gate_probe
from .precritic_collection import (
    collect_precritic_rollouts,
    prepare_precritic_collection,
)
from .precritic_controller_v1 import train_precritic_controller_v1
from .precritic_controller_v1_audit import run_precritic_controller_v1_audit
from .precritic_controller_v2 import train_precritic_controller_v2
from .precritic_controller_v3 import run_precritic_controller_v3_smoke
from .precritic_controller_v3_audit import run_precritic_controller_v3_generalization_audit
from .precritic_controller_v3_training import train_precritic_controller_v3
from .precritic_representation_audit import (
    run_precritic_representation_audit,
)
from .precritic_training_protocol import prepare_precritic_training_protocol
from .prompt_stability_audit import run_prompt_stability_audit
from .predictors import ActionModelPredictor, BudgetModelPredictor
from .training import train_action_controller, train_budget_allocator


def _config(path: str | None) -> ExperimentConfig:
    return ExperimentConfig.from_json(path) if path else ExperimentConfig()


def _backend(args: argparse.Namespace):
    if args.backend == "mock":
        return MockBackend()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Provide --api-key or OPENAI_API_KEY for the OpenAI-compatible backend")
    return OpenAIBackend(args.base_url, api_key, args.model, args.temperature, args.timeout)


def _evaluator(args: argparse.Namespace):
    evaluator_name = args.evaluator
    if evaluator_name is None:
        evaluator_name = "mock" if args.backend == "mock" else "exact"
    if evaluator_name == "mock":
        if args.backend != "mock":
            raise ValueError("MockEvaluator may only be paired with MockBackend")
        return MockEvaluator()
    return ExactMatchEvaluator()


def _encoder(args: argparse.Namespace):
    if args.encoder == "hashing":
        if not getattr(args, "allow_mock_encoder", False):
            raise ValueError("HashingEncoder is mock-only; pass --allow-mock-encoder explicitly")
        return HashingEncoder(args.embedding_dim)
    return MiniLMEncoder(args.minilm_model, args.device)


def _output_dir(args: argparse.Namespace) -> Path:
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def command_collect_budget(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _output_dir(args)
    config = _config(args.config)
    backend = _backend(args)
    evaluator = _evaluator(args)
    result = collect_budget_labels(
        read_jsonl(args.input), backend_engine(backend, config), evaluator, args.include_unsolved
    )
    write_jsonl(output_dir / "budget_rollouts.jsonl", result.rollouts)
    write_jsonl(output_dir / "budget_training.jsonl", result.training)
    update_metrics(output_dir / "metrics.json", "collect_budget", result.metrics, backend.mock_only)
    return result.metrics


def command_train_budget(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _output_dir(args)
    config = _config(args.config)
    encoder = _encoder(args)
    result = train_budget_allocator(
        read_jsonl(args.input or output_dir / "budget_training.jsonl"),
        encoder,
        config,
        output_dir / "budget_allocator.pt",
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.hidden_dim,
    )
    update_metrics(output_dir / "metrics.json", "train_budget", result.metrics, encoder.mock_only)
    return result.metrics


def command_collect_counterfactual(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _output_dir(args)
    config = _config(args.config)
    backend = _backend(args)
    evaluator = _evaluator(args)
    result = collect_action_labels(read_jsonl(args.input), backend_engine(backend, config), evaluator)
    write_jsonl(output_dir / "counterfactual_rollouts.jsonl", result.rollouts)
    write_jsonl(output_dir / "action_training.jsonl", result.training)
    update_metrics(
        output_dir / "metrics.json", "collect_counterfactual", result.metrics, backend.mock_only
    )
    return result.metrics


def command_train_action(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _output_dir(args)
    config = _config(args.config)
    encoder = _encoder(args)
    result = train_action_controller(
        read_jsonl(args.input or output_dir / "action_training.jsonl"),
        encoder,
        config,
        output_dir / "action_controller.pt",
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.hidden_dim,
        args.auxiliary_weight,
    )
    update_metrics(output_dir / "metrics.json", "train_action", result.metrics, encoder.mock_only)
    return result.metrics


def backend_engine(backend, config: ExperimentConfig) -> CollaborationEngine:
    return CollaborationEngine(backend=backend, config=config)


def command_pilot_logiqa(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args.config)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "local"
    backend = OpenAIBackend(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        extra_body=config.pilot_request_extra_body,
    )
    return run_logiqa_pilot(
        data_path=args.data_path,
        output_dir=args.output_dir,
        backend=backend,
        config=config,
        limit=args.limit,
        seed=args.seed,
    )


def command_audit_logiqa_pilot(args: argparse.Namespace) -> dict[str, Any]:
    return run_logiqa_audit(args.predictions, args.output_dir)


def command_replay_logiqa_prompts(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_logiqa_replay_settings(args.predictions)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "local"
    backend = OpenAIBackend(
        base_url=settings.base_url,
        api_key=api_key,
        model=settings.model,
        temperature=settings.temperature,
        timeout=args.timeout,
        extra_body=settings.extra_body,
    )
    return run_logiqa_prompt_replay(
        predictions_path=args.predictions,
        prompt_version=args.prompt_version,
        output_dir=args.output_dir,
        backend=backend,
    )


def command_audit_prompt_stability(args: argparse.Namespace) -> dict[str, Any]:
    return run_prompt_stability_audit(
        minimal_predictions=args.minimal_predictions,
        structured_predictions=args.structured_predictions,
        output_dir=args.output_dir,
    )


def command_audit_critic_gating(args: argparse.Namespace) -> dict[str, Any]:
    return run_critic_gating_audit(
        collection_rollouts=args.collection_rollouts,
        validation_predictions=args.validation_predictions,
        output_dir=args.output_dir,
    )


def command_probe_precritic_gate(args: argparse.Namespace) -> dict[str, Any]:
    return run_precritic_gate_probe(
        collection_rollouts=args.collection_rollouts,
        validation_predictions=args.validation_predictions,
        output_dir=args.output_dir,
    )


def command_prepare_precritic_collection(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_action_collection_settings()
    return prepare_precritic_collection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        settings=settings,
        mock_only=args.mock_only,
    )


def command_collect_precritic_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_action_collection_settings()
    if args.backend == "mock":
        backend = MockBackend()
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "local"
        backend = OpenAIBackend(
            base_url=settings.base_url,
            api_key=api_key,
            model=settings.model,
            temperature=settings.temperature,
            timeout=args.timeout,
            extra_body=settings.extra_body,
        )
    return collect_precritic_rollouts(
        data_path=args.data_path,
        output_dir=args.output_dir,
        backend=backend,
        settings=settings,
    )


def command_prepare_precritic_training_protocol(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return prepare_precritic_training_protocol(
        collection_200_path=args.collection_200,
        collection_800_path=args.collection_800,
        pilot_predictions_path=args.pilot_predictions,
        validation_predictions_path=args.validation_predictions,
        dev_data_path=args.dev_data_path,
        training_output_dir=args.training_output_dir,
        final_test_output_dir=args.final_test_output_dir,
    )


def command_train_precritic_controller_v1(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return train_precritic_controller_v1(
        training_path=args.training,
        training_manifest_path=args.training_manifest,
        validation_path=args.validation,
        old_probe_predictions_path=args.old_probe,
        old_probe_summary_path=args.old_probe_summary,
        final_test_manifest_path=args.final_test_manifest,
        output_dir=args.output_dir,
    )


def command_audit_precritic_controller_v1(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return run_precritic_controller_v1_audit(
        controller_dir=args.controller_dir,
        training_path=args.training,
        training_manifest_path=args.training_manifest,
        validation_path=args.validation,
        old_probe_predictions_path=args.old_probe,
        old_probe_summary_path=args.old_probe_summary,
        final_test_manifest_path=args.final_test_manifest,
        run_metadata_prefix=args.run_metadata_prefix,
        output_dir=args.output_dir,
    )


def command_train_precritic_controller_v2(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return train_precritic_controller_v2(
        training_path=args.training,
        training_manifest_path=args.training_manifest,
        validation_path=args.validation,
        validation_summary_path=args.validation_summary,
        old_probe_predictions_path=args.old_probe,
        old_probe_summary_path=args.old_probe_summary,
        controller_v1_dir=args.controller_v1_dir,
        final_test_manifest_path=args.final_test_manifest,
        output_dir=args.output_dir,
    )


def command_audit_precritic_representation(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return run_precritic_representation_audit(
        training_path=args.training,
        training_manifest_path=args.training_manifest,
        validation_path=args.validation,
        final_test_manifest_path=args.final_test_manifest,
        output_dir=args.output_dir,
    )


def command_smoke_precritic_controller_v3(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return run_precritic_controller_v3_smoke(
        training_path=args.training,
        training_manifest_path=args.training_manifest,
        final_test_manifest_path=args.final_test_manifest,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
    )


def command_train_precritic_controller_v3(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return train_precritic_controller_v3(
        training_path=args.training,
        training_manifest_path=args.training_manifest,
        validation_path=args.validation,
        final_test_manifest_path=args.final_test_manifest,
        controller_v1_dir=args.controller_v1_dir,
        controller_v2_dir=args.controller_v2_dir,
        output_dir=args.output_dir,
    )


def command_audit_precritic_controller_v3_generalization(
    args: argparse.Namespace,
) -> dict[str, Any]:
    return run_precritic_controller_v3_generalization_audit(
        controller_dir=args.controller_dir,
        training_path=args.training,
        training_manifest_path=args.training_manifest,
        validation_path=args.validation,
        final_test_manifest_path=args.final_test_manifest,
        output_dir=args.output_dir,
    )


def command_validate_logiqa_policies(args: argparse.Namespace) -> dict[str, Any]:
    settings = load_validation_settings(args.pilot_predictions)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "local"
    backend = OpenAIBackend(
        base_url=settings.base_url,
        api_key=api_key,
        model=settings.model,
        temperature=settings.temperature,
        timeout=args.timeout,
        extra_body=settings.extra_body,
    )
    return run_logiqa_policy_validation(
        pilot_predictions=args.pilot_predictions,
        output_dir=args.output_dir,
        backend=backend,
    )


def command_collect_logiqa_action_rollouts(
    args: argparse.Namespace,
) -> dict[str, Any]:
    settings = load_action_collection_settings()
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "local"
    backend = OpenAIBackend(
        base_url=settings.base_url,
        api_key=api_key,
        model=settings.model,
        temperature=settings.temperature,
        timeout=args.timeout,
        extra_body=settings.extra_body,
    )
    return run_logiqa_action_collection(
        data_path=args.data_path,
        output_dir=args.output_dir,
        backend=backend,
        settings=settings,
    )


def _toy_examples() -> list[dict[str, Any]]:
    return [
        {"id": "toy-zero", "query": "Return the base result. [toy=zero]", "difficulty": 0},
        {"id": "toy-low", "query": "Apply a small correction. [toy=low]", "difficulty": 1},
        {"id": "toy-medium", "query": "Solve a two-step task. [toy=medium]", "difficulty": 2},
        {"id": "toy-high", "query": "Solve a difficult task. [toy=high]", "difficulty": 3},
        {"id": "toy-unsolved", "query": "This toy task exceeds all tiers.", "difficulty": 4},
    ]


def command_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _output_dir(args)
    config = _config(args.config)
    config.save_json(output_dir / "smoke_config.json")
    examples = _toy_examples()
    toy_path = output_dir / "toy_data.jsonl"
    write_jsonl(toy_path, examples)

    budget_backend = MockBackend()
    budget_engine = backend_engine(budget_backend, config)
    budget_result = collect_budget_labels(examples, budget_engine, MockEvaluator())
    write_jsonl(output_dir / "budget_rollouts.jsonl", budget_result.rollouts)
    write_jsonl(output_dir / "budget_training.jsonl", budget_result.training)
    expected_labels = ["ZERO", "LOW", "MEDIUM", "HIGH", None]
    actual_labels = [record["budget_label"] for record in budget_result.rollouts]
    if actual_labels != expected_labels:
        raise AssertionError(f"Budget label selection failed: {actual_labels}")
    if sum(call["purpose"] == "solver" for call in budget_backend.calls) != len(examples):
        raise AssertionError("Budget collection did not reuse exactly one Solver state per query")
    for rollout in budget_result.rollouts:
        for candidate in rollout["candidates"]:
            limit = config.tier(candidate["tier"])
            cost = candidate["actual_cost"]
            if (
                cost["extra_completion_tokens"] > limit.extra_tokens
                or cost["extra_calls"] > limit.extra_calls
            ):
                raise AssertionError("A budget rollout exceeded its allocation")

    action_backend = MockBackend()
    action_engine = backend_engine(action_backend, config)
    action_result = collect_action_labels(examples, action_engine, MockEvaluator())
    write_jsonl(output_dir / "counterfactual_rollouts.jsonl", action_result.rollouts)
    write_jsonl(output_dir / "action_training.jsonl", action_result.training)

    probe = action_engine.solve_once("probe", {"difficulty": 1})
    calls_before = len(action_backend.calls)
    probe = action_engine.execute_action(probe, Action.SKIP, config.collection_limit)
    probe = action_engine.execute_action(probe, Action.STOP, config.collection_limit)
    if len(action_backend.calls) != calls_before:
        raise AssertionError("SKIP or STOP unexpectedly called the backend")

    encoder = HashingEncoder(args.embedding_dim)
    budget_train = train_budget_allocator(
        budget_result.training,
        encoder,
        config,
        output_dir / "budget_allocator.pt",
        epochs=args.budget_epochs,
        hidden_dim=args.budget_hidden_dim,
    )
    action_train = train_action_controller(
        action_result.training,
        encoder,
        config,
        output_dir / "action_controller.pt",
        epochs=args.action_epochs,
        hidden_dim=args.action_hidden_dim,
    )

    graph_backend = MockBackend()
    graph_engine = backend_engine(graph_backend, config)
    budget_predictor = BudgetModelPredictor(output_dir / "budget_allocator.pt", encoder, config)
    action_predictor = ActionModelPredictor(output_dir / "action_controller.pt", encoder, config)
    graph = build_workflow(graph_engine, budget_predictor, action_predictor)
    graph_result = graph.invoke(
        {
            "query": "Graph smoke query",
            "metadata": {"difficulty": 1},
            "max_budget": config.tier(BudgetTier.HIGH),
        }
    )
    final_state = graph_result["agent_state"]
    allocated = graph_result["allocated_budget"]
    if (
        not final_state.terminated
        or final_state.usage.extra_completion_tokens > allocated.extra_tokens
        or final_state.usage.extra_calls > allocated.extra_calls
    ):
        raise AssertionError("LangGraph runtime failed its termination or budget invariant")

    required = [
        "budget_rollouts.jsonl",
        "budget_training.jsonl",
        "counterfactual_rollouts.jsonl",
        "action_training.jsonl",
        "budget_allocator.pt",
        "action_controller.pt",
    ]
    for name in required:
        if not (output_dir / name).is_file():
            raise AssertionError(f"Missing smoke artifact: {name}")
    smoke_metrics = {
        "status": "passed",
        "device": "cpu",
        "network_used": False,
        "mock_only": True,
        "checks": {
            "budget_not_exceeded": True,
            "skip_stop_zero_calls": True,
            "counterfactual_branch_isolation": True,
            "label_selection": True,
            "controller_forward_backward_checkpoint": True,
            "checkpoint_load_and_inference": True,
            "langgraph_execution": True,
        },
        "artifacts": required + ["metrics.json"],
    }
    update_metrics(output_dir / "metrics.json", "collect_budget", budget_result.metrics, True)
    update_metrics(output_dir / "metrics.json", "collect_counterfactual", action_result.metrics, True)
    update_metrics(output_dir / "metrics.json", "train_budget", budget_train.metrics, True)
    update_metrics(output_dir / "metrics.json", "train_action", action_train.metrics, True)
    update_metrics(output_dir / "metrics.json", "smoke", smoke_metrics, True)
    return smoke_metrics


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["mock", "openai"], default="mock")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--model", default="localmodel")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--evaluator", choices=["mock", "exact"])


def _add_training_options(parser: argparse.ArgumentParser, action: bool = False) -> None:
    parser.add_argument("--encoder", choices=["minilm", "hashing"], default="minilm")
    parser.add_argument("--minilm-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--allow-mock-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=50 if action else 40)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128 if action else 96)
    if action:
        parser.add_argument("--auxiliary-weight", type=float, default=0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hierarchical-control")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_budget = subparsers.add_parser("collect-budget")
    collect_budget.add_argument("--input", required=True)
    collect_budget.add_argument("--output-dir", required=True)
    collect_budget.add_argument("--config")
    collect_budget.add_argument("--include-unsolved", action="store_true")
    _add_runtime_options(collect_budget)
    collect_budget.set_defaults(handler=command_collect_budget)

    train_budget = subparsers.add_parser("train-budget")
    train_budget.add_argument("--input")
    train_budget.add_argument("--output-dir", required=True)
    train_budget.add_argument("--config")
    _add_training_options(train_budget)
    train_budget.set_defaults(handler=command_train_budget)

    collect_action = subparsers.add_parser("collect-counterfactual")
    collect_action.add_argument("--input", required=True)
    collect_action.add_argument("--output-dir", required=True)
    collect_action.add_argument("--config")
    _add_runtime_options(collect_action)
    collect_action.set_defaults(handler=command_collect_counterfactual)

    train_action = subparsers.add_parser("train-action")
    train_action.add_argument("--input")
    train_action.add_argument("--output-dir", required=True)
    train_action.add_argument("--config")
    _add_training_options(train_action, action=True)
    train_action.set_defaults(handler=command_train_action)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", required=True)
    smoke.add_argument("--config")
    smoke.add_argument("--embedding-dim", type=int, default=64)
    smoke.add_argument("--budget-epochs", type=int, default=25)
    smoke.add_argument("--action-epochs", type=int, default=30)
    smoke.add_argument("--budget-hidden-dim", type=int, default=64)
    smoke.add_argument("--action-hidden-dim", type=int, default=96)
    smoke.set_defaults(handler=command_smoke)

    pilot = subparsers.add_parser("pilot-logiqa")
    pilot.add_argument("--data-path", required=True)
    pilot.add_argument("--limit", type=int, default=50)
    pilot.add_argument("--seed", type=int, default=20260810)
    pilot.add_argument("--output-dir", required=True)
    pilot.add_argument("--config")
    pilot.add_argument("--base-url", default="http://localhost:8000/v1")
    pilot.add_argument("--api-key")
    pilot.add_argument("--model", default="localmodel")
    pilot.add_argument("--temperature", type=float, default=0.0)
    pilot.add_argument("--timeout", type=float, default=120.0)
    pilot.set_defaults(handler=command_pilot_logiqa)

    audit = subparsers.add_parser("audit-logiqa-pilot")
    audit.add_argument("--predictions", "--input", dest="predictions", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.set_defaults(handler=command_audit_logiqa_pilot)

    replay = subparsers.add_parser("replay-logiqa-prompts")
    replay.add_argument("--predictions", required=True)
    replay.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default=MINIMAL_V1)
    replay.add_argument("--output-dir", required=True)
    replay.add_argument("--api-key")
    replay.add_argument("--timeout", type=float, default=120.0)
    replay.set_defaults(handler=command_replay_logiqa_prompts)

    stability = subparsers.add_parser("audit-prompt-stability")
    stability.add_argument(
        "--minimal-predictions",
        "--v1-predictions",
        dest="minimal_predictions",
        default="artifacts/pilot_logiqa/predictions.jsonl",
    )
    stability.add_argument(
        "--structured-predictions",
        "--v2-predictions",
        dest="structured_predictions",
        default=(
            "artifacts/pilot_logiqa/prompt_dev_structured_v2/predictions.jsonl"
        ),
    )
    stability.add_argument(
        "--output-dir",
        default="artifacts/pilot_logiqa/prompt_stability_audit",
    )
    stability.set_defaults(handler=command_audit_prompt_stability)

    critic_gating = subparsers.add_parser("audit-critic-gating")
    critic_gating.add_argument(
        "--collection-rollouts",
        "--collection",
        dest="collection_rollouts",
        default="artifacts/logiqa_action_collection_200/rollouts.jsonl",
    )
    critic_gating.add_argument(
        "--validation-predictions",
        "--validation",
        dest="validation_predictions",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    critic_gating.add_argument(
        "--output-dir",
        default="artifacts/critic_gating_audit",
    )
    critic_gating.set_defaults(handler=command_audit_critic_gating)

    precritic_probe = subparsers.add_parser("probe-precritic-gate")
    precritic_probe.add_argument(
        "--collection-rollouts",
        "--collection",
        dest="collection_rollouts",
        default="artifacts/logiqa_action_collection_200/rollouts.jsonl",
    )
    precritic_probe.add_argument(
        "--validation-predictions",
        "--validation",
        dest="validation_predictions",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    precritic_probe.add_argument(
        "--output-dir",
        default="artifacts/precritic_gate_probe",
    )
    precritic_probe.set_defaults(handler=command_probe_precritic_gate)

    prepare_precritic = subparsers.add_parser("prepare-precritic-collection")
    prepare_precritic.add_argument("--data-path", required=True)
    prepare_precritic.add_argument(
        "--output-dir",
        default="artifacts/logiqa_precritic_collection_800",
    )
    prepare_precritic.add_argument(
        "--mock-only",
        action="store_true",
        help="Prepare a separate split intended only for MockBackend flow tests.",
    )
    prepare_precritic.set_defaults(handler=command_prepare_precritic_collection)

    collect_precritic = subparsers.add_parser("collect-precritic-rollouts")
    collect_precritic.add_argument("--data-path", required=True)
    collect_precritic.add_argument(
        "--output-dir",
        default="artifacts/logiqa_precritic_collection_800",
    )
    collect_precritic.add_argument(
        "--backend", choices=("openai", "mock"), default="openai"
    )
    collect_precritic.add_argument("--api-key")
    collect_precritic.add_argument("--timeout", type=float, default=120.0)
    collect_precritic.set_defaults(handler=command_collect_precritic_rollouts)

    training_protocol = subparsers.add_parser(
        "prepare-precritic-training-protocol"
    )
    training_protocol.add_argument(
        "--collection-200",
        default="artifacts/logiqa_action_collection_200/rollouts.jsonl",
    )
    training_protocol.add_argument(
        "--collection-800",
        default="artifacts/logiqa_precritic_collection_800/rollouts.jsonl",
    )
    training_protocol.add_argument(
        "--pilot-predictions",
        default="artifacts/pilot_logiqa/predictions.jsonl",
    )
    training_protocol.add_argument(
        "--validation-predictions",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    training_protocol.add_argument(
        "--dev-data-path",
        default="/tmp/logiqa2-dev.txt",
    )
    training_protocol.add_argument(
        "--training-output-dir",
        default="artifacts/precritic_training_1000",
    )
    training_protocol.add_argument(
        "--final-test-output-dir",
        default="artifacts/logiqa_final_test_500",
    )
    training_protocol.set_defaults(
        handler=command_prepare_precritic_training_protocol
    )

    controller_v1 = subparsers.add_parser("train-precritic-controller-v1")
    controller_v1.add_argument(
        "--training",
        default="artifacts/precritic_training_1000/training_examples.jsonl",
    )
    controller_v1.add_argument(
        "--training-manifest",
        default="artifacts/precritic_training_1000/manifest.json",
    )
    controller_v1.add_argument(
        "--validation",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    controller_v1.add_argument(
        "--old-probe",
        default="artifacts/precritic_gate_probe/predictions.jsonl",
    )
    controller_v1.add_argument(
        "--old-probe-summary",
        default="artifacts/precritic_gate_probe/summary.json",
    )
    controller_v1.add_argument(
        "--final-test-manifest",
        default="artifacts/logiqa_final_test_500/split_manifest.json",
    )
    controller_v1.add_argument(
        "--output-dir",
        default="artifacts/precritic_controller_v1",
    )
    controller_v1.set_defaults(handler=command_train_precritic_controller_v1)

    controller_v1_audit = subparsers.add_parser(
        "audit-precritic-controller-v1"
    )
    controller_v1_audit.add_argument(
        "--controller-dir",
        default="artifacts/precritic_controller_v1",
    )
    controller_v1_audit.add_argument(
        "--training",
        default="artifacts/precritic_training_1000/training_examples.jsonl",
    )
    controller_v1_audit.add_argument(
        "--training-manifest",
        default="artifacts/precritic_training_1000/manifest.json",
    )
    controller_v1_audit.add_argument(
        "--validation",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    controller_v1_audit.add_argument(
        "--old-probe",
        default="artifacts/precritic_gate_probe/predictions.jsonl",
    )
    controller_v1_audit.add_argument(
        "--old-probe-summary",
        default="artifacts/precritic_gate_probe/summary.json",
    )
    controller_v1_audit.add_argument(
        "--final-test-manifest",
        default="artifacts/logiqa_final_test_500/split_manifest.json",
    )
    controller_v1_audit.add_argument(
        "--run-metadata-prefix",
        default="/tmp/counterfactualagent_precritic_controller_v1",
    )
    controller_v1_audit.add_argument(
        "--output-dir",
        default="artifacts/precritic_controller_v1/audit",
    )
    controller_v1_audit.set_defaults(
        handler=command_audit_precritic_controller_v1
    )

    controller_v2 = subparsers.add_parser(
        "train-precritic-controller-v2"
    )
    controller_v2.add_argument(
        "--training",
        default="artifacts/precritic_training_1000/training_examples.jsonl",
    )
    controller_v2.add_argument(
        "--training-manifest",
        default="artifacts/precritic_training_1000/manifest.json",
    )
    controller_v2.add_argument(
        "--validation",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    controller_v2.add_argument(
        "--validation-summary",
        default="artifacts/logiqa_policy_validation_100/summary.json",
    )
    controller_v2.add_argument(
        "--old-probe",
        default="artifacts/precritic_gate_probe/predictions.jsonl",
    )
    controller_v2.add_argument(
        "--old-probe-summary",
        default="artifacts/precritic_gate_probe/summary.json",
    )
    controller_v2.add_argument(
        "--controller-v1-dir",
        default="artifacts/precritic_controller_v1",
    )
    controller_v2.add_argument(
        "--final-test-manifest",
        default="artifacts/logiqa_final_test_500/split_manifest.json",
    )
    controller_v2.add_argument(
        "--output-dir",
        default="artifacts/precritic_controller_v2_factorized",
    )
    controller_v2.set_defaults(handler=command_train_precritic_controller_v2)

    representation_audit = subparsers.add_parser(
        "audit-precritic-representation"
    )
    representation_audit.add_argument(
        "--training",
        default="artifacts/precritic_training_1000/training_examples.jsonl",
    )
    representation_audit.add_argument(
        "--training-manifest",
        default="artifacts/precritic_training_1000/manifest.json",
    )
    representation_audit.add_argument(
        "--validation",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    representation_audit.add_argument(
        "--final-test-manifest",
        default="artifacts/logiqa_final_test_500/split_manifest.json",
    )
    representation_audit.add_argument(
        "--output-dir",
        default="artifacts/precritic_representation_audit",
    )
    representation_audit.set_defaults(
        handler=command_audit_precritic_representation
    )

    controller_v3_smoke = subparsers.add_parser(
        "smoke-precritic-controller-v3"
    )
    controller_v3_smoke.add_argument(
        "--training",
        default="artifacts/precritic_training_1000/training_examples.jsonl",
    )
    controller_v3_smoke.add_argument(
        "--training-manifest",
        default="artifacts/precritic_training_1000/manifest.json",
    )
    controller_v3_smoke.add_argument(
        "--final-test-manifest",
        default="artifacts/logiqa_final_test_500/split_manifest.json",
    )
    controller_v3_smoke.add_argument(
        "--output-dir",
        default="artifacts/precritic_controller_v3_smoke",
    )
    controller_v3_smoke.add_argument(
        "--sample-count", type=int, default=8
    )
    controller_v3_smoke.set_defaults(
        handler=command_smoke_precritic_controller_v3
    )

    controller_v3_train = subparsers.add_parser(
        "train-precritic-controller-v3"
    )
    controller_v3_train.add_argument(
        "--training",
        default="artifacts/precritic_training_1000/training_examples.jsonl",
    )
    controller_v3_train.add_argument(
        "--training-manifest",
        default="artifacts/precritic_training_1000/manifest.json",
    )
    controller_v3_train.add_argument(
        "--validation",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    controller_v3_train.add_argument(
        "--final-test-manifest",
        default="artifacts/logiqa_final_test_500/split_manifest.json",
    )
    controller_v3_train.add_argument(
        "--controller-v1-dir",
        default="artifacts/precritic_controller_v1",
    )
    controller_v3_train.add_argument(
        "--controller-v2-dir",
        default="artifacts/precritic_controller_v2_factorized",
    )
    controller_v3_train.add_argument(
        "--output-dir",
        default="artifacts/precritic_controller_v3",
    )
    controller_v3_train.set_defaults(
        handler=command_train_precritic_controller_v3
    )

    controller_v3_audit = subparsers.add_parser(
        "audit-precritic-controller-v3-generalization"
    )
    controller_v3_audit.add_argument(
        "--controller-dir",
        default="artifacts/precritic_controller_v3",
    )
    controller_v3_audit.add_argument(
        "--training",
        default="artifacts/precritic_training_1000/training_examples.jsonl",
    )
    controller_v3_audit.add_argument(
        "--training-manifest",
        default="artifacts/precritic_training_1000/manifest.json",
    )
    controller_v3_audit.add_argument(
        "--validation",
        default="artifacts/logiqa_policy_validation_100/predictions.jsonl",
    )
    controller_v3_audit.add_argument(
        "--final-test-manifest",
        default="artifacts/logiqa_final_test_500/split_manifest.json",
    )
    controller_v3_audit.add_argument(
        "--output-dir",
        default="artifacts/precritic_controller_v3/generalization_audit",
    )
    controller_v3_audit.set_defaults(
        handler=command_audit_precritic_controller_v3_generalization
    )

    validation = subparsers.add_parser("validate-logiqa-policies")
    validation.add_argument(
        "--pilot-predictions",
        default="artifacts/pilot_logiqa/predictions.jsonl",
    )
    validation.add_argument(
        "--output-dir",
        default="artifacts/logiqa_policy_validation_100",
    )
    validation.add_argument("--api-key")
    validation.add_argument("--timeout", type=float, default=120.0)
    validation.set_defaults(handler=command_validate_logiqa_policies)

    action_rollouts = subparsers.add_parser("collect-logiqa-action-rollouts")
    action_rollouts.add_argument("--data-path", required=True)
    action_rollouts.add_argument(
        "--output-dir",
        default="artifacts/logiqa_action_collection_200",
    )
    action_rollouts.add_argument("--api-key")
    action_rollouts.add_argument("--timeout", type=float, default=120.0)
    action_rollouts.set_defaults(handler=command_collect_logiqa_action_rollouts)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
