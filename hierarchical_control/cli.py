from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .backend import MockBackend, OpenAIBackend
from .collection import collect_action_labels, collect_budget_labels
from .config import Action, BudgetTier, ExperimentConfig
from .encoders import HashingEncoder, MiniLMEncoder
from .engine import CollaborationEngine
from .evaluator import ExactMatchEvaluator, MockEvaluator
from .graph import build_workflow
from .io_utils import read_jsonl, update_metrics, write_jsonl
from .logiqa_audit import run_logiqa_audit
from .logiqa_pilot import run_logiqa_pilot
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
            if cost["extra_tokens"] > limit.extra_tokens or cost["extra_calls"] > limit.extra_calls:
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
        or final_state.usage.extra_tokens > allocated.extra_tokens
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
