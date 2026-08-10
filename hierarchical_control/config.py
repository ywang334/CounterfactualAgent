from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class BudgetTier(str, Enum):
    ZERO = "ZERO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Action(str, Enum):
    SKIP = "SKIP"
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    FULL = "FULL"
    STOP = "STOP"


BUDGET_LABELS = [tier.value for tier in BudgetTier]
ACTION_LABELS = [action.value for action in Action]


@dataclass(frozen=True)
class BudgetLimit:
    extra_tokens: int
    extra_calls: int

    def __post_init__(self) -> None:
        if self.extra_tokens < 0 or self.extra_calls < 0:
            raise ValueError("Budget values must be non-negative")

    def to_dict(self) -> dict[str, int]:
        return {"extra_tokens": self.extra_tokens, "extra_calls": self.extra_calls}


def _default_tiers() -> dict[str, BudgetLimit]:
    # One critic and one refiner call at the tier's generation granularity.
    return {
        BudgetTier.ZERO.value: BudgetLimit(0, 0),
        BudgetTier.LOW.value: BudgetLimit(128, 2),
        BudgetTier.MEDIUM.value: BudgetLimit(384, 2),
        BudgetTier.HIGH.value: BudgetLimit(1024, 2),
    }


def _default_action_caps() -> dict[str, int]:
    return {
        Action.SHORT.value: 64,
        Action.MEDIUM.value: 192,
        Action.FULL.value: 512,
    }


def _default_pilot_caps() -> dict[str, int]:
    return {"solver": 512, "critic": 512, "refiner": 512}


def _default_pilot_extra_body() -> dict[str, Any]:
    return {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class ExperimentConfig:
    budget_tiers: dict[str, BudgetLimit] = field(default_factory=_default_tiers)
    action_token_caps: dict[str, int] = field(default_factory=_default_action_caps)
    solver_max_tokens: int = 256
    pilot_generation_caps: dict[str, int] = field(default_factory=_default_pilot_caps)
    pilot_request_extra_body: dict[str, Any] = field(default_factory=_default_pilot_extra_body)
    max_collaboration_steps: int = 2
    collection_limit: BudgetLimit = field(default_factory=lambda: BudgetLimit(4096, 16))
    reference_action: str = Action.SHORT.value
    quality_tolerance: float = 1e-8
    call_cost_weight: float = 1024.0
    seed: int = 7

    def __post_init__(self) -> None:
        missing_tiers = set(BUDGET_LABELS) - set(self.budget_tiers)
        missing_actions = {Action.SHORT.value, Action.MEDIUM.value, Action.FULL.value} - set(
            self.action_token_caps
        )
        if missing_tiers:
            raise ValueError(f"Missing budget tiers: {sorted(missing_tiers)}")
        if missing_actions:
            raise ValueError(f"Missing action caps: {sorted(missing_actions)}")
        if self.max_collaboration_steps < 1:
            raise ValueError("max_collaboration_steps must be positive")
        caps = [self.action_token_caps[a] for a in ("SHORT", "MEDIUM", "FULL")]
        if caps != sorted(caps) or caps[0] <= 0:
            raise ValueError("Action caps must be positive and ordered SHORT <= MEDIUM <= FULL")
        if self.reference_action not in ACTION_LABELS:
            raise ValueError(f"Unknown reference action: {self.reference_action}")
        if set(self.pilot_generation_caps) != {"solver", "critic", "refiner"}:
            raise ValueError("pilot_generation_caps must define solver, critic, and refiner")
        if any(value <= 0 for value in self.pilot_generation_caps.values()):
            raise ValueError("Pilot generation caps must be positive")
        if not isinstance(self.pilot_request_extra_body, dict):
            raise ValueError("pilot_request_extra_body must be an object")

    def tier(self, name: str | BudgetTier) -> BudgetLimit:
        key = name.value if isinstance(name, BudgetTier) else name
        return self.budget_tiers[key]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        payload = dict(payload)
        if "budget_tiers" in payload:
            payload["budget_tiers"] = {
                name: value if isinstance(value, BudgetLimit) else BudgetLimit(**value)
                for name, value in payload["budget_tiers"].items()
            }
        if "collection_limit" in payload and not isinstance(payload["collection_limit"], BudgetLimit):
            payload["collection_limit"] = BudgetLimit(**payload["collection_limit"])
        return cls(**payload)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
