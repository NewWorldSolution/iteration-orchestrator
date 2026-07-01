"""Deterministic model routing metadata for orchestrator tasks."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from orch.agents.base import AgentInvocationOptions

MODEL_TIERS = ("standard", "strong", "max")
REASONING_EFFORTS = ("low", "medium", "high", "max")
RISK_CATEGORIES = (
    "architecture_core_logic",
    "merge_critical_gate",
    "security_compliance",
    "schema_data_structure",
    "coordinator_state_hook_runtime",
    "financial_logic_client_deliverable",
    "routing_config",
    "unknown",
)

DEFAULT_MODEL_TIER = "strong"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_RISK_CATEGORY = "unknown"
ROUTING_WARNINGS_FILENAME = "model_routing_warnings.jsonl"
DUAL_MODEL_REQUIRED_CATEGORIES = (
    "architecture_core_logic",
    "merge_critical_gate",
)

_TIER_RANK = {value: idx for idx, value in enumerate(MODEL_TIERS)}
_EFFORT_RANK = {value: idx for idx, value in enumerate(REASONING_EFFORTS)}
_RISK_FLOORS = {
    "architecture_core_logic": ("max", "max"),
    "merge_critical_gate": ("max", "max"),
    "security_compliance": ("max", "high"),
    "schema_data_structure": ("max", "high"),
    "coordinator_state_hook_runtime": ("max", "max"),
    "financial_logic_client_deliverable": ("max", "max"),
    "routing_config": ("max", "max"),
    "unknown": ("max", "max"),
}


class ModelRoutingError(ValueError):
    """Raised when model routing metadata is malformed or unsupported."""


@dataclass(frozen=True)
class ModelRoutingDeclaration:
    model_tier: str | None = None
    reasoning_effort: str | None = None
    risk_category: str = DEFAULT_RISK_CATEGORY
    source_line: int = 0


@dataclass(frozen=True)
class ResolvedModelRouting:
    model_tier: str
    reasoning_effort: str
    risk_category: str
    declared_model_tier: str
    declared_reasoning_effort: str
    floor_model_tier: str
    floor_reasoning_effort: str
    unknown_risk: bool = False
    dual_model_required: bool = False
    source_line: int = 0


def validate_model_tier(value: str) -> str:
    if value not in _TIER_RANK:
        allowed = ", ".join(MODEL_TIERS)
        raise ModelRoutingError(
            f"invalid model_tier {value!r}; allowed: {allowed}"
        )
    return value


def validate_reasoning_effort(value: str) -> str:
    if value not in _EFFORT_RANK:
        allowed = ", ".join(REASONING_EFFORTS)
        raise ModelRoutingError(
            f"invalid reasoning_effort {value!r}; allowed: {allowed}"
        )
    return value


def validate_risk_category(value: str) -> str:
    if value not in RISK_CATEGORIES:
        allowed = ", ".join(RISK_CATEGORIES)
        raise ModelRoutingError(
            f"invalid risk_category {value!r}; allowed: {allowed}"
        )
    return value


def parse_model_routing_field(
    raw: str,
    *,
    source_line: int = 0,
) -> ModelRoutingDeclaration:
    """Parse ``key=value; ...`` metadata from a task detail line."""
    fields: dict[str, str] = {}
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ModelRoutingError(
                f"model routing field {part!r} must use key=value"
            )
        key, value = [piece.strip() for piece in part.split("=", 1)]
        if key not in {"model_tier", "reasoning_effort", "risk_category"}:
            raise ModelRoutingError(f"unknown model routing field {key!r}")
        if key in fields:
            raise ModelRoutingError(f"duplicate model routing field {key!r}")
        if not value:
            raise ModelRoutingError(f"model routing field {key!r} is empty")
        fields[key] = value

    model_tier = fields.get("model_tier")
    reasoning_effort = fields.get("reasoning_effort")
    risk_category = fields.get("risk_category", DEFAULT_RISK_CATEGORY)
    if model_tier is not None:
        validate_model_tier(model_tier)
    if reasoning_effort is not None:
        validate_reasoning_effort(reasoning_effort)
    validate_risk_category(risk_category)
    return ModelRoutingDeclaration(
        model_tier=model_tier,
        reasoning_effort=reasoning_effort,
        risk_category=risk_category,
        source_line=source_line,
    )


def _declaration_from_mapping(
    raw: Mapping[str, Any] | None,
) -> ModelRoutingDeclaration:
    if raw is None:
        return ModelRoutingDeclaration()
    if isinstance(raw, ModelRoutingDeclaration):
        return raw
    return ModelRoutingDeclaration(
        model_tier=raw.get("model_tier"),
        reasoning_effort=raw.get("reasoning_effort"),
        risk_category=raw.get("risk_category", DEFAULT_RISK_CATEGORY),
        source_line=int(raw.get("source_line", 0) or 0),
    )


def _max_model_tier(left: str, right: str) -> str:
    return left if _TIER_RANK[left] >= _TIER_RANK[right] else right


def _max_reasoning_effort(left: str, right: str) -> str:
    return left if _EFFORT_RANK[left] >= _EFFORT_RANK[right] else right


def requires_dual_model_agreement(
    routing_or_category: ResolvedModelRouting | str,
) -> bool:
    category = (
        routing_or_category.risk_category
        if isinstance(routing_or_category, ResolvedModelRouting)
        else routing_or_category
    )
    validate_risk_category(category)
    return category in DUAL_MODEL_REQUIRED_CATEGORIES


def resolve_model_routing(
    declaration: ModelRoutingDeclaration | Mapping[str, Any] | None = None,
) -> ResolvedModelRouting:
    declared = _declaration_from_mapping(declaration)
    declared_tier = validate_model_tier(
        declared.model_tier or DEFAULT_MODEL_TIER
    )
    declared_effort = validate_reasoning_effort(
        declared.reasoning_effort or DEFAULT_REASONING_EFFORT
    )
    risk_category = validate_risk_category(declared.risk_category)
    floor_tier, floor_effort = _RISK_FLOORS[risk_category]
    return ResolvedModelRouting(
        model_tier=_max_model_tier(declared_tier, floor_tier),
        reasoning_effort=_max_reasoning_effort(declared_effort, floor_effort),
        risk_category=risk_category,
        declared_model_tier=declared_tier,
        declared_reasoning_effort=declared_effort,
        floor_model_tier=floor_tier,
        floor_reasoning_effort=floor_effort,
        unknown_risk=(risk_category == DEFAULT_RISK_CATEGORY),
        dual_model_required=requires_dual_model_agreement(risk_category),
        source_line=declared.source_line,
    )


def routing_to_dict(routing: ResolvedModelRouting) -> dict[str, Any]:
    return asdict(routing)


def resolve_agent_routing_options(
    routing_config: Mapping[str, Any] | None,
    *,
    agent_name: str,
    routing: ResolvedModelRouting,
) -> AgentInvocationOptions:
    """Resolve provider args/env for an agent from resolved routing metadata."""
    agent_overrides = (routing_config or {}).get("agent_overrides") or {}
    agent_map = agent_overrides.get(agent_name) or {}
    raw_override = _raw_agent_override_for_cell(agent_map, routing)
    if raw_override is not None:
        return _agent_options_from_override(raw_override)
    return _agent_options_from_sugar(agent_map, routing)


def resolve_quality_gate_routing_options(
    routing_config: Mapping[str, Any] | None,
    *,
    agent_name: str,
) -> AgentInvocationOptions:
    """Resolve non-task QA/retro routing through the standard resolver."""
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="max",
            reasoning_effort="max",
            risk_category="coordinator_state_hook_runtime",
        )
    )
    return resolve_agent_routing_options(
        routing_config,
        agent_name=agent_name,
        routing=routing,
    )


def _raw_agent_override_for_cell(
    agent_map: Mapping[str, Any],
    routing: ResolvedModelRouting,
) -> Mapping[str, Any] | None:
    tier_map = agent_map.get(routing.model_tier) or {}
    if not isinstance(tier_map, Mapping):
        return None
    if routing.reasoning_effort not in tier_map:
        return None
    override = tier_map.get(routing.reasoning_effort) or {}
    if not isinstance(override, Mapping):
        return None
    return override


def _agent_options_from_override(
    override: Mapping[str, Any],
) -> AgentInvocationOptions:
    args = tuple(override.get("args") or ())
    env = dict(override.get("env") or {})
    return AgentInvocationOptions(args=args, env=env)


def _agent_options_from_sugar(
    agent_map: Mapping[str, Any],
    routing: ResolvedModelRouting,
) -> AgentInvocationOptions:
    args: list[str] = []
    env: dict[str, str] = {}

    model_flag = agent_map.get("model_flag")
    tier_models = agent_map.get("tier_models") or {}
    if (
        isinstance(model_flag, str)
        and model_flag
        and isinstance(tier_models, Mapping)
    ):
        model = tier_models.get(routing.model_tier)
        if isinstance(model, str) and model:
            args.extend([model_flag, model])

    effort_flags = agent_map.get("effort_flags") or {}
    if isinstance(effort_flags, Mapping):
        effort_override = effort_flags.get(routing.reasoning_effort) or {}
        if isinstance(effort_override, Mapping):
            args.extend(effort_override.get("args") or ())
            env.update(effort_override.get("env") or {})

    return AgentInvocationOptions(args=tuple(args), env=env)


def routing_warnings_path(log_dir: Path) -> Path:
    return log_dir / ROUTING_WARNINGS_FILENAME


def append_unknown_risk_warning(
    log_dir: Path,
    *,
    iteration: str,
    task_id: str,
    routing: ResolvedModelRouting,
) -> Path:
    """Append deterministic warning evidence for an unclassified task risk."""
    path = routing_warnings_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "iteration": iteration,
        "task_id": task_id,
        "risk_category": routing.risk_category,
        "model_tier": routing.model_tier,
        "reasoning_effort": routing.reasoning_effort,
        "warning": (
            "risk_category is unknown; classify the task before relying on "
            "risk-floor routing for high-blast-radius work"
        ),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=False) + "\n")
    return path
