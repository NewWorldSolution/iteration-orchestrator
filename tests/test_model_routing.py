"""Tests for deterministic model routing."""
from __future__ import annotations

import json

import pytest

from orch.model_routing import (
    MODEL_TIERS,
    REASONING_EFFORTS,
    ModelRoutingDeclaration,
    ModelRoutingError,
    append_unknown_risk_warning,
    requires_dual_model_agreement,
    resolve_agent_routing_options,
    resolve_model_routing,
    routing_warnings_path,
    validate_model_tier,
    validate_reasoning_effort,
)


def test_default_unknown_routing_resolves_to_max_max_without_dual():
    routing = resolve_model_routing()

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "max"
    assert routing.risk_category == "unknown"
    assert routing.dual_model_required is False


@pytest.mark.parametrize("tier", MODEL_TIERS)
def test_each_model_tier_value_validates(tier: str):
    assert validate_model_tier(tier) == tier


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_each_reasoning_effort_value_validates(effort: str):
    assert validate_reasoning_effort(effort) == effort


def test_invalid_model_tier_fails():
    with pytest.raises(ModelRoutingError, match="invalid model_tier"):
        validate_model_tier("tiny")


def test_invalid_reasoning_effort_fails():
    with pytest.raises(ModelRoutingError, match="invalid reasoning_effort"):
        validate_reasoning_effort("extreme")


@pytest.mark.parametrize(
    "risk_category",
    [
        "architecture_core_logic",
        "merge_critical_gate",
        "coordinator_state_hook_runtime",
        "financial_logic_client_deliverable",
        "routing_config",
    ],
)
def test_max_floor_categories_force_max_max(risk_category: str):
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category=risk_category,
        )
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "max"


@pytest.mark.parametrize(
    "risk_category",
    ["security_compliance", "schema_data_structure"],
)
def test_high_floor_categories_force_max_high(risk_category: str):
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category=risk_category,
        )
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "high"


def test_lower_author_declaration_is_raised_to_category_floor():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category="schema_data_structure",
        )
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "high"
    assert routing.declared_model_tier == "standard"
    assert routing.declared_reasoning_effort == "low"


def test_higher_author_declaration_is_preserved():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="max",
            reasoning_effort="max",
            risk_category="unknown",
        )
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "max"
    assert routing.floor_model_tier == "max"
    assert routing.floor_reasoning_effort == "max"


def test_unknown_category_writes_visible_warning_evidence(tmp_path):
    routing = resolve_model_routing(
        ModelRoutingDeclaration(risk_category="unknown")
    )

    path = append_unknown_risk_warning(
        tmp_path,
        iteration="demo-i1",
        task_id="I1-T1",
        routing=routing,
    )

    assert path == routing_warnings_path(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    assert payload == {
        "iteration": "demo-i1",
        "task_id": "I1-T1",
        "risk_category": "unknown",
        "model_tier": "max",
        "reasoning_effort": "max",
        "warning": (
            "risk_category is unknown; classify the task before relying on "
            "risk-floor routing for high-blast-radius work"
        ),
    }


def test_model_routing_unknown_risk_still_warns_and_uses_floor(tmp_path):
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category="unknown",
        )
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "max"
    assert routing.dual_model_required is False
    warning_path = append_unknown_risk_warning(
        tmp_path,
        iteration="demo-i1",
        task_id="I1-T1",
        routing=routing,
    )
    payload = json.loads(warning_path.read_text(encoding="utf-8"))
    assert payload["risk_category"] == "unknown"
    assert payload["model_tier"] == "max"
    assert payload["reasoning_effort"] == "max"


def test_model_routing_resolved_metadata_selects_configured_agent_args():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category="architecture_core_logic",
        )
    )
    routing_config = {
        "agent_overrides": {
            "codex": {
                "standard": {
                    "low": {
                        "args": ["--model", "weakened-fixture-model"],
                    },
                },
                "max": {
                    "max": {
                        "args": ["--model", "floor-fixture-model"],
                        "env": {"ORCH_REASONING_EFFORT": "max"},
                    },
                },
            },
        },
    }

    options = resolve_agent_routing_options(
        routing_config,
        agent_name="codex",
        routing=routing,
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "max"
    assert options.args == ("--model", "floor-fixture-model")
    assert options.env == {"ORCH_REASONING_EFFORT": "max"}


@pytest.mark.parametrize(
    "risk_category",
    ["architecture_core_logic", "merge_critical_gate"],
)
def test_required_categories_require_dual_model_agreement(risk_category: str):
    routing = resolve_model_routing(
        ModelRoutingDeclaration(risk_category=risk_category)
    )

    assert routing.dual_model_required is True
    assert requires_dual_model_agreement(routing) is True
    assert requires_dual_model_agreement(risk_category) is True


@pytest.mark.parametrize(
    "risk_category",
    [
        "security_compliance",
        "schema_data_structure",
        "coordinator_state_hook_runtime",
        "financial_logic_client_deliverable",
        "routing_config",
        "unknown",
    ],
)
def test_non_star_categories_do_not_require_dual_model_by_default(
    risk_category: str,
):
    routing = resolve_model_routing(
        ModelRoutingDeclaration(risk_category=risk_category)
    )

    assert routing.dual_model_required is False
    assert requires_dual_model_agreement(routing) is False


def test_agent_routing_options_empty_config_defaults_to_no_args_or_env():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category="routing_config",
        )
    )

    for routing_config in (None, {}, {"agent_overrides": {}}):
        options = resolve_agent_routing_options(
            routing_config,
            agent_name="codex",
            routing=routing,
        )

        assert options.args == ()
        assert options.env == {}


def test_codex_routing_sugar_generates_post_floor_model_effort_args_and_env():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category="schema_data_structure",
        )
    )
    effort_flags = {
        effort: {"args": ["--reasoning-effort", effort]}
        for effort in REASONING_EFFORTS
    }
    effort_flags["high"]["env"] = {"ORCH_REASONING_EFFORT": "high"}
    routing_config = {
        "agent_overrides": {
            "codex": {
                "model_flag": "--model",
                "tier_models": {
                    "standard": "codex-standard",
                    "strong": "codex-strong",
                    "max": "codex-max",
                },
                "effort_flags": effort_flags,
            },
        },
    }

    options = resolve_agent_routing_options(
        routing_config,
        agent_name="codex",
        routing=routing,
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "high"
    assert options.args == (
        "--model",
        "codex-max",
        "--reasoning-effort",
        "high",
    )
    assert options.env == {"ORCH_REASONING_EFFORT": "high"}


def test_claude_routing_sugar_generates_provider_style_args_and_env():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="max",
            reasoning_effort="max",
            risk_category="routing_config",
        )
    )
    effort_flags = {
        effort: {"args": ["--thinking-budget", str(index)]}
        for index, effort in enumerate(REASONING_EFFORTS, start=1)
    }
    effort_flags["max"] = {
        "args": ["--thinking-budget", "32000"],
        "env": {"CLAUDE_ROUTING_EFFORT": "max"},
    }
    routing_config = {
        "agent_overrides": {
            "claude": {
                "model_flag": "--model",
                "tier_models": {
                    "standard": "claude-haiku",
                    "strong": "claude-sonnet",
                    "max": "claude-opus",
                },
                "effort_flags": effort_flags,
            },
        },
    }

    options = resolve_agent_routing_options(
        routing_config,
        agent_name="claude",
        routing=routing,
    )

    assert options.args == (
        "--model",
        "claude-opus",
        "--thinking-budget",
        "32000",
    )
    assert options.env == {"CLAUDE_ROUTING_EFFORT": "max"}


def test_raw_cell_precedes_sugar_after_floor_raised_lookup():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category="architecture_core_logic",
        )
    )
    routing_config = {
        "agent_overrides": {
            "codex": {
                "model_flag": "--model",
                "tier_models": {
                    "standard": "sugar-standard-model",
                    "strong": "sugar-strong-model",
                    "max": "sugar-floor-model",
                },
                "effort_flags": {
                    effort: {"args": ["--reasoning-effort", effort]}
                    for effort in REASONING_EFFORTS
                },
                "standard": {
                    "low": {
                        "args": ["--model", "weakened-fixture-model"],
                    },
                },
                "max": {
                    "max": {
                        "args": ["--model", "floor-fixture-model"],
                        "env": {"ORCH_REASONING_EFFORT": "max"},
                    },
                },
            },
        },
    }

    options = resolve_agent_routing_options(
        routing_config,
        agent_name="codex",
        routing=routing,
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "max"
    assert options.args == ("--model", "floor-fixture-model")
    assert options.env == {"ORCH_REASONING_EFFORT": "max"}


def test_codex_project_mapping_uses_floored_effort_before_sugar():
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier="standard",
            reasoning_effort="low",
            risk_category="routing_config",
        )
    )
    routing_config = {
        "agent_overrides": {
            "codex": {
                "model_flag": "-m",
                "tier_models": {
                    "standard": "gpt-5.5",
                    "strong": "gpt-5.5",
                    "max": "gpt-5.5",
                },
                "effort_flags": {
                    "low": {"args": ["-c", "model_reasoning_effort=low"]},
                    "medium": {
                        "args": ["-c", "model_reasoning_effort=medium"]
                    },
                    "high": {"args": ["-c", "model_reasoning_effort=high"]},
                    "max": {"args": ["-c", "model_reasoning_effort=xhigh"]},
                },
            },
        },
    }

    options = resolve_agent_routing_options(
        routing_config,
        agent_name="codex",
        routing=routing,
    )

    assert routing.model_tier == "max"
    assert routing.reasoning_effort == "max"
    assert options.args == (
        "-m",
        "gpt-5.5",
        "-c",
        "model_reasoning_effort=xhigh",
    )
    assert options.env == {}


def test_quality_gate_routing_empty_config_uses_empty_options():
    from orch.model_routing import resolve_quality_gate_routing_options

    for routing_config in (None, {}, {"agent_overrides": {}}):
        options = resolve_quality_gate_routing_options(
            routing_config,
            agent_name="codex",
        )

        assert options.args == ()
        assert options.env == {}


def test_quality_gate_routing_uses_existing_resolver_max_profile():
    from orch.model_routing import resolve_quality_gate_routing_options

    routing_config = {
        "agent_overrides": {
            "codex": {
                "model_flag": "--model",
                "tier_models": {
                    "standard": "codex-standard",
                    "strong": "codex-strong",
                    "max": "codex-max",
                },
                "effort_flags": {
                    effort: {"args": ["--reasoning-effort", effort]}
                    for effort in REASONING_EFFORTS
                },
                "standard": {
                    "low": {"args": ["--model", "too-low"]},
                },
                "max": {
                    "max": {
                        "args": ["--model", "raw-quality-gate"],
                        "env": {"ORCH_REASONING_EFFORT": "max"},
                    },
                },
            },
        },
    }

    options = resolve_quality_gate_routing_options(
        routing_config,
        agent_name="codex",
    )

    assert options.args == ("--model", "raw-quality-gate")
    assert options.env == {"ORCH_REASONING_EFFORT": "max"}
