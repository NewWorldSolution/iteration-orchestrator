"""Config loading: core defaults deep-merged with project.yaml.

Project config overrides core. Agents and costs live only in project.yaml
for v1 (built-in defaults ship in a later version).
"""
from __future__ import annotations

import copy
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from orch.hooks import BLOCKING_HOOK_EVENTS

TASK_SCHEMA_POLICY_KEYS = (
    "forbidden_allowed_prefixes",
    "planning_allowed_prefixes",
    "planning_refusal_prefixes",
)

TASK_SCHEMA_POLICY_FLOOR: dict[str, tuple[str, ...]] = {
    "forbidden_allowed_prefixes": (
        ".git/",
        ".github/",
        "deploy/",
    ),
    "planning_allowed_prefixes": (
        "docs/",
        "iterations/",
    ),
    "planning_refusal_prefixes": (
        ".git/",
        ".github/",
        "deploy/",
    ),
}


CORE_DEFAULTS: dict[str, Any] = {
    "limits": {
        "impl_attempts": 3,
        "fix_rounds_acceptance": 3,
        "review_rounds": 2,
        "scope_auto_revert": 1,
        "max_diff_insertions_hard": 1500,
    },
    "auto_merge": {
        "max_fix_rounds_default": 1,
        "max_fix_rounds_high_risk": 0,
        "max_diff_insertions": 500,
        "ci_wait_seconds": 300,
        "no_ci": False,
    },
    "preflight": {
        "low_files": 2,    "low_lines": 80,
        "medium_files": 5, "medium_lines": 150,
        "high_files": 8,   "high_lines": 250,
        "refuse_files": 12, "refuse_lines": 400,
        "warn_allowed_files": 6, "warn_prompt_lines": 150,
    },
    "timeouts": {
        # Act as hang safety nets, not performance ceilings
        "impl_low": 900,   "impl_medium": 1800, "impl_high": 2700,
        "fix_low": 600,    "fix_medium": 900,   "fix_high": 1200,
        "review": 900,
        "qa": 900,
        "retro": 900,
        "acceptance": 900,
        "ci": 300,
        "task_kind_profiles": {},
    },
    "independence": {
        # session | model | model_family
        "level": "model_family",
    },
    "review": {
        # Strict regex; malformed verdict = STOP(REVIEW_MALFORMED)
        "verdict_regex": r"^Verdict:\s+(PASS|CHANGES REQUIRED|BLOCKED)\s*$",
        # Optional independent reviewer for dual-model agreement gates.
        "secondary_reviewer": "",
    },
    "tasks_md": {
        "status_values": [
            "WAITING", "IN_PROGRESS", "DONE", "BLOCKED", "NEEDS_HUMAN_MERGE",
        ],
    },
    "tasks_schema": {
        "forbidden_allowed_prefixes": [],
        "planning_allowed_prefixes": [],
        "planning_refusal_prefixes": [],
    },
    "paths": {
        "iteration_root": "iterations",
        "artifact_root": "tools/logs",
        "worktree_root": ".orch/worktrees",
        "project_config": ".orch/project.yaml",
        "task_board_filename": "tasks.md",
        "iteration_prompt_filename": "prompt.md",
        "task_prompts_dir": "prompts",
        "task_reviews_dir": "reviews",
        "prompt_factory_log_dirname": "prompt-factory",
        "generated_artifact_exclusion_prefixes": ["tools/logs/"],
    },
    "templates": {
        "prompt_rules_path": "templates/_prompt_rules.md",
        "review_template_path": "templates/_review_template.md",
    },
    "invariants": [],
    "scaffold": {
        "post_phase_docs_root": "docs/post-phase",
        "post_phase_iteration_root": "iterations/post-phase",
        "tooling_iteration_root": "iterations/tools",
        "post_phase_integration_branch": "post-phase-integration",
    },
    "ui_route_visibility": {
        "route_globs": [],
        "nav_anchor_paths": [],
    },
    "parallel": {
        "max_concurrency": 1,
    },
    "model_routing": {
        "agent_overrides": {},
    },
    "patterns": {
        "task_id": r"^TASK-(\d+)-(\d+)$",
        "task_detail_heading": (
            r"^###\s+(?P<id>TASK-\d+-\d+)\s+—\s+(?P<title>.+?)\s*$"
        ),
        "iteration_id": r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$",
        "phase_branch": r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$",
    },
    "hooks": {
        "handlers": [],
    },
}

AGENT_ROUTING_SUGAR_KEYS = {"model_flag", "tier_models", "effort_flags"}


PROJECT_SECTION_KEYS: dict[str, set[str] | None] = {
    "project": {
        "name",
        "main_branch",
        "phase_branch_pattern",
        "iteration_branch_pattern",
        "task_branch_pattern",
    },
    "stack": {
        "install",
        "test",
        "lint",
        "typecheck",
        "build",
        "test_env",
        "acceptance_timeout_seconds",
    },
    "risk": {
        "high_risk_globs",
        "sensitive_files",
        "forbidden_patterns",
    },
    "agents": None,
    "costs": None,
    "limits": set(CORE_DEFAULTS["limits"]),
    "auto_merge": set(CORE_DEFAULTS["auto_merge"]),
    "preflight": set(CORE_DEFAULTS["preflight"]),
    "timeouts": set(CORE_DEFAULTS["timeouts"]),
    "independence": set(CORE_DEFAULTS["independence"]),
    "review": set(CORE_DEFAULTS["review"]),
    "tasks_md": set(CORE_DEFAULTS["tasks_md"]),
    "tasks_schema": set(CORE_DEFAULTS["tasks_schema"]),
    "paths": set(CORE_DEFAULTS["paths"]),
    "templates": set(CORE_DEFAULTS["templates"]),
    "invariants": None,
    "scaffold": set(CORE_DEFAULTS["scaffold"]),
    "ui_route_visibility": set(CORE_DEFAULTS["ui_route_visibility"]),
    "parallel": set(CORE_DEFAULTS["parallel"]),
    "model_routing": set(CORE_DEFAULTS["model_routing"]),
    "patterns": set(CORE_DEFAULTS["patterns"]),
    "hooks": set(CORE_DEFAULTS["hooks"]),
}


class ConfigError(Exception):
    """Raised when project.yaml is missing, malformed, or invalid."""


@dataclass(frozen=True)
class LoadedConfig:
    """Frozen view of merged config plus the source path for error messages."""
    path: Path
    data: dict

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def agents(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["agents"]


def costs(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["costs"]


def limits(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["limits"]


def auto_merge(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["auto_merge"]


def timeouts(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["timeouts"]


def independence(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["independence"]


def risk(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["risk"]


def stack(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["stack"]


def review(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["review"]


def tasks_md(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["tasks_md"]


def task_schema_policy(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["tasks_schema"]


def paths(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["paths"]


def templates(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["templates"]


def invariants(cfg: LoadedConfig) -> list[dict[str, str]]:
    return cfg.data["invariants"]


def scaffold(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["scaffold"]


def ui_route_visibility(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["ui_route_visibility"]


def parallel(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["parallel"]


def model_routing(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["model_routing"]


def patterns(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["patterns"]


def hooks(cfg: LoadedConfig) -> dict[str, Any]:
    return cfg.data["hooks"]


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_project_yaml(path: Path) -> dict:
    if not path.exists():
        raise ConfigError(f"project config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level must be a mapping")
    _warn_unknown_project_keys(data, path)
    _validate_project(data, path)
    return data


def _warn_unknown_project_keys(data: dict, path: Path) -> None:
    for section in sorted(data):
        if section not in PROJECT_SECTION_KEYS:
            warnings.warn(
                f"{path}: unknown top-level config section '{section}'",
                UserWarning,
                stacklevel=2,
            )
            continue

        allowed_keys = PROJECT_SECTION_KEYS[section]
        if allowed_keys is None or not isinstance(data[section], dict):
            continue
        for key in sorted(data[section]):
            if key not in allowed_keys:
                warnings.warn(
                    f"{path}: unknown config key '{section}.{key}'",
                    UserWarning,
                    stacklevel=2,
                )


def _require(data: dict, section: str, keys: list[str] | None, path: Path) -> None:
    if section not in data:
        raise ConfigError(f"{path}: missing required section '{section}'")
    if not isinstance(data[section], dict):
        raise ConfigError(f"{path}: section '{section}' must be a mapping")
    if keys:
        for k in keys:
            if k not in data[section]:
                raise ConfigError(f"{path}: missing required key '{section}.{k}'")


def _validate_project(data: dict, path: Path) -> None:
    _require(data, "project",
             ["name", "main_branch", "phase_branch_pattern",
              "iteration_branch_pattern", "task_branch_pattern"], path)
    _require(data, "stack", ["test", "lint"], path)
    _require(data, "risk",
             ["high_risk_globs", "sensitive_files", "forbidden_patterns"], path)
    _require(data, "agents", None, path)
    _require(data, "costs", None, path)

    for field in ("high_risk_globs", "sensitive_files", "forbidden_patterns"):
        if not isinstance(data["risk"][field], list):
            raise ConfigError(f"{path}: risk.{field} must be a list")

    if not data["agents"]:
        raise ConfigError(f"{path}: 'agents' must be a non-empty mapping")
    for name, spec in data["agents"].items():
        if not isinstance(spec, dict):
            raise ConfigError(f"{path}: agent '{name}' must be a mapping")
        for req in ("cmd", "family"):
            if req not in spec:
                raise ConfigError(f"{path}: agent '{name}' missing '{req}'")

    if not isinstance(data["costs"], dict):
        raise ConfigError(f"{path}: 'costs' must be a mapping of family -> rates")
    for fam, rates in data["costs"].items():
        if (not isinstance(rates, dict)
                or "input" not in rates or "output" not in rates):
            raise ConfigError(
                f"{path}: costs['{fam}'] must have 'input' and 'output' keys")

    known_families = set(data["costs"])
    for name, spec in data["agents"].items():
        family = spec["family"]
        if not isinstance(family, str) or not family.strip():
            raise ConfigError(
                f"{path}: agent '{name}' family must be a non-empty string"
            )
        if family not in known_families:
            known = ", ".join(sorted(str(fam) for fam in known_families))
            raise ConfigError(
                f"{path}: agent '{name}' family '{family}' has no matching "
                f"costs entry; known families: {known}"
            )

    if "hooks" in data:
        _validate_hooks(data["hooks"], path)
    if "parallel" in data:
        _validate_parallel(data["parallel"], path)
    if "model_routing" in data:
        _validate_model_routing(
            data["model_routing"],
            path,
            known_agents=set(data["agents"]),
        )
    if "timeouts" in data:
        _validate_timeouts(data["timeouts"], path)
    if "tasks_schema" in data:
        _validate_task_schema_policy(data["tasks_schema"], path)
    if "invariants" in data:
        _validate_invariants(data["invariants"], path)
    if "patterns" in data:
        _validate_patterns(data["patterns"], path)


def _validate_invariants(invariants_section: Any, path: Path) -> None:
    if not isinstance(invariants_section, list):
        raise ConfigError(f"{path}: section 'invariants' must be a list")
    allowed_keys = {"name", "applies", "evidence", "status"}
    for idx, item in enumerate(invariants_section):
        label = f"invariants[{idx}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: {label} must be a mapping")
        unknown = sorted(set(item) - allowed_keys)
        if unknown:
            raise ConfigError(
                f"{path}: {label} has unknown key(s): {', '.join(unknown)}"
            )
        if "name" not in item:
            raise ConfigError(f"{path}: {label}.name is required")
        for key in allowed_keys:
            if key not in item:
                continue
            value = item[key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(
                    f"{path}: {label}.{key} must be a non-empty string"
                )
            if "\n" in value or "|" in value:
                raise ConfigError(
                    f"{path}: {label}.{key} must not contain newlines or '|'"
                )


def _validate_patterns(patterns_section: Any, path: Path) -> None:
    if not isinstance(patterns_section, dict):
        raise ConfigError(f"{path}: section 'patterns' must be a mapping")
    for key, value in patterns_section.items():
        label = f"patterns.{key}"
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{path}: {label} must be a non-empty regex string")
        try:
            re.compile(value)
        except re.error as exc:
            raise ConfigError(f"{path}: {label} has invalid regex: {exc}") from None


def effective_task_schema_policy(
    policy_section: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Return the non-droppable floor unioned with project extensions."""
    configured = policy_section or {}
    effective: dict[str, list[str]] = {}
    for key in TASK_SCHEMA_POLICY_KEYS:
        values = configured.get(key, [])
        effective[key] = _ordered_prefix_union(
            TASK_SCHEMA_POLICY_FLOOR[key],
            values if isinstance(values, list) else [],
        )
    return effective


def _ordered_prefix_union(
    floor: tuple[str, ...],
    extensions: list[str],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for prefix in (*floor, *tuple(extensions)):
        if prefix in seen:
            continue
        seen.add(prefix)
        out.append(prefix)
    return out


def _validate_task_schema_policy(policy_section: Any, path: Path) -> None:
    if not isinstance(policy_section, dict):
        raise ConfigError(f"{path}: section 'tasks_schema' must be a mapping")
    for key in TASK_SCHEMA_POLICY_KEYS:
        if key in policy_section:
            _validate_prefix_list(
                policy_section[key],
                path,
                f"tasks_schema.{key}",
            )


def _validate_prefix_list(value: Any, path: Path, label: str) -> None:
    if not isinstance(value, list):
        raise ConfigError(f"{path}: {label} must be a list")
    for idx, prefix in enumerate(value):
        item_label = f"{label}[{idx}]"
        if not isinstance(prefix, str) or not prefix.strip():
            raise ConfigError(f"{path}: {item_label} must be a non-empty string")
        raw = prefix.strip()
        if prefix != raw:
            raise ConfigError(
                f"{path}: {item_label} must not have surrounding whitespace"
            )
        prefix_path = Path(raw)
        if prefix_path.is_absolute() or any(part == ".." for part in prefix_path.parts):
            raise ConfigError(
                f"{path}: {item_label} must be repo-relative and must not "
                "contain '..'"
            )
        if any(ch in raw for ch in "*?[]"):
            raise ConfigError(f"{path}: {item_label} must not contain glob chars")
        if not raw.endswith("/"):
            raise ConfigError(f"{path}: {item_label} must end with '/'")


def _validate_timeouts(timeouts_section: Any, path: Path) -> None:
    if not isinstance(timeouts_section, dict):
        raise ConfigError(f"{path}: section 'timeouts' must be a mapping")
    profiles = timeouts_section.get("task_kind_profiles", {})
    if profiles is None:
        return
    if not isinstance(profiles, dict):
        raise ConfigError(
            f"{path}: timeouts.task_kind_profiles must be a mapping"
        )
    allowed_timeout_keys = {"impl", "fix", "review", "acceptance", "ci"}
    for kind, profile in profiles.items():
        label = f"timeouts.task_kind_profiles.{kind}"
        if not isinstance(kind, str) or not kind.strip():
            raise ConfigError(
                f"{path}: timeouts.task_kind_profiles keys must be "
                "non-empty strings"
            )
        if not isinstance(profile, dict):
            raise ConfigError(f"{path}: {label} must be a mapping")
        for key, value in profile.items():
            if key not in allowed_timeout_keys:
                allowed = ", ".join(sorted(allowed_timeout_keys))
                raise ConfigError(
                    f"{path}: {label}.{key} is not supported; allowed: "
                    f"{allowed}"
                )
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(
                    f"{path}: {label}.{key} must be a positive integer"
                )


def _validate_model_routing(
    model_routing_section: Any,
    path: Path,
    *,
    known_agents: set[str],
) -> None:
    from orch.model_routing import MODEL_TIERS, REASONING_EFFORTS

    if not isinstance(model_routing_section, dict):
        raise ConfigError(f"{path}: section 'model_routing' must be a mapping")
    agent_overrides = model_routing_section.get("agent_overrides", {})
    if agent_overrides is None:
        return
    if not isinstance(agent_overrides, dict):
        raise ConfigError(
            f"{path}: model_routing.agent_overrides must be a mapping"
        )
    for agent, tier_map in agent_overrides.items():
        agent_label = f"model_routing.agent_overrides.{agent}"
        if not isinstance(agent, str) or not agent.strip():
            raise ConfigError(
                f"{path}: model_routing.agent_overrides keys must be "
                "non-empty strings"
            )
        if agent not in known_agents:
            raise ConfigError(f"{path}: {agent_label} references unknown agent")
        if not isinstance(tier_map, dict):
            raise ConfigError(f"{path}: {agent_label} must be a mapping")
        _validate_agent_routing_sugar(tier_map, path, agent_label)
        for tier, effort_map in tier_map.items():
            if tier in AGENT_ROUTING_SUGAR_KEYS:
                continue
            tier_label = f"{agent_label}.{tier}"
            if tier not in MODEL_TIERS:
                allowed = ", ".join(MODEL_TIERS)
                raise ConfigError(
                    f"{path}: {tier_label} has invalid tier; allowed: "
                    f"{allowed}"
                )
            if not isinstance(effort_map, dict):
                raise ConfigError(f"{path}: {tier_label} must be a mapping")
            for effort, override in effort_map.items():
                override_label = f"{tier_label}.{effort}"
                if effort not in REASONING_EFFORTS:
                    allowed = ", ".join(REASONING_EFFORTS)
                    raise ConfigError(
                        f"{path}: {override_label} has invalid effort; "
                        f"allowed: {allowed}"
                )
                _validate_agent_override(override, path, override_label)


def _validate_agent_routing_sugar(
    agent_map: Any,
    path: Path,
    label: str,
) -> None:
    from orch.model_routing import MODEL_TIERS, REASONING_EFFORTS

    has_model_flag = "model_flag" in agent_map
    has_tier_models = "tier_models" in agent_map
    if has_model_flag:
        model_flag = agent_map["model_flag"]
        if not isinstance(model_flag, str) or not model_flag:
            raise ConfigError(
                f"{path}: {label}.model_flag must be a non-empty string"
            )
    if has_model_flag and not has_tier_models:
        raise ConfigError(
            f"{path}: {label}.model_flag requires tier_models"
        )
    if has_tier_models:
        if not has_model_flag:
            raise ConfigError(
                f"{path}: {label}.tier_models requires model_flag"
            )
        _validate_tier_models(
            agent_map["tier_models"],
            path,
            f"{label}.tier_models",
            MODEL_TIERS,
        )

    if "effort_flags" in agent_map:
        _validate_effort_flags(
            agent_map["effort_flags"],
            path,
            f"{label}.effort_flags",
            REASONING_EFFORTS,
        )


def _validate_tier_models(
    tier_models: Any,
    path: Path,
    label: str,
    model_tiers: tuple[str, ...],
) -> None:
    if not isinstance(tier_models, dict):
        raise ConfigError(f"{path}: {label} must be a mapping")
    missing = [tier for tier in model_tiers if tier not in tier_models]
    if missing:
        raise ConfigError(
            f"{path}: {label} must define every tier; missing: "
            f"{', '.join(missing)}"
        )
    for tier, model in tier_models.items():
        tier_label = f"{label}.{tier}"
        if tier not in model_tiers:
            allowed = ", ".join(model_tiers)
            raise ConfigError(
                f"{path}: {tier_label} has invalid tier; allowed: {allowed}"
            )
        if not isinstance(model, str) or not model:
            raise ConfigError(
                f"{path}: {tier_label} must be a non-empty string"
            )


def _validate_effort_flags(
    effort_flags: Any,
    path: Path,
    label: str,
    reasoning_efforts: tuple[str, ...],
) -> None:
    if not isinstance(effort_flags, dict):
        raise ConfigError(f"{path}: {label} must be a mapping")
    missing = [
        effort for effort in reasoning_efforts if effort not in effort_flags
    ]
    if missing:
        raise ConfigError(
            f"{path}: {label} must define every effort; missing: "
            f"{', '.join(missing)}"
        )
    for effort, override in effort_flags.items():
        override_label = f"{label}.{effort}"
        if effort not in reasoning_efforts:
            allowed = ", ".join(reasoning_efforts)
            raise ConfigError(
                f"{path}: {override_label} has invalid effort; "
                f"allowed: {allowed}"
            )
        _validate_agent_override(override, path, override_label)


def _validate_agent_override(override: Any, path: Path, label: str) -> None:
    if not isinstance(override, dict):
        raise ConfigError(f"{path}: {label} must be a mapping")
    unknown = sorted(set(override) - {"args", "env"})
    if unknown:
        raise ConfigError(
            f"{path}: {label} has unknown key(s): {', '.join(unknown)}"
        )
    args = override.get("args", [])
    if args is None:
        args = []
    if not isinstance(args, list) or any(
        not isinstance(arg, str) or arg == "" for arg in args
    ):
        raise ConfigError(f"{path}: {label}.args must be a list of strings")
    env = override.get("env", {})
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise ConfigError(f"{path}: {label}.env must be a mapping")
    for key, value in env.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(
                f"{path}: {label}.env keys must be non-empty strings"
            )
        if not isinstance(value, str):
            raise ConfigError(f"{path}: {label}.env values must be strings")


def _validate_parallel(parallel_section: Any, path: Path) -> None:
    if not isinstance(parallel_section, dict):
        raise ConfigError(f"{path}: section 'parallel' must be a mapping")
    if "max_concurrency" not in parallel_section:
        return
    value = parallel_section["max_concurrency"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(
            f"{path}: parallel.max_concurrency must be a positive integer"
        )


def _validate_hooks(hooks_section: Any, path: Path) -> None:
    if not isinstance(hooks_section, dict):
        raise ConfigError(f"{path}: section 'hooks' must be a mapping")
    handlers = hooks_section.get("handlers", [])
    if handlers is None:
        handlers = []
    if not isinstance(handlers, list):
        raise ConfigError(f"{path}: hooks.handlers must be a list")
    for idx, spec in enumerate(handlers):
        label = f"hooks.handlers[{idx}]"
        if not isinstance(spec, dict):
            raise ConfigError(f"{path}: {label} must be a mapping")
        for key in ("name", "events", "cmd"):
            if key not in spec:
                raise ConfigError(f"{path}: {label} missing '{key}'")
        if spec.get("type", "command") != "command":
            raise ConfigError(f"{path}: {label}.type must be 'command'")
        events = spec["events"]
        if not isinstance(events, list) or not events:
            raise ConfigError(f"{path}: {label}.events must be a non-empty list")
        for event in events:
            if not isinstance(event, str) or not event:
                raise ConfigError(f"{path}: {label}.events entries must be strings")
            if spec.get("required", False) and event not in BLOCKING_HOOK_EVENTS:
                raise ConfigError(
                    f"{path}: {label} sets required=true for non-blocking "
                    f"event '{event}'"
                )
        if not isinstance(spec["cmd"], str) or not spec["cmd"].strip():
            raise ConfigError(f"{path}: {label}.cmd must be a non-empty string")
        if "timeout" in spec:
            try:
                timeout = int(spec["timeout"])
            except (TypeError, ValueError):
                raise ConfigError(
                    f"{path}: {label}.timeout must be an integer"
                ) from None
            if timeout <= 0:
                raise ConfigError(f"{path}: {label}.timeout must be positive")


def load_config(project_yaml_path: Path) -> LoadedConfig:
    proj = load_project_yaml(project_yaml_path)
    merged = _deep_merge(CORE_DEFAULTS, proj)
    merged["tasks_schema"] = effective_task_schema_policy(merged.get("tasks_schema"))
    return LoadedConfig(path=project_yaml_path, data=merged)


def default_project_yaml_path(repo_root: Path) -> Path:
    return repo_root / Path(CORE_DEFAULTS["paths"]["project_config"])
