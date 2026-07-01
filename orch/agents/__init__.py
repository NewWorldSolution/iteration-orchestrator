"""Agent adapters — claude, codex, and generic shell.

Use :func:`build_adapter` to resolve a configured agent name from
``project.yaml`` into a concrete adapter instance. The factory looks at
``agents.<name>.type`` (``claude`` | ``codex`` | ``shell``) and passes the
rest of the spec through to the adapter.
"""
from __future__ import annotations

from orch.agents.base import (
    AgentAdapter,
    AgentInvocationOptions,
    AgentResult,
    apply_invocation_options,
    build_allowed_files_prefix,
    compose_prompt,
    run_with_timeout,
)
from orch.agents.claude import ClaudeAdapter
from orch.agents.codex import CodexAdapter
from orch.agents.shell import ShellAdapter

__all__ = [
    "AgentAdapter",
    "AgentInvocationOptions",
    "AgentResult",
    "ClaudeAdapter",
    "CodexAdapter",
    "ShellAdapter",
    "apply_invocation_options",
    "build_adapter",
    "build_allowed_files_prefix",
    "compose_prompt",
    "run_with_timeout",
]


def build_adapter(name: str, spec: dict) -> AgentAdapter:
    """Construct an adapter instance for an entry in ``agents:`` config.

    ``spec`` is the raw mapping from project.yaml. Required: ``cmd``,
    ``family``. ``type`` defaults to ``shell`` when omitted.
    """
    if "cmd" not in spec or "family" not in spec:
        raise ValueError(f"agent '{name}' missing 'cmd' or 'family' in config")
    kind = spec.get("type", "shell")
    cmd = spec["cmd"]
    family = spec["family"]
    if kind == "claude":
        return ClaudeAdapter(cmd=cmd, family=family)
    if kind == "codex":
        return CodexAdapter(cmd=cmd, family=family)
    if kind == "shell":
        return ShellAdapter(
            cmd=cmd,
            family=family,
            prompt_stdin=bool(spec.get("prompt_stdin", False)),
            cost_regex=spec.get("cost_regex"),
            agent_name=name,
        )
    raise ValueError(f"agent '{name}' has unknown type '{kind}'")
