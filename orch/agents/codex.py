"""Codex CLI adapter.

Invokes the ``codex`` CLI with the prompt piped on stdin. Token counts
fall back to the base heuristic unless terminal JSONL usage can be parsed
from ``--json`` output.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from orch.agents.base import (
    AgentInvocationOptions,
    AgentResult,
    apply_usage_capture,
    apply_invocation_options,
    dispatched_model_from_argv_env,
    ensure_usage_json_args,
    run_with_timeout,
    unwrap_codex_jsonl_stdout,
)
from orch.providers import ManagedProcessProvider


class CodexAdapter:
    name = "codex"

    def __init__(
        self,
        *,
        cmd: str = "codex exec",
        family: str = "openai",
        process_provider: ManagedProcessProvider | None = None,
    ) -> None:
        self.cmd = cmd
        self.family = family
        self.process_provider = process_provider

    def invoke(
        self,
        prompt: str,
        *,
        timeout: int,
        workdir: Path,
        routing_options: AgentInvocationOptions | None = None,
    ) -> AgentResult:
        argv = shlex.split(self.cmd)
        argv, env = apply_invocation_options(argv, routing_options)
        argv = ensure_usage_json_args(argv, "codex")
        dispatched_model = dispatched_model_from_argv_env(argv, env)
        result = run_with_timeout(
            argv,
            stdin_text=prompt,
            timeout=timeout,
            workdir=workdir,
            env=env,
            process_provider=self.process_provider,
        )
        raw_stdout = result.stdout
        return apply_usage_capture(
            result,
            provider="codex",
            raw_output=raw_stdout,
            answer_stdout=unwrap_codex_jsonl_stdout(raw_stdout),
            dispatched_model=dispatched_model,
        )
