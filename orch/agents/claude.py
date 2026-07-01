"""Claude CLI adapter.

Invokes ``claude -p`` (configurable via project.yaml) with the prompt piped
on stdin. Token counts fall back to the heuristic estimate in ``base.py``
unless terminal JSON usage can be parsed from ``--output-format json``.
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
    unwrap_claude_json_stdout,
)
from orch.providers import ManagedProcessProvider


class ClaudeAdapter:
    name = "claude"

    def __init__(
        self,
        *,
        cmd: str = "claude -p",
        family: str = "anthropic",
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
        argv = ensure_usage_json_args(argv, "claude")
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
            provider="claude",
            raw_output=raw_stdout,
            answer_stdout=unwrap_claude_json_stdout(raw_stdout),
            dispatched_model=dispatched_model,
        )
