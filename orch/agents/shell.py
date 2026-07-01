"""Generic shell adapter.

Lets the operator plug a new CLI into the orchestrator by editing
``project.yaml`` only — no code change. The ``cmd`` string may contain a
``{prompt}`` placeholder which is replaced at invocation time with the path
to a temp file holding the full prompt. If ``prompt_stdin: true`` is set in
the agent spec, the prompt is piped on stdin instead and the placeholder
(if present) is stripped.

If ``cost_regex`` is configured (``input_tokens`` and/or ``output_tokens``
patterns), the adapter scans stdout for the first match of each pattern
and promotes the result to an exact token count. Otherwise the base
heuristic applies.
"""
from __future__ import annotations

import re
import shlex
import tempfile
from pathlib import Path

from orch.agents.base import (
    AgentInvocationOptions,
    AgentResult,
    apply_invocation_options,
    dispatched_model_from_argv_env,
    run_with_timeout,
)
from orch.providers import ManagedProcessProvider


class ShellAdapter:
    name = "shell"

    def __init__(
        self,
        *,
        cmd: str,
        family: str,
        prompt_stdin: bool = False,
        cost_regex: dict | None = None,
        agent_name: str = "shell",
        process_provider: ManagedProcessProvider | None = None,
    ) -> None:
        if not cmd:
            raise ValueError("shell adapter requires a 'cmd' string")
        self.cmd = cmd
        self.family = family
        self.prompt_stdin = bool(prompt_stdin)
        self.cost_regex = dict(cost_regex or {})
        self.name = agent_name
        self.process_provider = process_provider

    def invoke(
        self,
        prompt: str,
        *,
        timeout: int,
        workdir: Path,
        routing_options: AgentInvocationOptions | None = None,
    ) -> AgentResult:
        if self.prompt_stdin:
            argv = shlex.split(self.cmd.replace("{prompt}", "").strip())
            argv, env = apply_invocation_options(argv, routing_options)
            dispatched_model = dispatched_model_from_argv_env(argv, env)
            result = run_with_timeout(
                argv,
                stdin_text=prompt,
                timeout=timeout,
                workdir=workdir,
                env=env,
                process_provider=self.process_provider,
            )
            result.model = dispatched_model
            self._maybe_exact_tokens(result, stdin_text=prompt)
            return result

        # File-based prompt: write to a temp file, substitute {prompt}.
        with tempfile.NamedTemporaryFile(
            "w", prefix="orch-prompt-", suffix=".md", delete=False
        ) as tf:
            tf.write(prompt)
            prompt_path = tf.name
        try:
            rendered = self.cmd.replace("{prompt}", prompt_path)
            argv = shlex.split(rendered)
            argv, env = apply_invocation_options(argv, routing_options)
            dispatched_model = dispatched_model_from_argv_env(argv, env)
            result = run_with_timeout(
                argv,
                stdin_text=None,
                timeout=timeout,
                workdir=workdir,
                env=env,
                process_provider=self.process_provider,
            )
            result.model = dispatched_model
            self._maybe_exact_tokens(result, stdin_text=prompt)
            return result
        finally:
            try:
                Path(prompt_path).unlink()
            except OSError:
                pass

    def _maybe_exact_tokens(self, result: AgentResult, *, stdin_text: str) -> None:
        if not self.cost_regex:
            return
        in_pat = self.cost_regex.get("input_tokens")
        out_pat = self.cost_regex.get("output_tokens")
        hit_in = hit_out = False
        if in_pat:
            m = re.search(in_pat, result.stdout)
            if m:
                try:
                    result.input_tokens = int(m.group(1))
                    hit_in = True
                except (ValueError, IndexError):
                    pass
        if out_pat:
            m = re.search(out_pat, result.stdout)
            if m:
                try:
                    result.output_tokens = int(m.group(1))
                    hit_out = True
                except (ValueError, IndexError):
                    pass
        # tokens_exact only when every configured pattern resolved.
        wants_in = bool(in_pat)
        wants_out = bool(out_pat)
        if (not wants_in or hit_in) and (not wants_out or hit_out):
            if wants_in or wants_out:
                result.tokens_exact = True
