"""Agent adapter protocol.

Adapters wrap CLI tools (claude, codex, shell)
into a uniform invocation surface. The orchestrator calls ``invoke`` with a
prompt, a timeout (tier-scaled per preflight), and a working directory, and
receives an :class:`AgentResult` with exit code, captured output, duration,
and token counts (exact or estimated).

Responsibilities common to every adapter:
    * Prepend the allowed-files prefix to the prompt.
    * Enforce the timeout with SIGTERM, wait 10 s, then SIGKILL; set
      ``partial=True`` on the result when the timeout fires.
    * Return estimated token counts (``len(chars) // 4``) as a fallback —
      concrete adapters may override when the underlying CLI reports exact
      figures or when ``cost_regex`` resolves them.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol

from orch.cost import estimate_tokens, parse_agent_usage
from orch.providers import (
    DEFAULT_SIGKILL_GRACE_SECONDS,
    ManagedProcessProvider,
    SubprocessManagedProcessProvider,
)

# SIGKILL grace (SIGTERM → 10 s → SIGKILL).
SIGKILL_GRACE_SECONDS = DEFAULT_SIGKILL_GRACE_SECONDS
_DEFAULT_PROCESS_PROVIDER = SubprocessManagedProcessProvider()


@dataclass
class AgentResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    input_tokens: int
    output_tokens: int
    tokens_exact: bool
    partial: bool = False  # set when the timeout killed the process
    provider: str | None = None
    model: str | None = None
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    parser_status: str | None = None
    parser_warning: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AgentInvocationOptions:
    """Provider-specific invocation additions resolved from project config."""

    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


def build_allowed_files_prefix(allowed_files: list[str]) -> str:
    """Compose the C2 prefix that is prepended to every implementer/fixer prompt."""
    if not allowed_files:
        return ""
    lines = ["You may ONLY create or modify the following files:"]
    for p in allowed_files:
        lines.append(f"- {p}")
    lines.append("")
    lines.append(
        "Do not touch any other files. Any changes outside this list will be reverted."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def compose_prompt(allowed_files: list[str], body: str) -> str:
    prefix = build_allowed_files_prefix(allowed_files)
    if not prefix:
        return body
    return prefix + body


class AgentAdapter(Protocol):
    """Uniform invocation surface for all CLI-backed agents."""

    name: str
    family: str

    def invoke(
        self,
        prompt: str,
        *,
        timeout: int,
        workdir: Path,
        routing_options: AgentInvocationOptions | None = None,
    ) -> AgentResult: ...


def apply_invocation_options(
    argv: list[str],
    routing_options: AgentInvocationOptions | None,
) -> tuple[list[str], dict[str, str] | None]:
    if routing_options is None:
        return argv, None
    rendered_argv = [*argv, *routing_options.args]
    env = dict(routing_options.env) if routing_options.env else None
    return rendered_argv, env


def ensure_usage_json_args(argv: Sequence[str], provider: str) -> list[str]:
    """Return argv with provider JSON-output flags added once."""
    out = list(argv)
    provider_key = provider.strip().lower()
    if provider_key in {"claude", "anthropic"}:
        if not _has_option(out, "--output-format"):
            out.extend(["--output-format", "json"])
    elif provider_key in {"codex", "openai"}:
        if "--json" not in out:
            out.append("--json")
    return out


def dispatched_model_from_argv_env(
    argv: Sequence[str], env: Mapping[str, str] | None = None
) -> str | None:
    """Best-effort model identity from already-dispatched CLI args/env."""
    args = list(argv)
    for idx, arg in enumerate(args):
        if arg in {"--model", "-m"} and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith("--model="):
            value = arg.split("=", 1)[1].strip()
            if value:
                return value
    for key in (
        "OPENAI_MODEL",
        "ANTHROPIC_MODEL",
        "CLAUDE_MODEL",
        "CODEX_MODEL",
        "ORCH_MODEL",
    ):
        value = (env or {}).get(key)
        if value:
            return str(value)
    return None


def dispatched_model_from_options(
    routing_options: AgentInvocationOptions | None,
) -> str | None:
    if routing_options is None:
        return None
    return dispatched_model_from_argv_env(
        routing_options.args,
        routing_options.env,
    )


def apply_usage_capture(
    result: AgentResult,
    *,
    provider: str,
    raw_output: str,
    answer_stdout: str | None,
    dispatched_model: str | None = None,
) -> AgentResult:
    """Promote parsed terminal usage onto an AgentResult in place."""
    usage = parse_agent_usage(provider, raw_output)
    warnings = tuple(usage.warnings)
    result.provider = usage.provider or provider
    result.parser_status = usage.parser_status
    result.parser_warning = "; ".join(warnings) if warnings else None
    result.model = usage.model or dispatched_model
    result.extra.update(
        {
            "usage_provider": result.provider,
            "raw_terminal_json": raw_output,
            "usage_parser_status": usage.parser_status,
            "usage_parser_warnings": list(warnings),
            "usage_model": usage.model,
            "dispatched_model": dispatched_model,
        }
    )
    if answer_stdout is not None:
        result.stdout = answer_stdout
        result.extra["stdout_unwrapped"] = True
    else:
        result.extra["stdout_unwrapped"] = False

    if usage.exact:
        result.input_tokens = usage.input_tokens
        result.output_tokens = usage.output_tokens
        result.cached_input_tokens = usage.cached_input_tokens
        result.cache_creation_input_tokens = usage.cache_creation_input_tokens
        result.reasoning_output_tokens = usage.reasoning_output_tokens
        result.tokens_exact = True
    return result


def cost_record_usage_kwargs(
    result: AgentResult,
    *,
    family: str,
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    """Fields from AgentResult that map to T1 CostLogger.record usage args."""
    return {
        "family": family,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "exact": result.tokens_exact,
        "provider": result.provider or provider or family,
        "model": result.model or model,
        "cached_input_tokens": result.cached_input_tokens,
        "cache_creation_input_tokens": result.cache_creation_input_tokens,
        "reasoning_output_tokens": result.reasoning_output_tokens,
        "parser_status": result.parser_status,
        "parser_warning": result.parser_warning,
    }


def unwrap_claude_json_stdout(raw_stdout: str) -> str | None:
    try:
        payload = json.loads(raw_stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("result", "text", "message"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    message = payload.get("message")
    if isinstance(message, dict):
        text = _text_from_content(message.get("content"))
        if text is not None:
            return text
    return _text_from_content(payload.get("content"))


def unwrap_codex_jsonl_stdout(raw_stdout: str) -> str | None:
    parts: list[str] = []
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        msg = event.get("msg") if isinstance(event.get("msg"), dict) else {}
        event_type = str(
            event.get("type")
            or event.get("event")
            or msg.get("type")
            or msg.get("event")
            or ""
        )
        if event_type == "turn.completed":
            continue
        text = _message_text_from_event(event)
        if text:
            parts.append(text)
    if not parts:
        return None
    return "\n".join(parts)


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def _message_text_from_event(event: dict) -> str | None:
    for key in ("message", "text", "content", "delta"):
        value = event.get(key)
        text = _text_from_content(value)
        if text:
            return text
    msg = event.get("msg")
    if isinstance(msg, dict):
        for key in ("message", "text", "content", "delta"):
            text = _text_from_content(msg.get(key))
            if text:
                return text
        item = msg.get("item")
        text = _text_from_content(item)
        if text:
            return text
    return _text_from_content(event.get("item"))


def _text_from_content(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "message", "output_text"):
            text = _text_from_content(value.get(key))
            if text:
                return text
        return None
    if isinstance(value, list):
        parts = [
            text for item in value
            if (text := _text_from_content(item))
        ]
        return "\n".join(parts) if parts else None
    return None


def run_with_timeout(
    argv: list[str],
    *,
    stdin_text: str | None,
    timeout: int,
    workdir: Path,
    env: dict | None = None,
    process_provider: ManagedProcessProvider | None = None,
) -> AgentResult:
    """Spawn a subprocess, enforce SIGTERM → grace → SIGKILL, capture streams.

    Returns an :class:`AgentResult` with ``partial=True`` if the timeout
    fires. Token counts default to heuristic estimates derived from the
    stdin (input) and stdout (output) character lengths; callers that can
    parse exact counts should overwrite them on the returned result.
    """
    provider = process_provider or _DEFAULT_PROCESS_PROVIDER
    result = provider.run(
        argv,
        stdin_text=stdin_text,
        timeout=timeout,
        workdir=workdir,
        env=env,
        kill_grace_seconds=SIGKILL_GRACE_SECONDS,
    )
    return AgentResult(
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=result.duration_s,
        input_tokens=estimate_tokens(stdin_text),
        output_tokens=estimate_tokens(result.stdout),
        tokens_exact=False,
        partial=result.partial,
    )
