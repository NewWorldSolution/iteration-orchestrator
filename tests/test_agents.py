"""Tests for orch.agents."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from orch.agents import (
    AgentInvocationOptions,
    ClaudeAdapter,
    CodexAdapter,
    ShellAdapter,
    build_adapter,
    build_allowed_files_prefix,
    compose_prompt,
)
from orch.providers import ManagedProcessResult


def test_allowed_files_prefix_empty():
    assert build_allowed_files_prefix([]) == ""


def test_allowed_files_prefix_lists_paths():
    prefix = build_allowed_files_prefix(["app/a.py", "app/b.py"])
    assert "You may ONLY create or modify" in prefix
    assert "- app/a.py" in prefix
    assert "- app/b.py" in prefix
    assert prefix.endswith("---\n\n") or prefix.endswith("---\n")


def test_compose_prompt_prepends_prefix():
    body = "Do the task."
    out = compose_prompt(["x.py"], body)
    assert out.startswith("You may ONLY")
    assert out.endswith(body)


def test_compose_prompt_no_prefix_when_no_files():
    assert compose_prompt([], "body") == "body"


# ---------------------------------------------------------------------------
# Real subprocess invocation via a trivial python -c command.
# Keeps tests hermetic without needing claude/codex installed.
# ---------------------------------------------------------------------------


def _py(code: str) -> str:
    return f'{sys.executable} -c {_sq(code)}'


def _sq(s: str) -> str:
    # shlex-safe single-quote wrap
    return "'" + s.replace("'", "'\\''") + "'"


def test_shell_adapter_stdin_roundtrip(tmp_path: Path):
    code = "import sys; s=sys.stdin.read(); print('LEN='+str(len(s)))"
    a = ShellAdapter(
        cmd=_py(code), family="openai", prompt_stdin=True, agent_name="pyshell"
    )
    res = a.invoke("hello world", timeout=30, workdir=tmp_path)
    assert res.exit_code == 0
    assert "LEN=11" in res.stdout
    assert res.partial is False
    # Heuristic estimate from the 11-char prompt: 11 // 4 = 2
    assert res.input_tokens == 2
    assert res.tokens_exact is False


def test_shell_adapter_prompt_file(tmp_path: Path):
    code = (
        "import sys; p=sys.argv[1]; "
        "print('PATH=' + p); "
        "print('BODY=' + open(p).read())"
    )
    a = ShellAdapter(
        cmd=f"{_py(code)} {{prompt}}",
        family="openai",
        prompt_stdin=False,
        agent_name="pyfile",
    )
    res = a.invoke("prompt-body-here", timeout=30, workdir=tmp_path)
    assert res.exit_code == 0
    assert "BODY=prompt-body-here" in res.stdout


def test_shell_adapter_cost_regex_promotes_to_exact(tmp_path: Path):
    code = (
        "print('input_tokens=123'); print('output_tokens=456'); print('done')"
    )
    a = ShellAdapter(
        cmd=_py(code),
        family="openai",
        prompt_stdin=True,
        cost_regex={
            "input_tokens":  r"input_tokens=(\d+)",
            "output_tokens": r"output_tokens=(\d+)",
        },
        agent_name="pyshell",
    )
    res = a.invoke("whatever", timeout=30, workdir=tmp_path)
    assert res.input_tokens == 123
    assert res.output_tokens == 456
    assert res.tokens_exact is True


def test_shell_adapter_cost_regex_partial_miss_stays_estimated(tmp_path: Path):
    code = "print('no usage info here')"
    a = ShellAdapter(
        cmd=_py(code),
        family="openai",
        prompt_stdin=True,
        cost_regex={"input_tokens": r"input_tokens=(\d+)"},
        agent_name="pyshell",
    )
    res = a.invoke("xxxxxxxx", timeout=30, workdir=tmp_path)
    assert res.tokens_exact is False


def test_timeout_kills_child_and_marks_partial(tmp_path: Path):
    code = "import time; time.sleep(30); print('never')"
    a = ShellAdapter(
        cmd=_py(code), family="openai", prompt_stdin=True, agent_name="pyshell"
    )
    res = a.invoke("x", timeout=1, workdir=tmp_path)
    assert res.partial is True
    assert res.exit_code != 0
    assert res.duration_s < 20  # much less than 30 s sleep


def test_nonzero_exit_code_captured(tmp_path: Path):
    code = "import sys; sys.exit(7)"
    a = ShellAdapter(
        cmd=_py(code), family="openai", prompt_stdin=True, agent_name="pyshell"
    )
    res = a.invoke("x", timeout=15, workdir=tmp_path)
    assert res.exit_code == 7
    assert res.partial is False


class FakeManagedProcessProvider:
    def __init__(
        self,
        *,
        stdout: str = "out",
        stderr: str = "err",
        exit_code: int = 0,
        partial: bool = False,
    ):
        self.calls: list[dict] = []
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.partial = partial

    def run(
        self,
        argv,
        *,
        stdin_text,
        timeout,
        workdir,
        env=None,
        kill_grace_seconds=10,
    ):
        self.calls.append(
            {
                "argv": list(argv),
                "stdin_text": stdin_text,
                "timeout": timeout,
                "workdir": workdir,
                "env": env,
                "kill_grace_seconds": kill_grace_seconds,
            }
        )
        return ManagedProcessResult(
            exit_code=self.exit_code,
            stdout=self.stdout,
            stderr=self.stderr,
            duration_s=1.25,
            partial=self.partial,
        )


def test_shell_adapter_uses_injected_process_provider(tmp_path: Path):
    provider = FakeManagedProcessProvider()
    adapter = ShellAdapter(
        cmd="agent --flag",
        family="openai",
        prompt_stdin=True,
        agent_name="fake",
        process_provider=provider,
    )

    res = adapter.invoke("prompt body", timeout=42, workdir=tmp_path)

    assert res.stdout == "out"
    assert res.stderr == "err"
    assert provider.calls == [
        {
            "argv": ["agent", "--flag"],
            "stdin_text": "prompt body",
            "timeout": 42,
            "workdir": tmp_path,
            "env": None,
            "kill_grace_seconds": 10,
        }
    ]


def test_agent_adapter_receives_routing_env_or_args(tmp_path: Path):
    provider = FakeManagedProcessProvider()
    adapter = CodexAdapter(
        cmd="codex exec",
        family="openai",
        process_provider=provider,
    )

    adapter.invoke(
        "prompt body",
        timeout=42,
        workdir=tmp_path,
        routing_options=AgentInvocationOptions(
            args=("--model", "fixture-codex-model", "--reasoning", "high"),
            env={"ORCH_REASONING_EFFORT": "high"},
        ),
    )

    assert provider.calls[0]["argv"] == [
        "codex",
        "exec",
        "--model",
        "fixture-codex-model",
        "--reasoning",
        "high",
        "--json",
    ]
    assert provider.calls[0]["env"] == {"ORCH_REASONING_EFFORT": "high"}


def test_claude_adapter_captures_json_usage_and_preserves_answer_stdout(
    tmp_path: Path,
):
    raw = json.dumps(
        {
            "type": "result",
            "model": "claude-sonnet-4-5",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 400,
            },
            "result": "Looks good.\nVerdict: PASS\n",
        }
    )
    provider = FakeManagedProcessProvider(stdout=raw)
    adapter = ClaudeAdapter(
        cmd="claude -p --model dispatched-sonnet",
        family="anthropic",
        process_provider=provider,
    )

    res = adapter.invoke("prompt body", timeout=42, workdir=tmp_path)

    assert provider.calls[0]["argv"] == [
        "claude",
        "-p",
        "--model",
        "dispatched-sonnet",
        "--output-format",
        "json",
    ]
    assert res.stdout == "Looks good.\nVerdict: PASS\n"
    assert "Verdict: PASS" in res.stdout
    assert res.tokens_exact is True
    assert res.input_tokens == 1000
    assert res.output_tokens == 200
    assert res.cached_input_tokens == 300
    assert res.cache_creation_input_tokens == 400
    assert res.provider == "claude"
    assert res.model == "claude-sonnet-4-5"
    assert res.parser_status == "parsed"
    assert res.extra["raw_terminal_json"] == raw
    assert res.extra["stdout_unwrapped"] is True


def test_codex_adapter_captures_jsonl_usage_and_preserves_answer_stdout(
    tmp_path: Path,
):
    raw = "\n".join(
        [
            json.dumps(
                {
                    "type": "agent_message",
                    "message": "Looks good.\nVerdict: PASS\n",
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "model": "gpt-5.4",
                    "usage": {
                        "input_tokens": 4000,
                        "cached_input_tokens": 1000,
                        "output_tokens": 500,
                        "reasoning_output_tokens": 200,
                    },
                }
            ),
        ]
    )
    provider = FakeManagedProcessProvider(stdout=raw)
    adapter = CodexAdapter(
        cmd="codex exec --model dispatched-gpt",
        family="openai",
        process_provider=provider,
    )

    res = adapter.invoke("prompt body", timeout=42, workdir=tmp_path)

    assert provider.calls[0]["argv"] == [
        "codex",
        "exec",
        "--model",
        "dispatched-gpt",
        "--json",
    ]
    assert res.stdout == "Looks good.\nVerdict: PASS\n"
    assert "Verdict: PASS" in res.stdout
    assert res.tokens_exact is True
    assert res.input_tokens == 4000
    assert res.output_tokens == 500
    assert res.cached_input_tokens == 1000
    assert res.reasoning_output_tokens == 200
    assert res.provider == "codex"
    assert res.model == "gpt-5.4"
    assert res.parser_status == "parsed"
    assert res.extra["raw_terminal_json"] == raw
    assert res.extra["stdout_unwrapped"] is True


def test_adapter_usage_flags_are_not_duplicated(tmp_path: Path):
    claude_provider = FakeManagedProcessProvider()
    ClaudeAdapter(
        cmd="claude -p --output-format json",
        family="anthropic",
        process_provider=claude_provider,
    ).invoke("prompt", timeout=42, workdir=tmp_path)

    codex_provider = FakeManagedProcessProvider()
    CodexAdapter(
        cmd="codex exec --json",
        family="openai",
        process_provider=codex_provider,
    ).invoke("prompt", timeout=42, workdir=tmp_path)

    assert claude_provider.calls[0]["argv"].count("--output-format") == 1
    assert codex_provider.calls[0]["argv"].count("--json") == 1


def test_usage_parse_failure_keeps_estimate_and_raw_stdout(tmp_path: Path):
    provider = FakeManagedProcessProvider(stdout="{not json")
    adapter = ClaudeAdapter(
        cmd="claude -p --model dispatched-sonnet",
        family="anthropic",
        process_provider=provider,
    )

    res = adapter.invoke("abcdefgh", timeout=42, workdir=tmp_path)

    assert res.stdout == "{not json"
    assert res.tokens_exact is False
    assert res.input_tokens == 2
    assert res.output_tokens == 2
    assert res.model == "dispatched-sonnet"
    assert res.parser_status == "parse_failed"
    assert "parse failed" in (res.parser_warning or "")
    assert res.extra["raw_terminal_json"] == "{not json"
    assert res.extra["stdout_unwrapped"] is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_adapter_claude():
    a = build_adapter("claude", {"type": "claude", "cmd": "claude -p", "family": "anthropic"})
    assert isinstance(a, ClaudeAdapter)
    assert a.family == "anthropic"


def test_build_adapter_codex():
    a = build_adapter("codex", {"type": "codex", "cmd": "codex", "family": "openai"})
    assert isinstance(a, CodexAdapter)


def test_build_adapter_shell_default_type():
    a = build_adapter(
        "cursor",
        {"cmd": "cursor-agent --prompt-file {prompt}", "family": "openai"},
    )
    assert isinstance(a, ShellAdapter)
    assert a.name == "cursor"


def test_build_adapter_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown type"):
        build_adapter("x", {"type": "martian", "cmd": "x", "family": "f"})


def test_build_adapter_missing_fields_raises():
    with pytest.raises(ValueError, match="missing"):
        build_adapter("x", {"type": "claude"})
