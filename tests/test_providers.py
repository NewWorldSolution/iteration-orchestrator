"""Tests for thin orchestrator provider interfaces."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from orch.providers import (
    ShellGhProvider,
    ShellGitProvider,
    SubprocessCommandProvider,
    SubprocessManagedProcessProvider,
)


class RecordingCommandProvider:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.calls: list[dict] = []
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def run(self, argv, *, cwd: Path, timeout: int | None = None):
        self.calls.append({"argv": list(argv), "cwd": cwd, "timeout": timeout})
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def test_subprocess_command_provider_captures_text_and_cwd(tmp_path: Path):
    provider = SubprocessCommandProvider()
    proc = provider.run(
        [
            sys.executable,
            "-c",
            "import os, sys; print(os.getcwd()); print('ERR', file=sys.stderr)",
        ],
        cwd=tmp_path,
        timeout=30,
    )

    assert proc.returncode == 0
    assert proc.stdout.strip() == str(tmp_path)
    assert proc.stderr.strip() == "ERR"


def test_shell_git_provider_prefixes_git_and_preserves_call_shape(tmp_path: Path):
    command_provider = RecordingCommandProvider(stdout="ok\n")
    provider = ShellGitProvider(command_provider=command_provider)

    proc = provider.run(["status", "--short"], cwd=tmp_path, timeout=12)

    assert proc.stdout == "ok\n"
    assert command_provider.calls == [
        {
            "argv": ["git", "status", "--short"],
            "cwd": tmp_path,
            "timeout": 12,
        }
    ]


def test_shell_gh_provider_prefixes_gh_and_preserves_call_shape(tmp_path: Path):
    command_provider = RecordingCommandProvider(stdout="[]\n")
    provider = ShellGhProvider(command_provider=command_provider)

    proc = provider.run(["pr", "list", "--json", "number"], cwd=tmp_path, timeout=7)

    assert proc.stdout == "[]\n"
    assert command_provider.calls == [
        {
            "argv": ["gh", "pr", "list", "--json", "number"],
            "cwd": tmp_path,
            "timeout": 7,
        }
    ]


def test_managed_process_provider_roundtrips_stdin_stdout_stderr(tmp_path: Path):
    provider = SubprocessManagedProcessProvider()
    code = (
        "import os, sys; "
        "body=sys.stdin.read(); "
        "print(os.getcwd()); "
        "print('IN=' + body); "
        "print('ERR=' + body, file=sys.stderr)"
    )

    result = provider.run(
        [sys.executable, "-c", code],
        stdin_text="hello",
        timeout=30,
        workdir=tmp_path,
    )

    assert result.exit_code == 0
    assert str(tmp_path) in result.stdout
    assert "IN=hello" in result.stdout
    assert "ERR=hello" in result.stderr
    assert result.partial is False


def test_managed_process_provider_kills_process_group_and_keeps_partial_output(
    tmp_path: Path,
):
    provider = SubprocessManagedProcessProvider()
    child_code = "import time; time.sleep(30)"
    parent_code = (
        "import subprocess, sys, time; "
        "print('parent-start', flush=True); "
        f"subprocess.Popen([{sys.executable!r}, '-c', {child_code!r}]); "
        "time.sleep(30)"
    )

    result = provider.run(
        [sys.executable, "-c", parent_code],
        stdin_text=None,
        timeout=1,
        workdir=tmp_path,
        kill_grace_seconds=2,
    )

    assert result.partial is True
    assert result.exit_code != 0
    assert "parent-start" in result.stdout
    assert result.duration_s < 10
