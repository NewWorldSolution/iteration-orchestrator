"""Thin subprocess-backed provider interfaces for orchestrator boundaries.

The orchestrator still shells out for all command execution. These
interfaces name the boundary and keep the default implementation identical
to the previous direct ``subprocess`` calls: captured text output, explicit
working directories, command timeouts, and process-group termination for
long-running managed agent processes.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

DEFAULT_SIGKILL_GRACE_SECONDS = 10


class CommandProvider(Protocol):
    """Run a short-lived command and return the raw CompletedProcess."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess: ...


class SubprocessCommandProvider:
    """``subprocess.run`` implementation used by shell-backed providers."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        package_parent = Path(__file__).resolve().parents[1]
        existing_pythonpath = env.get("PYTHONPATH", "")
        pythonpath_parts = [str(package_parent)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        return subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )


@dataclass(frozen=True)
class ManagedProcessResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    partial: bool


class ManagedProcessProvider(Protocol):
    """Run a managed process with process-group timeout handling."""

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin_text: str | None,
        timeout: int,
        workdir: Path,
        env: Mapping[str, str] | None = None,
        kill_grace_seconds: int = DEFAULT_SIGKILL_GRACE_SECONDS,
    ) -> ManagedProcessResult: ...


class SubprocessManagedProcessProvider:
    """``Popen`` implementation with SIGTERM -> grace -> SIGKILL semantics."""

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin_text: str | None,
        timeout: int,
        workdir: Path,
        env: Mapping[str, str] | None = None,
        kill_grace_seconds: int = DEFAULT_SIGKILL_GRACE_SECONDS,
    ) -> ManagedProcessResult:
        start = time.monotonic()
        merged_env = dict(os.environ)
        if env:
            merged_env.update(dict(env))

        proc = subprocess.Popen(
            list(argv),
            cwd=str(workdir),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            text=True,
            start_new_session=True,
        )

        partial = False
        try:
            stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            partial = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=kill_grace_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()

        exit_code = proc.returncode if proc.returncode is not None else -1
        return ManagedProcessResult(
            exit_code=exit_code,
            stdout=stdout or "",
            stderr=stderr or "",
            duration_s=time.monotonic() - start,
            partial=partial,
        )


class GitProvider(Protocol):
    """Run ``git`` with shell CLI semantics."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess: ...


class ShellGitProvider:
    """Git provider backed by the local ``git`` executable."""

    def __init__(self, command_provider: CommandProvider | None = None) -> None:
        self.command_provider = command_provider or SubprocessCommandProvider()

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess:
        return self.command_provider.run(["git", *list(args)], cwd=cwd, timeout=timeout)


class GhProvider(Protocol):
    """Run ``gh`` with shell CLI semantics."""

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess: ...


class ShellGhProvider:
    """GitHub CLI provider backed by the local ``gh`` executable."""

    def __init__(self, command_provider: CommandProvider | None = None) -> None:
        self.command_provider = command_provider or SubprocessCommandProvider()

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess:
        return self.command_provider.run(["gh", *list(args)], cwd=cwd, timeout=timeout)
