from __future__ import annotations

import json
import os
from pathlib import Path
import socket

import pytest

from orch.locks import (
    LOCK_GUARD_FILENAME,
    RUN_LOCK_FILENAME,
    RunLockError,
    RunLockInfo,
    RunStateLock,
)


def _lock(
    tmp_path: Path,
    *,
    command: str = "run",
    pid_status=lambda _pid, _host: "active",
) -> RunStateLock:
    return RunStateLock(
        log_dir=tmp_path / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        command=command,
        repo_root=tmp_path,
        orch_workdir=tmp_path / ".orch" / "worktrees" / "demo-i1",
        pid_status=pid_status,
    )


def _write_guard(
    tmp_path: Path,
    *,
    pid: int,
    hostname: str | None = None,
) -> Path:
    path = tmp_path / "tools" / "logs" / "demo-i1" / LOCK_GUARD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    info = RunLockInfo(
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        command="recover",
        pid=pid,
        hostname=hostname or socket.gethostname(),
        repo_root=str(tmp_path),
        orch_workdir=str(tmp_path / ".orch" / "worktrees" / "demo-i1"),
        created_at="2026-06-17T00:00:00Z",
        token="guard-token",
    )
    path.write_text(json.dumps(info.to_dict()))
    return path


def test_run_state_lock_acquire_and_release(tmp_path: Path):
    lock = _lock(tmp_path)

    lock.acquire()
    path = tmp_path / "tools" / "logs" / "demo-i1" / RUN_LOCK_FILENAME
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["iteration"] == "demo-i1"
    assert payload["iter_branch"] == "demo/iteration-1"
    assert payload["command"] == "run"
    assert payload["pid"] == os.getpid()
    assert payload["orch_workdir"].endswith(".orch/worktrees/demo-i1")

    lock.release()
    assert not path.exists()


def test_run_state_lock_blocks_active_existing_lock(tmp_path: Path):
    first = _lock(tmp_path, pid_status=lambda _pid, _host: "active")
    first.acquire()
    second = _lock(tmp_path, command="resume")

    try:
        try:
            second.acquire()
        except RunLockError as exc:
            msg = str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("second lock unexpectedly acquired")

        assert "active orch run lock" in msg
        assert "demo-i1" in msg
        assert "demo/iteration-1" in msg
        assert "run.lock" in msg
        assert "orch_workdir" in msg
        assert "Wait for the running orchestrator command" in msg
    finally:
        first.release()


def test_run_state_lock_reports_stale_existing_lock(tmp_path: Path):
    first = _lock(tmp_path, pid_status=lambda _pid, _host: "stale")
    first.acquire()
    second = _lock(tmp_path, command="resume", pid_status=lambda _pid, _host: "stale")

    try:
        try:
            second.acquire()
        except RunLockError as exc:
            msg = str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("second lock unexpectedly acquired")

        assert "stale orch run lock" in msg
        assert "No running process with pid" in msg
        assert "Remove" in msg
        assert "only after confirming no orchestrator run is active" in msg
        assert "Recovery note: lock removal is manual" in msg
    finally:
        first.release()


@pytest.mark.parametrize(
    ("pid_status", "expected"),
    [
        (
            "stale",
            [
                "stale orch run lock",
                "No running process with pid",
                "only after confirming no orchestrator run is active",
            ],
        ),
        (
            "active",
            [
                "active orch run lock",
                "Wait for the running orchestrator command to finish",
                "investigate that process before removing the lock",
            ],
        ),
        (
            "unknown",
            [
                "orch run lock with unknown process state",
                "cannot be verified from this host",
                "Confirm the other run is inactive before removing the lock",
            ],
        ),
    ],
)
def test_lock_diagnostics_cover_stale_active_and_unknown_states(
    tmp_path: Path,
    pid_status: str,
    expected: list[str],
):
    lock_root = tmp_path / pid_status
    first = _lock(lock_root, pid_status=lambda _pid, _host: pid_status)
    first.acquire()
    second = _lock(
        lock_root,
        command="resume",
        pid_status=lambda _pid, _host: pid_status,
    )

    try:
        with pytest.raises(RunLockError) as exc:
            second.acquire()

        msg = str(exc.value)
        for snippet in expected:
            assert snippet in msg
        assert "Next action:" in msg
        assert "Recovery note: lock removal is manual" in msg
        assert "Inspect the recorded process" in msg
        assert "remove the lock only after confirming no orchestrator run" in msg
    finally:
        first.release()


def test_run_state_lock_does_not_release_someone_elses_lock(tmp_path: Path):
    first = _lock(tmp_path)
    first.acquire()
    path = tmp_path / "tools" / "logs" / "demo-i1" / RUN_LOCK_FILENAME
    payload = json.loads(path.read_text())
    payload["token"] = "someone-else"
    path.write_text(json.dumps(payload))

    first.release()

    assert path.exists()


def test_run_state_lock_recovers_stale_acquisition_guard(tmp_path: Path):
    guard = _write_guard(tmp_path, pid=123456)
    lock = _lock(tmp_path, pid_status=lambda _pid, _host: "stale")

    lock.acquire()

    try:
        assert lock.acquired
        assert not guard.exists()
        assert (
            tmp_path / "tools" / "logs" / "demo-i1" / RUN_LOCK_FILENAME
        ).exists()
    finally:
        lock.release()


def test_run_state_lock_blocks_live_acquisition_guard(tmp_path: Path):
    guard = _write_guard(tmp_path, pid=os.getpid())
    lock = _lock(tmp_path, pid_status=lambda _pid, _host: "active")

    with pytest.raises(RunLockError) as exc:
        lock.acquire()

    msg = str(exc.value)
    assert "orch lock acquisition guard with active exists" in msg
    assert "pid=" in msg
    assert "Wait for the other orchestrator command" in msg
    assert guard.exists()
