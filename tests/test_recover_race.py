"""Concurrency characterization for recover/apply versus run locking."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import orch.cli as cli
import orch.lifecycle as lifecycle
from orch.locks import RunLockError, RunStateLock
from orch.state import STATUS_IN_PROGRESS, StateStore
from tests.test_runner import repo as _runner_repo


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _runner_repo.__wrapped__(tmp_path)


def _iteration_lock(repo: Path, *, command: str) -> RunStateLock:
    return RunStateLock(
        log_dir=repo / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        command=command,
        repo_root=repo,
        orch_workdir=repo / ".orch" / "worktrees" / "demo-i1",
    )


def test_recover_apply_blocks_run_before_removing_stale_lock(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    log_dir = repo / "tools" / "logs" / "demo-i1"
    state = StateStore(
        log_dir=log_dir, iteration="demo-i1", iter_branch="demo/iteration-1",
    )
    state.task_transition("I1-T1", STATUS_IN_PROGRESS)
    stale_lock = _iteration_lock(repo, command="run").acquire()

    removed = threading.Event()
    run_attempted = threading.Event()
    release_run_lock = threading.Event()
    results = SimpleNamespace(
        recover_rc=None,
        recover_error=None,
        run_acquired=False,
        run_error=None,
    )

    def build_stale_plan(
        store,
        *,
        repo_root: Path,
        iteration: str,
        log_dir: Path,
        worktree_root: Path | None = None,
    ):
        return lifecycle.build_recovery_plan(
            store,
            repo_root=repo_root,
            iteration=iteration,
            log_dir=log_dir,
            worktree_root=worktree_root,
            pid_status=lambda _pid, _host: "stale",
        )

    original_remove = cli.remove_recoverable_lock

    def remove_then_hold_window(plan, *, force_lock: bool):
        meta = original_remove(plan, force_lock=force_lock)
        removed.set()
        run_attempted.wait()
        return meta

    monkeypatch.chdir(repo)
    monkeypatch.setattr(cli, "build_recovery_plan", build_stale_plan)
    monkeypatch.setattr(cli, "remove_recoverable_lock", remove_then_hold_window)

    def recover_apply():
        try:
            results.recover_rc = cli.cmd_recover(
                SimpleNamespace(
                    iteration="demo-i1", apply=True, force_lock=False,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced in main thread
            results.recover_error = exc
            removed.set()
            run_attempted.set()

    def competing_run():
        removed.wait()
        run_lock = _iteration_lock(repo, command="run")
        try:
            run_lock.acquire()
        except RunLockError as exc:
            results.run_error = str(exc)
            run_attempted.set()
            return
        results.run_acquired = True
        run_attempted.set()
        release_run_lock.wait()
        run_lock.release()

    run_thread = threading.Thread(target=competing_run)
    recover_thread = threading.Thread(target=recover_apply)
    run_thread.start()
    recover_thread.start()
    recover_thread.join()
    release_run_lock.set()
    run_thread.join()
    stale_lock.release()

    assert results.recover_error is None
    assert results.recover_rc == 0
    assert results.run_acquired is False
    assert results.run_error is not None
