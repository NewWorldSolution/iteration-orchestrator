"""Iteration lifecycle helpers: HEAD-SHA guard, revert, cleanup, recover.

Small, self-contained operations the CLI dispatches to. They are separate
from :mod:`orch.runner` because they run *outside* the main loop —
before it (guard), or after it (revert / cleanup).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from orch.git_ops import (
    branch_exists,
    checkout,
    cleanup_orch_workdir,
    current_sha,
    fetch,
    git,
    orch_workdir,
    salvage_worktree,
    working_tree_clean,
)
from orch.locks import (
    RUN_LOCK_FILENAME,
    PidStatus,
    RunLockInfo,
    default_pid_status,
)
from orch.state import (
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    StateStore,
)
from orch.tasks_schema import TaskBoard


@dataclass
class SHAGuardResult:
    ok: bool
    reason: str = ""
    recorded_sha: str | None = None
    current_sha: str | None = None


class PhaseResolutionError(ValueError):
    """Raised when config asks for a phase but an iteration cannot resolve it."""


def _config_data(cfg) -> dict:
    return getattr(cfg, "data", cfg)


def extract_phase(cfg, iteration: str) -> str:
    """Resolve the phase token from the configured iteration id shape.

    This intentionally stays minimal: it centralizes the configured
    ``<phase>-i<n>`` slug contract so callers fail closed instead of each
    silently returning ``None`` from their own regex fallback.
    """
    project = _config_data(cfg).get("project", {})
    phase_branch_pattern = str(project.get("phase_branch_pattern") or "")
    if not phase_branch_pattern:
        raise PhaseResolutionError(
            "project.phase_branch_pattern is empty; cannot derive phase branch"
        )
    if "{phase}" not in phase_branch_pattern:
        raise PhaseResolutionError(
            "project.phase_branch_pattern must contain '{phase}' to derive "
            "a phase branch"
        )
    match = re.fullmatch(r"p?([A-Za-z0-9]+)-i\d+", iteration)
    if match is None:
        raise PhaseResolutionError(
            f"cannot resolve phase from iteration {iteration!r}; expected "
            "iteration id like 'alpha-i1' or 'demo-i1'"
        )
    return match.group(1)


def resolve_phase_branch(cfg, iteration: str) -> str | None:
    pattern = str(_config_data(cfg).get("project", {}).get("phase_branch_pattern") or "")
    if not pattern:
        return None
    return pattern.replace("{phase}", extract_phase(cfg, iteration))


def head_sha_guard(
    state: StateStore, *, cwd: Path, iter_branch: str, accept_external: bool
) -> SHAGuardResult:
    """Fail if the iter-branch HEAD moved since last recorded SHA.

    Returning ``ok=False`` asks the operator to rerun with
    ``--accept-external``.
    """
    recorded = state.snapshot.iter_branch_sha
    if not recorded:
        return SHAGuardResult(ok=True)
    try:
        fetch(cwd)
    except Exception as exc:
        state.append_event(
            kind="note",
            meta={
                "event": "head_sha_guard_fetch_failed",
                "iter_branch": iter_branch,
                "msg": str(exc),
            },
        )
    cur = current_sha(cwd, iter_branch)
    if cur == recorded:
        return SHAGuardResult(ok=True, recorded_sha=recorded, current_sha=cur)
    if accept_external:
        state.set_iter_branch_sha(cur)
        return SHAGuardResult(ok=True, recorded_sha=recorded, current_sha=cur)
    return SHAGuardResult(
        ok=False,
        reason=(
            f"iter-branch {iter_branch} moved: recorded {recorded}, "
            f"current {cur}. Re-run with --accept-external if intentional."
        ),
        recorded_sha=recorded,
        current_sha=cur,
    )


@dataclass
class RevertResult:
    ok: bool
    message: str
    revert_sha: str | None = None


def revert_task_merge(
    state: StateStore, board: TaskBoard, task_id: str, *, cwd: Path,
) -> RevertResult:
    """Revert the merge commit recorded for ``task_id`` on the iter branch.

    Uses ``git revert -m 1 <merge_sha> --no-edit``. The operator still
    needs to update tasks.md manually (or rerun with a fresh task) — we
    only touch git state here.
    """
    ts = state.tasks.get(task_id)
    if ts is None or not ts.merge_sha:
        return RevertResult(
            ok=False,
            message=f"no merge recorded for {task_id}",
        )
    checkout(cwd, board.iteration_branch)
    res = git(
        ["revert", "--no-edit", "-m", "1", ts.merge_sha],
        cwd=cwd,
    )
    if not res.ok:
        return RevertResult(ok=False, message=res.stderr.strip())
    new_sha = current_sha(cwd, "HEAD")
    state.append_event(
        kind="note", task=task_id,
        meta={"event": "revert", "revert_sha": new_sha,
              "reverted_merge": ts.merge_sha},
    )
    state.set_iter_branch_sha(new_sha)
    return RevertResult(ok=True, message=f"reverted at {new_sha}", revert_sha=new_sha)


@dataclass
class CleanupResult:
    deleted: list[str]
    skipped: list[tuple[str, str]]  # (branch, reason)


def cleanup_task_branches(
    state: StateStore, board: TaskBoard, *, cwd: Path, force: bool = False,
) -> CleanupResult:
    """Delete local task branches for DONE tasks.

    Only branches matching a known task's branch are deleted. Non-DONE or
    unknown branches are left alone so manual work is preserved.
    """
    deleted: list[str] = []
    skipped: list[tuple[str, str]] = []
    for t in board.tasks:
        br = t.branch
        if not br:
            continue
        if not branch_exists(cwd, br):
            continue
        ts = state.tasks.get(t.id)
        if not force and (ts is None or ts.status != STATUS_DONE):
            skipped.append((br, f"{t.id} not DONE"))
            continue
        res = git(["branch", "-D" if force else "-d", br], cwd=cwd)
        if res.ok:
            deleted.append(br)
        else:
            skipped.append((br, res.stderr.strip()))
    return CleanupResult(deleted=deleted, skipped=skipped)


@dataclass(frozen=True)
class RecoverLockPlan:
    path: Path
    exists: bool
    info: RunLockInfo | None = None
    pid_status: PidStatus | None = None
    read_error: str | None = None
    belongs_to_iteration: bool = True

    @property
    def removable_without_force(self) -> bool:
        return (
            self.exists
            and self.info is not None
            and self.belongs_to_iteration
            and self.pid_status == "stale"
        )

    @property
    def force_required(self) -> bool:
        return (
            self.exists
            and self.info is not None
            and self.belongs_to_iteration
            and self.pid_status in {"active", "unknown"}
        )


@dataclass(frozen=True)
class RecoverWorkdirPlan:
    path: Path
    exists: bool
    dirty: bool | None = None
    dirty_error: str | None = None


@dataclass(frozen=True)
class RecoverPlan:
    iteration: str
    lock: RecoverLockPlan
    workdir: RecoverWorkdirPlan
    in_progress_tasks: list[str]


@dataclass(frozen=True)
class RecoverSalvageResult:
    branch: str
    sha: str


class RecoverApplyError(RuntimeError):
    """Raised when a recover apply action must fail closed."""


def _read_run_lock(path: Path) -> tuple[RunLockInfo | None, str | None]:
    try:
        return RunLockInfo.from_dict(json.loads(path.read_text())), None
    except FileNotFoundError:
        return None, None
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        return None, str(exc)


def build_recovery_plan(
    state: StateStore,
    *,
    repo_root: Path,
    iteration: str,
    log_dir: Path,
    worktree_root: Path | None = None,
    pid_status: Callable[[int, str], PidStatus] = default_pid_status,
) -> RecoverPlan:
    lock_path = log_dir / RUN_LOCK_FILENAME
    lock_info: RunLockInfo | None = None
    lock_status: PidStatus | None = None
    lock_error: str | None = None
    belongs_to_iteration = True
    if lock_path.exists():
        lock_info, lock_error = _read_run_lock(lock_path)
        if lock_info is not None:
            lock_status = pid_status(lock_info.pid, lock_info.hostname)
            belongs_to_iteration = lock_info.iteration == iteration

    workdir = orch_workdir(
        repo_root, iteration, worktree_root=worktree_root
    )
    dirty: bool | None = None
    dirty_error: str | None = None
    if workdir.exists():
        try:
            dirty = not working_tree_clean(workdir)
        except Exception as exc:
            dirty_error = str(exc)

    in_progress = sorted(
        task_id for task_id, task in state.tasks.items()
        if task.status == STATUS_IN_PROGRESS
    )
    return RecoverPlan(
        iteration=iteration,
        lock=RecoverLockPlan(
            path=lock_path,
            exists=lock_path.exists(),
            info=lock_info,
            pid_status=lock_status,
            read_error=lock_error,
            belongs_to_iteration=belongs_to_iteration,
        ),
        workdir=RecoverWorkdirPlan(
            path=workdir,
            exists=workdir.exists(),
            dirty=dirty,
            dirty_error=dirty_error,
        ),
        in_progress_tasks=in_progress,
    )


def remove_recoverable_lock(
    plan: RecoverPlan, *, force_lock: bool,
) -> dict | None:
    lock = plan.lock
    if not lock.exists:
        return None
    if lock.info is None:
        raise RecoverApplyError(
            f"recover: cannot remove unreadable lock at {lock.path}: "
            f"{lock.read_error or 'unknown read error'}"
        )
    if not lock.belongs_to_iteration:
        raise RecoverApplyError(
            f"recover: lock at {lock.path} belongs to iteration "
            f"{lock.info.iteration!r}, not {plan.iteration!r}"
        )
    if lock.pid_status in {"active", "unknown"} and not force_lock:
        raise RecoverApplyError(
            f"recover: {lock.pid_status} lock at {lock.path} requires "
            "--force-lock after confirming no orchestrator run is active"
        )
    if lock.pid_status != "stale" and not force_lock:
        raise RecoverApplyError(
            f"recover: lock at {lock.path} is not safely removable "
            f"(pid_status={lock.pid_status!r})"
        )
    lock.path.unlink()
    return {
        "event": "recover_lock_removed",
        "path": str(lock.path),
        "pid": lock.info.pid,
        "hostname": lock.info.hostname,
        "pid_status": lock.pid_status,
        "forced": bool(force_lock and lock.pid_status in {"active", "unknown"}),
    }


def salvage_dirty_recover_workdir(
    state: StateStore,
    *,
    workdir: Path,
    iteration: str,
) -> RecoverSalvageResult | None:
    if not workdir.exists():
        return None
    try:
        dirty = not working_tree_clean(workdir)
    except Exception as exc:
        raise RecoverApplyError(
            f"recover: cannot determine workdir cleanliness for {workdir}: {exc}"
        ) from exc
    if not dirty:
        return None

    branch = f"salvage/{iteration}/recover"
    sha = salvage_worktree(
        workdir,
        branch,
        f"orch recover: salvage {iteration} dirty workdir before cleanup",
    )
    if sha is None:
        raise RecoverApplyError(
            f"recover: workdir {workdir} is dirty but salvage failed; "
            "not removing it"
        )
    state.append_event(
        kind="note",
        meta={
            "event": "recover_salvage",
            "branch": branch,
            "sha": sha,
            "workdir": str(workdir),
        },
    )
    return RecoverSalvageResult(branch=branch, sha=sha)


def cleanup_recover_workdir(
    state: StateStore,
    *,
    repo_root: Path,
    iteration: str,
    worktree_root: Path | None = None,
) -> bool:
    workdir = orch_workdir(
        repo_root, iteration, worktree_root=worktree_root
    )
    if not workdir.exists():
        return False
    cleanup_orch_workdir(
        repo_root, iteration, worktree_root=worktree_root
    )
    state.append_event(
        kind="note",
        meta={
            "event": "recover_workdir_cleaned",
            "workdir": str(workdir),
        },
    )
    return True


def reset_in_progress_tasks_for_recovery(state: StateStore) -> list[str]:
    reset: list[str] = []
    for task_id, task in list(state.tasks.items()):
        if task.status != STATUS_IN_PROGRESS:
            continue
        state.reset_task(task_id)
        state.append_event(
            kind="note",
            task=task_id,
            meta={"event": "recover_task_reset"},
        )
        reset.append(task_id)
    return reset
