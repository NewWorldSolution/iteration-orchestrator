"""Pure bounded-wave planner for explicitly parallel-safe tasks."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping

from orch.state import STATUS_DONE, STATUS_WAITING
from orch.tasks_schema import Task, TaskBoard


def plan_parallel_waves(
    board: TaskBoard,
    *,
    live_statuses: Mapping[str, str] | None = None,
    max_concurrency: int = 1,
) -> tuple[tuple[Task, ...], ...]:
    """Plan bounded waves of tasks that are safe to run concurrently.

    This is intentionally side-effect free. It does not create branches,
    worktrees, locks, or subprocesses; it only inspects task metadata.
    """
    if (
        isinstance(max_concurrency, bool)
        or not isinstance(max_concurrency, int)
        or max_concurrency < 1
    ):
        raise ValueError("max_concurrency must be a positive integer")

    validate_conflict_tokens(board)

    statuses = {task.id: task.status for task in board.tasks}
    if live_statuses:
        statuses.update(live_statuses)

    done = {tid for tid, status in statuses.items() if status == STATUS_DONE}
    planned: set[str] = set()
    waves: list[tuple[Task, ...]] = []
    candidates = [
        task
        for task in sorted(board.tasks, key=lambda item: item.index)
        if statuses.get(task.id, task.status) == STATUS_WAITING
    ]

    while True:
        available = done | planned
        ready = [
            task
            for task in candidates
            if task.id not in planned
            and task.parallel_safe.value
            and all(dep in available for dep in task.depends_on)
            and all(
                dep in available
                for dep in task.parallel_safe.requires_serial_after
            )
        ]
        if not ready:
            break

        wave: list[Task] = []
        for task in ready:
            if len(wave) >= max_concurrency:
                break
            if all(_can_share_wave(task, member) for member in wave):
                wave.append(task)

        if not wave:
            break
        waves.append(tuple(wave))
        planned.update(task.id for task in wave)

    return tuple(waves)


def validate_conflict_tokens(board: TaskBoard) -> None:
    """Fail closed when a parallel-safe task's conflict token matches nothing.

    A conflict token must reference either a known task id or a path that
    overlaps some task's allowed files. A token matching neither is almost
    certainly a typo or stale reference: ``_declared_conflict`` would silently
    skip it, so the planner could place genuinely-conflicting tasks in the same
    wave (conflict under-detection). Reject it instead of ignoring it.

    Only parallel-safe tasks are checked — non-parallel tasks never enter a
    wave, so their declarations are inert and must not block serial runs.
    """
    task_ids = {task.id for task in board.tasks}
    for task in board.tasks:
        if not task.parallel_safe.value:
            continue
        for token in task.parallel_safe.conflicts:
            if token in task_ids:
                continue
            if any(
                _path_groups_overlap(token, path)
                for other in board.tasks
                for path in other.allowed_files
            ):
                continue
            raise ValueError(
                f"task {task.id!r} declares conflict token {token!r} that "
                "matches no known task id and no task's allowed files; fix the "
                "typo or remove the stale conflict declaration"
            )


def _can_share_wave(left: Task, right: Task) -> bool:
    return (
        not _allowed_files_overlap(left, right)
        and not _declared_conflict(left, right)
        and not _declared_conflict(right, left)
    )


def _allowed_files_overlap(left: Task, right: Task) -> bool:
    return any(
        _path_groups_overlap(left_path, right_path)
        for left_path in left.allowed_files
        for right_path in right.allowed_files
    )


def _declared_conflict(source: Task, other: Task) -> bool:
    for token in source.parallel_safe.conflicts:
        if token == other.id:
            return True
        if any(_path_groups_overlap(token, path) for path in other.allowed_files):
            return True
    return False


def _path_groups_overlap(left: str, right: str) -> bool:
    left_norm = _normalize_path_group(left)
    right_norm = _normalize_path_group(right)
    if not left_norm or not right_norm:
        return False
    return (
        left_norm == right_norm
        or right_norm.startswith(f"{left_norm}/")
        or left_norm.startswith(f"{right_norm}/")
    )


def _normalize_path_group(value: str) -> str:
    stripped = value.strip().strip("`").strip().strip("/")
    while stripped.startswith("./"):
        stripped = stripped[2:]
    if not stripped:
        return ""
    return PurePosixPath(stripped).as_posix().rstrip("/")
