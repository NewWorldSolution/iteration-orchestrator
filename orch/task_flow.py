"""Pure DAG readiness helpers for the orchestrator runner.

This module intentionally does not mutate run state. Callers own any
``StateStore`` transitions so event ordering remains explicit in the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from orch.state import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_STOPPED_PREFIX,
    STATUS_WAITING,
)
from orch.tasks_schema import Task, TaskBoard


@dataclass(frozen=True)
class ReadySelection:
    task: Task | None
    blocked_upstream_task_ids: list[str]


def live_statuses(
    board: TaskBoard, state_statuses: Mapping[str, str]
) -> dict[str, str]:
    """Merge runtime statuses with board defaults using runner semantics."""
    live = dict(state_statuses)
    for task in board.tasks:
        live.setdefault(task.id, task.status)
    return live


def select_next_ready(board: TaskBoard, live: Mapping[str, str]) -> ReadySelection:
    """Select the next ready task and report tasks with blocked upstream deps.

    Preserves the runner's one-pass semantics: upstream-blocked task IDs are
    computed from the provided ``live`` snapshot without updating it while the
    board is scanned.
    """
    ready: list[Task] = []
    blocked_upstream: list[str] = []
    for task in board.tasks:
        status = live.get(task.id, task.status)
        if status != STATUS_WAITING:
            continue
        if any(is_upstream_blocked(dep, live) for dep in task.depends_on):
            blocked_upstream.append(task.id)
            continue
        if not all(live.get(dep) == STATUS_DONE for dep in task.depends_on):
            continue
        ready.append(task)
    ready.sort(key=lambda task: task.index)
    return ReadySelection(
        task=ready[0] if ready else None,
        blocked_upstream_task_ids=blocked_upstream,
    )


def is_upstream_blocked(dep: str, live: Mapping[str, str]) -> bool:
    status = live.get(dep, STATUS_WAITING)
    return (
        status.startswith(STATUS_STOPPED_PREFIX)
        or status == STATUS_BLOCKED_UPSTREAM
    )


def downstream_blocked_task_ids(
    board: TaskBoard, live: Mapping[str, str], stopped_id: str
) -> list[str]:
    blocked: list[str] = []
    for task in board.tasks:
        status = live.get(task.id, task.status)
        if status != STATUS_WAITING:
            continue
        if depends_transitively_on(board, task.id, stopped_id):
            blocked.append(task.id)
    return blocked


def depends_transitively_on(board: TaskBoard, task_id: str, target: str) -> bool:
    seen: set[str] = set()
    stack = list(board.by_id(task_id).depends_on)
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        if dep == target:
            return True
        try:
            stack.extend(board.by_id(dep).depends_on)
        except KeyError:
            pass
    return False
