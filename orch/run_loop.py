"""Run-loop DAG helpers for the orchestrator runner.

These helpers own the mutation-free readiness decisions. The runner passes in
its state store so the existing transition/event ordering stays unchanged.
"""
from __future__ import annotations

from orch.state import STATUS_BLOCKED_UPSTREAM
from orch.task_flow import (
    downstream_blocked_task_ids,
    live_statuses,
    select_next_ready,
)
from orch.tasks_schema import Task, TaskBoard


def pick_next_ready_task(board: TaskBoard, state) -> Task | None:
    live = live_statuses(
        board, {tid: task_state.status for tid, task_state in state.tasks.items()}
    )
    selection = select_next_ready(board, live)
    for task_id in selection.blocked_upstream_task_ids:
        # Mark BLOCKED_UPSTREAM once so it shows in the report.
        if state.tasks.get(task_id) is None or (
            state.tasks[task_id].status == "WAITING"
        ):
            state.task_transition(task_id, STATUS_BLOCKED_UPSTREAM)
    return selection.task


def mark_downstream_blocked(board: TaskBoard, state, stopped_id: str) -> None:
    live = {tid: task_state.status for tid, task_state in state.tasks.items()}
    for task_id in downstream_blocked_task_ids(board, live, stopped_id):
        state.task_transition(task_id, STATUS_BLOCKED_UPSTREAM)
