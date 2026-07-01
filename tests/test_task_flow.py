from __future__ import annotations

from pathlib import Path

from orch.state import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_STOPPED_PREFIX,
    STATUS_WAITING,
)
from orch.task_flow import (
    downstream_blocked_task_ids,
    is_upstream_blocked,
    live_statuses,
    select_next_ready,
)
from orch.tasks_schema import ExecutionPlan, Task, TaskBoard


def _task(
    task_id: str,
    *,
    status: str = STATUS_WAITING,
    depends_on: list[str] | None = None,
) -> Task:
    return Task(
        id=task_id,
        title=task_id,
        owner="TBD",
        status=status,
        depends_on=list(depends_on or []),
        branch=f"demo/{task_id.lower()}",
    )


def _board(*tasks: Task) -> TaskBoard:
    return TaskBoard(
        path=Path("tasks.md"),
        title="Demo",
        iteration_branch="demo/iteration-1",
        status=STATUS_WAITING,
        depends_on_header="none",
        blocks_header="none",
        execution_plan=ExecutionPlan(
            approach="task_by_task",
            qa="standard",
            note="test",
        ),
        tasks=list(tasks),
    )


def test_lowest_index_ready_task_is_selected_even_when_board_order_differs():
    board = _board(_task("I1-T2"), _task("I1-T1"))

    selection = select_next_ready(board, live_statuses(board, {}))

    assert selection.task is not None
    assert selection.task.id == "I1-T1"
    assert selection.blocked_upstream_task_ids == []


def test_missing_state_falls_back_to_board_status():
    board = _board(
        _task("I1-T1", status=STATUS_DONE),
        _task("I1-T2", depends_on=["I1-T1"]),
    )

    selection = select_next_ready(board, live_statuses(board, {}))

    assert selection.task is not None
    assert selection.task.id == "I1-T2"


def test_dependency_that_is_not_done_blocks_readiness_without_upstream_block():
    board = _board(
        _task("I1-T1", status=STATUS_IN_PROGRESS),
        _task("I1-T2", depends_on=["I1-T1"]),
    )

    selection = select_next_ready(board, live_statuses(board, {}))

    assert selection.task is None
    assert selection.blocked_upstream_task_ids == []


def test_upstream_stopped_or_blocked_dependency_is_reported():
    board = _board(
        _task("I1-T1"),
        _task("I1-T2", depends_on=["I1-T1"]),
        _task("I1-T3", depends_on=["I1-T2"]),
    )
    stopped_live = live_statuses(
        board, {"I1-T1": f"{STATUS_STOPPED_PREFIX}SCOPE"}
    )

    stopped_selection = select_next_ready(board, stopped_live)

    assert is_upstream_blocked("I1-T1", stopped_live)
    assert stopped_selection.task is None
    assert stopped_selection.blocked_upstream_task_ids == ["I1-T2"]

    blocked_live = live_statuses(board, {"I1-T1": STATUS_BLOCKED_UPSTREAM})

    blocked_selection = select_next_ready(board, blocked_live)

    assert is_upstream_blocked("I1-T1", blocked_live)
    assert blocked_selection.task is None
    assert blocked_selection.blocked_upstream_task_ids == ["I1-T2"]


def test_transitive_downstream_blocking_preserves_waiting_only_semantics():
    board = _board(
        _task("I1-T1"),
        _task("I1-T2", depends_on=["I1-T1"]),
        _task("I1-T3", depends_on=["I1-T2"]),
        _task("I1-T4", status=STATUS_DONE, depends_on=["I1-T3"]),
    )

    blocked = downstream_blocked_task_ids(board, {}, "I1-T1")

    assert blocked == ["I1-T2", "I1-T3"]
