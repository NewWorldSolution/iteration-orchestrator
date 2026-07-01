from __future__ import annotations

from pathlib import Path

import pytest

from orch.parallel import plan_parallel_waves, validate_conflict_tokens
from orch.state import STATUS_DONE, STATUS_WAITING
from orch.tasks_schema import (
    ExecutionPlan,
    ParallelSafetyDeclaration,
    Task,
    TaskBoard,
)


def _safe(
    *,
    conflicts: tuple[str, ...] = (),
    requires_serial_after: tuple[str, ...] = (),
) -> ParallelSafetyDeclaration:
    return ParallelSafetyDeclaration(
        value=True,
        reason="disjoint task",
        conflicts=conflicts,
        requires_serial_after=requires_serial_after,
    )


def _task(
    task_id: str,
    *,
    allowed_files: list[str],
    depends_on: list[str] | None = None,
    parallel_safe: ParallelSafetyDeclaration | None = None,
    status: str = STATUS_WAITING,
) -> Task:
    return Task(
        id=task_id,
        title=task_id,
        owner="TBD",
        status=status,
        depends_on=list(depends_on or []),
        branch=f"demo/{task_id.lower()}",
        allowed_files=allowed_files,
        parallel_safe=parallel_safe or ParallelSafetyDeclaration(),
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


def _ids(waves: tuple[tuple[Task, ...], ...]) -> list[list[str]]:
    return [[task.id for task in wave] for wave in waves]


def test_parallel_wave_planner_groups_disjoint_safe_tasks():
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"], parallel_safe=_safe()),
        _task("I1-T2", allowed_files=["src/b.py"], parallel_safe=_safe()),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T1", "I1-T2"]]


def test_parallel_wave_planner_refuses_overlapping_allowed_files():
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"], parallel_safe=_safe()),
        _task("I1-T2", allowed_files=["src/a.py"], parallel_safe=_safe()),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T1"], ["I1-T2"]]


def test_parallel_wave_planner_respects_declared_conflicts():
    board = _board(
        _task(
            "I1-T1",
            allowed_files=["src/a.py"],
            parallel_safe=_safe(conflicts=("I1-T2",)),
        ),
        _task("I1-T2", allowed_files=["src/b.py"], parallel_safe=_safe()),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T1"], ["I1-T2"]]


def test_planner_fails_closed_on_dangling_conflict_token():
    # "I1-T9" is neither a known task id nor a path any task touches — a typo
    # would silently disable conflict protection, so reject it (QA T5 note).
    board = _board(
        _task(
            "I1-T1",
            allowed_files=["src/a.py"],
            parallel_safe=_safe(conflicts=("I1-T9",)),
        ),
        _task("I1-T2", allowed_files=["src/b.py"], parallel_safe=_safe()),
    )

    with pytest.raises(ValueError, match="matches no known task id"):
        plan_parallel_waves(board, max_concurrency=2)


def test_planner_accepts_path_based_conflict_token():
    # A token that overlaps another task's allowed files is a valid, honored
    # declaration (and must NOT be rejected as dangling).
    board = _board(
        _task(
            "I1-T1",
            allowed_files=["src/a.py"],
            parallel_safe=_safe(conflicts=("src/b.py",)),
        ),
        _task("I1-T2", allowed_files=["src/b.py"], parallel_safe=_safe()),
    )

    validate_conflict_tokens(board)  # does not raise
    waves = plan_parallel_waves(board, max_concurrency=2)
    assert _ids(waves) == [["I1-T1"], ["I1-T2"]]


def test_validate_conflict_tokens_ignores_non_parallel_tasks():
    # A non-parallel-safe task never enters a wave; its stale token must not
    # block planning of the parallel-safe tasks.
    inert = _task(
        "I1-T3",
        allowed_files=["src/c.py"],
        parallel_safe=ParallelSafetyDeclaration(
            value=False, conflicts=("does-not-exist",)
        ),
    )
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"], parallel_safe=_safe()),
        _task("I1-T2", allowed_files=["src/b.py"], parallel_safe=_safe()),
        inert,
    )

    validate_conflict_tokens(board)  # does not raise
    waves = plan_parallel_waves(board, max_concurrency=2)
    assert _ids(waves) == [["I1-T1", "I1-T2"]]


def test_parallel_wave_planner_respects_max_concurrency():
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"], parallel_safe=_safe()),
        _task("I1-T2", allowed_files=["src/b.py"], parallel_safe=_safe()),
        _task("I1-T3", allowed_files=["src/c.py"], parallel_safe=_safe()),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T1", "I1-T2"], ["I1-T3"]]


def test_parallel_wave_planner_respects_dependencies():
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"], parallel_safe=_safe()),
        _task(
            "I1-T2",
            allowed_files=["src/b.py"],
            depends_on=["I1-T1"],
            parallel_safe=_safe(),
        ),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T1"], ["I1-T2"]]


def test_parallel_wave_planner_respects_requires_serial_after():
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"], parallel_safe=_safe()),
        _task(
            "I1-T2",
            allowed_files=["src/b.py"],
            parallel_safe=_safe(requires_serial_after=("I1-T1",)),
        ),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T1"], ["I1-T2"]]


def test_parallel_wave_planner_requires_explicit_parallel_safe():
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"]),
        _task("I1-T2", allowed_files=["src/b.py"], parallel_safe=_safe()),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T2"]]


def test_parallel_wave_planner_accepts_live_done_dependencies():
    board = _board(
        _task(
            "I1-T1",
            allowed_files=["src/a.py"],
            parallel_safe=_safe(),
            status=STATUS_DONE,
        ),
        _task(
            "I1-T2",
            allowed_files=["src/b.py"],
            depends_on=["I1-T1"],
            parallel_safe=_safe(),
        ),
    )

    waves = plan_parallel_waves(board, max_concurrency=2)

    assert _ids(waves) == [["I1-T2"]]


def test_parallel_wave_planner_rejects_invalid_max_concurrency():
    board = _board(
        _task("I1-T1", allowed_files=["src/a.py"], parallel_safe=_safe()),
    )

    with pytest.raises(ValueError, match="max_concurrency"):
        plan_parallel_waves(board, max_concurrency=0)
