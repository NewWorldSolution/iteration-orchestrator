"""Tests for orch.lifecycle (SHA guard, revert, cleanup)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import orch.lifecycle as lifecycle_mod
from orch.git_ops import (
    branch_exists,
    commit,
    current_sha,
    git,
    stage_all,
)
from orch.lifecycle import (
    PhaseResolutionError,
    cleanup_task_branches,
    extract_phase,
    head_sha_guard,
    reset_in_progress_tasks_for_recovery,
    resolve_phase_branch,
    revert_task_merge,
)
from orch.state import (
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_HUMAN_MERGE,
    STATUS_STOPPED_PREFIX,
    StateStore,
)
from orch.tasks_schema import TaskBoard, Task, ExecutionPlan


def _board(path: Path, iter_branch: str, tasks: list[Task]) -> TaskBoard:
    return TaskBoard(
        path=path, title="t",
        iteration_branch=iter_branch,
        status="IN_PROGRESS",
        depends_on_header="-", blocks_header="-",
        execution_plan=ExecutionPlan(approach="x", qa="x", note="x"),
        tasks=tasks,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(["init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git(["config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    git(["config", "user.name", "Tester"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("r\n")
    stage_all(tmp_path)
    commit(tmp_path, "init")
    git(["branch", "iter/x"], cwd=tmp_path, check=True)
    return tmp_path


def test_head_sha_guard_passes_when_no_recorded_sha(repo: Path, tmp_path: Path):
    s = StateStore(log_dir=tmp_path / "l", iteration="i", iter_branch="iter/x")
    r = head_sha_guard(s, cwd=repo, iter_branch="iter/x", accept_external=False)
    assert r.ok


def test_head_sha_guard_detects_drift(repo: Path, tmp_path: Path):
    s = StateStore(log_dir=tmp_path / "l", iteration="i", iter_branch="iter/x")
    s.set_iter_branch_sha("deadbeef" * 5)  # not real
    r = head_sha_guard(s, cwd=repo, iter_branch="iter/x", accept_external=False)
    assert not r.ok
    assert "moved" in r.reason


def test_head_sha_guard_accept_external_records_new(repo: Path, tmp_path: Path):
    s = StateStore(log_dir=tmp_path / "l", iteration="i", iter_branch="iter/x")
    s.set_iter_branch_sha("deadbeef" * 5)
    cur = current_sha(repo, "iter/x")
    r = head_sha_guard(s, cwd=repo, iter_branch="iter/x", accept_external=True)
    assert r.ok
    assert s.snapshot.iter_branch_sha == cur


def test_head_sha_guard_records_fetch_failure_note(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    s = StateStore(log_dir=tmp_path / "l", iteration="i", iter_branch="iter/x")
    s.set_iter_branch_sha(current_sha(repo, "iter/x"))

    def fail_fetch(cwd: Path):
        raise RuntimeError("network down")

    monkeypatch.setattr(lifecycle_mod, "fetch", fail_fetch)

    result = head_sha_guard(
        s, cwd=repo, iter_branch="iter/x", accept_external=False,
    )

    assert result.ok
    fetch_notes = [
        event for event in s.events
        if event.get("meta", {}).get("event") == "head_sha_guard_fetch_failed"
    ]
    assert fetch_notes
    assert fetch_notes[-1]["meta"]["iter_branch"] == "iter/x"
    assert "network down" in fetch_notes[-1]["meta"]["msg"]


def test_phase_resolution_uses_configured_phase_branch_pattern():
    cfg = SimpleNamespace(
        data={"project": {"phase_branch_pattern": "phase-{phase}"}}
    )

    assert extract_phase(cfg, "alpha-i1") == "alpha"
    assert extract_phase(cfg, "demo-i1") == "demo"
    assert resolve_phase_branch(cfg, "alpha-i1") == "phase-alpha"
    assert resolve_phase_branch(cfg, "demo-i1") == "phase-demo"
    with pytest.raises(PhaseResolutionError, match="cannot resolve phase"):
        extract_phase(cfg, "not-an-iteration")


def test_revert_task_merge_creates_revert_commit(repo: Path, tmp_path: Path):
    # Simulate: on iter/x, a task branch was merged (no-ff) producing a merge commit
    git(["checkout", "iter/x"], cwd=repo, check=True)
    git(["checkout", "-b", "feat/y"], cwd=repo, check=True)
    (repo / "x.py").write_text("x=1\n")
    stage_all(repo)
    commit(repo, "feat change")
    git(["checkout", "iter/x"], cwd=repo, check=True)
    git(["merge", "--no-ff", "-m", "merge feat/y", "feat/y"], cwd=repo, check=True)
    merge_sha = current_sha(repo, "iter/x")

    # State has the merge recorded for T1
    log_dir = tmp_path / "logs"
    s = StateStore(log_dir=log_dir, iteration="i", iter_branch="iter/x")
    s.record_pr("I1-T1", "https://example.com/pr/1")
    s.record_merge("I1-T1", auto_merged=True, merge_sha=merge_sha)

    board = _board(
        repo / "tasks.md", "iter/x",
        [Task(id="I1-T1", title="y", owner="-", status="DONE",
              depends_on=[], branch="feat/y")],
    )
    r = revert_task_merge(s, board, "I1-T1", cwd=repo)
    assert r.ok
    assert r.revert_sha
    assert current_sha(repo, "iter/x") == r.revert_sha
    # x.py should no longer have that line because the revert undid it
    assert not (repo / "x.py").exists() or (repo / "x.py").read_text() == ""


def test_revert_fails_when_no_merge_recorded(repo: Path, tmp_path: Path):
    s = StateStore(log_dir=tmp_path / "l", iteration="i", iter_branch="iter/x")
    board = _board(
        repo / "tasks.md", "iter/x",
        [Task(id="I1-T1", title="t", owner="-", status="WAITING",
              depends_on=[], branch="feat/y")],
    )
    r = revert_task_merge(s, board, "I1-T1", cwd=repo)
    assert not r.ok
    assert "no merge" in r.message


def test_cleanup_deletes_only_done_branches(repo: Path, tmp_path: Path):
    # Set up two task branches
    git(["checkout", "iter/x"], cwd=repo, check=True)
    git(["branch", "feat/a"], cwd=repo, check=True)
    git(["branch", "feat/b"], cwd=repo, check=True)
    assert branch_exists(repo, "feat/a")
    assert branch_exists(repo, "feat/b")

    s = StateStore(log_dir=tmp_path / "l", iteration="i", iter_branch="iter/x")
    s.task_transition("I1-T1", STATUS_DONE)  # only T1 is done
    board = _board(
        repo / "tasks.md", "iter/x",
        [
            Task(id="I1-T1", title="a", owner="-", status="DONE",
                 depends_on=[], branch="feat/a"),
            Task(id="I1-T2", title="b", owner="-", status="WAITING",
                 depends_on=[], branch="feat/b"),
        ],
    )
    result = cleanup_task_branches(s, board, cwd=repo)
    assert "feat/a" in result.deleted
    assert not branch_exists(repo, "feat/a")
    assert branch_exists(repo, "feat/b")
    assert any("feat/b" in b for b, _ in result.skipped)


def test_cleanup_force_deletes_non_done(repo: Path, tmp_path: Path):
    git(["checkout", "iter/x"], cwd=repo, check=True)
    # Create a branch that has extra commits (needs -D, not -d)
    git(["checkout", "-b", "feat/messy"], cwd=repo, check=True)
    (repo / "m.txt").write_text("m\n")
    stage_all(repo)
    commit(repo, "messy")
    git(["checkout", "iter/x"], cwd=repo, check=True)

    s = StateStore(log_dir=tmp_path / "l", iteration="i", iter_branch="iter/x")
    board = _board(
        repo / "tasks.md", "iter/x",
        [Task(id="I1-T1", title="m", owner="-", status="WAITING",
              depends_on=[], branch="feat/messy")],
    )
    result = cleanup_task_branches(s, board, cwd=repo, force=True)
    assert "feat/messy" in result.deleted


def test_recover_does_not_reset_stopped_done_or_human_merge_tasks(
    tmp_path: Path,
):
    store = StateStore(
        log_dir=tmp_path / "logs",
        iteration="i",
        iter_branch="iter/x",
    )
    store.task_transition("I1-T1", STATUS_IN_PROGRESS)
    store.task_transition("I1-T2", STATUS_DONE)
    store.task_transition(
        "I1-T3",
        STATUS_STOPPED_PREFIX + "INTERNAL",
        reason="INTERNAL",
        msg="boom",
    )
    store.task_transition("I1-T4", STATUS_NEEDS_HUMAN_MERGE)

    assert reset_in_progress_tasks_for_recovery(store) == ["I1-T1"]

    assert store.tasks["I1-T1"].status == "WAITING"
    assert store.tasks["I1-T2"].status == STATUS_DONE
    assert store.tasks["I1-T3"].status == STATUS_STOPPED_PREFIX + "INTERNAL"
    assert store.tasks["I1-T4"].status == STATUS_NEEDS_HUMAN_MERGE
    reset_events = [
        event for event in store.events
        if event.get("meta", {}).get("event") == "recover_task_reset"
    ]
    assert [event["task"] for event in reset_events] == ["I1-T1"]
