"""Crash/resume characterization tests for orchestrator kill points."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.git_ops import checkout, current_sha, git, merge_no_ff
from orch.merge import PrSnapshot
from orch.state import STATUS_DONE, STATUS_IN_PROGRESS
from tests.test_runner import (
    FakeAdapter,
    _assert_visible_then_edit,
    _edit_file_on_invoke,
    _make_runner,
    _reviewer_verdict,
    repo as _runner_repo,
)


class CrashAtKillPoint(BaseException):
    """Stand in for process death so runner ``except Exception`` does not fire."""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _runner_repo.__wrapped__(tmp_path)


def _open_pr_snapshot(pr_url: str, *, cwd: Path) -> PrSnapshot:
    return PrSnapshot(state="OPEN", merge_sha=None, rollup=[])


def _no_ci_runner(
    repo: Path,
    *,
    impl_script: list,
    review_script: list,
):
    calls = SimpleNamespace(open_pr=[], comment_pr=[])

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, f"https://example.com/pr/{len(calls.open_pr)}"

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    def fake_wait(branch, *, cwd, ci_wait_seconds, **kw):
        raise AssertionError("wait_for_ci must not run in no-CI mode")

    impl = FakeAdapter(
        name="claude", family="anthropic", script=list(impl_script),
    )
    rev = FakeAdapter(name="codex", family="openai", script=list(review_script))
    runner, store, cost = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=False,
        implementer="claude",
        reviewer="codex",
        allow_noop_acceptance_reason="crash fixture reaches no-CI kill point",
        deps_kwargs={
            "open_pr": fake_open_pr,
            "comment_pr": fake_comment,
            "wait_for_ci": fake_wait,
            "query_pr_state": _open_pr_snapshot,
        },
    )
    runner.cfg.data["auto_merge"]["no_ci"] = True
    runner._final_scope_gate = lambda: None
    runner._final_nav_discoverability_gate = lambda: None
    return runner, store, cost, calls, impl, rev


def _resume_no_ci_runner(
    repo: Path,
    *,
    impl_script: list,
    review_script: list,
):
    runner, store, cost, calls, impl, rev = _no_ci_runner(
        repo, impl_script=impl_script, review_script=review_script,
    )
    store.load()
    return runner, store, cost, calls, impl, rev


def _show_iter_file(repo: Path, relpath: str) -> str:
    return git(
        ["show", f"demo/iteration-1:{relpath}"], cwd=repo, check=True,
    ).stdout


def _tasks_md_status(repo: Path, task_id: str) -> str:
    text = _show_iter_file(repo, "iterations/demo-i1/tasks.md")
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) > 4 and cells[1] == task_id:
            return cells[4]
    raise AssertionError(f"missing tasks.md row for {task_id}")


def _last_iteration_verdict(store) -> str | None:
    for event in reversed(store.events):
        meta = event.get("meta") or {}
        if event.get("kind") == "iteration" and "verdict" in meta:
            return meta["verdict"]
    return None


def test_resume_reconciles_crash_after_merge_before_iter_sha(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    runner, store, *_ = _no_ci_runner(
        repo,
        impl_script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
        review_script=[_reviewer_verdict("Verdict: PASS\n")],
    )

    def crash_after_merge_no_ff(self, task, *, message):
        checkout(self.deps.cwd, self.iter_branch)
        merge_no_ff(self.deps.cwd, task.branch, message=message)
        raise CrashAtKillPoint("after merge_no_ff before iter_branch_sha")

    monkeypatch.setattr(
        type(runner), "_merge_task_branch_locally", crash_after_merge_no_ff,
    )

    with pytest.raises(CrashAtKillPoint):
        runner.run()

    merge_sha = current_sha(runner.deps.cwd, runner.iter_branch)
    assert _show_iter_file(repo, "src/a.py") == "a=1\n"
    assert store.tasks["I1-T1"].status == STATUS_IN_PROGRESS
    assert store.tasks["I1-T1"].merge_sha is None

    resumed, resumed_store, *_ = _resume_no_ci_runner(
        repo,
        impl_script=[
            _assert_visible_then_edit("src/a.py", "a=1\n", "src/b.py", "b=2\n"),
        ],
        review_script=[_reviewer_verdict("Verdict: PASS\n")],
    )

    assert resumed.run() == 0
    assert resumed_store.tasks["I1-T1"].status == STATUS_DONE
    assert resumed_store.tasks["I1-T1"].auto_merged is True
    assert resumed_store.tasks["I1-T1"].merge_sha == merge_sha
    assert resumed_store.tasks["I1-T2"].status == STATUS_DONE


def test_resume_reconciles_crash_after_record_merge_before_done(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    runner, store, *_ = _no_ci_runner(
        repo,
        impl_script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
        review_script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    original_transition = store.task_transition

    def crash_before_done(task: str, status: str, **kwargs):
        if task == "I1-T1" and status == STATUS_DONE:
            raise CrashAtKillPoint("after record_merge before DONE")
        original_transition(task, status, **kwargs)

    monkeypatch.setattr(store, "task_transition", crash_before_done)

    with pytest.raises(CrashAtKillPoint):
        runner.run()

    merge_sha = current_sha(runner.deps.cwd, runner.iter_branch)
    assert store.tasks["I1-T1"].status == STATUS_IN_PROGRESS
    assert store.tasks["I1-T1"].auto_merged is True
    assert store.tasks["I1-T1"].merge_sha == merge_sha

    resumed, resumed_store, *_ = _resume_no_ci_runner(
        repo,
        impl_script=[
            _assert_visible_then_edit("src/a.py", "a=1\n", "src/b.py", "b=2\n"),
        ],
        review_script=[_reviewer_verdict("Verdict: PASS\n")],
    )

    assert resumed.run() == 0
    assert resumed_store.tasks["I1-T1"].status == STATUS_DONE
    assert resumed_store.tasks["I1-T1"].auto_merged is True
    assert resumed_store.tasks["I1-T1"].merge_sha == merge_sha
    assert resumed_store.tasks["I1-T2"].status == STATUS_DONE


def test_resume_reconciles_crash_after_done_before_tasks_md_update(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    runner, store, *_ = _no_ci_runner(
        repo,
        impl_script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
        review_script=[_reviewer_verdict("Verdict: PASS\n")],
    )

    def crash_before_tasks_md_done(task):
        if task.id == "I1-T1":
            raise CrashAtKillPoint("after DONE before tasks.md update")
        raise AssertionError(f"unexpected task-board write for {task.id}")

    monkeypatch.setattr(runner, "_write_tasks_md_done", crash_before_tasks_md_done)

    with pytest.raises(CrashAtKillPoint):
        runner.run()

    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert _tasks_md_status(repo, "I1-T1") == "WAITING"

    resumed, resumed_store, *_ = _resume_no_ci_runner(
        repo,
        impl_script=[
            _assert_visible_then_edit("src/a.py", "a=1\n", "src/b.py", "b=2\n"),
        ],
        review_script=[_reviewer_verdict("Verdict: PASS\n")],
    )

    assert resumed.run() == 0
    assert resumed_store.tasks["I1-T1"].status == STATUS_DONE
    assert resumed_store.tasks["I1-T2"].status == STATUS_DONE
    assert _tasks_md_status(repo, "I1-T1") == "DONE"
    assert _tasks_md_status(repo, "I1-T2") == "DONE"


def test_resume_with_in_progress_task_finishes_non_ready(repo: Path):
    runner, store, *_ = _no_ci_runner(
        repo, impl_script=[], review_script=[],
    )
    store.task_transition("I1-T1", STATUS_IN_PROGRESS)

    rc = runner.run()

    assert rc == 1
    assert _last_iteration_verdict(store) != "READY"
