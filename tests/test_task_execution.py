"""Characterization tests for task-stop event order before T4 split."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_runner import (
    FakeAdapter,
    _edit_file_on_invoke,
    _make_runner,
    _reviewer_verdict,
    repo as _runner_repo,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _runner_repo.__wrapped__(tmp_path)


def _event_signature(event: dict) -> tuple[str, str | None, str | None, str | None]:
    meta = event.get("meta") or {}
    marker = (
        meta.get("event")
        or meta.get("status")
        or meta.get("verdict")
        or meta.get("phase")
        or ("sha" if "sha" in meta else None)
    )
    return (
        event.get("kind"),
        event.get("task"),
        event.get("step"),
        marker,
    )


def test_runner_split_preserves_failure_event_order(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("review text without verdict\n")],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 1

    assert [_event_signature(event) for event in store.events] == [
        ("iteration", None, None, None),
        ("sha", None, None, "sha"),
        ("note", None, None, "agents_resolved"),
        ("note", "I1-T1", None, "model_routing_resolved"),
        ("note", "I1-T1", None, "model_routing_unknown_risk"),
        ("task_transition", "I1-T1", None, "IN_PROGRESS"),
        ("task_meta", "I1-T1", None, None),
        ("note", "I1-T1", None, "pull_ff_only_failed"),
        ("impl_attempt", "I1-T1", "IMPL", "end"),
        ("task_meta", "I1-T1", None, None),
        ("review", "I1-T1", "REVIEW", "MALFORMED"),
        ("task_transition", "I1-T1", None, "STOPPED:REVIEW_MALFORMED"),
        ("task_transition", "I1-T2", None, "BLOCKED_UPSTREAM"),
        ("note", None, None, "resume_skipped_tasks"),
        ("iteration", None, None, "PARTIAL"),
        ("note", None, None, "orch_workdir_preserved"),
    ]
