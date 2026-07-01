"""Characterization tests for runner run-loop behavior before T4 split."""
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


def test_runner_split_preserves_happy_path_event_log(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

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
        ("review", "I1-T1", "REVIEW", "PASS"),
        ("confidence", "I1-T1", "REVIEW", None),
        ("triage", "I1-T1", "TRIAGE", "PASS"),
        ("sha", None, None, "sha"),
        ("task_transition", "I1-T1", None, "DONE"),
        ("note", "I1-T1", None, "tasks_md_done_push_failed"),
        ("note", "I1-T2", None, "model_routing_resolved"),
        ("note", "I1-T2", None, "model_routing_unknown_risk"),
        ("task_transition", "I1-T2", None, "IN_PROGRESS"),
        ("task_meta", "I1-T2", None, None),
        ("note", "I1-T2", None, "pull_ff_only_failed"),
        ("impl_attempt", "I1-T2", "IMPL", "end"),
        ("task_meta", "I1-T2", None, None),
        ("review", "I1-T2", "REVIEW", "PASS"),
        ("confidence", "I1-T2", "REVIEW", None),
        ("triage", "I1-T2", "TRIAGE", "PASS"),
        ("sha", None, None, "sha"),
        ("task_transition", "I1-T2", None, "DONE"),
        ("note", "I1-T2", None, "tasks_md_done_push_failed"),
        ("note", None, None, "final_scope_gate_passed"),
        ("note", None, None, "final_nav_discoverability_gate_passed"),
        ("iteration", None, None, "READY"),
    ]
