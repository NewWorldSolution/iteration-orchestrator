"""Tests for orch.state."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch.state import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_WAITING,
    STATUS_STOPPED_PREFIX,
    StateStore,
    TaskState,
)
from orch.hooks import HookVeto


def _store(tmp_path: Path, **kwargs) -> StateStore:
    return StateStore(
        log_dir=tmp_path,
        iteration=kwargs.get("iteration", "demo-i1"),
        iter_branch=kwargs.get("iter_branch", "demo/iteration-1"),
        hook_dispatcher=kwargs.get("hook_dispatcher"),
    )


class _RecordingDispatcher:
    def __init__(self, *, veto: bool = False) -> None:
        self.contexts = []
        self.incidents = []
        self.veto = veto

    def dispatch(self, context, emit_internal):
        self.contexts.append(context)
        if self.veto:
            emit_internal(
                {
                    "event": "hook_veto",
                    "hook_event": context.event_name,
                    "handler": "recording",
                    "required": True,
                    "blocking": context.blocking,
                    "reason": "test",
                    "msg": "test veto",
                    "metadata": {},
                }
            )
            raise HookVeto(
                handler="recording",
                event_name=context.event_name,
                reason="test",
                message="test veto",
            )


class _MutatingDispatcher:
    def __init__(self) -> None:
        self.contexts = []

    def dispatch(self, context, emit_internal):
        self.contexts.append(context)
        context.event["meta"]["extra"] = "mutated"
        context.payload["files"].append("mutated.py")


def test_new_store_has_no_events(tmp_path: Path):
    s = _store(tmp_path)
    assert s.events == []
    assert s.tasks == {}
    assert not s.exists()


def test_append_event_persists_atomically(tmp_path: Path):
    s = _store(tmp_path)
    s.mark_iteration_started()
    assert s.exists()
    raw = json.loads((tmp_path / "run_state.json").read_text())
    assert raw["iteration"] == "demo-i1"
    assert raw["iter_branch"] == "demo/iteration-1"
    assert len(raw["events"]) == 1
    assert raw["events"][0]["kind"] == "iteration"
    assert raw["started_at"] is not None


def test_task_transition_recomputes_snapshot(tmp_path: Path):
    s = _store(tmp_path)
    s.task_transition("I4-T1", STATUS_IN_PROGRESS)
    s.task_transition("I4-T1", STATUS_DONE)
    assert s.tasks["I4-T1"].status == STATUS_DONE
    # Snapshot derived: last write wins via replay
    assert s.tasks["I4-T1"].stop_reason is None


def test_task_transition_with_stop(tmp_path: Path):
    s = _store(tmp_path)
    s.task_transition(
        "I4-T2",
        STATUS_STOPPED_PREFIX + "CHECKS",
        reason="CHECKS",
        msg="pytest failed after 3 fix rounds",
    )
    t = s.tasks["I4-T2"]
    assert t.status == STATUS_STOPPED_PREFIX + "CHECKS"
    assert t.stop_reason == "CHECKS"
    assert "pytest failed" in t.stop_msg


def test_impl_and_fix_counters(tmp_path: Path):
    s = _store(tmp_path)
    for _ in range(2):
        s.record_impl_attempt_end(
            "I4-T1", agent="claude", exit_code=0, duration_s=120.0
        )
    for _ in range(3):
        s.record_fix_round_end(
            "I4-T1", cause="acceptance",
            agent="claude", exit_code=0, duration_s=60.0,
        )
    s.record_fix_round_end(
        "I4-T1", cause="review",
        agent="claude", exit_code=0, duration_s=60.0,
    )
    t = s.tasks["I4-T1"]
    assert t.impl_attempts == 2
    assert t.fix_rounds == 3
    assert t.review_fix_rounds == 1


def test_review_result(tmp_path: Path):
    s = _store(tmp_path)
    s.record_review_result(
        "I4-T1", reviewer="codex", verdict="PASS", round_num=1
    )
    t = s.tasks["I4-T1"]
    assert t.verdict == "PASS"
    assert t.reviewer == "codex"


def test_pr_and_merge(tmp_path: Path):
    s = _store(tmp_path)
    s.record_pr("I4-T1", "https://github.com/x/y/pull/1")
    s.record_merge("I4-T1", auto_merged=True, merge_sha="deadbeef")
    t = s.tasks["I4-T1"]
    assert t.pr_url == "https://github.com/x/y/pull/1"
    assert t.auto_merged is True
    assert t.merge_sha == "deadbeef"


def test_merge_journal_events_are_audit_only(tmp_path: Path):
    s = _store(tmp_path)
    s.record_merge_intent(
        "I4-T1",
        target_branch="demo/iteration-1",
        target_sha_before="base",
        task_branch="demo/i1/t1",
        task_sha="task",
        message="Merge demo/i1/t1",
        mode="no_ci_local",
    )
    s.record_merge_complete(
        "I4-T1",
        target_branch="demo/iteration-1",
        merge_sha="merge",
        target_sha_before="base",
        task_branch="demo/i1/t1",
        task_sha="task",
        mode="no_ci_local",
    )

    assert [event["kind"] for event in s.events] == [
        "merge_intent",
        "merge_complete",
    ]
    assert s.tasks["I4-T1"] == TaskState(id="I4-T1")


def test_task_meta_merges_fields(tmp_path: Path):
    s = _store(tmp_path)
    s.task_meta(
        "I4-T1",
        branch="demo/i1/t1-x",
        implementer="claude",
        diff_insertions=42,
    )
    t = s.tasks["I4-T1"]
    assert t.branch == "demo/i1/t1-x"
    assert t.implementer == "claude"
    assert t.diff_insertions == 42


def test_task_meta_ignores_unknown_fields(tmp_path: Path):
    s = _store(tmp_path)
    s.task_meta("I4-T1", nonsense="ignored", branch="ok")
    assert s.tasks["I4-T1"].branch == "ok"
    assert not hasattr(s.tasks["I4-T1"], "nonsense")


def test_set_iter_branch_sha(tmp_path: Path):
    s = _store(tmp_path)
    s.set_iter_branch_sha("abc123")
    assert s.snapshot.iter_branch_sha == "abc123"
    s.set_iter_branch_sha("def456")
    assert s.snapshot.iter_branch_sha == "def456"


def test_reload_rebuilds_snapshot_from_events(tmp_path: Path):
    s = _store(tmp_path)
    s.mark_iteration_started()
    s.task_transition("I4-T1", STATUS_IN_PROGRESS)
    s.record_impl_attempt_end("I4-T1", agent="claude", exit_code=0, duration_s=1)
    s.task_transition("I4-T1", STATUS_DONE)

    # Independently load a fresh store from the same file
    s2 = _store(tmp_path)
    s2.load()
    assert s2.snapshot.started_at is not None
    assert s2.tasks["I4-T1"].status == STATUS_DONE
    assert s2.tasks["I4-T1"].impl_attempts == 1
    assert len(s2.events) == 4


def test_load_mismatched_iteration_raises(tmp_path: Path):
    s = _store(tmp_path)
    s.mark_iteration_started()
    other = _store(tmp_path, iteration="other-iter")
    with pytest.raises(ValueError, match="does not match"):
        other.load()


def test_blocked_upstream_is_valid_status(tmp_path: Path):
    s = _store(tmp_path)
    s.task_transition("I4-T3", STATUS_BLOCKED_UPSTREAM)
    assert s.tasks["I4-T3"].status == STATUS_BLOCKED_UPSTREAM


def test_retry_reset_clears_recovery_fields_and_counters(tmp_path: Path):
    s = _store(tmp_path)
    s.task_transition("I4-T1", STATUS_IN_PROGRESS)
    s.record_impl_attempt_end(
        "I4-T1", agent="codex", exit_code=1, duration_s=10.0
    )
    s.record_fix_round_end(
        "I4-T1", cause="acceptance", agent="codex", exit_code=1,
        duration_s=20.0,
    )
    s.record_fix_round_end(
        "I4-T1", cause="review", agent="codex", exit_code=1,
        duration_s=30.0,
    )
    s.record_review_result(
        "I4-T1", reviewer="claude", verdict="CHANGES REQUIRED",
        round_num=1,
    )
    s.record_pr("I4-T1", "https://example.com/pr/1")
    s.record_merge("I4-T1", auto_merged=True, merge_sha="abc123")
    s.append_triage_decision(
        "I4-T1",
        action="defer",
        reason="needs operator",
        round_num=1,
        verdict="CHANGES REQUIRED",
        severity="should-fix",
        increments_defer_budget=True,
        confidence_history=[0.2, None],
    )
    s.record_confidence("I4-T1", round_num=1, value=0.4)
    s.task_transition(
        "I4-T1",
        STATUS_STOPPED_PREFIX + "REVIEW_FAIL",
        reason="REVIEW_FAIL",
        msg="review failed",
    )

    before = s.tasks["I4-T1"]
    assert before.impl_attempts == 1
    assert before.fix_rounds == 1
    assert before.review_fix_rounds == 1
    assert before.pr_url == "https://example.com/pr/1"
    assert before.verdict == "CHANGES REQUIRED"
    assert before.auto_merged is True
    assert before.merge_sha == "abc123"
    assert before.stop_reason == "REVIEW_FAIL"
    assert before.stop_msg == "review failed"
    assert before.triage_decisions
    assert before.defer_budget_used == 1
    assert before.confidence_history == [0.4]
    assert before.triage_outcome == "defer"

    s.reset_task("I4-T1")

    after = s.tasks["I4-T1"]
    assert after.status == STATUS_WAITING
    assert after.impl_attempts == 0
    assert after.fix_rounds == 0
    assert after.review_fix_rounds == 0
    # B-14 R4: pr_url is preserved across reset (the prior-PR reference must
    # survive so external-merge reconciliation can still find it); only the
    # outcome fields are cleared.
    assert after.pr_url == "https://example.com/pr/1"
    assert after.verdict is None
    assert after.auto_merged is False
    assert after.merge_sha is None
    assert after.stop_reason is None
    assert after.stop_msg is None
    assert after.triage_decisions == []
    assert after.defer_budget_used == 0
    assert after.confidence_history == []
    assert after.triage_outcome is None


def test_events_are_append_only_on_save(tmp_path: Path):
    s = _store(tmp_path)
    s.task_transition("I4-T1", STATUS_IN_PROGRESS)
    n = len(s.events)
    # Save again without adding events; event count unchanged
    s.save()
    raw = json.loads((tmp_path / "run_state.json").read_text())
    assert len(raw["events"]) == n


def test_unknown_event_kind_is_ignored_on_replay(tmp_path: Path):
    s = _store(tmp_path)
    # Hand-craft an unknown-kind event via the low-level API
    s.append_event(kind="exotic_new_kind", task="I4-T1", meta={"anything": 1})
    # Should not raise and should not create a phantom TaskState change
    assert s.tasks["I4-T1"] == TaskState(id="I4-T1")


def test_append_event_dispatches_after_persisting_snapshot(tmp_path: Path):
    dispatcher = _RecordingDispatcher()
    s = _store(tmp_path, hook_dispatcher=dispatcher)

    ev = s.append_event(
        kind="hook",
        task="I4-T1",
        meta={"event": "task.before_start", "extra": "value"},
        hook_blocking=True,
    )

    assert dispatcher.contexts[0].event == ev
    assert dispatcher.contexts[0].event_name == "task.before_start"
    assert dispatcher.contexts[0].blocking is True
    assert dispatcher.contexts[0].payload["extra"] == "value"
    raw = json.loads((tmp_path / "run_state.json").read_text())
    assert raw["events"][0]["meta"]["event"] == "task.before_start"


def test_hook_context_mutation_does_not_change_persisted_event(tmp_path: Path):
    dispatcher = _MutatingDispatcher()
    s = _store(tmp_path, hook_dispatcher=dispatcher)

    ev = s.append_event(
        kind="hook",
        task="I4-T1",
        meta={"event": "task.before_start", "files": ["a.py"]},
        hook_blocking=True,
    )

    raw = json.loads((tmp_path / "run_state.json").read_text())
    assert ev["meta"] == {"event": "task.before_start", "files": ["a.py"]}
    assert raw["events"][0]["meta"] == {
        "event": "task.before_start",
        "files": ["a.py"],
    }


def test_internal_hook_incidents_do_not_redispatch(tmp_path: Path):
    dispatcher = _RecordingDispatcher(veto=True)
    s = _store(tmp_path, hook_dispatcher=dispatcher)

    with pytest.raises(HookVeto):
        s.append_event(
            kind="hook",
            task="I4-T1",
            meta={"event": "task.before_start"},
            hook_blocking=True,
        )

    raw = json.loads((tmp_path / "run_state.json").read_text())
    assert [event["meta"].get("event") for event in raw["events"]] == [
        "task.before_start",
        "hook_veto",
    ]
    assert len(dispatcher.contexts) == 1
