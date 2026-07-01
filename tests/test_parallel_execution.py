"""Parallel-runner characterization tests before C2 extraction."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.agents.base import AgentResult
from orch.hooks import HookDispatcher, HookResult
from tests.test_runner import (
    FakeAdapter,
    FakeHookHandler,
    _enable_parallel_config,
    _enable_parallel_tasks,
    _make_runner,
    _parallel_edit_from_prompt,
    _parallel_edit_or_fail_b,
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


def _parallel_runner(
    repo: Path,
    *,
    impl_script: list,
    review_script: list,
    hook_dispatcher: HookDispatcher | None = None,
):
    impl = FakeAdapter(name="claude", family="anthropic", script=impl_script)
    rev = FakeAdapter(name="codex", family="openai", script=review_script)
    return _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        hook_dispatcher=hook_dispatcher,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
    )


PARALLEL_HAPPY_EVENT_LOG = [
    ("iteration", None, None, None),
    ("sha", None, None, "sha"),
    ("note", None, None, "agents_resolved"),
    ("note", None, None, "pull_ff_only_failed"),
    ("note", None, None, "parallel_wave_started"),
    ("note", "I1-T1", None, "model_routing_resolved"),
    ("note", "I1-T1", None, "model_routing_unknown_risk"),
    ("task_transition", "I1-T1", None, "IN_PROGRESS"),
    ("task_meta", "I1-T1", None, None),
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
    ("impl_attempt", "I1-T2", "IMPL", "end"),
    ("task_meta", "I1-T2", None, None),
    ("review", "I1-T2", "REVIEW", "PASS"),
    ("confidence", "I1-T2", "REVIEW", None),
    ("triage", "I1-T2", "TRIAGE", "PASS"),
    ("note", "I1-T2", None, "parallel_branch_refreshed"),
    ("sha", None, None, "sha"),
    ("task_transition", "I1-T2", None, "DONE"),
    ("note", "I1-T2", None, "tasks_md_done_push_failed"),
    ("note", None, None, "parallel_wave_finished"),
    ("note", None, None, "final_scope_gate_passed"),
    ("note", None, None, "final_nav_discoverability_gate_passed"),
    ("iteration", None, None, "READY"),
]


def test_parallel_runner_preserves_happy_wave_event_log(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    runner, store, _ = _parallel_runner(
        repo,
        impl_script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
        review_script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )

    assert runner.run() == 0

    assert [_event_signature(event) for event in store.events] == [
        *PARALLEL_HAPPY_EVENT_LOG
    ]


def test_parallel_review_artifacts_are_task_id_keyed_for_same_round(
    repo: Path,
):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)

    def review_with_task_marker(adapter, prompt, workdir):
        if "I1-T1" in prompt:
            marker = "reviewed I1-T1"
        elif "I1-T2" in prompt:
            marker = "reviewed I1-T2"
        else:
            raise AssertionError(f"unexpected review prompt: {prompt}")
        return AgentResult(
            exit_code=0,
            stdout=f"{marker}\nVerdict: PASS\n",
            stderr="",
            duration_s=0.1,
            input_tokens=30,
            output_tokens=10,
            tokens_exact=False,
        )

    runner, store, _ = _parallel_runner(
        repo,
        impl_script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
        review_script=[review_with_task_marker, review_with_task_marker],
    )

    assert runner.run() == 0
    assert store.tasks["I1-T1"].status == "DONE"
    assert store.tasks["I1-T2"].status == "DONE"
    reviews_dir = repo / "tools" / "logs" / "demo-i1" / "reviews"
    t1_review = reviews_dir / "review_I1-T1_r1.md"
    t2_review = reviews_dir / "review_I1-T2_r1.md"
    assert t1_review.read_text() == "reviewed I1-T1\nVerdict: PASS\n"
    assert t2_review.read_text() == "reviewed I1-T2\nVerdict: PASS\n"


def test_parallel_runner_preserves_wave_member_failure_event_order(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo, include_third=True)
    runner, store, _ = _parallel_runner(
        repo,
        impl_script=[_parallel_edit_or_fail_b, _parallel_edit_or_fail_b],
        review_script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )

    assert runner.run() == 1

    assert [_event_signature(event) for event in store.events] == [
        ("iteration", None, None, None),
        ("sha", None, None, "sha"),
        ("note", None, None, "agents_resolved"),
        ("note", None, None, "pull_ff_only_failed"),
        ("note", None, None, "parallel_wave_started"),
        ("note", "I1-T1", None, "model_routing_resolved"),
        ("note", "I1-T1", None, "model_routing_unknown_risk"),
        ("task_transition", "I1-T1", None, "IN_PROGRESS"),
        ("task_meta", "I1-T1", None, None),
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
        ("impl_attempt", "I1-T2", "IMPL", "end"),
        ("note", "I1-T2", None, None),
        ("task_transition", "I1-T2", None, "STOPPED:IMPL_FAILED"),
        ("note", None, None, "parallel_wave_finished"),
        ("note", None, None, "resume_skipped_tasks"),
        ("iteration", None, None, "PARTIAL"),
        ("note", None, None, "orch_workdir_preserved"),
    ]
    assert "I1-T3" not in store.tasks


def test_parallel_runner_preserves_second_member_refresh_event_log(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    runner, store, _ = _parallel_runner(
        repo,
        impl_script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
        review_script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )

    assert runner.run() == 0

    assert [_event_signature(event) for event in store.events] == [
        *PARALLEL_HAPPY_EVENT_LOG
    ]
    assert [
        event["task"]
        for event in store.events
        if event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "parallel_branch_refreshed"
    ] == ["I1-T2"]


def test_parallel_runner_preserves_branch_prepare_hook_veto_event_log(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)

    class _VetoT1(FakeHookHandler):
        def handle(self, context):
            self.contexts.append(context)
            if context.task_id == "I1-T1":
                return HookResult.veto(reason="policy", message="t1 blocked")
            return HookResult.ok()

    runner, store, _ = _parallel_runner(
        repo,
        impl_script=[_parallel_edit_from_prompt],
        review_script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
        hook_dispatcher=HookDispatcher([
            _VetoT1(event_name="task.before_branch_prepare")
        ]),
    )

    assert runner.run() == 1

    assert [_event_signature(event) for event in store.events] == [
        ("iteration", None, None, None),
        ("sha", None, None, "sha"),
        ("note", None, None, "agents_resolved"),
        ("hook", "I1-T1", "HOOK", "task.before_branch_prepare"),
        ("note", "I1-T1", None, "hook_veto"),
        ("task_transition", "I1-T1", None, "STOPPED:HOOK_VETO"),
        ("hook", "I1-T2", "HOOK", "task.before_branch_prepare"),
        ("note", None, None, "pull_ff_only_failed"),
        ("note", None, None, "parallel_wave_started"),
        ("note", "I1-T2", None, "model_routing_resolved"),
        ("note", "I1-T2", None, "model_routing_unknown_risk"),
        ("hook", "I1-T2", "HOOK", "task.before_start"),
        ("task_transition", "I1-T2", None, "IN_PROGRESS"),
        ("task_meta", "I1-T2", None, None),
        ("hook", "I1-T2", "HOOK", "task.before_implement"),
        ("impl_attempt", "I1-T2", "IMPL", "end"),
        ("task_meta", "I1-T2", None, None),
        ("hook", "I1-T2", "HOOK", "task.before_review"),
        ("review", "I1-T2", "REVIEW", "PASS"),
        ("confidence", "I1-T2", "REVIEW", None),
        ("triage", "I1-T2", "TRIAGE", "PASS"),
        ("sha", None, None, "sha"),
        ("task_transition", "I1-T2", None, "DONE"),
        ("note", "I1-T2", None, "tasks_md_done_push_failed"),
        ("note", None, None, "parallel_wave_finished"),
        ("note", None, None, "resume_skipped_tasks"),
        ("iteration", None, None, "PARTIAL"),
        ("note", None, None, "orch_workdir_preserved"),
    ]
