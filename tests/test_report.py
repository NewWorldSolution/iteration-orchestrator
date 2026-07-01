"""Tests for orch.report."""
from __future__ import annotations

from pathlib import Path

from orch.cost import CostLogger
from orch.improvements import append_record
from orch.report import build_report
from orch.state import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_HUMAN_MERGE,
    STATUS_STOPPED_PREFIX,
    StateStore,
)


RATES = {"anthropic": {"input": 3.0, "output": 15.0}}
HIGH_RISK = ["**/schema.sql", "**/auth*.py"]


def _fresh_store(tmp_path: Path) -> StateStore:
    s = StateStore(log_dir=tmp_path, iteration="demo-i1", iter_branch="demo/iteration-1")
    s.mark_iteration_started()
    return s


def test_empty_iteration_report_is_in_progress(tmp_path: Path):
    s = _fresh_store(tmp_path)
    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl",
        high_risk_globs=HIGH_RISK,
    )
    assert "Readiness report" in out
    assert "demo-i1" in out
    assert "ITERATION IN PROGRESS" in out
    # No tasks → no Blockers or Risk sections
    assert "## Blockers" not in out
    assert "## Risk heat map" not in out


def test_all_done_says_ready_for_qa(tmp_path: Path):
    s = _fresh_store(tmp_path)
    for k in range(1, 4):
        tid = f"I4-T{k}"
        s.task_transition(tid, STATUS_IN_PROGRESS)
        s.record_impl_attempt_end(tid, agent="claude", exit_code=0, duration_s=10)
        s.record_review_result(tid, reviewer="codex", verdict="PASS", round_num=1)
        s.record_merge(tid, auto_merged=True, merge_sha="abc")
        s.task_transition(tid, STATUS_DONE)
    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )
    assert "ITERATION READY FOR QA" in out
    for tid in ("I4-T1", "I4-T2", "I4-T3"):
        assert tid in out


def test_blockers_section_lists_stops_and_needs_human(tmp_path: Path):
    s = _fresh_store(tmp_path)
    s.task_transition("I4-T1", STATUS_STOPPED_PREFIX + "CHECKS",
                      reason="CHECKS", msg="pytest failed after 3 rounds")
    s.task_transition("I4-T2", STATUS_BLOCKED_UPSTREAM)
    s.record_pr("I4-T3", "https://example.com/pr/9")
    s.task_transition("I4-T3", STATUS_NEEDS_HUMAN_MERGE)
    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )
    assert "ITERATION PARTIAL — 3 blocker(s)" in out
    assert "## Blockers" in out
    assert "**I4-T1** STOPPED (CHECKS) — pytest failed after 3 rounds" in out
    assert "Recovery note: inspect the acceptance command output" in out
    assert "**I4-T2** BLOCKED_UPSTREAM" in out
    assert "unblock the upstream task first" in out
    assert "**I4-T3** NEEDS_HUMAN_MERGE" in out
    assert "resume with `--accept-external`" in out
    assert "https://example.com/pr/9" in out


def test_blockers_section_explains_hook_veto(tmp_path: Path):
    s = _fresh_store(tmp_path)
    s.task_transition(
        "I4-T1",
        STATUS_STOPPED_PREFIX + "HOOK_VETO",
        reason="HOOK_VETO",
        msg="hook veto at task.before_pr from policy: blocked",
    )

    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )

    assert "**I4-T1** STOPPED (HOOK_VETO)" in out
    assert "required pre-action hook vetoed the task" in out


def test_report_explains_dual_review_stop(tmp_path: Path):
    s = _fresh_store(tmp_path)
    s.task_transition(
        "I4-T1",
        STATUS_STOPPED_PREFIX + "DUAL_REVIEW_REQUIRED",
        reason="DUAL_REVIEW_REQUIRED",
        msg=(
            "dual-model agreement required for "
            "risk_category=architecture_core_logic, but no secondary "
            "reviewer is configured"
        ),
    )

    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )

    assert "**I4-T1** STOPPED (DUAL_REVIEW_REQUIRED)" in out
    assert "dual-model agreement required" in out
    assert "configure `--secondary-reviewer <agent>`" in out


def test_report_shows_dual_review_evidence(tmp_path: Path):
    s = _fresh_store(tmp_path)
    s.append_event(
        kind="note",
        task="I4-T1",
        meta={
            "event": "dual_review_passed",
            "primary_reviewer": "codex",
            "secondary_reviewer": "claude",
            "risk_category": "architecture_core_logic",
            "artifact": "tools/logs/demo-i1/reviews/dual_review_I4-T1_claude.md",
            "verdict": "PASS",
        },
    )

    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )

    assert "## Dual-model review" in out
    assert "**I4-T1** passed" in out
    assert "primary `codex`, secondary `claude`" in out
    assert "dual_review_I4-T1_claude.md" in out


def test_risk_heatmap_lists_high_risk_files(tmp_path: Path):
    s = _fresh_store(tmp_path)
    s.task_transition("I4-T1", STATUS_DONE)
    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl",
        high_risk_globs=HIGH_RISK,
        changed_files_by_task={
            "I4-T1": ["db/schema.sql", "app/auth_middleware.py", "app/routes.py"],
        },
    )
    assert "## Risk heat map" in out
    assert "db/schema.sql" in out
    assert "app/auth_middleware.py" in out
    # Non-high-risk file should not appear as a row in the heat map
    assert "| I4-T1 | app/routes.py" not in out


def test_cost_section_has_estimated_banner(tmp_path: Path):
    s = _fresh_store(tmp_path)
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    log.record(
        task="I4-T1", step="IMPL", agent="claude", family="anthropic",
        input_tokens=100_000, output_tokens=20_000, exact=True,
        duration_s=30, exit_code=0,
    )
    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )
    assert "## Cost (estimated)" in out
    assert "estimates using the rate table" in out
    assert "Invocations: 1" in out
    assert "Total: **$" in out


def test_report_shows_improvement_artifact_and_counts(tmp_path: Path):
    s = _fresh_store(tmp_path)
    append_record(
        tmp_path,
        {
            "id": "imp-001",
            "source_iteration": "demo-i1",
            "source_event": "retro.completed",
            "title": "Add fixture coverage",
            "problem": "Approved improvements need a durable control.",
            "classification": "quality",
            "impact": "medium",
            "effort": "small",
            "status": "approved",
            "control_mechanism": "fixture repo validation gate",
        },
    )

    out = build_report(
        state=s,
        cost_jsonl=tmp_path / "cost.jsonl",
        high_risk_globs=HIGH_RISK,
    )

    assert "## Six Sigma Improvements" in out
    assert "`tools/logs/demo-i1/improvements.jsonl`" in out
    assert "Validation: **OK**" in out
    assert "approved: 1" in out


def test_report_marks_timing_evidence_present(tmp_path: Path):
    s = _fresh_store(tmp_path)
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    log.record(
        task="I4-T1", step="IMPL", agent="claude", family="anthropic",
        input_tokens=100, output_tokens=50, exact=True,
        duration_s=360, exit_code=0,
    )
    (tmp_path / "timing.jsonl").write_text("{}\n", encoding="utf-8")

    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )

    assert "Timing evidence: **present**" in out
    assert "`tools/logs/demo-i1/timing.jsonl`" in out
    assert "Operator action:" not in out
    assert "Iteration **demo-i1** — wall time: **6m**" in out
    assert "Per-invocation breakdown:" in out
    assert "- I4-T1: 6min (claude-impl 6min)" in out


def test_report_marks_notes_timing_fallback(tmp_path: Path):
    s = _fresh_store(tmp_path)
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    log.record(
        task="I4-T1", step="IMPL", agent="claude", family="anthropic",
        input_tokens=100, output_tokens=50, exact=True,
        duration_s=360, exit_code=0,
    )
    (tmp_path / "notes.md").write_text(
        "Manual timing was captured outside orch.\n", encoding="utf-8"
    )

    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )

    assert "Timing evidence: **operator-provided fallback**" in out
    assert "`tools/logs/demo-i1/notes.md`" in out
    assert "Operator action:" not in out
    assert "Wall time: **operator fallback only**" in out
    assert "Iteration **demo-i1** — wall time:" not in out
    assert "Per-invocation breakdown:" in out


def test_report_marks_missing_timing_evidence(tmp_path: Path):
    s = _fresh_store(tmp_path)
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    log.record(
        task="I4-T1", step="IMPL", agent="claude", family="anthropic",
        input_tokens=100, output_tokens=50, exact=True,
        duration_s=360, exit_code=0,
    )

    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )

    assert "Timing evidence: **missing**" in out
    assert "`tools/logs/demo-i1/timing.jsonl`" in out
    assert "`tools/logs/demo-i1/notes.md`" in out
    assert "Operator action:" in out
    assert "Recovery note: for manual waits" in out
    assert "Wall time: **missing evidence**" in out
    assert "Iteration **demo-i1** — wall time:" not in out
    assert "Per-invocation breakdown:" in out


def test_artifact_paths_route_through_resolver(tmp_path: Path):
    s = _fresh_store(tmp_path)

    out = build_report(
        state=s,
        cost_jsonl=tmp_path / "cost.jsonl",
        high_risk_globs=HIGH_RISK,
        artifact_root_ref=".orch/artifacts",
    )

    assert "`.orch/artifacts/demo-i1/timing.jsonl`" in out
    assert "`.orch/artifacts/demo-i1/notes.md`" in out
    assert "`.orch/artifacts/demo-i1/improvements.jsonl`" in out
    assert "`tools/logs/demo-i1/timing.jsonl`" not in out


def test_report_handles_missing_cost_file(tmp_path: Path):
    s = _fresh_store(tmp_path)
    out = build_report(
        state=s, cost_jsonl=tmp_path / "nope.jsonl", high_risk_globs=HIGH_RISK,
    )
    assert "Invocations: 0" in out
    assert "Total: **$0.0000**" in out


def test_readiness_leads_with_walltime_per_task(tmp_path: Path):
    """Readiness report Cost section must lead with wall-time, not dollars.

    Wall-time is the operator's primary constraint; dollars are secondary.
    """
    from orch.cost import CostLogger
    from orch.state import STATUS_IN_PROGRESS, STATUS_DONE

    s = _fresh_store(tmp_path)
    cost_path = tmp_path / "cost.jsonl"

    # Seed one task with known duration.
    tid = "I7.5-T1"
    s.task_transition(tid, STATUS_IN_PROGRESS)
    logger = CostLogger(cost_path, RATES, "pre-i8-orch")
    logger.record(
        task=tid, step="IMPL", agent="codex", family="openai",
        input_tokens=100, output_tokens=200, exact=False,
        duration_s=300.0,   # 5.0 min
        exit_code=0, partial=False,
    )
    logger.record(
        task=tid, step="REVIEW", agent="claude", family="anthropic",
        input_tokens=150, output_tokens=100, exact=False,
        duration_s=120.0,   # 2.0 min
        exit_code=0, partial=False,
    )
    s.record_review_result(tid, reviewer="claude", verdict="PASS", round_num=1)
    s.record_merge(tid, auto_merged=True, merge_sha="abc123")
    s.task_transition(tid, STATUS_DONE)

    out = build_report(
        state=s,
        cost_jsonl=cost_path,
        high_risk_globs=HIGH_RISK,
    )

    # Wall-time must appear BEFORE dollar total in the output.
    wall_pos = out.find("Wall-time")
    dollar_pos = out.find("Total: **$")
    assert wall_pos != -1, "report must contain 'Wall-time' section"
    assert dollar_pos != -1, "report must still contain dollar total (as footnote)"
    assert wall_pos < dollar_pos, (
        f"Wall-time section (pos {wall_pos}) must precede dollar total (pos {dollar_pos})"
    )

    # Wall-time values must appear (5.0 min for IMPL, 2.0 min for REVIEW).
    assert "5.0" in out or "5.0 min" in out, "5.0 min wall-time not rendered"
    assert "7.0" in out or "7.0 min" in out, "7.0 min total wall-time not rendered"

    # Caveat must still be present.
    assert "verify against provider billing records" in out, (
        "cost caveat must remain"
    )


def test_walltime_renders_dash_for_tasks_without_records(tmp_path: Path):
    """Pending tasks (no cost records yet) must appear in the wall-time
    table with '-' cells per the T6 prompt contract."""
    from orch.state import STATUS_IN_PROGRESS

    s = _fresh_store(tmp_path)
    s.task_transition("I7.5-T1", STATUS_IN_PROGRESS)
    s.task_transition("I7.5-T2", STATUS_IN_PROGRESS)

    out = build_report(
        state=s,
        cost_jsonl=tmp_path / "cost.jsonl",  # no records on disk
        high_risk_globs=HIGH_RISK,
    )

    # Both pending tasks must appear as wall-time rows with '-' cells.
    assert "| I7.5-T1 | - | - | - | - |" in out, (
        "task with no records must render '-' cells in wall-time table"
    )
    assert "| I7.5-T2 | - | - | - | - |" in out, (
        "second pending task must also render '-' cells"
    )


def test_cost_dollar_lines_use_subscription_equivalent_label(tmp_path: Path):
    from orch.cost import HONEST_COST_LABEL

    s = _fresh_store(tmp_path)
    rates = {
        "anthropic": {
            "input": 3.0,
            "output": 15.0,
            "models": {"claude-sonnet-4-5": {"input": 3.0, "output": 15.0}},
        },
    }
    log = CostLogger(tmp_path / "cost.jsonl", rates, iteration="demo-i1")
    log.record(
        task="I4-T1",
        step="IMPL",
        agent="claude",
        family="anthropic",
        provider="claude",
        model="claude-sonnet-4-5",
        input_tokens=100_000,
        output_tokens=20_000,
        exact=True,
        duration_s=30,
        exit_code=0,
    )

    out = build_report(
        state=s, cost_jsonl=tmp_path / "cost.jsonl", high_risk_globs=HIGH_RISK,
    )

    dollar_lines = [line for line in out.splitlines() if "$" in line]
    assert dollar_lines
    assert all(HONEST_COST_LABEL in line for line in dollar_lines)
    assert "By provider/model:" in out
    assert "By agent:" in out
