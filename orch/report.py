"""Readiness report renderer.

Pure function over :class:`StateStore` + ``cost.jsonl`` — no git calls, no
model calls. The operator runs ``orch report <iter>`` at any point; at
end-of-run the orchestrator writes this same text to
``tools/logs/<iter>/readiness.md``.

Output is Markdown. Every cost figure is rendered with an
estimated banner so downstream readers don't mistake it for authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from orch.cost import (
    HONEST_COST_LABEL,
    CostSummary,
    load_records,
    summarize_costs,
)
from orch.improvements import (
    IMPROVEMENTS_FILENAME,
    ImprovementValidationError,
    improvements_path,
    read_records,
    status_counts,
)
from orch.recovery import (
    notes_timing_recovery_note,
    stop_reason_recovery_note,
)
from orch.state import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_NEEDS_HUMAN_MERGE,
    STATUS_STOPPED_PREFIX,
    StateStore,
    TaskState,
)
from orch.timing import TimingEvidence, detect_evidence


@dataclass
class ReportContext:
    state: StateStore
    cost: CostSummary
    high_risk_globs: list[str]
    changed_files_by_task: dict[str, list[str]]


def _verdict_label(t: TaskState) -> str:
    if t.verdict:
        return t.verdict
    if t.status == STATUS_DONE:
        return "PASS"
    return "-"


def _task_row(t: TaskState) -> str:
    auto = "yes" if t.auto_merged else ("no" if t.pr_url else "-")
    return (
        f"| {t.id} | {t.status} | {t.implementer or '-'} | {t.reviewer or '-'} "
        f"| {t.impl_attempts} | {t.fix_rounds}/{t.review_fix_rounds} "
        f"| {_verdict_label(t)} | {t.diff_insertions} | {auto} |"
    )


def _blocker_section(state: StateStore) -> list[str]:
    lines: list[str] = []
    stopped = [
        t for t in state.tasks.values()
        if t.status.startswith(STATUS_STOPPED_PREFIX)
    ]
    blocked_up = [
        t for t in state.tasks.values()
        if t.status == STATUS_BLOCKED_UPSTREAM
    ]
    needs_human = [
        t for t in state.tasks.values()
        if t.status == STATUS_NEEDS_HUMAN_MERGE
    ]
    if not (stopped or blocked_up or needs_human):
        return lines

    lines.append("## Blockers")
    lines.append("")
    iteration = state.snapshot.iteration
    for t in stopped:
        reason = t.stop_reason or "?"
        msg = f" — {t.stop_msg}" if t.stop_msg else ""
        lines.append(f"- **{t.id}** STOPPED ({reason}){msg}")
        recovery = stop_reason_recovery_note(reason, iteration=iteration)
        if recovery:
            lines.append(f"  - {recovery}")
    for t in blocked_up:
        lines.append(f"- **{t.id}** BLOCKED_UPSTREAM")
        lines.append(
            "  - Recovery note: unblock the upstream task first; do not "
            "edit downstream run state manually."
        )
    for t in needs_human:
        pr = f" — {t.pr_url}" if t.pr_url else ""
        lines.append(f"- **{t.id}** NEEDS_HUMAN_MERGE{pr}")
        lines.append(
            "  - Recovery note: inspect the PR and CI state, merge manually "
            "only when safe, then resume with `--accept-external`."
        )
    lines.append("")
    return lines


def _risk_heatmap(
    changed_files_by_task: dict[str, list[str]], high_risk_globs: list[str]
) -> list[str]:
    from fnmatch import fnmatch
    hits: list[tuple[str, str, str]] = []
    for tid, files in changed_files_by_task.items():
        for f in files:
            for g in high_risk_globs:
                if fnmatch(f, g):
                    hits.append((tid, f, g))
                    break
    if not hits:
        return []
    lines = ["## Risk heat map", "",
             "| Task | File | Glob |", "|------|------|------|"]
    for tid, f, g in hits:
        lines.append(f"| {tid} | {f} | {g} |")
    lines.append("")
    return lines


def _dual_review_section(state: StateStore) -> list[str]:
    events = [
        e for e in state.events
        if e.get("kind") == "note"
        and str(e.get("meta", {}).get("event", "")).startswith("dual_review_")
    ]
    if not events:
        return []

    lines = ["## Dual-model review", ""]
    for e in events:
        meta = e.get("meta", {})
        task = e.get("task") or "-"
        status = str(meta.get("event", "")).replace("dual_review_", "")
        primary = meta.get("primary_reviewer") or "-"
        secondary = meta.get("secondary_reviewer") or "-"
        risk = meta.get("risk_category") or "-"
        detail = meta.get("verdict") or meta.get("msg") or "-"
        artifact = meta.get("artifact")
        suffix = f" — `{artifact}`" if artifact else ""
        lines.append(
            f"- **{task}** {status}: primary `{primary}`, "
            f"secondary `{secondary}`, risk `{risk}`, result {detail}{suffix}"
        )
    lines.append("")
    return lines


def _handoff(state: StateStore) -> str:
    stopped = any(
        t.status.startswith(STATUS_STOPPED_PREFIX) for t in state.tasks.values()
    )
    needs_human = any(
        t.status == STATUS_NEEDS_HUMAN_MERGE for t in state.tasks.values()
    )
    blocked_up = any(
        t.status == STATUS_BLOCKED_UPSTREAM for t in state.tasks.values()
    )
    all_done = state.tasks and all(
        t.status == STATUS_DONE for t in state.tasks.values()
    )
    if all_done:
        return "**ITERATION READY FOR QA**"
    blockers = sum(
        1 for t in state.tasks.values()
        if t.status.startswith(STATUS_STOPPED_PREFIX)
        or t.status == STATUS_NEEDS_HUMAN_MERGE
        or t.status == STATUS_BLOCKED_UPSTREAM
    )
    if stopped or needs_human or blocked_up:
        return f"**ITERATION PARTIAL — {blockers} blocker(s)**"
    return "**ITERATION IN PROGRESS**"


def _timing_artifact(
    iteration: str,
    name: str,
    *,
    artifact_root_ref: str = "tools/logs",
) -> str:
    return f"{artifact_root_ref.rstrip('/')}/{iteration}/{name}"


def _timing_evidence_status_lines(
    evidence: TimingEvidence,
    iteration: str,
    *,
    artifact_root_ref: str = "tools/logs",
) -> list[str]:
    timing_ref = _timing_artifact(
        iteration, "timing.jsonl", artifact_root_ref=artifact_root_ref
    )
    notes_ref = _timing_artifact(
        iteration, "notes.md", artifact_root_ref=artifact_root_ref
    )
    if evidence.has_timing_log:
        return [
            f"Timing evidence: **present** — measured timing log found at "
            f"`{timing_ref}`.",
        ]
    if evidence.has_notes_fallback:
        return [
            "Timing evidence: **operator-provided fallback** — no "
            f"`{timing_ref}` was found; using fallback notes at "
            f"`{notes_ref}`.",
            "Treat fallback timing as operator-provided context, not measured "
            "`orch timing` evidence.",
        ]
    return [
        "Timing evidence: **missing** — no "
        f"`{timing_ref}` or `{notes_ref}` found.",
        "Operator action: run "
        f"`python -m orch timing --iter {iteration} start <label>` and "
        f"`python -m orch timing --iter {iteration} end <label>` "
        f"during the run, or add `{notes_ref}` explaining the missing/manual "
        "timing before claiming wall-time evidence.",
        notes_timing_recovery_note(iteration, artifact_root_ref),
    ]


def _improvements_section(
    log_dir: Path,
    iteration: str,
    *,
    artifact_root_ref: str = "tools/logs",
) -> list[str]:
    artifact = improvements_path(log_dir)
    artifact_ref = _timing_artifact(
        iteration,
        IMPROVEMENTS_FILENAME,
        artifact_root_ref=artifact_root_ref,
    )
    lines = ["## Six Sigma Improvements", ""]
    lines.append(f"- Artifact: `{artifact_ref}`")
    lines.append(
        "- Control gate: `approved` and `implemented` records require a "
        "non-empty `control_mechanism`."
    )
    try:
        records = read_records(artifact)
    except ImprovementValidationError as exc:
        lines.append(f"- Validation: **FAILED** — {exc}")
        lines.append("")
        return lines

    counts = status_counts(records)
    if counts:
        counts_text = ", ".join(
            f"{status}: {count}" for status, count in counts.items()
        )
    else:
        counts_text = "none"
    lines.append("- Validation: **OK**")
    lines.append(f"- Records: {len(records)} ({counts_text})")
    lines.append("")
    return lines


def render_report(
    ctx: ReportContext,
    cost_jsonl: Path | None = None,
    *,
    artifact_root_ref: str = "tools/logs",
) -> str:
    state = ctx.state
    cost = ctx.cost
    snap = state.snapshot

    lines: list[str] = []
    lines.append(f"# Readiness report — {snap.iteration}")
    lines.append("")
    lines.append(f"- Iteration branch: `{snap.iter_branch}`")
    if snap.iter_branch_sha:
        lines.append(f"- Iteration branch HEAD: `{snap.iter_branch_sha}`")
    lines.append(f"- Started: {snap.started_at or '-'}")
    lines.append(f"- Finished: {snap.finished_at or '-'}")
    lines.append("")
    lines.append(_handoff(state))
    lines.append("")

    # Tasks table
    lines.append("## Tasks")
    lines.append("")
    lines.append(
        "| ID | Status | Impl | Reviewer | Attempts | "
        "Fix (acc/rev) | Verdict | +Ins | Auto-merged |"
    )
    lines.append(
        "|----|--------|------|----------|----------|---------------|"
        "---------|------|-------------|"
    )
    for tid in sorted(state.tasks):
        lines.append(_task_row(state.tasks[tid]))
    lines.append("")

    lines.extend(_blocker_section(state))
    lines.extend(_dual_review_section(state))
    lines.extend(_risk_heatmap(ctx.changed_files_by_task, ctx.high_risk_globs))

    # Wall-time is the headline (operator's primary constraint);
    # dollars are a secondary footnote.
    records = load_records(cost_jsonl) if cost_jsonl else []
    log_dir = cost_jsonl.parent if cost_jsonl else state.log_dir
    timing_evidence = detect_evidence(log_dir)

    walltime_by_task_step: dict[str, dict[str, float]] = {}
    walltime_by_agent: dict[str, float] = {}
    invocations_by_task: dict[str, list[tuple[str, str, float]]] = {}
    has_duration = False
    for r in records:
        dur = float(r.get("duration_s", 0.0))
        if dur > 0:
            has_duration = True
        tid = r.get("task", "?")
        step = r.get("step", "?")
        agent = r.get("agent", "?")
        walltime_by_task_step.setdefault(tid, {})
        walltime_by_task_step[tid][step] = (
            walltime_by_task_step[tid].get(step, 0.0) + dur
        )
        walltime_by_agent[agent] = walltime_by_agent.get(agent, 0.0) + dur
        invocations_by_task.setdefault(tid, []).append((agent, step, dur))

    def _fmt_min(seconds: float) -> str:
        return f"{seconds / 60:.1f}" if seconds > 0 else "-"

    def _fmt_hm(seconds: float) -> str:
        total_min = int(round(seconds / 60))
        h, m = divmod(total_min, 60)
        if h:
            return f"{h}h {m}m"
        return f"{m}m"

    def _fmt_inv(seconds: float) -> str:
        # Match the operator-readable format (e.g. "218s", "13min").
        # Sub-5min
        # invocations stay in seconds so the per-tool ratio is visible.
        if seconds < 300:
            return f"{int(round(seconds))}s"
        return f"{int(round(seconds / 60))}min"

    _STEP_LABEL = {"IMPL": "impl", "FIX": "fix", "REVIEW": "review"}

    lines.append("## Wall-time (primary metric)")
    lines.append("")
    lines.extend(
        _timing_evidence_status_lines(
            timing_evidence,
            snap.iteration,
            artifact_root_ref=artifact_root_ref,
        )
    )
    lines.append("")
    grand_total_s = sum(
        sum(steps.values()) for steps in walltime_by_task_step.values()
    )
    if has_duration and timing_evidence.has_timing_log:
        lines.append(
            f"Iteration **{snap.iteration}** — wall time: "
            f"**{_fmt_hm(grand_total_s)}**"
        )
    elif timing_evidence.has_notes_fallback:
        lines.append(
            "Wall time: **operator fallback only** — `timing.jsonl` is "
            "absent, so cost-log durations below are context rather than "
            "measured timing evidence."
        )
    elif timing_evidence.is_missing:
        lines.append(
            "Wall time: **missing evidence** — unknown because `orch timing` "
            "was not used; cost-log durations below are not treated as "
            "measured wall-time evidence without `timing.jsonl` or "
            "`notes.md`."
        )
    elif has_duration:
        lines.append(
            "Wall time: **unclassified timing evidence** — cost-log "
            "durations are shown below for context."
        )
    else:
        lines.append(
            "Wall time: **unknown** — timing evidence exists, but "
            "`cost.jsonl` has no positive `duration_s` records, so "
            "per-step durations are unavailable."
        )
    lines.append("")
    # Always render the per-task table — pending tasks (no records yet)
    # appear with `-` cells per the T6 prompt contract; without a `-`
    # row they would silently disappear from the report.
    lines.append(
        "| Task | Impl (min) | Fix (min) | Review (min) | Total (min) |"
    )
    lines.append(
        "|------|-----------|-----------|--------------|-------------|"
    )
    all_task_ids = sorted(set(state.tasks) | set(walltime_by_task_step))
    for tid in all_task_ids:
        by_step = walltime_by_task_step.get(tid, {})
        impl_s = by_step.get("IMPL", 0.0)
        fix_s = by_step.get("FIX", 0.0)
        review_s = by_step.get("REVIEW", 0.0)
        task_total_s = sum(by_step.values())
        total_cell = (
            f"{task_total_s / 60:.1f}" if task_total_s > 0 else "-"
        )
        lines.append(
            f"| {tid} | {_fmt_min(impl_s)} | {_fmt_min(fix_s)} "
            f"| {_fmt_min(review_s)} | {total_cell} |"
        )
    if has_duration:
        lines.append(
            f"| **Total** | | | | {grand_total_s / 60:.1f} |"
        )
    lines.append("")
    if invocations_by_task:
        lines.append("Per-invocation breakdown:")
        # Render each task's invocations in the order they were logged so
        # the operator can see the impl/review/fix loop play out per-task,
        # in an operator-readable time format.
        for tid in sorted(invocations_by_task):
            invs = invocations_by_task[tid]
            task_total_s = sum(d for _, _, d in invs)
            chunks = [
                f"{agent}-{_STEP_LABEL.get(step, step.lower())} "
                f"{_fmt_inv(dur)}"
                for agent, step, dur in invs
            ]
            lines.append(
                f"- {tid}: {_fmt_inv(task_total_s)} "
                f"({' + '.join(chunks)})"
            )
        lines.append("")
    if walltime_by_agent:
        lines.append("Per tool:")
        for agent in sorted(walltime_by_agent):
            lines.append(
                f"- {agent}: {walltime_by_agent[agent] / 60:.1f} min"
            )
        lines.append("")

    lines.extend(
        _improvements_section(
            log_dir,
            snap.iteration,
            artifact_root_ref=artifact_root_ref,
        )
    )

    # Dollar estimates retained as a SECONDARY footnote. Heading text
    # `## Cost (estimated)` and the "verify against provider billing records"
    # caveat are pinned by tools/tests/orch/test_report.py — keep them.
    lines.append("## Cost (estimated)")
    lines.append("")
    lines.append("_Secondary metric — wall time above is the headline._")
    lines.append("")
    lines.append(
        "> Costs are estimates using the rate table in `.orch/project.yaml`; "
        "verify against provider billing records. Dollar figures "
        "are not authoritative — model pricing changes, exchange rates, "
        "and opaque provider adjustments make them unreliable."
    )
    lines.append("")
    lines.append(f"- Invocations: {cost.invocations}")
    lines.append(f"- Estimated records: {cost.estimated_count}")
    lines.append(f"- Partial (timed out): {cost.partial_count}")
    lines.append(f"- Unknown-cost records: {cost.unknown_count}")
    lines.append(f"- Cost/parser warnings: {cost.warning_count}")
    lines.append(f"- Total: **${cost.total_usd:.4f}** {HONEST_COST_LABEL}")
    if cost.by_provider_model:
        lines.append("")
        lines.append("By provider/model:")
        for k, detail in sorted(cost.by_provider_model.items()):
            lines.append(
                f"  - {k}: ${detail['cost_usd']:.4f} "
                f"{HONEST_COST_LABEL} "
                f"({detail['invocations']} invocation(s), "
                f"{detail['unknown_count']} unknown-cost)"
            )
    if cost.by_step:
        lines.append("")
        lines.append("By step:")
        for k, v in sorted(cost.by_step.items()):
            lines.append(f"  - {k}: ${v:.4f} {HONEST_COST_LABEL}")
    if cost.by_agent_detail:
        lines.append("")
        lines.append("By agent:")
        for k, detail in sorted(cost.by_agent_detail.items()):
            lines.append(
                f"  - {k}: ${detail['cost_usd']:.4f} "
                f"{HONEST_COST_LABEL} "
                f"({detail['invocations']} invocation(s), "
                f"{detail['unknown_count']} unknown-cost)"
            )
    elif cost.by_agent:
        lines.append("")
        lines.append("By agent:")
        for k, v in sorted(cost.by_agent.items()):
            lines.append(f"  - {k}: ${v:.4f} {HONEST_COST_LABEL}")
    if cost.by_task:
        lines.append("")
        lines.append("By task:")
        for k, v in sorted(cost.by_task.items()):
            lines.append(f"  - {k}: ${v:.4f} {HONEST_COST_LABEL}")
    lines.append("")

    return "\n".join(lines) + "\n"


def build_report(
    *,
    state: StateStore,
    cost_jsonl: Path,
    high_risk_globs: list[str],
    changed_files_by_task: dict[str, list[str]] | None = None,
    artifact_root_ref: str = "tools/logs",
) -> str:
    cost = summarize_costs(cost_jsonl)
    ctx = ReportContext(
        state=state,
        cost=cost,
        high_risk_globs=list(high_risk_globs or []),
        changed_files_by_task=dict(changed_files_by_task or {}),
    )
    return render_report(
        ctx,
        cost_jsonl=cost_jsonl,
        artifact_root_ref=artifact_root_ref,
    )
