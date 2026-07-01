"""Retrospective subcommand — 3 parallel perspectives on a completed iteration.

Each perspective is an independent LLM call with role-specific focus areas
plus shared context (diff, state, costs, QA report if available, previous
retros for cross-iteration pattern detection).

Output:
    tools/logs/<iter>/retro/<role>.md       — per-perspective report
    tools/logs/<iter>/retrospective.md      — main deliverable
"""
from __future__ import annotations

import sys
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from orch.agents import AgentAdapter
from orch.agents.base import cost_record_usage_kwargs, ensure_usage_json_args
from orch.config import LoadedConfig
from orch.cost import (
    HONEST_COST_LABEL,
    CostLogger,
    load_records,
    summarize_costs,
)
from orch.improvements import (
    IMPROVEMENTS_FILENAME,
    ImprovementValidationError,
    ensure_improvements_artifact,
    read_records,
    status_counts,
)
from orch.model_routing import resolve_quality_gate_routing_options
from orch.paths import OrchPaths, resolve_orch_paths
from orch.state import StateStore
from orch.tasks_schema import TaskBoard
from orch.team_mode import ReadOnlyTeamRole, run_read_only_team
from orch.timing import TimingEvidence, detect_evidence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RETRO_ROLES = ("developer", "product_owner", "scrum_master")
RETRO_TEAM_VERDICTS = ("COMPLETE", "COMPLETE_WITH_FOLLOWUPS", "INCOMPLETE")

_ROLE_PROMPTS: dict[str, str] = {
    "developer": (
        "You are reflecting as the **developer** on this iteration. Focus on:\n"
        "- Technical decisions: what worked, what didn't\n"
        "- Code quality and maintainability\n"
        "- Tooling and workflow friction\n"
        "- Technical debt introduced or resolved\n"
        "- Time/cost efficiency of implementation\n"
    ),
    "product_owner": (
        "You are reflecting as the **product owner**. Focus on:\n"
        "- Were the right features built for the right reasons?\n"
        "- Scope management: was scope appropriate?\n"
        "- Acceptance criteria clarity and completeness\n"
        "- Value delivered vs. effort spent\n"
        "- Stakeholder impact and readiness\n"
    ),
    "scrum_master": (
        "You are reflecting as the **scrum master**. Focus on:\n"
        "- Process health: did the orchestrator pipeline work smoothly?\n"
        "- Blockers and how they were resolved\n"
        "- Task dependency management\n"
        "- Communication gaps or assumptions\n"
        "- Improvement actions for the next iteration\n"
    ),
}

# v2.2 — prompt-quality assessment is asked of every role so each
# perspective can evaluate from its own angle. The retro aggregates
# these into a single actionable list for the next iteration's prompts.
_PROMPT_QUALITY_INSTRUCTIONS = (
    "For each task prompt and (implicitly) the review prompt, identify:\n"
    "- **Ambiguity / incompleteness** — wording that could be read two "
    "ways, or a missing invariant the implementer had to guess.\n"
    "- **Calibration** — was the review prompt too strict (rejecting "
    "acceptable diffs) or too weak (letting bugs through)? Cite the "
    "verdict history from the events when possible.\n"
    "- **Model-limitation vs. prompt-quality** — when convergence "
    "failed, classify: was the task beyond the chosen model's "
    "capability, or was the prompt under-specified? If a pair_swap "
    "event is present, the comparison between primary and swap "
    "outcomes is direct evidence.\n"
    "- **Concrete rewrites** — propose 1-3 specific prompt edits "
    "(add this invariant, clarify this field, tighten this "
    "calibration line) that the operator can paste into the next "
    "iteration's prompts/*.md or review template.\n"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RoleResult:
    role: str
    ok: bool
    text: str
    timed_out: bool = False


@dataclass
class RetroReport:
    roles: list[RoleResult]
    output_dir: Path | None = None

    @property
    def incomplete_roles(self) -> list[RoleResult]:
        return [r for r in self.roles if not r.ok]


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def _events_summary(state: StateStore) -> str:
    lines: list[str] = []
    for ev in state.events:
        kind = ev.get("kind", "?")
        task = ev.get("task") or ""
        meta = ev.get("meta") or {}
        ts = ev.get("ts", "")
        parts = [ts, kind]
        if task:
            parts.append(task)
        detail = ", ".join(f"{k}={v}" for k, v in meta.items())
        if detail:
            parts.append(detail)
        lines.append(" | ".join(parts))
    return "\n".join(lines) if lines else "(no events recorded)"


def _timing_artifact(
    iteration: str,
    name: str,
    *,
    artifact_root_ref: str = "tools/logs",
) -> str:
    return f"{artifact_root_ref.rstrip('/')}/{iteration}/{name}"


def _timing_evidence_text(
    evidence: TimingEvidence,
    iteration: str,
    *,
    artifact_root_ref: str = "tools/logs",
) -> str:
    timing_ref = _timing_artifact(
        iteration, "timing.jsonl", artifact_root_ref=artifact_root_ref
    )
    notes_ref = _timing_artifact(
        iteration, "notes.md", artifact_root_ref=artifact_root_ref
    )
    if evidence.has_timing_log:
        return (
            "Timing evidence: **present** — measured timing log found at "
            f"`{timing_ref}`."
        )
    if evidence.has_notes_fallback:
        return (
            "Timing evidence: **operator-provided fallback** — no "
            f"`{timing_ref}` was found; using fallback notes at "
            f"`{notes_ref}`. Treat this as operator-provided context, "
            "not measured `orch timing` evidence."
        )
    return (
        "Timing evidence: **missing** — no "
        f"`{timing_ref}` or `{notes_ref}` found.\n"
        "Operator action: run "
        f"`python -m orch timing --iter {iteration} start <label>` and "
        f"`python -m orch timing --iter {iteration} end <label>` during "
        f"the run, or add `{notes_ref}` explaining the missing/manual timing "
        "before claiming wall-time evidence."
    )


def _cost_summary_text(
    cost_path: Path,
    iteration: str | None = None,
    *,
    artifact_root_ref: str = "tools/logs",
) -> str:
    iteration_name = iteration or cost_path.parent.name
    timing_evidence = detect_evidence(cost_path.parent)
    lines = [
        _timing_evidence_text(
            timing_evidence,
            iteration_name,
            artifact_root_ref=artifact_root_ref,
        ),
        "",
    ]
    if not cost_path.exists():
        lines.append("(no cost data)")
        return "\n".join(lines)
    records = load_records(cost_path)
    summary = summarize_costs(cost_path)

    # Wall-time by task and agent (operator's primary constraint).
    # Older iterations may not have
    # `duration_s` populated — fall back to "unknown" rather than
    # showing a misleading 0.0.
    walltime_by_task: dict[str, float] = {}
    walltime_by_agent: dict[str, float] = {}
    has_duration = False
    for r in records:
        dur = float(r.get("duration_s", 0.0))
        if dur > 0:
            has_duration = True
        t = r.get("task", "?")
        a = r.get("agent", "?")
        walltime_by_task[t] = walltime_by_task.get(t, 0.0) + dur
        walltime_by_agent[a] = walltime_by_agent.get(a, 0.0) + dur

    lines.append("**Wall-time (operator's primary constraint — headline metric):**")
    if has_duration and timing_evidence.has_timing_log:
        for t, dur in sorted(walltime_by_task.items()):
            lines.append(f"  {t}: {dur / 60:.1f} min")
        lines.append(
            f"  Total: {sum(walltime_by_task.values()) / 60:.1f} min"
        )
        lines.append("")
        lines.append("By agent:")
        for a, dur in sorted(walltime_by_agent.items()):
            lines.append(f"  {a}: {dur / 60:.1f} min")
    elif timing_evidence.has_notes_fallback:
        lines.append(
            "  wall time: operator fallback only — `timing.jsonl` is absent; "
            "cost-log durations are context, not measured timing evidence."
        )
    elif timing_evidence.is_missing:
        lines.append(
            "  wall time: missing evidence — unknown because orch timing "
            "was not used; cost-log durations are not treated as measured "
            "wall-time evidence without `timing.jsonl` or `notes.md`."
        )
    else:
        lines.append(
            "  wall time: unknown — timing evidence exists, but cost records "
            "lack positive duration_s values."
        )
    lines.append("")
    lines.append(
        f"Dollar estimate (secondary, unreliable): "
        f"${summary.total_usd:.4f} "
        f"{HONEST_COST_LABEL} "
        f"({summary.invocations} invocations, "
        f"{summary.estimated_count} estimated, "
        f"{summary.partial_count} partial)"
    )
    if summary.by_task:
        lines.append("By task ($):")
        for t, c in summary.by_task.items():
            lines.append(f"  {t}: ${c:.4f} {HONEST_COST_LABEL}")
    return "\n".join(lines)


def _cost_extra(base: dict, result) -> dict:
    out = dict(base)
    if getattr(result, "extra", None):
        out["agent_result_extra"] = dict(result.extra)
    return out


def _load_qa_report(log_dir: Path) -> str:
    qa_report = log_dir / "qa_report.md"
    if qa_report.exists():
        return qa_report.read_text()
    return "(no QA report available)"


def _load_prompts(board: TaskBoard, orch_paths: OrchPaths) -> str:
    """Return the per-task prompt text plus the review prompt template.

    Used for the Prompt Quality Assessment section so the LLM can reason
    about the exact instructions the implementer and reviewer received.
    """
    parts: list[str] = []
    prompts_dir = orch_paths.task_prompts_dir(board.path.parent)
    if prompts_dir.exists():
        for task in board.tasks:
            slug = (
                task.branch.split("/")[-1] if "/" in task.branch
                else task.id.lower()
            )
            for candidate in (
                prompts_dir / f"{slug}.md",
                prompts_dir / f"{task.id.lower()}.md",
            ):
                if candidate.exists():
                    parts.append(
                        f"### Task prompt — {task.id} ({candidate.name})\n\n"
                        f"{candidate.read_text()}"
                    )
                    break
    reviews_dir = orch_paths.task_reviews_dir(board.path.parent)
    if reviews_dir.exists():
        for review_file in sorted(reviews_dir.glob("review-*.md")):
            parts.append(
                f"### Review criteria — {review_file.name}\n\n"
                f"{review_file.read_text()}"
            )
    if not parts:
        return "(no prompt files found)"
    return "\n\n---\n\n".join(parts)


def _swap_summary(state: StateStore) -> str:
    """Summarize any pair-swap events for the Prompt Quality Assessment."""
    swaps: list[str] = []
    for ev in state.events:
        if ev.get("kind") != "pair_swap":
            continue
        meta = ev.get("meta") or {}
        task = ev.get("task") or "?"
        swaps.append(
            f"- {task}: primary={meta.get('primary_implementer')}/"
            f"{meta.get('primary_reviewer')} -> swap="
            f"{meta.get('swap_implementer')}/{meta.get('swap_reviewer')} "
            f"(reason={meta.get('reason')})"
        )
    # Also list the per-task swap_outcome from the final snapshot so the
    # LLM can compare primary-pair failures against swap-pair outcomes.
    for tid, ts in state.snapshot.tasks.items():
        if ts.swap_attempted:
            swaps.append(
                f"  outcome for {tid}: {ts.swap_outcome or 'unknown'}"
            )
    return "\n".join(swaps) if swaps else "(no pair swaps triggered)"


def _load_previous_retros(logs_root: Path, current_iteration: str) -> str:
    """Load retrospective files from previous iterations for cross-iteration
    pattern detection."""
    retros: list[str] = []
    if not logs_root.exists():
        return "(no previous retrospectives)"
    for d in sorted(logs_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name == current_iteration:
            continue
        retro_file = d / "retrospective.md"
        if retro_file.exists():
            retros.append(
                f"### {d.name}\n\n{retro_file.read_text()}\n"
            )
    if not retros:
        return "(no previous retrospectives)"
    return "\n---\n\n".join(retros)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_retro_prompt(
    role: str,
    events_text: str,
    cost_text: str,
    qa_report: str,
    prev_retros: str,
    iteration: str,
    prompts_text: str,
    swap_text: str,
    improvement_ref: str,
) -> str:
    role_section = _ROLE_PROMPTS.get(role, f"You are the **{role}**.\n")
    role_prefix = {
        "developer": "RETRO-D",
        "product_owner": "RETRO-PO",
        "scrum_master": "RETRO-SM",
    }.get(role, "RETRO")
    return (
        f"{role_section}\n"
        f"## Instructions\n\n"
        f"Reflect on iteration **{iteration}** and produce a structured "
        "markdown retrospective. Use this exact section order:\n\n"
        "1. `## Outcome`\n"
        "2. `## What Worked`\n"
        "3. `## What Failed or Was Expensive`\n"
        "4. `## Root-Cause Classification`\n"
        "5. `## Prompt Quality Assessment`\n"
        "6. `## Improvement Records`\n"
        "7. `## Carry-Forward Recommendations`\n"
        "8. final line: `Retro Verdict: COMPLETE | "
        "COMPLETE_WITH_FOLLOWUPS | INCOMPLETE`\n\n"
        f"Use stable record IDs `{role_prefix}1`, `{role_prefix}2`, ... so "
        "retro conclusions can reference QA findings and feed the next "
        "planning prompt.\n\n"
        "Root-cause taxonomy: `prompt_ambiguity`, "
        "`prompt_missing_invariant`, `review_template_gap`, "
        "`review_calibration_too_strict`, `review_calibration_too_weak`, "
        "`qa_template_gap`, `deterministic_gate_missing`, "
        "`model_fit_or_pairing`, `operator_decision_gap`, "
        "`environment_or_ci`, `implementation_error`, "
        "`scope_or_branch_hygiene`.\n\n"
        "Required tables:\n\n"
        "```markdown\n"
        "| ID | Issue | Evidence | Primary cause | Secondary cause | Preventive control |\n"
        "|---|---|---|---|---|---|\n"
        f"| {role_prefix}1 | ... | event/QA finding/file | taxonomy | optional | control |\n"
        "```\n\n"
        "```markdown\n"
        "| ID | Action | Owner | Priority | Acceptance check | Control mechanism |\n"
        "|---|---|---|---|---|---|\n"
        f"| {role_prefix}2 | ... | operator/agent/tooling | P0/P1/P2 | check | gate/file/test |\n"
        "```\n\n"
        "Required shapes for `## Prompt Quality Assessment`:\n\n"
        "```markdown\n"
        "| ID | Task | Signal | Root cause | Template change needed? | Exact rewrite |\n"
        "|---|---|---|---|---|---|\n"
        "```\n\n"
        "```markdown\n"
        "| ID | Review | Calibration | Missed? | Too strict? | Change |\n"
        "|---|---|---|---|---|---|\n"
        "```\n\n"
        "```markdown\n"
        "| ID | QA finding | Should have been caught earlier? | Missing gate |\n"
        "|---|---|---|---|\n"
        "```\n\n"
        f"Do not approve or implement improvement actions autonomously. "
        f"Approved or implemented improvement records require an "
        f"operator-approved `control_mechanism` in "
        f"`{improvement_ref}`.\n"
        f"{_PROMPT_QUALITY_INSTRUCTIONS}\n\n"
        f"---\n\n"
        f"## Run Events\n\n{events_text}\n\n"
        f"## Pair Swaps\n\n{swap_text}\n\n"
        f"## Cost Summary\n\n{cost_text}\n\n"
        f"## QA Report\n\n{qa_report}\n\n"
        f"## Task Prompts and Review Criteria\n\n{prompts_text}\n\n"
        f"## Previous Retrospectives\n\n{prev_retros}\n"
    )


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_retro(
    *,
    cfg: LoadedConfig,
    board: TaskBoard,
    state: StateStore,
    cost: CostLogger,
    adapters: dict[str, AgentAdapter],
    iteration: str,
    cwd: Path,
    agent_name: str,
    roles: list[str] | None = None,
    timeout: int = 900,
) -> RetroReport:
    """Run retrospective with parallel LLM calls per role.

    Returns a RetroReport. Partial failure is tolerated.
    """
    active_roles = [r for r in (roles or RETRO_ROLES) if r in RETRO_ROLES]
    if not active_roles:
        active_roles = list(RETRO_ROLES)

    adapter = adapters[agent_name]
    orch_paths = resolve_orch_paths(cwd, cfg)
    log_dir = orch_paths.iteration_log_dir(iteration)
    retro_dir = log_dir / "retro"
    retro_dir.mkdir(parents=True, exist_ok=True)
    improvement_path = ensure_improvements_artifact(log_dir)

    # Gather context
    events_text = _events_summary(state)
    artifact_root_ref = orch_paths.artifact_root_ref
    timing_status_text = _timing_evidence_text(
        detect_evidence(log_dir),
        iteration,
        artifact_root_ref=artifact_root_ref,
    )
    cost_text = _cost_summary_text(
        log_dir / "cost.jsonl",
        iteration=iteration,
        artifact_root_ref=artifact_root_ref,
    )
    qa_report = _load_qa_report(log_dir)
    prev_retros = _load_previous_retros(
        orch_paths.artifact_root, iteration
    )
    prompts_text = _load_prompts(board, orch_paths)
    swap_text = _swap_summary(state)
    improvement_ref = orch_paths.artifact_ref(iteration, IMPROVEMENTS_FILENAME)

    # Build per-role prompts
    prompts: dict[str, str] = {}
    for role in active_roles:
        prompts[role] = _build_retro_prompt(
            role, events_text, cost_text, qa_report, prev_retros, iteration,
            prompts_text, swap_text, improvement_ref,
        )

    # Execute in parallel
    results: list[RoleResult] = []

    def _invoke_role(role: str) -> RoleResult:
        try:
            routing_options = resolve_quality_gate_routing_options(
                cfg.data.get("model_routing"),
                agent_name=agent_name,
            )
            result = adapter.invoke(
                prompts[role],
                timeout=timeout,
                workdir=cwd,
                routing_options=routing_options,
            )
            cost.record(
                task="RETRO", step="RETRO", agent=agent_name,
                **cost_record_usage_kwargs(
                    result,
                    family=adapter.family,
                    provider=agent_name,
                ),
                duration_s=result.duration_s,
                exit_code=result.exit_code,
                partial=result.partial,
                extra=_cost_extra({"role": role}, result),
            )
            if result.partial:
                return RoleResult(
                    role=role, ok=False, text=result.stdout,
                    timed_out=True,
                )
            if result.exit_code != 0:
                text = result.stdout or f"Agent exited {result.exit_code}"
                if result.stderr:
                    text = f"{text}\n\nstderr:\n{result.stderr}"
                return RoleResult(role=role, ok=False, text=text)
            return RoleResult(role=role, ok=True, text=result.stdout)
        except Exception as exc:
            return RoleResult(
                role=role, ok=False, text=f"Error: {exc}",
            )

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_invoke_role, role): role for role in active_roles
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by role order
    role_order = {r: i for i, r in enumerate(RETRO_ROLES)}
    results.sort(key=lambda r: role_order.get(r.role, 99))

    # Write per-role files
    for r in results:
        suffix = " (TIMEOUT)" if r.timed_out else ""
        (retro_dir / f"{r.role}.md").write_text(
            f"# Retrospective: {r.role}{suffix}\n\n{r.text}\n"
        )

    # Write combined report
    report_lines: list[str] = [f"# Retrospective — {iteration}\n"]
    ok_count = sum(1 for r in results if r.ok)
    report_lines.append(
        f"**Perspectives:** {ok_count}/{len(results)} completed\n\n---\n"
    )
    report_lines.append(f"\n## Timing Evidence\n\n{timing_status_text}\n")
    report_lines.append("\n## Six Sigma Improvement Records\n")
    report_lines.append(f"- Artifact: `{improvement_ref}`")
    report_lines.append(
        "- Control gate: `approved` and `implemented` records require a "
        "non-empty `control_mechanism`."
    )
    report_lines.append(
        "- Operator control: retrospectives may propose actions; they do "
        "not approve or implement improvements autonomously."
    )
    try:
        improvement_records = read_records(improvement_path)
    except ImprovementValidationError as exc:
        report_lines.append(f"- Validation: **FAILED** — {exc}\n")
    else:
        counts = status_counts(improvement_records)
        if counts:
            counts_text = ", ".join(
                f"{status}: {count}" for status, count in counts.items()
            )
        else:
            counts_text = "none"
        report_lines.append(
            f"- Current records: {len(improvement_records)} "
            f"({counts_text})\n"
        )
    for r in results:
        status = "OK" if r.ok else ("TIMEOUT" if r.timed_out else "ERROR")
        report_lines.append(
            f"\n## {r.role.replace('_', ' ').title()} [{status}]\n\n{r.text}\n"
        )

    report_text = "\n".join(report_lines)
    (log_dir / "retrospective.md").write_text(report_text)
    _log(f"Retrospective written to {log_dir / 'retrospective.md'}")

    return RetroReport(roles=results, output_dir=retro_dir)


def run_retro_team_mode(
    *,
    cfg: LoadedConfig,
    board: TaskBoard,
    state: StateStore,
    cost: CostLogger,
    adapters: dict[str, AgentAdapter],
    iteration: str,
    cwd: Path,
    agent_name: str,
    roles: list[str] | None = None,
    timeout: int = 900,
) -> RetroReport:
    """Run retrospective roles through the read-only team-mode artifact path."""
    active_roles = [r for r in (roles or RETRO_ROLES) if r in RETRO_ROLES]
    if not active_roles:
        active_roles = list(RETRO_ROLES)

    orch_paths = resolve_orch_paths(cwd, cfg)
    log_dir = orch_paths.iteration_log_dir(iteration)
    retro_dir = log_dir / "retro"
    retro_dir.mkdir(parents=True, exist_ok=True)
    improvement_path = ensure_improvements_artifact(log_dir)

    events_text = _events_summary(state)
    artifact_root_ref = orch_paths.artifact_root_ref
    cost_text = _cost_summary_text(
        log_dir / "cost.jsonl",
        iteration=iteration,
        artifact_root_ref=artifact_root_ref,
    )
    qa_report = _load_qa_report(log_dir)
    prev_retros = _load_previous_retros(orch_paths.artifact_root, iteration)
    prompts_text = _load_prompts(board, orch_paths)
    swap_text = _swap_summary(state)
    improvement_ref = orch_paths.artifact_ref(iteration, IMPROVEMENTS_FILENAME)

    prompts: dict[str, str] = {}
    for role in active_roles:
        prompts[role] = _build_retro_prompt(
            role, events_text, cost_text, qa_report, prev_retros, iteration,
            prompts_text, swap_text, improvement_ref,
        )

    team_roles = [
        ReadOnlyTeamRole(
            name=role,
            prompt=prompts[role],
            artifact_dir=retro_dir / "team" / role,
            result_filename="result.md",
            verdict_labels=RETRO_TEAM_VERDICTS,
        )
        for role in active_roles
    ]
    team_results = run_read_only_team(
        team_name=f"retro-{iteration}",
        roles=team_roles,
        cwd=cwd,
        command=_agent_command(cfg, agent_name),
        timeout=timeout,
        log_dir=log_dir,
        cost=cost,
        agent_name=agent_name,
        family=getattr(adapters.get(agent_name), "family", "unknown"),
        task="RETRO",
        step="RETRO_TEAM",
    )

    results = [
        RoleResult(
            role=result.role,
            ok=result.ok,
            text=result.text,
            timed_out=result.timed_out or result.killed,
        )
        for result in team_results
    ]
    role_order = {r: i for i, r in enumerate(RETRO_ROLES)}
    results.sort(key=lambda r: role_order.get(r.role, 99))

    for r in results:
        suffix = " (TIMEOUT)" if r.timed_out else ""
        (retro_dir / f"{r.role}.md").write_text(
            f"# Retrospective: {r.role}{suffix}\n\n{r.text}\n"
        )

    timing_status_text = _timing_evidence_text(
        detect_evidence(log_dir),
        iteration,
        artifact_root_ref=artifact_root_ref,
    )
    report_lines: list[str] = [f"# Retrospective — {iteration}\n"]
    ok_count = sum(1 for r in results if r.ok)
    report_lines.append(
        f"**Perspectives:** {ok_count}/{len(results)} completed\n\n---\n"
    )
    report_lines.append(f"\n## Timing Evidence\n\n{timing_status_text}\n")
    report_lines.append("\n## Six Sigma Improvement Records\n")
    report_lines.append(f"- Artifact: `{improvement_ref}`")
    report_lines.append(
        "- Control gate: `approved` and `implemented` records require a "
        "non-empty `control_mechanism`."
    )
    report_lines.append(
        "- Operator control: retrospectives may propose actions; they do "
        "not approve or implement improvements autonomously."
    )
    try:
        improvement_records = read_records(improvement_path)
    except ImprovementValidationError as exc:
        report_lines.append(f"- Validation: **FAILED** — {exc}\n")
    else:
        counts = status_counts(improvement_records)
        if counts:
            counts_text = ", ".join(
                f"{status}: {count}" for status, count in counts.items()
            )
        else:
            counts_text = "none"
        report_lines.append(
            f"- Current records: {len(improvement_records)} "
            f"({counts_text})\n"
        )
    for r in results:
        status = "OK" if r.ok else ("TIMEOUT" if r.timed_out else "ERROR")
        report_lines.append(
            f"\n## {r.role.replace('_', ' ').title()} [{status}]\n\n{r.text}\n"
        )

    report_text = "\n".join(report_lines)
    (log_dir / "retrospective.md").write_text(report_text)
    _log(f"Retrospective written to {log_dir / 'retrospective.md'}")

    return RetroReport(roles=results, output_dir=retro_dir)


def _agent_command(cfg: LoadedConfig, agent_name: str) -> tuple[str, ...]:
    spec = cfg.data.get("agents", {}).get(agent_name, {})
    cmd = str(spec.get("cmd") or agent_name).strip()
    if not cmd:
        raise ValueError(f"agent {agent_name!r} has no command")
    provider = str(spec.get("type") or spec.get("family") or agent_name)
    return tuple(ensure_usage_json_args(shlex.split(cmd), provider))


def _log(msg: str) -> None:
    print(f"[orch-retro] {msg}", file=sys.stderr, flush=True)
