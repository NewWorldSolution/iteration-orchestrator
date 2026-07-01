"""QA subcommand — 5 parallel LLM reviewers analyze a completed iteration.

Each reviewer is an independent, self-contained LLM call with role-specific
focus areas plus shared context (full diff, run_state events, cost summary,
task prompts).  No inter-agent communication.

Output:
    tools/logs/<iter>/qa/<role>.md   — per-reviewer report
    tools/logs/<iter>/qa_report.md   — main deliverable (all + optional synthesis)
    tools/logs/<iter>/qa/diff_base.txt — resolved diff base for auditability
"""
from __future__ import annotations

import re
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from orch.agents import AgentAdapter
from orch.agents.base import cost_record_usage_kwargs, ensure_usage_json_args
from orch.config import LoadedConfig
from orch.cost import HONEST_COST_LABEL, CostLogger, summarize_costs
from orch.git_ops import (
    BranchFreshnessCondition,
    classify_branch_freshness,
)
from orch.git_ops import diff_text as git_diff_text
from orch.lifecycle import PhaseResolutionError, resolve_phase_branch
from orch.model_routing import resolve_quality_gate_routing_options
from orch.paths import OrchPaths, resolve_orch_paths
from orch.state import StateStore
from orch.tasks_schema import TaskBoard
from orch.team_mode import ReadOnlyTeamRole, run_read_only_team

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QA_ROLES = ("security", "architecture", "test", "product", "process")
QA_TEAM_VERDICTS = ("PASS", "CHANGES_REQUIRED", "BLOCKED")

_ROLE_PROMPTS: dict[str, str] = {
    "security": (
        "You are a **security reviewer**. Focus on:\n"
        "- Authentication / authorization gaps\n"
        "- Input validation and injection risks (SQL, XSS, command)\n"
        "- Secrets or credentials in code\n"
        "- Insecure defaults, missing CSRF/CORS protections\n"
        "- Dependency vulnerabilities\n"
    ),
    "architecture": (
        "You are an **architecture reviewer**. Focus on:\n"
        "- Adherence to stated architecture (layers, patterns, stack)\n"
        "- Separation of concerns, coupling, cohesion\n"
        "- Database schema consistency\n"
        "- Performance implications\n"
        "- Migration safety\n"
    ),
    "test": (
        "You are a **test/quality reviewer**. Focus on:\n"
        "- Test coverage for new/changed code\n"
        "- Edge cases and error paths\n"
        "- Test isolation and determinism\n"
        "- Assertion quality (not just 'runs without error')\n"
        "- Missing integration or boundary tests\n"
    ),
    "product": (
        "You are a **product reviewer**. Focus on:\n"
        "- Does the implementation match the task requirements?\n"
        "- User-facing behavior correctness\n"
        "- Missing acceptance criteria\n"
        "- UX regressions or inconsistencies\n"
        "- Documentation gaps for user-visible changes\n"
    ),
    "process": (
        "You are a **process reviewer**. Focus on:\n"
        "- Commit hygiene and message quality\n"
        "- Scope compliance (did the iteration stay within bounds?)\n"
        "- Cost efficiency (tokens, attempts, fix rounds)\n"
        "- State-machine correctness (expected transitions)\n"
        "- Readiness for merge into the phase branch\n"
    ),
}


class QaDiffBaseError(Exception):
    """Raised when the diff base cannot be resolved."""


class QaEmptyDiffError(Exception):
    """Raised when QA would review an empty or placeholder diff."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RoleResult:
    role: str
    ok: bool
    text: str  # markdown report or error message
    timed_out: bool = False


@dataclass
class QaReport:
    roles: list[RoleResult]
    synthesis: str | None = None
    diff_base: str = ""
    output_dir: Path | None = None

    @property
    def incomplete_roles(self) -> list[RoleResult]:
        return [r for r in self.roles if not r.ok]


# ---------------------------------------------------------------------------
# Diff-base resolution
# ---------------------------------------------------------------------------

_DIFF_BASE_RE = re.compile(
    r"^\*\*Diff base:\*\*\s*`?([^`\s]+)`?", re.MULTILINE
)


def resolve_diff_base(
    board: TaskBoard,
    cfg: LoadedConfig,
    iteration: str,
) -> str:
    """Resolve the diff base commit/branch for QA comparison.

    Priority:
    1. Explicit ``**Diff base:**`` field in tasks.md source text
    2. Derived from project.yaml ``phase_branch_pattern`` + iteration ID
    3. Hard error ``QA_DIFF_BASE_UNRESOLVED``
    """
    # 1 — check tasks.md raw text for **Diff base:** field
    if board.path.exists():
        raw = board.path.read_text()
        m = _DIFF_BASE_RE.search(raw)
        if m:
            return m.group(1).strip()

    # 2 — derive from phase_branch_pattern
    try:
        phase_branch = resolve_phase_branch(cfg, iteration)
    except PhaseResolutionError as exc:
        raise QaDiffBaseError(
            f"QA_DIFF_BASE_UNRESOLVED: {exc}"
        ) from exc
    if phase_branch:
        return phase_branch

    raise QaDiffBaseError(
        f"QA_DIFF_BASE_UNRESOLVED: cannot determine diff base for "
        f"iteration '{iteration}'. Add '**Diff base:** <ref>' to tasks.md "
        f"or ensure project.yaml has phase_branch_pattern."
    )

# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_qa_prompt(
    role: str,
    diff: str,
    events_summary: str,
    cost_summary: str,
    task_prompts: str,
    review_artifacts: str,
    *,
    iteration: str,
    diff_base: str,
    iteration_branch: str,
) -> str:
    role_section = _ROLE_PROMPTS.get(role, f"You are a **{role} reviewer**.\n")
    role_prefix = {
        "security": "QA-S",
        "architecture": "QA-A",
        "test": "QA-T",
        "product": "QA-P",
        "process": "QA-PR",
    }.get(role, "QA")
    return (
        f"{role_section}\n"
        "## Instructions\n\n"
        f"Review iteration `{iteration}` as the `{role}` QA perspective. "
        "QA is not a larger per-task review; catch cross-task integration "
        "defects, missing evidence, stale-base issues, and process defects.\n\n"
        "Use this exact section order:\n\n"
        "1. `## Summary`\n"
        "2. `## Blocking Decision`\n"
        "3. `## Invariant Closure`\n"
        "4. `## Findings`\n"
        "5. `## Evidence`\n"
        "6. final line: `QA Verdict: OK | CONCERNS | BLOCK | INCOMPLETE`\n\n"
        "## Common Gates\n\n"
        "### Gate 1 - Diff Base and Inverse Diff\n\n"
        "Run or reason from these exact comparisons:\n\n"
        "```bash\n"
        f"git diff {diff_base}..{iteration_branch} --name-only\n"
        f"git diff {iteration_branch}..origin/<phase-branch> --name-only\n"
        "```\n\n"
        "Classify every inverse-diff file as expected post-branch phase work, "
        "missing sync/stale base, not applicable, or blocking unexplained diff.\n\n"
        "### Gate 2 - Allowed-File Union\n\n"
        "Forward diff must be inside the union of task allowed files plus "
        "documented orchestrator-owned status artifacts.\n\n"
        "### Gate 3 - Invariant Closure\n\n"
        "Fill one row per invariant from the iteration prompt/tasks and per "
        "applicable project invariant. Any failing blocking invariant means "
        "`QA Verdict: BLOCK`.\n\n"
        "### Gate 4 - Cross-Task Grep Gates\n\n"
        "Run all ten `_prompt_rules.md` Rule 5 signatures plus "
        "iteration-specific signatures. Every hit must map to documented "
        "exception, real finding, or false positive with evidence.\n\n"
        "### Gate 5 - Preserved Behavior\n\n"
        "Run the preserved-behavior fixture when required. If a required "
        "fixture is missing, report a Process finding and use `INCOMPLETE` or "
        "`BLOCK` depending on risk.\n\n"
        "### Gate 6 - Per-Task Review Audit\n\n"
        "Audit authored review prompts and runtime review artifacts. Identify "
        "missing artifacts, deferred findings, and suspicious gaps.\n\n"
        "### Gate 7 - Regression Evidence\n\n"
        "Record exact focused tests, full tests, lint, manual smoke, and CI "
        "evidence if available.\n\n"
        "## Findings Format\n\n"
        f"Use stable IDs `{role_prefix}1`, `{role_prefix}2`, ... and format "
        "each issue as:\n\n"
        f"- `[{role_prefix}1] [CRITICAL | SHOULD_FIX | FUTURE] <summary>`\n"
        "  - Task(s):\n"
        "  - File: `<file:line>`\n"
        "  - Evidence:\n"
        "  - Required fix or follow-up:\n\n"
        "Verdict guide: `OK` for complete evidence and no blocking findings; "
        "`CONCERNS` for only non-blocking findings; `BLOCK` for invariant, "
        "security, data integrity, or blocking regression; `INCOMPLETE` for "
        "missing required evidence or invalid diff base.\n\n"
        f"---\n\n"
        f"## Runtime Metadata\n\n"
        f"- Iteration: `{iteration}`\n"
        f"- Diff base: `{diff_base}`\n"
        f"- Iteration branch: `{iteration_branch}`\n\n"
        f"## Diff\n\n```diff\n{diff}\n```\n\n"
        f"## Run Events\n\n{events_summary}\n\n"
        f"## Cost Summary\n\n{cost_summary}\n\n"
        f"## Task Prompts\n\n{task_prompts}\n\n"
        f"## Review Prompts and Runtime Review Artifacts\n\n{review_artifacts}\n"
    )


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


def _cost_extra(base: dict, result) -> dict:
    out = dict(base)
    if getattr(result, "extra", None):
        out["agent_result_extra"] = dict(result.extra)
    return out


def _cost_summary_text(cost_path: Path) -> str:
    import json

    if not cost_path.exists():
        return "_No timing events recorded._\n"

    records: list[dict] = []
    for line in cost_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    if not records:
        return "_No timing events recorded._\n"

    def _fmt(seconds: float) -> str:
        total_seconds = int(round(seconds))
        minutes, secs = divmod(total_seconds, 60)
        return f"{minutes}:{secs:02d}"

    summary = summarize_costs(cost_path)
    by_task_by_tool: dict[str, dict[str, float]] = {}
    tool_labels: set[str] = set()
    grand_total_s = 0.0

    for record in records:
        task_id = str(record.get("task") or "?")
        agent = str(record.get("agent") or "other").strip().lower()
        step = str(record.get("step") or "").strip().lower()
        if step in {"impl", "fix", "review"}:
            tool = f"{agent} {step}"
        elif agent and step:
            tool = f"{agent} {step}"
        elif agent:
            tool = agent
        else:
            tool = "other"

        duration_s = float(record.get("duration_s", 0.0) or 0.0)
        grand_total_s += duration_s
        by_task_by_tool.setdefault(task_id, {})
        by_task_by_tool[task_id][tool] = (
            by_task_by_tool[task_id].get(tool, 0.0) + duration_s
        )
        tool_labels.add(tool)

    tools = sorted(tool_labels)
    lines = [
        "## Cost (wall-time)",
        "",
        "| Task | Wall time | " + " | ".join(tools) + " |",
        "|---|---|" + "|".join("---" for _ in tools) + "|",
    ]
    for task_id in sorted(by_task_by_tool):
        breakdown = by_task_by_tool[task_id]
        task_total_s = sum(breakdown.values())
        cells = [
            _fmt(breakdown[tool]) if tool in breakdown else "—"
            for tool in tools
        ]
        lines.append(
            f"| {task_id} | {_fmt(task_total_s)} | " + " | ".join(cells) + " |"
        )

    lines.extend(
        [
            "",
            f"**Total wall time:** {_fmt(grand_total_s)}",
            "",
            "---",
            "",
            (
                "_Estimated dollar cost for the iteration: "
                f"${summary.total_usd:.2f} "
                f"{HONEST_COST_LABEL} "
                "(verify against provider billing records.)_"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _task_prompts_text(board: TaskBoard, orch_paths: OrchPaths) -> str:
    lines: list[str] = []
    prompts_dir = orch_paths.task_prompts_dir(board.path.parent)
    for task in board.tasks:
        base = task.branch.split("/")[-1] if "/" in task.branch else task.id
        prompt_path = prompts_dir / f"{base}.md"
        alt_path = prompts_dir / f"{task.id.lower()}.md"
        text = ""
        for p in (prompt_path, alt_path):
            if p.exists():
                text = p.read_text()
                break
        if not text:
            text = f"Task {task.id}: {task.title}"
        lines.append(f"### {task.id}: {task.title}\n\n{text}\n")
    return "\n".join(lines)


def _review_artifacts_text(
    board: TaskBoard, log_dir: Path, orch_paths: OrchPaths
) -> str:
    parts: list[str] = []
    authored_reviews_dir = orch_paths.task_reviews_dir(board.path.parent)
    if authored_reviews_dir.exists():
        for path in sorted(authored_reviews_dir.glob("review-*.md")):
            parts.append(
                f"### Authored review prompt - {path.name}\n\n"
                f"{path.read_text()}\n"
            )
    runtime_reviews_dir = log_dir / "reviews"
    if runtime_reviews_dir.exists():
        for path in sorted(runtime_reviews_dir.glob("*.md")):
            parts.append(
                f"### Runtime review artifact - {path.name}\n\n"
                f"{path.read_text()}\n"
            )
    if not parts:
        return "(no authored review prompts or runtime review artifacts found)"
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_qa(
    *,
    cfg: LoadedConfig,
    board: TaskBoard,
    state: StateStore,
    cost: CostLogger,
    adapters: dict[str, AgentAdapter],
    iteration: str,
    cwd: Path,
    reviewer_agent: str,
    roles: list[str] | None = None,
    synthesize: bool = False,
    timeout: int = 900,
    allow_empty_diff_reason: str | None = None,
) -> QaReport:
    """Run QA review with parallel LLM calls per role.

    Returns a QaReport. Partial failure is tolerated — successful reviews
    are saved even if some roles time out or fail.
    """
    active_roles = [r for r in (roles or QA_ROLES) if r in QA_ROLES]
    if not active_roles:
        active_roles = list(QA_ROLES)

    adapter = adapters[reviewer_agent]
    orch_paths = resolve_orch_paths(cwd, cfg)
    log_dir = orch_paths.iteration_log_dir(iteration)
    qa_dir = log_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Resolve diff base
    diff_base = resolve_diff_base(board, cfg, iteration)
    iter_branch = board.iteration_branch

    # Fail closed when the diff base or the iteration branch does not resolve
    # in git. Otherwise git_diff_text below raises and is swallowed into a
    # "(diff unavailable: ...)" placeholder, and the QA reviewers would
    # "review" a non-existent diff and pass vacuously.
    freshness = classify_branch_freshness(
        cwd, branch=iter_branch, base_ref=diff_base
    )
    if freshness.condition in (
        BranchFreshnessCondition.MISSING_BASE,
        BranchFreshnessCondition.MISSING_BRANCH,
    ):
        raise QaDiffBaseError(
            f"QA_DIFF_BASE_UNRESOLVED: diff base '{diff_base}' and iteration "
            f"branch '{iter_branch}' did not both resolve in git "
            f"({freshness.condition.value}); refusing to run QA on a "
            "non-existent diff. Set '**Diff base:** <ref>' in tasks.md to an "
            "existing ref the iteration branch is ahead of, or fetch the "
            "phase branch first."
        )
    if (freshness.ahead_count or 0) == 0:
        # Base resolves but the iteration branch is not strictly ahead of it
        # (e.g. QA re-run after the iteration merged into the base). The QA
        # diff is empty/vacuous, so fail closed unless the operator explicitly
        # documents why this review may proceed.
        msg = (
            f"WARNING: iteration branch '{iter_branch}' is not ahead of diff "
            f"base '{diff_base}' ({freshness.condition.value}); the QA diff is "
            "empty/vacuous and reviewers will see no changes. This usually "
            "means QA is being re-run after the iteration merged into the base."
        )
        if not allow_empty_diff_reason:
            raise QaEmptyDiffError(
                msg
                + " Pass --allow-empty-diff <reason> to acknowledge this "
                "vacuous QA run."
            )
        _log(f"{msg} Override reason: {allow_empty_diff_reason}")

    (qa_dir / "diff_base.txt").write_text(diff_base + "\n")
    _log(f"QA diff base resolved: {diff_base}")

    # Gather context
    try:
        diff = git_diff_text(cwd, diff_base, iter_branch)
    except Exception as exc:
        msg = (
            f"QA diff unavailable between '{diff_base}' and '{iter_branch}': "
            f"{exc}; reviewers would see only a placeholder diff."
        )
        if not allow_empty_diff_reason:
            raise QaEmptyDiffError(
                msg
                + " Pass --allow-empty-diff <reason> to acknowledge this "
                "vacuous QA run."
            ) from exc
        _log(f"WARNING: {msg} Override reason: {allow_empty_diff_reason}")
        diff = f"(diff unavailable: {exc})"
    if not diff.strip():
        msg = (
            f"QA diff between '{diff_base}' and '{iter_branch}' is empty; "
            "reviewers would see no changes."
        )
        if not allow_empty_diff_reason:
            raise QaEmptyDiffError(
                msg
                + " Pass --allow-empty-diff <reason> to acknowledge this "
                "vacuous QA run."
            )
        _log(f"WARNING: {msg} Override reason: {allow_empty_diff_reason}")
    events_text = _events_summary(state)
    cost_text = _cost_summary_text(log_dir / "cost.jsonl")
    tasks_text = _task_prompts_text(board, orch_paths)
    reviews_text = _review_artifacts_text(board, log_dir, orch_paths)

    # Build per-role prompts
    prompts: dict[str, str] = {}
    for role in active_roles:
        prompts[role] = _build_qa_prompt(
            role,
            diff,
            events_text,
            cost_text,
            tasks_text,
            reviews_text,
            iteration=iteration,
            diff_base=diff_base,
            iteration_branch=iter_branch,
        )

    # Execute in parallel
    results: list[RoleResult] = []

    def _invoke_role(role: str) -> RoleResult:
        try:
            routing_options = resolve_quality_gate_routing_options(
                cfg.data.get("model_routing"),
                agent_name=reviewer_agent,
            )
            result = adapter.invoke(
                prompts[role],
                timeout=timeout,
                workdir=cwd,
                routing_options=routing_options,
            )
            cost.record(
                task="QA", step="QA", agent=reviewer_agent,
                **cost_record_usage_kwargs(
                    result,
                    family=adapter.family,
                    provider=reviewer_agent,
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

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_invoke_role, role): role for role in active_roles
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Sort by role order for consistent output
    role_order = {r: i for i, r in enumerate(QA_ROLES)}
    results.sort(key=lambda r: role_order.get(r.role, 99))

    # Write per-role files
    for r in results:
        suffix = " (TIMEOUT)" if r.timed_out else ""
        (qa_dir / f"{r.role}.md").write_text(
            f"# QA Review: {r.role}{suffix}\n\n{r.text}\n"
        )

    # Optional synthesis
    synthesis = None
    if synthesize:
        synthesis = _synthesize(
            results,
            adapter,
            reviewer_agent,
            cost,
            timeout,
            cwd,
            routing_config=cfg.data.get("model_routing"),
        )

    # Write combined report
    report_lines: list[str] = [f"# QA Report — {iteration}\n"]
    report_lines.append(f"**Diff base:** `{diff_base}`\n")
    ok_count = sum(1 for r in results if r.ok)
    report_lines.append(
        f"**Reviewers:** {ok_count}/{len(results)} completed\n\n---\n"
    )
    for r in results:
        status = "OK" if r.ok else ("TIMEOUT" if r.timed_out else "ERROR")
        report_lines.append(f"\n## {r.role.title()} [{status}]\n\n{r.text}\n")
    if synthesis:
        report_lines.append(f"\n---\n\n## Synthesis\n\n{synthesis}\n")

    report_text = "\n".join(report_lines)
    (log_dir / "qa_report.md").write_text(report_text)
    _log(f"QA report written to {log_dir / 'qa_report.md'}")

    return QaReport(
        roles=results,
        synthesis=synthesis,
        diff_base=diff_base,
        output_dir=qa_dir,
    )


def run_qa_team_mode(
    *,
    cfg: LoadedConfig,
    board: TaskBoard,
    state: StateStore,
    cost: CostLogger,
    adapters: dict[str, AgentAdapter],
    iteration: str,
    cwd: Path,
    reviewer_agent: str,
    roles: list[str] | None = None,
    synthesize: bool = False,
    timeout: int = 900,
    allow_empty_diff_reason: str | None = None,
) -> QaReport:
    """Run QA reviewers through the read-only team-mode artifact path."""
    active_roles = [r for r in (roles or QA_ROLES) if r in QA_ROLES]
    if not active_roles:
        active_roles = list(QA_ROLES)

    adapter = adapters[reviewer_agent]
    orch_paths = resolve_orch_paths(cwd, cfg)
    log_dir = orch_paths.iteration_log_dir(iteration)
    qa_dir = log_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    diff_base = resolve_diff_base(board, cfg, iteration)
    iter_branch = board.iteration_branch

    freshness = classify_branch_freshness(
        cwd, branch=iter_branch, base_ref=diff_base
    )
    if freshness.condition in (
        BranchFreshnessCondition.MISSING_BASE,
        BranchFreshnessCondition.MISSING_BRANCH,
    ):
        raise QaDiffBaseError(
            f"QA_DIFF_BASE_UNRESOLVED: diff base '{diff_base}' and iteration "
            f"branch '{iter_branch}' did not both resolve in git "
            f"({freshness.condition.value}); refusing to run QA on a "
            "non-existent diff. Set '**Diff base:** <ref>' in tasks.md to an "
            "existing ref the iteration branch is ahead of, or fetch the "
            "phase branch first."
        )
    if (freshness.ahead_count or 0) == 0:
        msg = (
            f"WARNING: iteration branch '{iter_branch}' is not ahead of diff "
            f"base '{diff_base}' ({freshness.condition.value}); the QA diff is "
            "empty/vacuous and reviewers will see no changes. This usually "
            "means QA is being re-run after the iteration merged into the base."
        )
        if not allow_empty_diff_reason:
            raise QaEmptyDiffError(
                msg
                + " Pass --allow-empty-diff <reason> to acknowledge this "
                "vacuous QA run."
            )
        _log(f"{msg} Override reason: {allow_empty_diff_reason}")

    (qa_dir / "diff_base.txt").write_text(diff_base + "\n")
    _log(f"QA diff base resolved: {diff_base}")

    try:
        diff = git_diff_text(cwd, diff_base, iter_branch)
    except Exception as exc:
        msg = (
            f"QA diff unavailable between '{diff_base}' and '{iter_branch}': "
            f"{exc}; reviewers would see only a placeholder diff."
        )
        if not allow_empty_diff_reason:
            raise QaEmptyDiffError(
                msg
                + " Pass --allow-empty-diff <reason> to acknowledge this "
                "vacuous QA run."
            ) from exc
        _log(f"WARNING: {msg} Override reason: {allow_empty_diff_reason}")
        diff = f"(diff unavailable: {exc})"
    if not diff.strip():
        msg = (
            f"QA diff between '{diff_base}' and '{iter_branch}' is empty; "
            "reviewers would see no changes."
        )
        if not allow_empty_diff_reason:
            raise QaEmptyDiffError(
                msg
                + " Pass --allow-empty-diff <reason> to acknowledge this "
                "vacuous QA run."
            )
        _log(f"WARNING: {msg} Override reason: {allow_empty_diff_reason}")

    events_text = _events_summary(state)
    cost_text = _cost_summary_text(log_dir / "cost.jsonl")
    tasks_text = _task_prompts_text(board, orch_paths)
    reviews_text = _review_artifacts_text(board, log_dir, orch_paths)

    prompts: dict[str, str] = {}
    for role in active_roles:
        prompts[role] = _build_qa_prompt(
            role,
            diff,
            events_text,
            cost_text,
            tasks_text,
            reviews_text,
            iteration=iteration,
            diff_base=diff_base,
            iteration_branch=iter_branch,
        )

    team_roles = [
        ReadOnlyTeamRole(
            name=role,
            prompt=prompts[role],
            artifact_dir=qa_dir / "team" / role,
            result_filename="review.md",
            verdict_labels=QA_TEAM_VERDICTS,
        )
        for role in active_roles
    ]
    team_results = run_read_only_team(
        team_name=f"qa-{iteration}",
        roles=team_roles,
        cwd=cwd,
        command=_agent_command(cfg, reviewer_agent),
        timeout=timeout,
        log_dir=log_dir,
        cost=cost,
        agent_name=reviewer_agent,
        family=getattr(adapters.get(reviewer_agent), "family", "unknown"),
        task="QA",
        step="QA_TEAM",
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
    role_order = {r: i for i, r in enumerate(QA_ROLES)}
    results.sort(key=lambda r: role_order.get(r.role, 99))

    for r in results:
        suffix = " (TIMEOUT)" if r.timed_out else ""
        (qa_dir / f"{r.role}.md").write_text(
            f"# QA Review: {r.role}{suffix}\n\n{r.text}\n"
        )

    synthesis = None
    if synthesize:
        synthesis = _synthesize(
            results,
            adapter,
            reviewer_agent,
            cost,
            timeout,
            cwd,
            routing_config=cfg.data.get("model_routing"),
        )

    report_lines: list[str] = [f"# QA Report — {iteration}\n"]
    report_lines.append(f"**Diff base:** `{diff_base}`\n")
    ok_count = sum(1 for r in results if r.ok)
    report_lines.append(
        f"**Reviewers:** {ok_count}/{len(results)} completed\n\n---\n"
    )
    for r in results:
        status = "OK" if r.ok else ("TIMEOUT" if r.timed_out else "ERROR")
        report_lines.append(f"\n## {r.role.title()} [{status}]\n\n{r.text}\n")
    if synthesis:
        report_lines.append(f"\n---\n\n## Synthesis\n\n{synthesis}\n")

    report_text = "\n".join(report_lines)
    (log_dir / "qa_report.md").write_text(report_text)
    _log(f"QA report written to {log_dir / 'qa_report.md'}")

    return QaReport(
        roles=results,
        synthesis=synthesis,
        diff_base=diff_base,
        output_dir=qa_dir,
    )


def _synthesize(
    results: list[RoleResult],
    adapter: AgentAdapter,
    agent_name: str,
    cost: CostLogger,
    timeout: int,
    cwd: Path,
    routing_config: dict | None = None,
) -> str | None:
    combined = "\n\n---\n\n".join(
        f"## {r.role}\n\n{r.text}" for r in results if r.ok
    )
    prompt = (
        "You are a synthesis reviewer. Read the following QA reports from "
        "5 independent reviewers and produce a unified summary with three "
        "tiers:\n\n"
        "1. **Critical** — must fix before merge\n"
        "2. **Should Fix** — important but not blocking\n"
        "3. **Future** — track for later iterations\n\n"
        "De-duplicate findings across reviewers. Be concise.\n\n"
        f"---\n\n{combined}\n"
    )
    try:
        routing_options = resolve_quality_gate_routing_options(
            routing_config,
            agent_name=agent_name,
        )
        result = adapter.invoke(
            prompt,
            timeout=timeout,
            workdir=cwd,
            routing_options=routing_options,
        )
        cost.record(
            task="QA", step="QA", agent=agent_name,
            **cost_record_usage_kwargs(
                result,
                family=adapter.family,
                provider=agent_name,
            ),
            duration_s=result.duration_s,
            exit_code=result.exit_code,
            partial=result.partial,
            extra=_cost_extra({"role": "synthesis"}, result),
        )
        return result.stdout
    except Exception as exc:
        return f"(synthesis failed: {exc})"


def _agent_command(cfg: LoadedConfig, agent_name: str) -> tuple[str, ...]:
    spec = cfg.data.get("agents", {}).get(agent_name, {})
    cmd = str(spec.get("cmd") or agent_name).strip()
    if not cmd:
        raise ValueError(f"agent {agent_name!r} has no command")
    provider = str(spec.get("type") or spec.get("family") or agent_name)
    return tuple(ensure_usage_json_args(shlex.split(cmd), provider))


def _log(msg: str) -> None:
    print(f"[orch-qa] {msg}", file=sys.stderr, flush=True)
