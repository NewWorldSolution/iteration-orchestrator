"""Guarded auto-merge and CI poll.

The orchestrator consults :func:`evaluate_auto_merge` after review passes.
The guard is a pure function over already-gathered facts; it returns a
:class:`MergeDecision` listing the specific conditions that failed so the
PR comment and readiness report can render a short, actionable reason
list. Anything that fails the guard routes the task to
``NEEDS_HUMAN_MERGE`` — the operator merges manually.

CI poll (:func:`wait_for_ci`) is a thin ``gh`` wrapper that sleeps up to
``ci_wait_seconds`` and returns the resolved status.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from orch.providers import (
    GhProvider,
    GitProvider,
    ShellGhProvider,
    ShellGitProvider,
)


@dataclass
class MergeDecision:
    should_auto_merge: bool
    reasons: list[str] = field(default_factory=list)  # reasons it *failed*

    @property
    def needs_human(self) -> bool:
        return not self.should_auto_merge


@dataclass(frozen=True)
class MergeStrategyContract:
    """Single source for orchestrator task-PR merge strategy assumptions."""

    name: str
    github_flag: str
    local_git_args: tuple[str, ...]


MERGE_STRATEGY = MergeStrategyContract(
    name="merge",
    github_flag="--merge",
    local_git_args=("--no-ff",),
)


def matches_high_risk(
    changed_files: list[str], high_risk_globs: list[str]
) -> list[str]:
    """Return changed files that match any of the high-risk globs."""
    hits: list[str] = []
    for p in changed_files:
        for g in high_risk_globs:
            if fnmatch(p, g):
                hits.append(p)
                break
    return hits


def evaluate_auto_merge(
    *,
    verdict: str,
    changed_files: list[str],
    sensitive_hits: list[str],
    forbidden_hits: list[str],
    diff_insertions: int,
    fix_rounds: int,
    high_risk_globs: list[str],
    auto_merge_cfg: dict,
    ci_passed: bool,
    unresolved_warnings: list[str] | None = None,
) -> MergeDecision:
    """Run the auto-merge guard. Returns a decision with human-readable reasons."""
    reasons: list[str] = []
    no_ci = bool(auto_merge_cfg.get("no_ci", False))

    if verdict != "PASS":
        reasons.append(f"verdict={verdict!r}, expected PASS")
    if sensitive_hits:
        reasons.append(f"sensitive files touched: {sorted(sensitive_hits)}")
    if forbidden_hits:
        reasons.append(f"forbidden patterns present: {sorted(forbidden_hits)}")

    max_ins = int(auto_merge_cfg.get("max_diff_insertions", 500))
    if diff_insertions > max_ins:
        reasons.append(
            f"diff_insertions={diff_insertions} > max {max_ins}"
        )

    hr_hits = matches_high_risk(changed_files, high_risk_globs)
    max_fr_default = int(auto_merge_cfg.get("max_fix_rounds_default", 1))
    max_fr_hr = int(auto_merge_cfg.get("max_fix_rounds_high_risk", 0))
    if hr_hits:
        if fix_rounds > max_fr_hr:
            reasons.append(
                f"high-risk files {hr_hits} with fix_rounds={fix_rounds} > {max_fr_hr}"
            )
    else:
        if fix_rounds > max_fr_default:
            reasons.append(
                f"fix_rounds={fix_rounds} > max {max_fr_default}"
            )

    if not ci_passed and not no_ci:
        reasons.append("CI not green within ci_wait_seconds")
    if unresolved_warnings:
        reasons.append(f"unresolved warnings: {unresolved_warnings}")

    return MergeDecision(should_auto_merge=not reasons, reasons=reasons)


@dataclass(frozen=True)
class TaskPrRequest:
    title: str
    body: str
    base: str
    head: str


def build_task_pr_request(
    *,
    task_id: str,
    task_title: str,
    iteration: str,
    iter_branch: str,
    task_branch: str,
    allowed_files: list[str] | None = None,
    test_cmd: str | None = None,
) -> TaskPrRequest:
    """Build the exact PR request fields used for a task branch.

    The body follows the CLAUDE.md "Final PR description standard" section
    layout (Delivers / Files changed / What was run / Scope / Follow-ups /
    Merge readiness) adapted to a single automated task PR, instead of the old
    vague one-liner that the standard explicitly forbids. It stays
    deterministic — no timestamps or random content — so the same task always
    renders the same body. The iteration->phase PR that the standard primarily
    governs remains operator-owned.
    """
    body = _render_task_pr_body(
        task_id=task_id,
        task_title=task_title,
        iteration=iteration,
        iter_branch=iter_branch,
        task_branch=task_branch,
        allowed_files=allowed_files or [],
        test_cmd=test_cmd,
    )
    return TaskPrRequest(
        title=f"{task_id}: {task_title}",
        body=body,
        base=iter_branch,
        head=task_branch,
    )


def _render_task_pr_body(
    *,
    task_id: str,
    task_title: str,
    iteration: str,
    iter_branch: str,
    task_branch: str,
    allowed_files: list[str],
    test_cmd: str | None,
) -> str:
    if allowed_files:
        files_block = "\n".join(f"- `{path}`" for path in allowed_files)
    else:
        files_block = (
            "- (no explicit allowed-file scope declared for this task; see the "
            "diff)"
        )
    what_was_run = [
        f"- Implemented by the orchestrator on branch `{task_branch}`.",
        "- Passed the per-task review gate before this PR was opened.",
    ]
    if test_cmd:
        what_was_run.append(f"- Scoped acceptance test: `{test_cmd}`")
    else:
        what_was_run.append("- No scoped acceptance test command was declared.")
    lines = [
        "## What this PR delivers",
        f"{task_id} — {task_title}. One task of orchestrator iteration "
        f"`{iteration}`.",
        "",
        "## Files changed",
        "Bounded to this task's allowed-file scope:",
        files_block,
        "",
        "## What was run",
        *what_was_run,
        "",
        "## Scope",
        f"Bounded to the allowed files above; part of iteration `{iteration}`. "
        "Iteration-level QA, retrospective, and the iteration→phase PR are "
        "handled separately.",
        "",
        "## Follow-ups surfaced",
        f"None at task level. Iteration-level QA/retro findings are recorded "
        f"against iteration `{iteration}`.",
        "",
        "## Merge readiness",
        f"Automated per-task PR into `{iter_branch}`. Merge readiness is "
        "decided by the orchestrator guard and the configured CI/no-CI merge "
        "policy; otherwise this PR is left for operator merge. The final "
        "iteration→phase PR description (CLAUDE.md \"Final PR description "
        "standard\") remains operator-owned.",
    ]
    return "\n".join(lines)


def build_needs_human_merge_meta(
    *,
    classification: str,
    msg_detail: str,
    ci_passed: bool,
    pr_url: str,
    iteration: str,
    task_id: str,
    decision_reasons: list[str] | None = None,
    artifact_root_ref: str = "tools/logs",
) -> dict:
    """Build the exact NEEDS_HUMAN_MERGE diagnostic event metadata."""
    meta: dict = {
        "event": "needs_human_merge",
        "classification": classification,
        "msg": msg_detail,
        "ci_passed": ci_passed,
        "recovery": human_merge_recovery_message(
            pr_url=pr_url,
            iteration=iteration,
            task_id=task_id,
            artifact_root_ref=artifact_root_ref,
        ),
    }
    if decision_reasons:
        meta["guard_reasons"] = decision_reasons
    return meta


_MERGE_SHA_RE = re.compile(r"\b([0-9a-f]{7,40})\b")


def parse_merge_sha(gh_output: str) -> str | None:
    try:
        data = json.loads(gh_output or "{}")
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        direct = data.get("merge_sha") or data.get("oid")
        if isinstance(direct, str) and _MERGE_SHA_RE.fullmatch(direct):
            return direct
        merge_commit = data.get("mergeCommit")
        if isinstance(merge_commit, dict):
            oid = merge_commit.get("oid")
            if isinstance(oid, str) and _MERGE_SHA_RE.fullmatch(oid):
                return oid
    m = _MERGE_SHA_RE.search(gh_output or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# gh wrappers — thin, easy to fake in tests
# ---------------------------------------------------------------------------


@dataclass
class CIStatus:
    passed: bool
    conclusion: str  # success | failure | timed_out | pending | unknown
    elapsed_s: float


_DEFAULT_GH_PROVIDER = ShellGhProvider()
_DEFAULT_GIT_PROVIDER = ShellGitProvider()


def _gh(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 60,
    provider: GhProvider | None = None,
) -> subprocess.CompletedProcess:
    return (provider or _DEFAULT_GH_PROVIDER).run(args, cwd=cwd, timeout=timeout)


# ---------------------------------------------------------------------------
# NEEDS_HUMAN_MERGE diagnostics + external-merge detection
# ---------------------------------------------------------------------------


def classify_pr_status(rollup: list[dict]) -> tuple[str, str]:
    """Convert a `gh pr view --json statusCheckRollup` payload to a
    (classification, human_message) pair.

    Used after the CI poll returns NOT-PASSED so the operator gets a
    diagnostic instead of a blank "needs human merge".
    """
    if not rollup:
        return (
            "no-checks-configured",
            "PR has no required status checks. Probable cause: "
            "paths-ignore filter excluded the diff, or branch protection "
            "doesn't require checks for this base branch.",
        )
    in_progress = [
        c.get("name", "?")
        for c in rollup
        if str(c.get("status", "")).upper() == "IN_PROGRESS"
    ]
    if in_progress:
        return (
            "checks-in-progress",
            f"Checks still running: {', '.join(in_progress)}.",
        )
    failed = [
        c.get("name", "?")
        for c in rollup
        if str(c.get("conclusion", "")).upper() == "FAILURE"
    ]
    if failed:
        return (
            "checks-failed",
            f"Checks FAILED: {', '.join(failed)}. Investigate before merging.",
        )
    return ("checks-pending", f"statusCheckRollup unclear: {rollup!r}")


def human_merge_recovery_message(
    *,
    pr_url: str,
    iteration: str,
    task_id: str,
    artifact_root_ref: str = "tools/logs",
) -> str:
    """Render the deterministic 5-step recovery procedure for a manual CI
    merge after the last task, with concrete substitutions."""
    notes_ref = f"{artifact_root_ref.rstrip('/')}/{iteration}/notes.md"
    return "\n".join(
        [
            f"Recovery for {task_id} (PR {pr_url}, iteration {iteration}):",
            "  1. Wait for GitHub Actions CI to go green on the PR.",
            f"  2. gh pr merge {pr_url} {MERGE_STRATEGY.github_flag}",
            "  3. git pull --ff-only on the iteration branch.",
            "  4. (orch resume auto-detects the merge — see step 5; the",
            "      run_state.json edit only matters when --accept-external is",
            "      passed before B2c shipped.)",
            f"  5. python -m orch resume {iteration} --accept-external",
            "Recovery note: record manual CI waits, merge stalls, or "
            "operator-side account changes in "
            f"{notes_ref} before claiming timing evidence.",
        ]
    )


@dataclass
class PrSnapshot:
    state: str            # OPEN | MERGED | CLOSED | unknown
    merge_sha: str | None
    rollup: list[dict]


def is_external_merge_complete(snapshot: PrSnapshot) -> bool:
    return snapshot.state == "MERGED" and bool(snapshot.merge_sha)


def query_pr_state(
    pr_url: str,
    *,
    cwd: Path,
    _run_gh=None,
) -> PrSnapshot:
    """Thin `gh pr view` wrapper used by NEEDS_HUMAN_MERGE diagnostics
    and `orch resume`'s external-merge auto-detection."""
    run_gh = _run_gh or _gh
    proc = run_gh(
        ["pr", "view", pr_url, "--json", "state,mergeCommit,statusCheckRollup"],
        cwd=cwd,
        timeout=60,
    )
    if proc.returncode != 0:
        return PrSnapshot(state="unknown", merge_sha=None, rollup=[])
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return PrSnapshot(state="unknown", merge_sha=None, rollup=[])
    state = str(data.get("state") or "unknown").upper()
    merge_commit = data.get("mergeCommit") or {}
    merge_sha = merge_commit.get("oid") or None
    rollup = data.get("statusCheckRollup") or []
    return PrSnapshot(state=state, merge_sha=merge_sha, rollup=rollup)


def _git(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 60,
    provider: GitProvider | None = None,
) -> subprocess.CompletedProcess:
    return (provider or _DEFAULT_GIT_PROVIDER).run(args, cwd=cwd, timeout=timeout)


def wait_for_ci(
    branch: str,
    *,
    cwd: Path,
    ci_wait_seconds: int,
    poll_interval_s: int = 10,
    _clock=time.monotonic,
    _sleep=time.sleep,
    _run_gh=None,
) -> CIStatus:
    """Poll ``gh pr checks <branch>`` until all checks resolve or time out.

    Abstraction hooks ``_clock`` / ``_sleep`` / ``_run_gh`` exist only for
    tests — production code passes nothing.
    """
    run_gh = _run_gh or _gh
    start = _clock()
    deadline = start + ci_wait_seconds

    while True:
        # `gh pr checks --json` exposes `bucket` (pass|fail|pending|skipping|
        # cancel), NOT `conclusion` — requesting `conclusion` made gh exit 1
        # on every poll, so green CI was never detected and every task parked
        # NEEDS_HUMAN_MERGE despite passing checks. Latent here under no-CI
        # mode; surfaced by a downstream project running under live CI.
        # `name` is diagnostics only.
        proc = run_gh(
            ["pr", "checks", branch, "--json", "name,bucket"],
            cwd=cwd,
            timeout=30,
        )
        rows: list[dict] = []
        if proc.returncode == 0 and proc.stdout.strip():
            try:
                rows = json.loads(proc.stdout)
            except json.JSONDecodeError:
                rows = []

        buckets = [str(r.get("bucket", "")).lower() for r in rows]

        if rows and all(b != "pending" for b in buckets):
            ok = all(b in ("pass", "skipping") for b in buckets)
            return CIStatus(
                passed=ok,
                conclusion="success" if ok else "failure",
                elapsed_s=_clock() - start,
            )

        if _clock() >= deadline:
            return CIStatus(
                passed=False,
                conclusion="timed_out",
                elapsed_s=_clock() - start,
            )
        _sleep(poll_interval_s)


def open_pr(
    *,
    cwd: Path,
    title: str,
    body: str,
    base: str,
    head: str,
    _run_git=None,
    _run_gh=None,
) -> tuple[bool, str]:
    """Create a PR via ``gh pr create``. Returns ``(ok, url_or_stderr)``.

    Force-pushes ``head`` to ``origin`` first. This is load-bearing: the
    repo accumulates stale task-branch refs on remote from previous
    iterations (same branch names, unrelated SHAs). Without this push
    ``gh pr create`` compares the stale remote head to the current remote
    base and either refuses with "No commits between" or opens a PR for
    the wrong commits. ``--force-with-lease`` preserves the safety net
    against concurrent pushes while overwriting stale refs.

    ``_run_git`` / ``_run_gh`` are test injection hooks; production wires
    in the default subprocess-backed runners.
    """
    run_git = _run_git or _git
    run_gh = _run_gh or _gh
    push = run_git(
        ["push", "--force-with-lease", "origin", head],
        cwd=cwd, timeout=120,
    )
    if push.returncode != 0:
        return False, f"git push failed: {push.stderr.strip()}"
    proc = run_gh(
        [
            "pr", "create",
            "--base", base,
            "--head", head,
            "--title", title,
            "--body", body,
        ],
        cwd=cwd,
        timeout=120,
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip()
    return True, proc.stdout.strip()


def merge_pr(*, cwd: Path, pr_url: str, _run_gh=None) -> tuple[bool, str]:
    """Merge ``pr_url`` using the shared task-PR merge strategy contract."""
    run_gh = _run_gh or _gh
    proc = run_gh(
        ["pr", "merge", pr_url, MERGE_STRATEGY.github_flag],
        cwd=cwd,
        timeout=120,
    )
    if proc.returncode != 0:
        return False, proc.stderr.strip()
    view = run_gh(
        ["pr", "view", pr_url, "--json", "mergeCommit"],
        cwd=cwd,
        timeout=60,
    )
    if view.returncode == 0 and parse_merge_sha(view.stdout):
        return True, view.stdout.strip()
    return True, proc.stdout.strip()


def comment_pr(*, cwd: Path, pr_url: str, body: str) -> bool:
    """Attach a human-readable guard report to the PR."""
    proc = _gh(
        ["pr", "comment", pr_url, "--body", body],
        cwd=cwd,
        timeout=60,
    )
    return proc.returncode == 0


def render_guard_comment(
    decision: MergeDecision,
    *,
    no_ci: bool = False,
    iter_branch: str | None = None,
) -> str:
    if decision.should_auto_merge:
        if no_ci:
            branch = f"`{iter_branch}`" if iter_branch else "the iteration branch"
            return "\n".join(
                [
                    "Auto-merge guards passed.",
                    "",
                    "no-CI mode: this PR was merged locally into "
                    f"{branch} by the orchestrator. GitHub will reflect it "
                    "as merged when the iteration branch is pushed after the "
                    "DONE/task-board commit. The per-task review diff is the "
                    "authoritative audit artifact until then.",
                ]
            )
        return "Auto-merge guards passed; merging."
    lines = [
        "Auto-merge blocked. The orchestrator left this PR open because:",
        "",
    ]
    for r in decision.reasons:
        lines.append(f"- {r}")
    lines.append("")
    if no_ci:
        lines.append(
            "no-CI mode skipped the CI wait, but a non-CI guard still blocked "
            "the local merge. Review the diff and resolve the guard before "
            "merging."
        )
    else:
        lines.append("Review the diff and merge manually if acceptable.")
    return "\n".join(lines)
