"""Thin wrappers around ``git`` and ``gh`` for the orchestrator.

Every call shells out — no libgit2, no
GitPython. The wrappers return structured results so higher layers can log
and branch on exit codes without parsing stderr everywhere.

Invariants:
    * Working directory is always passed explicitly via ``cwd``.
    * Destructive operations (``reset --hard``, branch deletion) live only
      in helpers whose name makes intent obvious.
    * HEAD-SHA guard support: :func:`current_sha` is the canonical way to
      read the iter-branch tip.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from orch.merge import MERGE_STRATEGY
from orch.providers import GitProvider, ShellGitProvider


class GitError(RuntimeError):
    """Raised when a git invocation fails unexpectedly."""


class WorktreePreflightError(GitError):
    """Raised when an orch worktree cannot be created for an actionable reason."""


@dataclass
class GitResult:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


_DEFAULT_GIT_PROVIDER = ShellGitProvider()


def orch_workdir(
    repo_root: Path,
    iteration: str,
    *,
    worktree_root: Path | None = None,
) -> Path:
    """Return the dedicated orch sub-worktree path for ``iteration``."""
    root = worktree_root or repo_root / ".orch" / "worktrees"
    if not root.is_absolute():
        root = repo_root / root
    return root / iteration


def task_workdir(
    repo_root: Path,
    iteration: str,
    task_id: str,
    *,
    worktree_root: Path | None = None,
) -> Path:
    """Return the dedicated parallel task worktree path for ``task_id``."""
    root = worktree_root or repo_root / ".orch" / "worktrees"
    if not root.is_absolute():
        root = repo_root / root
    safe_task = re.sub(r"[^A-Za-z0-9._-]+", "-", task_id).strip("-").lower()
    return root / f"{iteration}-{safe_task}"


def ensure_task_workdir(
    repo_root: Path,
    iteration: str,
    task_id: str,
    task_branch: str,
    base_ref: str,
    *,
    provider: GitProvider | None = None,
    worktree_root: Path | None = None,
) -> Path:
    """Create a per-task worktree rooted at ``base_ref``.

    Existing dirty task worktrees fail closed. The helper never removes or
    resets an existing dirty directory, preserving any interrupted work for
    operator inspection.
    """
    workdir = task_workdir(
        repo_root, iteration, task_id, worktree_root=worktree_root
    )
    if workdir.exists():
        if not working_tree_clean(workdir):
            raise WorktreePreflightError(
                f"parallel task worktree for {task_id} is dirty at {workdir}; "
                "inspect or salvage it before retrying"
            )
        return workdir
    workdir.parent.mkdir(parents=True, exist_ok=True)
    proc = (provider or _DEFAULT_GIT_PROVIDER).run(
        ["worktree", "add", "-B", task_branch, str(workdir), base_ref],
        cwd=repo_root,
        timeout=120,
    )
    if proc.returncode != 0:
        raise GitError(
            "git worktree add failed "
            f"({proc.returncode}): {(proc.stderr or '').strip()}"
        )
    return workdir


def _is_checked_out_elsewhere_error(stderr: str) -> bool:
    return "already checked out" in stderr.lower()


def _render_checked_out_elsewhere_message(
    *,
    iteration: str,
    iter_branch: str,
    workdir: Path,
    stderr: str,
) -> str:
    match = re.search(r"already checked out at ['\"]?([^'\"]+)['\"]?", stderr)
    other = match.group(1).strip() if match else "another worktree"
    return (
        f"orch workdir preflight failed for {iteration}: branch "
        f"{iter_branch!r} is already checked out at {other}. Git cannot add "
        f"the orch worktree at {workdir} while that branch is checked out "
        "elsewhere. Next action: run `git worktree list`; if the stale "
        f"checkout is the orch worktree, run `python -m orch "
        f"cleanup-workdir {iteration}`, otherwise switch or remove the other "
        "worktree before retrying."
    )


def ensure_orch_workdir(
    repo_root: Path,
    iteration: str,
    iter_branch: str,
    *,
    provider: GitProvider | None = None,
    worktree_root: Path | None = None,
) -> Path:
    """Create the orch sub-worktree if missing; return its path."""
    workdir = orch_workdir(
        repo_root, iteration, worktree_root=worktree_root
    )
    if workdir.exists():
        return workdir
    workdir.parent.mkdir(parents=True, exist_ok=True)
    proc = (provider or _DEFAULT_GIT_PROVIDER).run(
        ["worktree", "add", str(workdir), iter_branch],
        cwd=repo_root,
        timeout=120,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        if _is_checked_out_elsewhere_error(stderr):
            raise WorktreePreflightError(
                _render_checked_out_elsewhere_message(
                    iteration=iteration,
                    iter_branch=iter_branch,
                    workdir=workdir,
                    stderr=stderr,
                )
            )
        raise GitError(
            "git worktree add failed "
            f"({proc.returncode}): {stderr}"
        )
    return workdir


def cleanup_orch_workdir(
    repo_root: Path,
    iteration: str,
    *,
    provider: GitProvider | None = None,
    worktree_root: Path | None = None,
) -> None:
    """Remove the orch sub-worktree while preserving branches and commits."""
    workdir = orch_workdir(
        repo_root, iteration, worktree_root=worktree_root
    )
    if not workdir.exists():
        return
    proc = (provider or _DEFAULT_GIT_PROVIDER).run(
        ["worktree", "remove", str(workdir), "--force"],
        cwd=repo_root,
        timeout=120,
    )
    if proc.returncode != 0:
        raise GitError(
            "git worktree remove failed "
            f"({proc.returncode}): {(proc.stderr or '').strip()}"
        )


def git(
    args: list[str],
    *,
    cwd: Path,
    check: bool = False,
    timeout: int = 120,
    provider: GitProvider | None = None,
) -> GitResult:
    proc = (provider or _DEFAULT_GIT_PROVIDER).run(args, cwd=cwd, timeout=timeout)
    res = GitResult(
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    if check and not res.ok:
        raise GitError(
            f"git {' '.join(args)} failed ({res.exit_code}): {res.stderr.strip()}"
        )
    return res


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def current_sha(cwd: Path, ref: str = "HEAD") -> str:
    r = git(["rev-parse", ref], cwd=cwd, check=True)
    return r.stdout.strip()


def current_branch(cwd: Path) -> str:
    r = git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, check=True)
    return r.stdout.strip()


def branch_exists(cwd: Path, branch: str) -> bool:
    return git(
        ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], cwd=cwd
    ).ok


def diff_files(cwd: Path, base: str, head: str = "HEAD") -> list[str]:
    r = git(["diff", "--name-only", f"{base}...{head}"], cwd=cwd, check=True)
    return [line for line in r.stdout.splitlines() if line.strip()]


def diff_text(cwd: Path, base: str, head: str = "HEAD") -> str:
    r = git(["diff", f"{base}...{head}"], cwd=cwd, check=True)
    return r.stdout


@dataclass
class DiffStats:
    insertions: int
    deletions: int
    files: int


def diff_stats(cwd: Path, base: str, head: str = "HEAD") -> DiffStats:
    r = git(["diff", "--shortstat", f"{base}...{head}"], cwd=cwd, check=True)
    text = r.stdout.strip()
    if not text:
        return DiffStats(0, 0, 0)
    files = _first_int(re.search(r"(\d+) files? changed", text))
    ins = _first_int(re.search(r"(\d+) insertions?\(\+\)", text))
    dels = _first_int(re.search(r"(\d+) deletions?\(-\)", text))
    return DiffStats(insertions=ins, deletions=dels, files=files)


def _first_int(m: re.Match | None) -> int:
    return int(m.group(1)) if m else 0


class BranchFreshnessCondition(str, Enum):
    """Relationship between a branch and its expected base ref."""

    FRESH = "fresh"
    AHEAD = "ahead"
    BEHIND = "behind"
    DIVERGED = "diverged"
    MISSING_BRANCH = "missing_branch"
    MISSING_BASE = "missing_base"


@dataclass(frozen=True)
class BranchFreshness:
    branch: str
    base_ref: str
    condition: BranchFreshnessCondition
    branch_sha: str | None = None
    base_sha: str | None = None
    ahead_count: int | None = None
    behind_count: int | None = None

    @property
    def contains_base(self) -> bool:
        """True when ``branch`` contains every commit in ``base_ref``."""
        return self.condition in {
            BranchFreshnessCondition.FRESH,
            BranchFreshnessCondition.AHEAD,
        }

    @property
    def missing_base_commits(self) -> bool:
        return self.condition in {
            BranchFreshnessCondition.BEHIND,
            BranchFreshnessCondition.DIVERGED,
        }


def ref_exists(cwd: Path, ref: str) -> bool:
    return git(["rev-parse", "--verify", "--quiet", ref], cwd=cwd).ok


def upstream_ref(cwd: Path, branch: str) -> str | None:
    """Return the configured upstream ref for ``branch`` when available."""
    r = git(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name",
         f"{branch}@{{upstream}}"],
        cwd=cwd,
    )
    return r.stdout.strip() if r.ok and r.stdout.strip() else None


def preferred_remote_ref(
    cwd: Path, branch: str, *, remote: str = "origin"
) -> str:
    """Prefer ``remote/branch`` when it exists, otherwise return ``branch``."""
    if branch.startswith(f"{remote}/"):
        return branch
    remote_ref = f"{remote}/{branch}"
    if ref_exists(cwd, remote_ref):
        return remote_ref
    return branch


def classify_branch_freshness(
    cwd: Path, *, branch: str, base_ref: str
) -> BranchFreshness:
    """Classify whether ``branch`` contains the required ``base_ref`` commits.

    This helper performs no network operations. Callers that want remote
    currency should run ``fetch`` first, then pass the expected local or
    remote-tracking ref.
    """
    branch_sha = _rev_parse_optional(cwd, branch)
    if branch_sha is None:
        return BranchFreshness(
            branch=branch,
            base_ref=base_ref,
            condition=BranchFreshnessCondition.MISSING_BRANCH,
        )

    base_sha = _rev_parse_optional(cwd, base_ref)
    if base_sha is None:
        return BranchFreshness(
            branch=branch,
            base_ref=base_ref,
            condition=BranchFreshnessCondition.MISSING_BASE,
            branch_sha=branch_sha,
        )

    ahead = _rev_count(cwd, f"{base_ref}..{branch}") or 0
    behind = _rev_count(cwd, f"{branch}..{base_ref}") or 0
    if branch_sha == base_sha:
        condition = BranchFreshnessCondition.FRESH
    elif _is_ancestor(cwd, base_ref, branch):
        condition = BranchFreshnessCondition.AHEAD
    elif _is_ancestor(cwd, branch, base_ref):
        condition = BranchFreshnessCondition.BEHIND
    else:
        condition = BranchFreshnessCondition.DIVERGED

    return BranchFreshness(
        branch=branch,
        base_ref=base_ref,
        condition=condition,
        branch_sha=branch_sha,
        base_sha=base_sha,
        ahead_count=ahead,
        behind_count=behind,
    )


def render_branch_freshness_recovery(
    freshness: BranchFreshness, *, gate: str
) -> str:
    """Render operator-safe recovery guidance for a freshness failure."""
    condition = _condition_label(freshness.condition)
    counts = _count_summary(freshness)
    message = (
        f"branch freshness gate failed before {gate}: branch "
        f"'{freshness.branch}' is {condition} relative to expected base "
        f"'{freshness.base_ref}'{counts}."
    )
    if freshness.condition == BranchFreshnessCondition.MISSING_BRANCH:
        return (
            f"{message} Next action: verify the task branch name, create it "
            "from the expected base if this is a new task, then rerun."
        )
    if freshness.condition == BranchFreshnessCondition.MISSING_BASE:
        return (
            f"{message} Next action: run git fetch origin and verify the "
            "expected base ref name; if it still does not exist, stop and "
            "fix the branch configuration."
        )
    if freshness.condition == BranchFreshnessCondition.BEHIND:
        return (
            f"{message} Next action: merge '{freshness.base_ref}' into "
            f"'{freshness.branch}', resolve any conflicts, then "
            "rerun. If the branch should be recreated instead, stop and get "
            "operator approval before discarding work."
        )
    if freshness.condition == BranchFreshnessCondition.DIVERGED:
        return (
            f"{message} Next action: inspect the left/right history, merge "
            f"'{freshness.base_ref}' into '{freshness.branch}' to preserve "
            "work, resolve conflicts, then rerun. If the local work should "
            "be abandoned, stop and get operator approval first."
        )
    return (
        f"branch freshness gate passed before {gate}: branch "
        f"'{freshness.branch}' contains expected base "
        f"'{freshness.base_ref}'{counts}."
    )


def _rev_parse_optional(cwd: Path, ref: str) -> str | None:
    r = git(["rev-parse", "--verify", "--quiet", ref], cwd=cwd)
    return r.stdout.strip() if r.ok and r.stdout.strip() else None


def _rev_count(cwd: Path, rev_range: str) -> int | None:
    r = git(["rev-list", "--count", rev_range], cwd=cwd)
    if not r.ok:
        return None
    try:
        return int(r.stdout.strip() or "0")
    except ValueError:
        return None


def _is_ancestor(cwd: Path, ancestor: str, descendant: str) -> bool:
    return git(["merge-base", "--is-ancestor", ancestor, descendant], cwd=cwd).ok


def _condition_label(condition: BranchFreshnessCondition) -> str:
    return {
        BranchFreshnessCondition.FRESH: "fresh/equal",
        BranchFreshnessCondition.AHEAD: "ahead",
        BranchFreshnessCondition.BEHIND: "stale/behind",
        BranchFreshnessCondition.DIVERGED: "diverged",
        BranchFreshnessCondition.MISSING_BRANCH: "missing",
        BranchFreshnessCondition.MISSING_BASE: "missing-base",
    }[condition]


def _count_summary(freshness: BranchFreshness) -> str:
    if freshness.ahead_count is None or freshness.behind_count is None:
        return ""
    return (
        f" (ahead {freshness.ahead_count} commit(s), "
        f"behind {freshness.behind_count} commit(s))"
    )


# ---------------------------------------------------------------------------
# Mutating helpers — named to flag intent
# ---------------------------------------------------------------------------


def fetch(cwd: Path, remote: str = "origin") -> GitResult:
    return git(["fetch", "--prune", remote], cwd=cwd)


def checkout(cwd: Path, branch: str) -> GitResult:
    return git(["checkout", branch], cwd=cwd, check=True)


def pull_ff_only(cwd: Path) -> GitResult:
    return git(["pull", "--ff-only"], cwd=cwd)


def push_branch(cwd: Path, branch: str, remote: str = "origin") -> GitResult:
    return git(["push", remote, branch], cwd=cwd)


def create_or_reset_branch(cwd: Path, new_branch: str, base: str) -> GitResult:
    """Force-create ``new_branch`` pointing at ``base`` (reset if exists).

    Used at the top of each task to give the implementer a clean slate
    rooted at the current iter-branch tip.
    """
    if branch_exists(cwd, new_branch):
        git(["branch", "-f", new_branch, base], cwd=cwd, check=True)
        return checkout(cwd, new_branch)
    return git(["checkout", "-b", new_branch, base], cwd=cwd, check=True)


def revert_paths(cwd: Path, paths: list[str], base: str) -> GitResult:
    """Restore the named paths to their state at ``base`` (scope-violation recovery).

    Paths that exist at ``base`` are checked out from it; paths that did not
    exist at ``base`` (new files introduced on the current branch) are removed.
    """
    if not paths:
        return GitResult(0, "", "")
    restore, delete = [], []
    for p in paths:
        probe = git(["cat-file", "-e", f"{base}:{p}"], cwd=cwd, check=False)
        if probe.exit_code == 0:
            restore.append(p)
        else:
            delete.append(p)
    last: GitResult = GitResult(0, "", "")
    if restore:
        last = git(["checkout", base, "--", *restore], cwd=cwd, check=True)
    if delete:
        last = git(["rm", "-f", "--", *delete], cwd=cwd, check=True)
    return last


def working_tree_clean(cwd: Path) -> bool:
    """True when there are no uncommitted changes (staged or unstaged)."""
    r = git(["status", "--porcelain"], cwd=cwd, check=True)
    return r.stdout.strip() == ""


def salvage_worktree(cwd: Path, salvage_branch: str, message: str) -> str | None:
    """Park all uncommitted/untracked changes on ``salvage_branch``.

    Returns the salvage commit SHA, or ``None`` when there was nothing to
    salvage or the snapshot could not be created. Salvage is best-effort:
    failures must not turn the original stop into a git crash.
    """
    if working_tree_clean(cwd):
        return None
    original: str | None = None
    try:
        original = current_branch(cwd)
        git(["checkout", "-B", salvage_branch], cwd=cwd, check=True)
        stage_all(cwd)
        commit_res = commit(cwd, message)
        if not commit_res.ok:
            checkout(cwd, original)
            return None
        sha = current_sha(cwd)
        checkout(cwd, original)
        return sha
    except GitError:
        if original:
            try:
                checkout(cwd, original)
            except GitError:
                pass
        return None


def merge_no_ff(cwd: Path, branch: str, message: str) -> GitResult:
    """Merge ``branch`` into HEAD with the shared task-PR strategy."""
    return git(
        ["merge", *MERGE_STRATEGY.local_git_args, "-m", message, branch],
        cwd=cwd,
        check=True,
    )


def stage_all(cwd: Path) -> GitResult:
    return git(["add", "-A"], cwd=cwd)


def commit(cwd: Path, message: str, *, allow_empty: bool = False) -> GitResult:
    args = ["commit", "-m", message]
    if allow_empty:
        args.append("--allow-empty")
    return git(args, cwd=cwd)
