"""Deterministic checks.

All checks are pure functions over inputs the orchestrator has already
collected (changed files, diff text, config) or thin ``subprocess``
wrappers around acceptance commands. No model calls happen here.

The orchestrator calls them in this order:

    3a  check_scope                 → auto-revert once, else STOP(SCOPE)
    3b  check_tasks_md_touched      → auto-revert once, else STOP(SCOPE)
    3c  check_diff_size             → STOP(STRUCTURAL) if > hard limit
        check_forbidden_patterns    → STOP(STRUCTURAL) on any hit
        check_sensitive_files       → STOP(STRUCTURAL) on any hit
    3d  run_acceptance              → fix loop on failure

Conflict-marker detection moved to
``task_execution.diff_introduces_conflict_marker_pair`` (pair-aware; the old
single-marker ``check_conflict_markers`` was removed).
"""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from pathlib import PurePosixPath

from orch.config import CORE_DEFAULTS


# ---------------------------------------------------------------------------
# Scope checks
# ---------------------------------------------------------------------------


def check_scope(
    changed_files: list[str], allowed_files: list[str]
) -> list[str]:
    """Return files that fall outside the allowed set (possibly empty)."""
    allowed = set(allowed_files)
    return [p for p in changed_files if p not in allowed]


def check_tasks_md_touched(
    changed_files: list[str], tasks_md_path: str
) -> bool:
    """True if the agent wrote to tasks.md (forbidden — orchestrator owns it)."""
    return tasks_md_path in changed_files


@dataclass(frozen=True)
class ScopeExceptionEvidence:
    """Parsed operator-approved final scope exceptions.

    Missing evidence files are represented as an empty, valid object. Existing
    evidence files must be well-formed before their paths can authorize
    anything.
    """

    source: str
    approved_by: str | None = None
    reason: str | None = None
    paths: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def approved_paths(self) -> list[str]:
        return list(self.paths) if self.ok else []


def load_scope_exception_evidence(path: Path) -> ScopeExceptionEvidence:
    """Load ``tools/logs/<iter>/scope_exceptions.md`` if present."""
    if not path.exists():
        return ScopeExceptionEvidence(source=str(path))
    return parse_scope_exception_evidence(path.read_text(), source=str(path))


def parse_scope_exception_evidence(
    text: str, *, source: str = "<scope_exceptions>"
) -> ScopeExceptionEvidence:
    """Parse explicit operator-approved final scope exceptions.

    Required shape:

    ``approved_by: <operator identity>``
    ``reason: <why this exact file is allowed>``
    ``paths:``
    ``- exact/relative/file.py``

    Blank lines and ``#`` comments are ignored. Paths are exact relative file
    paths only: no absolutes, parent traversal, globs, or directory blankets.
    """
    approved_by: str | None = None
    reason: str | None = None
    paths: list[str] = []
    errors: list[str] = []
    in_paths = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("approved_by:"):
            if approved_by is not None:
                errors.append(f"{source}:{lineno}: duplicate approved_by")
            approved_by = line.split(":", 1)[1].strip()
            in_paths = False
            continue
        if lowered.startswith("reason:"):
            if reason is not None:
                errors.append(f"{source}:{lineno}: duplicate reason")
            reason = line.split(":", 1)[1].strip()
            in_paths = False
            continue
        if lowered == "paths:":
            in_paths = True
            continue
        if not in_paths:
            errors.append(
                f"{source}:{lineno}: expected approved_by:, reason:, or paths:"
            )
            continue

        entry = line[2:].strip() if line.startswith("- ") else line
        path_error = _validate_scope_exception_path(entry)
        if path_error is not None:
            errors.append(f"{source}:{lineno}: {path_error}")
            continue
        if entry in paths:
            errors.append(f"{source}:{lineno}: duplicate path '{entry}'")
            continue
        paths.append(entry)

    if not approved_by:
        errors.append(f"{source}: missing approved_by")
    if not reason:
        errors.append(f"{source}: missing reason")
    if not paths:
        errors.append(f"{source}: missing paths entries")

    return ScopeExceptionEvidence(
        source=source,
        approved_by=approved_by,
        reason=reason,
        paths=tuple(paths),
        errors=tuple(errors),
    )


def _validate_scope_exception_path(path: str) -> str | None:
    if not path:
        return "empty path entry"
    if path.endswith("/"):
        return f"path '{path}' must be an exact file path, not a directory"
    p = Path(path)
    if p.is_absolute() or any(part == ".." for part in p.parts):
        return f"path '{path}' must be relative and must not contain '..'"
    if any(ch in path for ch in "*?[]"):
        return f"path '{path}' must not contain glob chars"
    if path in (".", "./"):
        return f"path '{path}' must be an exact file path"
    return None


# ---------------------------------------------------------------------------
# Nav-discoverability detection (inward-gap rule)
# ---------------------------------------------------------------------------


DEFAULT_ROUTE_GLOBS = tuple(CORE_DEFAULTS["ui_route_visibility"]["route_globs"])
DEFAULT_NAV_ANCHOR_PATHS = tuple(
    CORE_DEFAULTS["ui_route_visibility"]["nav_anchor_paths"]
)


def is_route_visible_path(
    path: str, route_globs: list[str] | tuple[str, ...] | None = None
) -> bool:
    """Return True when ``path`` looks like a route-visible UI surface.

    Engine defaults are inert. Projects can declare route-visible patterns
    through ``ui_route_visibility.route_globs``.
    """
    if not path:
        return False
    if PurePosixPath(path).name == "__init__.py":
        return False
    patterns = tuple(route_globs or DEFAULT_ROUTE_GLOBS)
    return any(_path_matches_config_glob(path, pattern) for pattern in patterns)


# Path patterns that count as deterministic nav-anchor evidence in the
# iteration diff. When any of these files are also in the diff, the gate
# assumes nav was updated alongside the new surface and passes without
# requiring an evidence file.
def is_nav_anchor_path(
    path: str, nav_anchor_paths: list[str] | tuple[str, ...] | None = None
) -> bool:
    """Return True when ``path`` is a deterministic nav-anchor surface."""
    if not path:
        return False
    return path in set(nav_anchor_paths or DEFAULT_NAV_ANCHOR_PATHS)


def detect_route_visible_surfaces(
    changed_files: list[str],
    route_globs: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return changed paths that introduce route-visible UI surfaces."""
    return sorted({
        p for p in changed_files
        if is_route_visible_path(p, route_globs=route_globs)
    })


def detect_nav_anchor_updates(
    changed_files: list[str],
    nav_anchor_paths: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return changed paths that count as nav-anchor evidence."""
    return sorted({
        p for p in changed_files
        if is_nav_anchor_path(p, nav_anchor_paths=nav_anchor_paths)
    })


def _path_matches_config_glob(path: str, pattern: str) -> bool:
    candidate = PurePosixPath(path)
    configured = PurePosixPath(pattern)
    if candidate.is_absolute() or configured.is_absolute():
        return False
    return _match_glob_parts(tuple(configured.parts), tuple(candidate.parts))


def _match_glob_parts(
    pattern_parts: tuple[str, ...],
    path_parts: tuple[str, ...],
) -> bool:
    if not pattern_parts:
        return not path_parts
    head = pattern_parts[0]
    if head == "**":
        return (
            _match_glob_parts(pattern_parts[1:], path_parts)
            or bool(path_parts)
            and _match_glob_parts(pattern_parts, path_parts[1:])
        )
    if not path_parts:
        return False
    if not fnmatchcase(path_parts[0], head):
        return False
    return _match_glob_parts(pattern_parts[1:], path_parts[1:])


@dataclass(frozen=True)
class NavDiscoverabilityEvidence:
    """Parsed operator-approved no-nav or nav-elsewhere evidence.

    Mirrors :class:`ScopeExceptionEvidence`. Missing evidence is a valid,
    empty object — the gate then expects an in-diff nav anchor instead.
    Malformed evidence is an immediate block: an evidence file that exists
    must be well-formed before any path it lists is honoured.
    """

    source: str
    approved_by: str | None = None
    reason: str | None = None
    paths: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def approved_paths(self) -> list[str]:
        return list(self.paths) if self.ok else []


def load_nav_discoverability_evidence(path: Path) -> NavDiscoverabilityEvidence:
    """Load ``tools/logs/<iter>/nav_discoverability.md`` if present."""
    if not path.exists():
        return NavDiscoverabilityEvidence(source=str(path))
    return parse_nav_discoverability_evidence(
        path.read_text(), source=str(path)
    )


def parse_nav_discoverability_evidence(
    text: str, *, source: str = "<nav_discoverability>"
) -> NavDiscoverabilityEvidence:
    """Parse explicit operator-approved nav-discoverability evidence.

    Required shape (mirrors scope_exceptions for operator familiarity):

    ``approved_by: <operator identity>``
    ``reason: <why this surface is allowed without a nav link, or where
    the nav link lives>``
    ``paths:``
    ``- app/routes/admin_audit.py``

    Each listed path is an exact relative file path that the gate must
    have flagged as route-visible. The evidence does not allow paths the
    gate did not flag — those are matched in the runner against the live
    diff, so a typo here will not silently exempt unrelated files.

    Blank lines and ``#`` comments are ignored. No globs, absolutes, or
    parent traversal.
    """
    approved_by: str | None = None
    reason: str | None = None
    paths: list[str] = []
    errors: list[str] = []
    in_paths = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("approved_by:"):
            if approved_by is not None:
                errors.append(f"{source}:{lineno}: duplicate approved_by")
            approved_by = line.split(":", 1)[1].strip()
            in_paths = False
            continue
        if lowered.startswith("reason:"):
            if reason is not None:
                errors.append(f"{source}:{lineno}: duplicate reason")
            reason = line.split(":", 1)[1].strip()
            in_paths = False
            continue
        if lowered == "paths:":
            in_paths = True
            continue
        if not in_paths:
            errors.append(
                f"{source}:{lineno}: expected approved_by:, reason:, or paths:"
            )
            continue

        entry = line[2:].strip() if line.startswith("- ") else line
        path_error = _validate_scope_exception_path(entry)
        if path_error is not None:
            errors.append(f"{source}:{lineno}: {path_error}")
            continue
        if entry in paths:
            errors.append(f"{source}:{lineno}: duplicate path '{entry}'")
            continue
        paths.append(entry)

    if not approved_by:
        errors.append(f"{source}: missing approved_by")
    if not reason:
        errors.append(f"{source}: missing reason")
    if not paths:
        errors.append(f"{source}: missing paths entries")

    return NavDiscoverabilityEvidence(
        source=source,
        approved_by=approved_by,
        reason=reason,
        paths=tuple(paths),
        errors=tuple(errors),
    )


def check_tasks_md_status_only(
    base_text: str, head_text: str, task_ids: list[str]
) -> bool:
    """True when the only tasks.md changes are task-row Status cells."""
    base_lines = base_text.splitlines()
    head_lines = head_text.splitlines()
    if len(base_lines) != len(head_lines):
        return False

    task_id_set = set(task_ids)
    saw_change = False
    for base_line, head_line in zip(base_lines, head_lines):
        if base_line == head_line:
            continue
        base_cells = _split_md_table_row(base_line)
        head_cells = _split_md_table_row(head_line)
        if base_cells is None or head_cells is None:
            return False
        if len(base_cells) != len(head_cells) or len(base_cells) < 6:
            return False
        base_id = base_cells[0].strip()
        head_id = head_cells[0].strip()
        if base_id != head_id or base_id not in task_id_set:
            return False
        for idx, (base_cell, head_cell) in enumerate(zip(base_cells, head_cells)):
            if idx == 3:
                continue
            if base_cell != head_cell:
                return False
        if (
            not head_cells[3].strip()
            or base_cells[3].strip() == head_cells[3].strip()
        ):
            return False
        saw_change = True
    return saw_change


def _split_md_table_row(line: str) -> list[str] | None:
    if not line.lstrip().startswith("|") or "|" not in line:
        return None
    return line.strip().strip("|").split("|")


# ---------------------------------------------------------------------------
# Structural checks (diff content)
# ---------------------------------------------------------------------------


def check_forbidden_patterns(
    diff_text: str, patterns: list[str]
) -> list[str]:
    """Return the list of forbidden patterns that appear in added lines."""
    added = "\n".join(
        line[1:]
        for line in diff_text.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    return [p for p in patterns if p and p in added]


def check_sensitive_files(
    changed_files: list[str], sensitive_files: list[str]
) -> list[str]:
    """Return the intersection — any hit should halt structural checks."""
    sens = set(sensitive_files)
    # Match on either the full path or the basename (sensitive lists are
    # typically basenames like '.env').
    hits: list[str] = []
    for p in changed_files:
        if p in sens or Path(p).name in sens:
            hits.append(p)
    return hits


def check_diff_size(insertions: int, hard_limit: int) -> bool:
    """True when the diff exceeds the hard insertions cap."""
    return insertions > hard_limit


# ---------------------------------------------------------------------------
# Acceptance (test + lint [+ typecheck + build])
# ---------------------------------------------------------------------------

NOOP_ACCEPTANCE_COMMANDS = frozenset({"true", ":", "/bin/true"})


@dataclass
class CommandResult:
    name: str
    cmd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False


@dataclass
class AcceptanceReport:
    results: list[CommandResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.exit_code == 0 and not r.timed_out for r in self.results)

    @property
    def failures(self) -> list[CommandResult]:
        return [r for r in self.results if r.exit_code != 0 or r.timed_out]

    def combined_output(self) -> str:
        chunks: list[str] = []
        for r in self.results:
            status = (
                "TIMEOUT" if r.timed_out
                else ("OK" if r.exit_code == 0 else f"FAIL({r.exit_code})")
            )
            chunks.append(f"--- {r.name} [{status}] ---")
            if r.stdout.strip():
                chunks.append(r.stdout.rstrip())
            if r.stderr.strip():
                chunks.append(r.stderr.rstrip())
        return "\n".join(chunks)


def effective_acceptance_test_command(
    stack: dict,
    *,
    test_cmd_override: str | None = None,
) -> str | None:
    """Return the test command the acceptance step will actually execute."""
    return test_cmd_override or stack.get("test")


def is_noop_acceptance_command(cmd: str | None) -> bool:
    """Return True when a test command provides no acceptance signal."""
    if cmd is None:
        return True
    normalized = cmd.strip()
    if not normalized:
        return True
    try:
        parts = shlex.split(normalized)
    except ValueError:
        parts = []
    return normalized in NOOP_ACCEPTANCE_COMMANDS or (
        len(parts) == 1 and parts[0] in NOOP_ACCEPTANCE_COMMANDS
    )


def acceptance_test_command_is_noop(
    stack: dict,
    *,
    test_cmd_override: str | None = None,
) -> bool:
    return is_noop_acceptance_command(
        effective_acceptance_test_command(
            stack,
            test_cmd_override=test_cmd_override,
        )
    )


def run_acceptance(
    stack: dict,
    *,
    cwd: Path,
    timeout: int,
    test_cmd_override: str | None = None,
) -> AcceptanceReport:
    """Run each configured acceptance command once, collecting results.

    ``stack`` is the merged ``stack:`` section from config. Commands run in
    order: test, lint, typecheck (optional), build (optional). ``test_env``
    is merged into the process environment. The first-failure-wins policy
    is implemented by the caller (we run all and let them choose).

    If ``test_cmd_override`` is provided, it replaces the stack test command.
    This allows per-task test scoping (v1.5 — P1-3).
    """
    env = dict(os.environ)
    env.update(stack.get("test_env") or {})

    test_cmd = effective_acceptance_test_command(
        stack,
        test_cmd_override=test_cmd_override,
    )
    report = AcceptanceReport()
    order = [("test", test_cmd),
             ("lint", stack.get("lint")),
             ("typecheck", stack.get("typecheck")),
             ("build", stack.get("build"))]

    for name, cmd in order:
        if not cmd:
            continue
        report.results.append(_run_one(name, cmd, cwd=cwd, env=env, timeout=timeout))
    return report


def _run_one(
    name: str, cmd: str, *, cwd: Path, env: dict, timeout: int
) -> CommandResult:
    import time
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            name=name,
            cmd=cmd,
            exit_code=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration_s=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            cmd=cmd,
            exit_code=-1,
            stdout=(exc.stdout or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stdout, (bytes, bytearray)) else (exc.stdout or ""),
            stderr=(exc.stderr or b"").decode("utf-8", errors="replace")
                if isinstance(exc.stderr, (bytes, bytearray)) else (exc.stderr or ""),
            duration_s=time.monotonic() - start,
            timed_out=True,
        )
