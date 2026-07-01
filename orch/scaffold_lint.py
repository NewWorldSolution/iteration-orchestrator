"""Scaffold-consistency linter for iterations/.

Usage:
    python -m orch.scaffold_lint iterations/demo/demo-i14 iterations/tools/example ...
    python -m orch.scaffold_lint --all   # validate every iteration found under iterations/

Exits 0 on green, 1 on any inconsistency.

Checks:
1. Task count claimed in prompt.md (e.g. "Six tasks", "Seven tasks", "Three tasks")
   matches the number of task rows in tasks.md AND matches the number of
   prompts/t<k>-*.md files.
2. Every prompts/t<k>-*.md has a matching reviews/review-t<k>.md.
3. **Depends on:** fields use em-dash — for "no dependency", not double-hyphen --.
4. iteration directory has the four required files: prompt.md, tasks.md,
   prompts/, reviews/.
5. Runnable post-phase prompts must target the configured integration branch,
   not main.
"""
from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

# Running this file directly (`python tools/scaffold_lint.py`) puts tools/ on
# sys.path[0], not the repo root, so the absolute `orch.*` imports below
# (and their transitive `from orch...` imports) fail with ModuleNotFoundError.
# Bootstrap the repo root so these resolve identically whether the file is run as
# a script or imported as the `orch.scaffold_lint` module.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from orch.checks import is_route_visible_path  # noqa: E402
from orch.config import (  # noqa: E402
    CORE_DEFAULTS,
    ConfigError,
    default_project_yaml_path,
    load_config,
)
from orch.paths import resolve_orch_paths  # noqa: E402

WORD_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

def find_iteration_dirs(
    root: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> list[Path]:
    """Discover every iteration directory under root.

    An iteration directory is one that contains both prompt.md and tasks.md.
    """
    policy = policy or default_scaffold_policy()
    out: list[Path] = []
    for prompt in root.rglob(policy.prompt_filename):
        if (prompt.parent / policy.task_board_filename).exists():
            out.append(prompt.parent)
    return sorted(out)


def claimed_task_count(prompt_text: str) -> int | None:
    """Return the task count claimed in prompt.md, or None if no claim found.

    Looks for headings like '## Six tasks' or '## Seven tasks' or '## Tasks'
    or 'Three tasks (T1-T3)'. Returns the integer.
    """
    # Pattern: word number + "tasks"
    for word, n in WORD_TO_INT.items():
        if re.search(rf"\b{word}\b\s+tasks\b", prompt_text, re.IGNORECASE):
            return n
    # Pattern: digit + "tasks"
    m = re.search(r"\b(\d+)\s+tasks\b", prompt_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def task_rows_in_tasks_md(tasks_text: str) -> int:
    """Count the rows in the FIRST task table only.

    Real iteration tasks.md files (e.g., iterations/phase-demo/demo-i1/tasks.md)
    contain MULTIPLE markdown tables that reuse the same task IDs:
      - The primary Tasks table (canonical row count)
      - A secondary "Prompts & Reviews" or similar summary table later in the
        file that also lists T1, T2, T3 etc.

    A naive global count of `| T<n> |` rows double-counts and false-fails
    --all on the existing repo. The fix: count the FIRST contiguous block
    of task rows only. As soon as a non-task-row line appears after we've
    started counting (blank line, non-table prose, separator, new heading),
    we've reached the end of the first table and stop.

    Looks for table rows of shape:
        | T<n>  | Title | Status | ...
    or:
        | I<n>-T<k>  | Title | ...
    """
    count = 0
    seen_first = False
    for line in tasks_text.splitlines():
        s = line.strip()
        if re.match(r"\|\s*(T\d+|I\d+-T\d+)\b", s):
            count += 1
            seen_first = True
        elif seen_first:
            # We've started counting and hit a non-task-row line → end of
            # the first task table. Stop, don't count rows in any later table.
            break
    return count


def prompt_files(prompts_dir: Path) -> list[str]:
    """Return sorted list of t<k> prefixes for prompt files."""
    if not prompts_dir.exists():
        return []
    return sorted(
        p.stem.split("-")[0]
        for p in prompts_dir.glob("t*.md")
    )


def review_files(reviews_dir: Path) -> list[str]:
    """Return sorted list of t<k> prefixes for review files."""
    if not reviews_dir.exists():
        return []
    return sorted(
        p.stem.replace("review-", "").split("-")[0]
        for p in reviews_dir.glob("review-t*.md")
    )


def has_double_hyphen_no_dep(tasks_text: str) -> list[str]:
    """Return a list of bad lines that use '--' instead of '—' for no-dependency.

    Only the depends-on column (the last pipe-delimited cell of a task row) is
    inspected, so a ``--flag`` mentioned in a Title or Test cell (e.g.
    ``--skip-impl``) is not a false positive.
    """
    bad: list[str] = []
    for line in tasks_text.splitlines():
        s = line.strip()
        if not re.match(r"\|\s*(T\d+|I\d+-T\d+)\b", s):
            continue
        # Depends-on is the last pipe-delimited cell of the row.
        dep = s.strip("|").split("|")[-1].strip()
        if "--" in dep and "—" not in dep:
            bad.append(line)
    return bad


_TEST_FIELD_RE = re.compile(r"^\*\*Test:\*\*\s*(.*)$")
_ALLOWED_RUNNERS = ("pytest", "python -m", "ruff", "npm", "npx", "bash -c")


@dataclass(frozen=True)
class ScaffoldPolicy:
    iteration_root: Path
    prompt_filename: str
    task_board_filename: str
    task_prompts_dirname: str
    task_reviews_dirname: str
    generated_artifact_prefixes: tuple[str, ...]
    post_phase_docs_root: Path
    post_phase_iteration_root: Path
    tooling_iteration_root: Path
    post_phase_integration_branch: str
    route_globs: tuple[str, ...]
    task_detail_heading_pattern: str


def default_scaffold_policy() -> ScaffoldPolicy:
    paths = CORE_DEFAULTS["paths"]
    scaffold = CORE_DEFAULTS["scaffold"]
    visibility = CORE_DEFAULTS["ui_route_visibility"]
    patterns = CORE_DEFAULTS["patterns"]
    return ScaffoldPolicy(
        iteration_root=Path(paths["iteration_root"]),
        prompt_filename=str(paths["iteration_prompt_filename"]),
        task_board_filename=str(paths["task_board_filename"]),
        task_prompts_dirname=str(paths["task_prompts_dir"]),
        task_reviews_dirname=str(paths["task_reviews_dir"]),
        generated_artifact_prefixes=tuple(
            paths["generated_artifact_exclusion_prefixes"]
        ),
        post_phase_docs_root=Path(scaffold["post_phase_docs_root"]),
        post_phase_iteration_root=Path(scaffold["post_phase_iteration_root"]),
        tooling_iteration_root=Path(scaffold["tooling_iteration_root"]),
        post_phase_integration_branch=str(
            scaffold["post_phase_integration_branch"]
        ),
        route_globs=tuple(visibility["route_globs"]),
        task_detail_heading_pattern=str(patterns["task_detail_heading"]),
    )


def load_scaffold_policy(repo_root: Path) -> ScaffoldPolicy:
    config_path = default_project_yaml_path(repo_root)
    if not config_path.exists():
        return default_scaffold_policy()
    cfg = load_config(config_path)
    orch_paths = resolve_orch_paths(repo_root, cfg)
    scaffold = cfg.data["scaffold"]
    visibility = cfg.data["ui_route_visibility"]
    patterns = cfg.data["patterns"]
    return ScaffoldPolicy(
        iteration_root=Path(
            orch_paths.iteration_root.relative_to(repo_root).as_posix()
        ),
        prompt_filename=orch_paths.iteration_prompt_filename,
        task_board_filename=orch_paths.task_board_filename,
        task_prompts_dirname=orch_paths.task_prompts_dirname,
        task_reviews_dirname=orch_paths.task_reviews_dirname,
        generated_artifact_prefixes=orch_paths.generated_artifact_exclusion_prefixes,
        post_phase_docs_root=Path(scaffold["post_phase_docs_root"]),
        post_phase_iteration_root=Path(scaffold["post_phase_iteration_root"]),
        tooling_iteration_root=Path(scaffold["tooling_iteration_root"]),
        post_phase_integration_branch=str(
            scaffold["post_phase_integration_branch"]
        ),
        route_globs=tuple(visibility["route_globs"]),
        task_detail_heading_pattern=str(patterns["task_detail_heading"]),
    )

_DIRECT_MAIN_TARGET_PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = (
    (re.compile(r"\bgit\s+checkout\s+main(?:\s|$)"), "git checkout main", True),
    (
        re.compile(r"\bgit\s+checkout\s+-b\b[^\n]*\borigin/main\b"),
        "git checkout -b ... origin/main",
        True,
    ),
    (
        re.compile(r"\bBranch base:\s*`?origin/main`?\b", re.IGNORECASE),
        "Branch base: origin/main",
        False,
    ),
    (
        re.compile(r"\bbranch(?:ing)?\s+from\s+`?origin/main`?\b", re.IGNORECASE),
        "branch from origin/main",
        False,
    ),
    (
        re.compile(r"\bgh\s+pr\s+create\b[^\n]*\s--base(?:=|\s+)main\b"),
        "gh pr create --base main",
        True,
    ),
    (
        re.compile(r"^\s*(?:[-*]\s*)?PR base:\s*`?main`?\s*$", re.IGNORECASE),
        "PR base: main",
        False,
    ),
)
_MAIN_PR_LIST_RE = re.compile(
    r"\bgh\s+pr\s+list\b[^\n]*\s--base(?:=|\s+)main\b"
)


def _strip_inline_code(value: str) -> str:
    """If value is wrapped in a single pair of backticks, strip them.

    Tasks.md convention is to wrap the shell command in inline-code backticks,
    e.g. ``**Test:** `pytest tests/foo.py -v` ``. Strip the wrapper so the
    inner content can be validated.
    """
    v = value.strip()
    if len(v) >= 2 and v.startswith("`") and v.endswith("`"):
        # Only strip if inner content has no extra backticks (single inline span)
        inner = v[1:-1]
        if "`" not in inner:
            return inner
    return v


def _test_field_runnable(tasks_text: str) -> list[str]:
    """Validate every ``**Test:**`` line in tasks.md is a runnable shell command.

    Rules:
      1. The value must not contain unescaped backticks beyond an outer
         inline-code wrapper (an early launch hit ``STOPPED:CHECKS`` because
         prose backticks were interpreted as command substitution).
      2. The value must start with one of the allowed runners
         (``pytest``, ``python -m``, ``ruff``, ``npm``, ``npx``, ``bash -c``).
      3. ``shlex.split`` must succeed on the value (catches unbalanced quotes).

    Empty ``**Test:**`` lines (no command after the marker) are skipped — those
    indicate "manual test, no automation", not a malformed shell command.
    """
    errors: list[str] = []
    for lineno, line in enumerate(tasks_text.splitlines(), start=1):
        m = _TEST_FIELD_RE.match(line)
        if not m:
            continue
        raw = m.group(1).strip()
        if not raw:
            continue
        value = _strip_inline_code(raw)
        if not value:
            continue
        if "`" in value:
            errors.append(
                f"line {lineno}: **Test:** contains unescaped backticks "
                f"(would be interpreted as command substitution by bash): {raw!r}"
            )
            continue
        if not any(value == r or value.startswith(r + " ") for r in _ALLOWED_RUNNERS):
            errors.append(
                f"line {lineno}: **Test:** must start with one of "
                f"{_ALLOWED_RUNNERS}; got: {value.split()[0]!r}"
            )
            continue
        try:
            shlex.split(value)
        except ValueError as exc:
            errors.append(
                f"line {lineno}: **Test:** is not shell-tokenizable ({exc}): {value!r}"
            )
    return errors


def check_test_fields_runnable(tasks_text: str) -> list[str]:
    """Public wrapper over :func:`_test_field_runnable`.

    Lets other tools (e.g. the Prompt Factory renderer/validator) enforce the
    exact same ``**Test:**`` runnability rule that ``orch validate`` applies,
    so a draft can never render a ``**Test:**`` line that this lint would later
    reject. Single source of truth; keep callers using this rather than
    reimplementing the allowed-runner list.
    """
    return _test_field_runnable(tasks_text)


def _relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_post_phase_runnable_prompt(
    path: Path,
    repo_root: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> bool:
    """Return True for active post-phase prompts, not archived reports/docs."""
    policy = policy or default_scaffold_policy()
    rel = _relative_posix(path, repo_root)
    if not rel.endswith(".md"):
        return False
    docs_root = policy.post_phase_docs_root
    iter_root = policy.post_phase_iteration_root
    if rel.startswith((docs_root / policy.task_prompts_dirname).as_posix() + "/"):
        return not Path(rel).name.startswith("review-")
    if rel.startswith(docs_root.as_posix() + "/"):
        return Path(rel).parent.as_posix() == docs_root.as_posix() and (
            Path(rel).name.endswith("_prompt.md")
            or Path(rel).name.endswith("_recheck_prompt.md")
        )
    if rel == (iter_root / "_implementation-prompt-template.md").as_posix():
        return True
    if rel.startswith(iter_root.as_posix() + "/"):
        parts = rel.split("/")
        iter_parts = len(iter_root.parts)
        if len(parts) == iter_parts + 2 and parts[-1] == policy.prompt_filename:
            return True
        return (
            len(parts) == iter_parts + 3
            and parts[-2] == policy.task_prompts_dirname
        )
    return False


def _prompt_files_under(
    target: Path,
    repo_root: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> list[Path]:
    policy = policy or default_scaffold_policy()
    if target.is_file():
        return (
            [target]
            if _is_post_phase_runnable_prompt(
                target, repo_root, policy=policy
            )
            else []
        )
    if not target.is_dir():
        return []
    return sorted(
        p for p in target.rglob("*.md")
        if _is_post_phase_runnable_prompt(p, repo_root, policy=policy)
    )


def discover_post_phase_runnable_prompts(
    repo_root: Path,
    *,
    include_docs: bool = True,
    include_iterations: bool = True,
    policy: ScaffoldPolicy | None = None,
) -> list[Path]:
    """Find runnable post-phase prompt surfaces covered by this policy."""
    policy = policy or default_scaffold_policy()
    roots: list[Path] = []
    if include_docs:
        roots.append(repo_root / policy.post_phase_docs_root)
    if include_iterations:
        roots.append(repo_root / policy.post_phase_iteration_root)
    files: list[Path] = []
    for root in roots:
        files.extend(_prompt_files_under(root, repo_root, policy=policy))
    return sorted(set(files))


def _line_window(lines: list[str], index: int, before: int = 4, after: int = 4) -> str:
    start = max(0, index - before)
    end = min(len(lines), index + after + 1)
    return "\n".join(lines[start:end])


def _is_shell_command_context(line: str, in_fence: bool) -> bool:
    if in_fence:
        return True
    stripped = line.strip()
    if "`" in stripped:
        return False
    return (
        stripped.startswith(("git ", "gh ", "$ git ", "$ gh "))
        or "=$(gh pr list" in stripped
    )


def post_phase_integration_policy_errors_for_file(
    prompt_path: Path,
    repo_root: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    """Reject runnable post-phase prompts that target main directly."""
    policy = policy or default_scaffold_policy()
    if not _is_post_phase_runnable_prompt(
        prompt_path, repo_root, policy=policy
    ):
        return []
    rel = _relative_posix(prompt_path, repo_root)
    text = prompt_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    errors: list[str] = []
    in_fence = False
    for lineno, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        is_command_context = _is_shell_command_context(line, in_fence)
        for pattern, label, requires_command_context in _DIRECT_MAIN_TARGET_PATTERNS:
            if requires_command_context and not is_command_context:
                continue
            if pattern.search(line):
                errors.append(
                    f"{rel}:{lineno}: post-phase runnable prompt targets main "
                    f"via {label}: {line.strip()!r}; use branch base "
                    f"`origin/{policy.post_phase_integration_branch}` and PR "
                    f"base `{policy.post_phase_integration_branch}` instead."
                )
        if is_command_context and _MAIN_PR_LIST_RE.search(line):
            errors.append(
                f"{rel}:{lineno}: post-phase runnable prompt queries main "
                "with `gh pr list --base main`; use "
                f"`{policy.post_phase_integration_branch}` for current "
                "PR checks."
            )
    return errors


# ---------------------------------------------------------------------------
# Nav-discoverability scaffold check (inward-gap rule)
# ---------------------------------------------------------------------------

# Path patterns whose presence in a task's "Allowed files" block flags
# the task as introducing route-visible UI. Mirrors
# ``orch.checks.is_route_visible_path`` so prompt-time discipline and
# runtime gate agree on what counts.
def _is_route_visible_allowed_path(
    path: str,
    *,
    policy: ScaffoldPolicy | None = None,
) -> bool:
    policy = policy or default_scaffold_policy()
    if not path:
        return False
    return is_route_visible_path(path, route_globs=policy.route_globs)


# Markers that satisfy the inward-gap rule when they appear in a prompt or
# review body. Either an explicit nav-discoverability note OR an explicit
# no-nav justification / no-nav decision passes. The bare ``No-nav:`` /
# ``No nav:`` short form is intentionally NOT accepted — the contract is
# an explicit justification or decision so reviewers can audit intent.
_NAV_MARKER_RE = re.compile(
    r"(?:^|\n)(?:#{2,4}\s+|\*\*)\s*"
    r"(?:Nav discoverability"
    r"|Nav/discoverability"
    r"|Navigation discoverability"
    r"|No[- ]nav justification"
    r"|No[- ]nav decision)"
    r"\s*[:*]?",
    re.IGNORECASE,
)

# Detail section headers in iteration tasks.md files appear in two
# documented shapes:
#   ``### I<n>-T<k> — Title``  (phase iterations, em-dash only)
#   ``### T<k> — Title``        (tooling sprints / post-phase tracks)
#   ``### T<k> - Title``        (some tooling sprints use ASCII hyphen)
# All three must parse so forward-looking iterations are covered.
_DETAIL_HEADER_RE = re.compile(
    r"^###\s+(?P<id>(?:I\d+-)?T\d+)\s+[—-]\s+"
)


def _detail_header_id(
    line: str,
    *,
    policy: ScaffoldPolicy,
) -> str | None:
    try:
        configured = re.compile(policy.task_detail_heading_pattern)
    except re.error:
        configured = None
    if configured is not None:
        match = configured.match(line)
        if match is not None and "id" in match.groupdict():
            return match.group("id")
    legacy = _DETAIL_HEADER_RE.match(line)
    return legacy.group("id") if legacy is not None else None


def _post_phase_iteration(
    iter_dir: Path,
    repo_root: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> bool:
    """Return True for forward-looking iteration directories.

    The inward-gap rule applies to configured post-phase and tooling-sprint
    iteration roots.
    """
    policy = policy or default_scaffold_policy()
    rel = _relative_posix(iter_dir, repo_root)
    return (
        rel.startswith(policy.post_phase_iteration_root.as_posix() + "/")
        or rel.startswith(policy.tooling_iteration_root.as_posix() + "/")
    )


def extract_task_allowed_files(
    tasks_text: str,
    *,
    policy: ScaffoldPolicy | None = None,
) -> dict[str, list[str]]:
    """Return ``{task_id: [allowed_files]}`` from tasks.md detail sections.

    Parses ``### I<n>-T<k> — Title`` headers and their following
    ``**Allowed files:**`` fenced blocks. Inline arrow comments inside
    entries are stripped. Empty results when tasks.md has no detail
    sections (e.g. malformed or in-progress files).
    """
    policy = policy or default_scaffold_policy()
    out: dict[str, list[str]] = {}
    lines = tasks_text.splitlines()
    current_id: str | None = None
    in_allowed = False
    in_fence = False
    allowed: list[str] = []

    for line in lines:
        header_id = _detail_header_id(line, policy=policy)
        if header_id is not None:
            if current_id is not None:
                out[current_id] = allowed
            current_id = header_id
            allowed = []
            in_allowed = False
            in_fence = False
            continue
        if current_id is None:
            continue
        stripped = line.strip()
        if stripped.startswith("**Allowed files:**"):
            in_allowed = True
            in_fence = False
            continue
        if in_allowed and stripped.startswith("```"):
            if in_fence:
                in_fence = False
                in_allowed = False
            else:
                in_fence = True
            continue
        if in_allowed and in_fence:
            entry = stripped.strip("`").strip()
            # Strip inline arrow comments like "path  <- note".
            entry = re.split(r"\s+(?:<--|<-|->|→|←)\s+", entry, maxsplit=1)[0]
            entry = entry.strip()
            if entry:
                allowed.append(entry)

    if current_id is not None:
        out[current_id] = allowed
    return out


def has_nav_marker(text: str) -> bool:
    """Return True when ``text`` contains a recognised nav-discoverability
    or no-nav marker line."""
    return bool(_NAV_MARKER_RE.search(text))


_TASK_ID_RE = re.compile(r"(?:^|-)(T\d+)$")


def _task_artifact_prefixes(task_id: str) -> list[str]:
    prefixes: list[str] = []
    legacy = _TASK_ID_RE.search(task_id)
    if legacy is not None:
        prefixes.append(legacy.group(1).lower())
    normalized = re.sub(r"[^a-z0-9]+", "-", task_id.lower()).strip("-")
    if normalized:
        prefixes.append(normalized)
    out: list[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        seen.add(prefix)
        out.append(prefix)
    return out


def _task_artifact_text(
    task_id: str,
    *,
    prompts_dir: Path,
    reviews_dir: Path,
) -> tuple[str, list[str]]:
    """Aggregate every prompt + review file that maps to ``task_id``.

    Returns ``(combined_text, file_list)`` where ``file_list`` is the
    relative paths of artifacts that were read (for clearer error
    messages). Files that fail to read are skipped silently — the
    structural-files check already catches missing prompts/reviews.

    Supports both ``I<n>-T<k>`` and bare ``T<k>`` task IDs — prompt files
    are conventionally named ``t<k>-*.md`` in both cases.
    """
    prefixes = _task_artifact_prefixes(task_id)
    if not prefixes:
        return "", []
    combined = ""
    sources: list[str] = []
    prompt_candidates = sorted({
        candidate
        for prefix in prefixes
        for pattern in (f"{prefix}.md", f"{prefix}-*.md")
        for candidate in prompts_dir.glob(pattern)
    })
    for candidate in prompt_candidates:
        try:
            combined += candidate.read_text(encoding="utf-8") + "\n"
            sources.append(candidate.name)
        except OSError:
            continue
    if reviews_dir.is_dir():
        review_candidates = sorted({
            candidate
            for prefix in prefixes
            for pattern in (f"review-{prefix}.md", f"review-{prefix}-*.md")
            for candidate in reviews_dir.glob(pattern)
        })
        for candidate in review_candidates:
            try:
                combined += candidate.read_text(encoding="utf-8") + "\n"
                sources.append(candidate.name)
            except OSError:
                continue
    return combined, sources


def nav_discoverability_iteration_errors(
    iter_dir: Path,
    *,
    repo_root: Path | None = None,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    """Return inward-gap errors for one iteration directory."""
    repo_root = repo_root or Path.cwd()
    policy = policy or default_scaffold_policy()
    if not _post_phase_iteration(iter_dir, repo_root, policy=policy):
        return []
    tasks_path = iter_dir / policy.task_board_filename
    prompts_dir = iter_dir / policy.task_prompts_dirname
    reviews_dir = iter_dir / policy.task_reviews_dirname
    if not tasks_path.exists() or not prompts_dir.is_dir():
        return []

    tasks_text = tasks_path.read_text(encoding="utf-8")
    task_allowed = extract_task_allowed_files(tasks_text, policy=policy)
    errors: list[str] = []
    for tid, allowed in task_allowed.items():
        route_paths = [
            p for p in allowed
            if _is_route_visible_allowed_path(p, policy=policy)
        ]
        if not route_paths:
            continue
        combined, sources = _task_artifact_text(
            tid, prompts_dir=prompts_dir, reviews_dir=reviews_dir,
        )
        if not has_nav_marker(combined):
            errors.append(
                f"{iter_dir}: task {tid} declares route-visible allowed files "
                f"{route_paths} but neither its prompt nor its review "
                "contains a '**Nav discoverability:**' or "
                "'**No-nav justification:**' marker "
                f"(checked: {sources or 'no matching prompt/review files'}). "
                "Add explicit nav evidence in the task prompt/review or "
                "record an explicit no-nav decision."
            )
    return errors


# Allowed-files block in standalone post-phase prompts uses the same
# fenced-block convention as iteration prompts. Re-implementing the small
# extractor here keeps this module independent of scaffold_review.
_ALLOWED_HEADING_RE = re.compile(
    r"^(?:#{2,4}\s+Allowed Files\b|\*\*Allowed Files:\*\*)",
    re.IGNORECASE,
)


def _extract_allowed_block_paths(text: str) -> list[str]:
    """Pull file entries from the first ``Allowed files`` fenced block."""
    lines = text.splitlines()
    in_section = False
    in_fence = False
    out: list[str] = []
    for line in lines:
        if _ALLOWED_HEADING_RE.match(line.strip()):
            in_section = True
            continue
        if not in_section:
            continue
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                break
            in_fence = True
            continue
        if not in_fence:
            continue
        entry = stripped.strip("`").strip()
        if entry:
            out.append(entry)
    return out


def nav_discoverability_doc_prompt_errors(
    prompt_path: Path,
    repo_root: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    """Return inward-gap errors for a standalone post-phase prompt."""
    policy = policy or default_scaffold_policy()
    if not _is_post_phase_runnable_prompt(
        prompt_path, repo_root, policy=policy
    ):
        return []
    rel = _relative_posix(prompt_path, repo_root)
    # Only scan doc-style implementation prompts. Review prompts have no
    # Allowed-files block; iteration prompts use the iteration path.
    if not rel.startswith(policy.post_phase_docs_root.as_posix() + "/"):
        return []
    text = prompt_path.read_text(encoding="utf-8")
    allowed = _extract_allowed_block_paths(text)
    route_paths = [
        p for p in allowed
        if _is_route_visible_allowed_path(p, policy=policy)
    ]
    if not route_paths:
        return []
    if has_nav_marker(text):
        return []
    return [
        f"{rel}: prompt declares route-visible allowed files {route_paths} "
        "but does not contain a '**Nav discoverability:**' or "
        "'**No-nav justification:**' marker; add explicit nav evidence in "
        "the prompt body or record an explicit no-nav decision."
    ]


def nav_discoverability_doc_prompt_errors_all(
    repo_root: Path,
    *,
    targets: list[Path] | None = None,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    """Aggregate inward-gap errors across runnable post-phase prompts."""
    policy = policy or default_scaffold_policy()
    if targets is None:
        prompts = discover_post_phase_runnable_prompts(
            repo_root,
            include_docs=True,
            include_iterations=False,
            policy=policy,
        )
    else:
        prompts = sorted({
            prompt
            for target in targets
            for prompt in _prompt_files_under(
                target, repo_root, policy=policy
            )
        })
    errors: list[str] = []
    for prompt in prompts:
        errors.extend(
            nav_discoverability_doc_prompt_errors(
                prompt, repo_root, policy=policy
            )
        )
    return errors


def post_phase_integration_policy_errors(
    repo_root: Path,
    *,
    targets: list[Path] | None = None,
    include_docs: bool = True,
    include_iterations: bool = True,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    """Return all post-phase integration-branch policy errors."""
    policy = policy or default_scaffold_policy()
    prompt_files = (
        discover_post_phase_runnable_prompts(
            repo_root,
            include_docs=include_docs,
            include_iterations=include_iterations,
            policy=policy,
        )
        if targets is None
        else sorted(
            {
                prompt
                for target in targets
                for prompt in _prompt_files_under(
                    target, repo_root, policy=policy
                )
            }
        )
    )
    errors: list[str] = []
    for prompt in prompt_files:
        errors.extend(
            post_phase_integration_policy_errors_for_file(
                prompt, repo_root, policy=policy
            )
        )
    return errors


def lint_iteration(
    iter_dir: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    """Validate one iteration directory. Return list of error messages (empty = clean)."""
    policy = policy or default_scaffold_policy()
    errors: list[str] = []
    prompt_path = iter_dir / policy.prompt_filename
    tasks_path = iter_dir / policy.task_board_filename
    prompts_dir = iter_dir / policy.task_prompts_dirname
    reviews_dir = iter_dir / policy.task_reviews_dirname

    # Required files present
    if not prompt_path.exists():
        errors.append(f"{iter_dir}: missing {policy.prompt_filename}")
    if not tasks_path.exists():
        errors.append(f"{iter_dir}: missing {policy.task_board_filename}")
    if not prompts_dir.exists():
        errors.append(
            f"{iter_dir}: missing {policy.task_prompts_dirname}/ directory"
        )
    if not reviews_dir.exists():
        errors.append(
            f"{iter_dir}: missing {policy.task_reviews_dirname}/ directory"
        )
    if errors:
        return errors  # don't proceed if structure is broken

    prompt_text = prompt_path.read_text(encoding="utf-8")
    tasks_text = tasks_path.read_text(encoding="utf-8")

    # Check 1: task count consistency
    claimed = claimed_task_count(prompt_text)
    rows = task_rows_in_tasks_md(tasks_text)
    prompts_files_list = prompt_files(prompts_dir)
    if claimed is not None and rows > 0 and claimed != rows:
        errors.append(
            f"{iter_dir}: prompt.md claims {claimed} tasks but tasks.md has {rows} rows"
        )
    if (
        claimed is not None
        and len(prompts_files_list) > 0
        and claimed != len(prompts_files_list)
    ):
        errors.append(
            f"{iter_dir}: prompt.md claims {claimed} tasks but prompts/ has "
            f"{len(prompts_files_list)} files: {prompts_files_list}"
        )
    if rows > 0 and len(prompts_files_list) > 0 and rows != len(prompts_files_list):
        errors.append(
            f"{iter_dir}: tasks.md has {rows} rows but prompts/ has "
            f"{len(prompts_files_list)} files: {prompts_files_list}"
        )

    # Check 2: every prompt has a matching review
    reviews_files_list = review_files(reviews_dir)
    missing_reviews = sorted(set(prompts_files_list) - set(reviews_files_list))
    if missing_reviews:
        errors.append(
            f"{iter_dir}: prompts without matching reviews: {missing_reviews}"
        )

    # Check 3: depends-on uses em-dash, not '--'
    bad_dashes = has_double_hyphen_no_dep(tasks_text)
    if bad_dashes:
        errors.append(
            f"{iter_dir}: tasks.md uses '--' for no-dependency (use em-dash '—'). "
            f"Lines: {len(bad_dashes)}"
        )

    # Check 4: every **Test:** field is a runnable shell command
    for msg in _test_field_runnable(tasks_text):
        errors.append(f"{iter_dir}: {msg}")

    # Check 5: post-phase runnable prompts must not target main directly.
    errors.extend(
        post_phase_integration_policy_errors(
            Path.cwd(),
            targets=[iter_dir],
            include_docs=False,
            include_iterations=True,
            policy=policy,
        )
    )

    # Check 6: nav-discoverability inward-gap rule. Forward-
    # looking — only enforced for post-phase and tooling-sprint
    # iterations. Tasks that declare route-visible allowed files must
    # carry an explicit nav-discoverability or no-nav marker in their
    # prompt or review artifact.
    errors.extend(
        nav_discoverability_iteration_errors(
            iter_dir,
            repo_root=Path.cwd(),
            policy=policy,
        )
    )

    return errors


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    repo_root = Path.cwd()
    try:
        policy = load_scaffold_policy(repo_root)
    except ConfigError as exc:
        print(f"project.yaml error: {exc}", file=sys.stderr)
        return 1
    if "--all" in args:
        args.remove("--all")
        targets = find_iteration_dirs(
            repo_root / policy.iteration_root,
            policy=policy,
        )
        include_docs_policy = True
    else:
        targets = [Path(a) for a in args]
        include_docs_policy = False

    if not targets:
        print("No iteration directories to lint.", file=sys.stderr)
        return 0

    all_errors: list[str] = []
    for iter_dir in targets:
        if iter_dir.is_file():
            errors = post_phase_integration_policy_errors(
                repo_root,
                targets=[iter_dir],
                policy=policy,
            )
            errors.extend(
                nav_discoverability_doc_prompt_errors_all(
                    repo_root,
                    targets=[iter_dir],
                    policy=policy,
                )
            )
        else:
            errors = lint_iteration(iter_dir, policy=policy)
        all_errors.extend(errors)
    if include_docs_policy:
        all_errors.extend(
            post_phase_integration_policy_errors(
                repo_root,
                include_docs=True,
                include_iterations=False,
                policy=policy,
            )
        )
        all_errors.extend(
            nav_discoverability_doc_prompt_errors_all(repo_root, policy=policy)
        )

    if all_errors:
        for e in all_errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"OK: {len(targets)} iteration(s) lint clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
