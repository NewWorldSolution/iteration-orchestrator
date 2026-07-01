#!/usr/bin/env python3
"""Prompt-vs-disk consistency check for iterations/<phase>/<iter>/prompts/.

Iteration prompts can drift when they reference files that were renamed or
removed. This script catches that drift statically before the prompt is handed
to an implementer.

For each ``prompts/t<k>-*.md`` file in the iteration directory, scan the
sections under headings whose title matches one of:

    - "Read first"            (h2 ## or bold-only **Read first:** style)
    - "Current state on disk"
    - "Files to read"

Extract referenced paths from inline-code spans and from fenced code blocks,
and verify each exists relative to the repo root. Paths in the "Allowed files"
section are treated as CREATE intent (target paths) — those are collected for
skip-set membership but not existence-checked.

Exits 0 when every referenced path resolves; exits 1 with one line per stale
reference otherwise.

Usage:
    python -m orch.scaffold_review iterations/demo/demo-i9
    python -m orch.scaffold_review --all   # walk every iteration
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from orch.scaffold_lint import (
        ConfigError,
        ScaffoldPolicy,
        load_scaffold_policy,
        nav_discoverability_doc_prompt_errors,
        nav_discoverability_doc_prompt_errors_all,
        nav_discoverability_iteration_errors,
        post_phase_integration_policy_errors,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from scaffold_lint import (
        ConfigError,
        ScaffoldPolicy,
        load_scaffold_policy,
        nav_discoverability_doc_prompt_errors,
        nav_discoverability_doc_prompt_errors_all,
        nav_discoverability_iteration_errors,
        post_phase_integration_policy_errors,
    )

# Headings that mark a section whose paths must already exist on disk.
READ_HEADINGS = (
    "read first",
    "current state on disk",
    "files to read",
)
# Heading whose paths are CREATE intent — collect for skip-set, don't verify.
ALLOWED_HEADING = "allowed files"

# Heading line patterns. Match either Markdown heading syntax (## / ###) or
# the bold-only variant used in some prompts (**Read first (mandatory):**).
_HEADING_RE = re.compile(
    r"^(?:#{2,6}\s+|\*\*)(?P<title>[^*#\n][^*\n]*?)(?:\*\*)?\s*$"
)
_FENCE_RE = re.compile(r"^\s*```")
# Inline-code span: backtick-wrapped token.
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# Glob characters — skip these.
_GLOB_RE = re.compile(r"[*?\[]")
# Bare top-level filenames that are real references even without a slash.
_BARE_TOPLEVEL = {"CLAUDE.md", "README.md", "pyproject.toml", "Makefile"}
# Strip trailing punctuation like trailing ',' '.' ':' from a path token.
_STRIP_TRAIL = ",.:;)"


def _normalize_heading(line: str) -> str | None:
    m = _HEADING_RE.match(line.strip())
    if not m:
        return None
    title = m.group("title").strip().rstrip(":").strip()
    # Drop trailing parentheticals like "(mandatory)".
    title = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    return title.lower()


def _looks_like_path(token: str) -> bool:
    """A token is path-shaped if it has a slash or is a known top-level file.

    Bare filenames without a directory (e.g. ``legal.py``) are NOT treated as
    paths — those almost always appear as prose mentions ("you'll add
    ``legal.py`` here") rather than real cross-file references, and treating
    them as paths produces noisy false positives.
    """
    if not token:
        return False
    if _GLOB_RE.search(token):
        return False
    # Whitespace = prose, not a path.
    if any(c.isspace() for c in token):
        return False
    # Absolute filesystem paths and URL routes — out of scope.
    url_prefixes = ("http" + "://", "https" + "://")
    if token.startswith(("/", "~", *url_prefixes)):
        return False
    if "/" in token:
        return True
    return token in _BARE_TOPLEVEL


def _clean_token(token: str) -> str:
    token = token.strip()
    while token and token[0] in "\"'`":
        token = token[1:]
    while token and token[-1] in "\"'`":
        token = token[:-1]
    while token and token[-1] in _STRIP_TRAIL:
        token = token[:-1]
    token = token.strip()
    # Strip ``::symbol`` suffix (Python module::function notation).
    token = re.sub(r"::[A-Za-z_][\w]*$", "", token)
    # Strip ``:lineno`` suffix (file:line citation).
    token = re.sub(r":\d+$", "", token)
    return token


def _extract_fence_paths(lines: list[str]) -> list[tuple[int, str]]:
    """Inside a fenced block, treat each non-empty trimmed line as a candidate path."""
    out: list[tuple[int, str]] = []
    for lineno, raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # Drop inline trailing comments after the first whitespace if it looks
        # like prose ("app/main.py — explanation"). Take the leading token.
        first = re.split(r"\s+(?:[—\-]|#)\s+", s, maxsplit=1)[0].strip()
        token = _clean_token(first)
        if _looks_like_path(token):
            out.append((lineno, token))
    return out


def _extract_inline_paths(lineno: int, line: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for m in _INLINE_CODE_RE.finditer(line):
        token = _clean_token(m.group(1))
        if _looks_like_path(token):
            out.append((lineno, token))
    return out


def parse_prompt_paths(text: str) -> tuple[list[tuple[int, str]], set[str]]:
    """Return (paths_to_check, allowed_files_set).

    paths_to_check: list of (lineno, path) collected from READ_HEADINGS sections.
    allowed_files_set: set of normalized paths declared as CREATE intent under
    the "Allowed files" heading; these are skipped from existence checking
    even if they also appear in a read-section.
    """
    paths: list[tuple[int, str]] = []
    allowed: set[str] = set()

    lines = text.splitlines()
    i = 0
    in_fence = False
    fence_lines: list[tuple[int, str]] = []
    current_section: str | None = None  # "read" | "allowed" | None
    current_section_level: int | None = None  # heading level (2..6) or 0 for bold
    section_seen_content = False

    def end_section() -> None:
        nonlocal current_section, current_section_level, section_seen_content
        current_section = None
        current_section_level = None
        section_seen_content = False

    while i < len(lines):
        line = lines[i]

        # Toggle code-fence state.
        if _FENCE_RE.match(line):
            if in_fence:
                if current_section == "read":
                    paths.extend(_extract_fence_paths(fence_lines))
                    section_seen_content = True
                elif current_section == "allowed":
                    for _ln, p in _extract_fence_paths(fence_lines):
                        allowed.add(p)
                    section_seen_content = True
                fence_lines = []
                in_fence = False
            else:
                in_fence = True
            i += 1
            continue

        if in_fence:
            fence_lines.append((i + 1, line))
            i += 1
            continue

        # Bold-only "headings" introduce TIGHT blocks — terminate at the
        # first blank line following any consumed content.
        if (
            current_section is not None
            and current_section_level == 0
            and section_seen_content
            and line.strip() == ""
        ):
            end_section()
            i += 1
            continue

        # Heading detection (only outside fences).
        title = _normalize_heading(line)
        if title is not None:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
            else:
                level = 0  # bold-only "heading"

            # ## heading at same-or-shallower level terminates an active section.
            if current_section is not None and level != 0:
                if (
                    current_section_level == 0
                    or level <= current_section_level
                ):
                    end_section()

            if title in READ_HEADINGS:
                current_section = "read"
                current_section_level = level
                section_seen_content = False
            elif title == ALLOWED_HEADING:
                current_section = "allowed"
                current_section_level = level
                section_seen_content = False
            i += 1
            continue

        # Body line of a section: scan inline-code paths.
        if current_section == "read":
            found = _extract_inline_paths(i + 1, line)
            if found:
                paths.extend(found)
                section_seen_content = True
            elif line.strip():
                section_seen_content = True
        elif current_section == "allowed":
            found = _extract_inline_paths(i + 1, line)
            if found:
                for _ln, p in found:
                    allowed.add(p)
                section_seen_content = True
            elif line.strip():
                section_seen_content = True
        i += 1

    return paths, allowed


def _should_skip_existence(
    path: str,
    allowed: set[str],
    *,
    generated_artifact_prefixes: tuple[str, ...] = ("tools/logs/",),
) -> bool:
    if path in allowed:
        return True
    # Generated artifacts (orch logs, QA reports) — not source files.
    if any(
        path == prefix.rstrip("/") or path.startswith(prefix)
        for prefix in generated_artifact_prefixes
    ):
        return True
    return False


def review_prompt_file(
    prompt_path: Path,
    repo_root: Path,
    iteration_allowed: set[str] | None = None,
    *,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    """Return list of error strings for this prompt file (empty = clean).

    ``iteration_allowed`` is the union of every "Allowed files" set across all
    prompts in the same iteration. A later task can legitimately reference a
    file that an earlier task creates.
    """
    policy = policy or load_scaffold_policy(repo_root)
    text = prompt_path.read_text(encoding="utf-8")
    paths, allowed = parse_prompt_paths(text)
    if iteration_allowed:
        allowed = allowed | iteration_allowed
    errors: list[str] = []
    seen: set[tuple[int, str]] = set()
    for lineno, p in paths:
        if (lineno, p) in seen:
            continue
        seen.add((lineno, p))
        if _should_skip_existence(
            p,
            allowed,
            generated_artifact_prefixes=policy.generated_artifact_prefixes,
        ):
            continue
        if not (repo_root / p).exists():
            rel = (
                prompt_path.relative_to(repo_root)
                if prompt_path.is_relative_to(repo_root)
                else prompt_path
            )
            errors.append(f"{rel}:{lineno}: missing file reference: {p}")
    errors.extend(
        post_phase_integration_policy_errors(
            repo_root,
            targets=[prompt_path],
            policy=policy,
        )
    )
    errors.extend(
        nav_discoverability_doc_prompt_errors(
            prompt_path,
            repo_root,
            policy=policy,
        )
    )
    return errors


def collect_allowed(
    iter_dirs: list[Path],
    *,
    policy: ScaffoldPolicy | None = None,
) -> set[str]:
    """Union of every "Allowed files" set across the given iterations.

    Scoped intentionally narrow — passing a single iteration covers cross-task
    create-intent (T2 reads what T1 creates within one iteration). The caller
    must NOT pass every iteration in the repo: doing so would silently mask
    stale path references whenever any unrelated iteration happens to declare
    the same path, defeating the prompt-vs-disk guarantee.
    """
    policy = policy or load_scaffold_policy(Path.cwd())
    allowed: set[str] = set()
    for iter_dir in iter_dirs:
        prompts_dir = iter_dir / policy.task_prompts_dirname
        if not prompts_dir.is_dir():
            continue
        for prompt in prompts_dir.glob("t*.md"):
            _, declared = parse_prompt_paths(prompt.read_text(encoding="utf-8"))
            allowed |= declared
    return allowed


def review_iteration(
    iter_dir: Path,
    repo_root: Path,
    *,
    policy: ScaffoldPolicy | None = None,
) -> list[str]:
    policy = policy or load_scaffold_policy(repo_root)
    prompts_dir = iter_dir / policy.task_prompts_dirname
    if not prompts_dir.is_dir():
        return []
    prompt_files = sorted(prompts_dir.glob("t*.md"))
    # Same-iteration aggregation only: T2 can read files T1 creates within the
    # same iteration, but unrelated iteration history is not exempted.
    iteration_allowed = collect_allowed([iter_dir], policy=policy)
    errors: list[str] = []
    for prompt in prompt_files:
        errors.extend(
            review_prompt_file(
                prompt,
                repo_root,
                iteration_allowed,
                policy=policy,
            )
        )
    prompt_md = iter_dir / policy.prompt_filename
    if prompt_md.exists():
        errors.extend(
            post_phase_integration_policy_errors(
                repo_root,
                targets=[prompt_md],
                policy=policy,
            )
        )
    errors.extend(
        nav_discoverability_iteration_errors(
            iter_dir,
            repo_root=repo_root,
            policy=policy,
        )
    )
    return errors


def find_iteration_dirs(root: Path, *, policy: ScaffoldPolicy) -> list[Path]:
    out: list[Path] = []
    for prompt in root.rglob(policy.prompt_filename):
        if (prompt.parent / policy.task_board_filename).exists():
            out.append(prompt.parent)
    return sorted(out)


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
        print("No iteration directories to review.", file=sys.stderr)
        return 0

    all_errors: list[str] = []
    for iter_dir in targets:
        if iter_dir.is_file():
            all_errors.extend(
                post_phase_integration_policy_errors(
                    repo_root,
                    targets=[iter_dir],
                    policy=policy,
                )
            )
            all_errors.extend(
                nav_discoverability_doc_prompt_errors_all(
                    repo_root,
                    targets=[iter_dir],
                    policy=policy,
                )
            )
        else:
            all_errors.extend(review_iteration(iter_dir, repo_root, policy=policy))
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

    print(f"OK: {len(targets)} iteration(s) reviewed clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
