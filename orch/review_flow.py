"""Review-prompt helpers for task review rounds."""
from __future__ import annotations

from pathlib import Path

from orch.tasks_schema import Task


RUNTIME_FALLBACK_REVIEW_CONTRACT = """## Runtime Fallback Review Contract

No authored review prompt exists for this task (legacy iteration without a
reviews/ directory). Apply this minimal contract:

- Changed files must stay within the allowed files listed in the metadata.
- The change must match the task title/intent; tests and project invariants
  must not be weakened.

Calibration: PASS only when blocking requirements hold; CHANGES REQUIRED with
Severity: should-fix for concrete non-blocking issues; Severity: block for
bugs/missing requirements/security issues; BLOCKED when correctness cannot be
determined.

Your response must end with exactly one of these trailing blocks:

Verdict: PASS

Verdict: CHANGES REQUIRED
Severity: should-fix

Verdict: CHANGES REQUIRED
Severity: block

Verdict: BLOCKED

No non-empty line may follow the verdict block.
"""


def review_prompt_candidates(
    reviews_dir: Path,
    task: Task,
) -> list[Path]:
    branch_stem = task.branch.split("/")[-1] if "/" in task.branch else ""
    task_stem = task.id.lower()
    task_num = task_stem.split("-t")[-1]
    stems = [
        branch_stem,
        task_stem,
        f"t{task_num}" if task_num else "",
    ]

    candidates: list[Path] = []
    seen: set[Path] = set()
    for stem in stems:
        if not stem:
            continue
        path = reviews_dir / f"review-{stem}.md"
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    if reviews_dir.exists():
        for pattern in (f"review-{task_stem}-*.md", f"review-t{task_num}-*.md"):
            for path in sorted(reviews_dir.glob(pattern)):
                if path not in seen:
                    candidates.append(path)
                    seen.add(path)
    return candidates


def load_review_prompt_contract(
    reviews_dir: Path,
    candidates: list[Path],
) -> tuple[str | None, Path | None]:
    for path in candidates:
        if path.exists():
            return path.read_text(), path
    if reviews_dir.exists():
        return None, None
    return "", None


def build_task_review_prompt_text(
    *,
    task: Task,
    contract: str | None,
    contract_path: Path | None,
    candidates: list[Path],
    cwd: Path,
    fresh_diff: str,
    base_sha: str,
    reviewer_role: str,
    round_num: int,
    max_rounds: int,
    primary_reviewer: str | None = None,
) -> tuple[str, str | None]:
    if contract is None:
        expected = ", ".join(
            str(p.relative_to(cwd)) if p.is_relative_to(cwd) else str(p)
            for p in candidates
        )
        return "", "review prompt missing; expected one of: " + expected

    task_title = task.title
    source = (
        str(contract_path.relative_to(cwd))
        if contract_path and contract_path.is_relative_to(cwd)
        else str(contract_path or "(no authored review prompt found)")
    )
    independence = ""
    if reviewer_role == "secondary":
        independence = (
            "\n## Secondary Review Independence\n\n"
            f"The primary reviewer was `{primary_reviewer}`. Do not rely "
            "on prior review results; inspect the fresh diff and apply "
            "the same review contract independently.\n"
        )
    prompt = (
        f"You are reviewing code for task `{task.id}: {task_title}`.\n\n"
        "## Review Prompt Contract\n\n"
        f"Source: `{source}`\n\n"
        f"{contract or RUNTIME_FALLBACK_REVIEW_CONTRACT}\n\n"
        "## Runtime Review Metadata\n\n"
        f"- Task: `{task.id}`\n"
        f"- Title: `{task_title}`\n"
        f"- Review role: `{reviewer_role}`\n"
        f"- Round: `{round_num}/{max_rounds}`\n"
        f"- Diff base: `{base_sha}`\n"
        f"- Allowed files: `{task.allowed_files}`\n"
        f"{independence}\n"
        "## Fresh Diff\n\n"
        f"```diff\n{fresh_diff}\n```\n\n"
        "End with the exact trailing verdict block required by the "
        "review prompt contract. No non-empty line may follow it.\n"
    )
    return prompt, None


def extract_review_findings(stdout: str) -> str:
    # Strip everything after the last "Verdict:" line for the fixer prompt.
    lines = stdout.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().lower().startswith("verdict:"):
            return "\n".join(lines[:i]).strip()
    return stdout.strip()
