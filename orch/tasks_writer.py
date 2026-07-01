"""Mutate a single task's status cell in tasks.md.

The orchestrator is the sole writer of tasks.md — implementers and fixers
must never touch it (checked in step 3b). This module keeps that write
scoped: only the target row's Status cell is updated, preserving column
widths and spacing. If the target status matches the existing value the
file is not rewritten (idempotent).
"""
from __future__ import annotations

import re
from pathlib import Path

def update_task_status(
    path: Path, task_id: str, new_status: str
) -> bool:
    """Rewrite tasks.md changing ``task_id``'s Status cell to ``new_status``.

    Returns True when the file was updated, False when no change was needed.
    Raises FileNotFoundError if the file is missing, or ValueError if the
    target row cannot be found.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text().splitlines(keepends=True)
    out = list(lines)
    changed = False
    matched = False
    row_re = re.compile(rf"^(\s*\|\s*)({re.escape(task_id)})(\s*\|)(.*)$")

    for i, line in enumerate(lines):
        m = row_re.match(line.rstrip("\n"))
        if not m:
            continue
        matched = True
        rest = m.group(4)  # everything after the "| <id> |" preamble
        # rest has form: "<title>|<owner>|<status>|<deps>|<branch>|"
        cells = rest.split("|")
        # cells = [title_with_spaces, owner, status, deps, branch, "", ...]
        if len(cells) < 5:
            raise ValueError(
                f"tasks.md: row for {task_id} has too few cells: {line.rstrip()}"
            )
        status_cell = cells[2]
        left = _leading_ws(status_cell)
        right = _trailing_ws(status_cell)
        current = status_cell.strip()
        if current == new_status:
            matched = True
            break
        # Preserve the column width: pad the new status to the old cell width.
        old_width = len(status_cell)
        replacement = f"{left}{new_status}{right}"
        if len(replacement) < old_width:
            replacement += " " * (old_width - len(replacement))
        elif len(replacement) > old_width:
            # Only pad right to the length of new_status + one space if we
            # overran — don't let a longer label squash neighbouring columns.
            replacement = f"{left}{new_status} "
        cells[2] = replacement
        new_rest = "|".join(cells)
        preserved_eol = "\n" if line.endswith("\n") else ""
        out[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{new_rest}{preserved_eol}"
        changed = True
        break

    if not matched:
        raise ValueError(f"tasks.md: no row found for task id '{task_id}'")
    if changed:
        path.write_text("".join(out))
    return changed


def _leading_ws(s: str) -> str:
    i = 0
    while i < len(s) and s[i].isspace():
        i += 1
    return s[:i]


def _trailing_ws(s: str) -> str:
    i = len(s)
    while i > 0 and s[i - 1].isspace():
        i -= 1
    return s[i:]
