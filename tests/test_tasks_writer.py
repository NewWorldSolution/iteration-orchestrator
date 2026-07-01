"""Tests for orch.tasks_writer."""
from __future__ import annotations

from pathlib import Path

import pytest

from orch.tasks_writer import update_task_status


MD = """\
# Demo
## Task Board

**Status:** WAITING

---

## Tasks

| ID    | Title   | Owner | Status  | Depends on | Branch      |
|-------|---------|-------|---------|------------|-------------|
| I1-T1 | Alpha   | TBD   | WAITING | \u2014     | d/i1/t1     |
| I1-T2 | Beta    | TBD   | WAITING | I1-T1      | d/i1/t2     |
"""


def _write(tmp_path: Path, content: str = MD) -> Path:
    p = tmp_path / "tasks.md"
    p.write_text(content)
    return p


def test_updates_target_row_only(tmp_path: Path):
    p = _write(tmp_path)
    assert update_task_status(p, "I1-T1", "DONE")
    text = p.read_text()
    assert "| I1-T1 | Alpha   | TBD   | DONE" in text
    # Row below is untouched
    assert "| I1-T2 | Beta    | TBD   | WAITING" in text


def test_example_update_output_is_preserved(tmp_path: Path):
    p = _write(tmp_path)

    assert update_task_status(p, "I1-T1", "DONE")

    assert p.read_text() == """\
# Demo
## Task Board

**Status:** WAITING

---

## Tasks

| ID    | Title   | Owner | Status  | Depends on | Branch      |
|-------|---------|-------|---------|------------|-------------|
| I1-T1 | Alpha   | TBD   | DONE    | \u2014     | d/i1/t1     |
| I1-T2 | Beta    | TBD   | WAITING | I1-T1      | d/i1/t2     |
"""


def test_updates_generic_task_id_row(tmp_path: Path):
    p = _write(
        tmp_path,
        MD.replace("I1-T1", "TASK-1-1").replace("I1-T2", "TASK-1-2"),
    )

    assert update_task_status(p, "TASK-1-1", "DONE")

    text = p.read_text()
    assert "| TASK-1-1 | Alpha   | TBD   | DONE" in text
    assert "| TASK-1-2 | Beta    | TBD   | WAITING" in text


def test_task_id_is_matched_literally(tmp_path: Path):
    p = _write(
        tmp_path,
        """\
# Demo
## Tasks

| ID        | Title | Owner | Status  | Depends on | Branch  |
|-----------|-------|-------|---------|------------|---------|
| TASK-1x1  | Wrong | TBD   | WAITING | \u2014     | d/wrong |
| TASK-1.1  | Right | TBD   | WAITING | \u2014     | d/right |
""",
    )

    assert update_task_status(p, "TASK-1.1", "DONE")

    text = p.read_text()
    assert "| TASK-1x1  | Wrong | TBD   | WAITING" in text
    assert "| TASK-1.1  | Right | TBD   | DONE" in text


def test_idempotent_noop_when_already_set(tmp_path: Path):
    p = _write(tmp_path)
    update_task_status(p, "I1-T1", "DONE")
    before = p.read_text()
    assert update_task_status(p, "I1-T1", "DONE") is False
    assert p.read_text() == before


def test_missing_row_raises(tmp_path: Path):
    p = _write(tmp_path)
    with pytest.raises(ValueError, match="no row"):
        update_task_status(p, "I9-T9", "DONE")


def test_too_few_cells_error_is_preserved(tmp_path: Path):
    p = _write(
        tmp_path,
        """\
# Demo
## Tasks

| ID    | Title | Owner |
|-------|-------|-------|
| I1-T1 | Alpha | TBD   |
""",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"tasks\.md: row for I1-T1 has too few cells: "
            r"\| I1-T1 \| Alpha \| TBD   \|"
        ),
    ):
        update_task_status(p, "I1-T1", "DONE")


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        update_task_status(tmp_path / "nope.md", "I1-T1", "DONE")
