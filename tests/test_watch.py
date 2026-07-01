"""Tests for the read-only orch watch command."""
from __future__ import annotations

from pathlib import Path

from orch.cli import main
from orch.state import StateStore


PROJECT_YAML = """\
project:
  name: demo
  main_branch: main
  phase_branch_pattern: "phase-{phase}"
  iteration_branch_pattern: "{phase}/iteration-{n}"
  task_branch_pattern: "{phase}/i{n}/t{k}-{slug}"
stack:
  test: "echo test"
  lint: "echo lint"
risk:
  high_risk_globs: []
  sensitive_files: []
  forbidden_patterns: []
agents:
  claude:
    type: claude
    cmd: "claude -p"
    family: anthropic
  codex:
    type: codex
    cmd: "codex"
    family: openai
costs:
  anthropic: {input: 3.0, output: 15.0}
  openai: {input: 2.5, output: 10.0}
"""


TASKS_MD = """\
# Iteration demo-i1
## Task Board

**Status:** WAITING
**Iteration branch:** `demo/iteration-1`
**Depends on:** none
**Blocks:** none

---

## Execution Plan
- approach: task_by_task
- qa: standard
- note: runtime

---

## Tasks

| ID    | Title    | Owner | Status  | Depends on | Branch            |
|-------|----------|-------|---------|------------|-------------------|
| I4-T1 | Do thing | TBD   | WAITING | \u2014     | `demo/i1/t1-thing` |

---

## Task Details

### I4-T1 \u2014 Do thing

**Allowed files:**
```
app/thing.py
```
"""


def test_watch_tails_events_without_mutating_state(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(PROJECT_YAML)
    iter_dir = tmp_path / "iterations" / "phase-demo" / "demo-i1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tasks.md").write_text(TASKS_MD)
    log_dir = tmp_path / "tools" / "logs" / "demo-i1"
    store = StateStore(
        log_dir=log_dir,
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
    )
    store.append_event(kind="note", task="I4-T1", meta={"event": "first"})
    store.append_event(kind="note", task="I4-T1", meta={"event": "second"})
    before = store.path.read_bytes()
    monkeypatch.chdir(tmp_path)

    assert main(["watch", "demo-i1", "--limit", "1"]) == 0

    out = capsys.readouterr().out
    assert "second" in out
    assert "first" not in out
    assert store.path.read_bytes() == before
