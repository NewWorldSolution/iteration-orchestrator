"""Integration tests for the IterationRunner main loop.

Uses a real temp git repo plus fake agent adapters so the full state
machine runs end-to-end without spawning claude/codex/gh.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

import orch.runner as runner_mod
from orch.agents.base import AgentResult
from orch.config import costs, limits, load_config
from orch.cost import CostLogger
from orch.git_ops import (
    branch_exists,
    DiffStats,
    checkout,
    commit,
    create_or_reset_branch,
    current_sha,
    ensure_orch_workdir,
    GitError,
    GitResult,
    git,
    orch_workdir,
    stage_all,
    task_workdir,
    working_tree_clean,
)
from orch.hooks import HookDispatcher, HookResult
from orch.merge import PrSnapshot
from orch.runner import (
    STATUS_DONE,
    IterationRunner,
    RunnerError,
    RunOptions,
    RunnerDeps,
)
from orch.state import STATUS_NEEDS_HUMAN_MERGE
from orch.state import STATUS_IN_PROGRESS
from orch.state import StateStore, STATUS_STOPPED_PREFIX
from orch.tasks_schema import parse_tasks_md


PROJECT_YAML = """\
project:
  name: demo
  main_branch: main
  phase_branch_pattern: "phase-{phase}"
  iteration_branch_pattern: "{phase}/iteration-{n}"
  task_branch_pattern: "{phase}/i{n}/t{k}-{slug}"
stack:
  test: "true"
  lint: "true"
  test_env:
    ENVIRONMENT: test
  acceptance_timeout_seconds: 30
risk:
  high_risk_globs: ["**/schema.sql"]
  sensitive_files: [".env"]
  forbidden_patterns: ["nosec"]
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
patterns:
  task_id: '^I(\\d+)-T(\\d+)$'
  task_detail_heading: '^###\\s+(?P<id>I\\d+-T\\d+)\\s+—\\s+(?P<title>.+?)\\s*$'
  phase_branch: '^phase-[A-Za-z0-9][A-Za-z0-9_-]*$'
"""

EXAMPLE_ROUTE_VISIBILITY_CONFIG = """
ui_route_visibility:
  route_globs:
    - "app/routes/*.py"
    - "app/templates/**/*.html"
  nav_anchor_paths:
    - "app/templates/base.html"
    - "app/templates/_nav.html"
    - "app/templates/partials/_nav.html"
    - "app/templates/nav.html"
"""

EXAMPLE_NAV_ANCHOR_PATHS = [
    "app/templates/base.html",
    "app/templates/_nav.html",
    "app/templates/partials/_nav.html",
    "app/templates/nav.html",
]

TASKS_MD = """\
# Demo iteration
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

| ID    | Title     | Owner | Status  | Depends on | Branch      |
|-------|-----------|-------|---------|------------|-------------|
| I1-T1 | First     | TBD   | WAITING | \u2014     | d/i1/t1     |
| I1-T2 | Second    | TBD   | WAITING | I1-T1      | d/i1/t2     |

---

## Task Details

### I1-T1 \u2014 First

**Allowed files:**
```
src/a.py
```

### I1-T2 \u2014 Second

**Allowed files:**
```
src/b.py
```
"""

TASK_DIFF_CAP_OVERRIDE = (
    "max_diff_insertions_hard=1800; approved_by=operator; "
    "evidence=iterations/demo-i1/reviews/review-t1.md"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeAdapter:
    name: str
    family: str
    # Each call returns the next script entry (cycling on exhaustion)
    script: list[Callable] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def invoke(self, prompt, *, timeout, workdir, routing_options=None):
        self.calls.append(
            {
                "prompt": prompt,
                "timeout": timeout,
                "workdir": Path(workdir),
                "routing_options": routing_options,
            }
        )
        if not self.script:
            return AgentResult(
                exit_code=0, stdout="", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )
        fn = self.script.pop(0)
        return fn(self, prompt, workdir)


class FakeHookHandler:
    def __init__(
        self,
        *,
        event_name: str,
        result: HookResult | None = None,
        required: bool = True,
    ) -> None:
        self.name = "fake-hook"
        self.event_name = event_name
        self.required = required
        self.result = result or HookResult.ok()
        self.contexts = []

    def handles(self, event_name: str) -> bool:
        return event_name == self.event_name

    def handle(self, context):
        self.contexts.append(context)
        return self.result


def _edit_file_on_invoke(relpath: str, content: str):
    def do(adapter, prompt, workdir):
        p = Path(workdir) / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        stage_all(Path(workdir))
        commit(Path(workdir), f"{adapter.name}: edit {relpath}")
        return AgentResult(
            exit_code=0, stdout="", stderr="", duration_s=0.1,
            input_tokens=50, output_tokens=20, tokens_exact=False,
        )
    return do


def _exact_edit_file_on_invoke(
    relpath: str,
    content: str,
    *,
    provider: str,
    model: str | None = None,
    input_tokens: int = 1000,
    output_tokens: int = 200,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
):
    def do(adapter, prompt, workdir):
        p = Path(workdir) / relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        stage_all(Path(workdir))
        commit(Path(workdir), f"{adapter.name}: edit {relpath}")
        return AgentResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=0.1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_exact=True,
            provider=provider,
            model=model,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            parser_status="parsed",
            extra={"raw_terminal_json": "{}"},
        )
    return do


def _assert_visible_then_edit(
    visible_relpath: str,
    visible_content: str,
    edit_relpath: str,
    edit_content: str,
):
    def do(adapter, prompt, workdir):
        visible = Path(workdir) / visible_relpath
        assert visible.read_text() == visible_content
        p = Path(workdir) / edit_relpath
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(edit_content)
        stage_all(Path(workdir))
        commit(Path(workdir), f"{adapter.name}: edit {edit_relpath}")
        return AgentResult(
            exit_code=0, stdout="", stderr="", duration_s=0.1,
            input_tokens=50, output_tokens=20, tokens_exact=False,
        )
    return do


def _reviewer_verdict(text: str):
    def do(adapter, prompt, workdir):
        return AgentResult(
            exit_code=0, stdout=text, stderr="", duration_s=0.1,
            input_tokens=30, output_tokens=10, tokens_exact=False,
        )
    return do


def _exact_reviewer_verdict(
    text: str,
    *,
    provider: str,
    model: str | None = None,
    input_tokens: int = 4000,
    output_tokens: int = 500,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
):
    def do(adapter, prompt, workdir):
        return AgentResult(
            exit_code=0,
            stdout=text,
            stderr="",
            duration_s=0.1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_exact=True,
            provider=provider,
            model=model,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            parser_status="parsed",
            extra={"raw_terminal_json": "{}"},
        )
    return do


def _parallel_edit_from_prompt(adapter, prompt, workdir):
    if "src/a.py" in prompt:
        relpath = "src/a.py"
        content = "def a():\n    return 1\n"
    elif "src/b.py" in prompt:
        relpath = "src/b.py"
        content = "def b():\n    return 2\n"
    elif "src/c.py" in prompt:
        relpath = "src/c.py"
        content = "def c():\n    return 3\n"
    else:
        raise AssertionError(f"unexpected prompt: {prompt}")
    p = Path(workdir) / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    stage_all(Path(workdir))
    commit(Path(workdir), f"{adapter.name}: edit {relpath}")
    return AgentResult(
        exit_code=0, stdout="", stderr="", duration_s=0.1,
        input_tokens=50, output_tokens=20, tokens_exact=False,
    )


def _parallel_edit_or_fail_b(adapter, prompt, workdir):
    if "src/b.py" in prompt:
        return AgentResult(
            exit_code=2,
            stdout="",
            stderr="forced failure",
            duration_s=0.1,
            input_tokens=20,
            output_tokens=0,
            tokens_exact=False,
        )
    return _parallel_edit_from_prompt(adapter, prompt, workdir)


def _partial_timeout_write_untracked(adapter, prompt, workdir):
    attempt = len(adapter.calls)
    p = Path(workdir) / "src" / f"timeout-{attempt}.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"attempt {attempt}\n")
    return AgentResult(
        exit_code=124,
        stdout="",
        stderr="timed out",
        duration_s=900.0,
        input_tokens=50,
        output_tokens=0,
        tokens_exact=False,
        partial=True,
    )


def _partial_timeout_no_write(adapter, prompt, workdir):
    return AgentResult(
        exit_code=124,
        stdout="",
        stderr="timed out",
        duration_s=900.0,
        input_tokens=50,
        output_tokens=0,
        tokens_exact=False,
        partial=True,
    )


def _seed_recorded_agents(
    store: StateStore, *, implementer: str = "codex", reviewer: str = "claude"
) -> None:
    store.append_event(
        kind="note",
        meta={
            "event": "agents_resolved",
            "implementer": implementer,
            "reviewer": reviewer,
            "source": "flags",
        },
    )


def _agents_resolved_events(store: StateStore) -> list[dict]:
    return [
        event for event in store.events
        if event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "agents_resolved"
    ]


def _update_tasks_md(repo: Path, content: str) -> None:
    checkout(repo, "main")
    (repo / "iterations" / "demo-i1" / "tasks.md").write_text(content)
    stage_all(repo)
    commit(repo, "update tasks md")
    git(["branch", "-f", "phase-demo", "HEAD"], cwd=repo, check=True)
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=repo, check=True)


def _enable_parallel_config(repo: Path, max_concurrency: int = 2) -> None:
    checkout(repo, "main")
    project_path = repo / ".orch" / "project.yaml"
    project_path.write_text(
        project_path.read_text()
        + f"\nparallel:\n  max_concurrency: {max_concurrency}\n"
    )
    stage_all(repo)
    commit(repo, "enable parallel config")
    git(["branch", "-f", "phase-demo", "HEAD"], cwd=repo, check=True)
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=repo, check=True)


def _append_project_config(repo: Path, content: str) -> None:
    checkout(repo, "main")
    project_path = repo / ".orch" / "project.yaml"
    project_path.write_text(project_path.read_text() + content)
    stage_all(repo)
    commit(repo, "update project config")
    git(["branch", "-f", "phase-demo", "HEAD"], cwd=repo, check=True)
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=repo, check=True)


def _parallel_tasks_md(
    *,
    overlap: bool = False,
    include_third: bool = False,
) -> str:
    t2_allowed = "src/a.py" if overlap else "src/b.py"
    rows = [
        "| I1-T1 | First     | TBD   | WAITING | \u2014     | d/i1/t1     |",
        "| I1-T2 | Second    | TBD   | WAITING | \u2014     | d/i1/t2     |",
    ]
    details = [
        "### I1-T1 \u2014 First\n\n"
        "**Allowed files:**\n"
        "```\n"
        "src/a.py\n"
        "```\n\n"
        "**Parallel safe:** yes; reason=disjoint files; conflicts=none; "
        "requires_serial_after=none\n",
        "### I1-T2 \u2014 Second\n\n"
        "**Allowed files:**\n"
        "```\n"
        f"{t2_allowed}\n"
        "```\n\n"
        "**Parallel safe:** yes; reason=disjoint files; conflicts=none; "
        "requires_serial_after=none\n",
    ]
    if include_third:
        rows.append(
            "| I1-T3 | Third     | TBD   | WAITING | \u2014     | d/i1/t3     |"
        )
        details.append(
            "### I1-T3 \u2014 Third\n\n"
            "**Allowed files:**\n"
            "```\n"
            "src/c.py\n"
            "```\n\n"
            "**Parallel safe:** yes; reason=disjoint files; conflicts=none; "
            "requires_serial_after=none\n"
        )
    return (
        "# Demo iteration\n"
        "## Task Board\n\n"
        "**Status:** WAITING\n"
        "**Iteration branch:** `demo/iteration-1`\n"
        "**Depends on:** none\n"
        "**Blocks:** none\n\n"
        "---\n\n"
        "## Execution Plan\n"
        "- approach: task_by_task\n"
        "- qa: standard\n"
        "- note: runtime\n\n"
        "---\n\n"
        "## Tasks\n\n"
        "| ID    | Title     | Owner | Status  | Depends on | Branch      |\n"
        "|-------|-----------|-------|---------|------------|-------------|\n"
        + "\n".join(rows)
        + "\n\n---\n\n"
        "## Task Details\n\n"
        + "\n\n".join(details)
        + "\n"
    )


def _enable_parallel_tasks(
    repo: Path, *, overlap: bool = False, include_third: bool = False
) -> None:
    _update_tasks_md(
        repo,
        _parallel_tasks_md(overlap=overlap, include_third=include_third),
    )


def _add_t1_diff_cap_override(repo: Path) -> None:
    _update_tasks_md(
        repo,
        TASKS_MD.replace(
            "### I1-T1 \u2014 First\n\n",
            "### I1-T1 \u2014 First\n\n"
            f"**Diff cap override:** `{TASK_DIFF_CAP_OVERRIDE}`\n\n",
        ),
    )


def _add_t1_model_routing(repo: Path, risk_category: str) -> None:
    _update_tasks_md(
        repo,
        TASKS_MD.replace(
            "### I1-T1 \u2014 First\n\n",
            "### I1-T1 \u2014 First\n\n"
            "**Model routing:** "
            f"`model_tier=standard; reasoning_effort=low; "
            f"risk_category={risk_category}`\n\n",
        ),
    )


def _add_t1_task_kind(repo: Path, task_kind: str) -> None:
    _update_tasks_md(
        repo,
        TASKS_MD.replace(
            "### I1-T1 \u2014 First\n\n",
            "### I1-T1 \u2014 First\n\n"
            f"**Task kind:** `{task_kind}`\n\n",
        ),
    )


def _add_review_prompt(repo: Path, name: str, text: str) -> None:
    reviews_dir = repo / "iterations" / "demo-i1" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    (reviews_dir / name).write_text(text)
    stage_all(repo)
    commit(repo, f"test: add {name}")
    git(["branch", "-f", "phase-demo", "HEAD"], cwd=repo, check=True)
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=repo, check=True)


def _fake_large_diff_for_targets(*targets: str):
    target_set = set(targets)

    def fake_diff_stats(cwd: Path, base: str, head: str = "HEAD") -> DiffStats:
        files = runner_mod.diff_files(cwd, base, head)
        if any(path in target_set for path in files):
            return DiffStats(insertions=1600, deletions=0, files=len(files))
        return DiffStats(insertions=0, deletions=0, files=len(files))

    return fake_diff_stats


def _inject_final_scope_leak(
    monkeypatch: pytest.MonkeyPatch,
    *,
    relpath: str = "src/outside.py",
    after_task: str = "I1-T2",
    force_add: bool = False,
) -> None:
    original = IterationRunner._write_tasks_md_done

    def patched(self: IterationRunner, task):
        original(self, task)
        if task.id != after_task:
            return
        checkout(self.deps.cwd, self.iter_branch)
        target = self.deps.cwd / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("OUTSIDE = True\n")
        if force_add:
            git(["add", "-f", relpath], cwd=self.deps.cwd, check=True)
        else:
            stage_all(self.deps.cwd)
        commit(self.deps.cwd, "test: inject final scope leak")

    monkeypatch.setattr(IterationRunner, "_write_tasks_md_done", patched)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    # Init repo with iter branch and an initial commit.
    git(["init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git(["config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    git(["config", "user.name", "Tester"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("r\n")
    (tmp_path / ".gitignore").write_text("tools/logs/\n")
    stage_all(tmp_path)
    commit(tmp_path, "init")
    # Create the iter branch the board references
    git(["branch", "demo/iteration-1"], cwd=tmp_path, check=True)
    # Project files
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(PROJECT_YAML)
    iter_dir = tmp_path / "iterations" / "demo-i1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tasks.md").write_text(TASKS_MD)
    stage_all(tmp_path)
    commit(tmp_path, "project files")
    # Force the configured phase branch and iteration branch to match HEAD.
    git(["branch", "-f", "phase-demo", "HEAD"], cwd=tmp_path, check=True)
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=tmp_path, check=True)
    return tmp_path


def _make_runner(
    repo: Path,
    adapters,
    *,
    dry_run=True,
    hook_dispatcher=None,
    deps_kwargs: dict | None = None,
    **opts,
):
    cfg = load_config(repo / ".orch" / "project.yaml")
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
        hook_dispatcher=hook_dispatcher,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(repo_root=repo, cwd=orch_cwd, **(deps_kwargs or {}))
    options = RunOptions(dry_run=dry_run, poll_ci=False, **opts)
    return IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters=adapters, options=options, deps=deps,
    ), store, cost


# ---------------------------------------------------------------------------
# Happy path (dry-run, so no PR/merge) — both tasks DONE
# ---------------------------------------------------------------------------


def test_happy_path_two_tasks_done(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, cost = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        hook_dispatcher=HookDispatcher([]),
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 0
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE
    assert all(event["kind"] != "hook" for event in store.events)
    # One IMPL + one REVIEW per task = 4 cost records.
    lines = (repo / "tools/logs/demo-i1/cost.jsonl").read_text().splitlines()
    assert len(lines) == 4


def test_runner_phase_branch_resolution_is_config_driven_fail_closed(repo: Path):
    runner, _, _ = _make_runner(
        repo,
        {
            "claude": FakeAdapter(name="claude", family="anthropic"),
            "codex": FakeAdapter(name="codex", family="openai"),
        },
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    runner.iteration = "demo-i1"
    assert runner._resolve_phase_branch() == "phase-demo"
    runner.iteration = "not-an-iteration"
    with pytest.raises(RunnerError, match="cannot resolve phase"):
        runner._resolve_phase_branch()


def test_parallel_default_serial_preserves_existing_order(repo: Path):
    _enable_parallel_tasks(repo)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    assert [
        event["task"]
        for event in store.events
        if event.get("kind") == "task_transition"
        and event.get("meta", {}).get("status") == STATUS_IN_PROGRESS
    ] == ["I1-T1", "I1-T2"]
    assert not any(
        event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "parallel_wave_started"
        for event in store.events
    )


def test_parallel_runner_executes_disjoint_ready_tasks_in_one_wave(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    wave_events = [
        event for event in store.events
        if event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "parallel_wave_started"
    ]
    assert len(wave_events) == 1
    assert wave_events[0]["meta"]["tasks"] == ["I1-T1", "I1-T2"]
    assert wave_events[0]["meta"]["lock_held"] is True
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE
    assert task_workdir(repo, "demo-i1", "I1-T1").exists()
    assert task_workdir(repo, "demo-i1", "I1-T2").exists()


def test_parallel_wave_second_member_merge_survives_freshness_gate(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    refresh_events = [
        event for event in store.events
        if event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "parallel_branch_refreshed"
    ]
    assert [event["task"] for event in refresh_events] == ["I1-T2"]
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE
    assert not any(
        event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "branch_freshness_gate"
        for event in store.events
    )


def test_parallel_no_ci_local_merge_blocks_noop_acceptance_without_override(
    repo: Path,
):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    calls = SimpleNamespace(open_pr=[], merge_pr=[], comment_pr=[])

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, f"https://example.com/pr/{len(calls.open_pr)}"

    def fake_merge_pr(*, cwd, pr_url):
        calls.merge_pr.append(pr_url)
        raise AssertionError("gh merge must not run in no-CI mode")

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    cfg = load_config(repo / ".orch" / "project.yaml")
    cfg.data["parallel"] = {"max_concurrency": 2}
    cfg.data["auto_merge"]["no_ci"] = True
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(
        repo_root=repo,
        cwd=orch_cwd,
        open_pr=fake_open_pr,
        merge_pr=fake_merge_pr,
        comment_pr=fake_comment,
        run_lock=SimpleNamespace(acquired=True),
    )
    opts = RunOptions(
        implementer="claude", reviewer="codex",
        dry_run=False, poll_ci=True,
    )
    runner = IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": impl, "codex": rev}, options=opts, deps=deps,
    )

    rc = runner.run()

    assert rc == 1
    assert len(calls.open_pr) == 2
    assert calls.merge_pr == []
    assert calls.comment_pr == []
    assert store.tasks["I1-T1"].status == STATUS_NEEDS_HUMAN_MERGE
    assert store.tasks["I1-T2"].status == STATUS_NEEDS_HUMAN_MERGE
    assert all(
        store.tasks[task_id].stop_reason == "NOOP_ACCEPTANCE"
        for task_id in ["I1-T1", "I1-T2"]
    )
    block_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "noop_acceptance_blocked"
    ]
    assert [e["task"] for e in block_events] == ["I1-T1", "I1-T2"]
    assert all(e["meta"]["test_cmd"] == "true" for e in block_events)


def test_parallel_runner_refuses_overlapping_allowed_files(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo, overlap=True)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/a.py", "def a():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    assert not any(
        event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "parallel_wave_started"
        for event in store.events
    )
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE


def test_parallel_runner_records_deterministic_event_order(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    signatures = [
        (
            event.get("kind"),
            event.get("task"),
            (event.get("meta") or {}).get("event")
            or (event.get("meta") or {}).get("status")
            or (event.get("meta") or {}).get("verdict")
            or (event.get("meta") or {}).get("phase"),
        )
        for event in store.events
        if event.get("kind") in {"note", "task_transition", "impl_attempt", "review"}
        # pull_ff_only_failed is an IO/environment diagnostic (the local test
        # repo has no remote), orthogonal to the orchestration event order this
        # test pins — exclude it so the characterization is environment-robust.
        and (event.get("meta") or {}).get("event") != "pull_ff_only_failed"
    ]
    assert signatures[:8] == [
        ("note", None, "agents_resolved"),
        ("note", None, "parallel_wave_started"),
        ("note", "I1-T1", "model_routing_resolved"),
        ("note", "I1-T1", "model_routing_unknown_risk"),
        ("task_transition", "I1-T1", STATUS_IN_PROGRESS),
        ("impl_attempt", "I1-T1", "end"),
        ("review", "I1-T1", "PASS"),
        ("task_transition", "I1-T1", STATUS_DONE),
    ]


def test_pull_ff_only_failure_is_surfaced_not_swallowed(
    repo: Path, monkeypatch: pytest.MonkeyPatch
):
    # B-14 R5: a non-zero `git pull --ff-only` must be recorded (so a stale
    # local branch is never silently relied upon), but stay non-fatal.
    runner, store, _ = _make_runner(repo, {})
    monkeypatch.setattr(
        runner_mod,
        "pull_ff_only",
        lambda cwd: GitResult(
            exit_code=1, stdout="", stderr="fatal: Not possible to fast-forward",
        ),
    )

    ok = runner._pull_iter_branch_ff(context="unit refresh", task="I1-T1")

    assert ok is False
    notes = [
        e for e in store.events
        if (e.get("meta") or {}).get("event") == "pull_ff_only_failed"
    ]
    assert len(notes) == 1
    assert notes[0]["task"] == "I1-T1"
    assert notes[0]["meta"]["context"] == "unit refresh"
    assert notes[0]["meta"]["exit_code"] == 1
    assert "fast-forward" in notes[0]["meta"]["stderr"]


def test_pull_ff_only_success_emits_no_note(
    repo: Path, monkeypatch: pytest.MonkeyPatch
):
    runner, store, _ = _make_runner(repo, {})
    monkeypatch.setattr(
        runner_mod,
        "pull_ff_only",
        lambda cwd: GitResult(exit_code=0, stdout="Already up to date.", stderr=""),
    )

    assert runner._pull_iter_branch_ff(context="unit refresh") is True
    assert not [
        e for e in store.events
        if (e.get("meta") or {}).get("event") == "pull_ff_only_failed"
    ]


def test_parallel_runner_stops_cleanly_when_one_wave_task_fails(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo, include_third=True)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_parallel_edit_or_fail_b, _parallel_edit_or_fail_b],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 1

    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_STOPPED_PREFIX + "IMPL_FAILED"
    assert "I1-T3" not in store.tasks
    assert len(impl.calls) == 2
    assert any(
        event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "parallel_wave_finished"
        and event.get("meta", {}).get("stopped") is True
        for event in store.events
    )


def test_parallel_runner_holds_iteration_lock_for_shared_mutation(repo: Path):
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 1

    assert impl.calls == []
    assert any(
        event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "stop_global"
        and event.get("meta", {}).get("reason") == "INTERNAL"
        and "iteration lock" in event.get("meta", {}).get("msg", "")
        for event in store.events
    )


def test_conflict_marker_check_requires_pair_detection():
    setext_diff = "\n".join([
        "diff --git a/docs.md b/docs.md",
        "+++ b/docs.md",
        "+Heading",
        "+=======",
    ])
    conflict_diff = "\n".join([
        "diff --git a/src/a.py b/src/a.py",
        "+++ b/src/a.py",
        "+<<<<<<< HEAD",
        "+a = 1",
        "+=======",
        "+a = 2",
        "+>>>>>>> other",
    ])

    assert not runner_mod._diff_introduces_conflict_marker_pair(setext_diff)
    assert runner_mod._diff_introduces_conflict_marker_pair(conflict_diff)


def test_resume_defaults_to_recorded_agents(repo: Path):
    codex = FakeAdapter(
        name="codex", family="openai",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    claude = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": claude, "codex": codex}, dry_run=True,
    )
    _seed_recorded_agents(store, implementer="codex", reviewer="claude")

    assert runner.run() == 0

    assert len(codex.calls) == 2
    assert store.tasks["I1-T1"].implementer == "codex"
    assert store.tasks["I1-T1"].reviewer == "claude"
    resolved = _agents_resolved_events(store)
    assert resolved[-1]["meta"] == {
        "event": "agents_resolved",
        "implementer": "codex",
        "reviewer": "claude",
        "source": "run_state",
    }


def test_resume_legacy_state_falls_back_to_task_meta(repo: Path):
    codex = FakeAdapter(
        name="codex", family="openai",
        script=[_edit_file_on_invoke("src/b.py", "def b():\n    return 2\n")],
    )
    claude = FakeAdapter(
        name="claude", family="anthropic",
        script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": claude, "codex": codex}, dry_run=True,
    )
    store.task_meta("I1-T1", implementer="codex", reviewer="claude")
    store.task_transition("I1-T1", STATUS_DONE)

    assert runner.run() == 0

    assert len(codex.calls) == 1
    assert store.tasks["I1-T2"].implementer == "codex"
    assert store.tasks["I1-T2"].reviewer == "claude"
    assert _agents_resolved_events(store)[-1]["meta"]["source"] == "run_state"


def test_resume_flag_mismatch_errors_without_override(repo: Path):
    codex = FakeAdapter(name="codex", family="openai")
    claude = FakeAdapter(name="claude", family="anthropic")
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "codex": codex},
        dry_run=True,
        implementer="claude",
    )
    _seed_recorded_agents(store, implementer="codex", reviewer="claude")

    with pytest.raises(RunnerError) as excinfo:
        runner.run()

    msg = str(excinfo.value)
    assert "recorded: implementer=codex, reviewer=claude" in msg
    assert "requested: implementer=claude, reviewer=claude" in msg
    assert "Pass --override-agents" in msg
    assert codex.calls == []
    assert claude.calls == []


def test_resume_override_agents_proceeds(repo: Path):
    claude = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    codex = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "codex": codex},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        override_agents=True,
    )
    _seed_recorded_agents(store, implementer="codex", reviewer="claude")

    assert runner.run() == 0

    assert len(claude.calls) == 2
    assert store.tasks["I1-T1"].implementer == "claude"
    assert store.tasks["I1-T1"].reviewer == "codex"
    assert _agents_resolved_events(store)[-1]["meta"] == {
        "event": "agents_resolved",
        "implementer": "claude",
        "reviewer": "codex",
        "source": "flags_override",
    }


def test_resume_reports_stopped_tasks(repo: Path):
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )
    store.task_transition(
        "I1-T1",
        STATUS_STOPPED_PREFIX + "IMPL_TIMEOUT",
        reason="IMPL_TIMEOUT",
        msg="timeout",
    )

    assert runner.run() == 1

    assert store.tasks["I1-T1"].status == STATUS_STOPPED_PREFIX + "IMPL_TIMEOUT"
    assert impl.calls == []
    skipped = [
        event for event in store.events
        if event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "resume_skipped_tasks"
    ]
    assert skipped
    assert skipped[-1]["meta"]["tasks"] == [
        {"id": "I1-T1", "status": STATUS_STOPPED_PREFIX + "IMPL_TIMEOUT"}
    ]
    assert "orch retry <iter> <task>" in skipped[-1]["meta"]["msg"]


def test_runner_records_model_routing_metadata_and_unknown_warning(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    routing_events = [
        event for event in store.events
        if event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "model_routing_resolved"
    ]
    assert [event["task"] for event in routing_events] == ["I1-T1", "I1-T2"]
    assert routing_events[0]["meta"]["model_tier"] == "max"
    assert routing_events[0]["meta"]["reasoning_effort"] == "max"
    assert routing_events[0]["meta"]["risk_category"] == "unknown"

    warning_path = repo / "tools/logs/demo-i1/model_routing_warnings.jsonl"
    warnings = [
        json.loads(line)
        for line in warning_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [warning["task_id"] for warning in warnings] == ["I1-T1", "I1-T2"]

    first_cost = json.loads(
        (repo / "tools/logs/demo-i1/cost.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert first_cost["extra"]["model_routing"]["model_tier"] == "max"
    assert first_cost["extra"]["model_routing"]["reasoning_effort"] == "max"


def test_existing_agent_config_without_routing_map_still_runs(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert impl.calls[0]["routing_options"].args == ()
    assert impl.calls[0]["routing_options"].env == {}


def test_runner_passes_configured_model_routing_options_to_adapter(repo: Path):
    _add_t1_model_routing(repo, "security_compliance")
    _append_project_config(
        repo,
        """
model_routing:
  agent_overrides:
    claude:
      max:
        high:
          args: ["--model", "fixture-claude-model"]
    codex:
      max:
        high:
          env:
            ORCH_REASONING_EFFORT: "high"
""",
    )
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, _, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0
    assert impl.calls[0]["routing_options"].args == (
        "--model",
        "fixture-claude-model",
    )
    assert rev.calls[0]["routing_options"].env == {
        "ORCH_REASONING_EFFORT": "high"
    }


def test_runner_cost_records_exact_usage_and_dispatched_model(repo: Path):
    _add_t1_model_routing(repo, "security_compliance")
    _append_project_config(
        repo,
        """
model_routing:
  agent_overrides:
    claude:
      max:
        high:
          args: ["--model", "fixture-claude-dispatched"]
    codex:
      max:
        high:
          args: ["--model", "fixture-codex-dispatched"]
""",
    )
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _exact_edit_file_on_invoke(
                "src/a.py",
                "def a():\n    return 1\n",
                provider="claude",
                model="claude-json-model",
                input_tokens=1000,
                output_tokens=200,
                cached_input_tokens=300,
                cache_creation_input_tokens=400,
            ),
            _exact_edit_file_on_invoke(
                "src/b.py",
                "def b():\n    return 2\n",
                provider="claude",
                model="claude-json-model",
            ),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _exact_reviewer_verdict(
                "Looks good.\nVerdict: PASS\n",
                provider="codex",
                input_tokens=4000,
                output_tokens=500,
                cached_input_tokens=1000,
                reasoning_output_tokens=200,
            ),
            _exact_reviewer_verdict(
                "Looks good.\nVerdict: PASS\n",
                provider="codex",
                model="codex-json-model",
            ),
        ],
    )
    runner, _, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    records = [
        json.loads(line)
        for line in (
            repo / "tools/logs/demo-i1/cost.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    impl_rec = next(
        rec for rec in records
        if rec["task"] == "I1-T1" and rec["step"] == "IMPL"
    )
    review_rec = next(
        rec for rec in records
        if rec["task"] == "I1-T1" and rec["step"] == "REVIEW"
    )
    assert impl_rec["estimated"] is False
    assert impl_rec["provider"] == "claude"
    assert impl_rec["model"] == "claude-json-model"
    assert impl_rec["cached_input_tokens"] == 300
    assert impl_rec["cache_creation_input_tokens"] == 400
    assert impl_rec["parser_status"] == "parsed"
    # Raw CLI dump is stripped from the persisted record; the parsed usage
    # fields asserted above remain intact.
    assert "raw_terminal_json" not in impl_rec["extra"]["agent_result_extra"]
    assert review_rec["estimated"] is False
    assert review_rec["provider"] == "codex"
    assert review_rec["model"] == "fixture-codex-dispatched"
    assert review_rec["cached_input_tokens"] == 1000
    assert review_rec["reasoning_output_tokens"] == 200
    assert review_rec["parser_status"] == "parsed"


def test_model_routing_does_not_change_independence_check(repo: Path):
    _add_t1_model_routing(repo, "architecture_core_logic")
    _append_project_config(
        repo,
        """
model_routing:
  agent_overrides:
    claude:
      max:
        max:
          args: ["--model", "fixture-claude-model"]
""",
    )
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="anthropic")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 1
    assert impl.calls == []
    assert rev.calls == []
    assert any(
        event.get("kind") == "note"
        and event.get("meta", {}).get("reason") == "INDEPENDENCE"
        for event in store.events
    )


def test_task_kind_timeout_profile_defaults_to_existing_timeout(repo: Path):
    _add_t1_task_kind(repo, "characterization")
    _append_project_config(
        repo,
        """
timeouts:
  task_kind_profiles:
    characterization: {}
""",
    )
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0
    assert impl.calls[0]["timeout"] == 900
    assert any(
        event.get("kind") == "note"
        and event.get("task") == "I1-T1"
        and event.get("meta", {}).get("event")
        == "task_kind_timeout_profile_applied"
        for event in store.events
    )


def test_task_kind_timeout_profile_overrides_impl_timeout(repo: Path):
    _add_t1_task_kind(repo, "characterization")
    _append_project_config(
        repo,
        """
timeouts:
  task_kind_profiles:
    characterization:
      impl: 1234
""",
    )
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, _, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0
    assert impl.calls[0]["timeout"] == 1234


def test_unknown_task_kind_timeout_profile_fails_closed(repo: Path):
    _add_t1_task_kind(repo, "characterization")
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 1
    assert impl.calls == []
    assert rev.calls == []
    # Fails closed under a dedicated CONFIG reason, not the PREFLIGHT_SIZE
    # diff-size label (QA-A4): an unknown profile is a config error, not a
    # too-large diff.
    assert store.tasks["I1-T1"].status == STATUS_STOPPED_PREFIX + "CONFIG"
    stop_msg = store.tasks["I1-T1"].stop_msg or ""
    assert "unknown task_kind timeout profile" in stop_msg
    assert "config value was invalid" in stop_msg  # recovery note wired


def test_dual_required_task_without_secondary_stops_before_pr(repo: Path):
    _add_t1_model_routing(repo, "architecture_core_logic")
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )
    calls = SimpleNamespace(open_pr=[])

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, "https://example.com/pr/1"

    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=False,
        deps_kwargs={"open_pr": fake_open_pr},
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "DUAL_REVIEW_REQUIRED"
    assert t1.stop_reason == "DUAL_REVIEW_REQUIRED"
    assert "no secondary reviewer is configured" in t1.stop_msg
    assert "Recovery note: configure `--secondary-reviewer <agent>`" in (
        t1.stop_msg or ""
    )
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"
    assert calls.open_pr == []


def test_dual_required_rejects_same_family_secondary(repo: Path):
    _add_t1_model_routing(repo, "merge_critical_gate")
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )
    secondary = FakeAdapter(name="codex2", family="openai")
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev, "codex2": secondary},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        secondary_reviewer="codex2",
    )

    assert runner.run() == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "DUAL_REVIEW_REQUIRED"
    assert "different model family" in t1.stop_msg
    assert secondary.calls == []


def test_dual_required_proceeds_when_both_reviewers_pass(repo: Path):
    _add_t1_model_routing(repo, "architecture_core_logic")
    claude = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _exact_reviewer_verdict(
                "Independent pass.\nVerdict: PASS\n",
                provider="claude",
                model="claude-secondary-model",
                input_tokens=2222,
                output_tokens=333,
                cached_input_tokens=444,
                cache_creation_input_tokens=555,
            ),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    codex = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Primary pass.\nVerdict: PASS\n"),
            _reviewer_verdict("T2 pass.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "codex": codex},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        secondary_reviewer="claude",
    )

    assert runner.run() == 0
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE
    dual_events = [
        event for event in store.events
        if event.get("kind") == "note"
        and event.get("meta", {}).get("event") == "dual_review_passed"
    ]
    assert len(dual_events) == 1
    assert dual_events[0]["task"] == "I1-T1"
    assert dual_events[0]["meta"]["secondary_reviewer"] == "claude"
    assert (
        repo / "tools/logs/demo-i1/reviews/dual_review_I1-T1_claude.md"
    ).exists()
    lines = (repo / "tools/logs/demo-i1/cost.jsonl").read_text().splitlines()
    assert len(lines) == 5
    secondary_cost = json.loads(lines[2])
    assert secondary_cost["extra"]["review_role"] == "secondary"
    assert secondary_cost["extra"]["primary_reviewer"] == "codex"
    assert secondary_cost["estimated"] is False
    assert secondary_cost["provider"] == "claude"
    assert secondary_cost["model"] == "claude-secondary-model"
    assert secondary_cost["input_tokens"] == 2222
    assert secondary_cost["cached_input_tokens"] == 444
    assert secondary_cost["cache_creation_input_tokens"] == 555
    assert secondary_cost["parser_status"] == "parsed"


def test_dual_required_stops_when_primary_review_is_not_pass(repo: Path):
    _add_t1_model_routing(repo, "architecture_core_logic")
    claude = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _reviewer_verdict("Secondary should not run.\nVerdict: PASS\n"),
        ],
    )
    codex = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict(
                "Deferred nit.\nVerdict: CHANGES REQUIRED\nSeverity: should-fix\n"
            ),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "codex": codex},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        secondary_reviewer="claude",
    )

    assert runner.run() == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "DUAL_REVIEW_FAIL"
    assert "primary reviewer 'codex' returned Verdict: CHANGES REQUIRED" in (
        t1.stop_msg or ""
    )
    assert len(claude.calls) == 1


@pytest.mark.parametrize("verdict", ["BLOCKED", "CHANGES REQUIRED"])
def test_dual_required_stops_when_secondary_does_not_pass(
    repo: Path, verdict: str,
):
    _add_t1_model_routing(repo, "merge_critical_gate")
    claude = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _reviewer_verdict(f"Independent concern.\nVerdict: {verdict}\n"),
        ],
    )
    codex = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Primary pass.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "codex": codex},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        secondary_reviewer="claude",
    )

    assert runner.run() == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "DUAL_REVIEW_FAIL"
    assert t1.stop_reason == "DUAL_REVIEW_FAIL"
    assert f"Verdict: {verdict}" in (t1.stop_msg or "")
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"


def test_dual_required_stops_on_secondary_malformed_verdict(repo: Path):
    _add_t1_model_routing(repo, "architecture_core_logic")
    claude = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _reviewer_verdict("Looks independent but no final line.\n"),
        ],
    )
    codex = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Primary pass.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "codex": codex},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        secondary_reviewer="claude",
    )

    assert runner.run() == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "DUAL_REVIEW_MALFORMED"
    assert t1.stop_reason == "DUAL_REVIEW_MALFORMED"
    assert "secondary review output was malformed" in (t1.stop_msg or "")


@pytest.mark.parametrize(
    ("secondary_result", "expected_reason", "expected_msg"),
    [
        (
            AgentResult(
                exit_code=0,
                stdout="Looks good.\nVerdict: PASS\n",
                stderr="",
                duration_s=10.0,
                input_tokens=30,
                output_tokens=10,
                tokens_exact=False,
                partial=True,
            ),
            "DUAL_REVIEW_MALFORMED",
            "secondary reviewer timed out",
        ),
        (
            AgentResult(
                exit_code=2,
                stdout="Looks good.\nVerdict: PASS\n",
                stderr="adapter failed",
                duration_s=1.0,
                input_tokens=30,
                output_tokens=10,
                tokens_exact=False,
            ),
            "DUAL_REVIEW_FAIL",
            "secondary reviewer 'claude' exited 2",
        ),
    ],
)
def test_dual_required_stops_on_secondary_invocation_failure(
    repo: Path,
    secondary_result: AgentResult,
    expected_reason: str,
    expected_msg: str,
):
    _add_t1_model_routing(repo, "architecture_core_logic")

    def secondary_review(adapter, prompt, workdir):
        return secondary_result

    claude = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            secondary_review,
        ],
    )
    codex = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Primary pass.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "codex": codex},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        secondary_reviewer="claude",
    )

    assert runner.run() == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + expected_reason
    assert t1.stop_reason == expected_reason
    assert expected_msg in (t1.stop_msg or "")
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"
    assert (
        repo / "tools/logs/demo-i1/reviews/dual_review_I1-T1_claude.md"
    ).exists()


def test_non_dual_task_behavior_unchanged_without_secondary(repo: Path):
    _add_t1_model_routing(repo, "security_compliance")
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE


def test_runner_emits_pre_action_hook_events(repo: Path):
    handler = FakeHookHandler(event_name="task.before_pr")
    dispatcher = HookDispatcher([handler])
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        hook_dispatcher=dispatcher, implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    hook_names = [
        event["meta"]["event"]
        for event in store.events
        if event["kind"] == "hook"
    ]
    assert "task.before_start" in hook_names
    assert "task.before_branch_prepare" in hook_names
    assert "task.before_implement" in hook_names
    assert "task.before_review" in hook_names
    # Dry-run skips PR creation, so before_pr/before_merge are not emitted.
    assert "task.before_pr" not in hook_names
    assert handler.contexts == []


def test_blocking_hook_veto_stops_task_before_implement(repo: Path):
    handler = FakeHookHandler(
        event_name="task.before_implement",
        result=HookResult.veto(
            reason="prompt_policy",
            message="prompt policy failed",
        ),
    )
    dispatcher = HookDispatcher([handler])
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        hook_dispatcher=dispatcher, implementer="claude", reviewer="codex",
    )

    assert runner.run() == 1
    assert impl.calls == []
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "HOOK_VETO"
    assert t1.stop_reason == "HOOK_VETO"
    assert "prompt policy failed" in t1.stop_msg
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"
    assert any(
        event["meta"].get("event") == "hook_veto"
        for event in store.events
        if event["kind"] == "note"
    )


def test_parallel_wave_emits_before_branch_prepare_hook_per_task(repo: Path):
    # QA-A3 parity: the parallel path must emit task.before_branch_prepare for
    # each wave member, matching the serial path.
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)
    handler = FakeHookHandler(event_name="task.before_branch_prepare")
    dispatcher = HookDispatcher([handler])
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_parallel_edit_from_prompt, _parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        hook_dispatcher=dispatcher,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    prepared = [
        event["task"]
        for event in store.events
        if event["kind"] == "hook"
        and event["meta"]["event"] == "task.before_branch_prepare"
    ]
    assert sorted(prepared) == ["I1-T1", "I1-T2"]


def test_parallel_before_branch_prepare_veto_drops_only_that_task(repo: Path):
    # A veto on the parallel before_branch_prepare hook stops that task before
    # its worktree is created; the independent sibling still completes.
    _enable_parallel_config(repo)
    _enable_parallel_tasks(repo)

    class _VetoT1(FakeHookHandler):
        def handle(self, context):
            self.contexts.append(context)
            if context.task_id == "I1-T1":
                return HookResult.veto(reason="policy", message="t1 blocked")
            return HookResult.ok()

    handler = _VetoT1(event_name="task.before_branch_prepare")
    dispatcher = HookDispatcher([handler])
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_parallel_edit_from_prompt],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        hook_dispatcher=dispatcher,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude", reviewer="codex",
    )

    runner.run()

    assert store.tasks["I1-T1"].status == STATUS_STOPPED_PREFIX + "HOOK_VETO"
    assert store.tasks["I1-T1"].stop_reason == "HOOK_VETO"
    assert store.tasks["I1-T2"].status == STATUS_DONE
    # The vetoed task never reached implementation; only the sibling ran.
    assert len(impl.calls) == 1


def test_review_fail_blocks_downstream(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
        ],
    )
    # Reviewer says BLOCKED — T1 stops, T2 becomes BLOCKED_UPSTREAM.
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("nope\nVerdict: BLOCKED\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 1
    assert store.tasks["I1-T1"].status.startswith(STATUS_STOPPED_PREFIX)
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"


def test_unhandled_task_exception_stops_as_internal(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    def fail_execute(task, implementer, reviewer):
        raise RuntimeError("boom")

    monkeypatch.setattr(runner, "_execute_task", fail_execute)

    assert runner.run() == 1

    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "INTERNAL"
    assert t1.stop_reason == "INTERNAL"
    assert "RuntimeError" in (t1.stop_msg or "")
    assert "boom" in (t1.stop_msg or "")
    assert "python -m orch retry demo-i1 <task>" in (t1.stop_msg or "")
    internal_events = [
        event for event in store.events
        if event.get("kind") == "note"
        and event.get("task") == "I1-T1"
        and event.get("meta", {}).get("event") == "internal_error"
    ]
    assert internal_events
    assert internal_events[-1]["meta"]["exception_type"] == "RuntimeError"
    assert internal_events[-1]["meta"]["msg"] == "boom"
    assert not [
        event for event in store.events
        if event.get("kind") == "pair_swap"
    ]


def test_internal_stop_blocks_downstream_tasks(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    def fail_execute(task, implementer, reviewer):
        raise ValueError("broken state")

    monkeypatch.setattr(runner, "_execute_task", fail_execute)

    assert runner.run() == 1

    assert store.tasks["I1-T1"].status == STATUS_STOPPED_PREFIX + "INTERNAL"
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"


def test_review_changes_required_triggers_fix_then_pass(repo: Path):
    # First impl edit, then reviewer requests changes, fixer edits again,
    # second review passes.
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            # Fixer round
            _edit_file_on_invoke("src/a.py", "def a():\n    return 2\n"),
            # T2 implementation (no fix)
            _edit_file_on_invoke("src/b.py", "def b():\n    return 3\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("nits...\nVerdict: CHANGES REQUIRED\n"),
            _reviewer_verdict("all good\nVerdict: PASS\n"),
            _reviewer_verdict("ok\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 0
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_DONE
    assert t1.review_fix_rounds == 1


def test_runner_fix_cost_record_carries_exact_usage(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _exact_edit_file_on_invoke(
                "src/a.py",
                "def a():\n    return 2\n",
                provider="claude",
                model="claude-fix-model",
                input_tokens=1234,
                output_tokens=234,
                cached_input_tokens=345,
                cache_creation_input_tokens=456,
            ),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 3\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("nits...\nVerdict: CHANGES REQUIRED\n"),
            _reviewer_verdict("all good\nVerdict: PASS\n"),
            _reviewer_verdict("ok\nVerdict: PASS\n"),
        ],
    )
    runner, _, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    records = [
        json.loads(line)
        for line in (
            repo / "tools/logs/demo-i1/cost.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    [fix_rec] = [
        rec for rec in records
        if rec["task"] == "I1-T1" and rec["step"] == "FIX"
    ]
    assert fix_rec["estimated"] is False
    assert fix_rec["provider"] == "claude"
    assert fix_rec["model"] == "claude-fix-model"
    assert fix_rec["input_tokens"] == 1234
    assert fix_rec["cached_input_tokens"] == 345
    assert fix_rec["cache_creation_input_tokens"] == 456
    assert fix_rec["parser_status"] == "parsed"
    assert fix_rec["cause"] == "review"


def test_malformed_review_stops(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "x = 1\n")],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("I forgot the verdict line\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 1
    assert "REVIEW_MALFORMED" in store.tasks["I1-T1"].status


def test_independence_violation_halts_before_invocation(repo: Path):
    impl = FakeAdapter(name="claude", family="anthropic")
    # Reviewer shares family with implementer — default level is model_family.
    rev = FakeAdapter(name="codex", family="anthropic")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 1
    # No task touched
    assert all(
        t.status == "WAITING"
        for t in store.tasks.values()
        if t.id in {"I1-T1", "I1-T2"}
    ) or store.tasks == {}


def test_tasks_md_updated_on_success(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, _, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    assert runner.run() == 0
    main_text = (repo / "iterations" / "demo-i1" / "tasks.md").read_text()
    worktree_text = git(
        ["show", "demo/iteration-1:iterations/demo-i1/tasks.md"],
        cwd=repo,
        check=True,
    ).stdout
    assert "| I1-T1 | First     | TBD   | WAITING" in main_text
    assert "| I1-T2 | Second    | TBD   | WAITING" in main_text
    assert "| I1-T1 | First     | TBD   | DONE" in worktree_text
    assert "| I1-T2 | Second    | TBD   | DONE" in worktree_text


def test_done_commit_pushes_iteration_branch(repo: Path):
    pushes: list[dict] = []

    def fake_push(cwd: Path, branch: str):
        board_text = git(
            ["show", f"{branch}:iterations/demo-i1/tasks.md"],
            cwd=cwd,
            check=True,
        ).stdout
        pushes.append({"branch": branch, "tasks_md": board_text})
        return SimpleNamespace(ok=True, stdout="pushed\n", stderr="")

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        deps_kwargs={"push_branch": fake_push},
    )

    assert runner.run() == 0

    assert [push["branch"] for push in pushes] == [
        "demo/iteration-1",
        "demo/iteration-1",
    ]
    assert "| I1-T1 | First     | TBD   | DONE" in pushes[0]["tasks_md"]
    assert "| I1-T2 | Second    | TBD   | WAITING" in pushes[0]["tasks_md"]
    assert "| I1-T2 | Second    | TBD   | DONE" in pushes[1]["tasks_md"]
    pushed_events = [
        event for event in store.events
        if event.get("meta", {}).get("event") == "tasks_md_done_pushed"
    ]
    assert [event["task"] for event in pushed_events] == ["I1-T1", "I1-T2"]


def test_final_scope_gate_allows_allowed_union_and_status_updates(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    gate_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "final_scope_gate_passed"
    ]
    assert gate_events
    meta = gate_events[-1]["meta"]
    assert meta["allowed_files"] == ["src/a.py", "src/b.py"]
    assert meta["tasks_md_status_only"] is True
    assert "iterations/demo-i1/tasks.md" not in meta["changed_files"]


def test_final_scope_gate_ignores_tools_logs_run_artifacts(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    _inject_final_scope_leak(
        monkeypatch,
        relpath="tools/logs/demo-i1/artifact.txt",
        force_add=True,
    )
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    gate_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "final_scope_gate_passed"
    ]
    assert gate_events
    assert "tools/logs/demo-i1/artifact.txt" not in (
        gate_events[-1]["meta"]["changed_files"]
    )


def test_final_scope_gate_blocks_accumulated_outward_leak(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    _inject_final_scope_leak(monkeypatch)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "final_scope_gate_failed"
    ]
    assert failed
    assert failed[-1]["meta"]["reason"] == "SCOPE"
    assert "src/outside.py" in failed[-1]["meta"]["msg"]
    assert "Allowed files" in failed[-1]["meta"]["msg"]


def test_final_scope_gate_allows_operator_approved_exception(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    _inject_final_scope_leak(monkeypatch)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    store.log_dir.mkdir(parents=True, exist_ok=True)
    (store.log_dir / "scope_exceptions.md").write_text(
        "\n".join([
            "approved_by: operator",
            "reason: approved review fold-in",
            "paths:",
            "- src/outside.py",
        ])
    )

    assert runner.run() == 0

    applied = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "final_scope_exception_applied"
    ]
    assert applied
    assert applied[-1]["meta"]["paths"] == ["src/outside.py"]
    assert applied[-1]["meta"]["approved_by"] == "operator"
    assert applied[-1]["meta"]["reason"] == "approved review fold-in"


def test_final_scope_gate_rejects_malformed_exception_evidence(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    _inject_final_scope_leak(monkeypatch)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    store.log_dir.mkdir(parents=True, exist_ok=True)
    (store.log_dir / "scope_exceptions.md").write_text(
        "\n".join([
            "reason: missing approval identity",
            "paths:",
            "- src/outside.py",
        ])
    )

    rc = runner.run()

    assert rc == 1
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "final_scope_gate_failed"
    ]
    assert failed
    assert "malformed scope exception evidence" in failed[-1]["meta"]["msg"]
    assert "missing approved_by" in failed[-1]["meta"]["msg"]


# ---------------------------------------------------------------------------
# Final nav-discoverability gate
# ---------------------------------------------------------------------------


def _use_route_visible_tasks_md(
    repo: Path,
    *,
    include_nav_anchor: bool = False,
    configure_visibility: bool = True,
) -> None:
    """Swap the demo tasks.md to declare route-visible allowed files.

    Two task variants:
      - I1-T1 always touches ``app/routes/widgets.py`` (route-visible).
      - I1-T2 touches ``app/templates/widgets/list.html`` (route-visible)
        and optionally ``app/templates/base.html`` (nav anchor) when
        ``include_nav_anchor`` is true.
    """
    nav_line = "app/templates/base.html\n" if include_nav_anchor else ""
    content = TASKS_MD.replace(
        "**Allowed files:**\n```\nsrc/a.py\n```",
        "**Allowed files:**\n```\napp/routes/widgets.py\n```",
        1,
    ).replace(
        "**Allowed files:**\n```\nsrc/b.py\n```",
        "**Allowed files:**\n```\napp/templates/widgets/list.html\n"
        f"{nav_line}```",
        1,
    )
    _update_tasks_md(repo, content)
    if configure_visibility:
        _append_project_config(repo, EXAMPLE_ROUTE_VISIBILITY_CONFIG)


def test_final_nav_gate_passes_when_no_route_visible_files(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _edit_file_on_invoke("src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    gate_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_passed"
    ]
    assert gate_events
    assert gate_events[-1]["meta"]["route_visible_surfaces"] == []
    assert (
        "no route-visible surfaces" in gate_events[-1]["meta"]["reason"]
    )


def test_final_nav_gate_generic_config_is_inert_for_example_paths(repo: Path):
    _use_route_visible_tasks_md(repo, configure_visibility=False)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("app/routes/widgets.py", "x = 1\n"),
            _edit_file_on_invoke(
                "app/templates/widgets/list.html", "<h1>list</h1>\n",
            ),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    gate_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_passed"
    ]
    assert gate_events
    assert gate_events[-1]["meta"]["route_visible_surfaces"] == []
    assert gate_events[-1]["meta"]["nav_anchor_updates"] == []
    assert (
        "no route-visible surfaces" in gate_events[-1]["meta"]["reason"]
    )


def test_final_nav_gate_passes_when_nav_anchor_in_diff(repo: Path):
    _use_route_visible_tasks_md(repo, include_nav_anchor=True)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("app/routes/widgets.py", "x = 1\n"),
            _edit_file_on_invoke(
                "app/templates/widgets/list.html", "<h1>list</h1>\n",
            ),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    # T2 is allowed to touch both the template AND base.html. Replace
    # the simple single-file edit step with a combined-edit step so the
    # iteration diff contains the nav anchor.
    def _combined_t2(adapter, prompt, workdir):
        for rel, body in (
            ("app/templates/widgets/list.html", "<h1>list</h1>\n"),
            ("app/templates/base.html", "<nav>x</nav>\n"),
        ):
            target = Path(workdir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body)
        stage_all(Path(workdir))
        commit(Path(workdir), f"{adapter.name}: edit list + base")
        return AgentResult(
            exit_code=0, stdout="", stderr="", duration_s=0.1,
            input_tokens=50, output_tokens=20, tokens_exact=False,
        )
    impl.script[1] = _combined_t2

    assert runner.run() == 0

    gate_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_passed"
    ]
    assert gate_events
    meta = gate_events[-1]["meta"]
    assert "app/routes/widgets.py" in meta["route_visible_surfaces"]
    assert "app/templates/widgets/list.html" in meta["route_visible_surfaces"]
    assert "app/templates/base.html" in meta["nav_anchor_updates"]


def test_final_nav_gate_blocks_when_no_nav_anchor_and_no_evidence(
    repo: Path,
):
    _use_route_visible_tasks_md(repo, include_nav_anchor=False)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("app/routes/widgets.py", "x = 1\n"),
            _edit_file_on_invoke(
                "app/templates/widgets/list.html", "<h1>list</h1>\n",
            ),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_failed"
    ]
    assert failed
    msg = failed[-1]["meta"]["msg"]
    assert "app/routes/widgets.py" in msg
    assert "app/templates/widgets/list.html" in msg
    for anchor in EXAMPLE_NAV_ANCHOR_PATHS:
        assert anchor in msg
    assert "nav_discoverability.md" in msg


def test_final_nav_gate_allows_operator_approved_no_nav_exception(
    repo: Path,
):
    _use_route_visible_tasks_md(repo, include_nav_anchor=False)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("app/routes/widgets.py", "x = 1\n"),
            _edit_file_on_invoke(
                "app/templates/widgets/list.html", "<h1>list</h1>\n",
            ),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    store.log_dir.mkdir(parents=True, exist_ok=True)
    (store.log_dir / "nav_discoverability.md").write_text(
        "\n".join([
            "approved_by: operator",
            "reason: widgets sub-page reached from admin index, not nav",
            "paths:",
            "- app/routes/widgets.py",
            "- app/templates/widgets/list.html",
        ])
    )

    assert runner.run() == 0

    applied = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_exception_applied"
    ]
    assert applied
    meta = applied[-1]["meta"]
    assert set(meta["approved_paths"]) == {
        "app/routes/widgets.py",
        "app/templates/widgets/list.html",
    }
    assert meta["approved_by"] == "operator"
    assert "widgets sub-page" in meta["reason"]


def test_final_nav_gate_rejects_partial_evidence_coverage(repo: Path):
    """Evidence that covers only some route-visible files still blocks."""
    _use_route_visible_tasks_md(repo, include_nav_anchor=False)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("app/routes/widgets.py", "x = 1\n"),
            _edit_file_on_invoke(
                "app/templates/widgets/list.html", "<h1>list</h1>\n",
            ),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    store.log_dir.mkdir(parents=True, exist_ok=True)
    (store.log_dir / "nav_discoverability.md").write_text(
        "\n".join([
            "approved_by: operator",
            "reason: only covers the route, forgot the template",
            "paths:",
            "- app/routes/widgets.py",
        ])
    )

    rc = runner.run()

    assert rc == 1
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_failed"
    ]
    assert failed
    msg = failed[-1]["meta"]["msg"]
    assert "app/templates/widgets/list.html" in msg
    assert failed[-1]["meta"]["missing_paths"] == [
        "app/templates/widgets/list.html"
    ]


def test_final_nav_gate_rejects_malformed_evidence(repo: Path):
    _use_route_visible_tasks_md(repo, include_nav_anchor=False)
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("app/routes/widgets.py", "x = 1\n"),
            _edit_file_on_invoke(
                "app/templates/widgets/list.html", "<h1>list</h1>\n",
            ),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    store.log_dir.mkdir(parents=True, exist_ok=True)
    (store.log_dir / "nav_discoverability.md").write_text(
        "\n".join([
            "reason: missing approval identity",
            "paths:",
            "- app/routes/widgets.py",
            "- app/templates/widgets/list.html",
        ])
    )

    rc = runner.run()

    assert rc == 1
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_failed"
    ]
    assert failed
    msg = failed[-1]["meta"]["msg"]
    assert "malformed" in msg
    assert "missing approved_by" in msg
    # Malformed evidence must still surface the operator-facing remediation
    # text: the uncovered route-visible paths, the nav-anchor expectation,
    # and the evidence file path.
    assert "app/routes/widgets.py" in msg
    assert "app/templates/widgets/list.html" in msg
    for anchor in EXAMPLE_NAV_ANCHOR_PATHS:
        assert anchor in msg
    assert "nav_discoverability.md" in msg
    assert failed[-1]["meta"]["missing_paths"] == [
        "app/routes/widgets.py",
        "app/templates/widgets/list.html",
    ]


# ---------------------------------------------------------------------------
# Non-dry-run PR/merge path with fake gh layer
# ---------------------------------------------------------------------------


def test_guarded_merge_auto_merges_on_pass(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    # Skip T2 by running only T1 — remove T2 from the board via options?
    # Simpler: let both run; supply enough scripts.
    impl.script.append(_edit_file_on_invoke("src/b.py", "b=2\n"))
    rev.script.append(_reviewer_verdict("Verdict: PASS\n"))

    calls = SimpleNamespace(
        open_pr=[], merge_pr=[], comment_pr=[], wait_for_ci=[],
    )

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, f"https://example.com/pr/{len(calls.open_pr)}"

    def fake_merge_pr(*, cwd, pr_url):
        calls.merge_pr.append(pr_url)
        return True, f"merged deadbeef{len(calls.merge_pr)}"

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    def fake_wait(branch, *, cwd, ci_wait_seconds, **kw):
        calls.wait_for_ci.append(branch)
        return SimpleNamespace(
            passed=True, conclusion="success", elapsed_s=1.0,
        )

    cfg = load_config(repo / ".orch" / "project.yaml")
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    handler = FakeHookHandler(event_name="task.before_pr")
    dispatcher = HookDispatcher([handler])
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
        hook_dispatcher=dispatcher,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(
        repo_root=repo,
        cwd=orch_cwd,
        open_pr=fake_open_pr, merge_pr=fake_merge_pr,
        comment_pr=fake_comment, wait_for_ci=fake_wait,
    )
    opts = RunOptions(
        implementer="claude", reviewer="codex",
        dry_run=False, poll_ci=True,
    )
    runner = IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": impl, "codex": rev}, options=opts, deps=deps,
    )
    rc = runner.run()
    assert rc == 0
    assert len(calls.open_pr) == 2
    assert len(calls.merge_pr) == 2
    assert all(t.auto_merged for t in store.tasks.values()
               if t.status == STATUS_DONE)
    assert len(calls.wait_for_ci) == 2
    hook_names = [
        event["meta"]["event"]
        for event in store.events
        if event["kind"] == "hook"
    ]
    assert hook_names.count("task.before_pr") == 2
    assert hook_names.count("task.before_merge") == 2


def test_guarded_merge_conflict_routes_to_human_merge(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    calls = SimpleNamespace(
        open_pr=[], merge_pr=[], comment_pr=[], wait_for_ci=[],
    )

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, "https://example.com/pr/1"

    def fake_merge_pr(*, cwd, pr_url):
        calls.merge_pr.append(pr_url)
        return False, "merge conflict: src/a.py"

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    def fake_wait(branch, *, cwd, ci_wait_seconds, **kw):
        calls.wait_for_ci.append(branch)
        return SimpleNamespace(
            passed=True, conclusion="success", elapsed_s=1.0,
        )

    def fake_query(pr_url, *, cwd):
        assert pr_url == "https://example.com/pr/1"
        return PrSnapshot(state="OPEN", merge_sha=None, rollup=[])

    cfg = load_config(repo / ".orch" / "project.yaml")
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(
        repo_root=repo,
        cwd=orch_cwd,
        open_pr=fake_open_pr,
        merge_pr=fake_merge_pr,
        comment_pr=fake_comment,
        wait_for_ci=fake_wait,
        query_pr_state=fake_query,
    )
    opts = RunOptions(
        implementer="claude", reviewer="codex",
        dry_run=False, poll_ci=True,
    )
    runner = IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": impl, "codex": rev}, options=opts, deps=deps,
    )

    rc = runner.run()

    assert rc == 1
    assert calls.open_pr == [("I1-T1: First", "demo/iteration-1", "d/i1/t1")]
    assert calls.merge_pr == ["https://example.com/pr/1"]
    assert calls.wait_for_ci == ["d/i1/t1"]
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_NEEDS_HUMAN_MERGE
    assert t1.auto_merged is False
    assert t1.merge_sha is None
    merge_failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("task") == "I1-T1"
        and e.get("meta", {}).get("event") == "merge_failed"
    ]
    assert merge_failed[-1]["meta"]["msg"] == "merge conflict: src/a.py"


def test_no_ci_local_merge_blocks_noop_acceptance_without_override(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    calls = SimpleNamespace(open_pr=[], merge_pr=[], comment_pr=[])

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, "https://example.com/pr/1"

    def fake_merge_pr(*, cwd, pr_url):
        calls.merge_pr.append(pr_url)
        raise AssertionError("gh merge must not run in no-CI mode")

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    cfg = load_config(repo / ".orch" / "project.yaml")
    cfg.data["auto_merge"]["no_ci"] = True
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(
        repo_root=repo,
        cwd=orch_cwd,
        open_pr=fake_open_pr,
        merge_pr=fake_merge_pr,
        comment_pr=fake_comment,
    )
    opts = RunOptions(
        implementer="claude", reviewer="codex",
        dry_run=False, poll_ci=True,
    )
    runner = IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": impl, "codex": rev}, options=opts, deps=deps,
    )

    rc = runner.run()

    assert rc == 1
    assert calls.open_pr == [("I1-T1: First", "demo/iteration-1", "d/i1/t1")]
    assert calls.merge_pr == []
    assert calls.comment_pr == []
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_NEEDS_HUMAN_MERGE
    assert t1.stop_reason == "NOOP_ACCEPTANCE"
    assert "--allow-noop-acceptance" in (t1.stop_msg or "")
    block_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "noop_acceptance_blocked"
    ]
    assert block_events[-1]["meta"]["test_cmd"] == "true"


def test_no_ci_local_merge_advances_iteration_without_ci_or_gh_merge(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _assert_visible_then_edit("src/a.py", "a=1\n", "src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    calls = SimpleNamespace(
        open_pr=[], merge_pr=[], comment_pr=[], wait_for_ci=[],
    )

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, f"https://example.com/pr/{len(calls.open_pr)}"

    def fake_merge_pr(*, cwd, pr_url):
        calls.merge_pr.append(pr_url)
        raise AssertionError("merge_pr must not run in no-CI mode")

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    def fake_wait(branch, *, cwd, ci_wait_seconds, **kw):
        calls.wait_for_ci.append(branch)
        raise AssertionError("wait_for_ci must not run in no-CI mode")

    cfg = load_config(repo / ".orch" / "project.yaml")
    cfg.data["auto_merge"]["no_ci"] = True
    cfg.data["stack"]["test"] = f"{sys.executable} -c \"print('tested')\""
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    handler = FakeHookHandler(event_name="task.before_merge")
    dispatcher = HookDispatcher([handler])
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
        hook_dispatcher=dispatcher,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(
        repo_root=repo,
        cwd=orch_cwd,
        open_pr=fake_open_pr,
        merge_pr=fake_merge_pr,
        comment_pr=fake_comment,
        wait_for_ci=fake_wait,
    )
    opts = RunOptions(
        implementer="claude", reviewer="codex",
        dry_run=False, poll_ci=True,
    )
    runner = IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": impl, "codex": rev}, options=opts, deps=deps,
    )

    rc = runner.run()

    assert rc == 0
    assert len(calls.open_pr) == 2
    assert len(calls.comment_pr) == 2
    assert calls.wait_for_ci == []
    assert calls.merge_pr == []
    assert all("no-CI mode" in body for _, body in calls.comment_pr)
    assert all("pushed after the DONE/task-board commit" in body for _, body in calls.comment_pr)
    assert len(handler.contexts) == 2
    assert all(ctx.payload["no_ci"] is True for ctx in handler.contexts)
    assert len(impl.calls) == 2
    t1 = store.tasks["I1-T1"]
    t2 = store.tasks["I1-T2"]
    assert t1.status == STATUS_DONE
    assert t2.status == STATUS_DONE
    assert t1.pr_url == "https://example.com/pr/1"
    assert t2.pr_url == "https://example.com/pr/2"
    assert t1.auto_merged is True
    assert t2.auto_merged is True
    assert t1.merge_sha
    assert t2.merge_sha
    merge_events = [
        event for event in store.events
        if event.get("kind") in {"merge_intent", "merge_complete", "merge"}
    ]
    assert [(event["kind"], event["task"]) for event in merge_events] == [
        ("merge_intent", "I1-T1"),
        ("merge_complete", "I1-T1"),
        ("merge", "I1-T1"),
        ("merge_intent", "I1-T2"),
        ("merge_complete", "I1-T2"),
        ("merge", "I1-T2"),
    ]
    assert merge_events[0]["meta"]["target_sha_before"] != (
        merge_events[0]["meta"]["task_sha"]
    )
    assert merge_events[1]["meta"]["merge_sha"] == t1.merge_sha
    assert merge_events[4]["meta"]["merge_sha"] == t2.merge_sha
    assert git(
        ["show", "demo/iteration-1:src/a.py"], cwd=repo, check=True,
    ).stdout == "a=1\n"
    assert git(
        ["show", "demo/iteration-1:src/b.py"], cwd=repo, check=True,
    ).stdout == "b=2\n"


def test_no_ci_local_merge_allows_noop_acceptance_with_reason(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "a=1\n"),
            _assert_visible_then_edit("src/a.py", "a=1\n", "src/b.py", "b=2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    calls = SimpleNamespace(open_pr=[], comment_pr=[])

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, f"https://example.com/pr/{len(calls.open_pr)}"

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    cfg = load_config(repo / ".orch" / "project.yaml")
    cfg.data["auto_merge"]["no_ci"] = True
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(
        repo_root=repo,
        cwd=orch_cwd,
        open_pr=fake_open_pr,
        comment_pr=fake_comment,
    )
    opts = RunOptions(
        implementer="claude", reviewer="codex",
        allow_noop_acceptance_reason="operator will run full suite",
        dry_run=False, poll_ci=True,
    )
    runner = IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": impl, "codex": rev}, options=opts, deps=deps,
    )

    rc = runner.run()

    assert rc == 0
    assert [t.status for t in store.tasks.values()] == [
        STATUS_DONE,
        STATUS_DONE,
    ]
    allow_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "noop_acceptance_allowed"
    ]
    assert [e["task"] for e in allow_events] == ["I1-T1", "I1-T2"]
    assert all(
        e["meta"]["reason"] == "operator will run full suite"
        for e in allow_events
    )
    assert all(e["meta"]["test_cmd"] == "true" for e in allow_events)


def test_no_ci_local_merge_failure_stops_needs_human_merge(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "a=1\n")],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    calls = SimpleNamespace(open_pr=[], comment_pr=[])

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, "https://example.com/pr/1"

    def fake_comment(*, cwd, pr_url, body):
        calls.comment_pr.append((pr_url, body))
        return True

    def fail_local_merge(self, task, *, message):
        raise GitError("local merge exploded")

    monkeypatch.setattr(
        IterationRunner, "_merge_task_branch_locally", fail_local_merge,
    )
    cfg = load_config(repo / ".orch" / "project.yaml")
    cfg.data["auto_merge"]["no_ci"] = True
    cfg.data["stack"]["test"] = f"{sys.executable} -c \"print('tested')\""
    board = parse_tasks_md(repo / "iterations" / "demo-i1" / "tasks.md")
    log_dir = repo / "tools" / "logs" / "demo-i1"
    store = StateStore(
        log_dir=log_dir, iteration="demo-i1",
        iter_branch=board.iteration_branch,
    )
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=costs(cfg),
        iteration="demo-i1",
    )
    orch_cwd = ensure_orch_workdir(repo, "demo-i1", board.iteration_branch)
    deps = RunnerDeps(
        repo_root=repo,
        cwd=orch_cwd,
        open_pr=fake_open_pr,
        comment_pr=fake_comment,
    )
    opts = RunOptions(
        implementer="claude", reviewer="codex",
        dry_run=False, poll_ci=True,
    )
    runner = IterationRunner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": impl, "codex": rev}, options=opts, deps=deps,
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_NEEDS_HUMAN_MERGE
    assert t1.auto_merged is False
    assert t1.merge_sha is None
    assert calls.open_pr == [("I1-T1: First", "demo/iteration-1", "d/i1/t1")]
    assert calls.comment_pr == []
    failure_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "local_merge_failed"
    ]
    assert failure_events[-1]["meta"]["msg"] == "local merge exploded"


def test_resume_reconciles_externally_merged_human_merge_task(repo: Path):
    impl = FakeAdapter(name="claude", family="anthropic", script=[])
    rev = FakeAdapter(name="codex", family="openai", script=[])
    query_calls = []

    def fake_query_pr_state(pr_url: str, *, cwd: Path):
        query_calls.append({"pr_url": pr_url, "cwd": cwd})
        return PrSnapshot(
            state="MERGED",
            merge_sha="deadbeef" * 5,
            rollup=[],
        )

    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        deps_kwargs={"query_pr_state": fake_query_pr_state},
    )
    store.record_pr("I1-T1", "https://example.com/pr/1")
    store.task_transition("I1-T1", STATUS_NEEDS_HUMAN_MERGE)
    store.task_transition("I1-T2", STATUS_DONE)
    runner._final_scope_gate = lambda: None
    runner._final_nav_discoverability_gate = lambda: None

    rc = runner.run()

    assert rc == 0
    assert query_calls == [
        {"pr_url": "https://example.com/pr/1", "cwd": runner.deps.cwd}
    ]
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_DONE
    assert t1.auto_merged is False
    assert t1.merge_sha == "deadbeef" * 5
    assert t1.stop_reason == "external_merge_detected"
    assert t1.stop_msg == f"PR merged externally at {'deadbeef' * 5}"
    assert impl.calls == []
    assert rev.calls == []


def test_external_merge_scan_includes_in_progress_tasks(repo: Path):
    impl = FakeAdapter(name="claude", family="anthropic", script=[])
    rev = FakeAdapter(name="codex", family="openai", script=[])
    query_calls = []

    def fake_query_pr_state(pr_url: str, *, cwd: Path):
        query_calls.append({"pr_url": pr_url, "cwd": cwd})
        return PrSnapshot(
            state="MERGED",
            merge_sha="feedface" * 5,
            rollup=[],
        )

    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        deps_kwargs={"query_pr_state": fake_query_pr_state},
    )
    store.record_pr("I1-T1", "https://example.com/pr/1")
    store.task_transition("I1-T1", STATUS_IN_PROGRESS)

    runner._check_external_merges()

    assert query_calls == [
        {"pr_url": "https://example.com/pr/1", "cwd": runner.deps.cwd}
    ]
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_DONE
    assert t1.auto_merged is False
    assert t1.merge_sha == "feedface" * 5
    assert t1.stop_reason == "external_merge_detected"


def test_runner_uses_dedicated_sub_worktree_and_cleans_up_on_success(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, _, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    workdir = runner.deps.cwd

    assert workdir == orch_workdir(repo, "demo-i1")
    assert runner.deps.repo_root == repo
    assert workdir.exists()
    assert runner.run() == 0
    assert not workdir.exists()
    assert all(call["workdir"] == workdir for call in impl.calls)
    assert all(call["workdir"] == workdir for call in rev.calls)


def test_runner_preserves_sub_worktree_on_failure(repo: Path):
    impl = FakeAdapter(
        name="claude", family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(
        name="codex", family="openai",
        script=[_reviewer_verdict("nope\nVerdict: BLOCKED\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    workdir = runner.deps.cwd

    assert runner.run() == 1
    assert workdir.exists()
    note_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "orch_workdir_preserved"
    ]
    assert note_events
    assert str(workdir) in note_events[-1]["meta"]["msg"]


# ---------------------------------------------------------------------------
# v2.2 — bounded one-shot model-pair swap on REVIEW_FAIL
# ---------------------------------------------------------------------------


def test_v22_pair_swap_recovers_after_review_fail(repo: Path):
    """Primary pair fails review convergence; swapped pair converges."""
    # Primary pair: claude impl, codex reviewer.
    #   IMPL #1  — claude writes src/a.py
    #   REVIEW #1 — codex CHANGES REQUIRED
    #   FIX      — claude re-edits src/a.py
    #   REVIEW #2 — codex CHANGES REQUIRED (REVIEW_FAIL, round 2)
    # Swap: codex impl, claude reviewer.
    #   IMPL     — codex re-writes src/a.py from clean branch
    #   REVIEW   — claude PASS
    # Then T2 runs on the primary pair again.
    claude = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            # Primary IMPL
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            # Primary FIX (reviewer requested change)
            _edit_file_on_invoke("src/a.py", "def a():\n    return 2\n"),
            # Swap REVIEW — now claude reviews
            _reviewer_verdict("clean\nVerdict: PASS\n"),
            # T2 IMPL (back to primary pair — claude implements)
            _edit_file_on_invoke("src/b.py", "def b():\n    return 3\n"),
        ],
    )
    codex = FakeAdapter(
        name="codex", family="openai",
        script=[
            # Primary REVIEW round 1
            _reviewer_verdict("nit.\nVerdict: CHANGES REQUIRED\n"),
            # Primary REVIEW round 2 → REVIEW_FAIL
            _reviewer_verdict("still wrong.\nVerdict: CHANGES REQUIRED\n"),
            # Swap IMPL — now codex implements
            _edit_file_on_invoke("src/a.py", "def a():\n    return 99\n"),
            # T2 REVIEW (back to primary pair — codex reviews)
            _reviewer_verdict("good\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": claude, "codex": codex}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 0
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_DONE
    assert t1.swap_attempted is True
    assert t1.swap_implementer == "codex"
    assert t1.swap_reviewer == "claude"
    assert t1.swap_reason == "REVIEW_FAIL"
    assert t1.swap_outcome == "PASS"
    # A single pair_swap event is recorded.
    swap_events = [e for e in store.events if e["kind"] == "pair_swap"]
    assert len(swap_events) == 1
    assert swap_events[0]["task"] == "I1-T1"
    # T2 ran on primary pair and completed.
    assert store.tasks["I1-T2"].status == STATUS_DONE


def test_v22_pair_swap_is_one_shot_on_double_fail(repo: Path):
    """Swap is only attempted once; if the swapped pair also fails, stop."""
    claude = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/a.py", "def a():\n    return 2\n"),
            # Swap REVIEW round 1 — claude reviews, rejects
            _reviewer_verdict("nope\nVerdict: CHANGES REQUIRED\n"),
            # Swap REVIEW round 2 — rejects again → REVIEW_FAIL again
            _reviewer_verdict("still nope\nVerdict: CHANGES REQUIRED\n"),
        ],
    )
    codex = FakeAdapter(
        name="codex", family="openai",
        script=[
            _reviewer_verdict("Verdict: CHANGES REQUIRED\n"),
            _reviewer_verdict("Verdict: CHANGES REQUIRED\n"),
            # Swap IMPL
            _edit_file_on_invoke("src/a.py", "def a():\n    return 99\n"),
            # Swap FIX (claude-the-reviewer requested changes on round 1)
            _edit_file_on_invoke("src/a.py", "def a():\n    return 100\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": claude, "codex": codex}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.swap_attempted is True
    assert t1.swap_outcome == "REVIEW_FAIL"
    # Only one pair_swap event — second REVIEW_FAIL does NOT swap again.
    swap_events = [e for e in store.events if e["kind"] == "pair_swap"]
    assert len(swap_events) == 1
    # T2 cascades to BLOCKED_UPSTREAM.
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"


def test_v22_pair_swap_skipped_on_non_review_stop(repo: Path):
    """Swap only fires for REVIEW_FAIL — scope/structural/etc. bypass it."""
    # Claude writes an out-of-scope file so _structural_checks flips to
    # STOP_SCOPE (after auto-revert wipes the diff → empty diff guard).
    # No reviewer calls expected.
    claude = FakeAdapter(
        name="claude", family="anthropic",
        script=[
            _edit_file_on_invoke("src/outside.py", "x = 1\n"),
        ],
    )
    codex = FakeAdapter(name="codex", family="openai", script=[])
    runner, store, _ = _make_runner(
        repo, {"claude": claude, "codex": codex}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    rc = runner.run()
    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.swap_attempted is False  # no swap for SCOPE
    swap_events = [e for e in store.events if e["kind"] == "pair_swap"]
    assert swap_events == []


def test_v22_pair_swap_rejects_same_family_swap(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    claude = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/a.py", "def a():\n    return 2\n"),
        ],
    )
    sonnet = FakeAdapter(
        name="sonnet",
        family="anthropic",
        script=[
            _reviewer_verdict("Verdict: CHANGES REQUIRED\n"),
            _reviewer_verdict("Verdict: CHANGES REQUIRED\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": claude, "sonnet": sonnet},
        dry_run=True,
        implementer="claude",
        reviewer="sonnet",
    )
    monkeypatch.setattr(
        runner,
        "_check_global_independence",
        lambda _implementer, _reviewer: SimpleNamespace(
            ok=True, reason="forced global pass",
        ),
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "REVIEW_FAIL"
    assert t1.stop_reason == "REVIEW_FAIL"
    assert t1.swap_attempted is False
    assert t1.swap_implementer is None
    assert t1.swap_reviewer is None
    assert [e for e in store.events if e["kind"] == "pair_swap"] == []
    assert store.tasks["I1-T2"].status == "BLOCKED_UPSTREAM"


def test_impl_nonzero_exit_with_commit_stops_as_impl_failed(repo: Path):
    """Non-zero adapter exit must STOP as IMPL_FAILED even when a commit exists."""

    def _commit_then_fail(adapter, prompt, workdir):
        p = Path(workdir) / "src" / "a.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("def a():\n    return 1\n")
        stage_all(Path(workdir))
        commit(Path(workdir), f"{adapter.name}: partial src/a.py")
        return AgentResult(
            exit_code=1,
            stdout="partial output",
            stderr="rate limit exceeded",
            duration_s=10.0,
            input_tokens=50,
            output_tokens=0,
            tokens_exact=False,
        )

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_commit_then_fail],
    )
    rev = FakeAdapter(name="codex", family="openai", script=[])
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.stop_reason == "IMPL_FAILED"
    assert "will not switch accounts" in (t1.stop_msg or "")
    note_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and "impl_fast_exit" in e.get("meta", {}).get("msg", "")
    ]
    assert note_events, "expected impl_fast_exit note event for 0-token fast exit"
    assert note_events[0]["meta"]["exit_code"] == 1
    assert note_events[0]["meta"]["stderr_tail"] == "rate limit exceeded"
    assert rev.calls == []


def test_all_partial_attempts_stop_as_impl_timeout_with_salvage(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _partial_timeout_write_untracked,
            _partial_timeout_write_untracked,
            _partial_timeout_write_untracked,
        ],
    )
    rev = FakeAdapter(name="codex", family="openai", script=[])
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "IMPL_TIMEOUT"
    assert t1.stop_reason == "IMPL_TIMEOUT"
    salvage_branch = "salvage/demo-i1/I1-T1"
    assert branch_exists(runner.deps.cwd, salvage_branch)
    assert git(
        ["show", f"{salvage_branch}:src/timeout-3.txt"],
        cwd=runner.deps.cwd,
        check=True,
    ).stdout == "attempt 3\n"
    assert current_sha(runner.deps.cwd, "d/i1/t1") == current_sha(
        runner.deps.cwd, "demo/iteration-1"
    )
    assert working_tree_clean(runner.deps.cwd)
    salvage_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "impl_salvage"
    ]
    assert len(salvage_events) == 1
    assert salvage_events[0]["meta"]["branch"] == salvage_branch
    assert len(salvage_events[0]["meta"]["sha"]) == 40
    assert salvage_branch in (t1.stop_msg or "")
    assert "NOT reviewed or resumed automatically" in (t1.stop_msg or "")
    assert rev.calls == []


def test_all_partial_attempts_clean_tree_no_salvage(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _partial_timeout_no_write,
            _partial_timeout_no_write,
            _partial_timeout_no_write,
        ],
    )
    rev = FakeAdapter(name="codex", family="openai", script=[])
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "IMPL_TIMEOUT"
    assert t1.stop_reason == "IMPL_TIMEOUT"
    assert not branch_exists(runner.deps.cwd, "salvage/demo-i1/I1-T1")
    assert working_tree_clean(runner.deps.cwd)
    assert not [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "impl_salvage"
    ]
    assert rev.calls == []


def test_exit_failure_path_stays_impl_failed(repo: Path):
    def _exit_failure(adapter, prompt, workdir):
        return AgentResult(
            exit_code=2,
            stdout="",
            stderr="adapter exploded",
            duration_s=61.0,
            input_tokens=50,
            output_tokens=1,
            tokens_exact=False,
        )

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_exit_failure],
    )
    rev = FakeAdapter(name="codex", family="openai", script=[])
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "IMPL_FAILED"
    assert t1.stop_reason == "IMPL_FAILED"
    assert "adapter exited 2 (attempt 1)" in (t1.stop_msg or "")
    impl_events = [
        e for e in store.events
        if e.get("kind") == "impl_attempt"
        and e.get("task") == "I1-T1"
    ]
    assert len(impl_events) == 1
    assert impl_events[0]["meta"]["classification"] == "unknown"
    assert impl_events[0]["meta"]["stderr_tail"] == "adapter exploded"
    assert not branch_exists(runner.deps.cwd, "salvage/demo-i1/I1-T1")
    assert working_tree_clean(runner.deps.cwd)
    assert rev.calls == []


def test_impl_failure_classifier_detects_auth_errors():
    assert runner_mod._classify_impl_failure(
        "Authentication failed: gh auth status required", 1
    ) == "auth"


def test_fix_round_out_of_scope_file_fails_task_before_review(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A fix round that edits a file outside allowed_files must STOP:SCOPE."""

    def fake_run_acceptance(*args, **kwargs):
        return SimpleNamespace(
            ok=False,
            combined_output=lambda: "acceptance failing",
        )

    monkeypatch.setitem(
        IterationRunner._checks_with_fix_loop.__globals__,
        "run_acceptance",
        fake_run_acceptance,
    )

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/outside.py", "x = 1\n"),
        ],
    )
    rev = FakeAdapter(name="codex", family="openai", script=[])
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.stop_reason == "SCOPE"
    review_events = [e for e in store.events if e.get("kind") == "review"]
    assert not review_events
    assert rev.calls == []


def test_diff_cap_default_blocks_large_task_without_override(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        runner_mod,
        "diff_stats",
        _fake_large_diff_for_targets("src/a.py"),
    )
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(name="codex", family="openai", script=[])
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    assert limits(runner.cfg)["max_diff_insertions_hard"] == 1500
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.stop_reason == "STRUCTURAL"
    assert "diff insertions 1600 over hard cap 1500" in t1.stop_msg
    assert rev.calls == []


def test_task_diff_cap_override_is_scoped_to_one_task(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    _add_t1_diff_cap_override(repo)
    monkeypatch.setattr(
        runner_mod,
        "diff_stats",
        _fake_large_diff_for_targets("src/a.py", "src/b.py"),
    )
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    assert limits(runner.cfg)["max_diff_insertions_hard"] == 1500
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status.startswith(STATUS_STOPPED_PREFIX)
    assert store.tasks["I1-T2"].stop_reason == "STRUCTURAL"
    assert "diff insertions 1600 over hard cap 1500" in (
        store.tasks["I1-T2"].stop_msg
    )
    override_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "diff_cap_override_applied"
    ]
    assert len(override_events) == 1
    assert override_events[0]["task"] == "I1-T1"
    assert override_events[0]["meta"]["scope"] == "I1-T1"
    assert override_events[0]["meta"]["max_diff_insertions_hard"] == 1800
    assert override_events[0]["meta"]["approved_by"] == "operator"
    assert override_events[0]["meta"]["evidence"] == (
        "iterations/demo-i1/reviews/review-t1.md"
    )
    assert len(rev.calls) == 1


def test_task_review_loads_authored_review_prompt(repo: Path):
    _add_review_prompt(
        repo,
        "review-t1.md",
        "# Review T1\n\nCUSTOM REVIEW CONTRACT FOR T1\n\nVerdict: PASS\n",
    )
    _add_review_prompt(
        repo,
        "review-t2.md",
        "# Review T2\n\nCUSTOM REVIEW CONTRACT FOR T2\n\nVerdict: PASS\n",
    )
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, _, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    prompts = [call["prompt"] for call in rev.calls]
    assert any("CUSTOM REVIEW CONTRACT FOR T1" in p for p in prompts)
    assert any("CUSTOM REVIEW CONTRACT FOR T2" in p for p in prompts)
    assert all("## Fresh Diff" in p for p in prompts)


def test_task_review_fallback_includes_verdict_contract(repo: Path):
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, _, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 0

    prompts = [call["prompt"] for call in rev.calls]
    assert all("Runtime Fallback Review Contract" in p for p in prompts)
    assert all("Verdict: CHANGES REQUIRED" in p for p in prompts)
    assert all("Severity: should-fix" in p for p in prompts)


def test_task_review_fails_closed_when_scaffolded_review_missing(repo: Path):
    reviews_dir = repo / "iterations" / "demo-i1" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    (reviews_dir / "review-unrelated.md").write_text("unused\n")
    stage_all(repo)
    commit(repo, "test: add unrelated review prompt")
    git(["branch", "-f", "phase-demo", "HEAD"], cwd=repo, check=True)
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=repo, check=True)

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    assert runner.run() == 1
    assert store.tasks["I1-T1"].status.startswith(STATUS_STOPPED_PREFIX)
    assert store.tasks["I1-T1"].stop_reason == "STRUCTURAL"
    assert "review prompt missing" in store.tasks["I1-T1"].stop_msg
    assert rev.calls == []


def test_partial_review_with_pass_verdict_line_stops_malformed(repo: Path):
    """A timed-out reviewer with Verdict: PASS must STOP as REVIEW_MALFORMED."""

    def _partial_review(adapter, prompt, workdir):
        return AgentResult(
            exit_code=0,
            stdout="Analysis looks good.\n\nVerdict: PASS\n",
            stderr="",
            duration_s=10.0,
            input_tokens=30,
            output_tokens=10,
            tokens_exact=False,
            partial=True,
        )

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_partial_review],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.stop_reason == "REVIEW_MALFORMED"
    assert (
        "tools/logs/demo-i1/reviews/review_I1-T1_r1.md"
        in (t1.stop_msg or "")
    )
    review_events = [
        e for e in store.events
        if e.get("kind") == "review"
    ]
    assert review_events, "expected at least one review event"
    assert review_events[-1].get("meta", {}).get("verdict") == "MALFORMED"


def test_branch_freshness_gate_stops_stale_iteration_before_impl(repo: Path):
    """If the phase base advances, the next task refuses before implementer."""
    checkout(repo, "phase-demo")
    (repo / "phase-advance.txt").write_text("phase\n")
    stage_all(repo)
    commit(repo, "phase advances")
    checkout(repo, "main")

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.stop_reason == "BRANCH_FRESHNESS"
    assert "demo/iteration-1" in t1.stop_msg
    assert "phase-demo" in t1.stop_msg
    assert "Next action:" in t1.stop_msg
    assert impl.calls == []
    assert rev.calls == []


def test_skip_impl_preserves_existing_task_branch(repo: Path):
    create_or_reset_branch(repo, "d/i1/t1", "demo/iteration-1")
    (repo / "src" / "a.py").parent.mkdir(parents=True, exist_ok=True)
    (repo / "src" / "a.py").write_text("def a():\n    return 42\n")
    stage_all(repo)
    commit(repo, "manual implementation for skip-impl")
    manual_sha = current_sha(repo)
    checkout(repo, "main")

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/b.py", "def b():\n    return 2\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Verdict: PASS\n"),
            _reviewer_verdict("Verdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        skip_impl_tasks=["I1-T1"],
    )

    rc = runner.run()

    assert rc == 0
    assert len(impl.calls) == 1
    assert "Task I1-T2" in impl.calls[0]["prompt"]
    assert len(rev.calls) == 2
    prompt = rev.calls[0]["prompt"]
    assert "## Fresh Diff" in prompt
    assert "+def a():" in prompt
    assert "+    return 42" in prompt
    assert current_sha(repo, "d/i1/t1") == manual_sha
    assert store.tasks["I1-T1"].status == STATUS_DONE
    preserved_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "skip_impl_branch_preserved"
    ]
    assert len(preserved_events) == 1
    assert preserved_events[0]["meta"] == {
        "event": "skip_impl_branch_preserved",
        "branch": "d/i1/t1",
        "sha": manual_sha,
    }


def test_skip_impl_missing_branch_stops_branch_freshness(repo: Path):
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        skip_impl_tasks=["I1-T1"],
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "BRANCH_FRESHNESS"
    assert t1.stop_reason == "BRANCH_FRESHNESS"
    gate_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "branch_freshness_gate"
    ]
    assert gate_events[-1]["meta"]["gate"] == "skip-impl"
    assert gate_events[-1]["meta"]["condition"] == "missing_branch"
    assert impl.calls == []
    assert rev.calls == []


def test_skip_impl_stale_branch_stops_branch_freshness(repo: Path):
    git(["branch", "d/i1/t1", "demo/iteration-1"], cwd=repo, check=True)
    checkout(repo, "demo/iteration-1")
    (repo / "src" / "base.py").parent.mkdir(parents=True, exist_ok=True)
    (repo / "src" / "base.py").write_text("BASE = True\n")
    stage_all(repo)
    commit(repo, "iteration advances")
    checkout(repo, "main")

    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
        skip_impl_tasks=["I1-T1"],
    )

    rc = runner.run()

    assert rc == 1
    t1 = store.tasks["I1-T1"]
    assert t1.status == STATUS_STOPPED_PREFIX + "BRANCH_FRESHNESS"
    assert t1.stop_reason == "BRANCH_FRESHNESS"
    gate_events = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "branch_freshness_gate"
    ]
    assert gate_events[-1]["meta"]["gate"] == "skip-impl"
    assert gate_events[-1]["meta"]["condition"] == "behind"
    assert impl.calls == []
    assert rev.calls == []


def test_branch_freshness_gate_stops_stale_task_before_pr(
    repo: Path, monkeypatch: pytest.MonkeyPatch,
):
    """A task branch missing current iteration commits never opens a PR."""
    git(["branch", "old-base", "demo/iteration-1"], cwd=repo, check=True)
    checkout(repo, "demo/iteration-1")
    (repo / "src" / "base.py").parent.mkdir(parents=True, exist_ok=True)
    (repo / "src" / "base.py").write_text("BASE = True\n")
    stage_all(repo)
    commit(repo, "iteration advances")
    checkout(repo, "main")

    original_create_or_reset = runner_mod.create_or_reset_branch

    def create_stale_task_branch(cwd: Path, new_branch: str, base: str):
        return original_create_or_reset(cwd, new_branch, "old-base")

    monkeypatch.setattr(
        runner_mod, "create_or_reset_branch", create_stale_task_branch
    )

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("src/a.py", "def a():\n    return 1\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Verdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=False,
        implementer="claude", reviewer="codex",
    )
    calls = SimpleNamespace(open_pr=[])

    def fake_open_pr(*, cwd, title, body, base, head):
        calls.open_pr.append((title, base, head))
        return True, "https://example.com/pr/1"

    runner.deps.open_pr = fake_open_pr

    rc = runner.run()

    assert rc == 1
    assert calls.open_pr == []
    t1 = store.tasks["I1-T1"]
    assert t1.status.startswith(STATUS_STOPPED_PREFIX)
    assert t1.stop_reason == "BRANCH_FRESHNESS"
    assert "d/i1/t1" in t1.stop_msg
    assert "demo/iteration-1" in t1.stop_msg
    assert "diverged" in t1.stop_msg


# ---------------------------------------------------------------------------
# Close-out objective 6 — runner-level fail-closed on a base ref that does not
# yield a meaningful diff (regression guard for the gate short-circuit).
# ---------------------------------------------------------------------------


def test_final_scope_gate_fails_closed_on_missing_base_ref(repo: Path):
    """A resolved-but-non-existent diff base must fail closed (not crash via
    diff_files check=True, not pass vacuously)."""
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    # The resolved phase base (phase-demo) no longer exists in git.
    git(["branch", "-D", "phase-demo"], cwd=repo, check=True)

    msg = runner._final_scope_gate()

    assert msg is not None
    assert "could not resolve" in msg
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "final_scope_gate_failed"
    ]
    assert failed, "expected a final_scope_gate_failed note event"


def test_final_scope_gate_fails_closed_on_vacuous_post_merge_base(repo: Path):
    """When the base already contains the iteration branch (post-merge shape),
    base...iter is empty; the gate must fail closed rather than pass on an
    empty changed-file set."""
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    # Advance phase-demo past demo/iteration-1 so the iteration is BEHIND its
    # base (ahead_count == 0) — the post-merge vacuous case.
    checkout(repo, "phase-demo")
    (repo / "later_on_phase.txt").write_text("later\n")
    stage_all(repo)
    commit(repo, "commit on phase after iteration")

    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    msg = runner._final_scope_gate()

    assert msg is not None
    assert "vacuous" in msg
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event") == "final_scope_gate_failed"
    ]
    assert failed, "expected a final_scope_gate_failed note event"


def test_final_nav_gate_fails_closed_on_missing_base_ref(repo: Path):
    """Final nav-discoverability gate shares the obj-6 freshness guard with the
    scope gate, so a resolved-but-non-existent diff base must fail closed there
    too (regression guard for the nav side of the gate)."""
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )
    # The resolved phase base (phase-demo) no longer exists in git.
    git(["branch", "-D", "phase-demo"], cwd=repo, check=True)

    msg = runner._final_nav_discoverability_gate()

    assert msg is not None
    assert "could not resolve" in msg
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_failed"
    ]
    assert failed, "expected a final_nav_discoverability_gate_failed note event"


def test_final_nav_gate_fails_closed_on_vacuous_post_merge_base(repo: Path):
    """When the base already contains the iteration branch (post-merge shape),
    base...iter is vacuous; the nav gate must fail closed rather than review an
    empty diff — mirroring the scope gate's obj-6 behavior."""
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    # Advance phase-demo past demo/iteration-1 so the iteration is no longer
    # strictly ahead of its base (ahead_count == 0) — the post-merge vacuous
    # case, identical to the scope-gate fixture above.
    checkout(repo, "phase-demo")
    (repo / "later_on_phase.txt").write_text("later\n")
    stage_all(repo)
    commit(repo, "commit on phase after iteration")

    runner, store, _ = _make_runner(
        repo, {"claude": impl, "codex": rev}, dry_run=True,
        implementer="claude", reviewer="codex",
    )

    msg = runner._final_nav_discoverability_gate()

    assert msg is not None
    assert "vacuous" in msg
    failed = [
        e for e in store.events
        if e.get("kind") == "note"
        and e.get("meta", {}).get("event")
        == "final_nav_discoverability_gate_failed"
    ]
    assert failed, "expected a final_nav_discoverability_gate_failed note event"


def _planning_tasks_md(
    *,
    allowed_1: str = "docs/plan-a.md",
    allowed_2: str | None = None,
) -> str:
    rows = [
        "| I1-T1 | First     | TBD   | WAITING | \u2014     | d/i1/t1     |",
    ]
    details = [
        "### I1-T1 \u2014 First\n\n"
        "**Allowed files:**\n"
        "```\n"
        f"{allowed_1}\n"
        "```\n",
    ]
    if allowed_2 is not None:
        rows.append(
            "| I1-T2 | Second    | TBD   | WAITING | \u2014     | d/i1/t2     |"
        )
        details.append(
            "### I1-T2 \u2014 Second\n\n"
            "**Allowed files:**\n"
            "```\n"
            f"{allowed_2}\n"
            "```\n"
        )
    return (
        "# Demo iteration\n"
        "## Task Board\n\n"
        "**Status:** WAITING\n"
        "**Iteration branch:** `demo/iteration-1`\n"
        "**Depends on:** none\n"
        "**Blocks:** none\n\n"
        "---\n\n"
        "## Execution Plan\n"
        "- approach: task_by_task\n"
        "- qa: standard\n"
        "- note: planning\n\n"
        "---\n\n"
        "## Tasks\n\n"
        "| ID    | Title     | Owner | Status  | Depends on | Branch      |\n"
        "|-------|-----------|-------|---------|------------|-------------|\n"
        + "\n".join(rows)
        + "\n\n---\n\n"
        "## Task Details\n\n"
        + "\n\n".join(details)
        + "\n"
    )


def test_planning_team_default_off_uses_existing_serial_runner(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _update_tasks_md(repo, _planning_tasks_md())
    monkeypatch.setattr(
        runner_mod,
        "run_planning_team",
        lambda **kwargs: pytest.fail("default-off run must not use team mode"),
    )
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[_edit_file_on_invoke("docs/plan-a.md", "serial plan\n")],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )

    assert runner.run() == 0

    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert not [
        event for event in store.events
        if (event.get("meta") or {}).get("event") == "planning_team_spawn"
    ]


def test_planning_team_mode_runs_guarded_team_implementation(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from orch.planning_team import PlanningTeamResult

    _update_tasks_md(repo, _planning_tasks_md())
    calls = []

    def fake_run_planning_team(**kwargs):
        calls.append(kwargs)
        [candidate] = kwargs["candidates"]
        for rel in candidate.allowed_files:
            path = kwargs["cwd"] / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("team plan\n", encoding="utf-8")
        return [
            PlanningTeamResult(
                task_id=candidate.task_id,
                ok=True,
                status="completed",
                text="team plan complete",
                artifact_dir=candidate.artifact_dir,
                changed_files=tuple(candidate.allowed_files),
                exit_code=0,
                duration_s=0.2,
                input_tokens=10,
                output_tokens=5,
                tokens_exact=True,
                provider="claude",
                model="claude-planning-model",
                cached_input_tokens=3,
                cache_creation_input_tokens=4,
                parser_status="parsed",
                raw_terminal_json="{}",
            )
        ]

    monkeypatch.setattr(runner_mod, "run_planning_team", fake_run_planning_team)
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[_reviewer_verdict("Looks good.\nVerdict: PASS\n")],
    )
    runner, store, _ = _make_runner(
        repo,
        {
            "claude": FakeAdapter(name="claude", family="anthropic"),
            "codex": rev,
        },
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
        team_mode="planning",
    )

    assert runner.run() == 0

    assert len(calls) == 1
    assert calls[0]["command"] == (
        "claude",
        "-p",
        "--output-format",
        "json",
    )
    assert calls[0]["team_name"] == "planning-demo-i1-I1-T1"
    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert [
        event for event in store.events
        if (event.get("meta") or {}).get("event") == "planning_team_spawn"
    ]
    cost_records = [
        json.loads(line)
        for line in (
            repo / "tools/logs/demo-i1/cost.jsonl"
        ).read_text().splitlines()
    ]
    impl_records = [record for record in cost_records if record["step"] == "IMPL"]
    assert impl_records[0]["extra"]["team_mode"] == "planning"
    assert impl_records[0]["extra"]["cost_estimate"] is False
    assert impl_records[0]["estimated"] is False
    assert impl_records[0]["provider"] == "claude"
    assert impl_records[0]["model"] == "claude-planning-model"
    assert impl_records[0]["cached_input_tokens"] == 3
    assert impl_records[0]["cache_creation_input_tokens"] == 4
    assert impl_records[0]["parser_status"] == "parsed"
    # Raw CLI dump is stripped from the persisted record (top-level key).
    assert "raw_terminal_json" not in impl_records[0]["extra"]
    assert (
        impl_records[0]["extra"]["planning_review_independence_policy"]
        == "cross_vendor"
    )


def test_planning_team_mode_refuses_runtime_allowed_file(repo: Path):
    _update_tasks_md(repo, _planning_tasks_md(allowed_1="app/main.py"))
    runner, store, _ = _make_runner(
        repo,
        {
            "claude": FakeAdapter(name="claude", family="anthropic"),
            "codex": FakeAdapter(name="codex", family="openai"),
        },
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
        team_mode="planning",
    )

    assert runner.run() == 1

    stop_events = [
        event for event in store.events
        if (event.get("meta") or {}).get("event") == "stop_global"
    ]
    assert stop_events
    assert stop_events[-1]["meta"]["reason"] == "CONFIG"
    assert "app/main.py" in stop_events[-1]["meta"]["msg"]


def test_planning_team_mode_serializes_overlapping_allowed_files(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _update_tasks_md(
        repo,
        _planning_tasks_md(
            allowed_1="docs/shared.md",
            allowed_2="docs/shared.md",
        ),
    )
    monkeypatch.setattr(
        runner_mod,
        "run_planning_team",
        lambda **kwargs: pytest.fail("overlap must use non-team serial path"),
    )
    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _edit_file_on_invoke("docs/shared.md", "first serial plan\n"),
            _edit_file_on_invoke("docs/shared.md", "second serial plan\n"),
        ],
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
        team_mode="planning",
    )

    assert runner.run() == 0

    assert store.tasks["I1-T1"].status == STATUS_DONE
    assert store.tasks["I1-T2"].status == STATUS_DONE
    assert [
        event for event in store.events
        if (event.get("meta") or {}).get("event") == "planning_team_serialized"
    ]
    assert not [
        event for event in store.events
        if (event.get("meta") or {}).get("event") == "planning_team_spawn"
    ]


def test_planning_team_policy_blocks_same_vendor_even_with_session_independence(
    repo: Path,
):
    _update_tasks_md(repo, _planning_tasks_md())
    runner, store, _ = _make_runner(
        repo,
        {
            "claude": FakeAdapter(name="claude", family="anthropic"),
            "codex": FakeAdapter(name="codex", family="anthropic"),
        },
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
        independence="session",
        team_mode="planning",
    )

    assert runner.run() == 1

    stop_events = [
        event for event in store.events
        if (event.get("meta") or {}).get("event") == "stop_global"
    ]
    assert stop_events[-1]["meta"]["reason"] == "INDEPENDENCE"
    assert stop_events[-1]["meta"]["policy"] == "cross_vendor"


def test_project_routing_sugar_flows_from_task_declaration_to_invocation_options(
    repo: Path,
):
    _add_t1_model_routing(repo, "routing_config")
    _append_project_config(
        repo,
        """
model_routing:
  agent_overrides:
    claude:
      model_flag: "--model"
      tier_models:
        standard: "claude-haiku-4-5"
        strong: "claude-sonnet-4-6"
        max: "claude-opus-4-8"
    codex:
      model_flag: "-m"
      tier_models:
        standard: "gpt-5.5"
        strong: "gpt-5.5"
        max: "gpt-5.5"
      effort_flags:
        low:
          args: ["-c", "model_reasoning_effort=low"]
        medium:
          args: ["-c", "model_reasoning_effort=medium"]
        high:
          args: ["-c", "model_reasoning_effort=high"]
        max:
          args: ["-c", "model_reasoning_effort=xhigh"]
""",
    )
    impl = FakeAdapter(
        name="codex",
        family="openai",
        script=[
            _edit_file_on_invoke("src/a.py", "def a():\n    return 1\n"),
            _edit_file_on_invoke("src/b.py", "def b():\n    return 2\n"),
        ],
    )
    rev = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
            _reviewer_verdict("Looks good.\nVerdict: PASS\n"),
        ],
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": rev, "codex": impl},
        dry_run=True,
        implementer="codex",
        reviewer="claude",
    )

    assert runner.run() == 0

    t1_routing_events = [
        event for event in store.events
        if event.get("task") == "I1-T1"
        and (event.get("meta") or {}).get("event") == "model_routing_resolved"
    ]
    assert t1_routing_events
    assert t1_routing_events[-1]["meta"]["risk_category"] == "routing_config"
    assert t1_routing_events[-1]["meta"]["model_tier"] == "max"
    assert t1_routing_events[-1]["meta"]["reasoning_effort"] == "max"
    assert impl.calls[0]["routing_options"].args == (
        "-m",
        "gpt-5.5",
        "-c",
        "model_reasoning_effort=xhigh",
    )
    assert impl.calls[0]["routing_options"].env == {}
    assert rev.calls[0]["routing_options"].args == (
        "--model",
        "claude-opus-4-8",
    )
    assert rev.calls[0]["routing_options"].env == {}
