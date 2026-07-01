"""Tests for orch.cli subcommands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from orch import cli as cli_mod
from orch.cli import _build_runner, _run_scaffold_lint, main
from orch.git_ops import (
    WorktreePreflightError,
    branch_exists,
    commit,
    ensure_orch_workdir,
    git,
    orch_workdir,
    stage_all,
)
from orch.improvements import append_record
from orch.lifecycle import PhaseResolutionError
from orch.locks import RUN_LOCK_FILENAME, RunLockInfo, RunStateLock
from orch.state import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_HUMAN_MERGE,
    STATUS_STOPPED_PREFIX,
    StateStore,
)


VALID_PROJECT_YAML = """\
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
  high_risk_globs: ["**/schema.sql"]
  sensitive_files: [".env"]
  forbidden_patterns: ["except: pass"]
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


VALID_TASKS_MD = """\
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
- note: implementer and reviewer chosen at runtime

---

## Tasks

| ID    | Title    | Owner | Status  | Depends on | Branch             |
|-------|----------|-------|---------|------------|--------------------|
| I4-T1 | Do thing | TBD   | WAITING | \u2014     | `demo/i1/t1-thing`  |

---

## Task Details

### I4-T1 \u2014 Do thing

**Allowed files:**
```
app/thing.py
```

**Done when:** it works.
"""


def _completed_review_report():
    return SimpleNamespace(roles=[SimpleNamespace(role="security", ok=True)])


def _incomplete_review_report(role: str, *, timed_out: bool = False):
    return SimpleNamespace(
        roles=[SimpleNamespace(role=role, ok=False, timed_out=timed_out)]
    )


def _write_iteration_scaffold(
    iter_dir: Path,
    *,
    tasks_text: str = VALID_TASKS_MD,
    prompt_text: str = "# Test iteration\n\n",
) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "tasks.md").write_text(tasks_text)
    (iter_dir / "prompt.md").write_text(prompt_text)
    prompts = iter_dir / "prompts"
    reviews = iter_dir / "reviews"
    prompts.mkdir(exist_ok=True)
    reviews.mkdir(exist_ok=True)
    (prompts / "t1-do-thing.md").write_text("# T1\n\n")
    (reviews / "review-t1.md").write_text("# Review T1\n\n")


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(VALID_PROJECT_YAML)
    iter_dir = tmp_path / "iterations" / "phase-demo" / "demo-i1"
    _write_iteration_scaffold(iter_dir)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_validate_ok(workdir: Path, capsys: pytest.CaptureFixture):
    assert main(["validate", "demo-i1"]) == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out
    assert "demo/iteration-1" in captured.out


def test_phase_branch_resolution_is_config_driven_fail_closed(workdir: Path):
    cfg = cli_mod.load_config(workdir / ".orch" / "project.yaml")

    assert cli_mod._resolve_phase_branch(cfg, "demo-i1") == "phase-demo"
    with pytest.raises(PhaseResolutionError, match="cannot resolve phase"):
        cli_mod._resolve_phase_branch(cfg, "not-an-iteration")


def test_validate_unknown_iter(workdir: Path, capsys: pytest.CaptureFixture):
    assert main(["validate", "missing-iteration"]) == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_iter_dir_multi_match_is_clear_error(
    workdir: Path,
    capsys: pytest.CaptureFixture,
):
    other = workdir / "iterations" / "other-phase" / "demo-i1"
    other.mkdir(parents=True)
    (other / "tasks.md").write_text(VALID_TASKS_MD)

    assert main(["validate", "demo-i1"]) == 1

    err = capsys.readouterr().err
    assert "ambiguous" in err
    assert "phase-demo/demo-i1" in err
    assert "other-phase/demo-i1" in err


def test_cli_uses_configured_iteration_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(
        VALID_PROJECT_YAML
        + "\npaths:\n"
        + "  iteration_root: custom-iterations\n"
    )
    iter_dir = tmp_path / "custom-iterations" / "phase-demo" / "demo-i1"
    _write_iteration_scaffold(iter_dir)
    monkeypatch.chdir(tmp_path)

    assert main(["validate", "demo-i1"]) == 0
    out = capsys.readouterr().out
    assert "custom-iterations/phase-demo/demo-i1/tasks.md" in out


def test_status_with_no_state(workdir: Path, capsys: pytest.CaptureFixture):
    assert main(["status", "demo-i1"]) == 0
    out = capsys.readouterr().out
    assert "iteration:   demo-i1" in out
    assert "demo/iteration-1" in out
    assert "tasks on board: 1" in out


def test_status_reflects_persisted_events(
    workdir: Path, capsys: pytest.CaptureFixture
):
    log_dir = workdir / "tools" / "logs" / "demo-i1"
    log_dir.mkdir(parents=True)
    s = StateStore(log_dir=log_dir, iteration="demo-i1", iter_branch="demo/iteration-1")
    s.mark_iteration_started()
    s.task_transition("I4-T1", STATUS_DONE)

    assert main(["status", "demo-i1"]) == 0
    out = capsys.readouterr().out
    assert "DONE               1" in out


def _recover_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> Path:
    repo = tmp_path / name
    repo.mkdir()
    git(["init", "-q", "-b", "main"], cwd=repo, check=True)
    git(["config", "user.email", "t@e.st"], cwd=repo, check=True)
    git(["config", "user.name", "Tester"], cwd=repo, check=True)
    (repo / ".orch").mkdir()
    (repo / ".orch" / "project.yaml").write_text(VALID_PROJECT_YAML)
    iter_dir = repo / "iterations" / "phase-demo" / "demo-i1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tasks.md").write_text(VALID_TASKS_MD)
    (repo / "README.md").write_text("demo\n")
    stage_all(repo)
    commit(repo, "init recover repo")
    git(["branch", "demo/iteration-1"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    return repo


def _recover_store(repo: Path) -> StateStore:
    return StateStore(
        log_dir=repo / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
    )


def _write_recover_lock(repo: Path, status: str) -> Path:
    if status == "active":
        pid = os.getpid()
        hostname = socket.gethostname()
    elif status == "unknown":
        pid = os.getpid()
        hostname = "other-host"
    else:
        pid = 999999
        hostname = socket.gethostname()
    log_dir = repo / "tools" / "logs" / "demo-i1"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / RUN_LOCK_FILENAME
    info = RunLockInfo(
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        command="run",
        pid=pid,
        hostname=hostname,
        repo_root=str(repo),
        orch_workdir=str(orch_workdir(repo, "demo-i1")),
        created_at="2026-06-11T00:00:00Z",
        token="stale-token",
    )
    path.write_text(json.dumps(info.to_dict()))
    return path


def test_recover_dry_run_reports_lock_workdir_and_in_progress_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    repo = _recover_repo(tmp_path, monkeypatch, "dry-run")
    store = _recover_store(repo)
    store.task_transition("I4-T1", STATUS_IN_PROGRESS)
    lock_path = _write_recover_lock(repo, "stale")
    worktree = ensure_orch_workdir(repo, "demo-i1", "demo/iteration-1")
    (worktree / "leftover.txt").write_text("interrupted\n")

    assert main(["recover", "demo-i1"]) == 0

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert str(lock_path) in out
    assert "pid_status=stale" in out
    assert str(worktree) in out
    assert "dirty" in out
    assert "salvage/demo-i1/recover" in out
    assert "I4-T1 -> WAITING" in out
    assert "python -m orch resume demo-i1" in out
    assert lock_path.exists()
    assert worktree.exists()
    assert (worktree / "leftover.txt").exists()
    store.load()
    assert store.tasks["I4-T1"].status == STATUS_IN_PROGRESS


def test_recover_apply_removes_stale_lock_cleans_workdir_and_resets_in_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = _recover_repo(tmp_path, monkeypatch, "dirty-apply")
    store = _recover_store(repo)
    store.task_transition("I4-T1", STATUS_IN_PROGRESS)
    lock_path = _write_recover_lock(repo, "stale")
    worktree = ensure_orch_workdir(repo, "demo-i1", "demo/iteration-1")
    (worktree / "leftover.txt").write_text("interrupted\n")

    assert main(["recover", "demo-i1", "--apply"]) == 0

    assert not lock_path.exists()
    assert not worktree.exists()
    assert branch_exists(repo, "salvage/demo-i1/recover")
    assert git(
        ["show", "salvage/demo-i1/recover:leftover.txt"],
        cwd=repo,
        check=True,
    ).stdout == "interrupted\n"
    store.load()
    assert store.tasks["I4-T1"].status == "WAITING"
    event_names = [event.get("meta", {}).get("event") for event in store.events]
    assert "recover_lock_removed" in event_names
    assert "recover_salvage" in event_names
    assert "recover_workdir_cleaned" in event_names
    assert "recover_task_reset" in event_names
    assert event_names.index("recover_salvage") < event_names.index(
        "recover_workdir_cleaned"
    )

    clean_repo = _recover_repo(tmp_path, monkeypatch, "clean-apply")
    clean_store = _recover_store(clean_repo)
    clean_store.task_transition("I4-T1", STATUS_IN_PROGRESS)
    _write_recover_lock(clean_repo, "stale")
    clean_worktree = ensure_orch_workdir(
        clean_repo, "demo-i1", "demo/iteration-1"
    )

    assert main(["recover", "demo-i1", "--apply"]) == 0

    assert not clean_worktree.exists()
    assert not branch_exists(clean_repo, "salvage/demo-i1/recover")
    clean_store.load()
    clean_events = [
        event.get("meta", {}).get("event") for event in clean_store.events
    ]
    assert "recover_salvage" not in clean_events
    assert "recover_workdir_cleaned" in clean_events


@pytest.mark.parametrize("status", ["active", "unknown"])
def test_recover_refuses_active_or_unknown_lock_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    status: str,
):
    repo = _recover_repo(tmp_path, monkeypatch, f"{status}-lock")
    store = _recover_store(repo)
    store.task_transition("I4-T1", STATUS_IN_PROGRESS)
    lock_path = _write_recover_lock(repo, status)

    assert main(["recover", "demo-i1", "--apply"]) == 1

    err = capsys.readouterr().err
    assert status in err
    assert "--force-lock" in err
    assert lock_path.exists()
    store.load()
    assert store.tasks["I4-T1"].status == STATUS_IN_PROGRESS


def test_recover_force_lock_records_forced_lock_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = _recover_repo(tmp_path, monkeypatch, "force-lock")
    store = _recover_store(repo)
    lock_path = _write_recover_lock(repo, "unknown")

    assert main(["recover", "demo-i1", "--apply", "--force-lock"]) == 0

    assert not lock_path.exists()
    store.load()
    lock_events = [
        event for event in store.events
        if event.get("meta", {}).get("event") == "recover_lock_removed"
    ]
    assert lock_events
    assert lock_events[-1]["meta"]["forced"] is True
    assert lock_events[-1]["meta"]["pid_status"] == "unknown"


def test_qa_and_retro_use_configured_timeouts(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (workdir / ".orch" / "project.yaml").write_text(
        VALID_PROJECT_YAML + "\ntimeouts:\n  qa: 321\n  retro: 654\n"
    )
    calls = []

    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": object(), "codex": object()},
    )
    monkeypatch.setattr(cli_mod, "cmd_run", lambda args: 0)

    def fake_run_qa(**kwargs):
        calls.append(("qa", kwargs["timeout"]))
        return _completed_review_report()

    def fake_run_retro(**kwargs):
        calls.append(("retro", kwargs["timeout"]))
        return _completed_review_report()

    monkeypatch.setattr(cli_mod, "run_qa", fake_run_qa)
    monkeypatch.setattr(cli_mod, "run_retro", fake_run_retro)

    assert main(["qa", "demo-i1"]) == 0
    assert main(["retro", "demo-i1"]) == 0
    assert main(["iteration", "demo-i1", "--dry-run"]) == 0

    assert calls == [
        ("qa", 321),
        ("retro", 654),
        ("qa", 321),
        ("retro", 654),
    ]


def test_qa_partial_failure_fails_closed_unless_allowed(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )
    monkeypatch.setattr(
        cli_mod,
        "run_qa",
        lambda **kwargs: _incomplete_review_report(
            "security", timed_out=True
        ),
    )

    assert main(["qa", "demo-i1"]) == 1
    err = capsys.readouterr().err
    assert "QA incomplete" in err
    assert "security" in err

    assert main(["qa", "demo-i1", "--allow-partial"]) == 0


def test_retro_partial_failure_fails_closed_unless_allowed(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )
    monkeypatch.setattr(
        cli_mod,
        "run_retro",
        lambda **kwargs: _incomplete_review_report("developer"),
    )

    assert main(["retro", "demo-i1"]) == 1
    err = capsys.readouterr().err
    assert "Retro incomplete" in err
    assert "developer" in err

    assert main(["retro", "demo-i1", "--allow-partial"]) == 0


def test_qa_empty_diff_fails_closed_unless_reason_is_provided(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    reasons = []
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )

    def fake_run_qa(**kwargs):
        reason = kwargs["allow_empty_diff_reason"]
        reasons.append(reason)
        if reason is None:
            raise cli_mod.QaEmptyDiffError("empty/vacuous diff")
        return _completed_review_report()

    monkeypatch.setattr(cli_mod, "run_qa", fake_run_qa)

    assert main(["qa", "demo-i1"]) == 1
    assert "empty/vacuous diff" in capsys.readouterr().err

    assert main(
        [
            "qa",
            "demo-i1",
            "--allow-empty-diff",
            "post-merge audit",
        ]
    ) == 0
    assert reasons == [None, "post-merge audit"]


def test_iteration_threads_review_gate_overrides(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(cli_mod, "cmd_run", lambda args: 0)
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )

    def fake_run_qa(**kwargs):
        calls.append(
            (
                "qa",
                kwargs["allow_empty_diff_reason"],
            )
        )
        return _incomplete_review_report("security")

    def fake_run_retro(**kwargs):
        calls.append(("retro", None))
        return _incomplete_review_report("developer")

    monkeypatch.setattr(cli_mod, "run_qa", fake_run_qa)
    monkeypatch.setattr(cli_mod, "run_retro", fake_run_retro)

    assert main(
        [
            "iteration",
            "demo-i1",
            "--dry-run",
            "--allow-partial",
            "--allow-empty-diff",
            "post-merge audit",
        ]
    ) == 0
    assert calls == [("qa", "post-merge audit"), ("retro", None)]


def test_qa_command_respects_iteration_lock(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    lock = RunStateLock(
        log_dir=workdir / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        command="run",
        repo_root=workdir,
        orch_workdir=orch_workdir(workdir, "demo-i1"),
    ).acquire()
    monkeypatch.setattr(
        cli_mod,
        "run_qa",
        lambda **kwargs: pytest.fail("run_qa should not run under lock"),
    )

    try:
        assert main(["qa", "demo-i1"]) == 1
    finally:
        lock.release()

    err = capsys.readouterr().err
    assert "active orch run lock" in err
    assert "command='run'" in err


def test_retro_command_respects_iteration_lock(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    lock = RunStateLock(
        log_dir=workdir / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        command="qa",
        repo_root=workdir,
        orch_workdir=orch_workdir(workdir, "demo-i1"),
    ).acquire()
    monkeypatch.setattr(
        cli_mod,
        "run_retro",
        lambda **kwargs: pytest.fail("run_retro should not run under lock"),
    )

    try:
        assert main(["retro", "demo-i1"]) == 1
    finally:
        lock.release()

    err = capsys.readouterr().err
    assert "active orch run lock" in err
    assert "command='qa'" in err


def test_iteration_chain_passes_retro_agent_flag(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(cli_mod, "cmd_run", lambda args: 0)
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": object(), "codex": object()},
    )

    def fake_run_retro(**kwargs):
        calls.append(kwargs["agent_name"])
        return _completed_review_report()

    monkeypatch.setattr(cli_mod, "run_retro", fake_run_retro)

    assert main(
        [
            "iteration",
            "demo-i1",
            "--dry-run",
            "--skip-qa",
            "--retro-agent",
            "codex",
        ]
    ) == 0

    assert calls == ["codex"]


def test_retro_parser_accepts_team_mode_but_iteration_does_not():
    parser = cli_mod.build_parser()

    retro_args = parser.parse_args(["retro", "demo-i1", "--team-mode"])
    assert retro_args.team_mode is True
    assert parser.parse_args(["retro", "demo-i1"]).team_mode is False

    with pytest.raises(SystemExit):
        parser.parse_args(["iteration", "demo-i1", "--team-mode"])


def test_qa_parser_accepts_team_mode_but_iteration_does_not():
    parser = cli_mod.build_parser()

    qa_args = parser.parse_args(["qa", "demo-i1", "--team-mode"])
    assert qa_args.team_mode is True
    assert parser.parse_args(["qa", "demo-i1"]).team_mode is False

    with pytest.raises(SystemExit):
        parser.parse_args(["iteration", "demo-i1", "--team-mode"])


def test_retro_team_mode_dispatches_to_team_runner_only(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )
    monkeypatch.setattr(
        cli_mod,
        "run_retro",
        lambda **kwargs: pytest.fail("serial run_retro should not run"),
    )

    def fake_run_retro_team_mode(**kwargs):
        calls.append(kwargs)
        return _completed_review_report()

    monkeypatch.setattr(
        cli_mod,
        "run_retro_team_mode",
        fake_run_retro_team_mode,
    )

    assert main(["retro", "demo-i1", "--team-mode"]) == 0
    assert len(calls) == 1
    assert calls[0]["iteration"] == "demo-i1"
    assert calls[0]["agent_name"] == "claude"


def test_qa_team_mode_dispatches_to_team_runner_only(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )
    monkeypatch.setattr(
        cli_mod,
        "run_qa",
        lambda **kwargs: pytest.fail("serial run_qa should not run"),
    )

    def fake_run_qa_team_mode(**kwargs):
        calls.append(kwargs)
        return _completed_review_report()

    monkeypatch.setattr(cli_mod, "run_qa_team_mode", fake_run_qa_team_mode)

    assert main(["qa", "demo-i1", "--team-mode"]) == 0
    assert len(calls) == 1
    assert calls[0]["iteration"] == "demo-i1"
    assert calls[0]["reviewer_agent"] == "claude"


def test_iteration_chain_does_not_use_retro_team_mode(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(cli_mod, "cmd_run", lambda args: 0)
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )
    monkeypatch.setattr(
        cli_mod,
        "run_retro_team_mode",
        lambda **kwargs: pytest.fail("iteration chain must not use team mode"),
    )

    def fake_run_retro(**kwargs):
        calls.append(kwargs)
        return _completed_review_report()

    monkeypatch.setattr(cli_mod, "run_retro", fake_run_retro)

    assert main(["iteration", "demo-i1", "--dry-run", "--skip-qa"]) == 0
    assert len(calls) == 1
    assert calls[0]["iteration"] == "demo-i1"


def test_iteration_chain_does_not_use_qa_team_mode(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []
    monkeypatch.setattr(cli_mod, "cmd_run", lambda args: 0)
    monkeypatch.setattr(
        cli_mod,
        "_build_adapters",
        lambda cfg: {"claude": SimpleNamespace(family="anthropic")},
    )
    monkeypatch.setattr(
        cli_mod,
        "run_qa_team_mode",
        lambda **kwargs: pytest.fail("iteration chain must not use team mode"),
    )

    def fake_run_qa(**kwargs):
        calls.append(kwargs)
        return _completed_review_report()

    monkeypatch.setattr(cli_mod, "run_qa", fake_run_qa)

    assert main(["iteration", "demo-i1", "--dry-run", "--skip-retro"]) == 0
    assert len(calls) == 1
    assert calls[0]["iteration"] == "demo-i1"


def test_run_flags_accept_override_agents():
    parser = cli_mod.build_parser()

    assert parser.parse_args(
        ["run", "demo-i1", "--override-agents"]
    ).override_agents is True
    assert parser.parse_args(["run", "demo-i1"]).override_agents is False
    assert parser.parse_args(
        ["resume", "demo-i1", "--override-agents"]
    ).override_agents is True
    assert parser.parse_args(["resume", "demo-i1"]).override_agents is False


def test_run_flags_accept_noop_acceptance_reason():
    parser = cli_mod.build_parser()

    assert parser.parse_args(
        ["run", "demo-i1", "--allow-noop-acceptance", "manual suite"]
    ).allow_noop_acceptance == "manual suite"
    assert parser.parse_args(
        ["resume", "demo-i1", "--allow-noop-acceptance", "manual suite"]
    ).allow_noop_acceptance == "manual suite"
    assert parser.parse_args(
        ["iteration", "demo-i1", "--allow-noop-acceptance", "manual suite"]
    ).allow_noop_acceptance == "manual suite"
    assert parser.parse_args(["run", "demo-i1"]).allow_noop_acceptance is None


def test_run_flags_accept_planning_team_mode_only_with_value():
    parser = cli_mod.build_parser()

    assert parser.parse_args(
        ["run", "demo-i1", "--team-mode", "planning"]
    ).team_mode == "planning"
    assert parser.parse_args(
        ["resume", "demo-i1", "--team-mode", "planning"]
    ).team_mode == "planning"
    assert parser.parse_args(
        ["iteration", "demo-i1", "--team-mode", "planning"]
    ).team_mode == "planning"
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "demo-i1", "--team-mode"])
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "demo-i1", "--team-mode", "edit"])


def test_report_to_stdout(workdir: Path, capsys: pytest.CaptureFixture):
    assert main(["report", "demo-i1"]) == 0
    out = capsys.readouterr().out
    assert "# Readiness report — demo-i1" in out
    assert "## Cost (estimated)" in out


def test_report_to_file(workdir: Path, tmp_path: Path, capsys: pytest.CaptureFixture):
    out_path = workdir / "tools" / "logs" / "demo-i1" / "readiness.md"
    assert main(["report", "demo-i1", "--out", str(out_path)]) == 0
    assert out_path.exists()
    text = out_path.read_text()
    assert "Readiness report" in text
    stdout = capsys.readouterr().out
    assert str(out_path) in stdout


def test_improvements_validate_and_list(workdir: Path, capsys: pytest.CaptureFixture):
    log_dir = workdir / "tools" / "logs" / "demo-i1"
    append_record(
        log_dir,
        {
            "id": "imp-001",
            "source_iteration": "demo-i1",
            "source_event": "retro.completed",
            "title": "Pin the control gate",
            "problem": "Approved improvements need visible controls.",
            "classification": "quality",
            "impact": "medium",
            "effort": "small",
            "status": "approved",
            "control_mechanism": "golden event-log test",
        },
    )

    assert main(["improvements", "validate", "demo-i1"]) == 0
    out = capsys.readouterr().out
    assert "OK: 1 improvement record(s)" in out
    assert "tools/logs/demo-i1/improvements.jsonl" in out

    assert main(["improvements", "list", "demo-i1"]) == 0
    out = capsys.readouterr().out
    assert '"id": "imp-001"' in out
    assert '"control_mechanism": "golden event-log test"' in out


def test_prompt_factory_validate_and_render(
    workdir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    draft = {
        "phase_or_iteration_id": "demo-i1",
        "iteration_branch": "demo/iteration-1",
        "final_pr": "TBD",
        "depends_on": "none",
        "blocks": "none",
        "execution_plan": {
            "approach": "task_by_task",
            "qa": "standard",
            "note": "operator chooses agents",
        },
        "tasks": [
            {
                "id": "I4-T1",
                "title": "Do thing",
                "dependencies": [],
                "allowed_files": ["app/thing.py"],
                "test_note": "Use the existing app smoke test.",
                "prompt_summary": "Implement the thing.",
                "review_summary": "Review the thing.",
                "risk_category": "unknown",
            },
        ],
    }
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    assert main(["prompt-factory", "validate", str(draft_path)]) == 0
    out = capsys.readouterr().out
    assert "OK: prompt factory draft has 1 task(s)" in out
    assert "tasks_schema.py" in out

    assert main(["prompt-factory", "render", str(draft_path)]) == 0
    out = capsys.readouterr().out
    assert "# Prompt Factory Draft demo-i1" in out
    assert "| I4-T1 | Do thing | TBD | WAITING" in out


def test_prompt_factory_review_package_and_status(
    workdir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    draft = {
        "phase_or_iteration_id": "demo-i1",
        "iteration_branch": "demo/iteration-1",
        "final_pr": "TBD",
        "depends_on": "none",
        "blocks": "none",
        "execution_plan": {
            "approach": "task_by_task",
            "qa": "standard",
            "note": "operator chooses agents",
        },
        "tasks": [
            {
                "id": "I4-T1",
                "title": "Do thing",
                "dependencies": [],
                "allowed_files": ["app/thing.py"],
                "test_note": "Use the existing app smoke test.",
                "prompt_summary": "Implement the thing.",
                "review_summary": "Review the thing.",
                "risk_category": "unknown",
            },
        ],
    }
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    assert main(["prompt-factory", "review-package", str(draft_path)]) == 0
    out = capsys.readouterr().out
    assert "gate status: INCOMPLETE" in out

    artifact_dir = workdir / "tools" / "logs" / "prompt-factory" / "demo-i1"
    assert (artifact_dir / "prompt_expert_review_prompt.md").exists()
    assert (artifact_dir / "technical_reviewer_review_prompt.md").exists()
    assert (artifact_dir / "review_gate.json").exists()
    assert not (workdir / "iterations" / "demo-i1").exists()

    (artifact_dir / "prompt_expert_review.md").write_text(
        "Verdict: PASS\n",
        encoding="utf-8",
    )
    (artifact_dir / "technical_reviewer_review.md").write_text(
        "Verdict: PASS\n",
        encoding="utf-8",
    )

    assert main(["prompt-factory", "review-status", "demo-i1"]) == 0
    out = capsys.readouterr().out
    assert "demo-i1: PASS" in out
    gate = json.loads((artifact_dir / "review_gate.json").read_text())
    assert gate["status"] == "PASS"
    assert gate["roles"]["prompt_expert"]["verdict"] == "PASS"
    assert gate["roles"]["technical_reviewer"]["verdict"] == "PASS"


def test_prompt_factory_approve_check_and_materialize_dry_run(
    workdir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    draft = {
        "phase_or_iteration_id": "demo-i1",
        "iteration_branch": "demo/iteration-1",
        "final_pr": "TBD",
        "depends_on": "none",
        "blocks": "none",
        "execution_plan": {
            "approach": "task_by_task",
            "qa": "standard",
            "note": "operator chooses agents",
        },
        "tasks": [
            {
                "id": "I4-T1",
                "title": "Do thing",
                "dependencies": [],
                "allowed_files": ["app/thing.py"],
                "test_note": "Use the existing app smoke test.",
                "prompt_summary": "Implement the thing.",
                "review_summary": "Review the thing.",
                "risk_category": "unknown",
            },
        ],
    }
    gate = {
        "draft_id": "demo-i1",
        "status": "PASS",
        "required_roles": ["prompt_expert", "technical_reviewer"],
        "roles": {
            "prompt_expert": {"state": "PRESENT", "verdict": "PASS"},
            "technical_reviewer": {"state": "PRESENT", "verdict": "PASS"},
        },
    }
    approval = {
        "draft_id": "demo-i1",
        "approved_by": "operator",
        "approved_at": "2026-05-25T12:00:00Z",
        "decision": "approved",
        "review_gate_status": "PASS",
        "notes": "approved in test fixture",
    }
    draft_path = tmp_path / "draft.json"
    gate_path = tmp_path / "review_gate.json"
    approval_path = tmp_path / "approval.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    assert main(
        [
            "prompt-factory",
            "approve-check",
            str(draft_path),
            str(gate_path),
            str(approval_path),
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "OK: Prompt Factory approval gate passed for demo-i1" in out

    target = Path("iterations") / "phase-demo" / "demo-i1-generated"
    assert main(
        [
            "prompt-factory",
            "materialize",
            str(draft_path),
            str(gate_path),
            str(approval_path),
            "--target",
            str(target),
            "--dry-run",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "would write 4 file(s)" in out
    assert "demo-i1-generated" in out
    assert not (workdir / target).exists()


def test_missing_project_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                              capsys: pytest.CaptureFixture):
    monkeypatch.chdir(tmp_path)
    assert main(["validate", "anything"]) == 1
    err = capsys.readouterr().err
    assert "project.yaml" in err


def test_validate_fails_when_prior_action_items_not_addressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """validate must fail when prior retro exists but prompt.md lacks the
    carried-forward section.
    """
    iter_dir = tmp_path / "iterations" / "tools" / "demo-i1"
    _write_iteration_scaffold(
        iter_dir,
        tasks_text=VALID_TASKS_MD.replace(
            "**Depends on:** none",
            "**Depends on:** `demo-i14`",
        ),
        prompt_text="# Target iteration\n\nSome context.\n",
    )

    prev_log = tmp_path / "tools" / "logs" / "demo-i14"
    prev_log.mkdir(parents=True)
    (prev_log / "retrospective.md").write_text(
        "# Retrospective\n\n## Action items\n- Fix the foo\n- Improve bar\n"
    )

    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(VALID_PROJECT_YAML)

    monkeypatch.chdir(tmp_path)
    result = main(["validate", "demo-i1"])
    assert result == 1
    err = capsys.readouterr().err
    assert "lacks a '## Carried-forward action items' section" in err
    assert str(prev_log / "retrospective.md") in err


def test_validate_passes_when_prior_action_items_marked_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """validate must pass when prompt.md marks prior retro items closed."""
    iter_dir = tmp_path / "iterations" / "tools" / "demo-i1"
    _write_iteration_scaffold(
        iter_dir,
        tasks_text=VALID_TASKS_MD.replace(
            "**Depends on:** none",
            "**Depends on:** `demo-i14`",
        ),
        prompt_text=(
            "# Target iteration\n\n"
            "## Carried-forward action items\n\n"
            "(none — all closed in iteration demo-i14)\n"
        ),
    )

    prev_log = tmp_path / "tools" / "logs" / "demo-i14"
    prev_log.mkdir(parents=True)
    (prev_log / "retrospective.md").write_text("# Retrospective\n\nDone.\n")

    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(VALID_PROJECT_YAML)

    monkeypatch.chdir(tmp_path)
    result = main(["validate", "demo-i1"])
    assert result == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_validate_extracts_iter_id_from_prose_depends_on(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """validate must extract the prior iteration id from prose headers."""
    iter_dir = tmp_path / "iterations" / "tools" / "demo-i1"
    _write_iteration_scaffold(
        iter_dir,
        tasks_text=VALID_TASKS_MD.replace(
            "**Depends on:** none",
            "**Depends on:** Demo-I14 (debt cleanup) merged to `phase-demo`",
        ),
        prompt_text="# Target iteration\n\nSome context.\n",
    )

    prev_log = tmp_path / "tools" / "logs" / "demo-i14"
    prev_log.mkdir(parents=True)
    (prev_log / "retrospective.md").write_text(
        "# Retrospective\n\n## Action items\n- Do thing\n"
    )

    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(VALID_PROJECT_YAML)

    monkeypatch.chdir(tmp_path)
    result = main(["validate", "demo-i1"])
    assert result == 1
    err = capsys.readouterr().err
    assert "demo-i14" in err
    assert "Carried-forward action items" in err


def test_build_runner_uses_existing_orch_workdir(
    workdir: Path, monkeypatch: pytest.MonkeyPatch,
):
    orch_dir = workdir / ".orch" / "worktrees" / "demo-i1"
    orch_dir.mkdir(parents=True)
    for rel in [".orch/project.yaml", "iterations/phase-demo/demo-i1/tasks.md"]:
        src = workdir / rel
        dst = orch_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())

    called = {"count": 0}

    def fake_ensure(repo_root, iteration, iter_branch, *, worktree_root=None):
        called["count"] += 1
        assert repo_root == workdir
        assert iteration == "demo-i1"
        assert iter_branch == "demo/iteration-1"
        assert worktree_root == workdir / ".orch" / "worktrees"
        return orch_dir

    monkeypatch.setattr(cli_mod, "ensure_orch_workdir", fake_ensure)

    args = SimpleNamespace(
        iteration="demo-i1",
        implementer=None,
        reviewer=None,
        independence=None,
        stop_on_first_failure=False,
        accept_external=False,
        skip_impl=[],
        dry_run=True,
    )
    runner = _build_runner(args, require_state=False)

    assert called["count"] == 1
    assert runner.deps.repo_root == workdir
    assert runner.deps.cwd == orch_dir
    runner.deps.run_lock.release()


def test_build_runner_uses_configured_worktree_root(
    workdir: Path, monkeypatch: pytest.MonkeyPatch,
):
    with (workdir / ".orch" / "project.yaml").open("a", encoding="utf-8") as fh:
        fh.write("\npaths:\n  worktree_root: .orch/custom-worktrees\n")
    orch_dir = workdir / ".orch" / "custom-worktrees" / "demo-i1"
    orch_dir.mkdir(parents=True)
    for rel in [".orch/project.yaml", "iterations/phase-demo/demo-i1/tasks.md"]:
        src = workdir / rel
        dst = orch_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())

    called = {"worktree_root": None}

    def fake_ensure(repo_root, iteration, iter_branch, *, worktree_root=None):
        assert repo_root == workdir
        assert iteration == "demo-i1"
        assert iter_branch == "demo/iteration-1"
        called["worktree_root"] = worktree_root
        return orch_dir

    monkeypatch.setattr(cli_mod, "ensure_orch_workdir", fake_ensure)

    args = SimpleNamespace(
        iteration="demo-i1",
        implementer=None,
        reviewer=None,
        independence=None,
        stop_on_first_failure=False,
        accept_external=False,
        skip_impl=[],
        dry_run=True,
    )
    runner = _build_runner(args, require_state=False)

    assert called["worktree_root"] == workdir / ".orch" / "custom-worktrees"
    assert runner.deps.cwd == orch_dir
    assert runner.deps.run_lock.path == (
        workdir / "tools" / "logs" / "demo-i1" / RUN_LOCK_FILENAME
    )
    runner.deps.run_lock.release()


def test_build_runner_wires_secondary_reviewer(
    workdir: Path, monkeypatch: pytest.MonkeyPatch,
):
    orch_dir = workdir / ".orch" / "worktrees" / "demo-i1"
    orch_dir.mkdir(parents=True)
    for rel in [".orch/project.yaml", "iterations/phase-demo/demo-i1/tasks.md"]:
        src = workdir / rel
        dst = orch_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text())

    monkeypatch.setattr(
        cli_mod,
        "ensure_orch_workdir",
        lambda *args, **kwargs: orch_dir,
    )
    args = SimpleNamespace(
        iteration="demo-i1",
        implementer=None,
        reviewer=None,
        secondary_reviewer="claude",
        independence=None,
        stop_on_first_failure=False,
        accept_external=False,
        skip_impl=[],
        allow_noop_acceptance="manual suite",
        dry_run=True,
    )

    runner = _build_runner(args, require_state=False)

    assert runner.options.secondary_reviewer == "claude"
    assert runner.options.allow_noop_acceptance_reason == "manual suite"
    runner.deps.run_lock.release()


def test_build_runner_blocks_when_iteration_lock_exists(
    workdir: Path,
    capsys: pytest.CaptureFixture,
):
    lock = RunStateLock(
        log_dir=workdir / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        command="run",
        repo_root=workdir,
        orch_workdir=orch_workdir(workdir, "demo-i1"),
    ).acquire()
    args = SimpleNamespace(
        iteration="demo-i1",
        implementer=None,
        reviewer=None,
        independence=None,
        stop_on_first_failure=False,
        accept_external=False,
        skip_impl=[],
        dry_run=True,
    )

    try:
        result = _build_runner(args, require_state=False)
    finally:
        lock.release()

    assert result == 1
    err = capsys.readouterr().err
    assert "active orch run lock" in err
    assert "run.lock" in err
    assert "demo-i1" in err
    assert "demo/iteration-1" in err


def test_resume_head_sha_guard_runs_before_runner_build(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    store = StateStore(
        log_dir=workdir / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
    )
    store.set_iter_branch_sha("old-sha")
    orch_dir = workdir / ".orch" / "worktrees" / "demo-i1"
    orch_dir.mkdir(parents=True)
    guard_calls = []

    def fake_guard(store_arg, *, cwd: Path, iter_branch: str, accept_external: bool):
        guard_calls.append(
            {
                "sha": store_arg.snapshot.iter_branch_sha,
                "cwd": cwd,
                "iter_branch": iter_branch,
                "accept_external": accept_external,
            }
        )
        return SimpleNamespace(ok=False, reason="sha drift before runner build")

    def fail_runner(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("IterationRunner should not be constructed")

    monkeypatch.setattr(
        cli_mod,
        "ensure_orch_workdir",
        lambda *args, **kwargs: orch_dir,
    )
    monkeypatch.setattr(cli_mod, "head_sha_guard", fake_guard)
    monkeypatch.setattr(cli_mod, "IterationRunner", fail_runner)
    monkeypatch.setattr(cli_mod, "build_adapter", lambda name, spec: object())

    args = SimpleNamespace(
        iteration="demo-i1",
        implementer=None,
        reviewer=None,
        secondary_reviewer=None,
        independence=None,
        stop_on_first_failure=False,
        accept_external=False,
        override_agents=False,
        skip_impl=[],
        dry_run=True,
    )

    assert _build_runner(args, require_state=True) == 1

    assert guard_calls == [
        {
            "sha": "old-sha",
            "cwd": orch_dir,
            "iter_branch": "demo/iteration-1",
            "accept_external": False,
        }
    ]
    assert "sha drift before runner build" in capsys.readouterr().err


def test_run_with_state_rejects_head_sha_mismatch(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    store = StateStore(
        log_dir=workdir / "tools" / "logs" / "demo-i1",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
    )
    store.set_iter_branch_sha("old-sha")
    orch_dir = workdir / ".orch" / "worktrees" / "demo-i1"
    orch_dir.mkdir(parents=True)
    guard_calls = []

    def fake_guard(store_arg, *, cwd: Path, iter_branch: str, accept_external: bool):
        guard_calls.append(
            {
                "sha": store_arg.snapshot.iter_branch_sha,
                "cwd": cwd,
                "iter_branch": iter_branch,
                "accept_external": accept_external,
            }
        )
        return SimpleNamespace(ok=False, reason="sha drift on run")

    def fail_runner(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("IterationRunner should not be constructed")

    monkeypatch.setattr(
        cli_mod,
        "ensure_orch_workdir",
        lambda *args, **kwargs: orch_dir,
    )
    monkeypatch.setattr(cli_mod, "head_sha_guard", fake_guard)
    monkeypatch.setattr(cli_mod, "IterationRunner", fail_runner)
    monkeypatch.setattr(cli_mod, "build_adapter", lambda name, spec: object())

    args = SimpleNamespace(
        iteration="demo-i1",
        implementer=None,
        reviewer=None,
        secondary_reviewer=None,
        independence=None,
        stop_on_first_failure=False,
        accept_external=False,
        override_agents=False,
        skip_impl=[],
        dry_run=True,
    )

    assert _build_runner(args, require_state=False) == 1

    assert guard_calls == [
        {
            "sha": "old-sha",
            "cwd": orch_dir,
            "iter_branch": "demo/iteration-1",
            "accept_external": False,
        }
    ]
    assert "sha drift on run" in capsys.readouterr().err


def test_run_reports_friendly_worktree_checked_out_elsewhere(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    def fake_ensure(
        repo_root: Path,
        iteration: str,
        iter_branch: str,
        *,
        worktree_root: Path | None = None,
    ):
        assert worktree_root == workdir / ".orch" / "worktrees"
        raise WorktreePreflightError(
            "orch workdir preflight failed for demo-i1: branch "
            "'demo/iteration-1' is already checked out at /tmp/other. "
            "Next action: run `python -m orch cleanup-workdir demo-i1`."
        )

    monkeypatch.setattr(cli_mod, "ensure_orch_workdir", fake_ensure)

    assert main(["run", "demo-i1", "--dry-run"]) == 1

    err = capsys.readouterr().err
    assert "already checked out" in err
    assert "demo/iteration-1" in err
    assert "cleanup-workdir demo-i1" in err


class _RetryBoard:
    def __init__(self, deps: dict[str, list[str]]) -> None:
        self.tasks = [
            SimpleNamespace(id=task_id, depends_on=depends_on)
            for task_id, depends_on in deps.items()
        ]

    def by_id(self, task_id: str):
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)


def test_retry_resets_only_target_and_blocked_upstream_tasks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    store = StateStore(
        log_dir=tmp_path / "logs",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
    )
    store.task_transition(
        "I1-T1",
        STATUS_STOPPED_PREFIX + "REVIEW_FAIL",
        reason="REVIEW_FAIL",
        msg="needs retry",
    )
    store.record_pr("I1-T1", "https://example.com/pr/1")
    store.task_transition("I1-T2", STATUS_BLOCKED_UPSTREAM)
    store.task_transition("I1-T3", STATUS_DONE)
    store.record_merge("I1-T3", auto_merged=True, merge_sha="done-sha")
    store.task_transition("I1-T4", STATUS_NEEDS_HUMAN_MERGE)
    store.record_pr("I1-T4", "https://example.com/pr/4")
    store.task_transition(
        "I1-T5",
        STATUS_STOPPED_PREFIX + "IMPL_FAILED",
        reason="IMPL_FAILED",
        msg="separate stop",
    )
    store.task_transition("I1-T6", STATUS_BLOCKED_UPSTREAM)
    board = _RetryBoard(
        {
            "I1-T1": [],
            "I1-T2": ["I1-T1"],
            "I1-T3": [],
            "I1-T4": [],
            "I1-T5": [],
            "I1-T6": ["I1-T5"],
        }
    )

    result = cli_mod._cmd_retry_locked(
        SimpleNamespace(iteration="demo-i1", task="I1-T1"),
        board,
        store,
    )

    assert result == 0
    assert store.tasks["I1-T1"].status == "WAITING"
    assert store.tasks["I1-T2"].status == "WAITING"
    assert store.tasks["I1-T3"].status == STATUS_DONE
    assert store.tasks["I1-T3"].merge_sha == "done-sha"
    assert store.tasks["I1-T4"].status == STATUS_NEEDS_HUMAN_MERGE
    assert store.tasks["I1-T4"].pr_url == "https://example.com/pr/4"
    assert store.tasks["I1-T5"].status == STATUS_STOPPED_PREFIX + "IMPL_FAILED"
    assert store.tasks["I1-T5"].stop_reason == "IMPL_FAILED"
    assert store.tasks["I1-T6"].status == STATUS_BLOCKED_UPSTREAM
    assert "reset I1-T1 to WAITING" in capsys.readouterr().out


def test_retry_unblocks_only_downstream_tasks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
):
    test_retry_resets_only_target_and_blocked_upstream_tasks(tmp_path, capsys)


def test_cleanup_workdir_subcommand(
    workdir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    called = {}

    def fake_cleanup(
        repo_root: Path,
        iteration: str,
        *,
        worktree_root: Path | None = None,
    ):
        called["repo_root"] = repo_root
        called["iteration"] = iteration
        called["worktree_root"] = worktree_root

    monkeypatch.setattr(cli_mod, "cleanup_orch_workdir", fake_cleanup)
    assert main(["cleanup-workdir", "demo-i1"]) == 0
    assert called == {
        "repo_root": workdir,
        "iteration": "demo-i1",
        "worktree_root": workdir / ".orch" / "worktrees",
    }
    out = capsys.readouterr().out
    assert "removed orch workdir for demo-i1" in out


class RecordingCommandProvider:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.calls: list[dict] = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def run(self, argv, *, cwd: Path, timeout: int | None = None):
        self.calls.append({"argv": list(argv), "cwd": cwd, "timeout": timeout})
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def test_run_scaffold_lint_uses_command_provider(tmp_path: Path):
    iter_dir = tmp_path / "iterations" / "demo"
    iter_dir.mkdir(parents=True)
    provider = RecordingCommandProvider()

    rc = _run_scaffold_lint(
        tmp_path,
        iter_dir,
        command_provider=provider,
    )

    assert rc == 0
    assert len(provider.calls) == 1
    assert provider.calls[0]["argv"] == [
        sys.executable,
        "-m",
        "orch.scaffold_lint",
        str(iter_dir),
    ]
    assert provider.calls[0]["cwd"] == tmp_path
    assert provider.calls[0]["timeout"] is None


def test_run_scaffold_lint_fails_closed(tmp_path: Path, capsys: pytest.CaptureFixture):
    iter_dir = tmp_path / "iterations" / "demo"
    iter_dir.mkdir(parents=True)
    provider = RecordingCommandProvider(returncode=1, stderr="lint failed\n")

    rc = _run_scaffold_lint(
        tmp_path,
        iter_dir,
        command_provider=provider,
    )

    captured = capsys.readouterr()
    assert rc == 1
    assert "lint failed" in captured.err
    assert "validate failed: scaffold_lint did not pass" in captured.err


# --- B-14 R8: same-family QA/retro independence warning -------------------


def test_qa_warns_when_reviewer_shares_recorded_implementer_family(
    capsys: pytest.CaptureFixture,
):
    store = SimpleNamespace(
        tasks={"I1-T1": SimpleNamespace(implementer="codex")},
        events=[],
    )
    adapters = {
        "codex": SimpleNamespace(family="openai"),
        "codex-alt": SimpleNamespace(family="openai"),
        "claude": SimpleNamespace(family="anthropic"),
    }

    cli_mod._warn_if_same_family_as_implementer(
        store, adapters, "codex-alt", role="QA reviewer"
    )

    err = capsys.readouterr().err
    assert "same model family" in err
    assert "QA reviewer 'codex-alt'" in err
    assert "implementer 'codex'" in err


def test_qa_no_warning_for_cross_family_reviewer(
    capsys: pytest.CaptureFixture,
):
    store = SimpleNamespace(
        tasks={"I1-T1": SimpleNamespace(implementer="codex")},
        events=[],
    )
    adapters = {
        "codex": SimpleNamespace(family="openai"),
        "claude": SimpleNamespace(family="anthropic"),
    }

    cli_mod._warn_if_same_family_as_implementer(
        store, adapters, "claude", role="QA reviewer"
    )

    assert capsys.readouterr().err == ""


def test_recorded_implementer_falls_back_to_agents_resolved_event():
    store = SimpleNamespace(
        tasks={},  # no per-task implementer recorded
        events=[
            {"meta": {"event": "agents_resolved", "implementer": "codex"}},
        ],
    )

    assert cli_mod._recorded_implementer_names(store) == {"codex"}
