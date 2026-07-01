"""Tests for the retrospective subcommand — 3 parallel perspectives.

Uses FakeAdapter and real temp git repos, matching the patterns from
test_runner.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from orch.agents.base import AgentResult
from orch.config import costs, load_config
from orch.cost import CostLogger
from orch.git_ops import commit, git, stage_all
from orch.retro import RETRO_ROLES, run_retro
from orch.state import StateStore
from orch.tasks_schema import parse_tasks_md

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
patterns:
  task_id: '^I(\\d+)-T(\\d+)$'
  task_detail_heading: '^###\\s+(?P<id>I\\d+-T\\d+)\\s+\u2014\\s+(?P<title>.+?)\\s*$'
  phase_branch: '^phase-[A-Za-z0-9][A-Za-z0-9_-]*$'
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
"""

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

---

## Task Details

### I1-T1 \u2014 First

**Allowed files:**
```
src/a.py
```
"""


@dataclass
class FakeAdapter:
    name: str
    family: str
    script: list[Callable] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def invoke(self, prompt, *, timeout, workdir):
        self.calls.append({"prompt": prompt, "timeout": timeout})
        if not self.script:
            return AgentResult(
                exit_code=0,
                stdout="## What went well\n- Good\n## What could improve\n- Better\n## Action items\n- Do X",
                stderr="", duration_s=0.1,
                input_tokens=100, output_tokens=50, tokens_exact=False,
            )
        fn = self.script.pop(0)
        return fn(self, prompt, workdir)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(["init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git(["config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    git(["config", "user.name", "Tester"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("r\n")
    (tmp_path / ".gitignore").write_text("tools/logs/\n")
    stage_all(tmp_path)
    commit(tmp_path, "init")
    git(["branch", "demo/iteration-1"], cwd=tmp_path, check=True)
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(PROJECT_YAML)
    iter_dir = tmp_path / "iterations" / "demo-i1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tasks.md").write_text(TASKS_MD)
    stage_all(tmp_path)
    commit(tmp_path, "project files")
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=tmp_path, check=True)
    return tmp_path


def _make_deps(repo: Path):
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
    return cfg, board, store, cost


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelExecution:
    def test_all_three_roles_run(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        assert len(report.roles) == 3
        assert all(r.ok for r in report.roles)
        assert len(adapter.calls) == 3
        role_names = {r.role for r in report.roles}
        assert role_names == set(RETRO_ROLES)

    def test_output_files_created(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        retro_dir = repo / "tools" / "logs" / "demo-i1" / "retro"
        assert retro_dir.exists()
        for role in RETRO_ROLES:
            assert (retro_dir / f"{role}.md").exists()
        assert (repo / "tools" / "logs" / "demo-i1" / "retrospective.md").exists()

    def test_improvement_artifact_is_visible(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        log_dir = repo / "tools" / "logs" / "demo-i1"
        assert (log_dir / "improvements.jsonl").exists()
        text = (log_dir / "retrospective.md").read_text()
        assert "## Six Sigma Improvement Records" in text
        assert "`tools/logs/demo-i1/improvements.jsonl`" in text
        assert "non-empty `control_mechanism`" in text
        assert "do not approve or implement improvements autonomously" in text


class TestPartialFailure:
    def test_one_timeout_others_succeed(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        call_num = {"n": 0}

        def ordered_invoke(adapter, prompt, workdir):
            call_num["n"] += 1
            if call_num["n"] == 1:
                return AgentResult(
                    exit_code=-1, stdout="partial...", stderr="",
                    duration_s=10.0, input_tokens=10, output_tokens=5,
                    tokens_exact=False, partial=True,
                )
            return AgentResult(
                exit_code=0, stdout="OK", stderr="", duration_s=0.1,
                input_tokens=100, output_tokens=50, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic",
            script=[ordered_invoke] * 3,
        )
        report = run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        ok_count = sum(1 for r in report.roles if r.ok)
        timeout_count = sum(1 for r in report.roles if r.timed_out)
        assert ok_count == 2
        assert timeout_count == 1
        # Report still written
        assert (repo / "tools" / "logs" / "demo-i1" / "retrospective.md").exists()

    def test_error_does_not_crash(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        call_num = {"n": 0}

        def conditional(adapter, prompt, workdir):
            call_num["n"] += 1
            if call_num["n"] == 2:
                raise RuntimeError("boom")
            return AgentResult(
                exit_code=0, stdout="Fine.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic",
            script=[conditional] * 3,
        )
        report = run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        ok_count = sum(1 for r in report.roles if r.ok)
        err_count = sum(1 for r in report.roles if not r.ok)
        assert ok_count == 2
        assert err_count == 1

    def test_nonzero_exit_marks_role_failed(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        call_num = {"n": 0}

        def conditional(adapter, prompt, workdir):
            call_num["n"] += 1
            if call_num["n"] == 2:
                return AgentResult(
                    exit_code=2, stdout="failed", stderr="bad role",
                    duration_s=0.1, input_tokens=10, output_tokens=10,
                    tokens_exact=False,
                )
            return AgentResult(
                exit_code=0, stdout="Fine.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic",
            script=[conditional] * 3,
        )
        report = run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )

        failed = [r for r in report.roles if not r.ok]
        assert len(failed) == 1
        assert "bad role" in failed[0].text


class TestRoleFiltering:
    def test_subset_of_roles(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
            roles=["developer", "scrum_master"],
        )
        assert len(report.roles) == 2
        assert {r.role for r in report.roles} == {"developer", "scrum_master"}

    def test_invalid_roles_ignored(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
            roles=["developer", "nonexistent"],
        )
        assert len(report.roles) == 1
        assert report.roles[0].role == "developer"


class TestQaReportIntegration:
    def test_reads_qa_report_when_available(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        # Create a fake QA report
        log_dir = repo / "tools" / "logs" / "demo-i1"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "qa_report.md").write_text(
            "# QA Report\n## Security\nFound XSS vulnerability\n"
        )
        captured_prompts = []

        def capture_prompt(adapter, prompt, workdir):
            captured_prompts.append(prompt)
            return AgentResult(
                exit_code=0, stdout="Noted.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic",
            script=[capture_prompt] * 3,
        )
        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        # QA report content should appear in prompts
        assert any("XSS vulnerability" in p for p in captured_prompts)

    def test_handles_missing_qa_report(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        # Should not raise even without qa_report.md
        report = run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        assert len(report.roles) == 3


class TestCostRecords:
    def test_cost_recorded_per_role(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        cost_file = repo / "tools" / "logs" / "demo-i1" / "cost.jsonl"
        assert cost_file.exists()
        records = [json.loads(line) for line in cost_file.read_text().splitlines()]
        assert len(records) == 3
        for rec in records:
            assert rec["step"] == "RETRO"
            assert rec["task"] == "RETRO"
            assert rec["agent"] == "claude"
            assert "role" in rec.get("extra", {})

    def test_exact_usage_model_recorded_per_role(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)

        def exact_retro(adapter, prompt, workdir):
            return AgentResult(
                exit_code=0,
                stdout="## What went well\n- Good\n## Action items\n- None",
                stderr="",
                duration_s=0.1,
                input_tokens=1000,
                output_tokens=200,
                tokens_exact=True,
                provider="claude",
                model="claude-retro-model",
                cached_input_tokens=300,
                cache_creation_input_tokens=400,
                parser_status="parsed",
                extra={"raw_terminal_json": "{}"},
            )

        adapter = FakeAdapter(
            name="claude",
            family="anthropic",
            script=[exact_retro] * 3,
        )
        run_retro(
            cfg=cfg,
            board=board,
            state=store,
            cost=cost,
            adapters={"claude": adapter},
            iteration="demo-i1",
            cwd=repo,
            agent_name="claude",
        )

        records = [
            json.loads(line)
            for line in (
                repo / "tools" / "logs" / "demo-i1" / "cost.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert len(records) == 3
        assert all(rec["estimated"] is False for rec in records)
        assert all(rec["provider"] == "claude" for rec in records)
        assert all(rec["model"] == "claude-retro-model" for rec in records)
        assert all(rec["cached_input_tokens"] == 300 for rec in records)
        assert all(
            rec["cache_creation_input_tokens"] == 400 for rec in records
        )
        assert all(rec["parser_status"] == "parsed" for rec in records)
        # Raw CLI dump is stripped from each persisted record; the parsed
        # usage fields asserted above remain intact.
        assert all(
            "raw_terminal_json" not in rec["extra"]["agent_result_extra"]
            for rec in records
        )


class TestTimingEvidenceContext:
    def _seed_cost_duration(self, cost: CostLogger):
        cost.record(
            task="I1-T1", step="IMPL", agent="codex", family="openai",
            input_tokens=100, output_tokens=50, exact=True,
            duration_s=120.0, exit_code=0,
        )

    def test_timing_jsonl_status_reaches_prompts_and_report(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        self._seed_cost_duration(cost)
        log_dir = repo / "tools" / "logs" / "demo-i1"
        (log_dir / "timing.jsonl").write_text("{}\n", encoding="utf-8")
        captured_prompts = []

        def capture(adapter, prompt, workdir):
            captured_prompts.append(prompt)
            return AgentResult(
                exit_code=0, stdout="Noted.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic", script=[capture] * 3,
        )

        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )

        assert captured_prompts
        assert all("Timing evidence: **present**" in p for p in captured_prompts)
        assert all("Operator action:" not in p for p in captured_prompts)
        report_text = (
            repo / "tools" / "logs" / "demo-i1" / "retrospective.md"
        ).read_text()
        assert "Timing evidence: **present**" in report_text

    def test_notes_fallback_status_reaches_prompts_and_report(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        self._seed_cost_duration(cost)
        log_dir = repo / "tools" / "logs" / "demo-i1"
        (log_dir / "notes.md").write_text(
            "Manual timing captured in operator notes.\n", encoding="utf-8"
        )
        captured_prompts = []

        def capture(adapter, prompt, workdir):
            captured_prompts.append(prompt)
            return AgentResult(
                exit_code=0, stdout="Noted.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic", script=[capture] * 3,
        )

        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )

        assert captured_prompts
        assert all(
            "Timing evidence: **operator-provided fallback**" in p
            for p in captured_prompts
        )
        assert all("`tools/logs/demo-i1/notes.md`" in p for p in captured_prompts)
        assert all("Operator action:" not in p for p in captured_prompts)
        report_text = (
            repo / "tools" / "logs" / "demo-i1" / "retrospective.md"
        ).read_text()
        assert "Timing evidence: **operator-provided fallback**" in report_text
        assert "`tools/logs/demo-i1/notes.md`" in report_text

    def test_missing_timing_status_reaches_prompts_and_report(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        self._seed_cost_duration(cost)
        captured_prompts = []

        def capture(adapter, prompt, workdir):
            captured_prompts.append(prompt)
            return AgentResult(
                exit_code=0, stdout="Noted.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic", script=[capture] * 3,
        )

        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )

        assert captured_prompts
        assert all("Timing evidence: **missing**" in p for p in captured_prompts)
        assert all("Operator action:" in p for p in captured_prompts)
        assert all("  Total: 2.0 min" not in p for p in captured_prompts)
        report_text = (
            repo / "tools" / "logs" / "demo-i1" / "retrospective.md"
        ).read_text()
        assert "Timing evidence: **missing**" in report_text
        assert "Operator action:" in report_text


def test_retro_cost_summary_uses_honesty_label(tmp_path: Path):
    from orch.retro import _cost_summary_text

    log_dir = tmp_path / "tools" / "logs" / "demo-i1"
    log_dir.mkdir(parents=True)
    cost_path = log_dir / "cost.jsonl"
    cost_path.write_text(
        '{"ts": "2026-04-30T10:00:00Z", "iteration": "demo-i1", '
        '"task": "T1", "step": "IMPL", "agent": "claude", '
        '"duration_s": 60.0, "est_cost_usd": 0.012}\n',
        encoding="utf-8",
    )

    out = _cost_summary_text(cost_path)

    assert (
        "estimated equivalent API cost (subscription — not billed per request)"
        in out
    )


class TestOutputFileSchema:
    def test_retrospective_structure(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        report_file = repo / "tools" / "logs" / "demo-i1" / "retrospective.md"
        text = report_file.read_text()
        assert "# Retrospective" in text
        assert "**Perspectives:**" in text
        assert "Developer" in text
        assert "Product Owner" in text
        assert "Scrum Master" in text

    def test_per_role_file_has_header(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        retro_dir = repo / "tools" / "logs" / "demo-i1" / "retro"
        for role in RETRO_ROLES:
            text = (retro_dir / f"{role}.md").read_text()
            assert text.startswith(f"# Retrospective: {role}")


def test_retro_prompt_uses_root_cause_contract(repo: Path):
    cfg, board, store, cost = _make_deps(repo)
    adapter = FakeAdapter(name="claude", family="anthropic")

    run_retro(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
        agent_name="claude",
    )

    developer_prompt = next(
        call["prompt"] for call in adapter.calls
        if "developer" in call["prompt"]
    )
    assert "`## Root-Cause Classification`" in developer_prompt
    assert "`## Improvement Records`" in developer_prompt
    assert "Retro Verdict: COMPLETE | COMPLETE_WITH_FOLLOWUPS | INCOMPLETE" in developer_prompt
    assert "`RETRO-D1`, `RETRO-D2`" in developer_prompt
    assert "`prompt_missing_invariant`" in developer_prompt
    assert "| ID | Task | Signal | Root cause | Template change needed? | Exact rewrite |" in developer_prompt
    assert "| ID | Review | Calibration | Missed? | Too strict? | Change |" in developer_prompt
    assert "| ID | QA finding | Should have been caught earlier? | Missing gate |" in developer_prompt


class TestPreviousRetros:
    def test_reads_previous_iteration_retros(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        # Create a previous iteration's retrospective
        prev_dir = repo / "tools" / "logs" / "demo-i0"
        prev_dir.mkdir(parents=True, exist_ok=True)
        (prev_dir / "retrospective.md").write_text(
            "# Previous Retro\nWe had deployment issues.\n"
        )
        captured_prompts = []

        def capture(adapter, prompt, workdir):
            captured_prompts.append(prompt)
            return AgentResult(
                exit_code=0, stdout="Noted.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic",
            script=[capture] * 3,
        )
        run_retro(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            agent_name="claude",
        )
        # Previous retro content should appear in prompts
        assert any("deployment issues" in p for p in captured_prompts)


def test_retro_team_mode_uses_read_only_team_artifacts(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from orch import retro as retro_mod
    from orch.team_mode import ReadOnlyTeamResult

    cfg, board, store, cost = _make_deps(repo)
    captured = {}

    def fake_run_read_only_team(**kwargs):
        captured.update(kwargs)
        return [
            ReadOnlyTeamResult(
                role=role.name,
                ok=True,
                status="completed",
                text=f"## Outcome\n{role.name} team artifact\n",
                artifact_dir=role.artifact_dir,
                verdict="COMPLETE",
                exit_code=0,
            )
            for role in kwargs["roles"]
        ]

    monkeypatch.setattr(retro_mod, "run_read_only_team", fake_run_read_only_team)

    report = retro_mod.run_retro_team_mode(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"claude": FakeAdapter(name="claude", family="anthropic")},
        iteration="demo-i1",
        cwd=repo,
        agent_name="claude",
        timeout=654,
    )

    assert len(report.roles) == 3
    assert all(result.ok for result in report.roles)
    assert captured["team_name"] == "retro-demo-i1"
    assert captured["command"] == (
        "claude",
        "-p",
        "--output-format",
        "json",
    )
    assert captured["timeout"] == 654
    assert [role.name for role in captured["roles"]] == list(RETRO_ROLES)
    assert all(
        role.artifact_dir
        == repo / "tools" / "logs" / "demo-i1" / "retro" / "team" / role.name
        for role in captured["roles"]
    )
    assert all("Write only the declared artifacts" not in role.prompt for role in captured["roles"])

    retro_dir = repo / "tools" / "logs" / "demo-i1" / "retro"
    for role in RETRO_ROLES:
        assert (retro_dir / f"{role}.md").exists()
    report_text = (repo / "tools" / "logs" / "demo-i1" / "retrospective.md").read_text()
    assert "**Perspectives:** 3/3 completed" in report_text
    assert "team artifact" in report_text


def test_retro_team_mode_malformed_artifact_becomes_incomplete_role(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from orch import retro as retro_mod
    from orch.team_mode import ReadOnlyTeamResult

    cfg, board, store, cost = _make_deps(repo)

    def fake_run_read_only_team(**kwargs):
        return [
            ReadOnlyTeamResult(
                role="developer",
                ok=False,
                status="malformed",
                text="developer: verdict.txt has 'BROKEN'",
                artifact_dir=kwargs["roles"][0].artifact_dir,
                error="bad verdict",
            ),
            ReadOnlyTeamResult(
                role="product_owner",
                ok=True,
                status="completed",
                text="ok",
                artifact_dir=kwargs["roles"][1].artifact_dir,
                verdict="COMPLETE",
                exit_code=0,
            ),
            ReadOnlyTeamResult(
                role="scrum_master",
                ok=True,
                status="completed",
                text="ok",
                artifact_dir=kwargs["roles"][2].artifact_dir,
                verdict="COMPLETE",
                exit_code=0,
            ),
        ]

    monkeypatch.setattr(retro_mod, "run_read_only_team", fake_run_read_only_team)

    report = retro_mod.run_retro_team_mode(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"claude": FakeAdapter(name="claude", family="anthropic")},
        iteration="demo-i1",
        cwd=repo,
        agent_name="claude",
    )

    incomplete = report.incomplete_roles
    assert [role.role for role in incomplete] == ["developer"]
    report_text = (repo / "tools" / "logs" / "demo-i1" / "retrospective.md").read_text()
    assert "Developer [ERROR]" in report_text
    assert "bad verdict" in report_text or "BROKEN" in report_text


def _fake_adapter_invoke_with_routing(
    self,
    prompt,
    *,
    timeout,
    workdir,
    routing_options=None,
):
    self.calls.append(
        {
            "prompt": prompt,
            "timeout": timeout,
            "routing_options": routing_options,
        }
    )
    if not self.script:
        return AgentResult(
            exit_code=0,
            stdout=(
                "## What went well\n- Good\n"
                "## What could improve\n- Better\n"
                "## Action items\n- Do X"
            ),
            stderr="",
            duration_s=0.1,
            input_tokens=100,
            output_tokens=50,
            tokens_exact=False,
        )
    fn = self.script.pop(0)
    return fn(self, prompt, workdir)


FakeAdapter.invoke = _fake_adapter_invoke_with_routing


def _append_quality_gate_routing_config(repo: Path) -> None:
    project_path = repo / ".orch" / "project.yaml"
    project_path.write_text(
        project_path.read_text()
        + """
model_routing:
  agent_overrides:
    codex:
      model_flag: "-m"
      tier_models:
        standard: "fixture-standard"
        strong: "fixture-strong"
        max: "fixture-max"
      effort_flags:
        low:
          args: ["-c", "model_reasoning_effort=low"]
        medium:
          args: ["-c", "model_reasoning_effort=medium"]
        high:
          args: ["-c", "model_reasoning_effort=high"]
        max:
          args: ["-c", "model_reasoning_effort=xhigh"]
"""
    )


def test_retro_empty_config_passes_empty_routing_options_to_role(
    repo: Path,
):
    cfg, board, store, cost = _make_deps(repo)
    adapter = FakeAdapter(name="codex", family="openai")

    report = run_retro(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"codex": adapter},
        iteration="demo-i1",
        cwd=repo,
        agent_name="codex",
        roles=["developer"],
    )

    assert len(report.roles) == 1
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["routing_options"].args == ()
    assert adapter.calls[0]["routing_options"].env == {}


def test_retro_quality_gate_routing_reaches_role_invocation(repo: Path):
    _append_quality_gate_routing_config(repo)
    cfg, board, store, cost = _make_deps(repo)
    adapter = FakeAdapter(name="codex", family="openai")

    run_retro(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"codex": adapter},
        iteration="demo-i1",
        cwd=repo,
        agent_name="codex",
        roles=["developer"],
    )

    assert len(adapter.calls) == 1
    assert adapter.calls[0]["routing_options"].args == (
        "-m",
        "fixture-max",
        "-c",
        "model_reasoning_effort=xhigh",
    )
    assert adapter.calls[0]["routing_options"].env == {}
