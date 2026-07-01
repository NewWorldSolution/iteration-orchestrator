"""Tests for the QA subcommand — parallel LLM reviewers.

Uses FakeAdapter and real temp git repos, matching the patterns from
test_runner.py.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

from orch.agents.base import AgentResult
from orch.config import LoadedConfig, costs, load_config
from orch.cost import CostLogger
from orch.git_ops import commit, git, stage_all
from orch.qa import (
    QA_ROLES,
    QA_TEAM_VERDICTS,
    QaDiffBaseError,
    QaEmptyDiffError,
    resolve_diff_base,
    run_qa,
    run_qa_team_mode,
)
from orch.state import StateStore
from orch.tasks_schema import parse_tasks_md

# ---------------------------------------------------------------------------
# Fixtures (same pattern as test_runner.py)
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

TASKS_MD_WITH_DIFF_BASE = """\
# Demo iteration
## Task Board

**Status:** WAITING
**Iteration branch:** `demo/iteration-1`
**Depends on:** none
**Blocks:** none
**Diff base:** `main`

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
                stdout="## Summary\nAll good.\n\n## Findings\n- None\n\n## Verdict\nPASS",
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
    # The derived QA diff base for demo-i1 is `phase-demo` (phase_branch_pattern
    # "phase-{phase}"). It must exist as a real ref and the iteration branch
    # must be ahead of it, otherwise run_qa now fails closed (the old fixture
    # left `phase-demo` non-existent and QA reviewed a "(diff unavailable)"
    # placeholder vacuously).
    git(["branch", "phase-demo"], cwd=tmp_path, check=True)
    git(["branch", "demo/iteration-1"], cwd=tmp_path, check=True)
    # Project files
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(PROJECT_YAML)
    iter_dir = tmp_path / "iterations" / "demo-i1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "tasks.md").write_text(TASKS_MD)
    stage_all(tmp_path)
    commit(tmp_path, "project files")
    git(["branch", "-f", "demo/iteration-1", "HEAD"], cwd=tmp_path, check=True)
    return tmp_path


def _make_deps(repo: Path, tasks_md_content: str | None = None):
    """Build cfg, board, store, cost, adapters for QA tests."""
    if tasks_md_content:
        (repo / "iterations" / "demo-i1" / "tasks.md").write_text(tasks_md_content)
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
    def test_all_five_roles_run(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        adapters = {"claude": adapter}
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters=adapters, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        assert len(report.roles) == 5
        assert all(r.ok for r in report.roles)
        assert len(adapter.calls) == 5
        # All 5 role names present
        role_names = {r.role for r in report.roles}
        assert role_names == set(QA_ROLES)

    def test_output_files_created(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        qa_dir = repo / "tools" / "logs" / "demo-i1" / "qa"
        assert qa_dir.exists()
        for role in QA_ROLES:
            assert (qa_dir / f"{role}.md").exists()
        assert (repo / "tools" / "logs" / "demo-i1" / "qa_report.md").exists()
        assert (qa_dir / "diff_base.txt").exists()


class TestPartialFailure:
    def test_one_timeout_others_succeed(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)

        def conditional_invoke(adapter, prompt, workdir):
            if "security reviewer" in prompt:
                return AgentResult(
                    exit_code=-1, stdout="partial...", stderr="",
                    duration_s=10.0, input_tokens=10, output_tokens=5,
                    tokens_exact=False, partial=True,
                )
            return AgentResult(
                exit_code=0, stdout="All good.\n## Verdict\nPASS",
                stderr="", duration_s=0.1,
                input_tokens=100, output_tokens=50, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic",
            script=[conditional_invoke] * 5,
        )
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        # 4 succeed, 1 times out
        ok_count = sum(1 for r in report.roles if r.ok)
        timeout_count = sum(1 for r in report.roles if r.timed_out)
        assert ok_count == 4
        assert timeout_count == 1
        # Report file still written
        assert (repo / "tools" / "logs" / "demo-i1" / "qa_report.md").exists()

    def test_error_does_not_crash_others(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)

        def conditional_invoke(adapter, prompt, workdir):
            if "architecture reviewer" in prompt:
                raise RuntimeError("boom")
            return AgentResult(
                exit_code=0, stdout="Fine.", stderr="", duration_s=0.1,
                input_tokens=10, output_tokens=10, tokens_exact=False,
            )

        adapter = FakeAdapter(
            name="claude", family="anthropic",
            script=[conditional_invoke] * 5,
        )
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        ok_count = sum(1 for r in report.roles if r.ok)
        err_count = sum(1 for r in report.roles if not r.ok)
        assert ok_count == 4
        assert err_count == 1

    def test_nonzero_exit_marks_role_failed(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)

        def conditional_invoke(adapter, prompt, workdir):
            if "product reviewer" in prompt:
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
            script=[conditional_invoke] * 5,
        )
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )

        failed = [r for r in report.roles if not r.ok]
        assert [r.role for r in failed] == ["product"]
        assert "bad role" in failed[0].text


class TestRoleFiltering:
    def test_subset_of_roles(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
            roles=["security", "test"],
        )
        assert len(report.roles) == 2
        assert {r.role for r in report.roles} == {"security", "test"}
        assert len(adapter.calls) == 2

    def test_invalid_roles_ignored(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
            roles=["security", "nonexistent"],
        )
        assert len(report.roles) == 1
        assert report.roles[0].role == "security"


def test_qa_prompt_uses_iteration_contract_and_review_artifacts(repo: Path):
    reviews_dir = repo / "iterations" / "demo-i1" / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    (reviews_dir / "review-t1.md").write_text("CUSTOM AUTHORED REVIEW\n")
    cfg, board, store, cost = _make_deps(repo)
    adapter = FakeAdapter(name="claude", family="anthropic")

    run_qa(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
        reviewer_agent="claude", roles=["security"],
    )

    prompt = adapter.calls[0]["prompt"]
    assert "final line: `QA Verdict: OK | CONCERNS | BLOCK | INCOMPLETE`" in prompt
    assert "### Gate 1 - Diff Base and Inverse Diff" in prompt
    assert "### Gate 4 - Cross-Task Grep Gates" in prompt
    assert "`_prompt_rules.md` Rule 5 signatures" in prompt
    assert "`[QA-S1] [CRITICAL | SHOULD_FIX | FUTURE] <summary>`" in prompt
    assert "CUSTOM AUTHORED REVIEW" in prompt


def test_qa_team_mode_uses_read_only_team_artifacts(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from orch import qa as qa_mod
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
                text=f"## Summary\n{role.name} team artifact\n",
                artifact_dir=role.artifact_dir,
                verdict="PASS",
                exit_code=0,
            )
            for role in kwargs["roles"]
        ]

    monkeypatch.setattr(qa_mod, "run_read_only_team", fake_run_read_only_team)

    report = run_qa_team_mode(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"claude": FakeAdapter(name="claude", family="anthropic")},
        iteration="demo-i1",
        cwd=repo,
        reviewer_agent="claude",
        timeout=321,
    )

    assert len(report.roles) == 5
    assert all(result.ok for result in report.roles)
    assert captured["team_name"] == "qa-demo-i1"
    assert captured["command"] == (
        "claude",
        "-p",
        "--output-format",
        "json",
    )
    assert captured["timeout"] == 321
    assert captured["task"] == "QA"
    assert captured["step"] == "QA_TEAM"
    assert captured["agent_name"] == "claude"
    assert captured["family"] == "anthropic"
    assert [role.name for role in captured["roles"]] == list(QA_ROLES)
    assert all(role.result_filename == "review.md" for role in captured["roles"])
    assert all(
        role.verdict_labels == QA_TEAM_VERDICTS for role in captured["roles"]
    )
    assert all(
        role.artifact_dir
        == repo / "tools" / "logs" / "demo-i1" / "qa" / "team" / role.name
        for role in captured["roles"]
    )
    assert all(
        "Write only the declared artifacts" not in role.prompt
        for role in captured["roles"]
    )

    qa_dir = repo / "tools" / "logs" / "demo-i1" / "qa"
    for role in QA_ROLES:
        assert (qa_dir / f"{role}.md").exists()
    assert (qa_dir / "diff_base.txt").read_text().strip() == "phase-demo"
    report_text = (
        repo / "tools" / "logs" / "demo-i1" / "qa_report.md"
    ).read_text()
    assert "**Reviewers:** 5/5 completed" in report_text
    assert "team artifact" in report_text


def test_qa_team_mode_malformed_artifact_becomes_incomplete_role(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from orch import qa as qa_mod
    from orch.team_mode import ReadOnlyTeamResult

    cfg, board, store, cost = _make_deps(repo)

    def fake_run_read_only_team(**kwargs):
        return [
            ReadOnlyTeamResult(
                role="security",
                ok=False,
                status="malformed",
                text="security: verdict.txt has 'BROKEN'",
                artifact_dir=kwargs["roles"][0].artifact_dir,
                error="bad verdict",
            ),
            *[
                ReadOnlyTeamResult(
                    role=role.name,
                    ok=True,
                    status="completed",
                    text=f"{role.name} ok",
                    artifact_dir=role.artifact_dir,
                    verdict="PASS",
                    exit_code=0,
                )
                for role in kwargs["roles"][1:]
            ],
        ]

    monkeypatch.setattr(qa_mod, "run_read_only_team", fake_run_read_only_team)

    report = run_qa_team_mode(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"claude": FakeAdapter(name="claude", family="anthropic")},
        iteration="demo-i1",
        cwd=repo,
        reviewer_agent="claude",
    )

    incomplete = report.incomplete_roles
    assert [role.role for role in incomplete] == ["security"]
    report_text = (
        repo / "tools" / "logs" / "demo-i1" / "qa_report.md"
    ).read_text()
    assert "Security [ERROR]" in report_text
    assert "BROKEN" in report_text


def test_qa_team_mode_stuck_role_becomes_timeout(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from orch import qa as qa_mod
    from orch.team_mode import ReadOnlyTeamResult

    cfg, board, store, cost = _make_deps(repo)

    def fake_run_read_only_team(**kwargs):
        return [
            ReadOnlyTeamResult(
                role="security",
                ok=False,
                status="killed",
                text="security: team agent status=killed; missing_artifacts=review.md",
                artifact_dir=kwargs["roles"][0].artifact_dir,
                timed_out=True,
                killed=True,
                exit_code=-9,
                missing_artifacts=("review.md",),
                error="killed by Step-1 stuck path",
            ),
            *[
                ReadOnlyTeamResult(
                    role=role.name,
                    ok=True,
                    status="completed",
                    text=f"{role.name} ok",
                    artifact_dir=role.artifact_dir,
                    verdict="PASS",
                    exit_code=0,
                )
                for role in kwargs["roles"][1:]
            ],
        ]

    monkeypatch.setattr(qa_mod, "run_read_only_team", fake_run_read_only_team)

    report = run_qa_team_mode(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"claude": FakeAdapter(name="claude", family="anthropic")},
        iteration="demo-i1",
        cwd=repo,
        reviewer_agent="claude",
    )

    assert [role.role for role in report.incomplete_roles] == ["security"]
    assert report.incomplete_roles[0].timed_out is True
    role_file = repo / "tools" / "logs" / "demo-i1" / "qa" / "security.md"
    assert role_file.read_text().startswith("# QA Review: security (TIMEOUT)")
    report_text = (
        repo / "tools" / "logs" / "demo-i1" / "qa_report.md"
    ).read_text()
    assert "Security [TIMEOUT]" in report_text
    assert "status=killed" in report_text


class TestDiffBaseResolution:
    def test_from_tasks_md_field(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo, TASKS_MD_WITH_DIFF_BASE)
        diff_base = resolve_diff_base(board, cfg, "demo-i1")
        assert diff_base == "main"

    def test_from_project_yaml_pattern(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        # iteration "demo-i1" -> phase "demo" -> "phase-demo"
        diff_base = resolve_diff_base(board, cfg, "demo-i1")
        assert diff_base == "phase-demo"

    def test_invalid_iteration_name_fails_closed(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        with pytest.raises(QaDiffBaseError, match="cannot resolve phase"):
            resolve_diff_base(board, cfg, "weird")

    def test_hard_error_when_unresolvable(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        # Override config to remove phase_branch_pattern
        data = copy.deepcopy(cfg.data)
        data["project"]["phase_branch_pattern"] = ""
        bad_cfg = LoadedConfig(path=cfg.path, data=data)
        # iteration doesn't match p<phase>-i<n> pattern
        with pytest.raises(QaDiffBaseError, match="QA_DIFF_BASE_UNRESOLVED"):
            resolve_diff_base(board, bad_cfg, "weird")

    def test_diff_base_logged_to_file(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo, TASKS_MD_WITH_DIFF_BASE)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude", roles=["security"],
            allow_empty_diff_reason="diff-base audit fixture",
        )
        diff_base_file = repo / "tools" / "logs" / "demo-i1" / "qa" / "diff_base.txt"
        assert diff_base_file.exists()
        assert diff_base_file.read_text().strip() == "main"


class TestCostRecords:
    def test_cost_recorded_per_role(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        cost_file = repo / "tools" / "logs" / "demo-i1" / "cost.jsonl"
        assert cost_file.exists()
        records = [json.loads(line) for line in cost_file.read_text().splitlines()]
        assert len(records) == 5
        for rec in records:
            assert rec["step"] == "QA"
            assert rec["task"] == "QA"
            assert rec["agent"] == "claude"
            assert "role" in rec.get("extra", {})

    def test_synthesis_adds_extra_cost_record(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude", synthesize=True,
        )
        cost_file = repo / "tools" / "logs" / "demo-i1" / "cost.jsonl"
        records = [json.loads(line) for line in cost_file.read_text().splitlines()]
        # 5 roles + 1 synthesis = 6
        assert len(records) == 6
        synthesis_rec = [r for r in records if r.get("extra", {}).get("role") == "synthesis"]
        assert len(synthesis_rec) == 1

    def test_exact_usage_model_recorded_for_roles_and_synthesis(
        self,
        repo: Path,
    ):
        cfg, board, store, cost = _make_deps(repo)

        def exact_role(adapter, prompt, workdir):
            return AgentResult(
                exit_code=0,
                stdout="## Summary\nAll good.\n\n## Verdict\nPASS",
                stderr="",
                duration_s=0.1,
                input_tokens=1000,
                output_tokens=200,
                tokens_exact=True,
                provider="claude",
                model="claude-role-model",
                cached_input_tokens=300,
                cache_creation_input_tokens=400,
                parser_status="parsed",
                extra={"raw_terminal_json": "{}"},
            )

        def exact_synthesis(adapter, prompt, workdir):
            return AgentResult(
                exit_code=0,
                stdout="## Synthesis\nAll clear.",
                stderr="",
                duration_s=0.1,
                input_tokens=2000,
                output_tokens=300,
                tokens_exact=True,
                provider="claude",
                model="claude-synthesis-model",
                cached_input_tokens=500,
                cache_creation_input_tokens=600,
                parser_status="parsed",
                extra={"raw_terminal_json": "{\"usage\": true}"},
            )

        adapter = FakeAdapter(
            name="claude",
            family="anthropic",
            script=[exact_role] * 5 + [exact_synthesis],
        )
        run_qa(
            cfg=cfg,
            board=board,
            state=store,
            cost=cost,
            adapters={"claude": adapter},
            iteration="demo-i1",
            cwd=repo,
            reviewer_agent="claude",
            synthesize=True,
        )

        records = [
            json.loads(line)
            for line in (
                repo / "tools" / "logs" / "demo-i1" / "cost.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        role_records = [
            rec for rec in records
            if rec["extra"].get("role") in QA_ROLES
        ]
        synthesis = next(
            rec for rec in records
            if rec["extra"].get("role") == "synthesis"
        )
        assert len(role_records) == 5
        assert all(rec["estimated"] is False for rec in role_records)
        assert all(rec["provider"] == "claude" for rec in role_records)
        assert all(rec["model"] == "claude-role-model" for rec in role_records)
        assert all(rec["cached_input_tokens"] == 300 for rec in role_records)
        assert all(
            rec["cache_creation_input_tokens"] == 400
            for rec in role_records
        )
        assert all(rec["parser_status"] == "parsed" for rec in role_records)
        # Raw CLI dump is stripped from each persisted record; the parsed
        # usage fields asserted above remain intact.
        assert all(
            "raw_terminal_json" not in rec["extra"]["agent_result_extra"]
            for rec in role_records
        )
        assert synthesis["estimated"] is False
        assert synthesis["model"] == "claude-synthesis-model"
        assert synthesis["cached_input_tokens"] == 500
        assert synthesis["cache_creation_input_tokens"] == 600
        assert synthesis["parser_status"] == "parsed"


class TestSynthesis:
    def test_synthesis_reads_role_reports(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude", synthesize=True,
        )
        assert report.synthesis is not None
        report_file = repo / "tools" / "logs" / "demo-i1" / "qa_report.md"
        text = report_file.read_text()
        assert "Synthesis" in text


class TestOutputFileSchema:
    def test_qa_report_structure(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        report_file = repo / "tools" / "logs" / "demo-i1" / "qa_report.md"
        text = report_file.read_text()
        assert "# QA Report" in text
        assert "**Diff base:**" in text
        assert "**Reviewers:**" in text
        for role in QA_ROLES:
            assert role.title() in text

    def test_per_role_file_has_header(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        qa_dir = repo / "tools" / "logs" / "demo-i1" / "qa"
        for role in QA_ROLES:
            text = (qa_dir / f"{role}.md").read_text()
            assert text.startswith(f"# QA Review: {role}")


class TestIdempotency:
    def test_running_twice_overwrites(self, repo: Path):
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude", roles=["security"],
        )
        first_text = (
            repo / "tools" / "logs" / "demo-i1" / "qa" / "security.md"
        ).read_text()

        # Second run with different output
        def custom(adapter, prompt, workdir):
            return AgentResult(
                exit_code=0, stdout="SECOND RUN OUTPUT", stderr="",
                duration_s=0.1, input_tokens=10, output_tokens=10,
                tokens_exact=False,
            )
        adapter2 = FakeAdapter(
            name="claude", family="anthropic", script=[custom],
        )
        cost2 = CostLogger(
            path=repo / "tools" / "logs" / "demo-i1" / "cost.jsonl",
            cost_table=costs(cfg), iteration="demo-i1",
        )
        run_qa(
            cfg=cfg, board=board, state=store, cost=cost2,
            adapters={"claude": adapter2}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude", roles=["security"],
        )
        second_text = (
            repo / "tools" / "logs" / "demo-i1" / "qa" / "security.md"
        ).read_text()
        assert "SECOND RUN OUTPUT" in second_text
        assert second_text != first_text


def test_qa_cost_summary_leads_with_walltime(tmp_path):
    """qa.py::_cost_summary_text must lead with wall-time per task + per-tool
    minutes (mirrors T6's report.py / retro.py treatment). Dollars are a
    footnote.
    """
    from orch.qa import _cost_summary_text

    log_dir = tmp_path / "tools" / "logs" / "test-iter"
    log_dir.mkdir(parents=True)
    cost_path = log_dir / "cost.jsonl"

    # Two records: one impl call (codex), one review call (claude).
    cost_path.write_text(
        '{"ts": "2026-04-30T10:00:00Z", "iteration": "test-iter", "task": "T1", '
        '"step": "IMPL", "agent": "codex", "duration_s": 245.0, "est_cost_usd": 0.012}\n'
        '{"ts": "2026-04-30T10:05:00Z", "iteration": "test-iter", "task": "T1", '
        '"step": "REVIEW", "agent": "claude", "duration_s": 60.0, "est_cost_usd": 0.005}\n'
    )

    out = _cost_summary_text(cost_path)

    # Wall-time content must precede dollars
    walltime_idx = out.find("Wall time")
    dollar_idx = out.find("$")
    assert walltime_idx >= 0, "wall-time table missing"
    assert dollar_idx >= 0, "dollar footnote missing"
    assert walltime_idx < dollar_idx, (
        "dollars must appear AFTER the wall-time table; current output: "
        + out
    )
    # Caveat preserved (any of the canonical phrasings)
    assert (
        "estimated" in out.lower()
        or "verify against" in out.lower()
    ), f"cost-estimate caveat missing; current output: {out}"
    assert (
        "estimated equivalent API cost (subscription — not billed per request)"
        in out
    )
    # Wall-time minutes-format present (e.g. "4:05")
    import re
    assert re.search(r"\b\d+:\d{2}\b", out), (
        "no MM:SS wall-time formatting found in output: " + out
    )


class TestDiffBaseFailClosed:
    """Objective 6 — a non-existent diff base must fail closed, not let QA
    review a "(diff unavailable)" placeholder and pass vacuously."""

    def test_run_qa_raises_on_nonexistent_diff_base(self, repo: Path):
        ghost = TASKS_MD_WITH_DIFF_BASE.replace("`main`", "`ghost-ref-zzz`")
        cfg, board, store, cost = _make_deps(repo, ghost)
        adapter = FakeAdapter(name="claude", family="anthropic")
        with pytest.raises(QaDiffBaseError) as exc:
            run_qa(
                cfg=cfg, board=board, state=store, cost=cost,
                adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
                reviewer_agent="claude",
            )
        assert "ghost-ref-zzz" in str(exc.value)
        # Reviewers must never run against a non-existent diff.
        assert adapter.calls == []

    def test_run_qa_runs_when_iteration_branch_ahead_of_base(self, repo: Path):
        # demo/iteration-1 is ahead of phase-demo in the fixture, so QA runs.
        cfg, board, store, cost = _make_deps(repo)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
        )
        assert len(report.roles) == 5
        assert adapter.calls

    def test_run_qa_raises_when_not_ahead_of_base_without_override(
        self, repo: Path, capsys
    ):
        # In the repo fixture, `main` and `demo/iteration-1` are both at the
        # "project files" commit, so with diff base = `main` the iteration
        # branch is NOT strictly ahead: ahead_count == 0 (the post-merge /
        # vacuous re-run shape). QA must fail closed by default so reviewers
        # cannot pass an empty diff.
        cfg, board, store, cost = _make_deps(repo, TASKS_MD_WITH_DIFF_BASE)
        adapter = FakeAdapter(name="claude", family="anthropic")
        with pytest.raises(QaEmptyDiffError, match="empty/vacuous"):
            run_qa(
                cfg=cfg, board=board, state=store, cost=cost,
                adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
                reviewer_agent="claude",
            )
        captured = capsys.readouterr()
        assert captured.err == ""
        assert adapter.calls == []

    def test_run_qa_allows_empty_diff_with_explicit_reason(
        self, repo: Path, capsys
    ):
        cfg, board, store, cost = _make_deps(repo, TASKS_MD_WITH_DIFF_BASE)
        adapter = FakeAdapter(name="claude", family="anthropic")
        report = run_qa(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters={"claude": adapter}, iteration="demo-i1", cwd=repo,
            reviewer_agent="claude",
            allow_empty_diff_reason="post-merge audit",
        )
        captured = capsys.readouterr()
        assert "not ahead of diff base" in captured.err
        assert "empty/vacuous" in captured.err
        assert "post-merge audit" in captured.err
        assert len(report.roles) == 5
        assert adapter.calls


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
                "## Summary\nAll good.\n\n"
                "## Findings\n- None\n\n"
                "## Verdict\nPASS"
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


def test_qa_empty_config_passes_empty_routing_options_to_role_and_synthesis(
    repo: Path,
):
    cfg, board, store, cost = _make_deps(repo)
    adapter = FakeAdapter(name="codex", family="openai")

    report = run_qa(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"codex": adapter},
        iteration="demo-i1",
        cwd=repo,
        reviewer_agent="codex",
        roles=["security"],
        synthesize=True,
    )

    assert len(report.roles) == 1
    assert report.synthesis is not None
    assert len(adapter.calls) == 2
    for call in adapter.calls:
        assert call["routing_options"].args == ()
        assert call["routing_options"].env == {}


def test_qa_quality_gate_routing_reaches_role_and_synthesis_invocations(
    repo: Path,
):
    _append_quality_gate_routing_config(repo)
    cfg, board, store, cost = _make_deps(repo)
    adapter = FakeAdapter(name="codex", family="openai")

    run_qa(
        cfg=cfg,
        board=board,
        state=store,
        cost=cost,
        adapters={"codex": adapter},
        iteration="demo-i1",
        cwd=repo,
        reviewer_agent="codex",
        roles=["security"],
        synthesize=True,
    )

    expected_args = (
        "-m",
        "fixture-max",
        "-c",
        "model_reasoning_effort=xhigh",
    )
    assert len(adapter.calls) == 2
    for call in adapter.calls:
        assert call["routing_options"].args == expected_args
        assert call["routing_options"].env == {}
