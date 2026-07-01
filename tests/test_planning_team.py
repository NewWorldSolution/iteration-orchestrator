"""Tests for docs-only planning-team mode."""
from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import pytest

from orch.git_ops import commit, git, stage_all
from orch.planning_team import (
    PlanningTeamCandidate,
    PlanningTeamRefusal,
    run_planning_team,
    validate_planning_candidate,
)


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(["init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git(["config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    git(["config", "user.name", "Tester"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("tools/logs/\nplanner-writer.py\n")
    (tmp_path / "README.md").write_text("planning team tests\n")
    stage_all(tmp_path)
    commit(tmp_path, "init")
    return tmp_path


@pytest.fixture
def planner_writer(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "planner-writer.py",
        f"""\
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import re
        import sys

        prompt = sys.stdin.read()
        cwd = Path.cwd()
        declared = re.findall(r"^- `([^`]+)`$", prompt, re.MULTILINE)
        for rel in declared:
            path = cwd / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"planned {{rel}}\\n", encoding="utf-8")
        if "WRITE_FORBIDDEN" in prompt:
            path = cwd / "app" / "forbidden.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("FORBIDDEN = True\\n", encoding="utf-8")
        if "WRITE_UNDECLARED_DOC" in prompt:
            path = cwd / "docs" / "undeclared.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("undeclared\\n", encoding="utf-8")
        result_paths = [
            Path(raw)
            for raw in json.loads(os.environ["ORCH_TEAM_RESULT_PATHS"])
        ]
        for path in result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "verdict.txt":
                text = "READY_FOR_REVIEW\\n"
            else:
                text = f"{{path.name}} for {{os.environ['ORCH_TEAM_AGENT_NAME']}}\\n"
            path.write_text(text, encoding="utf-8")
        """,
    )


@pytest.fixture
def usage_planner_writer(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "planner-writer.py",
        f"""\
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import re
        import sys

        prompt = sys.stdin.read()
        cwd = Path.cwd()
        declared = re.findall(r"^- `([^`]+)`$", prompt, re.MULTILINE)
        for rel in declared:
            path = cwd / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"planned {{rel}}\\n", encoding="utf-8")
        result_paths = [
            Path(raw)
            for raw in json.loads(os.environ["ORCH_TEAM_RESULT_PATHS"])
        ]
        for path in result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "verdict.txt":
                text = "READY_FOR_REVIEW\\n"
            else:
                text = f"{{path.name}} for {{os.environ['ORCH_TEAM_AGENT_NAME']}}\\n"
            path.write_text(text, encoding="utf-8")
        print(json.dumps({{
            "type": "result",
            "model": "claude-planning-model",
            "usage": {{
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 400,
            }},
            "result": "ignored stdout",
        }}))
        """,
    )


def _candidate(
    repo: Path,
    task_id: str = "I1-T1",
    *,
    prompt: str = "write the plan",
    allowed_files: tuple[str, ...] = ("docs/plan.md",),
) -> PlanningTeamCandidate:
    return PlanningTeamCandidate(
        task_id=task_id,
        title=f"Task {task_id}",
        prompt=prompt,
        allowed_files=allowed_files,
        artifact_dir=repo / "tools" / "logs" / "demo-i1" / "planning" / task_id,
    )


def test_planning_team_writes_declared_docs_and_iteration_files(
    repo: Path,
    planner_writer: Path,
):
    candidate = _candidate(
        repo,
        allowed_files=(
            "docs/post-phase-organization/plan.md",
            "iterations/demo-i1/prompt.md",
        ),
    )

    results = run_planning_team(
        team_name="planning-demo",
        candidates=[candidate],
        cwd=repo,
        command=(str(planner_writer),),
        timeout=3,
    )

    assert len(results) == 1
    result = results[0]
    assert result.ok is True
    assert result.status == "completed"
    assert set(result.changed_files) == set(candidate.allowed_files)
    for rel in candidate.allowed_files:
        assert (repo / rel).read_text(encoding="utf-8").startswith("planned")
    assert (candidate.artifact_dir / "result.md").exists()
    assert (candidate.artifact_dir / "transcript.md").exists()
    assert (candidate.artifact_dir / "verdict.txt").read_text() == (
        "READY_FOR_REVIEW\n"
    )
    assert (candidate.artifact_dir / "timing.jsonl").exists()


def test_planning_team_parses_terminal_usage(
    repo: Path,
    usage_planner_writer: Path,
):
    candidate = _candidate(repo)

    [result] = run_planning_team(
        team_name="planning-demo",
        candidates=[candidate],
        cwd=repo,
        command=(str(usage_planner_writer), "claude"),
        timeout=3,
    )

    assert result.ok is True
    assert result.tokens_exact is True
    assert result.provider == "claude"
    assert result.model == "claude-planning-model"
    assert result.input_tokens == 1000
    assert result.output_tokens == 200
    assert result.cached_input_tokens == 300
    assert result.cache_creation_input_tokens == 400
    assert result.parser_status == "parsed"
    assert "claude-planning-model" in (result.raw_terminal_json or "")
    assert set(result.changed_files) == set(candidate.allowed_files)


def test_planning_team_refuses_forbidden_runtime_write(
    repo: Path,
    planner_writer: Path,
):
    candidate = _candidate(repo, prompt="WRITE_FORBIDDEN")

    with pytest.raises(PlanningTeamRefusal) as exc:
        run_planning_team(
            team_name="planning-demo",
            candidates=[candidate],
            cwd=repo,
            command=(str(planner_writer),),
            timeout=3,
        )

    assert "app/forbidden.py" in str(exc.value)
    assert "path must be under one of: docs/, iterations/" in str(exc.value)


def test_planning_team_refuses_undeclared_docs_write(
    repo: Path,
    planner_writer: Path,
):
    candidate = _candidate(repo, prompt="WRITE_UNDECLARED_DOC")

    with pytest.raises(PlanningTeamRefusal) as exc:
        run_planning_team(
            team_name="planning-demo",
            candidates=[candidate],
            cwd=repo,
            command=(str(planner_writer),),
            timeout=3,
        )

    assert "docs/undeclared.md" in str(exc.value)
    assert "undeclared path" in str(exc.value)


def test_planning_team_refuses_overlapping_candidates_before_spawn(
    repo: Path,
    planner_writer: Path,
):
    first = _candidate(repo, "I1-T1", allowed_files=("docs/shared.md",))
    second = _candidate(repo, "I1-T2", allowed_files=("docs/shared.md",))

    with pytest.raises(PlanningTeamRefusal) as exc:
        run_planning_team(
            team_name="planning-demo",
            candidates=[first, second],
            cwd=repo,
            command=(str(planner_writer),),
            timeout=3,
        )

    assert "overlapping allowed file 'docs/shared.md'" in str(exc.value)
    assert not (first.artifact_dir / "result.md").exists()
    assert not (second.artifact_dir / "result.md").exists()


@pytest.mark.parametrize(
    "path",
    [
        "app/main.py",
        "db/schema.sql",
        "src/orchestrator/runner.py",
        "tests/test_app.py",
        "deploy/render.yaml",
        "seed/demo.sql",
        ".github/workflows/ci.yml",
    ],
)
def test_planning_candidate_refuses_design_blacklist_paths(
    repo: Path,
    path: str,
):
    with pytest.raises(PlanningTeamRefusal):
        validate_planning_candidate(_candidate(repo, allowed_files=(path,)))
