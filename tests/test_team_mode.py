"""Tests for read-only Agent Team helper surface."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import pytest

from orch.cost import CostLogger
from orch.team_mode import (
    ReadOnlyTeamRole,
    load_required_artifacts,
    run_read_only_team,
)


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def artifact_writer(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "artifact-writer.py",
        f"""\
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import sys

        prompt = sys.stdin.read()
        paths = [Path(raw) for raw in json.loads(os.environ["ORCH_TEAM_RESULT_PATHS"])]
        by_name = {{path.name: path for path in paths}}
        role = os.environ["ORCH_TEAM_AGENT_NAME"]
        values = {{
            "result.md": f"## Outcome for {{role}}\\n",
            "transcript.md": f"transcript for {{role}}\\n",
            "verdict.txt": "COMPLETE\\n",
            "tests.txt": "not run - mock\\n",
            "files.txt": "none\\n",
            "blockers.txt": "",
        }}
        if "BAD_VERDICT" in prompt:
            values["verdict.txt"] = "NOT_A_LABEL\\n"
        if "OMIT_VERDICT" in prompt:
            values.pop("verdict.txt")
        for name, value in values.items():
            path = by_name[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        """,
    )


@pytest.fixture
def usage_artifact_writer(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "usage-artifact-writer.py",
        f"""\
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import sys

        sys.stdin.read()
        paths = [Path(raw) for raw in json.loads(os.environ["ORCH_TEAM_RESULT_PATHS"])]
        by_name = {{path.name: path for path in paths}}
        values = {{
            "result.md": "## Outcome\\nartifact text\\n",
            "transcript.md": "transcript\\n",
            "verdict.txt": "COMPLETE\\n",
            "tests.txt": "not run - mock\\n",
            "files.txt": "none\\n",
            "blockers.txt": "",
        }}
        for name, value in values.items():
            path = by_name[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        print(json.dumps({{
            "type": "result",
            "model": "claude-team-model",
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


@pytest.fixture
def review_artifact_writer(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "review-artifact-writer.py",
        f"""\
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import sys

        sys.stdin.read()
        paths = [Path(raw) for raw in json.loads(os.environ["ORCH_TEAM_RESULT_PATHS"])]
        by_name = {{path.name: path for path in paths}}
        role = os.environ["ORCH_TEAM_AGENT_NAME"]
        values = {{
            "review.md": f"## QA Review for {{role}}\\n",
            "transcript.md": f"transcript for {{role}}\\n",
            "verdict.txt": "PASS\\n",
            "tests.txt": "not run - mock\\n",
            "files.txt": "reviewed diff only\\n",
            "blockers.txt": "",
        }}
        for name, value in values.items():
            path = by_name[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        """,
    )


@pytest.fixture
def stuck_agent(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "stuck-agent.py",
        f"""\
        #!{sys.executable}
        import time

        time.sleep(30)
        """,
    )


def _role(name: str, root: Path, prompt: str = "write artifacts") -> ReadOnlyTeamRole:
    return ReadOnlyTeamRole(
        name=name,
        prompt=prompt,
        artifact_dir=root / name,
        verdict_labels=("COMPLETE", "COMPLETE_WITH_FOLLOWUPS", "INCOMPLETE"),
    )


def _cost_logger(path: Path) -> CostLogger:
    return CostLogger(
        path=path,
        cost_table={"anthropic": {"input": 3.0, "output": 15.0}},
        iteration="demo-i1",
    )


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_run_read_only_team_loads_disk_artifacts_and_records_evidence(
    artifact_writer: Path,
    tmp_path: Path,
):
    log_dir = tmp_path / "tools" / "logs" / "demo-i1"
    roles = [_role(name, log_dir / "retro" / "team") for name in ("a", "b", "c")]
    cost = _cost_logger(log_dir / "cost.jsonl")

    results = run_read_only_team(
        team_name="retro-demo",
        roles=roles,
        cwd=tmp_path,
        command=(str(artifact_writer),),
        timeout=3,
        log_dir=log_dir,
        cost=cost,
        agent_name="claude",
        family="anthropic",
        task="RETRO",
        step="RETRO_TEAM",
    )

    assert [result.role for result in results] == ["a", "b", "c"]
    assert all(result.ok for result in results)
    assert all(result.verdict == "COMPLETE" for result in results)
    assert all(result.text.startswith("## Outcome") for result in results)

    timing_records = _jsonl(log_dir / "timing.jsonl")
    assert len(timing_records) == 6
    assert {record["kind"] for record in timing_records} == {"spawn", "completion"}
    assert {record["role"] for record in timing_records} == {"a", "b", "c"}
    assert all(record["team_mode"] is True for record in timing_records)

    cost_records = _jsonl(log_dir / "cost.jsonl")
    assert len(cost_records) == 3
    assert all(record["step"] == "RETRO_TEAM" for record in cost_records)
    assert all(record["extra"]["cost_estimate"] is True for record in cost_records)
    assert all(record["extra"]["team_mode"] is True for record in cost_records)


def test_run_read_only_team_supports_review_artifact_filename(
    review_artifact_writer: Path,
    tmp_path: Path,
):
    log_dir = tmp_path / "tools" / "logs" / "demo-i1"
    role = ReadOnlyTeamRole(
        name="security",
        prompt="write QA review artifacts",
        artifact_dir=log_dir / "qa" / "team" / "security",
        result_filename="review.md",
        verdict_labels=("PASS", "CHANGES_REQUIRED", "BLOCKED"),
    )
    cost = _cost_logger(log_dir / "cost.jsonl")

    results = run_read_only_team(
        team_name="qa-demo",
        roles=[role],
        cwd=tmp_path,
        command=(str(review_artifact_writer),),
        timeout=3,
        log_dir=log_dir,
        cost=cost,
        agent_name="claude",
        family="anthropic",
        task="QA",
        step="QA_TEAM",
    )

    result = results[0]
    assert result.ok is True
    assert result.text.startswith("## QA Review")
    assert result.verdict == "PASS"
    assert (role.artifact_dir / "review.md").exists()
    assert not (role.artifact_dir / "result.md").exists()

    timing_records = _jsonl(log_dir / "timing.jsonl")
    assert {record["kind"] for record in timing_records} == {
        "spawn",
        "completion",
    }
    assert all(record["role"] == "security" for record in timing_records)
    assert all(record["team_mode"] is True for record in timing_records)

    cost_records = _jsonl(log_dir / "cost.jsonl")
    assert len(cost_records) == 1
    assert cost_records[0]["task"] == "QA"
    assert cost_records[0]["step"] == "QA_TEAM"
    assert cost_records[0]["extra"]["role"] == "security"
    assert cost_records[0]["extra"]["cost_estimate"] is True


def test_run_read_only_team_parses_terminal_usage_without_changing_artifacts(
    usage_artifact_writer: Path,
    tmp_path: Path,
):
    log_dir = tmp_path / "tools" / "logs" / "demo-i1"
    role = _role("developer", log_dir / "retro" / "team")
    cost = _cost_logger(log_dir / "cost.jsonl")

    results = run_read_only_team(
        team_name="retro-demo",
        roles=[role],
        cwd=tmp_path,
        command=(str(usage_artifact_writer),),
        timeout=3,
        log_dir=log_dir,
        cost=cost,
        agent_name="claude",
        family="anthropic",
        task="RETRO",
        step="RETRO_TEAM",
    )

    result = results[0]
    assert result.ok is True
    assert result.text == "## Outcome\nartifact text\n"
    assert result.tokens_exact is True
    assert result.provider == "claude"
    assert result.model == "claude-team-model"
    assert result.input_tokens == 1000
    assert result.output_tokens == 200
    assert result.cached_input_tokens == 300
    assert result.cache_creation_input_tokens == 400
    assert result.parser_status == "parsed"
    assert "claude-team-model" in (result.raw_terminal_json or "")

    cost_records = _jsonl(log_dir / "cost.jsonl")
    assert len(cost_records) == 1
    rec = cost_records[0]
    assert rec["estimated"] is False
    assert rec["provider"] == "claude"
    assert rec["model"] == "claude-team-model"
    assert rec["cached_input_tokens"] == 300
    assert rec["cache_creation_input_tokens"] == 400
    assert rec["parser_status"] == "parsed"
    assert rec["extra"]["cost_estimate"] is False
    # Raw CLI dump is stripped from the persisted record; it stays on the
    # in-memory result (asserted above) for live diagnostics.
    assert "raw_terminal_json" not in rec["extra"]


def test_load_required_artifacts_rejects_bad_verdict(tmp_path: Path):
    role = _role("developer", tmp_path)
    role.artifact_dir.mkdir(parents=True)
    for filename in role.artifact_filenames():
        text = "COMPLETE\n" if filename == "verdict.txt" else "x\n"
        if filename == "blockers.txt":
            text = ""
        (role.artifact_dir / filename).write_text(text, encoding="utf-8")
    (role.artifact_dir / "verdict.txt").write_text("BROKEN\n", encoding="utf-8")

    with pytest.raises(Exception, match="expected one of"):
        load_required_artifacts(role)


def test_run_read_only_team_fails_closed_on_missing_artifact(
    artifact_writer: Path,
    tmp_path: Path,
):
    log_dir = tmp_path / "tools" / "logs" / "demo-i1"
    role = _role("developer", log_dir / "retro" / "team", prompt="OMIT_VERDICT")

    results = run_read_only_team(
        team_name="retro-demo",
        roles=[role],
        cwd=tmp_path,
        command=(str(artifact_writer),),
        timeout=3,
        log_dir=log_dir,
        agent_name="claude",
        family="anthropic",
        task="RETRO",
        step="RETRO_TEAM",
    )

    assert len(results) == 1
    assert results[0].ok is False
    assert results[0].status == "missing"
    assert "verdict.txt" in results[0].missing_artifacts


def test_run_read_only_team_reports_stuck_agent_as_killed(
    stuck_agent: Path,
    tmp_path: Path,
):
    log_dir = tmp_path / "tools" / "logs" / "demo-i1"
    role = _role("developer", log_dir / "retro" / "team")
    cost = _cost_logger(log_dir / "cost.jsonl")

    results = run_read_only_team(
        team_name="retro-demo",
        roles=[role],
        cwd=tmp_path,
        command=(str(stuck_agent),),
        timeout=1,
        log_dir=log_dir,
        cost=cost,
        agent_name="claude",
        family="anthropic",
        task="RETRO",
        step="RETRO_TEAM",
    )

    result = results[0]
    assert result.ok is False
    assert result.status == "killed"
    assert result.timed_out is True
    assert result.killed is True
    assert result.exit_code != 0
    assert not (role.artifact_dir / "result.md").exists()

    cost_records = _jsonl(log_dir / "cost.jsonl")
    assert cost_records[0]["partial"] is True
    assert cost_records[0]["extra"]["cost_estimate"] is True
