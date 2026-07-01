"""Tests for the pure team-spawn primitive."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import textwrap

import pytest

from orch import teams


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def happy_writer(tmp_path: Path) -> Path:
    return _write_executable(
        tmp_path / "happy-agent.py",
        f"""\
        #!{sys.executable}
        import json
        import os
        from pathlib import Path
        import sys
        import time

        prompt = sys.stdin.read()
        delay = 0.0
        for token in prompt.split():
            if token.startswith("DELAY="):
                delay = float(token.split("=", 1)[1])
        if delay:
            time.sleep(delay)

        paths = [Path(raw) for raw in json.loads(os.environ["ORCH_TEAM_RESULT_PATHS"])]
        if "OMIT_LAST" in prompt:
            paths = paths[:-1]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"artifact from {{os.environ['ORCH_TEAM_AGENT_NAME']}}\\n",
                encoding="utf-8",
            )
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


def _team(name: str, command: Path, *, timeout: int = 2) -> teams.Team:
    team = teams.create_team(name)
    team.command = (str(command),)
    team.stuck_timeout_s = timeout
    return team


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_public_surface_signatures_and_constant():
    assert list(inspect.signature(teams.create_team).parameters) == ["name"]
    assert list(inspect.signature(teams.spawn_agent).parameters) == [
        "team",
        "name",
        "prompt",
        "cwd",
        "result_paths",
    ]
    assert list(inspect.signature(teams.wait_for_results).parameters) == [
        "paths",
        "timeout",
    ]
    assert list(inspect.signature(teams.kill_team).parameters) == ["team"]
    assert teams.STUCK_AGENT_TIMEOUT_S == 120


def test_concurrent_happy_wave_returns_all_completed(happy_writer: Path, tmp_path: Path):
    result_dir = tmp_path / "wave-results"
    team = _team("happy-wave", happy_writer, timeout=5)
    paths_a = (result_dir / "agent-a" / "result.md",)
    paths_b = (result_dir / "agent-b" / "result.md",)

    try:
        agent_a = teams.spawn_agent(team, "agent-a", "DELAY=0.25", tmp_path, paths_a)
        agent_b = teams.spawn_agent(team, "agent-b", "DELAY=0.25", tmp_path, paths_b)

        assert len(team.agents) == 2
        assert not (agent_a.future.done() and agent_b.future.done())

        results = teams.wait_for_results([*paths_a, *paths_b], timeout=5)

        assert set(results) == {"agent-a", "agent-b"}
        assert {result.status for result in results.values()} == {"completed"}
        assert all(result.exit_code == 0 for result in results.values())
        assert all(result.timed_out is False for result in results.values())
        assert all(result.killed is False for result in results.values())
        assert paths_a[0].read_text(encoding="utf-8") == "artifact from agent-a\n"
        assert paths_b[0].read_text(encoding="utf-8") == "artifact from agent-b\n"
        teams.kill_team(team)
        teams.kill_team(team)
        with pytest.raises(RuntimeError):
            teams.spawn_agent(team, "late", "x", tmp_path, (result_dir / "late.md",))
    finally:
        teams.kill_team(team)


def test_contract_violation_reports_missing_path(happy_writer: Path, tmp_path: Path):
    result_dir = tmp_path / "contract"
    team = _team("contract", happy_writer, timeout=3)
    paths = (
        result_dir / "result.md",
        result_dir / "transcript.md",
    )

    try:
        teams.spawn_agent(team, "writer", "OMIT_LAST", tmp_path, paths)
        results = teams.wait_for_results(paths, timeout=3)

        result = results["writer"]
        assert result.status == "missing"
        assert result.exit_code == 0
        assert result.missing_paths == (paths[1].resolve(strict=False),)
        assert teams.contract_ok(result_dir, ["result.md", "transcript.md"]) is False
        (result_dir / "empty.md").write_text("", encoding="utf-8")
        assert teams.contract_ok(result_dir, ["result.md", "empty.md"]) is False
    finally:
        teams.kill_team(team)


def test_hung_agent_is_killed_and_reported(stuck_agent: Path, tmp_path: Path):
    result_path = tmp_path / "hung" / "result.md"
    team = _team("hung", stuck_agent, timeout=1)

    try:
        agent = teams.spawn_agent(team, "stuck", "never writes", tmp_path, (result_path,))
        results = teams.wait_for_results((result_path,), timeout=5)

        result = results["stuck"]
        process_result = agent.future.result()
        assert process_result.partial is True
        assert process_result.exit_code != 0
        assert process_result.duration_s < 10
        assert result.status == "killed"
        assert result.timed_out is True
        assert result.killed is True
        assert result.exit_code != 0
        assert result.missing_paths == (result_path.resolve(strict=False),)
    finally:
        teams.kill_team(team)


def test_multi_agent_kill_isolation(
    happy_writer: Path,
    stuck_agent: Path,
    tmp_path: Path,
):
    result_dir = tmp_path / "mixed"
    team = _team("mixed", happy_writer, timeout=5)
    happy_a = (result_dir / "happy-a" / "result.md",)
    happy_b = (result_dir / "happy-b" / "result.md",)
    stuck = (result_dir / "stuck" / "result.md",)

    try:
        teams.spawn_agent(team, "happy-a", "DELAY=0.1", tmp_path, happy_a)
        teams.spawn_agent(team, "happy-b", "DELAY=0.1", tmp_path, happy_b)
        team.command = (str(stuck_agent),)
        team.stuck_timeout_s = 1
        teams.spawn_agent(team, "stuck", "never writes", tmp_path, stuck)

        results = teams.wait_for_results([*happy_a, *happy_b, *stuck], timeout=5)

        assert results["happy-a"].status == "completed"
        assert results["happy-b"].status == "completed"
        assert results["stuck"].status == "killed"
        assert results["stuck"].timed_out is True
        assert results["stuck"].killed is True
        assert happy_a[0].exists()
        assert happy_b[0].exists()
        assert not stuck[0].exists()
    finally:
        teams.kill_team(team)


def test_purity_guard_and_default_off_imports():
    source_path = Path(teams.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    import_lines = [
        line
        for line in source.splitlines()
        if line.startswith(("from ", "import "))
    ]
    violations = [
        line
        for line in import_lines
        if "orch" in line and "providers" not in line
    ]

    assert violations == []
    assert "orch.state" not in source
    assert "orch.timing" not in source
    assert "run_state" not in source
    assert "os.killpg" not in source

    orch_dir = source_path.parent
    importers = []
    for path in orch_dir.glob("*.py"):
        if path == source_path:
            continue
        text = path.read_text(encoding="utf-8")
        if "import teams" in text or "import orch.teams" in text:
            importers.append(path.name)
    assert sorted(importers) == ["planning_team.py", "team_mode.py"]


def test_timing_jsonl_lands_in_caller_dir_without_cost_fields(
    happy_writer: Path,
    tmp_path: Path,
):
    result_dir = tmp_path / "timing"
    paths_a = (result_dir / "a-result.md",)
    paths_b = (result_dir / "b-result.md",)
    team = _team("timing", happy_writer, timeout=3)

    try:
        teams.spawn_agent(team, "a", "DELAY=0.05", tmp_path, paths_a)
        teams.spawn_agent(team, "b", "DELAY=0.05", tmp_path, paths_b)
        results = teams.wait_for_results([*paths_a, *paths_b], timeout=5)

        assert {result.status for result in results.values()} == {"completed"}
        timing_path = result_dir / "timing.jsonl"
        assert timing_path.exists()
        assert timing_path.parent == result_dir
        assert "tools" not in timing_path.parts or "logs" not in timing_path.parts

        records = _records(timing_path)
        assert len(records) == 4
        by_agent = {agent: [] for agent in ("a", "b")}
        for record in records:
            by_agent[record["agent"]].append(record)
            assert "cost" not in record
            assert "first_tool_use" not in record
        assert {
            agent: [record["kind"] for record in agent_records]
            for agent, agent_records in by_agent.items()
        } == {
            "a": ["spawn", "completion"],
            "b": ["spawn", "completion"],
        }
    finally:
        teams.kill_team(team)


class _SlowImmediateExecutor:
    def __init__(self, *, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s
        self.shutdown_calls = 0

    def submit(self, fn, *args):
        import concurrent.futures
        import time

        time.sleep(self.delay_s)
        agent_name = args[1]
        result_paths = args[4]
        for path in result_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"artifact from {agent_name}\n", encoding="utf-8")
        future = concurrent.futures.Future()
        future.set_result(
            teams.providers.ManagedProcessResult(
                exit_code=0,
                stdout="",
                stderr="",
                duration_s=self.delay_s,
                partial=False,
            )
        )
        return future

    def shutdown(self, *, wait=True, cancel_futures=False):
        self.shutdown_calls += 1


def _registry_owner(path: Path):
    with teams._REGISTRY_LOCK:
        return teams._AGENTS_BY_RESULT_PATH.get(path.resolve(strict=False))


def test_concurrent_same_path_spawn_registers_exactly_one_agent(tmp_path: Path):
    import concurrent.futures
    import threading

    result_path = tmp_path / "race" / "result.md"
    team = teams.create_team("registry-race")
    team.executor = _SlowImmediateExecutor(delay_s=0.05)
    workers = 8
    start = threading.Barrier(workers)

    def attempt(index: int):
        start.wait()
        try:
            agent = teams.spawn_agent(
                team,
                f"agent-{index}",
                "write",
                tmp_path,
                (result_path,),
            )
            return ("ok", agent.name)
        except ValueError as exc:
            return ("error", str(exc))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(attempt, range(workers)))

        winners = [value for status, value in outcomes if status == "ok"]
        errors = [value for status, value in outcomes if status == "error"]
        assert len(winners) == 1
        assert len(errors) == workers - 1
        assert all("result_paths already registered" in error for error in errors)

        winner = winners[0]
        assert _registry_owner(result_path) is team.agents[winner]

        results = teams.wait_for_results((result_path,), timeout=1)
        assert set(results) == {winner}
        assert results[winner].status == "completed"
        assert _registry_owner(result_path) is None
    finally:
        teams.kill_team(team)


def test_wait_for_results_deregisters_terminal_agents_without_kill_team(
    happy_writer: Path,
    tmp_path: Path,
):
    result_dir = tmp_path / "collect-cleanup"
    paths_a = (result_dir / "a" / "result.md",)
    paths_b = (result_dir / "b" / "result.md",)
    team = _team("collect-cleanup", happy_writer, timeout=3)

    try:
        agent_a = teams.spawn_agent(team, "a", "DELAY=0.05", tmp_path, paths_a)
        agent_b = teams.spawn_agent(team, "b", "DELAY=0.05", tmp_path, paths_b)
        assert _registry_owner(paths_a[0]) is agent_a
        assert _registry_owner(paths_b[0]) is agent_b

        results = teams.wait_for_results([*paths_a, *paths_b], timeout=5)

        assert {result.status for result in results.values()} == {"completed"}
        assert _registry_owner(paths_a[0]) is None
        assert _registry_owner(paths_b[0]) is None
    finally:
        teams.kill_team(team)


def test_wait_for_results_keeps_running_agents_registered_until_kill(
    stuck_agent: Path,
    tmp_path: Path,
):
    result_path = tmp_path / "still-running" / "result.md"
    team = _team("still-running", stuck_agent, timeout=1)

    try:
        agent = teams.spawn_agent(team, "stuck", "never writes", tmp_path, (result_path,))
        results = teams.wait_for_results((result_path,), timeout=0.1)

        assert results["stuck"].status == "timed_out"
        assert results["stuck"].timed_out is True
        assert results["stuck"].killed is False
        assert agent.future.done() is False
        assert _registry_owner(result_path) is agent
    finally:
        teams.kill_team(team)

    assert _registry_owner(result_path) is None


def test_kill_team_is_idempotent_after_collect_cleanup(
    happy_writer: Path,
    tmp_path: Path,
):
    result_path = tmp_path / "idempotent" / "result.md"
    team = _team("idempotent", happy_writer, timeout=3)

    teams.spawn_agent(team, "writer", "DELAY=0.05", tmp_path, (result_path,))
    results = teams.wait_for_results((result_path,), timeout=5)

    assert results["writer"].status == "completed"
    assert _registry_owner(result_path) is None

    teams.kill_team(team)
    teams.kill_team(team)

    assert _registry_owner(result_path) is None
