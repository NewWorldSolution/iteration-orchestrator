"""Pure Agent Team spawn primitive for Wave 7 Step 1.

This module is intentionally default-off: no orchestrator command imports it
yet, and it does not read or write run state. Agents communicate completion
through caller-declared disk artifacts, while process spawning and timeout
termination are delegated to ``providers.SubprocessManagedProcessProvider``.

Step 1 can only observe a proxy for the design's stuck-agent rule: no declared
artifact and no captured output before the subprocess timeout. The richer
"0 tool uses + 0 tokens for 120s" signal requires real agent telemetry and is
deferred to Step 2. Likewise, ``kill_team`` cannot perform an immediate
pre-timeout external abort because the current provider exposes only a blocking
launcher; it waits for the launcher's own timeout kill path instead.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from threading import Lock

from orch import providers


STUCK_AGENT_TIMEOUT_S = 120

_DEFAULT_AGENT_COMMAND = ("claude", "-p")
_REGISTRY_LOCK = Lock()
_TIMING_LOCK = Lock()
_AGENTS_BY_RESULT_PATH: dict[Path, "Agent"] = {}


@dataclass(frozen=True)
class AgentResult:
    name: str
    status: str
    timed_out: bool
    killed: bool
    exit_code: int | None
    artifact_paths: tuple[Path, ...]
    missing_paths: tuple[Path, ...]
    stdout: str = ""
    stderr: str = ""
    duration_s: float | None = None


@dataclass(eq=False)
class Agent:
    name: str
    cwd: Path
    result_paths: tuple[Path, ...]
    future: Future[providers.ManagedProcessResult]
    timing_dir: Path


@dataclass
class Team:
    name: str
    agents: dict[str, Agent]
    executor: ThreadPoolExecutor
    command: tuple[str, ...]
    provider: providers.ManagedProcessProvider
    stuck_timeout_s: int
    accepting_agents: bool = True


def create_team(name):
    """Create a lightweight in-process team handle."""
    return Team(
        name=name,
        agents={},
        executor=ThreadPoolExecutor(thread_name_prefix=f"orch-team-{_slug(name)}"),
        command=_DEFAULT_AGENT_COMMAND,
        provider=providers.SubprocessManagedProcessProvider(),
        stuck_timeout_s=STUCK_AGENT_TIMEOUT_S,
    )


def spawn_agent(team, name, prompt, cwd, result_paths):
    """Launch one agent through the managed-process provider without blocking."""
    cwd_path = Path(cwd)
    paths = tuple(_path_key(path, cwd=cwd_path) for path in result_paths)
    if not paths:
        raise ValueError("result_paths must declare at least one artifact path")
    if len(set(paths)) != len(paths):
        raise ValueError("result_paths must not contain duplicates")
    if not team.command:
        raise ValueError("team.command must include an executable")

    timing_dir = _common_parent(paths)
    command = tuple(team.command)
    provider = team.provider
    stuck_timeout_s = int(team.stuck_timeout_s)

    with _REGISTRY_LOCK:
        if not team.accepting_agents:
            raise RuntimeError(f"team {team.name!r} is not accepting agents")
        if name in team.agents:
            raise ValueError(f"agent {name!r} already exists in team {team.name!r}")

        collisions = [
            path
            for path in paths
            if path in _AGENTS_BY_RESULT_PATH
            and _AGENTS_BY_RESULT_PATH[path].name != name
        ]
        if collisions:
            rendered = ", ".join(str(path) for path in collisions)
            raise ValueError(f"result_paths already registered: {rendered}")

        future = team.executor.submit(
            _run_agent,
            team.name,
            name,
            prompt,
            cwd_path,
            paths,
            timing_dir,
            command,
            provider,
            stuck_timeout_s,
        )
        agent = Agent(
            name=name,
            cwd=cwd_path,
            result_paths=paths,
            future=future,
            timing_dir=timing_dir,
        )
        team.agents[name] = agent
        for path in paths:
            _AGENTS_BY_RESULT_PATH[path] = agent
        return agent


def wait_for_results(paths, timeout):
    """Wait for caller-declared paths and return per-agent structured results."""
    requested_paths = tuple(_path_key(path) for path in paths)
    if not requested_paths:
        return {}

    grouped, unregistered = _agents_for_paths(requested_paths)
    deadline = time.monotonic() + timeout
    wait_timed_out = False
    known_agents = tuple(grouped)

    while True:
        paths_exist = all(path.exists() for path in requested_paths)
        futures_done = all(agent.future.done() for agent in known_agents)
        if paths_exist and futures_done:
            break
        if known_agents and futures_done:
            break
        now = time.monotonic()
        if now >= deadline:
            wait_timed_out = True
            break
        time.sleep(min(0.05, max(0.0, deadline - now)))

    results = {
        agent.name: _result_for_agent(
            agent,
            artifact_paths=tuple(agent_paths),
            wait_timed_out=wait_timed_out,
        )
        for agent, agent_paths in grouped.items()
    }
    if unregistered:
        missing = tuple(path for path in unregistered if not path.exists())
        results["<unregistered>"] = AgentResult(
            name="<unregistered>",
            status="missing",
            timed_out=wait_timed_out,
            killed=False,
            exit_code=None,
            artifact_paths=tuple(unregistered),
            missing_paths=missing,
        )
    _deregister_agents(agent for agent in grouped if agent.future.done())
    return results


def kill_team(team):
    """Close the team and wait for provider-managed timeout termination."""
    team.accepting_agents = False
    for agent in tuple(team.agents.values()):
        try:
            agent.future.result()
        except Exception:
            pass
    team.executor.shutdown(wait=True, cancel_futures=True)
    _deregister_agents(tuple(team.agents.values()))


def _deregister_agents(agents) -> None:
    """Remove terminal agent path registrations without touching new owners."""
    with _REGISTRY_LOCK:
        for agent in agents:
            for path in agent.result_paths:
                if _AGENTS_BY_RESULT_PATH.get(path) is agent:
                    del _AGENTS_BY_RESULT_PATH[path]


def contract_ok(directory, required):
    """Return True when caller-supplied required files exist and are non-empty."""
    root = Path(directory)
    for name in required:
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or path.stat().st_size == 0:
            return False
    return True


def _run_agent(
    team_name: str,
    agent_name: str,
    prompt: str,
    cwd: Path,
    result_paths: tuple[Path, ...],
    timing_dir: Path,
    command: tuple[str, ...],
    provider: providers.ManagedProcessProvider,
    stuck_timeout_s: int,
) -> providers.ManagedProcessResult:
    start = time.monotonic()
    _write_timing(
        timing_dir,
        {
            "ts": _now_iso(),
            "kind": "spawn",
            "team": team_name,
            "agent": agent_name,
            "artifact_paths": [str(path) for path in result_paths],
        },
    )
    try:
        result = provider.run(
            command,
            stdin_text=prompt,
            timeout=stuck_timeout_s,
            workdir=cwd,
            env={
                "ORCH_TEAM_NAME": team_name,
                "ORCH_TEAM_AGENT_NAME": agent_name,
                "ORCH_TEAM_RESULT_PATHS": json.dumps(
                    [str(path) for path in result_paths]
                ),
            },
        )
    except Exception as exc:
        result = providers.ManagedProcessResult(
            exit_code=-1,
            stdout="",
            stderr=str(exc),
            duration_s=time.monotonic() - start,
            partial=False,
        )
    _write_timing(
        timing_dir,
        {
            "ts": _now_iso(),
            "kind": "completion",
            "team": team_name,
            "agent": agent_name,
            "exit_code": result.exit_code,
            "duration_s": result.duration_s,
            "timed_out": result.partial,
            "killed": result.partial,
        },
    )
    return result


def _result_for_agent(
    agent: Agent,
    *,
    artifact_paths: tuple[Path, ...],
    wait_timed_out: bool,
) -> AgentResult:
    missing = tuple(path for path in artifact_paths if not path.exists())
    if not agent.future.done():
        return AgentResult(
            name=agent.name,
            status="timed_out",
            timed_out=True,
            killed=False,
            exit_code=None,
            artifact_paths=artifact_paths,
            missing_paths=missing,
        )

    process_result = agent.future.result()
    if process_result.partial:
        return AgentResult(
            name=agent.name,
            status="killed",
            timed_out=True,
            killed=True,
            exit_code=process_result.exit_code,
            artifact_paths=artifact_paths,
            missing_paths=missing,
            stdout=process_result.stdout,
            stderr=process_result.stderr,
            duration_s=process_result.duration_s,
        )
    if missing or process_result.exit_code != 0:
        return AgentResult(
            name=agent.name,
            status="missing",
            timed_out=wait_timed_out,
            killed=False,
            exit_code=process_result.exit_code,
            artifact_paths=artifact_paths,
            missing_paths=missing,
            stdout=process_result.stdout,
            stderr=process_result.stderr,
            duration_s=process_result.duration_s,
        )
    return AgentResult(
        name=agent.name,
        status="completed",
        timed_out=False,
        killed=False,
        exit_code=process_result.exit_code,
        artifact_paths=artifact_paths,
        missing_paths=(),
        stdout=process_result.stdout,
        stderr=process_result.stderr,
        duration_s=process_result.duration_s,
    )


def _agents_for_paths(
    paths: tuple[Path, ...],
) -> tuple[dict[Agent, list[Path]], list[Path]]:
    grouped: dict[Agent, list[Path]] = {}
    unregistered: list[Path] = []
    with _REGISTRY_LOCK:
        for path in paths:
            agent = _AGENTS_BY_RESULT_PATH.get(path)
            if agent is None:
                unregistered.append(path)
            else:
                grouped.setdefault(agent, []).append(path)
    return grouped, unregistered


def _common_parent(paths: tuple[Path, ...]) -> Path:
    parents = [str(path.parent) for path in paths]
    return Path(os.path.commonpath(parents))


def _path_key(path, *, cwd: Path | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = (cwd or Path.cwd()) / candidate
    return candidate.resolve(strict=False)


def _write_timing(directory: Path, record: dict) -> None:
    path = directory / "timing.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TIMING_LOCK, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in str(value)) or "team"
