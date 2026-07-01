"""Read-only Agent Team helpers for retro/QA style orchestrator reports."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from orch import teams as _teams
from orch.agents.base import dispatched_model_from_argv_env
from orch.cost import CostLogger, estimate_tokens, parse_agent_usage


_DEFAULT_REQUIRED_ARTIFACTS = (
    "transcript.md",
    "verdict.txt",
    "tests.txt",
    "files.txt",
    "blockers.txt",
)
_MAY_BE_EMPTY = {"blockers.txt"}


class TeamModeArtifactError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: str = "malformed",
        missing_artifacts: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.status = status
        self.missing_artifacts = tuple(missing_artifacts)


@dataclass(frozen=True)
class ReadOnlyTeamRole:
    name: str
    prompt: str
    artifact_dir: Path
    result_filename: str = "result.md"
    verdict_labels: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = _DEFAULT_REQUIRED_ARTIFACTS

    def artifact_filenames(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for filename in (self.result_filename, *self.required_artifacts):
            if filename not in seen:
                seen.add(filename)
                ordered.append(filename)
        return tuple(ordered)


@dataclass(frozen=True)
class ReadOnlyTeamResult:
    role: str
    ok: bool
    status: str
    text: str
    artifact_dir: Path
    artifacts: dict[str, str] = field(default_factory=dict)
    verdict: str | None = None
    timed_out: bool = False
    killed: bool = False
    exit_code: int | None = None
    missing_artifacts: tuple[str, ...] = ()
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_exact: bool = False
    provider: str | None = None
    model: str | None = None
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    parser_status: str | None = None
    parser_warning: str | None = None
    raw_terminal_json: str | None = None
    duration_s: float = 0.0


def load_required_artifacts(role: ReadOnlyTeamRole) -> dict[str, str]:
    """Load and validate the caller-declared artifact contract for one role."""
    artifacts: dict[str, str] = {}
    missing: list[str] = []
    for filename in role.artifact_filenames():
        path = role.artifact_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        text = path.read_text(encoding="utf-8")
        if filename not in _MAY_BE_EMPTY and not text.strip():
            raise TeamModeArtifactError(
                f"{role.name}: artifact {filename} is empty",
                status="malformed",
            )
        artifacts[filename] = text
    if missing:
        raise TeamModeArtifactError(
            f"{role.name}: missing required artifact(s): {', '.join(missing)}",
            status="missing",
            missing_artifacts=missing,
        )

    verdict = artifacts.get("verdict.txt", "").strip()
    if not verdict:
        raise TeamModeArtifactError(
            f"{role.name}: verdict.txt is empty",
            status="malformed",
        )
    if role.verdict_labels and verdict not in role.verdict_labels:
        allowed = ", ".join(role.verdict_labels)
        raise TeamModeArtifactError(
            f"{role.name}: verdict.txt has {verdict!r}; expected one of {allowed}",
            status="malformed",
        )
    return artifacts


def run_read_only_team(
    *,
    team_name: str,
    roles: Sequence[ReadOnlyTeamRole],
    cwd: Path,
    command: Sequence[str],
    timeout: int,
    log_dir: Path,
    cost: CostLogger | None = None,
    agent_name: str = "team",
    family: str = "unknown",
    task: str = "READ_ONLY_TEAM",
    step: str = "READ_ONLY_TEAM",
) -> list[ReadOnlyTeamResult]:
    """Spawn roles through ``teams.py`` and load only declared artifacts."""
    active_roles = list(roles)
    if not active_roles:
        return []
    team = _teams.create_team(team_name)
    team.command = tuple(command)
    team.stuck_timeout_s = int(timeout)

    declared_paths: list[Path] = []
    try:
        for role in active_roles:
            role.artifact_dir.mkdir(parents=True, exist_ok=True)
            result_paths = tuple(
                role.artifact_dir / filename
                for filename in role.artifact_filenames()
            )
            declared_paths.extend(result_paths)
            _teams.spawn_agent(
                team,
                role.name,
                _prompt_with_artifact_contract(role),
                cwd,
                result_paths,
            )

        wait_timeout = max(1, int(timeout) + 1)
        team_results = _teams.wait_for_results(declared_paths, timeout=wait_timeout)
        provider = _usage_provider(agent_name, family)
        results = [
            _result_from_team(role, team_results.get(role.name), provider=provider)
            for role in active_roles
        ]
        record_team_mode_evidence(
            log_dir=log_dir,
            results=results,
            cost=cost,
            agent_name=agent_name,
            family=family,
            task=task,
            step=step,
            command=command,
        )
        return results
    finally:
        _teams.kill_team(team)


def record_team_mode_evidence(
    *,
    log_dir: Path,
    results: Sequence[ReadOnlyTeamResult],
    cost: CostLogger | None = None,
    agent_name: str = "team",
    family: str = "unknown",
    task: str = "READ_ONLY_TEAM",
    step: str = "READ_ONLY_TEAM",
    command: Sequence[str] = (),
) -> None:
    """Append team-mode timing and estimated cost evidence to orch logs."""
    log_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        _copy_role_timing(log_dir, result)
        if cost is None:
            continue
        extra = {
            "role": result.role,
            "team_mode": True,
            "cost_estimate": not result.tokens_exact,
            "status": result.status,
            "verdict": result.verdict,
        }
        if result.raw_terminal_json is not None:
            extra["raw_terminal_json"] = result.raw_terminal_json
        cost.record(
            task=task,
            step=step,
            agent=agent_name,
            family=family,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            exact=result.tokens_exact,
            provider=result.provider or _usage_provider(agent_name, family),
            model=result.model or dispatched_model_from_argv_env(command),
            cached_input_tokens=result.cached_input_tokens,
            cache_creation_input_tokens=result.cache_creation_input_tokens,
            reasoning_output_tokens=result.reasoning_output_tokens,
            parser_status=result.parser_status,
            parser_warning=result.parser_warning,
            duration_s=result.duration_s,
            exit_code=result.exit_code if result.exit_code is not None else -1,
            partial=result.timed_out or result.killed,
            extra=extra,
        )


def _result_from_team(
    role: ReadOnlyTeamRole,
    team_result: Any | None,
    *,
    provider: str = "",
) -> ReadOnlyTeamResult:
    input_tokens = estimate_tokens(role.prompt)
    if team_result is None:
        text = f"{role.name}: no team result was returned"
        return ReadOnlyTeamResult(
            role=role.name,
            ok=False,
            status="missing",
            text=text,
            artifact_dir=role.artifact_dir,
            error=text,
            input_tokens=input_tokens,
            output_tokens=estimate_tokens(text),
            provider=provider,
            parser_status="no_usage",
        )

    if team_result.status != "completed":
        missing = tuple(path.name for path in team_result.missing_paths)
        detail = team_result.stderr or team_result.stdout
        text = (
            f"{role.name}: team agent status={team_result.status}; "
            f"missing_artifacts={', '.join(missing) or 'none'}"
        )
        if detail:
            text = f"{text}\n\n{detail}"
        usage = _parse_team_usage(team_result, provider=provider)
        return ReadOnlyTeamResult(
            role=role.name,
            ok=False,
            status=team_result.status,
            text=text,
            artifact_dir=role.artifact_dir,
            timed_out=team_result.timed_out,
            killed=team_result.killed,
            exit_code=team_result.exit_code,
            missing_artifacts=missing,
            error=text,
            input_tokens=usage.get("input_tokens", input_tokens),
            output_tokens=usage.get("output_tokens", estimate_tokens(text)),
            tokens_exact=bool(usage.get("tokens_exact")),
            provider=usage.get("provider"),
            model=usage.get("model"),
            cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
            cache_creation_input_tokens=int(
                usage.get("cache_creation_input_tokens", 0)
            ),
            reasoning_output_tokens=int(
                usage.get("reasoning_output_tokens", 0)
            ),
            parser_status=usage.get("parser_status"),
            parser_warning=usage.get("parser_warning"),
            raw_terminal_json=usage.get("raw_terminal_json"),
            duration_s=team_result.duration_s or 0.0,
        )

    try:
        artifacts = load_required_artifacts(role)
    except TeamModeArtifactError as exc:
        text = str(exc)
        usage = _parse_team_usage(team_result, provider=provider)
        return ReadOnlyTeamResult(
            role=role.name,
            ok=False,
            status=exc.status,
            text=text,
            artifact_dir=role.artifact_dir,
            timed_out=team_result.timed_out,
            killed=team_result.killed,
            exit_code=team_result.exit_code,
            missing_artifacts=exc.missing_artifacts,
            error=text,
            input_tokens=usage.get("input_tokens", input_tokens),
            output_tokens=usage.get("output_tokens", estimate_tokens(text)),
            tokens_exact=bool(usage.get("tokens_exact")),
            provider=usage.get("provider"),
            model=usage.get("model"),
            cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
            cache_creation_input_tokens=int(
                usage.get("cache_creation_input_tokens", 0)
            ),
            reasoning_output_tokens=int(
                usage.get("reasoning_output_tokens", 0)
            ),
            parser_status=usage.get("parser_status"),
            parser_warning=usage.get("parser_warning"),
            raw_terminal_json=usage.get("raw_terminal_json"),
            duration_s=team_result.duration_s or 0.0,
        )

    text = artifacts[role.result_filename]
    verdict = artifacts.get("verdict.txt", "").strip() or None
    usage = _parse_team_usage(team_result, provider=provider)
    return ReadOnlyTeamResult(
        role=role.name,
        ok=True,
        status="completed",
        text=text,
        artifact_dir=role.artifact_dir,
        artifacts=artifacts,
        verdict=verdict,
        exit_code=team_result.exit_code,
        input_tokens=usage.get("input_tokens", input_tokens),
        output_tokens=usage.get("output_tokens", estimate_tokens(text)),
        tokens_exact=bool(usage.get("tokens_exact")),
        provider=usage.get("provider"),
        model=usage.get("model"),
        cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
        cache_creation_input_tokens=int(
            usage.get("cache_creation_input_tokens", 0)
        ),
        reasoning_output_tokens=int(usage.get("reasoning_output_tokens", 0)),
        parser_status=usage.get("parser_status"),
        parser_warning=usage.get("parser_warning"),
        raw_terminal_json=usage.get("raw_terminal_json"),
        duration_s=team_result.duration_s or 0.0,
    )


def _parse_team_usage(team_result: Any, *, provider: str) -> dict:
    raw = "\n".join(
        part for part in (
            getattr(team_result, "stdout", ""),
            getattr(team_result, "stderr", ""),
        )
        if part
    )
    if not raw.strip():
        return {"parser_status": "no_usage"}
    usage = parse_agent_usage(provider, raw)
    data: dict[str, Any] = {
        "tokens_exact": usage.exact,
        "provider": usage.provider,
        "model": usage.model,
        "parser_status": usage.parser_status,
        "parser_warning": "; ".join(usage.warnings) if usage.warnings else None,
        "raw_terminal_json": raw,
    }
    if usage.exact:
        data.update(
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "reasoning_output_tokens": usage.reasoning_output_tokens,
            }
        )
    return data


def _usage_provider(agent_name: str, family: str) -> str:
    lowered = f"{agent_name} {family}".lower()
    if "codex" in lowered or "openai" in lowered:
        return "codex"
    if "claude" in lowered or "anthropic" in lowered:
        return "claude"
    return family or agent_name


def _prompt_with_artifact_contract(role: ReadOnlyTeamRole) -> str:
    artifact_lines = "\n".join(
        f"- `{filename}` -> `{role.artifact_dir / filename}`"
        for filename in role.artifact_filenames()
    )
    verdict_line = ""
    if role.verdict_labels:
        verdict_line = (
            "Write exactly one of these labels to `verdict.txt`: "
            + " | ".join(role.verdict_labels)
            + "."
        )
    return (
        "You are running in read-only orch team mode.\n"
        "Write only the declared artifacts below; do not edit source, tests, "
        "docs, configuration, git state, or any undeclared path.\n\n"
        f"{artifact_lines}\n\n"
        "`blockers.txt` may be empty when there are no blockers. All other "
        "declared artifacts must be non-empty. "
        f"{verdict_line}\n\n"
        "---\n\n"
        f"{role.prompt}"
    )


def _copy_role_timing(log_dir: Path, result: ReadOnlyTeamResult) -> None:
    source = result.artifact_dir / "timing.jsonl"
    if not source.exists():
        return
    target = log_dir / "timing.jsonl"
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        record["team_mode"] = True
        record["role"] = result.role
        record["artifact_dir"] = str(result.artifact_dir)
        _append_jsonl(target, record)


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
