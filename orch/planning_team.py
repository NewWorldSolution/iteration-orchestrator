"""Docs-only planning-team runner guard for Wave 7 Step 3."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from typing import Sequence

from orch import teams
from orch.agents.base import dispatched_model_from_argv_env
from orch.cost import estimate_tokens, parse_agent_usage
from orch.tasks_schema import (
    Task,
    normalize_relative_task_path,
    planning_path_refusal_reason,
)


PLANNING_REVIEW_INDEPENDENCE_POLICY = "cross_vendor"
PLANNING_RESULT_FILES = ("result.md", "transcript.md", "verdict.txt")
PLANNING_VERDICTS = ("READY_FOR_REVIEW", "NEEDS_OPERATOR_DECISION", "BLOCKED")


class PlanningTeamError(RuntimeError):
    """Base class for planning-team refusals and execution failures."""


class PlanningTeamRefusal(PlanningTeamError):
    """Raised when a candidate cannot safely use planning-team mode."""


@dataclass(frozen=True)
class PlanningTeamCandidate:
    task_id: str
    title: str
    prompt: str
    allowed_files: tuple[str, ...]
    artifact_dir: Path

    @classmethod
    def from_task(
        cls,
        task: Task,
        *,
        prompt: str,
        artifact_root: Path,
    ) -> "PlanningTeamCandidate":
        return cls(
            task_id=task.id,
            title=task.title,
            prompt=prompt,
            allowed_files=tuple(task.allowed_files),
            artifact_dir=artifact_root / task.id,
        )


@dataclass(frozen=True)
class PlanningTeamResult:
    task_id: str
    ok: bool
    status: str
    text: str
    artifact_dir: Path
    changed_files: tuple[str, ...] = ()
    missing_files: tuple[str, ...] = ()
    error: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    killed: bool = False
    duration_s: float = 0.0
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
    artifacts: dict[str, str] = field(default_factory=dict)


def validate_planning_candidate(candidate: PlanningTeamCandidate) -> None:
    """Fail closed unless every declared write is docs/iterations only."""
    if not candidate.allowed_files:
        raise PlanningTeamRefusal(
            f"{candidate.task_id}: planning-team mode requires a non-empty "
            "Allowed files block"
        )
    for raw_path in candidate.allowed_files:
        reason = planning_path_refusal_reason(raw_path)
        if reason is not None:
            raise PlanningTeamRefusal(
                f"{candidate.task_id}: allowed file {raw_path!r} refused: "
                f"{reason}"
            )


def validate_planning_wave(
    candidates: Sequence[PlanningTeamCandidate],
) -> None:
    """Validate a concurrent planning wave before any agent is spawned."""
    seen: dict[str, str] = {}
    for candidate in candidates:
        validate_planning_candidate(candidate)
        for raw_path in candidate.allowed_files:
            path = normalize_relative_task_path(raw_path)
            owner = seen.get(path)
            if owner is not None and owner != candidate.task_id:
                raise PlanningTeamRefusal(
                    "planning-team candidates have overlapping allowed "
                    f"file {path!r}: {owner} and {candidate.task_id}"
                )
            seen[path] = candidate.task_id


def has_overlapping_allowed_files(tasks: Sequence[Task]) -> bool:
    """Return True when task allowed-file sets are not disjoint."""
    seen: set[str] = set()
    for task in tasks:
        for raw_path in task.allowed_files:
            path = normalize_relative_task_path(raw_path)
            if path in seen:
                return True
            seen.add(path)
    return False


def run_planning_team(
    *,
    team_name: str,
    candidates: Sequence[PlanningTeamCandidate],
    cwd: Path,
    command: Sequence[str],
    timeout: int,
) -> list[PlanningTeamResult]:
    """Run docs/iterations planners and verify declared writes on disk."""
    active = list(candidates)
    if not active:
        return []
    validate_planning_wave(active)

    cwd = Path(cwd)
    before = _changed_paths(cwd, artifact_roots=[c.artifact_dir for c in active])
    if before:
        raise PlanningTeamRefusal(
            "planning-team mode requires a clean worktree before spawn; "
            f"found existing changes: {sorted(before)}"
        )

    team = teams.create_team(team_name)
    team.command = tuple(command)
    team.stuck_timeout_s = int(timeout)
    result_paths: list[Path] = []
    try:
        for candidate in active:
            candidate.artifact_dir.mkdir(parents=True, exist_ok=True)
            paths = tuple(candidate.artifact_dir / name for name in PLANNING_RESULT_FILES)
            result_paths.extend(paths)
            teams.spawn_agent(
                team,
                candidate.task_id,
                _prompt_with_planning_contract(candidate),
                cwd,
                paths,
            )

        raw_results = teams.wait_for_results(
            result_paths,
            timeout=max(1, int(timeout) + 1),
        )
    finally:
        teams.kill_team(team)

    changed = _changed_paths(cwd, artifact_roots=[c.artifact_dir for c in active])
    _assert_changed_files_declared(active, changed)
    return [
        _result_for_candidate(
            candidate,
            raw_results.get(candidate.task_id),
            changed,
            cwd=cwd,
            provider=_usage_provider(command),
            dispatched_model=dispatched_model_from_argv_env(command),
        )
        for candidate in active
    ]


def _prompt_with_planning_contract(candidate: PlanningTeamCandidate) -> str:
    allowed = "\n".join(f"- `{path}`" for path in candidate.allowed_files)
    artifact_lines = "\n".join(
        f"- `{filename}` -> `{candidate.artifact_dir / filename}`"
        for filename in PLANNING_RESULT_FILES
    )
    verdicts = " | ".join(PLANNING_VERDICTS)
    return (
        "You are running in docs-only orch planning-team mode.\n"
        "Write only the declared docs/iterations files for this task. Do not "
        "edit runtime source, tests, orchestrator source, deployment, migration, seed, "
        "git state, or undeclared paths.\n\n"
        "Declared task output files:\n"
        f"{allowed}\n\n"
        "Required planning-team control artifacts:\n"
        f"{artifact_lines}\n\n"
        f"Write exactly one of these labels to `verdict.txt`: {verdicts}.\n"
        "All control artifacts and declared output files must be non-empty.\n\n"
        "---\n\n"
        f"Task {candidate.task_id}: {candidate.title}\n\n"
        f"{candidate.prompt}"
    )


def _result_for_candidate(
    candidate: PlanningTeamCandidate,
    team_result,
    changed_files: set[str],
    *,
    cwd: Path,
    provider: str,
    dispatched_model: str | None = None,
) -> PlanningTeamResult:
    input_tokens = estimate_tokens(candidate.prompt)
    declared = {
        normalize_relative_task_path(item)
        for item in candidate.allowed_files
    }
    candidate_changed = tuple(
        sorted(path for path in changed_files if path in declared)
    )
    artifacts = _load_control_artifacts(candidate)
    output_text = artifacts.get("result.md", "")
    if team_result is None:
        text = f"{candidate.task_id}: no planning team result was returned"
        return PlanningTeamResult(
            task_id=candidate.task_id,
            ok=False,
            status="missing",
            text=text,
            artifact_dir=candidate.artifact_dir,
            changed_files=candidate_changed,
            missing_files=tuple(
                path for path in candidate.allowed_files if not _non_empty(cwd, path)
            ),
            error=text,
            input_tokens=input_tokens,
            output_tokens=estimate_tokens(text),
            provider=provider,
            model=dispatched_model,
            parser_status="no_usage",
        )
    if team_result.status != "completed":
        missing = tuple(path.name for path in team_result.missing_paths)
        text = (
            f"{candidate.task_id}: planning team status={team_result.status}; "
            f"missing_artifacts={', '.join(missing) or 'none'}"
        )
        detail = team_result.stderr or team_result.stdout
        if detail:
            text = f"{text}\n\n{detail}"
        usage = _parse_team_usage(team_result, provider=provider)
        return PlanningTeamResult(
            task_id=candidate.task_id,
            ok=False,
            status=team_result.status,
            text=text,
            artifact_dir=candidate.artifact_dir,
            changed_files=candidate_changed,
            missing_files=missing,
            error=text,
            exit_code=team_result.exit_code,
            timed_out=team_result.timed_out,
            killed=team_result.killed,
            duration_s=team_result.duration_s or 0.0,
            input_tokens=usage.get("input_tokens", input_tokens),
            output_tokens=usage.get("output_tokens", estimate_tokens(text)),
            tokens_exact=bool(usage.get("tokens_exact")),
            provider=usage.get("provider"),
            model=usage.get("model") or dispatched_model,
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
        )

    missing_control = tuple(
        name
        for name in PLANNING_RESULT_FILES
        if name not in artifacts or not artifacts[name].strip()
    )
    verdict = artifacts.get("verdict.txt", "").strip()
    missing_declared = tuple(
        path
        for path in candidate.allowed_files
        if not _non_empty(cwd, path)
    )
    if missing_control or missing_declared or verdict not in PLANNING_VERDICTS:
        parts: list[str] = []
        if missing_control:
            parts.append(f"missing control artifacts: {', '.join(missing_control)}")
        if missing_declared:
            parts.append(f"missing declared files: {', '.join(missing_declared)}")
        if verdict not in PLANNING_VERDICTS:
            parts.append(f"bad verdict: {verdict!r}")
        text = f"{candidate.task_id}: " + "; ".join(parts)
        usage = _parse_team_usage(team_result, provider=provider)
        return PlanningTeamResult(
            task_id=candidate.task_id,
            ok=False,
            status="malformed",
            text=text,
            artifact_dir=candidate.artifact_dir,
            artifacts=artifacts,
            changed_files=candidate_changed,
            missing_files=missing_control + missing_declared,
            error=text,
            exit_code=team_result.exit_code,
            duration_s=team_result.duration_s or 0.0,
            input_tokens=usage.get("input_tokens", input_tokens),
            output_tokens=usage.get("output_tokens", estimate_tokens(text)),
            tokens_exact=bool(usage.get("tokens_exact")),
            provider=usage.get("provider"),
            model=usage.get("model") or dispatched_model,
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
        )

    usage = _parse_team_usage(team_result, provider=provider)
    return PlanningTeamResult(
        task_id=candidate.task_id,
        ok=True,
        status="completed",
        text=output_text,
        artifact_dir=candidate.artifact_dir,
        artifacts=artifacts,
        changed_files=candidate_changed,
        exit_code=team_result.exit_code,
        duration_s=team_result.duration_s or 0.0,
        input_tokens=usage.get("input_tokens", input_tokens),
        output_tokens=usage.get("output_tokens", estimate_tokens(output_text)),
        tokens_exact=bool(usage.get("tokens_exact")),
        provider=usage.get("provider"),
        model=usage.get("model") or dispatched_model,
        cached_input_tokens=int(usage.get("cached_input_tokens", 0)),
        cache_creation_input_tokens=int(
            usage.get("cache_creation_input_tokens", 0)
        ),
        reasoning_output_tokens=int(usage.get("reasoning_output_tokens", 0)),
        parser_status=usage.get("parser_status"),
        parser_warning=usage.get("parser_warning"),
        raw_terminal_json=usage.get("raw_terminal_json"),
    )


def _parse_team_usage(team_result, *, provider: str) -> dict:
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
    data: dict = {
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


def _usage_provider(command: Sequence[str]) -> str:
    rendered = " ".join(str(part) for part in command).lower()
    if "codex" in rendered:
        return "codex"
    if "claude" in rendered:
        return "claude"
    return "unknown"


def _load_control_artifacts(candidate: PlanningTeamCandidate) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for name in PLANNING_RESULT_FILES:
        path = candidate.artifact_dir / name
        if path.exists():
            artifacts[name] = path.read_text(encoding="utf-8")
    return artifacts


def _assert_changed_files_declared(
    candidates: Sequence[PlanningTeamCandidate],
    changed_files: set[str],
) -> None:
    declared = {
        normalize_relative_task_path(path)
        for candidate in candidates
        for path in candidate.allowed_files
    }
    for path in sorted(changed_files):
        reason = planning_path_refusal_reason(path)
        if reason is not None:
            raise PlanningTeamRefusal(
                f"planning-team write refused for {path!r}: {reason}"
            )
        if path not in declared:
            raise PlanningTeamRefusal(
                f"planning-team write refused for undeclared path {path!r}"
            )


def _changed_paths(cwd: Path, *, artifact_roots: Sequence[Path]) -> set[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise PlanningTeamError(
            "planning-team mode requires a git worktree: "
            + result.stderr.strip()
        )
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if not line:
            continue
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        rel = normalize_relative_task_path(raw.strip('"'))
        if rel and not _is_under_artifact_root(cwd, rel, artifact_roots):
            paths.add(rel)
    return paths


def _is_under_artifact_root(
    cwd: Path,
    rel_path: str,
    artifact_roots: Sequence[Path],
) -> bool:
    path = (cwd / rel_path).resolve()
    for root in artifact_roots:
        try:
            path.relative_to(root.resolve())
        except ValueError:
            continue
        return True
    return False


def _non_empty(cwd: Path, rel_path: str) -> bool:
    path = cwd / normalize_relative_task_path(rel_path)
    return path.is_file() and path.stat().st_size > 0
