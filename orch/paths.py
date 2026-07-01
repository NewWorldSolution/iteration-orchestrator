"""Config-backed path resolver for orchestrator-owned filesystem roots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from orch.config import ConfigError, LoadedConfig


@dataclass(frozen=True)
class OrchPaths:
    """Resolved orchestrator paths for one repository checkout."""

    repo_root: Path
    iteration_root: Path
    artifact_root: Path
    worktree_root: Path
    project_config: Path
    task_board_filename: str
    iteration_prompt_filename: str
    task_prompts_dirname: str
    task_reviews_dirname: str
    prompt_factory_log_dirname: str
    generated_artifact_exclusion_prefixes: tuple[str, ...]

    @property
    def artifact_root_ref(self) -> str:
        return _relative_ref(self.repo_root, self.artifact_root)

    def iteration_log_dir(self, iteration: str) -> Path:
        return self.artifact_root / iteration

    def artifact_ref(self, iteration: str, name: str) -> str:
        return _relative_ref(self.repo_root, self.iteration_log_dir(iteration) / name)

    def worktree_dir(self, iteration: str) -> Path:
        return self.worktree_root / iteration

    def task_board_path(self, iter_dir: Path) -> Path:
        return Path(iter_dir) / self.task_board_filename

    def prompt_path(self, iter_dir: Path) -> Path:
        return Path(iter_dir) / self.iteration_prompt_filename

    def task_prompts_dir(self, iter_dir: Path) -> Path:
        return Path(iter_dir) / self.task_prompts_dirname

    def task_reviews_dir(self, iter_dir: Path) -> Path:
        return Path(iter_dir) / self.task_reviews_dirname

    def prompt_factory_artifact_dir(self, draft_id: str) -> Path:
        return self.artifact_root / self.prompt_factory_log_dirname / draft_id

    def is_generated_artifact_path(self, path: str | Path) -> bool:
        rel = _relative_posix(self.repo_root, path)
        if rel is None:
            return False
        return any(
            rel == prefix.rstrip("/") or rel.startswith(prefix)
            for prefix in self.generated_artifact_exclusion_prefixes
        )


def resolve_orch_paths(repo_root: Path, cfg: LoadedConfig | Mapping[str, Any]) -> OrchPaths:
    """Resolve path-related config into absolute repo-local paths."""

    data = cfg.data if isinstance(cfg, LoadedConfig) else cfg
    paths_cfg = _mapping(data, "paths")
    repo = Path(repo_root)

    return OrchPaths(
        repo_root=repo,
        iteration_root=repo / _relative_path(
            paths_cfg, "iteration_root", "paths.iteration_root"
        ),
        artifact_root=repo / _relative_path(
            paths_cfg, "artifact_root", "paths.artifact_root"
        ),
        worktree_root=repo / _relative_path(
            paths_cfg, "worktree_root", "paths.worktree_root"
        ),
        project_config=repo / _relative_path(
            paths_cfg, "project_config", "paths.project_config"
        ),
        task_board_filename=_filename(
            paths_cfg, "task_board_filename", "paths.task_board_filename"
        ),
        iteration_prompt_filename=_filename(
            paths_cfg,
            "iteration_prompt_filename",
            "paths.iteration_prompt_filename",
        ),
        task_prompts_dirname=_filename(
            paths_cfg, "task_prompts_dir", "paths.task_prompts_dir"
        ),
        task_reviews_dirname=_filename(
            paths_cfg, "task_reviews_dir", "paths.task_reviews_dir"
        ),
        prompt_factory_log_dirname=_filename(
            paths_cfg,
            "prompt_factory_log_dirname",
            "paths.prompt_factory_log_dirname",
        ),
        generated_artifact_exclusion_prefixes=_prefixes(
            paths_cfg,
            "generated_artifact_exclusion_prefixes",
            "paths.generated_artifact_exclusion_prefixes",
        ),
    )


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key}: must be a mapping")
    return value


def _relative_path(
    data: Mapping[str, Any],
    key: str,
    label: str,
) -> Path:
    value = _string(data, key, label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError(
            f"{label}: must be a relative repo-local path without '..'"
        )
    return Path(*path.parts)


def _filename(data: Mapping[str, Any], key: str, label: str) -> str:
    value = _string(data, key, label)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ConfigError(f"{label}: must be a single relative path segment")
    return value


def _string(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}: must be a non-empty string")
    return value.strip()


def _prefixes(data: Mapping[str, Any], key: str, label: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"{label}: must be a list of relative path prefixes")
    prefixes: list[str] = []
    for idx, item in enumerate(value):
        item_label = f"{label}[{idx}]"
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{item_label}: must be a non-empty string")
        if any(ch in item for ch in "*?[]"):
            raise ConfigError(f"{item_label}: must not contain glob characters")
        raw = item.strip()
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts:
            raise ConfigError(
                f"{item_label}: must be a relative repo-local prefix without '..'"
            )
        normalized = path.as_posix().rstrip("/")
        if normalized in ("", "."):
            raise ConfigError(f"{item_label}: must not resolve to the repo root")
        prefixes.append(f"{normalized}/")
    return tuple(prefixes)


def _relative_ref(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_posix(repo_root: Path, path: str | Path) -> str | None:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(repo_root)
        except ValueError:
            return None
    return candidate.as_posix()
