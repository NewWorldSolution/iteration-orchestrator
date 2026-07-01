"""Tests for orch.paths."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from orch.config import ConfigError, load_config
from orch.paths import resolve_orch_paths
from tests.test_config import VALID_PROJECT_YAML, _write


def test_resolve_orch_paths_defaults_match_current_layout(tmp_path: Path):
    repo = tmp_path / "repo"
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))

    paths = resolve_orch_paths(repo, cfg)

    assert paths.iteration_root == repo / "iterations"
    assert paths.artifact_root == repo / "tools" / "logs"
    assert paths.worktree_root == repo / ".orch" / "worktrees"
    assert paths.project_config == repo / ".orch" / "project.yaml"
    assert paths.iteration_log_dir("demo-i1") == (
        repo / "tools" / "logs" / "demo-i1"
    )
    assert paths.artifact_ref("demo-i1", "readiness_report.md") == (
        "tools/logs/demo-i1/readiness_report.md"
    )
    assert paths.worktree_dir("demo-i1") == (
        repo / ".orch" / "worktrees" / "demo-i1"
    )
    assert paths.prompt_factory_artifact_dir("draft-1") == (
        repo / "tools" / "logs" / "prompt-factory" / "draft-1"
    )


def test_resolve_orch_paths_project_overrides(tmp_path: Path):
    repo = tmp_path / "repo"
    config = VALID_PROJECT_YAML + """\

paths:
  iteration_root: work/iterations
  artifact_root: .orch/artifacts
  worktree_root: .orch/wt
  project_config: config/orch.yaml
  task_board_filename: board.md
  iteration_prompt_filename: context.md
  task_prompts_dir: task-prompts
  task_reviews_dir: task-reviews
  prompt_factory_log_dirname: pf
  generated_artifact_exclusion_prefixes:
    - .orch/artifacts/
"""
    cfg = load_config(_write(tmp_path, config))

    paths = resolve_orch_paths(repo, cfg)
    iter_dir = repo / "work" / "iterations" / "demo"

    assert paths.iteration_root == repo / "work" / "iterations"
    assert paths.artifact_root == repo / ".orch" / "artifacts"
    assert paths.worktree_root == repo / ".orch" / "wt"
    assert paths.project_config == repo / "config" / "orch.yaml"
    assert paths.task_board_path(iter_dir) == iter_dir / "board.md"
    assert paths.prompt_path(iter_dir) == iter_dir / "context.md"
    assert paths.task_prompts_dir(iter_dir) == iter_dir / "task-prompts"
    assert paths.task_reviews_dir(iter_dir) == iter_dir / "task-reviews"
    assert paths.prompt_factory_artifact_dir("draft-1") == (
        repo / ".orch" / "artifacts" / "pf" / "draft-1"
    )
    assert paths.generated_artifact_exclusion_prefixes == (".orch/artifacts/",)
    assert paths.is_generated_artifact_path(".orch/artifacts/demo/log.txt")
    assert not paths.is_generated_artifact_path("orch/artifacts/demo/log.txt")


def test_orch_paths_generated_artifact_exclusion_uses_config(tmp_path: Path):
    repo = tmp_path / "repo"
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    paths = resolve_orch_paths(repo, cfg)

    assert paths.is_generated_artifact_path("tools/logs/demo/run_state.json")
    assert paths.is_generated_artifact_path(
        repo / "tools" / "logs" / "demo" / "run_state.json"
    )
    assert not paths.is_generated_artifact_path("tools/logs-other/demo.txt")
    assert not paths.is_generated_artifact_path("docs/report.md")
    assert not paths.is_generated_artifact_path(
        tmp_path / "outside" / "tools" / "logs" / "demo.txt"
    )


def test_default_config_preserves_existing_prompt_and_review_paths(tmp_path: Path):
    repo = tmp_path / "repo"
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    paths = resolve_orch_paths(repo, cfg)
    iter_dir = repo / "iterations" / "phase" / "demo-i1"

    assert paths.prompt_path(iter_dir) == iter_dir / "prompt.md"
    assert paths.task_board_path(iter_dir) == iter_dir / "tasks.md"
    assert paths.task_prompts_dir(iter_dir) == iter_dir / "prompts"
    assert paths.task_reviews_dir(iter_dir) == iter_dir / "reviews"


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("artifact_root", "/tmp/orch", "paths.artifact_root"),
        ("artifact_root", "../logs", "paths.artifact_root"),
        ("task_board_filename", "nested/tasks.md", "paths.task_board_filename"),
        (
            "generated_artifact_exclusion_prefixes",
            ["tools/[logs]/"],
            r"generated_artifact_exclusion_prefixes\[0\]",
        ),
        (
            "generated_artifact_exclusion_prefixes",
            ["/tmp/orch/"],
            r"generated_artifact_exclusion_prefixes\[0\]",
        ),
        (
            "generated_artifact_exclusion_prefixes",
            ["../logs/"],
            r"generated_artifact_exclusion_prefixes\[0\]",
        ),
    ],
)
def test_resolve_orch_paths_rejects_absolute_escape_or_invalid_type(
    tmp_path: Path,
    key: str,
    value,
    match: str,
):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    data = copy.deepcopy(cfg.data)
    data["paths"][key] = value

    with pytest.raises(ConfigError, match=match):
        resolve_orch_paths(tmp_path / "repo", data)


def test_resolve_orch_paths_rejects_invalid_path_value_type(tmp_path: Path):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    data = copy.deepcopy(cfg.data)
    data["paths"]["iteration_root"] = ["iterations"]

    with pytest.raises(ConfigError, match="paths.iteration_root"):
        resolve_orch_paths(tmp_path / "repo", data)
