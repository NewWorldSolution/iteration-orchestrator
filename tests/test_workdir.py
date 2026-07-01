"""Workdir lifecycle tests for the dedicated orch sub-worktree."""
from __future__ import annotations

from pathlib import Path

from orch.git_ops import cleanup_orch_workdir, ensure_orch_workdir, git, orch_workdir


def test_repo_root_and_orch_workdir_are_distinct(tmp_path: Path):
    git(["init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git(["config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    git(["config", "user.name", "Tester"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("r\n")
    git(["add", "-A"], cwd=tmp_path, check=True)
    git(["commit", "-m", "init"], cwd=tmp_path, check=True)
    git(["branch", "demo/iteration-1"], cwd=tmp_path, check=True)

    workdir = ensure_orch_workdir(tmp_path, "demo-i1", "demo/iteration-1")

    assert tmp_path != workdir
    assert orch_workdir(tmp_path, "demo-i1") == workdir

    cleanup_orch_workdir(tmp_path, "demo-i1")
