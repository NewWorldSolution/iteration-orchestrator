"""Tests for orch.git_ops using a real throwaway git repo."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from orch.git_ops import (
    BranchFreshnessCondition,
    GitError,
    WorktreePreflightError,
    branch_exists,
    classify_branch_freshness,
    checkout,
    cleanup_orch_workdir,
    commit,
    create_or_reset_branch,
    current_branch,
    current_sha,
    diff_files,
    diff_stats,
    diff_text,
    ensure_orch_workdir,
    git,
    orch_workdir,
    render_branch_freshness_recovery,
    revert_paths,
    salvage_worktree,
    stage_all,
    working_tree_clean,
)


class RecordingGitProvider:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.calls: list[dict] = []
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def run(self, args, *, cwd: Path, timeout: int = 120):
        self.calls.append({"args": list(args), "cwd": cwd, "timeout": timeout})
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(["init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git(["config", "user.email", "t@e.st"], cwd=tmp_path, check=True)
    git(["config", "user.name", "Tester"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("hello\n")
    stage_all(tmp_path)
    commit(tmp_path, "init")
    return tmp_path


def test_current_sha_and_branch(repo: Path):
    sha = current_sha(repo)
    assert len(sha) == 40
    assert current_branch(repo) == "main"


def test_branch_creation_and_reset(repo: Path):
    assert not branch_exists(repo, "feat/x")
    create_or_reset_branch(repo, "feat/x", "main")
    assert branch_exists(repo, "feat/x")
    assert current_branch(repo) == "feat/x"

    # Add a commit on feat/x, then reset back to main
    (repo / "b.txt").write_text("b\n")
    stage_all(repo)
    commit(repo, "add b on feat/x")
    sha_x = current_sha(repo)

    checkout(repo, "main")
    create_or_reset_branch(repo, "feat/x", "main")
    # Now feat/x should point at main again, not sha_x
    assert current_sha(repo) != sha_x
    assert current_sha(repo) == current_sha(repo, "main")


def test_diff_files_and_stats(repo: Path):
    base_sha = current_sha(repo)
    create_or_reset_branch(repo, "task/x", "main")
    (repo / "c.txt").write_text("c1\nc2\nc3\n")
    (repo / "a.txt").write_text("hello\nworld\n")
    stage_all(repo)
    commit(repo, "edits")

    files = diff_files(repo, base_sha, "HEAD")
    assert set(files) == {"a.txt", "c.txt"}

    stats = diff_stats(repo, base_sha, "HEAD")
    assert stats.files == 2
    assert stats.insertions >= 4
    assert stats.deletions == 0


def test_diff_text_round_trip(repo: Path):
    base = current_sha(repo)
    create_or_reset_branch(repo, "task/y", "main")
    (repo / "a.txt").write_text("hello\nchanged\n")
    stage_all(repo)
    commit(repo, "change a")
    text = diff_text(repo, base)
    assert "+changed" in text


def test_revert_paths_restores_file(repo: Path):
    base = current_sha(repo)
    create_or_reset_branch(repo, "task/r", "main")
    (repo / "a.txt").write_text("TAMPERED\n")
    (repo / "outside.txt").write_text("new file\n")
    stage_all(repo)
    commit(repo, "tamper")

    # Revert only a.txt relative to base
    revert_paths(repo, ["a.txt"], base)
    # Commit the revert so diff reflects HEAD
    stage_all(repo)
    commit(repo, "revert a")
    assert (repo / "a.txt").read_text() == "hello\n"
    # outside.txt remained
    assert (repo / "outside.txt").exists()


def test_git_check_raises_on_failure(tmp_path: Path):
    with pytest.raises(GitError):
        git(["rev-parse", "HEAD"], cwd=tmp_path, check=True)


def test_git_uses_injected_provider(tmp_path: Path):
    provider = RecordingGitProvider(stdout="main\n")

    result = git(
        ["rev-parse", "--abbrev-ref", "HEAD"],
        cwd=tmp_path,
        timeout=9,
        provider=provider,
    )

    assert result.stdout == "main\n"
    assert provider.calls == [
        {
            "args": ["rev-parse", "--abbrev-ref", "HEAD"],
            "cwd": tmp_path,
            "timeout": 9,
        }
    ]


def test_ensure_orch_workdir_uses_git_provider(tmp_path: Path):
    provider = RecordingGitProvider()

    workdir = ensure_orch_workdir(
        tmp_path,
        "demo-i1",
        "demo/iteration-1",
        provider=provider,
    )

    assert workdir == tmp_path / ".orch" / "worktrees" / "demo-i1"
    assert provider.calls == [
        {
            "args": [
                "worktree",
                "add",
                str(tmp_path / ".orch" / "worktrees" / "demo-i1"),
                "demo/iteration-1",
            ],
            "cwd": tmp_path,
            "timeout": 120,
        }
    ]


def test_ensure_orch_workdir_detects_branch_checked_out_elsewhere(
    tmp_path: Path,
):
    provider = RecordingGitProvider(
        returncode=128,
        stderr=(
            "fatal: 'demo/iteration-1' is already checked out at "
            "'/tmp/other-worktree'\n"
        ),
    )

    with pytest.raises(WorktreePreflightError) as exc_info:
        ensure_orch_workdir(
            tmp_path,
            "demo-i1",
            "demo/iteration-1",
            provider=provider,
        )

    msg = str(exc_info.value)
    assert "demo-i1" in msg
    assert "demo/iteration-1" in msg
    assert "/tmp/other-worktree" in msg
    assert "git worktree list" in msg
    assert "cleanup-workdir demo-i1" in msg
    assert "raw git" not in msg.lower()


def test_ensure_orch_workdir_generic_failure_stays_git_error(tmp_path: Path):
    provider = RecordingGitProvider(
        returncode=128,
        stderr="fatal: not a valid object name: demo/iteration-1\n",
    )

    with pytest.raises(GitError) as exc_info:
        ensure_orch_workdir(
            tmp_path,
            "demo-i1",
            "demo/iteration-1",
            provider=provider,
        )

    assert not isinstance(exc_info.value, WorktreePreflightError)
    assert "git worktree add failed" in str(exc_info.value)


def test_empty_diff_stats(repo: Path):
    stats = diff_stats(repo, "HEAD", "HEAD")
    assert stats.insertions == 0 and stats.deletions == 0 and stats.files == 0


def test_orch_workdir_path_resolution(repo: Path):
    path = orch_workdir(repo, "demo-i1")
    assert path == repo / ".orch" / "worktrees" / "demo-i1"
    assert path != repo


def test_orch_workdir_uses_configured_worktree_root(repo: Path):
    assert orch_workdir(
        repo,
        "demo-i1",
        worktree_root=repo / ".orch" / "custom-worktrees",
    ) == repo / ".orch" / "custom-worktrees" / "demo-i1"
    assert orch_workdir(
        repo,
        "demo-i1",
        worktree_root=Path(".orch/custom-worktrees"),
    ) == repo / ".orch" / "custom-worktrees" / "demo-i1"


def test_ensure_orch_workdir_uses_configured_worktree_root(tmp_path: Path):
    provider = RecordingGitProvider()
    custom_root = tmp_path / ".orch" / "custom-worktrees"

    workdir = ensure_orch_workdir(
        tmp_path,
        "demo-i1",
        "demo/iteration-1",
        provider=provider,
        worktree_root=custom_root,
    )

    assert workdir == custom_root / "demo-i1"
    assert provider.calls[0]["args"] == [
        "worktree",
        "add",
        str(custom_root / "demo-i1"),
        "demo/iteration-1",
    ]


def test_ensure_orch_workdir_is_idempotent(repo: Path):
    git(["branch", "demo/iteration-1"], cwd=repo, check=True)
    first = ensure_orch_workdir(repo, "demo-i1", "demo/iteration-1")
    second = ensure_orch_workdir(repo, "demo-i1", "demo/iteration-1")
    assert first == second
    assert first.exists()
    assert current_branch(first) == "demo/iteration-1"


def test_cleanup_orch_workdir_preserves_branch(repo: Path):
    git(["branch", "demo/iteration-1"], cwd=repo, check=True)
    workdir = ensure_orch_workdir(repo, "demo-i1", "demo/iteration-1")
    checkout(workdir, "demo/iteration-1")
    (workdir / "inside.txt").write_text("hello\n")
    stage_all(workdir)
    commit(workdir, "worktree commit")
    head_before = current_sha(workdir)

    cleanup_orch_workdir(repo, "demo-i1")

    assert not workdir.exists()
    assert current_sha(repo, "demo/iteration-1") == head_before


def test_salvage_worktree_preserves_work_and_restores_clean_tree(repo: Path):
    original_branch = current_branch(repo)
    original_sha = current_sha(repo)
    (repo / "a.txt").write_text("hello\ntracked change\n")
    (repo / "untracked.txt").write_text("new file\n")

    sha = salvage_worktree(
        repo,
        "salvage/demo-i1/I1-T1",
        "orch salvage: I1-T1 uncommitted work at terminal stop",
    )

    assert sha is not None
    assert current_branch(repo) == original_branch
    assert current_sha(repo, original_branch) == original_sha
    assert working_tree_clean(repo)
    assert branch_exists(repo, "salvage/demo-i1/I1-T1")
    assert current_sha(repo, "salvage/demo-i1/I1-T1") == sha
    assert git(
        ["show", "salvage/demo-i1/I1-T1:a.txt"],
        cwd=repo,
        check=True,
    ).stdout == "hello\ntracked change\n"
    assert git(
        ["show", "salvage/demo-i1/I1-T1:untracked.txt"],
        cwd=repo,
        check=True,
    ).stdout == "new file\n"

    assert salvage_worktree(repo, "salvage/demo-i1/I1-T1-clean", "clean") is None
    assert not branch_exists(repo, "salvage/demo-i1/I1-T1-clean")
    assert current_branch(repo) == original_branch


def test_branch_freshness_equal_branch_is_fresh(repo: Path):
    result = classify_branch_freshness(repo, branch="main", base_ref="main")

    assert result.condition == BranchFreshnessCondition.FRESH
    assert result.contains_base
    assert result.ahead_count == 0
    assert result.behind_count == 0


def test_branch_freshness_ahead_branch_contains_base(repo: Path):
    create_or_reset_branch(repo, "feat/ahead", "main")
    (repo / "ahead.txt").write_text("ahead\n")
    stage_all(repo)
    commit(repo, "ahead")

    result = classify_branch_freshness(repo, branch="feat/ahead", base_ref="main")

    assert result.condition == BranchFreshnessCondition.AHEAD
    assert result.contains_base
    assert result.ahead_count == 1
    assert result.behind_count == 0


def test_branch_freshness_behind_branch_is_stale(repo: Path):
    git(["branch", "feat/behind", "main"], cwd=repo, check=True)
    (repo / "base.txt").write_text("base\n")
    stage_all(repo)
    commit(repo, "base advances")

    result = classify_branch_freshness(
        repo, branch="feat/behind", base_ref="main"
    )
    message = render_branch_freshness_recovery(result, gate="task start")

    assert result.condition == BranchFreshnessCondition.BEHIND
    assert not result.contains_base
    assert result.behind_count == 1
    assert "feat/behind" in message
    assert "main" in message
    assert "stale/behind" in message
    assert "Next action:" in message
    assert "merge" in message
    assert "reset --hard" not in message


def test_branch_freshness_diverged_branch_is_not_just_behind(repo: Path):
    git(["branch", "feat/diverged", "main"], cwd=repo, check=True)
    create_or_reset_branch(repo, "feat/diverged", "main")
    (repo / "branch.txt").write_text("branch\n")
    stage_all(repo)
    commit(repo, "branch work")
    checkout(repo, "main")
    (repo / "base.txt").write_text("base\n")
    stage_all(repo)
    commit(repo, "base work")

    result = classify_branch_freshness(
        repo, branch="feat/diverged", base_ref="main"
    )
    message = render_branch_freshness_recovery(result, gate="merge")

    assert result.condition == BranchFreshnessCondition.DIVERGED
    assert not result.contains_base
    assert result.ahead_count == 1
    assert result.behind_count == 1
    assert "diverged" in message
    assert "left/right history" in message
