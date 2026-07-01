"""Tests for tools/scaffold_lint.py — iteration scaffold consistency linter."""
from __future__ import annotations

from pathlib import Path

from orch import scaffold_lint as sl


VALID_PROJECT_YAML = """\
project:
  name: demo
  main_branch: main
  phase_branch_pattern: "phase-{phase}"
  iteration_branch_pattern: "{phase}/iteration-{n}"
  task_branch_pattern: "{phase}/i{n}/t{k}-{slug}"
stack:
  test: "echo test"
  lint: "echo lint"
risk:
  high_risk_globs: []
  sensitive_files: []
  forbidden_patterns: []
agents:
  claude:
    type: claude
    cmd: "claude -p"
    family: anthropic
  codex:
    type: codex
    cmd: "codex"
    family: openai
costs:
  anthropic: {input: 3.0, output: 15.0}
  openai: {input: 2.5, output: 10.0}
"""


def _make_iter(tmp_path: Path, prompt_text: str, tasks_text: str,
                prompts: list[str], reviews: list[str]) -> Path:
    """Helper: build an iteration directory under tmp_path."""
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    (iter_dir / "prompt.md").write_text(prompt_text)
    (iter_dir / "tasks.md").write_text(tasks_text)
    pdir = iter_dir / "prompts"
    pdir.mkdir()
    for p in prompts:
        (pdir / p).write_text("# stub\n")
    rdir = iter_dir / "reviews"
    rdir.mkdir()
    for r in reviews:
        (rdir / r).write_text("# stub\n")
    return iter_dir


def _write_project_config(tmp_path: Path, extra: str = "") -> None:
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(
        VALID_PROJECT_YAML + extra
    )


def test_scaffold_lint_passes_clean_iteration(tmp_path: Path):
    """Three tasks claimed, three rows, three prompts, three reviews → green."""
    iter_dir = _make_iter(
        tmp_path,
        prompt_text="## Three tasks\n\nstuff\n",
        tasks_text="| T1 | A | WAIT | x | n/a | — |\n| T2 | B | WAIT | x | n/a | T1 |\n| T3 | C | WAIT | x | n/a | T2 |\n",
        prompts=["t1-a.md", "t2-b.md", "t3-c.md"],
        reviews=["review-t1.md", "review-t2.md", "review-t3.md"],
    )
    errors = sl.lint_iteration(iter_dir)
    assert errors == [], f"expected clean, got: {errors}"


def test_scaffold_lint_catches_task_count_drift(tmp_path: Path):
    """The pre-i8-orch bug: prompt.md says 'Six tasks' but tasks.md has 7 rows."""
    iter_dir = _make_iter(
        tmp_path,
        prompt_text="## Six tasks\n\nstuff\n",
        tasks_text="\n".join(
            f"| T{i} | task {i} | WAIT | x | n/a | — |" for i in range(1, 8)
        ),
        prompts=[f"t{i}-x.md" for i in range(1, 8)],
        reviews=[f"review-t{i}.md" for i in range(1, 8)],
    )
    errors = sl.lint_iteration(iter_dir)
    assert any("claims 6 tasks but tasks.md has 7" in e for e in errors), errors


def test_scaffold_lint_catches_missing_review(tmp_path: Path):
    """prompts/t3-x.md exists but no reviews/review-t3.md."""
    iter_dir = _make_iter(
        tmp_path,
        prompt_text="## Three tasks\n",
        tasks_text="| T1 | A | WAIT | x | n/a | — |\n| T2 | B | WAIT | x | n/a | — |\n| T3 | C | WAIT | x | n/a | — |\n",
        prompts=["t1-a.md", "t2-b.md", "t3-c.md"],
        reviews=["review-t1.md", "review-t2.md"],   # missing review-t3
    )
    errors = sl.lint_iteration(iter_dir)
    assert any("missing matching reviews" in e or "without matching reviews" in e for e in errors), errors


def test_scaffold_lint_catches_double_hyphen_no_dep(tmp_path: Path):
    """tasks.md uses '--' instead of '—' for no-dependency."""
    iter_dir = _make_iter(
        tmp_path,
        prompt_text="## One tasks\n",
        tasks_text="| T1 | A | WAIT | x | n/a | -- |\n",  # double hyphen, not em-dash
        prompts=["t1-a.md"],
        reviews=["review-t1.md"],
    )
    errors = sl.lint_iteration(iter_dir)
    assert any("'--' for no-dependency" in e for e in errors), errors


def test_double_hyphen_no_dep_ignores_flag_in_title():
    """A '--flag' in the Title/Test cell is not a no-dependency violation —
    only the depends-on column (last cell) is inspected."""
    # depends-on column correctly uses '—'; '--skip-impl' is in the title cell.
    row = "| T3 | `--skip-impl` preserves the task branch | WAIT | x | n/a | T2 |\n"
    assert sl.has_double_hyphen_no_dep(row) == []
    # a genuine '--' in the depends-on column is still caught.
    assert sl.has_double_hyphen_no_dep("| T1 | A | WAIT | x | n/a | -- |\n") != []


def test_scaffold_lint_main_returns_1_on_drift(tmp_path: Path, capsys):
    """End-to-end: main() returns 1 when an iteration has drift."""
    iter_dir = _make_iter(
        tmp_path,
        prompt_text="## Six tasks\n",
        tasks_text="| T1 | A | WAIT | x | n/a | — |\n",
        prompts=["t1-a.md"],
        reviews=["review-t1.md"],
    )
    rc = sl.main([str(iter_dir)])
    assert rc == 1
    out = capsys.readouterr()
    assert "FAIL" in (out.err + out.out)


def test_scaffold_lint_main_returns_0_on_clean(tmp_path: Path, capsys):
    """End-to-end: main() returns 0 on a clean iteration."""
    iter_dir = _make_iter(
        tmp_path,
        prompt_text="## One tasks\n",
        tasks_text="| T1 | A | WAIT | x | n/a | — |\n",
        prompts=["t1-a.md"],
        reviews=["review-t1.md"],
    )
    rc = sl.main([str(iter_dir)])
    assert rc == 0


def test_scaffold_lint_flags_post_phase_prompt_targeting_main(
    tmp_path: Path,
    monkeypatch,
):
    """post-phase runnable task prompts must use the integration branch."""
    _write_project_config(
        tmp_path,
        "\nscaffold:\n"
        "  post_phase_iteration_root: iterations/post-phase\n"
        "  post_phase_integration_branch: post-phase-integration\n",
    )
    iter_dir = tmp_path / "iterations" / "post-phase" / "cleanup"
    iter_dir.mkdir(parents=True)
    (iter_dir / "prompt.md").write_text("## One tasks\n")
    (iter_dir / "tasks.md").write_text("| T1 | A | WAIT | x | n/a | — |\n")
    prompts_dir = iter_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "t1-bad.md").write_text(
        "## Pre-Flight\n\n```bash\ngit checkout main\n```\n"
    )
    reviews_dir = iter_dir / "reviews"
    reviews_dir.mkdir()
    (reviews_dir / "review-t1.md").write_text("# review\n")

    monkeypatch.chdir(tmp_path)
    policy = sl.load_scaffold_policy(tmp_path)
    errors = sl.lint_iteration(iter_dir, policy=policy)
    assert any("git checkout main" in err for err in errors), errors
    assert any("post-phase-integration" in err for err in errors), errors


def test_scaffold_policy_uses_configured_post_phase_roots(
    tmp_path: Path,
    monkeypatch,
):
    _write_project_config(
        tmp_path,
        "\nscaffold:\n"
        + "  post_phase_iteration_root: custom/PostPhase\n"
        + "  tooling_iteration_root: custom/tools\n"
        + "  post_phase_integration_branch: custom-integration\n"
    )
    iter_dir = tmp_path / "custom" / "PostPhase" / "cleanup"
    iter_dir.mkdir(parents=True)
    (iter_dir / "prompt.md").write_text("## One tasks\n")
    (iter_dir / "tasks.md").write_text("| T1 | A | WAIT | x | n/a | — |\n")
    prompts_dir = iter_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "t1-bad.md").write_text(
        "## Pre-Flight\n\n```bash\ngit checkout main\n```\n"
    )
    reviews_dir = iter_dir / "reviews"
    reviews_dir.mkdir()
    (reviews_dir / "review-t1.md").write_text("# review\n")

    monkeypatch.chdir(tmp_path)
    policy = sl.load_scaffold_policy(tmp_path)
    errors = sl.lint_iteration(iter_dir, policy=policy)

    assert any("git checkout main" in err for err in errors), errors
    assert any("custom-integration" in err for err in errors), errors


def test_scaffold_nav_uses_configured_task_detail_pattern(
    tmp_path: Path,
    monkeypatch,
):
    _write_project_config(
        tmp_path,
        "\nscaffold:\n"
        + "  tooling_iteration_root: custom/tools\n"
        + "ui_route_visibility:\n"
        + "  route_globs: [\"ui/pages/*.html\"]\n"
        + "patterns:\n"
        + "  task_id: \"^TASK-[a-z]+$\"\n"
        + "  task_detail_heading: \"^###\\\\s+(?P<id>TASK-[a-z]+)\\\\s+—\\\\s+(?P<title>.+?)\\\\s*$\"\n"
    )
    iter_dir = tmp_path / "custom" / "tools" / "demo"
    iter_dir.mkdir(parents=True)
    (iter_dir / "prompt.md").write_text("## One tasks\n")
    (iter_dir / "tasks.md").write_text(
        "### TASK-alpha — Alpha\n\n"
        "**Allowed files:**\n"
        "```\n"
        "ui/pages/new.html\n"
        "```\n"
    )
    (iter_dir / "prompts").mkdir()
    (iter_dir / "reviews").mkdir()

    monkeypatch.chdir(tmp_path)
    policy = sl.load_scaffold_policy(tmp_path)

    errors = sl.nav_discoverability_iteration_errors(
        iter_dir,
        repo_root=tmp_path,
        policy=policy,
    )

    assert any("TASK-alpha" in err for err in errors), errors
    assert any("ui/pages/new.html" in err for err in errors), errors


def test_nav_marker_from_t30_does_not_satisfy_t3(tmp_path: Path, monkeypatch):
    _write_project_config(
        tmp_path,
        "\nscaffold:\n"
        "  tooling_iteration_root: iterations/tools\n"
        "ui_route_visibility:\n"
        "  route_globs:\n"
        "    - app/routes/*.py\n",
    )
    iter_dir = tmp_path / "iterations" / "tools" / "demo"
    iter_dir.mkdir(parents=True)
    (iter_dir / "prompt.md").write_text("## One tasks\n")
    (iter_dir / "tasks.md").write_text(
        "### T3 — Third\n\n"
        "**Allowed files:**\n"
        "```\n"
        "app/routes/new_page.py\n"
        "```\n"
    )
    prompts_dir = iter_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "t30-other.md").write_text(
        "**Nav discoverability:** covered elsewhere\n"
    )
    reviews_dir = iter_dir / "reviews"
    reviews_dir.mkdir()

    monkeypatch.chdir(tmp_path)
    policy = sl.load_scaffold_policy(tmp_path)

    errors = sl.nav_discoverability_iteration_errors(
        iter_dir,
        repo_root=tmp_path,
        policy=policy,
    )

    assert any("task T3" in err for err in errors), errors
    assert any("app/routes/new_page.py" in err for err in errors), errors
