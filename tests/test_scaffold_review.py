"""Tests for tools/scaffold_review.py config-backed policy."""
from __future__ import annotations

from pathlib import Path

from orch import scaffold_review as sr


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


def test_scaffold_review_uses_configured_iteration_root_and_artifact_prefix(
    tmp_path: Path,
    monkeypatch,
):
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(
        VALID_PROJECT_YAML
        + "\npaths:\n"
        + "  iteration_root: custom-iterations\n"
        + "  generated_artifact_exclusion_prefixes:\n"
        + "    - .orch/artifacts/\n"
    )
    iter_dir = tmp_path / "custom-iterations" / "demo-i1"
    iter_dir.mkdir(parents=True)
    (iter_dir / "prompt.md").write_text("## One tasks\n", encoding="utf-8")
    (iter_dir / "tasks.md").write_text("| T1 | A |\n", encoding="utf-8")
    prompts_dir = iter_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "t1-demo.md").write_text(
        "## Read first\n\n`.orch/artifacts/demo-i1/report.md`\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    assert sr.main(["--all"]) == 0
