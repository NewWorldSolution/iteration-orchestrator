"""Characterization tests for runner review prompt flow before T4 split."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_runner import (
    FakeAdapter,
    _add_review_prompt,
    _make_runner,
    repo as _runner_repo,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _runner_repo.__wrapped__(tmp_path)


AUTHORED_REVIEW = """# Review - T1

## Verdict Output Contract

End with `Verdict: PASS`.

## Gate

- Check the exact authored review contract survives the runner split.
"""


def test_runner_split_preserves_review_prompt_content(repo: Path):
    _add_review_prompt(repo, "review-t1.md", AUTHORED_REVIEW)
    impl = FakeAdapter(name="claude", family="anthropic")
    rev = FakeAdapter(name="codex", family="openai")
    runner, _, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        implementer="claude",
        reviewer="codex",
    )
    task = runner.board.by_id("I1-T1")

    prompt = runner._build_task_review_prompt(
        task,
        fresh_diff="diff --git a/src/a.py b/src/a.py\n+VALUE = 1\n",
        base_sha="base-sha",
        reviewer_role="primary",
        round_num=1,
        max_rounds=2,
    )

    assert prompt == (
        "You are reviewing code for task `I1-T1: First`.\n\n"
        "## Review Prompt Contract\n\n"
        "Source: `iterations/demo-i1/reviews/review-t1.md`\n\n"
        f"{AUTHORED_REVIEW}\n\n"
        "## Runtime Review Metadata\n\n"
        "- Task: `I1-T1`\n"
        "- Title: `First`\n"
        "- Review role: `primary`\n"
        "- Round: `1/2`\n"
        "- Diff base: `base-sha`\n"
        "- Allowed files: `['src/a.py']`\n"
        "\n"
        "## Fresh Diff\n\n"
        "```diff\n"
        "diff --git a/src/a.py b/src/a.py\n+VALUE = 1\n"
        "\n"
        "```\n\n"
        "End with the exact trailing verdict block required by the "
        "review prompt contract. No non-empty line may follow it.\n"
    )
