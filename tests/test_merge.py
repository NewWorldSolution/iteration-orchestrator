"""Tests for orch.merge (guard logic + CI poll harness)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from orch.merge import (
    MERGE_STRATEGY,
    PrSnapshot,
    _gh,
    _git,
    build_needs_human_merge_meta,
    build_task_pr_request,
    evaluate_auto_merge,
    human_merge_recovery_message,
    is_external_merge_complete,
    matches_high_risk,
    merge_pr,
    open_pr,
    parse_merge_sha,
    render_guard_comment,
    wait_for_ci,
)


AUTO_MERGE_CFG = {
    "max_fix_rounds_default": 1,
    "max_fix_rounds_high_risk": 0,
    "max_diff_insertions": 500,
    "ci_wait_seconds": 300,
}
HIGH_RISK_GLOBS = ["**/schema.sql", "**/auth*.py", "**/*migration*"]


def _ok_kwargs(**overrides):
    kw = dict(
        verdict="PASS",
        changed_files=["app/routes.py", "app/service.py"],
        sensitive_hits=[],
        forbidden_hits=[],
        diff_insertions=120,
        fix_rounds=0,
        high_risk_globs=HIGH_RISK_GLOBS,
        auto_merge_cfg=AUTO_MERGE_CFG,
        ci_passed=True,
    )
    kw.update(overrides)
    return kw


def test_happy_path_auto_merges():
    d = evaluate_auto_merge(**_ok_kwargs())
    assert d.should_auto_merge
    assert d.reasons == []


def test_merge_strategy_contract_is_shared(tmp_path: Path):
    calls = []

    def fake_gh(args, *, cwd, timeout):
        calls.append(list(args))
        if args[:2] == ["pr", "view"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"mergeCommit": {"oid": "deadbeef1"}}
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="merged\n", stderr="")

    ok, message = merge_pr(
        cwd=tmp_path, pr_url="https://example.com/pr/1", _run_gh=fake_gh
    )

    assert ok
    assert parse_merge_sha(message) == "deadbeef1"
    assert calls[0] == [
        "pr",
        "merge",
        "https://example.com/pr/1",
        MERGE_STRATEGY.github_flag,
    ]
    assert MERGE_STRATEGY.local_git_args == ("--no-ff",)
    recovery = human_merge_recovery_message(
        pr_url="https://example.com/pr/1",
        iteration="demo-i1",
        task_id="I1-T1",
    )
    assert f"gh pr merge https://example.com/pr/1 {MERGE_STRATEGY.github_flag}" in recovery


def test_human_merge_recovery_honors_configured_artifact_root():
    recovery = human_merge_recovery_message(
        pr_url="https://example.com/pr/1",
        iteration="demo-i1",
        task_id="I1-T1",
        artifact_root_ref="custom/logroot",
    )
    # The notes hint routes through the configured root, not the hardcoded one.
    assert "custom/logroot/demo-i1/notes.md" in recovery
    assert "tools/logs/demo-i1/notes.md" not in recovery


def test_fail_verdict_blocks():
    d = evaluate_auto_merge(**_ok_kwargs(verdict="CHANGES REQUIRED"))
    assert d.needs_human
    assert any("verdict" in r for r in d.reasons)


def test_sensitive_hits_block():
    d = evaluate_auto_merge(**_ok_kwargs(sensitive_hits=[".env"]))
    assert d.needs_human
    assert any("sensitive" in r for r in d.reasons)


def test_forbidden_hits_block():
    d = evaluate_auto_merge(**_ok_kwargs(forbidden_hits=["except: pass"]))
    assert d.needs_human


def test_diff_over_cap_blocks():
    d = evaluate_auto_merge(**_ok_kwargs(diff_insertions=501))
    assert d.needs_human
    assert any("501" in r for r in d.reasons)


def test_fix_rounds_at_default_cap_ok():
    # default cap is 1; fix_rounds=1 should pass
    d = evaluate_auto_merge(**_ok_kwargs(fix_rounds=1))
    assert d.should_auto_merge


def test_fix_rounds_over_default_cap_blocks():
    d = evaluate_auto_merge(**_ok_kwargs(fix_rounds=2))
    assert d.needs_human


def test_high_risk_file_zero_fix_rounds_ok():
    d = evaluate_auto_merge(**_ok_kwargs(
        changed_files=["db/schema.sql"], fix_rounds=0,
    ))
    assert d.should_auto_merge


def test_high_risk_file_any_fix_round_blocks():
    d = evaluate_auto_merge(**_ok_kwargs(
        changed_files=["db/schema.sql", "app/x.py"], fix_rounds=1,
    ))
    assert d.needs_human
    assert any("high-risk" in r for r in d.reasons)


def test_ci_not_passed_blocks():
    d = evaluate_auto_merge(**_ok_kwargs(ci_passed=False))
    assert d.needs_human


def test_no_ci_guard_skips_ci_reason_when_enabled():
    d = evaluate_auto_merge(
        **_ok_kwargs(
            auto_merge_cfg={**AUTO_MERGE_CFG, "no_ci": True},
            ci_passed=False,
        )
    )

    assert d.should_auto_merge
    assert d.reasons == []


def test_no_ci_guard_still_blocks_non_ci_reasons():
    d = evaluate_auto_merge(
        **_ok_kwargs(
            auto_merge_cfg={**AUTO_MERGE_CFG, "no_ci": True},
            ci_passed=False,
            sensitive_hits=[".env"],
        )
    )

    assert d.needs_human
    assert d.reasons == ["sensitive files touched: ['.env']"]


def test_unresolved_warnings_block():
    d = evaluate_auto_merge(**_ok_kwargs(unresolved_warnings=["flaky retest needed"]))
    assert d.needs_human


def test_matches_high_risk_globs():
    hits = matches_high_risk(
        ["app/auth.py", "app/routes.py", "db/20240101_migration_x.py"],
        HIGH_RISK_GLOBS,
    )
    assert set(hits) == {"app/auth.py", "db/20240101_migration_x.py"}


def test_render_guard_comment_lists_reasons():
    d = evaluate_auto_merge(**_ok_kwargs(
        verdict="CHANGES REQUIRED", diff_insertions=600,
    ))
    body = render_guard_comment(d)
    assert "Auto-merge blocked" in body
    assert "verdict" in body
    assert "600" in body


def test_render_guard_comment_passing():
    d = evaluate_auto_merge(**_ok_kwargs())
    assert "passed" in render_guard_comment(d)


def test_build_task_pr_request_pins_exact_fields():
    request = build_task_pr_request(
        task_id="I9-T2",
        task_title="Add report filters",
        iteration="demo-i9",
        iter_branch="demo/iteration-9",
        task_branch="demo/i9/t2-report-filters",
        allowed_files=["app/reports.py", "tests/test_reports.py"],
        test_cmd="pytest tests/test_reports.py -q",
    )

    assert request.title == "I9-T2: Add report filters"
    assert request.base == "demo/iteration-9"
    assert request.head == "demo/i9/t2-report-filters"
    # Structured body follows the CLAUDE.md final-PR standard section layout,
    # not the old forbidden one-liner.
    body = request.body
    for header in (
        "## What this PR delivers",
        "## Files changed",
        "## What was run",
        "## Scope",
        "## Follow-ups surfaced",
        "## Merge readiness",
    ):
        assert header in body, f"missing section {header!r}"
    assert "I9-T2 — Add report filters" in body
    assert "`app/reports.py`" in body
    assert "`tests/test_reports.py`" in body
    assert "pytest tests/test_reports.py -q" in body
    assert "operator-owned" in body
    # Deterministic: same inputs render the same body.
    again = build_task_pr_request(
        task_id="I9-T2",
        task_title="Add report filters",
        iteration="demo-i9",
        iter_branch="demo/iteration-9",
        task_branch="demo/i9/t2-report-filters",
        allowed_files=["app/reports.py", "tests/test_reports.py"],
        test_cmd="pytest tests/test_reports.py -q",
    )
    assert again.body == body
    assert "Automated PR for" not in body


def test_build_task_pr_request_without_scope_or_test_is_explicit():
    request = build_task_pr_request(
        task_id="I1-T1",
        task_title="Docs note",
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        task_branch="demo/iteration-1-tasks/i1-t1-docs-note",
    )
    body = request.body
    assert "no explicit allowed-file scope" in body
    assert "No scoped acceptance test command was declared." in body


def test_build_needs_human_merge_meta_pins_exact_shape_without_guard_reasons():
    meta = build_needs_human_merge_meta(
        classification="checks-in-progress",
        msg_detail="Checks still running: test.",
        ci_passed=False,
        pr_url="https://github.com/acme/repo/pull/42",
        iteration="demo-i9",
        task_id="I9-T2",
    )

    assert meta == {
        "event": "needs_human_merge",
        "classification": "checks-in-progress",
        "msg": "Checks still running: test.",
        "ci_passed": False,
        "recovery": "\n".join([
            "Recovery for I9-T2 (PR https://github.com/acme/repo/pull/42, iteration demo-i9):",
            "  1. Wait for GitHub Actions CI to go green on the PR.",
            "  2. gh pr merge https://github.com/acme/repo/pull/42 --merge",
            "  3. git pull --ff-only on the iteration branch.",
            "  4. (orch resume auto-detects the merge — see step 5; the",
            "      run_state.json edit only matters when --accept-external is",
            "      passed before B2c shipped.)",
            "  5. python -m orch resume demo-i9 --accept-external",
            "Recovery note: record manual CI waits, merge stalls, or "
            "operator-side account changes in "
            "tools/logs/demo-i9/notes.md before claiming timing evidence.",
        ]),
    }


def test_build_needs_human_merge_meta_pins_guard_reasons_key_when_present():
    meta = build_needs_human_merge_meta(
        classification="checks-failed",
        msg_detail="Checks FAILED: test. Investigate before merging.",
        ci_passed=False,
        pr_url="https://github.com/acme/repo/pull/42",
        iteration="demo-i9",
        task_id="I9-T2",
        decision_reasons=["CI not green within ci_wait_seconds"],
    )

    assert meta["guard_reasons"] == ["CI not green within ci_wait_seconds"]
    assert set(meta) == {
        "event",
        "classification",
        "msg",
        "ci_passed",
        "recovery",
        "guard_reasons",
    }


def test_parse_merge_sha_uses_json_output():
    assert (
        parse_merge_sha(
            json.dumps({"mergeCommit": {"oid": "deadbeef1"}})
        )
        == "deadbeef1"
    )
    assert parse_merge_sha(json.dumps({"merge_sha": "cafebabe2"})) == "cafebabe2"


def test_parse_merge_sha_preserves_runner_semantics():
    assert parse_merge_sha("merged deadbeef1") == "deadbeef1"
    assert parse_merge_sha("abc1234 then deadbeef") == "abc1234"
    assert parse_merge_sha("") is None
    assert parse_merge_sha("merged ABCDEF1") is None
    assert parse_merge_sha("short abc123") is None


def test_is_external_merge_complete_requires_merged_state_and_sha():
    assert is_external_merge_complete(
        PrSnapshot(state="MERGED", merge_sha="deadbeef", rollup=[])
    )
    assert not is_external_merge_complete(
        PrSnapshot(state="OPEN", merge_sha="deadbeef", rollup=[])
    )
    assert not is_external_merge_complete(
        PrSnapshot(state="MERGED", merge_sha=None, rollup=[])
    )


def test_human_merge_recovery_mentions_manual_wait_notes():
    msg = human_merge_recovery_message(
        pr_url="https://example.com/pr/1", iteration="demo-i1", task_id="I1-T1"
    )

    assert "python -m orch resume demo-i1 --accept-external" in msg
    assert "tools/logs/demo-i1/notes.md" in msg
    assert "manual CI waits" in msg


# ---------------------------------------------------------------------------
# wait_for_ci with fake gh runner
# ---------------------------------------------------------------------------


def _fake_gh(responses):
    """Return a function mimicking subprocess.run for gh pr checks."""
    it = iter(responses)

    def runner(args, *, cwd, timeout):
        body = next(it)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(body),
            stderr="",
        )
    return runner


def test_wait_for_ci_success(tmp_path: Path):
    # gh pr checks --json exposes `bucket` (pass|fail|pending|skipping|cancel);
    # fakes mirror the real payload shape.
    run_gh = _fake_gh([
        [{"name": "test", "bucket": "pending"}],
        [{"name": "test", "bucket": "pass"},
         {"name": "lint", "bucket": "skipping"}],
    ])
    fake_clock = iter([0.0, 5.0, 10.0, 20.0])
    status = wait_for_ci(
        "feat/x",
        cwd=tmp_path,
        ci_wait_seconds=300,
        poll_interval_s=0,
        _clock=lambda: next(fake_clock),
        _sleep=lambda s: None,
        _run_gh=run_gh,
    )
    assert status.passed
    assert status.conclusion == "success"


def test_wait_for_ci_queries_bucket_not_conclusion(tmp_path: Path):
    # Regression: requesting the non-existent `conclusion` field made every
    # poll exit 1, so green CI was never seen and tasks parked
    # NEEDS_HUMAN_MERGE.
    seen_args: list[list[str]] = []

    def recording_gh(args, *, cwd, timeout):
        seen_args.append(list(args))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"name": "test", "bucket": "pass"}]),
            stderr="",
        )

    status = wait_for_ci(
        "feat/x", cwd=tmp_path, ci_wait_seconds=60,
        poll_interval_s=0,
        _clock=lambda: 0.0,
        _sleep=lambda s: None,
        _run_gh=recording_gh,
    )
    assert status.passed
    json_field = seen_args[0][seen_args[0].index("--json") + 1]
    assert "conclusion" not in json_field
    assert "bucket" in json_field


def test_wait_for_ci_pr_checks_field_contract_is_exact(tmp_path: Path):
    seen_args: list[list[str]] = []

    def recording_gh(args, *, cwd, timeout):
        seen_args.append(list(args))
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"name": "test", "bucket": "pass"}]),
            stderr="",
        )

    status = wait_for_ci(
        "feat/x", cwd=tmp_path, ci_wait_seconds=60,
        poll_interval_s=0,
        _clock=lambda: 0.0,
        _sleep=lambda s: None,
        _run_gh=recording_gh,
    )

    assert status.passed
    assert seen_args == [["pr", "checks", "feat/x", "--json", "name,bucket"]]


def test_wait_for_ci_failure(tmp_path: Path):
    run_gh = _fake_gh([
        [{"name": "test", "bucket": "fail"}],
    ])
    status = wait_for_ci(
        "feat/x", cwd=tmp_path, ci_wait_seconds=60,
        poll_interval_s=0,
        _clock=lambda: 0.0,
        _sleep=lambda s: None,
        _run_gh=run_gh,
    )
    assert not status.passed
    assert status.conclusion == "failure"


def test_wait_for_ci_cancel_bucket_is_failure(tmp_path: Path):
    run_gh = _fake_gh([
        [{"name": "test", "bucket": "pass"},
         {"name": "e2e", "bucket": "cancel"}],
    ])
    status = wait_for_ci(
        "feat/x", cwd=tmp_path, ci_wait_seconds=60,
        poll_interval_s=0,
        _clock=lambda: 0.0,
        _sleep=lambda s: None,
        _run_gh=run_gh,
    )
    assert not status.passed
    assert status.conclusion == "failure"


class RecordingCliProvider:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.calls: list[dict] = []
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def run(self, args, *, cwd: Path, timeout: int):
        self.calls.append({"args": list(args), "cwd": cwd, "timeout": timeout})
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def test_merge_gh_wrapper_uses_provider(tmp_path: Path):
    provider = RecordingCliProvider(stdout="{}\n")

    proc = _gh(["pr", "view", "1"], cwd=tmp_path, timeout=11, provider=provider)

    assert proc.stdout == "{}\n"
    assert provider.calls == [
        {"args": ["pr", "view", "1"], "cwd": tmp_path, "timeout": 11}
    ]


def test_merge_git_wrapper_uses_provider(tmp_path: Path):
    provider = RecordingCliProvider(stdout="pushed\n")

    proc = _git(["push", "origin", "head"], cwd=tmp_path, timeout=22, provider=provider)

    assert proc.stdout == "pushed\n"
    assert provider.calls == [
        {"args": ["push", "origin", "head"], "cwd": tmp_path, "timeout": 22}
    ]


# ---------------------------------------------------------------------------
# v2.3 — open_pr must force-push head before gh pr create
# ---------------------------------------------------------------------------


def _capturing_runner():
    """Return a fake subprocess runner that records every call."""
    calls: list[dict] = []

    def make(returncode: int = 0, stdout: str = "", stderr: str = ""):
        def runner(args, *, cwd, timeout):
            calls.append(
                {"cmd": args[0], "args": list(args), "cwd": cwd,
                 "timeout": timeout}
            )
            return SimpleNamespace(
                returncode=returncode, stdout=stdout, stderr=stderr,
            )
        return runner
    return calls, make


def test_open_pr_force_pushes_head_before_gh_create(tmp_path: Path):
    calls, make = _capturing_runner()
    ok, url = open_pr(
        cwd=tmp_path,
        title="T1", body="b",
        base="demo/iteration-7",
        head="demo/i7/t1-routes-gate",
        _run_git=make(returncode=0, stdout="pushed\n"),
        _run_gh=make(returncode=0, stdout="https://example.com/pr/42\n"),
    )
    assert ok
    assert url == "https://example.com/pr/42"
    # git push called with --force-with-lease and the head branch, and it
    # ran before the gh call.
    assert calls[0]["cmd"] == "push"
    assert "--force-with-lease" in calls[0]["args"]
    assert "origin" in calls[0]["args"]
    assert "demo/i7/t1-routes-gate" in calls[0]["args"]
    assert calls[1]["cmd"] == "pr"
    # gh receives --head pointing at the same branch we just pushed.
    assert "--head" in calls[1]["args"]
    assert calls[1]["args"][
        calls[1]["args"].index("--head") + 1
    ] == "demo/i7/t1-routes-gate"


def test_open_pr_reports_push_failure_without_calling_gh(tmp_path: Path):
    # If the push fails, gh pr create must not run — otherwise we'd open
    # a PR against a stale remote head.
    gh_calls: list[dict] = []

    def gh_runner(args, *, cwd, timeout):
        gh_calls.append({"args": list(args)})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def git_runner(args, *, cwd, timeout):
        return SimpleNamespace(
            returncode=1, stdout="",
            stderr="! [rejected] head -> head (non-fast-forward)",
        )

    ok, err = open_pr(
        cwd=tmp_path,
        title="T", body="b",
        base="iter", head="t",
        _run_git=git_runner, _run_gh=gh_runner,
    )
    assert not ok
    assert "git push failed" in err
    assert "non-fast-forward" in err
    assert gh_calls == []


def test_wait_for_ci_timeout(tmp_path: Path):
    run_gh = _fake_gh([
        [{"name": "test", "bucket": "pending"}],
    ] * 10)
    clock_values = iter([0.0, 5.0, 100.0, 200.0, 400.0])
    status = wait_for_ci(
        "feat/x", cwd=tmp_path, ci_wait_seconds=60,
        poll_interval_s=0,
        _clock=lambda: next(clock_values),
        _sleep=lambda s: None,
        _run_gh=run_gh,
    )
    assert not status.passed
    assert status.conclusion == "timed_out"
