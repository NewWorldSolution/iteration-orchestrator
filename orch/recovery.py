"""Operator-facing recovery notes for orchestrator blocked states.

These helpers intentionally render text only. They do not retry, switch
accounts, clean locks, mutate run state, or choose credentials.
"""
from __future__ import annotations


def append_recovery_note(message: str, note: str | None) -> str:
    """Append ``note`` to ``message`` without duplicating existing text."""
    if not note or note in message:
        return message
    if "Recovery note:" in message:
        return message
    return f"{message}\n\n{note}"


def impl_failure_recovery_note(
    classification: str | None, *, iteration: str
) -> str:
    label = classification or "unknown"
    if label == "rate_limit":
        return (
            "Recovery note: provider rate limit or quota exhaustion was "
            "detected. The orchestrator will not switch accounts or model "
            "families automatically. The operator may pause until reset, or "
            "switch accounts outside the orchestrator, then run "
            f"`python -m orch resume {iteration} --accept-external`."
        )
    if label == "auth":
        return (
            "Recovery note: authentication or credential configuration appears "
            "to be failing. Check the agent/CLI login and `gh auth status` "
            "outside the orchestrator; do not paste secrets into logs. Resume "
            f"with `python -m orch resume {iteration} --accept-external` "
            "after credentials are fixed."
        )
    if label == "disk":
        return (
            "Recovery note: disk-space failure detected. Free space or move the "
            "worktree outside the orchestrator, then resume only after the "
            "workspace is clean."
        )
    if label == "oom":
        return (
            "Recovery note: the implementer process was likely killed for "
            "memory pressure. Retry once after freeing memory; if it repeats, "
            "split the task before resuming."
        )
    if label == "permission":
        return (
            "Recovery note: filesystem permission failure detected. Inspect "
            "the worktree ownership/permissions manually; do not run "
            "destructive cleanup without operator approval."
        )
    return (
        "Recovery note: inspect the implementer stderr tail and the worktree "
        "state before retrying. Retry once only if the failure is clearly "
        "transient; otherwise split the task or ask the operator."
    )


def stop_reason_recovery_note(
    reason: str | None,
    *,
    iteration: str,
    review_artifact: str | None = None,
) -> str | None:
    if reason == "IMPL_TIMEOUT":
        return (
            "Recovery note: implementation timed out. Retry once with "
            f"`python -m orch resume {iteration}`; if it repeats, split "
            "the task before continuing."
        )
    if reason == "CHECKS":
        return (
            "Recovery note: inspect the acceptance command output, fix the "
            "failing tests in scope, and resume only after the same command "
            "passes locally."
        )
    if reason == "SCOPE":
        return (
            "Recovery note: revise the task prompt or allowed-files list. Do "
            "not delete or reset unrelated work without explicit operator "
            "approval."
        )
    if reason == "STRUCTURAL":
        return (
            "Recovery note: address the structural finding directly "
            "(conflicts, forbidden patterns, sensitive paths, or diff size). "
            "If the diff is too large, split the task."
        )
    if reason == "PREFLIGHT_SIZE":
        return (
            "Recovery note: the task is too large before implementation. Split "
            "the scope into smaller PRs and rerun validation before spending "
            "model time."
        )
    if reason == "CONFIG":
        return (
            "Recovery note: an orchestrator config value was invalid (e.g. an "
            "unknown task_kind timeout profile). Fix the offending field in "
            "tasks.md or the orch config so it resolves to a defined profile, "
            "then rerun validation before spending model time."
        )
    if reason == "REVIEW_MALFORMED":
        artifact = (
            f" Inspect `{review_artifact}` for the saved reviewer output."
            if review_artifact else ""
        )
        return (
            "Recovery note: reviewer output was malformed or partial. Retry "
            f"once; if it repeats, ask the operator to adjust the review "
            f"prompt.{artifact}"
        )
    if reason == "REVIEW_FAIL":
        artifact = (
            f" Inspect `{review_artifact}` for the latest saved review."
            if review_artifact else ""
        )
        return (
            "Recovery note: review found blocking issues. Fix the concrete "
            "findings or ask the operator whether to split/defer; do not "
            f"bypass review.{artifact}"
        )
    if reason == "DUAL_REVIEW_REQUIRED":
        return (
            "Recovery note: configure `--secondary-reviewer <agent>` or "
            "`review.secondary_reviewer` with a reviewer from a different "
            "model family than the primary reviewer, then resume. Do not "
            "bypass dual-model agreement for star-risk tasks."
        )
    if reason == "DUAL_REVIEW_FAIL":
        artifact = (
            f" Inspect `{review_artifact}` for the secondary review output."
            if review_artifact else ""
        )
        return (
            "Recovery note: secondary review did not approve the task. Fix "
            "the concrete finding, split/defer with operator approval, or "
            f"rerun only after the issue is addressed.{artifact}"
        )
    if reason == "DUAL_REVIEW_MALFORMED":
        artifact = (
            f" Inspect `{review_artifact}` for the saved secondary output."
            if review_artifact else ""
        )
        return (
            "Recovery note: secondary review output was malformed, partial, "
            "or unusable. Retry once or adjust the review prompt/config; do "
            f"not treat the primary review as sufficient.{artifact}"
        )
    if reason == "BRANCH_FRESHNESS":
        return (
            "Recovery note: refresh the branch against the expected base and "
            "rerun the freshness gate. Do not force-push shared integration "
            "branches."
        )
    if reason == "HOOK_VETO":
        return (
            "Recovery note: a required pre-action hook vetoed the task. "
            "Inspect the preceding hook_veto or hook_failure event in "
            "run_state.json, fix the hook input or policy issue, then retry "
            "the task."
        )
    if reason == "INTERNAL":
        return (
            "Recovery note: inspect run_state.json and the orchestrator logs, "
            "fix the internal issue, then rerun the task with "
            f"`python -m orch retry {iteration} <task>`. After B-7/T2 "
            f"lands, `python -m orch recover {iteration}` can clear "
            "stale locks/worktrees before retrying."
        )
    if reason == "INDEPENDENCE":
        return (
            "Recovery note: choose implementer and reviewer from different "
            "model families, then rerun."
        )
    return None


def report_blocker_recovery_note(reason: str | None) -> str | None:
    if not reason:
        return None
    return stop_reason_recovery_note(reason, iteration="<iter>")


def notes_timing_recovery_note(
    iteration: str, artifact_root_ref: str = "tools/logs"
) -> str:
    notes_ref = f"{artifact_root_ref.rstrip('/')}/{iteration}/notes.md"
    return (
        "Recovery note: for manual waits, CI stalls, account switches, quota "
        f"pauses, or review delays, add `{notes_ref}` with "
        "the reason, start/end time, and operator action taken."
    )


def lock_recovery_note(lock_path: str) -> str:
    return (
        "Recovery note: lock removal is manual. Inspect the recorded process "
        f"and `{lock_path}` first; remove the lock only after confirming no "
        "orchestrator run is active."
    )
