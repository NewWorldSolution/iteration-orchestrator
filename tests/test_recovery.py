from __future__ import annotations

from orch.recovery import (
    append_recovery_note,
    impl_failure_recovery_note,
    notes_timing_recovery_note,
    stop_reason_recovery_note,
)


def test_impl_rate_limit_note_is_operator_side_only() -> None:
    note = impl_failure_recovery_note("rate_limit", iteration="demo-i1")

    assert "will not switch accounts" in note
    assert "operator may pause" in note
    assert "python -m orch resume demo-i1 --accept-external" in note


def test_impl_auth_note_is_safe_for_credentials() -> None:
    note = impl_failure_recovery_note("auth", iteration="demo-i1")

    assert "gh auth status" in note
    assert "do not paste secrets into logs" in note


def test_review_malformed_note_points_to_artifact() -> None:
    note = stop_reason_recovery_note(
        "REVIEW_MALFORMED",
        iteration="demo-i1",
        review_artifact="tools/logs/demo-i1/reviews/review_I1-T1_r1.md",
    )

    assert note is not None
    assert "Retry once" in note
    assert "tools/logs/demo-i1/reviews/review_I1-T1_r1.md" in note


def test_internal_stop_note_mentions_retry_and_recover() -> None:
    note = stop_reason_recovery_note("INTERNAL", iteration="demo-i1")

    assert note is not None
    assert "run_state.json" in note
    assert "python -m orch retry demo-i1 <task>" in note
    assert "python -m orch recover demo-i1" in note


def test_notes_timing_recovery_note_honors_configured_root() -> None:
    note = notes_timing_recovery_note("demo-i1", artifact_root_ref="custom/root")

    assert "custom/root/demo-i1/notes.md" in note
    assert "tools/logs" not in note


def test_config_stop_note_points_at_offending_field() -> None:
    note = stop_reason_recovery_note("CONFIG", iteration="demo-i1")

    assert note is not None
    assert "config value was invalid" in note
    assert "task_kind timeout profile" in note
    assert "rerun validation" in note


def test_append_recovery_note_is_idempotent() -> None:
    msg = "failed\n\nRecovery note: inspect first."

    assert append_recovery_note(msg, "Recovery note: another note.") == msg


def test_manual_wait_note_uses_notes_md() -> None:
    note = notes_timing_recovery_note("demo-i1")

    assert "tools/logs/demo-i1/notes.md" in note
    assert "manual waits" in note
