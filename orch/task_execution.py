"""Task execution helpers split out of the orchestrator runner.

The runner still owns the task state machine. This module carries the small,
pure diagnostics that are shared by implementation and structural gates.
"""
from __future__ import annotations


def diff_introduces_conflict_marker_pair(diff_text: str) -> bool:
    added: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].lstrip())
    has_start = any(line.startswith("<<<<<<<") for line in added)
    has_middle = any(line.startswith("=======") for line in added)
    has_end = any(line.startswith(">>>>>>>") for line in added)
    return has_start and has_middle and has_end


# Codex account rotation hint. The orchestrator cannot detect which
# provider account is in use; it can only PROMPT the operator. When
# IMPL_FAILED stderr matches a rate-limit signature, we append the
# rotation playbook to the stop_msg.
RATE_LIMIT_HINT = (
    "Rate-limit hit on codex. Recovery:\n"
    "  1. Switch to an alternate provider account on your machine "
    "(operator-side step, not visible to orch)\n"
    "  2. Run: python -m orch resume <iter> --accept-external\n"
    "If all accounts are exhausted, pause and wait for limit reset — "
    "do NOT switch the implementer role to Claude (would violate "
    "model-family-independence)."
)


def rate_limit_signature(stderr: str | None) -> bool:
    """Return True if stderr text suggests a provider rate-limit failure.

    Heuristic — matches the common patterns codex / claude CLIs surface
    on quota exhaustion. Local fallback in case the structured failure
    diagnostics have not landed.
    """
    if not stderr:
        return False
    needle = stderr.lower()
    return any(
        p in needle
        for p in ("rate limit", "rate_limit", "ratelimit", "quota", "429")
    )


# Fast-fail diagnostics for IMPL_FAILED. Heuristically bucket
# the failure so the operator sees the cause in run_state.json instead
# of a bare exit code.
IMPL_FAILURE_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rate_limit", ("rate limit", "rate_limit", "ratelimit", "quota", "429")),
    (
        "auth",
        (
            "authentication",
            "unauthorized",
            "invalid api key",
            "api key",
            "not logged in",
            "login required",
            "401",
            "403",
            "gh auth",
        ),
    ),
    ("disk", ("enospc", "no space left", "disk full")),
    ("permission", ("permission denied", "eacces", "operation not permitted")),
)
STDERR_TAIL_MAX_LINES = 50
STDERR_TAIL_MAX_BYTES = 2048


def classify_impl_failure(stderr: str | None, exit_code: int) -> str:
    """Bucket an impl-agent failure into a coarse classification.

    Returns one of: ``rate_limit``, ``auth``, ``disk``, ``oom``, ``permission``,
    ``unknown``. Order matters — rate-limit signatures are checked
    before generic permission text since codex sometimes surfaces both.
    Exit code 137 (SIGKILL, typical OOM-killer) wins over text matching
    because the stderr is often empty in that case.
    """
    if exit_code == 137:
        return "oom"
    needle = (stderr or "").lower()
    if needle:
        for label, patterns in IMPL_FAILURE_BUCKETS:
            if any(p in needle for p in patterns):
                return label
    return "unknown"


def stderr_tail(stderr: str | None) -> str:
    """Truncate stderr to the last 50 lines / 2 KB for run_state inclusion."""
    if not stderr:
        return ""
    lines = stderr.splitlines()[-STDERR_TAIL_MAX_LINES:]
    tail = "\n".join(lines)
    if len(tail.encode("utf-8")) > STDERR_TAIL_MAX_BYTES:
        # Trim from the front: keep the most recent bytes, which carry
        # the actual error message.
        tail = tail.encode("utf-8")[-STDERR_TAIL_MAX_BYTES:].decode(
            "utf-8", errors="replace"
        )
    return tail
