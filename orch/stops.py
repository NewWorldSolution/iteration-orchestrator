"""Shared orchestrator stop primitives.

This leaf module keeps stop labels importable without making future runner
submodules import ``orch.runner``.
"""
from __future__ import annotations


# Stop reason labels - the canonical set of run stop reasons.
STOP_PREFLIGHT = "PREFLIGHT_SIZE"
STOP_IMPL_FAILED = "IMPL_FAILED"
STOP_IMPL_TIMEOUT = "IMPL_TIMEOUT"
STOP_SCOPE = "SCOPE"
STOP_STRUCTURAL = "STRUCTURAL"
STOP_CHECKS = "CHECKS"
STOP_REVIEW_MALFORMED = "REVIEW_MALFORMED"
STOP_REVIEW_FAIL = "REVIEW_FAIL"
STOP_INDEPENDENCE = "INDEPENDENCE"
STOP_INTERNAL = "INTERNAL"
STOP_BRANCH_FRESHNESS = "BRANCH_FRESHNESS"
STOP_HOOK_VETO = "HOOK_VETO"
STOP_DUAL_REVIEW_REQUIRED = "DUAL_REVIEW_REQUIRED"
STOP_DUAL_REVIEW_FAIL = "DUAL_REVIEW_FAIL"
STOP_DUAL_REVIEW_MALFORMED = "DUAL_REVIEW_MALFORMED"
# Config/profile-resolution error the operator can fix (e.g. an unknown
# task_kind timeout profile). Distinct from PREFLIGHT_SIZE, which refuses an
# oversized diff - conflating the two misleads triage.
STOP_CONFIG = "CONFIG"


class _TaskStopped(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


__all__ = [
    "STOP_BRANCH_FRESHNESS",
    "STOP_CHECKS",
    "STOP_CONFIG",
    "STOP_DUAL_REVIEW_FAIL",
    "STOP_DUAL_REVIEW_MALFORMED",
    "STOP_DUAL_REVIEW_REQUIRED",
    "STOP_HOOK_VETO",
    "STOP_IMPL_FAILED",
    "STOP_IMPL_TIMEOUT",
    "STOP_INDEPENDENCE",
    "STOP_INTERNAL",
    "STOP_PREFLIGHT",
    "STOP_REVIEW_FAIL",
    "STOP_REVIEW_MALFORMED",
    "STOP_SCOPE",
    "STOP_STRUCTURAL",
    "_TaskStopped",
]
