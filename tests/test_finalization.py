"""Characterization tests for runner import surface before T4 split."""
from __future__ import annotations

import orch.runner as runner
import orch.stops as stops


def test_runner_import_surface_remains_backward_compatible():
    expected = {
        "IterationRunner",
        "RunnerDeps",
        "RunnerError",
        "RunOptions",
        "STATUS_DONE",
        "STOP_CHECKS",
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
        "_classify_impl_failure",
        "_diff_introduces_conflict_marker_pair",
        "_extract_review_findings",
        "_rate_limit_signature",
        "_stderr_tail",
    }

    assert not [name for name in sorted(expected) if not hasattr(runner, name)]


def test_runner_reexports_stop_primitives_for_backward_compatibility():
    expected = {
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
    }

    for name in expected:
        assert getattr(runner, name) == getattr(stops, name)
    assert runner._TaskStopped is stops._TaskStopped


def test_stops_star_export_includes_task_stopped():
    assert "_TaskStopped" in stops.__all__
    assert all(name.startswith("STOP_") or name == "_TaskStopped" for name in stops.__all__)
