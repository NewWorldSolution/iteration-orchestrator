from __future__ import annotations

import pytest

from orch.checks import (
    NavDiscoverabilityEvidence,
    ScopeExceptionEvidence,
    parse_nav_discoverability_evidence,
    parse_scope_exception_evidence,
)
from orch.final_gates import (
    evaluate_final_outward_scope_gate,
    evaluate_final_nav_discoverability_gate,
    final_nav_discoverability_gate_input,
    final_scope_base_ref_yields_diff,
    unresolved_final_nav_discoverability_base_ref_decision,
    unresolved_final_scope_base_ref_decision,
)
from orch.git_ops import BranchFreshness, BranchFreshnessCondition


TASKS_MD = "iterations/demo-i1/tasks.md"
BASE_REF = "origin/phase-demo"
EVIDENCE_SOURCE = "tools/logs/demo-i1/scope_exceptions.md"
NAV_EVIDENCE_SOURCE = "tools/logs/demo-i1/nav_discoverability.md"
ROUTE_PATH = "app/routes/widgets.py"
TEMPLATE_PATH = "app/templates/widgets/list.html"
NAV_ANCHOR_PATH = "app/templates/base.html"
EXAMPLE_ROUTE_GLOBS = ["app/routes/*.py", "app/templates/**/*.html"]
EXAMPLE_NAV_ANCHOR_PATHS = [
    "app/templates/base.html",
    "app/templates/_nav.html",
    "app/templates/partials/_nav.html",
    "app/templates/nav.html",
]


def _missing_evidence() -> ScopeExceptionEvidence:
    return ScopeExceptionEvidence(source=EVIDENCE_SOURCE)


def _missing_nav_evidence() -> NavDiscoverabilityEvidence:
    return NavDiscoverabilityEvidence(source=NAV_EVIDENCE_SOURCE)


def _nav_evidence(*paths: str) -> NavDiscoverabilityEvidence:
    return parse_nav_discoverability_evidence(
        "\n".join(
            [
                "approved_by: operator",
                "reason: widgets are reachable from a parent page",
                "paths:",
                *(f"- {path}" for path in paths),
            ]
        ),
        source=NAV_EVIDENCE_SOURCE,
    )


def _example_nav_gate_input(changed_files: list[str]):
    return final_nav_discoverability_gate_input(
        changed_files,
        route_globs=EXAMPLE_ROUTE_GLOBS,
        nav_anchor_paths=EXAMPLE_NAV_ANCHOR_PATHS,
    )


def _expected_nav_failure_msg(
    *,
    route_visible: list[str],
    missing: list[str],
    nav_anchor_paths: list[str] | None = None,
) -> str:
    anchors = (
        EXAMPLE_NAV_ANCHOR_PATHS if nav_anchor_paths is None else nav_anchor_paths
    )
    if anchors:
        anchor_text = (
            f"none of the configured nav-anchor paths {anchors} were updated "
            "in the iteration diff"
        )
        remediation = "add the nav link in one of those configured anchors"
    else:
        anchor_text = (
            "no nav-anchor paths are configured for this project and no "
            "nav-anchor file was updated in the iteration diff"
        )
        remediation = (
            "configure ui_route_visibility.nav_anchor_paths for this project "
            "and add the nav link there"
        )
    return (
        "final nav-discoverability gate failed: route-visible surfaces "
        f"{route_visible} were added or changed, but {anchor_text} and "
        f"{missing} are not covered by operator-approved no-nav evidence; "
        f"{remediation} or list these paths in {NAV_EVIDENCE_SOURCE} with "
        "approved_by, "
        "reason, and exact relative paths."
    )


def test_final_scope_gate_allows_final_status_only_tasks_md() -> None:
    decision = evaluate_final_outward_scope_gate(
        base_ref=BASE_REF,
        changed_files=[TASKS_MD, "src/a.py"],
        allowed_files=["src/a.py"],
        tasks_md_path=TASKS_MD,
        tasks_md_status_only=True,
        evidence=_missing_evidence(),
    )

    assert decision.passed
    assert decision.failure_meta is None
    assert decision.exception_meta is None
    assert decision.passed_meta == {
        "event": "final_scope_gate_passed",
        "base_ref": BASE_REF,
        "changed_files": ["src/a.py"],
        "allowed_file_source": TASKS_MD,
        "allowed_files": ["src/a.py"],
        "tasks_md_status_only": True,
    }


def test_final_scope_gate_blocks_unexpected_outward_files() -> None:
    decision = evaluate_final_outward_scope_gate(
        base_ref=BASE_REF,
        changed_files=["src/a.py", "docs/leak.md"],
        allowed_files=["src/a.py"],
        tasks_md_path=TASKS_MD,
        tasks_md_status_only=False,
        evidence=_missing_evidence(),
    )

    expected_msg = (
        "final scope gate failed: disallowed files ['docs/leak.md']; "
        "allowed-file source is the union of every 'Allowed files' block "
        f"in {TASKS_MD}; explicit operator exceptions must be listed in "
        f"{EVIDENCE_SOURCE} with approved_by, reason, and exact relative paths."
    )
    assert not decision.passed
    assert decision.message == expected_msg
    assert decision.failure_meta == {
        "event": "final_scope_gate_failed",
        "reason": "SCOPE",
        "msg": expected_msg,
        "changed_files": ["src/a.py", "docs/leak.md"],
        "allowed_file_source": TASKS_MD,
        "allowed_files": ["src/a.py"],
    }
    assert decision.exception_meta is None
    assert decision.passed_meta is None


def test_final_scope_gate_clean_allowed_diff_passes() -> None:
    decision = evaluate_final_outward_scope_gate(
        base_ref=BASE_REF,
        changed_files=["src/a.py", "src/b.py"],
        allowed_files=["src/a.py", "src/b.py"],
        tasks_md_path=TASKS_MD,
        tasks_md_status_only=False,
        evidence=_missing_evidence(),
    )

    assert decision.passed
    assert decision.passed_meta == {
        "event": "final_scope_gate_passed",
        "base_ref": BASE_REF,
        "changed_files": ["src/a.py", "src/b.py"],
        "allowed_file_source": TASKS_MD,
        "allowed_files": ["src/a.py", "src/b.py"],
        "tasks_md_status_only": False,
    }


@pytest.mark.parametrize(
    "changed_files",
    [
        [],
        ["tools/logs/demo-i1/run_state.json"],
    ],
)
def test_final_scope_gate_empty_relevant_diff_passes(
    changed_files: list[str],
) -> None:
    decision = evaluate_final_outward_scope_gate(
        base_ref=BASE_REF,
        changed_files=changed_files,
        allowed_files=["src/a.py"],
        tasks_md_path=TASKS_MD,
        tasks_md_status_only=False,
        evidence=_missing_evidence(),
    )

    assert decision.passed
    assert decision.failure_meta is None
    assert decision.passed_meta == {
        "event": "final_scope_gate_passed",
        "base_ref": BASE_REF,
        "changed_files": [],
        "allowed_file_source": TASKS_MD,
        "allowed_files": ["src/a.py"],
        "tasks_md_status_only": False,
    }


def test_final_scope_gate_returns_exception_metadata_for_runner() -> None:
    evidence = parse_scope_exception_evidence(
        "\n".join(
            [
                "approved_by: operator",
                "reason: approved fold-in",
                "paths:",
                "- docs/foldin.md",
            ]
        ),
        source=EVIDENCE_SOURCE,
    )
    decision = evaluate_final_outward_scope_gate(
        base_ref=BASE_REF,
        changed_files=["src/a.py", "docs/foldin.md"],
        allowed_files=["src/a.py"],
        tasks_md_path=TASKS_MD,
        tasks_md_status_only=False,
        evidence=evidence,
    )

    assert decision.passed
    assert decision.exception_meta == {
        "event": "final_scope_exception_applied",
        "paths": ["docs/foldin.md"],
        "approved_by": "operator",
        "reason": "approved fold-in",
        "source": EVIDENCE_SOURCE,
    }
    assert decision.passed_meta == {
        "event": "final_scope_gate_passed",
        "base_ref": BASE_REF,
        "changed_files": ["src/a.py", "docs/foldin.md"],
        "allowed_file_source": TASKS_MD,
        "allowed_files": ["src/a.py"],
        "tasks_md_status_only": False,
    }


def test_final_scope_gate_unresolved_base_metadata_matches_runner() -> None:
    decision = unresolved_final_scope_base_ref_decision(
        iter_branch="demo/iteration-1",
        tasks_md_path=TASKS_MD,
    )

    expected_msg = (
        "final scope gate failed: could not resolve the phase/upstream "
        "diff base for iteration branch 'demo/iteration-1'"
    )
    assert not decision.passed
    assert decision.message == expected_msg
    assert decision.failure_meta == {
        "event": "final_scope_gate_failed",
        "reason": "SCOPE",
        "msg": expected_msg,
        "changed_files": [],
        "allowed_file_source": TASKS_MD,
        "allowed_files": [],
    }


def test_final_nav_gate_clean_required_evidence_passes() -> None:
    gate_input = _example_nav_gate_input([ROUTE_PATH, TEMPLATE_PATH])
    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
        evidence=_nav_evidence(ROUTE_PATH, TEMPLATE_PATH),
    )

    assert gate_input.requires_evidence
    assert decision.passed
    assert decision.failure_meta is None
    assert decision.passed_meta is None
    assert decision.exception_meta == {
        "event": "final_nav_discoverability_exception_applied",
        "base_ref": BASE_REF,
        "route_visible_surfaces": [ROUTE_PATH, TEMPLATE_PATH],
        "approved_paths": [ROUTE_PATH, TEMPLATE_PATH],
        "approved_by": "operator",
        "reason": "widgets are reachable from a parent page",
        "source": NAV_EVIDENCE_SOURCE,
    }


def test_final_nav_gate_missing_evidence_fails_with_runner_shape() -> None:
    gate_input = _example_nav_gate_input([ROUTE_PATH, TEMPLATE_PATH])
    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
        evidence=_missing_nav_evidence(),
    )

    expected_msg = _expected_nav_failure_msg(
        route_visible=[ROUTE_PATH, TEMPLATE_PATH],
        missing=[ROUTE_PATH, TEMPLATE_PATH],
    )
    assert not decision.passed
    assert decision.message == expected_msg
    assert decision.failure_meta == {
        "event": "final_nav_discoverability_gate_failed",
        "reason": "SCOPE",
        "msg": expected_msg,
        "route_visible_surfaces": [ROUTE_PATH, TEMPLATE_PATH],
        "approved_paths": [],
        "missing_paths": [ROUTE_PATH, TEMPLATE_PATH],
    }
    assert decision.exception_meta is None
    assert decision.passed_meta is None


def test_final_nav_gate_failure_message_lists_example_configured_anchors(
) -> None:
    gate_input = _example_nav_gate_input([ROUTE_PATH, TEMPLATE_PATH])
    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
        evidence=_missing_nav_evidence(),
    )

    assert not decision.passed
    for anchor in EXAMPLE_NAV_ANCHOR_PATHS:
        assert anchor in decision.message
    assert "one of those configured anchors" in decision.message


def test_final_nav_gate_failure_message_uses_custom_configured_anchors() -> None:
    custom_route = "web/pages/widgets.py"
    custom_anchors = [
        "web/templates/layout.html",
        "web/templates/navigation.html",
    ]
    gate_input = final_nav_discoverability_gate_input(
        [custom_route],
        route_globs=["web/pages/*.py"],
        nav_anchor_paths=custom_anchors,
    )

    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
        evidence=_missing_nav_evidence(),
    )

    expected_msg = _expected_nav_failure_msg(
        route_visible=[custom_route],
        missing=[custom_route],
        nav_anchor_paths=custom_anchors,
    )
    assert gate_input.route_visible_surfaces == [custom_route]
    assert gate_input.nav_anchor_paths == custom_anchors
    assert not decision.passed
    assert decision.message == expected_msg
    assert "web/templates/layout.html" in decision.message
    assert "web/templates/navigation.html" in decision.message
    assert "app/templates/base.html" not in decision.message


def test_final_nav_gate_failure_message_without_configured_anchors_is_generic(
) -> None:
    custom_route = "web/pages/widgets.py"
    gate_input = final_nav_discoverability_gate_input(
        [custom_route],
        route_globs=["web/pages/*.py"],
        nav_anchor_paths=[],
    )

    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
        evidence=_missing_nav_evidence(),
    )

    expected_msg = _expected_nav_failure_msg(
        route_visible=[custom_route],
        missing=[custom_route],
        nav_anchor_paths=[],
    )
    assert gate_input.route_visible_surfaces == [custom_route]
    assert gate_input.nav_anchor_paths == []
    assert not decision.passed
    assert decision.message == expected_msg
    assert "ui_route_visibility.nav_anchor_paths" in decision.message
    assert "app/templates/base.html" not in decision.message


def test_final_nav_gate_malformed_evidence_fails_with_runner_shape() -> None:
    gate_input = _example_nav_gate_input([ROUTE_PATH, TEMPLATE_PATH])
    evidence = parse_nav_discoverability_evidence(
        "\n".join(
            [
                "reason: missing approval identity",
                "paths:",
                f"- {ROUTE_PATH}",
                f"- {TEMPLATE_PATH}",
            ]
        ),
        source=NAV_EVIDENCE_SOURCE,
    )
    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
        evidence=evidence,
    )

    base_msg = _expected_nav_failure_msg(
        route_visible=[ROUTE_PATH, TEMPLATE_PATH],
        missing=[ROUTE_PATH, TEMPLATE_PATH],
    )
    expected_msg = (
        f"{base_msg} Evidence file at {NAV_EVIDENCE_SOURCE} is "
        f"malformed: {list(evidence.errors)}"
    )
    assert not decision.passed
    assert decision.message == expected_msg
    assert decision.failure_meta == {
        "event": "final_nav_discoverability_gate_failed",
        "reason": "SCOPE",
        "msg": expected_msg,
        "route_visible_surfaces": [ROUTE_PATH, TEMPLATE_PATH],
        "approved_paths": [],
        "missing_paths": [ROUTE_PATH, TEMPLATE_PATH],
    }


def test_final_nav_gate_uncovered_route_visible_path_handling_is_preserved(
) -> None:
    gate_input = _example_nav_gate_input(
        [
            "tools/logs/demo-i1/run_state.json",
            ROUTE_PATH,
            "app/routes/widgets/nested.py",
            "app/routes/__init__.py",
            TEMPLATE_PATH,
            "docs/discoverability.md",
        ]
    )
    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
        evidence=_nav_evidence(ROUTE_PATH),
    )

    expected_msg = _expected_nav_failure_msg(
        route_visible=[ROUTE_PATH, TEMPLATE_PATH],
        missing=[TEMPLATE_PATH],
    )
    assert gate_input.route_visible_surfaces == [ROUTE_PATH, TEMPLATE_PATH]
    assert not decision.passed
    assert decision.message == expected_msg
    assert decision.failure_meta == {
        "event": "final_nav_discoverability_gate_failed",
        "reason": "SCOPE",
        "msg": expected_msg,
        "route_visible_surfaces": [ROUTE_PATH, TEMPLATE_PATH],
        "approved_paths": [ROUTE_PATH],
        "missing_paths": [TEMPLATE_PATH],
    }


def test_final_nav_gate_nav_anchor_pass_metadata_matches_runner() -> None:
    gate_input = _example_nav_gate_input([
        ROUTE_PATH,
        TEMPLATE_PATH,
        NAV_ANCHOR_PATH,
    ])
    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
    )

    assert not gate_input.requires_evidence
    assert decision.passed
    assert decision.passed_meta == {
        "event": "final_nav_discoverability_gate_passed",
        "base_ref": BASE_REF,
        "route_visible_surfaces": [
            ROUTE_PATH,
            NAV_ANCHOR_PATH,
            TEMPLATE_PATH,
        ],
        "nav_anchor_updates": [NAV_ANCHOR_PATH],
        "reason": "nav anchor updated in iteration diff",
    }


def test_final_nav_gate_generic_defaults_are_inert() -> None:
    gate_input = final_nav_discoverability_gate_input([
        ROUTE_PATH,
        TEMPLATE_PATH,
        NAV_ANCHOR_PATH,
    ])
    decision = evaluate_final_nav_discoverability_gate(
        base_ref=BASE_REF,
        gate_input=gate_input,
    )

    assert gate_input.route_visible_surfaces == []
    assert gate_input.nav_anchor_updates == []
    assert not gate_input.requires_evidence
    assert decision.passed
    assert decision.passed_meta == {
        "event": "final_nav_discoverability_gate_passed",
        "base_ref": BASE_REF,
        "route_visible_surfaces": [],
        "nav_anchor_updates": [],
        "reason": "no route-visible surfaces in iteration diff",
    }


def test_final_nav_gate_unresolved_base_metadata_matches_runner() -> None:
    decision = unresolved_final_nav_discoverability_base_ref_decision(
        iter_branch="demo/iteration-1",
    )

    expected_msg = (
        "final nav-discoverability gate failed: could not resolve "
        "the phase/upstream diff base for iteration branch "
        "'demo/iteration-1'"
    )
    assert not decision.passed
    assert decision.message == expected_msg
    assert decision.failure_meta == {
        "event": "final_nav_discoverability_gate_failed",
        "reason": "SCOPE",
        "msg": expected_msg,
        "route_visible_surfaces": [],
        "approved_paths": [],
        "missing_paths": [],
    }


# ---------------------------------------------------------------------------
# Close-out batch objective 6 — base ref must yield a non-vacuous diff.
# ---------------------------------------------------------------------------


def _freshness(
    condition: BranchFreshnessCondition, ahead: int | None = None
) -> BranchFreshness:
    return BranchFreshness(
        branch="demo/iteration-1",
        base_ref="phase-demo",
        condition=condition,
        ahead_count=ahead,
    )


def test_final_scope_base_ref_yields_diff_requires_strictly_ahead():
    C = BranchFreshnessCondition
    # Meaningful: iteration branch carries commits the base does not have.
    assert final_scope_base_ref_yields_diff(_freshness(C.AHEAD, 2)) is True
    assert final_scope_base_ref_yields_diff(_freshness(C.DIVERGED, 1)) is True
    # Vacuous / unresolved → must fail closed.
    assert final_scope_base_ref_yields_diff(_freshness(C.FRESH, 0)) is False
    assert final_scope_base_ref_yields_diff(_freshness(C.BEHIND, 0)) is False
    assert final_scope_base_ref_yields_diff(_freshness(C.MISSING_BASE)) is False
    assert (
        final_scope_base_ref_yields_diff(_freshness(C.MISSING_BRANCH)) is False
    )


def test_unresolved_scope_decision_appends_detail():
    decision = unresolved_final_scope_base_ref_decision(
        iter_branch="demo/iteration-1",
        tasks_md_path=TASKS_MD,
        detail="base ref 'phase-demo' is behind; ... vacuous",
    )
    assert not decision.passed
    assert "vacuous" in decision.message
    assert decision.failure_meta is not None
