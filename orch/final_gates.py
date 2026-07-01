"""Pure final gate decisions for the orchestrator.

The runner owns git, filesystem reads, and state writes. This module keeps the
last-step gate decisions deterministic over values the runner already collected.
"""
from __future__ import annotations

from dataclasses import dataclass

from orch.checks import (
    NavDiscoverabilityEvidence,
    ScopeExceptionEvidence,
    check_scope,
    detect_nav_anchor_updates,
    detect_route_visible_surfaces,
)
from orch.git_ops import BranchFreshness, BranchFreshnessCondition


def final_scope_base_ref_yields_diff(freshness: BranchFreshness) -> bool:
    """True iff ``base_ref...iter_branch`` is a meaningful, non-vacuous diff.

    The final scope and nav gates compare ``base_ref...iter_branch`` (three-dot).
    That diff is only meaningful when the iteration branch carries commits the
    base ref does not already contain. It is *vacuous* — an empty changed-file
    set that would pass the gate silently — when:

      * the base ref does not resolve (``MISSING_BASE``),
      * the iteration branch does not resolve (``MISSING_BRANCH``),
      * the branch tip equals the base (``FRESH``, ``ahead == 0``), or
      * the base already contains the branch after a merge (``BEHIND``,
        ``ahead == 0``).

    Returning False here lets the runner fail the gate closed instead of
    treating a post-merge / unresolved base as "nothing leaked".
    """
    if freshness.condition in {
        BranchFreshnessCondition.MISSING_BASE,
        BranchFreshnessCondition.MISSING_BRANCH,
    }:
        return False
    return (freshness.ahead_count or 0) > 0


@dataclass(frozen=True)
class FinalScopeGateDecision:
    """Decision payload for the final outward scope gate."""

    message: str | None
    failure_meta: dict[str, object] | None = None
    exception_meta: dict[str, object] | None = None
    passed_meta: dict[str, object] | None = None

    @property
    def passed(self) -> bool:
        return self.message is None


@dataclass(frozen=True)
class FinalNavDiscoverabilityGateInput:
    """Path-pattern inputs for the final nav-discoverability gate."""

    route_visible_surfaces: list[str]
    nav_anchor_updates: list[str]
    nav_anchor_paths: list[str]

    @property
    def requires_evidence(self) -> bool:
        return bool(self.route_visible_surfaces) and not self.nav_anchor_updates


@dataclass(frozen=True)
class FinalNavDiscoverabilityGateDecision:
    """Decision payload for the final nav-discoverability gate."""

    message: str | None
    failure_meta: dict[str, object] | None = None
    exception_meta: dict[str, object] | None = None
    passed_meta: dict[str, object] | None = None

    @property
    def passed(self) -> bool:
        return self.message is None


def unresolved_final_scope_base_ref_decision(
    *,
    iter_branch: str,
    tasks_md_path: str,
    stop_reason: str = "SCOPE",
    detail: str | None = None,
) -> FinalScopeGateDecision:
    """Return the existing failure shape for an unresolved iteration base.

    ``detail`` is appended when the base ref resolved to a string but does not
    yield a usable diff (e.g. it is missing in git, or already contains the
    iteration branch after a merge), so the failure message is accurate rather
    than implying the base name itself was never derived.
    """
    msg = (
        "final scope gate failed: could not resolve the phase/upstream "
        f"diff base for iteration branch '{iter_branch}'"
    )
    if detail:
        msg = f"{msg} ({detail})"
    return FinalScopeGateDecision(
        message=msg,
        failure_meta=_final_scope_failure_meta(
            msg=msg,
            changed_files=[],
            allowed_files=[],
            tasks_md_path=tasks_md_path,
            stop_reason=stop_reason,
        ),
    )


def unresolved_final_nav_discoverability_base_ref_decision(
    *,
    iter_branch: str,
    stop_reason: str = "SCOPE",
    detail: str | None = None,
) -> FinalNavDiscoverabilityGateDecision:
    """Return the existing failure shape for an unresolved iteration base.

    ``detail`` is appended when the base ref resolved to a string but does not
    yield a usable diff (missing in git, or already contains the iteration
    branch), so the message is accurate.
    """
    msg = (
        "final nav-discoverability gate failed: could not resolve "
        "the phase/upstream diff base for iteration branch "
        f"'{iter_branch}'"
    )
    if detail:
        msg = f"{msg} ({detail})"
    return FinalNavDiscoverabilityGateDecision(
        message=msg,
        failure_meta=_final_nav_failure_meta(
            msg=msg,
            route_visible=[],
            approved=[],
            missing=[],
            stop_reason=stop_reason,
        ),
    )


def evaluate_final_outward_scope_gate(
    *,
    base_ref: str,
    changed_files: list[str],
    allowed_files: list[str],
    tasks_md_path: str,
    tasks_md_status_only: bool,
    evidence: ScopeExceptionEvidence,
    stop_reason: str = "SCOPE",
    generated_artifact_prefixes: tuple[str, ...] = ("tools/logs/",),
) -> FinalScopeGateDecision:
    """Evaluate the final accumulated outward scope gate.

    ``changed_files`` is the raw final diff file list. The runner decides
    whether ``tasks.md`` is status-only by reading git refs; this helper only
    applies that already-computed fact.
    """
    relevant_changed = _final_scope_relevant_changed_files(
        changed_files=changed_files,
        tasks_md_path=tasks_md_path,
        tasks_md_status_only=tasks_md_status_only,
        generated_artifact_prefixes=generated_artifact_prefixes,
    )
    allowed = list(allowed_files)

    if evidence.errors:
        msg = (
            "final scope gate failed: malformed scope exception evidence "
            f"at {evidence.source}: {list(evidence.errors)}"
        )
        return FinalScopeGateDecision(
            message=msg,
            failure_meta=_final_scope_failure_meta(
                msg=msg,
                changed_files=relevant_changed,
                allowed_files=allowed,
                tasks_md_path=tasks_md_path,
                stop_reason=stop_reason,
            ),
        )

    approved = evidence.approved_paths
    disallowed = check_scope(relevant_changed, [*allowed, *approved])
    if disallowed:
        msg = render_final_scope_failure(
            disallowed=disallowed,
            allowed_source=tasks_md_path,
            exception_source=evidence.source,
        )
        return FinalScopeGateDecision(
            message=msg,
            failure_meta=_final_scope_failure_meta(
                msg=msg,
                changed_files=relevant_changed,
                allowed_files=allowed,
                tasks_md_path=tasks_md_path,
                stop_reason=stop_reason,
            ),
        )

    approved_set = set(approved)
    allowed_set = set(allowed)
    approved_used = sorted(
        p for p in relevant_changed if p in approved_set and p not in allowed_set
    )
    exception_meta: dict[str, object] | None = None
    if approved_used:
        exception_meta = {
            "event": "final_scope_exception_applied",
            "paths": approved_used,
            "approved_by": evidence.approved_by,
            "reason": evidence.reason,
            "source": evidence.source,
        }

    return FinalScopeGateDecision(
        message=None,
        exception_meta=exception_meta,
        passed_meta={
            "event": "final_scope_gate_passed",
            "base_ref": base_ref,
            "changed_files": relevant_changed,
            "allowed_file_source": tasks_md_path,
            "allowed_files": allowed,
            "tasks_md_status_only": tasks_md_status_only,
        },
    )


def final_nav_discoverability_gate_input(
    changed_files: list[str],
    *,
    generated_artifact_prefixes: tuple[str, ...] = ("tools/logs/",),
    route_globs: list[str] | tuple[str, ...] | None = None,
    nav_anchor_paths: list[str] | tuple[str, ...] | None = None,
) -> FinalNavDiscoverabilityGateInput:
    """Return deterministic path-pattern inputs for the final nav gate."""
    relevant_changed = [
        p for p in changed_files
        if not _is_generated_artifact(p, generated_artifact_prefixes)
    ]
    return FinalNavDiscoverabilityGateInput(
        route_visible_surfaces=detect_route_visible_surfaces(
            relevant_changed, route_globs=route_globs
        ),
        nav_anchor_updates=detect_nav_anchor_updates(
            relevant_changed, nav_anchor_paths=nav_anchor_paths
        ),
        nav_anchor_paths=list(nav_anchor_paths or ()),
    )


def evaluate_final_nav_discoverability_gate(
    *,
    base_ref: str,
    gate_input: FinalNavDiscoverabilityGateInput,
    evidence: NavDiscoverabilityEvidence | None = None,
    stop_reason: str = "SCOPE",
) -> FinalNavDiscoverabilityGateDecision:
    """Evaluate the final accumulated nav-discoverability gate.

    The runner supplies diff files and loads operator evidence only when
    ``gate_input.requires_evidence`` is true. This helper keeps the pass/fail
    decision and report-visible metadata shape deterministic.
    """
    route_visible = gate_input.route_visible_surfaces
    nav_anchors = gate_input.nav_anchor_updates

    if not route_visible:
        return FinalNavDiscoverabilityGateDecision(
            message=None,
            passed_meta={
                "event": "final_nav_discoverability_gate_passed",
                "base_ref": base_ref,
                "route_visible_surfaces": [],
                "nav_anchor_updates": [],
                "reason": "no route-visible surfaces in iteration diff",
            },
        )

    if nav_anchors:
        return FinalNavDiscoverabilityGateDecision(
            message=None,
            passed_meta={
                "event": "final_nav_discoverability_gate_passed",
                "base_ref": base_ref,
                "route_visible_surfaces": route_visible,
                "nav_anchor_updates": nav_anchors,
                "reason": "nav anchor updated in iteration diff",
            },
        )

    if evidence is None:
        raise ValueError(
            "nav-discoverability evidence is required when no nav anchor "
            "is present"
        )

    if evidence.errors:
        base_msg = render_final_nav_discoverability_failure(
            route_visible=route_visible,
            missing=route_visible,
            evidence_source=evidence.source,
            nav_anchor_paths=gate_input.nav_anchor_paths,
        )
        msg = (
            f"{base_msg} Evidence file at {evidence.source} is "
            f"malformed: {list(evidence.errors)}"
        )
        return FinalNavDiscoverabilityGateDecision(
            message=msg,
            failure_meta=_final_nav_failure_meta(
                msg=msg,
                route_visible=route_visible,
                approved=[],
                missing=route_visible,
                stop_reason=stop_reason,
            ),
        )

    approved_paths = evidence.approved_paths
    approved = set(approved_paths)
    missing = [p for p in route_visible if p not in approved]
    if approved and not approved_paths:
        # Should not happen — approved_paths guards on .ok. Defensive.
        missing = list(route_visible)
    if missing or not approved_paths:
        msg = render_final_nav_discoverability_failure(
            route_visible=route_visible,
            missing=missing or route_visible,
            evidence_source=evidence.source,
            nav_anchor_paths=gate_input.nav_anchor_paths,
        )
        return FinalNavDiscoverabilityGateDecision(
            message=msg,
            failure_meta=_final_nav_failure_meta(
                msg=msg,
                route_visible=route_visible,
                approved=approved_paths,
                missing=missing,
                stop_reason=stop_reason,
            ),
        )

    return FinalNavDiscoverabilityGateDecision(
        message=None,
        exception_meta={
            "event": "final_nav_discoverability_exception_applied",
            "base_ref": base_ref,
            "route_visible_surfaces": route_visible,
            "approved_paths": approved_paths,
            "approved_by": evidence.approved_by,
            "reason": evidence.reason,
            "source": evidence.source,
        },
    )


def render_final_scope_failure(
    *,
    disallowed: list[str],
    allowed_source: str,
    exception_source: str,
) -> str:
    return (
        "final scope gate failed: disallowed files "
        f"{disallowed}; allowed-file source is the union of every "
        f"'Allowed files' block in {allowed_source}; explicit operator "
        f"exceptions must be listed in {exception_source} with "
        "approved_by, reason, and exact relative paths."
    )


def render_final_nav_discoverability_failure(
    *,
    route_visible: list[str],
    missing: list[str],
    evidence_source: str,
    nav_anchor_paths: list[str] | tuple[str, ...] | None = None,
) -> str:
    anchors = list(nav_anchor_paths or ())
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
        f"{remediation} or list these paths in {evidence_source} with "
        "approved_by, "
        "reason, and exact relative paths."
    )


def _final_scope_relevant_changed_files(
    *,
    changed_files: list[str],
    tasks_md_path: str,
    tasks_md_status_only: bool,
    generated_artifact_prefixes: tuple[str, ...],
) -> list[str]:
    relevant = [
        p for p in changed_files
        if not _is_generated_artifact(p, generated_artifact_prefixes)
    ]
    if tasks_md_status_only:
        relevant = [p for p in relevant if p != tasks_md_path]
    return relevant


def _is_generated_artifact(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def _final_scope_failure_meta(
    *,
    msg: str,
    changed_files: list[str],
    allowed_files: list[str],
    tasks_md_path: str,
    stop_reason: str,
) -> dict[str, object]:
    return {
        "event": "final_scope_gate_failed",
        "reason": stop_reason,
        "msg": msg,
        "changed_files": changed_files,
        "allowed_file_source": tasks_md_path,
        "allowed_files": allowed_files,
    }


def _final_nav_failure_meta(
    *,
    msg: str,
    route_visible: list[str],
    approved: list[str],
    missing: list[str],
    stop_reason: str,
) -> dict[str, object]:
    return {
        "event": "final_nav_discoverability_gate_failed",
        "reason": stop_reason,
        "msg": msg,
        "route_visible_surfaces": route_visible,
        "approved_paths": approved,
        "missing_paths": missing,
    }
