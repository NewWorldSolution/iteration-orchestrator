"""Failure-triage classifier.

Pure decision logic. The runner gathers ``TriageInput`` from per-task
state + the latest review parse, calls :func:`classify`, and routes the
returned ``TriageDecision``. No I/O, no side effects.

The inputs and decision-tree contract are part of the orchestrator design
notes; any behavior change here must update those notes in the same commit.
"""
from __future__ import annotations

from dataclasses import dataclass

# Action codes — strings (not Enum) for cheap JSON round-trip on the
# event log.
ACTION_PROCEED      = "PROCEED"
ACTION_FIX_NOW      = "FIX_NOW"
ACTION_DEFER_TO_QA  = "DEFER_TO_QA"
ACTION_STOP_HUMAN   = "STOP_HUMAN"

# Severity values (mirror review.py's _SEVERITY_RE captures, lowercased).
SEVERITY_SHOULD_FIX = "should-fix"
SEVERITY_BLOCK      = "block"

DEFAULT_DEFER_BUDGET_MAX       = 2
DEFAULT_CONFIDENCE_DROP_THRESH = 0.3
REPEAT_FAILURE_WINDOW          = 3


@dataclass(frozen=True)
class TriageInput:
    verdict: str                                      # "PASS" | "CHANGES REQUIRED" | "BLOCKED"
    severity: str | None                              # "should-fix" | "block" | None
    structural_failure: bool = False
    defer_budget_used: int = 0
    defer_budget_max: int = DEFAULT_DEFER_BUDGET_MAX
    recent_findings: tuple[str, ...] = ()             # most recent last
    confidence_history: tuple[float | None, ...] = () # most recent last
    confidence_drop_threshold: float = DEFAULT_CONFIDENCE_DROP_THRESH


@dataclass(frozen=True)
class TriageDecision:
    action: str
    reason: str
    increments_defer_budget: bool = False


def classify(inp: TriageInput) -> TriageDecision:
    """Map review-round inputs to a routing decision.

    Order matters: repeat-failure and confidence-drop checks fire before
    the severity/budget logic so a pathological should-fix loop can't tie
    up the defer budget indefinitely.
    """
    # 1. PASS short-circuits — terminal success.
    if inp.verdict == "PASS":
        return TriageDecision(ACTION_PROCEED, "verdict PASS")

    # 2. BLOCKED short-circuits — terminal stop, no salvage path.
    if inp.verdict == "BLOCKED":
        return TriageDecision(ACTION_STOP_HUMAN, "verdict BLOCKED")

    # 3. Defensive: structural failures already raised this task.
    if inp.structural_failure:
        return TriageDecision(
            ACTION_STOP_HUMAN, "structural failure flag set"
        )

    # 4. Repeat-failure detection. Fingerprint identity over the
    #    last `REPEAT_FAILURE_WINDOW` rounds == fixer is making no
    #    progress; escalate.
    window = inp.recent_findings[-REPEAT_FAILURE_WINDOW:]
    if (
        len(window) >= REPEAT_FAILURE_WINDOW
        and len(set(window)) == 1
        and window[0]   # ignore empty fingerprints
    ):
        return TriageDecision(
            ACTION_STOP_HUMAN,
            f"same review finding {REPEAT_FAILURE_WINDOW} rounds in a row",
        )

    # 5. Confidence-drop trigger. Any consecutive numeric pair
    #    whose drop exceeds the threshold flags reviewer regression.
    numeric = [c for c in inp.confidence_history if c is not None]
    if len(numeric) >= 2:
        for prev, curr in zip(numeric, numeric[1:]):
            drop = prev - curr
            if drop > inp.confidence_drop_threshold:
                return TriageDecision(
                    ACTION_STOP_HUMAN,
                    f"confidence dropped {drop:.2f} > "
                    f"{inp.confidence_drop_threshold:.2f} "
                    f"({prev:.2f} → {curr:.2f})",
                )

    # 6. CHANGES REQUIRED + severity routing.
    if inp.severity == SEVERITY_SHOULD_FIX:
        if inp.defer_budget_used < inp.defer_budget_max:
            return TriageDecision(
                ACTION_DEFER_TO_QA,
                f"should-fix within budget "
                f"({inp.defer_budget_used + 1}/{inp.defer_budget_max})",
                increments_defer_budget=True,
            )
        return TriageDecision(
            ACTION_STOP_HUMAN,
            f"should-fix budget exhausted "
            f"({inp.defer_budget_used}/{inp.defer_budget_max})",
        )

    # 7. severity == "block" or None (fail-safe). Iterate until passes
    #    or the surrounding fix-loop hits its own round budget.
    return TriageDecision(
        ACTION_FIX_NOW, "block severity (or unspecified)"
    )


# --- helpers ---------------------------------------------------------------


def fingerprint_findings(text: str) -> str:
    """Stable fingerprint for repeat-failure detection.

    Hashes the normalized review-finding payload so trivial rephrasing
    (whitespace, line ordering noise) doesn't break repeat detection.
    Empty / whitespace-only input → empty string (skipped by classifier).
    """
    import hashlib
    norm = " ".join(text.split())
    if not norm:
        return ""
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()
