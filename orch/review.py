"""AI review: verdict parsing and independence enforcement.

The orchestrator invokes a reviewer adapter with the fresh diff and a
review prompt. The reviewer's stdout must end with a verdict line matching
``config.review.verdict_regex``:

    Verdict: PASS
    Verdict: CHANGES REQUIRED
    Verdict: BLOCKED

Anything else → STOP(REVIEW_MALFORMED). The parser is deliberately strict:
ambiguous verdicts produce worse outcomes than a re-run.

Independence: reviewer.family MUST differ from implementer.family when
``independence.level == model_family``. Violations STOP before invocation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from orch.config import (
    LoadedConfig,
    independence as _independence_section,
    review as _review_section,
)


class Verdict(str, Enum):
    PASS = "PASS"
    CHANGES_REQUIRED = "CHANGES REQUIRED"
    BLOCKED = "BLOCKED"


@dataclass
class ReviewParseResult:
    verdict: Verdict | None
    raw_line: str | None
    malformed: bool
    message: str = ""
    severity: str | None = None  # "should-fix" | "block" | None


# Optional severity tag. Only meaningful when verdict is CHANGES_REQUIRED.
# Absence is treated as "block" (fail-safe; matches the original fix-loop
# behaviour). Unknown tokens degrade to None.
_SEVERITY_RE = re.compile(
    r"^Severity:\s+(should-fix|block)\s*$", re.IGNORECASE
)
_SEVERITY_ANY_RE = re.compile(r"^Severity:\s*\S", re.IGNORECASE)


def _non_empty_lines_reversed(text: str) -> list[str]:
    return [ln for ln in reversed(text.splitlines()) if ln.strip()]


def parse_verdict(
    text: str, regex: str | LoadedConfig
) -> ReviewParseResult:
    """Parse the reviewer's final-line verdict, plus optional Severity tag.

    Contract:
      * Final non-empty line matches the verdict regex (current behaviour).
      * OR final non-empty line matches ``^Severity: should-fix|block$``
        AND the previous non-empty line matches the verdict regex.

    Anything else → ``malformed=True``. Reviewers that don't emit a
    Severity line keep working unchanged (severity=None).

    ``regex`` accepts the raw pattern string or a ``LoadedConfig`` — in the
    latter case the ``review(cfg)["verdict_regex"]`` accessor resolves it.
    Failure semantics are preserved either way.
    """
    if isinstance(regex, LoadedConfig):
        regex = _review_section(regex)["verdict_regex"]
    rev = _non_empty_lines_reversed(text)
    if not rev:
        return ReviewParseResult(
            verdict=None,
            raw_line=None,
            malformed=True,
            message="no non-empty line in reviewer output",
        )

    severity: str | None = None
    final = rev[0].strip()
    sev_m = _SEVERITY_RE.match(final)
    sev_any = _SEVERITY_ANY_RE.match(final)
    if sev_m:
        severity = sev_m.group(1).lower()
        if len(rev) < 2:
            return ReviewParseResult(
                verdict=None,
                raw_line=final,
                malformed=True,
                message="Severity line present but no preceding Verdict line",
            )
        verdict_line = rev[1].strip()
    elif sev_any:
        # Looks like a Severity line but not the canonical form — treat
        # token as unknown (severity=None) and require the verdict line
        # to be the line above (fail-safe: do not silently accept
        # malformed final lines).
        if len(rev) < 2:
            return ReviewParseResult(
                verdict=None,
                raw_line=final,
                malformed=True,
                message=(
                    f"unknown severity token in final line {final!r}; "
                    "no preceding Verdict line"
                ),
            )
        verdict_line = rev[1].strip()
        severity = None
    else:
        verdict_line = final

    pat = re.compile(regex)
    m = pat.match(verdict_line)
    if not m:
        return ReviewParseResult(
            verdict=None,
            raw_line=verdict_line,
            malformed=True,
            message=(
                f"verdict line does not match verdict regex: {verdict_line!r}"
            ),
        )
    label = m.group(1).strip()
    try:
        verdict = Verdict(label)
    except ValueError:
        return ReviewParseResult(
            verdict=None,
            raw_line=verdict_line,
            malformed=True,
            message=f"unknown verdict label '{label}'",
        )
    return ReviewParseResult(
        verdict=verdict,
        raw_line=verdict_line,
        malformed=False,
        severity=severity,
    )


@dataclass
class IndependenceCheck:
    ok: bool
    reason: str = ""


def check_independence(
    implementer_family: str,
    reviewer_family: str,
    level: str | LoadedConfig,
    *,
    implementer_name: str = "",
    reviewer_name: str = "",
) -> IndependenceCheck:
    """Enforce reviewer independence. ``level`` is one of session | model | model_family.

    ``level`` accepts the raw string or a ``LoadedConfig`` — in the latter
    case the ``independence(cfg)["level"]`` accessor resolves it. The
    unknown-level rejection path is preserved (e.g. an empty string or a
    misspelt value still returns ``ok=False``).
    """
    if isinstance(level, LoadedConfig):
        level = _independence_section(level)["level"]
    if level == "session":
        # Any distinct invocation counts. Always OK.
        return IndependenceCheck(ok=True)
    if level == "model":
        # Distinct adapter instance. Same family permitted, same adapter name not.
        if implementer_name and implementer_name == reviewer_name:
            return IndependenceCheck(
                ok=False,
                reason=f"independence=model requires distinct adapters; both are '{implementer_name}'",
            )
        return IndependenceCheck(ok=True)
    if level == "model_family":
        if implementer_family == reviewer_family:
            return IndependenceCheck(
                ok=False,
                reason=(
                    f"reviewer family '{reviewer_family}' matches implementer "
                    f"family — independence level 'model_family' violated"
                ),
            )
        return IndependenceCheck(ok=True)
    return IndependenceCheck(
        ok=False, reason=f"unknown independence level '{level}'"
    )


def decide_next_action(
    verdict: Verdict, *, round_num: int, max_rounds: int
) -> str:
    """Map verdict + round to a high-level action name.

    Returns one of:
      * ``"accept"`` — PASS, proceed to PR/merge.
      * ``"fix"``    — CHANGES REQUIRED within budget, run fixer.
      * ``"stop_review_fail"`` — BLOCKED, or CHANGES REQUIRED past budget.
    """
    if verdict is Verdict.PASS:
        return "accept"
    if verdict is Verdict.BLOCKED:
        return "stop_review_fail"
    # CHANGES_REQUIRED
    if round_num >= max_rounds:
        return "stop_review_fail"
    return "fix"
