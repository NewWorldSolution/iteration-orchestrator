"""Pre-flight task size estimate.

Pure file reads + arithmetic — **never invokes a model**. Given the parsed
task (allowed files, prompt text) and the merged config, returns a size
tier and any warn/refuse diagnostics.

Tiers drive timeout scaling (C3) downstream:

    LOW    → impl_low,    fix_low
    MEDIUM → impl_medium, fix_medium
    HIGH   → impl_high,   fix_high
    REFUSE → STOP(PREFLIGHT_SIZE) before any invocation

Thresholds come from ``config.preflight.*`` (see CORE_DEFAULTS).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from orch.config import LoadedConfig, timeouts as _timeouts_section


class Tier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    REFUSE = "REFUSE"


@dataclass
class PreflightResult:
    tier: Tier
    num_allowed_files: int
    prompt_lines: int
    warnings: list[str] = field(default_factory=list)
    refuse_reasons: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.tier is Tier.REFUSE


def estimate(
    *,
    allowed_files: list[str],
    prompt_text: str,
    preflight_cfg: dict,
    test_cmd: str | None = None,
    task_kind: str | None = None,
) -> PreflightResult:
    """Classify a task into a size tier and emit warn/refuse diagnostics."""
    nf = len(allowed_files)
    nl = _count_nonempty_lines(prompt_text)

    low_f = preflight_cfg.get("low_files", 2)
    low_l = preflight_cfg.get("low_lines", 80)
    med_f = preflight_cfg.get("medium_files", 5)
    med_l = preflight_cfg.get("medium_lines", 150)
    high_f = preflight_cfg.get("high_files", 8)
    high_l = preflight_cfg.get("high_lines", 250)
    ref_f = preflight_cfg.get("refuse_files", 12)
    ref_l = preflight_cfg.get("refuse_lines", 400)
    warn_f = preflight_cfg.get("warn_allowed_files", 6)
    warn_l = preflight_cfg.get("warn_prompt_lines", 150)

    refuse: list[str] = []
    if nf >= ref_f:
        refuse.append(
            f"allowed-files count {nf} ≥ refuse threshold {ref_f}"
        )
    if nl >= ref_l:
        refuse.append(
            f"prompt length {nl} lines ≥ refuse threshold {ref_l}"
        )

    warnings: list[str] = []
    if nf >= warn_f and not refuse:
        warnings.append(f"allowed-files count {nf} ≥ warn threshold {warn_f}")
    if nl >= warn_l and not refuse:
        warnings.append(f"prompt length {nl} lines ≥ warn threshold {warn_l}")

    if refuse:
        tier = Tier.REFUSE
    elif nf >= high_f or nl >= high_l:
        tier = Tier.HIGH
    elif nf >= med_f or nl >= med_l:
        tier = Tier.MEDIUM
    elif nf >= low_f or nl >= low_l:
        tier = Tier.LOW
    else:
        tier = Tier.LOW

    if (
        tier is Tier.LOW
        and task_kind is None
        and _looks_verification_heavy(allowed_files, test_cmd)
    ):
        tier = Tier.MEDIUM
        warnings.append(
            "verification-heavy task without task_kind auto-bumped "
            "from LOW to MEDIUM"
        )

    return PreflightResult(
        tier=tier,
        num_allowed_files=nf,
        prompt_lines=nl,
        warnings=warnings,
        refuse_reasons=refuse,
    )


def timeouts_for_tier(
    tier: Tier, timeouts_cfg: dict | LoadedConfig
) -> dict:
    """Resolve tier-scaled impl / fix / review / acceptance / ci timeouts.

    ``timeouts_cfg`` accepts either the already-sliced ``timeouts`` section
    or a ``LoadedConfig`` — in the latter case the ``timeouts(cfg)``
    accessor extracts the section. Failure behaviour is unchanged: a
    missing key still surfaces as ``KeyError``.
    """
    if isinstance(timeouts_cfg, LoadedConfig):
        timeouts_cfg = _timeouts_section(timeouts_cfg)
    if tier is Tier.LOW:
        impl = timeouts_cfg["impl_low"]
        fix = timeouts_cfg["fix_low"]
    elif tier is Tier.MEDIUM:
        impl = timeouts_cfg["impl_medium"]
        fix = timeouts_cfg["fix_medium"]
    else:  # HIGH or REFUSE (no invocation will run; keep HIGH numbers)
        impl = timeouts_cfg["impl_high"]
        fix = timeouts_cfg["fix_high"]
    return {
        "impl": impl,
        "fix": fix,
        "review": timeouts_cfg["review"],
        "acceptance": timeouts_cfg["acceptance"],
        "ci": timeouts_cfg["ci"],
    }


def _count_nonempty_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _looks_verification_heavy(
    allowed_files: list[str], test_cmd: str | None,
) -> bool:
    if test_cmd and _is_runnable_test_command(test_cmd):
        return True
    return any(_is_test_path(path) for path in allowed_files)


def _is_runnable_test_command(command: str) -> bool:
    normalized = command.strip().lower()
    if not normalized or normalized == "true":
        return False
    return any(
        token in normalized
        for token in ("pytest", "unittest", "tox", "vitest", "jest", "npm test")
    )


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip().lower()
    parts = [part for part in normalized.split("/") if part]
    if any(part in {"test", "tests", "__tests__"} for part in parts[:-1]):
        return True
    name = parts[-1] if parts else normalized
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".spec.js")
        or name.endswith(".test.ts")
        or name.endswith(".spec.ts")
    )
