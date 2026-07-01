"""Tests for orch.preflight."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from orch.config import CORE_DEFAULTS, LoadedConfig
from orch.preflight import Tier, estimate, timeouts_for_tier


PF = CORE_DEFAULTS["preflight"]
TIMEOUTS = CORE_DEFAULTS["timeouts"]


def _cfg_from_defaults() -> LoadedConfig:
    return LoadedConfig(
        path=Path("test-project.yaml"), data=copy.deepcopy(CORE_DEFAULTS)
    )


def _run(files: int, lines: int):
    return estimate(
        allowed_files=[f"f{i}.py" for i in range(files)],
        prompt_text="x\n" * lines,
        preflight_cfg=PF,
    )


def test_tier_low_when_small():
    r = _run(1, 10)
    assert r.tier is Tier.LOW
    assert r.refused is False
    assert r.warnings == []


def test_verification_heavy_low_task_auto_bumps_to_medium_without_task_kind():
    r = estimate(
        allowed_files=["src/report.py", "tests/test_report.py"],
        prompt_text="small change\n",
        preflight_cfg=PF,
    )

    assert r.tier is Tier.MEDIUM
    assert any("auto-bumped from LOW to MEDIUM" in w for w in r.warnings)


def test_declared_task_kind_preserves_size_based_low_tier():
    r = estimate(
        allowed_files=["src/report.py", "tests/test_report.py"],
        prompt_text="small change\n",
        preflight_cfg=PF,
        task_kind="characterization",
    )

    assert r.tier is Tier.LOW
    assert r.warnings == []


def test_explicit_test_command_auto_bumps_low_task_to_medium():
    r = estimate(
        allowed_files=["src/report.py"],
        prompt_text="small change\n",
        preflight_cfg=PF,
        test_cmd="pytest tests/test_report.py -q",
    )

    assert r.tier is Tier.MEDIUM


def test_tier_medium_by_files():
    r = _run(PF["medium_files"], 10)
    assert r.tier is Tier.MEDIUM


def test_tier_high_by_lines():
    r = _run(1, PF["high_lines"])
    assert r.tier is Tier.HIGH


def test_tier_refuse_by_files():
    r = _run(PF["refuse_files"], 10)
    assert r.tier is Tier.REFUSE
    assert r.refused
    assert any("allowed-files" in rr for rr in r.refuse_reasons)


def test_tier_refuse_by_lines():
    r = _run(1, PF["refuse_lines"])
    assert r.refused
    assert any("prompt length" in rr for rr in r.refuse_reasons)


def test_warning_emitted_between_warn_and_refuse():
    warn_f = PF["warn_allowed_files"]
    r = _run(warn_f, 10)
    assert r.tier in (Tier.MEDIUM, Tier.HIGH)
    assert any("warn threshold" in w for w in r.warnings)


def test_no_warning_once_refused():
    r = _run(PF["refuse_files"], PF["refuse_lines"])
    # Refuse reasons populated, warnings suppressed to avoid duplicate noise
    assert r.refuse_reasons
    assert r.warnings == []


def test_timeouts_for_tier_scales_impl_and_fix():
    tlow = timeouts_for_tier(Tier.LOW, TIMEOUTS)
    tmed = timeouts_for_tier(Tier.MEDIUM, TIMEOUTS)
    thi = timeouts_for_tier(Tier.HIGH, TIMEOUTS)
    assert tlow["impl"] < tmed["impl"] < thi["impl"]
    assert tlow["fix"] < tmed["fix"] < thi["fix"]
    # review / acceptance / ci are fixed across tiers
    assert tlow["review"] == tmed["review"] == thi["review"]
    assert tlow["acceptance"] == thi["acceptance"]
    assert tlow["ci"] == thi["ci"]


# ---------------------------------------------------------------------------
# LoadedConfig dispatch
# ---------------------------------------------------------------------------


def test_timeouts_for_tier_accepts_loaded_config():
    """timeouts_for_tier(tier, cfg) resolves the slice via timeouts(cfg)."""
    cfg = _cfg_from_defaults()
    for tier in (Tier.LOW, Tier.MEDIUM, Tier.HIGH, Tier.REFUSE):
        direct = timeouts_for_tier(tier, TIMEOUTS)
        via_cfg = timeouts_for_tier(tier, cfg)
        assert direct == via_cfg


def test_timeouts_for_tier_loaded_config_missing_keys_raises():
    """Failure parity: a LoadedConfig missing timeouts keys raises KeyError."""
    bad = LoadedConfig(path=Path("test"), data={"timeouts": {}})
    with pytest.raises(KeyError):
        timeouts_for_tier(Tier.LOW, bad)
