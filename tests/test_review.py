"""Tests for orch.review."""
from __future__ import annotations

import copy
from pathlib import Path

from orch.config import CORE_DEFAULTS, LoadedConfig
from orch.review import (
    Verdict,
    check_independence,
    decide_next_action,
    parse_verdict,
)

REGEX = CORE_DEFAULTS["review"]["verdict_regex"]


def _cfg_from_defaults() -> LoadedConfig:
    return LoadedConfig(
        path=Path("test-project.yaml"), data=copy.deepcopy(CORE_DEFAULTS)
    )


def test_parse_verdict_pass():
    r = parse_verdict("blah\nVerdict: PASS\n", REGEX)
    assert r.malformed is False
    assert r.verdict is Verdict.PASS


def test_parse_verdict_changes_required():
    r = parse_verdict("stuff\n\nVerdict: CHANGES REQUIRED\n", REGEX)
    assert r.verdict is Verdict.CHANGES_REQUIRED


def test_parse_verdict_blocked():
    r = parse_verdict("Verdict: BLOCKED\n", REGEX)
    assert r.verdict is Verdict.BLOCKED


def test_verdict_followed_by_trailing_prose_is_malformed():
    """Verdict line followed by any trailing prose = malformed.

    This pins the final-non-empty-line contract from runner.py:698.
    """
    text = (
        "My analysis:\n"
        "The code looks correct.\n\n"
        "Verdict: PASS\n\n"
        "P.S. One minor nit: variable name could be more descriptive.\n"
    )
    r = parse_verdict(text, REGEX)
    assert r.malformed is True, (
        "trailing prose after Verdict line must produce malformed result"
    )
    assert r.verdict is None


def test_verdict_as_true_final_line_passes():
    """Verdict on the actual final non-empty line is accepted."""
    text = "My analysis: all good.\n\nVerdict: PASS\n"
    r = parse_verdict(text, REGEX)
    assert r.malformed is False
    assert r.verdict is Verdict.PASS


def test_verdict_changes_required_as_final_line():
    """CHANGES REQUIRED as final non-empty line is accepted."""
    text = "Found issue at line 42.\n\nVerdict: CHANGES REQUIRED\n"
    r = parse_verdict(text, REGEX)
    assert r.malformed is False
    assert r.verdict is Verdict.CHANGES_REQUIRED


def test_verdict_mid_output_only_is_malformed():
    """Verdict: PASS buried mid-output with no final verdict = malformed."""
    text = "Verdict: PASS\n\nActually let me reconsider. The approach is wrong.\n"
    r = parse_verdict(text, REGEX)
    assert r.malformed is True, (
        "verdict only in middle of output must be malformed under final-line contract"
    )


def test_empty_output_is_malformed():
    """Empty or whitespace-only output = malformed."""
    r = parse_verdict("   \n\n  \n", REGEX)
    assert r.malformed is True
    assert "no non-empty line" in r.message


def test_parse_verdict_missing_is_malformed():
    r = parse_verdict("the reviewer rambled\n", REGEX)
    assert r.malformed
    assert r.verdict is None


def test_parse_verdict_unknown_label_is_malformed():
    r = parse_verdict("Verdict: MAYBE\n", "^Verdict:\\s+(\\w+)\\s*$")
    assert r.malformed
    assert r.verdict is None


def test_independence_model_family_blocks_same_family():
    r = check_independence("anthropic", "anthropic", "model_family")
    assert not r.ok
    assert "model_family" in r.reason


def test_independence_model_family_allows_cross_family():
    r = check_independence("anthropic", "openai", "model_family")
    assert r.ok


def test_independence_model_level_blocks_same_adapter():
    r = check_independence(
        "anthropic", "anthropic", "model",
        implementer_name="claude", reviewer_name="claude",
    )
    assert not r.ok


def test_independence_model_level_allows_distinct_adapters():
    r = check_independence(
        "anthropic", "anthropic", "model",
        implementer_name="claude", reviewer_name="claude-opus",
    )
    assert r.ok


def test_independence_session_always_ok():
    r = check_independence("anthropic", "anthropic", "session")
    assert r.ok


def test_independence_unknown_level_rejects():
    r = check_independence("a", "b", "bogus")
    assert not r.ok


def test_decide_next_action():
    assert decide_next_action(Verdict.PASS, round_num=1, max_rounds=2) == "accept"
    assert decide_next_action(Verdict.CHANGES_REQUIRED, round_num=1, max_rounds=2) == "fix"
    assert decide_next_action(
        Verdict.CHANGES_REQUIRED, round_num=2, max_rounds=2
    ) == "stop_review_fail"
    assert decide_next_action(
        Verdict.BLOCKED, round_num=1, max_rounds=2
    ) == "stop_review_fail"


# ---------------------------------------------------------------------------
# LoadedConfig dispatch
# ---------------------------------------------------------------------------


def test_parse_verdict_accepts_loaded_config():
    """parse_verdict resolves the regex via review(cfg) when given a LoadedConfig."""
    cfg = _cfg_from_defaults()
    direct = parse_verdict("Verdict: PASS\n", REGEX)
    via_cfg = parse_verdict("Verdict: PASS\n", cfg)
    assert direct.verdict is via_cfg.verdict is Verdict.PASS
    assert direct.malformed is False
    assert via_cfg.malformed is False


def test_parse_verdict_loaded_config_malformed_path_matches_string_path():
    """Malformed-output behaviour is identical for both input forms."""
    cfg = _cfg_from_defaults()
    rambled = "the reviewer rambled\n"
    assert parse_verdict(rambled, REGEX).malformed is True
    assert parse_verdict(rambled, cfg).malformed is True


def test_check_independence_accepts_loaded_config_level():
    """check_independence resolves level via independence(cfg) when given LoadedConfig.

    The default ``CORE_DEFAULTS["independence"]["level"]`` is ``model_family``,
    so same-family pairs must still be rejected; cross-family must pass.
    """
    cfg = _cfg_from_defaults()
    same = check_independence("anthropic", "anthropic", cfg)
    cross = check_independence("anthropic", "openai", cfg)
    assert same.ok is False
    assert "model_family" in same.reason
    assert cross.ok is True


def test_check_independence_loaded_config_with_override_level():
    """Project overrides flow through: independence.level=session ⇒ always ok."""
    data = copy.deepcopy(CORE_DEFAULTS)
    data["independence"] = {"level": "session"}
    cfg = LoadedConfig(path=Path("test"), data=data)
    r = check_independence("anthropic", "anthropic", cfg)
    assert r.ok is True
