"""Tests for orch.cost."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from orch.config import CORE_DEFAULTS, LoadedConfig
from orch.cost import (
    CostLogger,
    compute_cost_usd,
    estimate_tokens,
    load_records,
    sanitize_persisted_extra,
    summarize_costs,
)

RATES = {
    "anthropic": {"input": 3.0, "output": 15.0},
    "openai":    {"input": 2.5, "output": 10.0},
}


def _cfg_with_costs(rates: dict[str, dict[str, float]]) -> LoadedConfig:
    """Build a LoadedConfig whose ``costs`` section equals ``rates``."""
    data = copy.deepcopy(CORE_DEFAULTS)
    data["costs"] = copy.deepcopy(rates)
    return LoadedConfig(path=Path("test-project.yaml"), data=data)


def test_estimate_tokens_basic():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    # 4 chars == 1 token heuristic
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_compute_cost_usd():
    # 1M input tokens at $3 = $3
    assert compute_cost_usd("anthropic", 1_000_000, 0, RATES) == 3.0
    # 1M output tokens at $15 = $15
    assert compute_cost_usd("anthropic", 0, 1_000_000, RATES) == 15.0
    # Mixed
    c = compute_cost_usd("openai", 500_000, 100_000, RATES)
    assert abs(c - (0.5 * 2.5 + 0.1 * 10.0)) < 1e-9


def test_compute_cost_unknown_family_returns_zero():
    assert compute_cost_usd("martian", 1000, 1000, RATES) == 0.0


def test_record_exact_tokens(tmp_path: Path):
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    rec = log.record(
        task="I4-T1",
        step="IMPL",
        agent="claude",
        family="anthropic",
        input_tokens=10_000,
        output_tokens=2_000,
        exact=True,
        duration_s=45.2,
        exit_code=0,
        model="claude-sonnet-4-6",
        prompt_path="iterations/x/prompts/t1.md",
    )
    assert rec.estimated is False
    assert rec.partial is False
    assert rec.est_cost_usd > 0
    # File line written
    records = load_records(tmp_path / "cost.jsonl")
    assert len(records) == 1
    assert records[0]["task"] == "I4-T1"
    assert records[0]["step"] == "IMPL"
    assert records[0]["estimated"] is False


def test_sanitize_persisted_extra_strips_raw_keeps_parsed():
    extra = {
        "model_routing": {"tier": "max"},
        "raw_terminal_json": "x" * 100_000,
        "agent_result_extra": {
            "raw_terminal_json": "x" * 100_000,
            "usage_parser_status": "parsed",
            "usage_model": "claude-json-model",
            "dispatched_model": "claude-dispatched",
        },
    }
    cleaned = sanitize_persisted_extra(extra)
    # Bulky raw blob dropped at the top level and inside agent_result_extra.
    assert "raw_terminal_json" not in cleaned
    assert "raw_terminal_json" not in cleaned["agent_result_extra"]
    # Every other key — routing + parsed usage metadata — is preserved.
    assert cleaned["model_routing"] == {"tier": "max"}
    assert cleaned["agent_result_extra"]["usage_parser_status"] == "parsed"
    assert cleaned["agent_result_extra"]["usage_model"] == "claude-json-model"
    assert cleaned["agent_result_extra"]["dispatched_model"] == "claude-dispatched"
    # The input mapping is not mutated.
    assert "raw_terminal_json" in extra
    assert "raw_terminal_json" in extra["agent_result_extra"]


def test_sanitize_persisted_extra_handles_empty_and_missing_nested():
    assert sanitize_persisted_extra(None) == {}
    assert sanitize_persisted_extra({}) == {}
    # Non-dict agent_result_extra is left untouched (only dicts are descended).
    assert sanitize_persisted_extra({"agent_result_extra": "n/a"}) == {
        "agent_result_extra": "n/a"
    }


def test_record_strips_raw_terminal_json_but_keeps_usage_fields(tmp_path: Path):
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    log.record(
        task="I4-T1",
        step="REVIEW",
        agent="claude",
        family="anthropic",
        input_tokens=10_000,
        output_tokens=2_000,
        exact=True,
        duration_s=12.0,
        exit_code=0,
        model="claude-json-model",
        cached_input_tokens=300,
        cache_creation_input_tokens=400,
        reasoning_output_tokens=50,
        parser_status="parsed",
        extra={
            "role": "synthesis",
            "raw_terminal_json": "x" * 100_000,
            "agent_result_extra": {
                "raw_terminal_json": "x" * 100_000,
                "usage_parser_status": "parsed",
            },
        },
    )
    [persisted] = load_records(tmp_path / "cost.jsonl")
    # Raw blob is absent on disk, top-level and nested.
    assert "raw_terminal_json" not in persisted["extra"]
    assert "raw_terminal_json" not in persisted["extra"]["agent_result_extra"]
    # Sibling extra context survives.
    assert persisted["extra"]["role"] == "synthesis"
    assert (
        persisted["extra"]["agent_result_extra"]["usage_parser_status"] == "parsed"
    )
    # Parsed usage columns are intact.
    assert persisted["model"] == "claude-json-model"
    assert persisted["input_tokens"] == 10_000
    assert persisted["output_tokens"] == 2_000
    assert persisted["cached_input_tokens"] == 300
    assert persisted["cache_creation_input_tokens"] == 400
    assert persisted["reasoning_output_tokens"] == 50
    assert persisted["parser_status"] == "parsed"


def test_record_estimated_and_partial(tmp_path: Path):
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    # Simulate a timeout-killed invocation with estimated tokens
    rec = log.record(
        task="I4-T2",
        step="IMPL",
        agent="codex",
        family="openai",
        input_tokens=estimate_tokens("x" * 2000),
        output_tokens=estimate_tokens("y" * 800),
        exact=False,
        duration_s=2700.0,
        exit_code=-15,
        partial=True,
    )
    assert rec.estimated is True
    assert rec.partial is True
    assert rec.input_tokens == 500
    assert rec.output_tokens == 200


def test_fix_record_carries_cause(tmp_path: Path):
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    log.record(
        task="I4-T1",
        step="FIX",
        agent="claude",
        family="anthropic",
        input_tokens=500,
        output_tokens=200,
        exact=True,
        duration_s=30.0,
        exit_code=0,
        cause="acceptance",
    )
    log.record(
        task="I4-T1",
        step="FIX",
        agent="claude",
        family="anthropic",
        input_tokens=500,
        output_tokens=200,
        exact=True,
        duration_s=30.0,
        exit_code=0,
        cause="review",
    )
    rows = load_records(tmp_path / "cost.jsonl")
    assert [r["cause"] for r in rows] == ["acceptance", "review"]


def test_summarize_empty(tmp_path: Path):
    s = summarize_costs(tmp_path / "cost.jsonl")
    assert s.total_usd == 0.0
    assert s.invocations == 0
    assert s.by_task == {} == s.by_step == s.by_agent


def test_summarize_aggregates(tmp_path: Path):
    log = CostLogger(tmp_path / "cost.jsonl", RATES, iteration="demo-i1")
    for i in range(3):
        log.record(
            task=f"I4-T{i+1}",
            step="IMPL",
            agent="claude",
            family="anthropic",
            input_tokens=1_000_000,
            output_tokens=0,
            exact=True,
            duration_s=10.0,
            exit_code=0,
        )
    log.record(
        task="I4-T1",
        step="REVIEW",
        agent="codex",
        family="openai",
        input_tokens=1_000_000,
        output_tokens=0,
        exact=True,
        duration_s=10.0,
        exit_code=0,
    )
    s = summarize_costs(tmp_path / "cost.jsonl")
    assert s.invocations == 4
    # 3 claude IMPL at $3 + 1 codex REVIEW at $2.5 = $11.5
    assert abs(s.total_usd - 11.5) < 1e-6
    assert set(s.by_task) == {"I4-T1", "I4-T2", "I4-T3"}
    assert s.by_step["IMPL"] == 9.0
    assert s.by_step["REVIEW"] == 2.5
    assert s.estimated_count == 0
    assert s.partial_count == 0


def test_jsonl_is_append_only(tmp_path: Path):
    path = tmp_path / "cost.jsonl"
    log = CostLogger(path, RATES, iteration="x")
    for i in range(3):
        log.record(
            task="T1", step="IMPL", agent="a", family="anthropic",
            input_tokens=10, output_tokens=10, exact=True,
            duration_s=1, exit_code=0,
        )
    text = path.read_text().splitlines()
    assert len(text) == 3
    for line in text:
        # Each line is standalone JSON
        json.loads(line)


# ---------------------------------------------------------------------------
# LoadedConfig dispatch
# ---------------------------------------------------------------------------


def test_compute_cost_usd_accepts_loaded_config():
    """compute_cost_usd resolves rates via costs(cfg) when given a LoadedConfig."""
    cfg = _cfg_with_costs(RATES)
    direct = compute_cost_usd("anthropic", 1_000_000, 0, RATES)
    via_cfg = compute_cost_usd("anthropic", 1_000_000, 0, cfg)
    assert direct == via_cfg == 3.0


def test_compute_cost_unknown_family_via_loaded_config_returns_zero():
    """Unknown-family fallback (0.0) is preserved across both input shapes."""
    cfg = _cfg_with_costs(RATES)
    assert compute_cost_usd("martian", 1_000, 1_000, cfg) == 0.0


def test_cost_logger_accepts_loaded_config(tmp_path: Path):
    """CostLogger pulls the cost table from costs(cfg) when given a LoadedConfig."""
    cfg = _cfg_with_costs(RATES)
    log = CostLogger(tmp_path / "cost.jsonl", cfg, iteration="demo-i1")
    assert log.cost_table == RATES
    rec = log.record(
        task="I4-T1",
        step="IMPL",
        agent="claude",
        family="anthropic",
        input_tokens=1_000_000,
        output_tokens=0,
        exact=True,
        duration_s=1.0,
        exit_code=0,
    )
    # 1M input tokens at $3 → $3 — identical to the sliced-dict path.
    assert rec.est_cost_usd == 3.0


def test_cost_logger_sliced_and_loaded_config_records_match(tmp_path: Path):
    """Same invocation produces identical CostRecord for dict vs LoadedConfig input."""
    cfg = _cfg_with_costs(RATES)
    log_direct = CostLogger(tmp_path / "a.jsonl", RATES, iteration="iter")
    log_via_cfg = CostLogger(tmp_path / "b.jsonl", cfg, iteration="iter")
    kwargs = dict(
        task="T", step="IMPL", agent="codex", family="openai",
        input_tokens=500_000, output_tokens=100_000,
        exact=True, duration_s=2.0, exit_code=0,
    )
    a = log_direct.record(**kwargs)
    b = log_via_cfg.record(**kwargs)
    assert a.est_cost_usd == b.est_cost_usd
    assert a.family == b.family
    assert a.input_tokens == b.input_tokens


# ---------------------------------------------------------------------------
# T1 cost-tracking: per-model/cache-aware pricing and CLI usage fixtures
# ---------------------------------------------------------------------------


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "cost_usage"

MODEL_RATES = {
    "anthropic": {
        "input": "3.0",
        "output": "15.0",
        "cache_read_input": "0.30",
        "cache_creation_input": "3.75",
        "models": {
            "claude-sonnet-4-5": {
                "input": "3.0",
                "output": "15.0",
                "cache_read_input": "0.30",
                "cache_creation_input": "3.75",
            },
        },
    },
    "openai": {
        "input": "2.0",
        "cached_input": "0.20",
        "output": "8.0",
        "models": {
            "gpt-5.4": {
                "input": "2.0",
                "cached_input": "0.20",
                "output": "8.0",
            },
        },
    },
}


def test_compute_cost_uses_model_rates_before_family_fallback():
    table = {
        "openai": {
            "input": "2.0",
            "output": "8.0",
            "models": {
                "gpt-expensive": {"input": "5.0", "output": "20.0"},
            },
        },
    }

    model_cost = compute_cost_usd(
        "openai",
        1_000_000,
        1_000_000,
        table,
        model="gpt-expensive",
    )
    fallback_cost = compute_cost_usd(
        "openai",
        1_000_000,
        1_000_000,
        {"openai": {"input": "2.0", "output": "8.0"}},
        model="gpt-expensive",
    )

    assert model_cost == 25.0
    assert fallback_cost == 10.0


def test_parse_claude_final_usage_fixture_and_anthropic_cache_math(
    tmp_path: Path,
):
    from orch.cost import parse_agent_usage

    usage = parse_agent_usage(
        "claude",
        (FIXTURE_DIR / "claude_final_usage.json").read_text(),
    )

    assert usage.exact is True
    assert usage.parser_status == "parsed"
    assert usage.model == "claude-sonnet-4-5"
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 200
    assert usage.cached_input_tokens == 300
    assert usage.cache_creation_input_tokens == 400

    log = CostLogger(tmp_path / "cost.jsonl", MODEL_RATES, iteration="iter")
    rec = log.record(
        task="T1",
        step="IMPL",
        agent="claude",
        family="anthropic",
        provider=usage.provider,
        model=usage.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_output_tokens=usage.reasoning_output_tokens,
        exact=usage.exact,
        parser_status=usage.parser_status,
        parser_warning="; ".join(usage.warnings) or None,
        duration_s=1,
        exit_code=0,
    )

    # (1000 * 3) + (300 * .30) + (400 * 3.75) + (200 * 15)
    assert rec.est_cost_usd == 0.00759
    assert rec.cost_unknown is False
    assert rec.cached_input_tokens == 300
    assert rec.cache_creation_input_tokens == 400


def test_parse_codex_turn_completed_fixture_and_openai_cache_subset_math(
    tmp_path: Path,
):
    from orch.cost import parse_agent_usage

    usage = parse_agent_usage(
        "codex",
        (FIXTURE_DIR / "codex_turn_completed.jsonl").read_text(),
    )

    assert usage.exact is True
    assert usage.parser_status == "parsed"
    assert usage.model == "gpt-5.4"
    assert usage.input_tokens == 9000
    assert usage.cached_input_tokens == 3000
    assert usage.output_tokens == 1200
    assert usage.reasoning_output_tokens == 500

    log = CostLogger(tmp_path / "cost.jsonl", MODEL_RATES, iteration="iter")
    rec = log.record(
        task="T1",
        step="REVIEW",
        agent="codex",
        family="openai",
        provider=usage.provider,
        model=usage.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        reasoning_output_tokens=usage.reasoning_output_tokens,
        exact=usage.exact,
        parser_status=usage.parser_status,
        parser_warning="; ".join(usage.warnings) or None,
        duration_s=1,
        exit_code=0,
    )

    # Source: https://developers.openai.com/api/docs/guides/reasoning
    # Responses usage reports reasoning under output token details;
    # output_tokens already includes those reasoning tokens. Bill output once:
    # ((9000 - 3000) * 2) + (3000 * .20) + (1200 * 8).
    assert rec.est_cost_usd == 0.0222
    assert rec.reasoning_output_tokens == 500


def test_unknown_model_records_tokens_and_marks_cost_unknown(tmp_path: Path):
    log = CostLogger(tmp_path / "cost.jsonl", MODEL_RATES, iteration="iter")
    rec = log.record(
        task="T1",
        step="IMPL",
        agent="codex",
        family="openai",
        model="not-in-rate-table",
        input_tokens=123,
        output_tokens=45,
        exact=True,
        duration_s=1,
        exit_code=0,
    )

    assert rec.input_tokens == 123
    assert rec.output_tokens == 45
    assert rec.est_cost_usd is None
    assert rec.cost_unknown is True
    assert "unknown model" in (rec.cost_warning or "")

    summary = summarize_costs(tmp_path / "cost.jsonl")
    assert summary.total_usd == 0.0
    assert summary.unknown_count == 1
    detail = summary.by_provider_model["openai/not-in-rate-table"]
    assert detail["unknown_count"] == 1
    assert detail["input_tokens"] == 123


def test_parse_malformed_usage_returns_warning_without_raising():
    from orch.cost import parse_agent_usage

    usage = parse_agent_usage("codex", "{not json}\n")

    assert usage.exact is False
    assert usage.parser_status == "no_usage"
    assert usage.input_tokens == 0
    assert usage.warnings


def test_summarize_costs_exposes_json_serializable_model_and_agent_rollups(
    tmp_path: Path,
):
    from dataclasses import asdict

    log = CostLogger(tmp_path / "cost.jsonl", MODEL_RATES, iteration="iter")
    log.record(
        task="T1",
        step="IMPL",
        agent="claude",
        family="anthropic",
        provider="claude",
        model="claude-sonnet-4-5",
        input_tokens=1000,
        output_tokens=200,
        cached_input_tokens=300,
        cache_creation_input_tokens=400,
        exact=True,
        duration_s=1,
        exit_code=0,
    )
    log.record(
        task="T1",
        step="REVIEW",
        agent="codex",
        family="openai",
        provider="codex",
        model="gpt-5.4",
        input_tokens=9000,
        output_tokens=1200,
        cached_input_tokens=3000,
        reasoning_output_tokens=500,
        exact=True,
        duration_s=1,
        exit_code=0,
    )

    summary = summarize_costs(tmp_path / "cost.jsonl")

    json.dumps(asdict(summary))
    assert summary.by_provider_model["claude/claude-sonnet-4-5"][
        "cached_input_tokens"
    ] == 300
    assert summary.by_provider_model["codex/gpt-5.4"][
        "reasoning_output_tokens"
    ] == 500
    assert summary.by_agent_detail["claude"]["cost_usd"] == 0.0076
    assert summary.by_agent_detail["codex"]["cost_usd"] == 0.0222
