"""Per-invocation cost logging.

Every model invocation appends one JSON line
to ``tools/logs/<iter>/cost.jsonl`` — no aggregation service, no billing
platform. Totals are recomputed from the file when the readiness report
runs.

Token source priority:
    1. Exact counts reported by the CLI (``exact=True`` on ``record``)
    2. ``cost_regex`` captured from agent stdout (caller resolves; passes
       ``exact=True`` when resolved)
    3. Fallback: ``len(chars) // 4`` with ``estimated: True``

Rates are operator-maintained in ``.orch/project.yaml``.
Every cost figure in the readiness report is rendered under an
"estimated" banner so it is not treated as authoritative.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from orch.config import LoadedConfig, costs as _costs_section
from orch.state import utcnow_iso

# Per-1M-token rates live in project.yaml. CostTable is a simple alias.
CostTable = dict[str, Any]
HONEST_COST_LABEL = (
    "estimated equivalent API cost "
    "(subscription — not billed per request)"
)
_PER_MILLION = Decimal("1000000")
_RECORD_PLACES = Decimal("0.000001")
_SUMMARY_PLACES = Decimal("0.0001")


def _resolve_cost_table(table: CostTable | LoadedConfig) -> CostTable:
    """Accept either a sliced CostTable or a LoadedConfig.

    Routes LoadedConfig inputs through the ``costs(cfg)`` accessor so the
    consumer never reaches into ``cfg.data`` directly.
    """
    if isinstance(table, LoadedConfig):
        return _costs_section(table)
    return table


def estimate_tokens(text: str | None) -> int:
    """Heuristic: len(chars) // 4. Returns 0 for empty/None input."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _rate_value(rates: dict[str, Any], *names: str) -> Decimal:
    for name in names:
        value = rates.get(name)
        if value is not None:
            return Decimal(str(value))
    return Decimal("0")


@dataclass(frozen=True)
class ParsedUsage:
    provider: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_output_tokens: int = 0
    exact: bool = False
    parser_status: str = "parse_failed"
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostComputation:
    cost_usd: Decimal | None
    cost_unknown: bool
    warning: str | None
    rate_source: str | None


def _usage_failure(provider: str, status: str, warning: str) -> ParsedUsage:
    return ParsedUsage(
        provider=provider,
        exact=False,
        parser_status=status,
        warnings=(warning,),
    )


def _nested_int(data: dict[str, Any], path: tuple[str, ...]) -> int:
    cur: Any = data
    for part in path:
        if not isinstance(cur, dict):
            return 0
        cur = cur.get(part)
    return _non_negative_int(cur)


def _openai_cached_tokens(usage: dict[str, Any]) -> int:
    return max(
        _non_negative_int(usage.get("cached_input_tokens")),
        _nested_int(usage, ("input_tokens_details", "cached_tokens")),
        _nested_int(usage, ("prompt_tokens_details", "cached_tokens")),
    )


def _openai_reasoning_tokens(usage: dict[str, Any]) -> int:
    return max(
        _non_negative_int(usage.get("reasoning_output_tokens")),
        _nested_int(usage, ("output_tokens_details", "reasoning_tokens")),
    )


def parse_agent_usage(provider: str, raw_json: str) -> ParsedUsage:
    """Parse terminal Claude/Codex CLI usage without raising.

    Claude reports one final JSON object with a ``usage`` mapping. Codex
    reports JSONL events; only terminal ``turn.completed`` events are
    usage-authoritative.
    """
    provider_key = provider.strip().lower()
    if not raw_json or not raw_json.strip():
        return _usage_failure(provider_key, "no_usage", "empty usage payload")

    if provider_key in {"claude", "anthropic"}:
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            return _usage_failure(
                provider_key,
                "parse_failed",
                f"claude usage JSON parse failed: {exc}",
            )
        if not isinstance(payload, dict):
            return _usage_failure(
                provider_key, "parse_failed", "claude usage payload is not an object"
            )
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return _usage_failure(
                provider_key, "no_usage", "claude final JSON has no usage object"
            )
        return ParsedUsage(
            provider=provider_key,
            model=payload.get("model") or usage.get("model"),
            input_tokens=_non_negative_int(usage.get("input_tokens")),
            output_tokens=_non_negative_int(usage.get("output_tokens")),
            cached_input_tokens=_non_negative_int(
                usage.get("cache_read_input_tokens")
            ),
            cache_creation_input_tokens=_non_negative_int(
                usage.get("cache_creation_input_tokens")
            ),
            reasoning_output_tokens=0,
            exact=True,
            parser_status="parsed",
        )

    if provider_key in {"codex", "openai"}:
        warnings: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cached_input_tokens = 0
        reasoning_output_tokens = 0
        model: str | None = None
        completed = 0
        for line_no, line in enumerate(raw_json.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                warnings.append(f"codex JSONL line {line_no} parse failed: {exc}")
                continue
            if not isinstance(event, dict):
                warnings.append(f"codex JSONL line {line_no} is not an object")
                continue

            msg = event.get("msg") if isinstance(event.get("msg"), dict) else {}
            event_type = (
                event.get("type")
                or event.get("event")
                or msg.get("type")
                or msg.get("event")
            )
            if event_type != "turn.completed":
                continue

            usage = event.get("usage") or msg.get("usage")
            if not isinstance(usage, dict):
                warnings.append(
                    f"codex turn.completed line {line_no} has no usage object"
                )
                continue

            event_model = event.get("model") or msg.get("model") or usage.get("model")
            if event_model:
                if model and event_model != model:
                    warnings.append(
                        f"codex completed turns reported multiple models: "
                        f"{model}, {event_model}"
                    )
                model = model or str(event_model)

            completed += 1
            input_tokens += _non_negative_int(usage.get("input_tokens"))
            output_tokens += _non_negative_int(usage.get("output_tokens"))
            cached_input_tokens += _openai_cached_tokens(usage)
            reasoning_output_tokens += _openai_reasoning_tokens(usage)

        if completed == 0:
            return ParsedUsage(
                provider=provider_key,
                exact=False,
                parser_status="no_usage",
                warnings=tuple(warnings or ["codex JSONL had no turn.completed usage"]),
            )
        status = "parsed_with_warnings" if warnings else "parsed"
        return ParsedUsage(
            provider=provider_key,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=0,
            reasoning_output_tokens=reasoning_output_tokens,
            exact=True,
            parser_status=status,
            warnings=tuple(warnings),
        )

    return _usage_failure(
        provider_key, "unsupported_provider", f"unsupported usage provider: {provider}"
    )


def _family_rates(
    family: str,
    model: str | None,
    table: CostTable | LoadedConfig,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    resolved = _resolve_cost_table(table)
    family_rates = resolved.get(family)
    if not isinstance(family_rates, dict):
        return None, None, f"unknown cost family: {family}"

    base_rates = {
        key: value for key, value in family_rates.items()
        if key != "models"
    }
    models = family_rates.get("models")
    if model and isinstance(models, dict) and models:
        model_rates = models.get(model)
        if not isinstance(model_rates, dict):
            known = ", ".join(sorted(str(k) for k in models)) or "(none)"
            return (
                None,
                None,
                f"unknown model for cost table: {family}/{model} "
                f"(known: {known})",
            )
        merged = dict(base_rates)
        merged.update(model_rates)
        return merged, "model", None
    return base_rates, "family", None


def _compute_cost(
    family: str,
    input_tokens: int,
    output_tokens: int,
    table: CostTable | LoadedConfig,
    *,
    model: str | None = None,
    provider: str | None = None,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
) -> CostComputation:
    rates, rate_source, warning = _family_rates(family, model, table)
    if rates is None:
        return CostComputation(
            cost_usd=None,
            cost_unknown=True,
            warning=warning,
            rate_source=None,
        )

    billing_family = (family or provider or "").lower()
    input_count = _non_negative_int(input_tokens)
    output_count = _non_negative_int(output_tokens)
    cached_count = _non_negative_int(cached_input_tokens)
    creation_count = _non_negative_int(cache_creation_input_tokens)
    warnings: list[str] = []

    if billing_family == "anthropic":
        cost_units = (
            Decimal(input_count) * _rate_value(rates, "input")
            + Decimal(cached_count)
            * _rate_value(rates, "cache_read_input", "cached_input", "input")
            + Decimal(creation_count)
            * _rate_value(rates, "cache_creation_input", "input")
            + Decimal(output_count) * _rate_value(rates, "output")
        )
    elif billing_family == "openai":
        billable_cached = cached_count
        if billable_cached > input_count:
            billable_cached = input_count
            warnings.append("cached_input_tokens exceeded input_tokens; clamped")
        billable_uncached = input_count - billable_cached
        # OpenAI reports reasoning under output token details; docs state
        # reasoning is billed as output, so output_tokens is charged once.
        _ = reasoning_output_tokens
        cost_units = (
            Decimal(billable_uncached) * _rate_value(rates, "input")
            + Decimal(billable_cached)
            * _rate_value(rates, "cached_input", "cache_read_input", "input")
            + Decimal(output_count) * _rate_value(rates, "output")
        )
        if creation_count:
            warnings.append("openai cache creation tokens are recorded but not billed")
    else:
        cost_units = (
            Decimal(input_count) * _rate_value(rates, "input")
            + Decimal(cached_count)
            * _rate_value(rates, "cache_read_input", "cached_input")
            + Decimal(creation_count) * _rate_value(rates, "cache_creation_input")
            + Decimal(output_count) * _rate_value(rates, "output")
        )
    return CostComputation(
        cost_usd=cost_units / _PER_MILLION,
        cost_unknown=False,
        warning="; ".join(warnings) if warnings else None,
        rate_source=rate_source,
    )


def compute_cost_usd(
    family: str,
    input_tokens: int,
    output_tokens: int,
    table: CostTable | LoadedConfig,
    *,
    model: str | None = None,
    provider: str | None = None,
    cached_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    reasoning_output_tokens: int = 0,
) -> float:
    result = _compute_cost(
        family,
        input_tokens,
        output_tokens,
        table,
        model=model,
        provider=provider,
        cached_input_tokens=cached_input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
    )
    if result.cost_unknown or result.cost_usd is None:
        return 0.0
    return float(result.cost_usd)


# Keys in a cost-record ``extra`` payload that hold bulky raw CLI output
# (the full ``claude --output-format json`` / ``codex exec --json`` dump).
# They are dropped before the record is persisted to ``cost.jsonl``: the
# parsed usage counters live in dedicated CostRecord columns and the
# review/answer text is stored in its own review artifact, so the raw blob on
# disk is pure redundancy that bloats every report/qa/retro read of the file.
# The full blob stays on the in-memory ``AgentResult.extra`` for live
# diagnostics and parser fallback. (cost-tracking T2 [FUTURE] finding.)
RAW_OUTPUT_EXTRA_KEYS: tuple[str, ...] = ("raw_terminal_json",)


def sanitize_persisted_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of ``extra`` with bulky raw-output blobs stripped.

    Removes every key in :data:`RAW_OUTPUT_EXTRA_KEYS` from the top level and
    from the nested ``agent_result_extra`` mapping that the runner/qa/retro
    helpers attach. Every other key — parsed usage metadata, model routing,
    role, verdict, ... — is preserved. The input mapping is never mutated.
    """
    if not extra:
        return {}
    cleaned = {
        key: value
        for key, value in extra.items()
        if key not in RAW_OUTPUT_EXTRA_KEYS
    }
    nested = cleaned.get("agent_result_extra")
    if isinstance(nested, dict):
        cleaned["agent_result_extra"] = {
            key: value
            for key, value in nested.items()
            if key not in RAW_OUTPUT_EXTRA_KEYS
        }
    return cleaned


@dataclass
class CostRecord:
    ts: str
    iteration: str
    task: str
    step: str                # IMPL | FIX | REVIEW
    agent: str
    family: str
    model: str | None
    prompt_path: str | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cache_creation_input_tokens: int
    reasoning_output_tokens: int
    estimated: bool
    est_cost_usd: float | None
    duration_s: float
    exit_code: int
    partial: bool            # True when the invocation was killed by TIMEOUT
    cause: str | None = None  # for FIX: 'acceptance' | 'review' (C4)
    provider: str | None = None
    cost_unknown: bool = False
    cost_warning: str | None = None
    rate_source: str | None = None
    parser_status: str | None = None
    parser_warning: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CostLogger:
    """Append-only JSONL writer."""

    def __init__(
        self,
        path: Path,
        cost_table: CostTable | LoadedConfig,
        iteration: str,
    ) -> None:
        self.path = path
        self.cost_table = _resolve_cost_table(cost_table)
        self.iteration = iteration

    def record(
        self,
        *,
        task: str,
        step: str,
        agent: str,
        family: str,
        input_tokens: int,
        output_tokens: int,
        exact: bool,
        duration_s: float,
        exit_code: int,
        partial: bool = False,
        model: str | None = None,
        prompt_path: str | None = None,
        cause: str | None = None,
        provider: str | None = None,
        cached_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        reasoning_output_tokens: int = 0,
        parser_status: str | None = None,
        parser_warning: str | None = None,
        extra: dict | None = None,
    ) -> CostRecord:
        cost = _compute_cost(
            family,
            input_tokens,
            output_tokens,
            self.cost_table,
            model=model,
            provider=provider,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
        )
        serialized_cost: float | None
        if cost.cost_unknown or cost.cost_usd is None:
            serialized_cost = None
        else:
            serialized_cost = float(cost.cost_usd.quantize(_RECORD_PLACES))
        rec = CostRecord(
            ts=utcnow_iso(),
            iteration=self.iteration,
            task=task,
            step=step,
            agent=agent,
            family=family,
            model=model,
            prompt_path=prompt_path,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cached_input_tokens=_non_negative_int(cached_input_tokens),
            cache_creation_input_tokens=_non_negative_int(
                cache_creation_input_tokens
            ),
            reasoning_output_tokens=_non_negative_int(reasoning_output_tokens),
            estimated=not exact,
            est_cost_usd=serialized_cost,
            duration_s=round(duration_s, 3),
            exit_code=int(exit_code),
            partial=bool(partial),
            cause=cause,
            provider=provider or family,
            cost_unknown=cost.cost_unknown,
            cost_warning=cost.warning,
            rate_source=cost.rate_source,
            parser_status=parser_status,
            parser_warning=parser_warning,
            extra=sanitize_persisted_extra(extra),
        )
        self._append(rec)
        return rec

    def _append(self, rec: CostRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec), sort_keys=False) + "\n")


# ---------------------------------------------------------------------------
# Summarisation — used by the readiness report
# ---------------------------------------------------------------------------


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


@dataclass
class CostSummary:
    total_usd: float
    invocations: int
    by_task: dict[str, float]
    by_step: dict[str, float]
    by_agent: dict[str, float]
    estimated_count: int
    partial_count: int
    unknown_count: int = 0
    warning_count: int = 0
    by_provider_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_agent_detail: dict[str, dict[str, Any]] = field(default_factory=dict)


def _record_cost(r: dict[str, Any]) -> Decimal | None:
    if r.get("cost_unknown"):
        return None
    value = r.get("est_cost_usd")
    if value is None:
        value = r.get("estimated_cost_usd", 0.0)
    try:
        return Decimal(str(value or 0.0))
    except Exception:
        return Decimal("0")


def _round_summary(value: Decimal) -> float:
    return float(value.quantize(_SUMMARY_PLACES))


def _rollup_key(r: dict[str, Any]) -> str:
    provider = str(r.get("provider") or r.get("family") or "?")
    model = str(r.get("model") or "(family fallback)")
    return f"{provider}/{model}"


def _add_detail(
    target: dict[str, dict[str, Any]],
    key: str,
    r: dict[str, Any],
    cost: Decimal | None,
) -> None:
    detail = target.setdefault(
        key,
        {
            "_cost": Decimal("0"),
            "invocations": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "reasoning_output_tokens": 0,
            "unknown_count": 0,
        },
    )
    detail["invocations"] += 1
    detail["input_tokens"] += _non_negative_int(r.get("input_tokens"))
    detail["output_tokens"] += _non_negative_int(r.get("output_tokens"))
    detail["cached_input_tokens"] += _non_negative_int(
        r.get("cached_input_tokens")
    )
    detail["cache_creation_input_tokens"] += _non_negative_int(
        r.get("cache_creation_input_tokens")
    )
    detail["reasoning_output_tokens"] += _non_negative_int(
        r.get("reasoning_output_tokens")
    )
    if cost is None:
        detail["unknown_count"] += 1
    else:
        detail["_cost"] += cost


def _finalize_detail(
    target: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, detail in target.items():
        clean = dict(detail)
        cost = clean.pop("_cost")
        clean["cost_usd"] = _round_summary(cost)
        out[key] = clean
    return out


def summarize_costs(path: Path) -> CostSummary:
    records = load_records(path)
    by_task: dict[str, Decimal] = {}
    by_step: dict[str, Decimal] = {}
    by_agent: dict[str, Decimal] = {}
    by_provider_model: dict[str, dict[str, Any]] = {}
    by_agent_detail: dict[str, dict[str, Any]] = {}
    total = Decimal("0")
    estimated = 0
    partial = 0
    unknown = 0
    warnings = 0
    for r in records:
        cost = _record_cost(r)
        _add_detail(by_provider_model, _rollup_key(r), r, cost)
        _add_detail(by_agent_detail, str(r.get("agent", "?")), r, cost)
        if cost is None:
            unknown += 1
        else:
            total += cost
            by_task[r.get("task", "?")] = (
                by_task.get(r.get("task", "?"), Decimal("0")) + cost
            )
            by_step[r.get("step", "?")] = (
                by_step.get(r.get("step", "?"), Decimal("0")) + cost
            )
            by_agent[r.get("agent", "?")] = (
                by_agent.get(r.get("agent", "?"), Decimal("0")) + cost
            )
        if r.get("estimated"):
            estimated += 1
        if r.get("partial"):
            partial += 1
        if r.get("cost_warning") or r.get("parser_warning"):
            warnings += 1
    return CostSummary(
        total_usd=_round_summary(total),
        invocations=len(records),
        by_task={k: _round_summary(v) for k, v in by_task.items()},
        by_step={k: _round_summary(v) for k, v in by_step.items()},
        by_agent={k: _round_summary(v) for k, v in by_agent.items()},
        estimated_count=estimated,
        partial_count=partial,
        unknown_count=unknown,
        warning_count=warnings,
        by_provider_model=_finalize_detail(by_provider_model),
        by_agent_detail=_finalize_detail(by_agent_detail),
    )
