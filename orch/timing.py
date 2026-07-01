"""Manual phase-timing harness for iterations.

Writes append-only events to ``tools/logs/<iter>/timing.jsonl``. Each line is
one JSON object with keys: ``ts`` (ISO8601 UTC), ``kind`` ("start"|"end"),
``label`` (str).

Pairs are matched by label: the first unmatched start with a given label pairs
with the next end of the same label. Out-of-order events, missing ends, or
duplicate starts are reported by ``summarize()`` rather than failing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class TimingEvidence:
    """Evidence policy for an iteration log directory."""

    status: str
    timing_path: Path
    notes_path: Path

    @property
    def has_timing_log(self) -> bool:
        return self.status == "timing_jsonl"

    @property
    def has_notes_fallback(self) -> bool:
        return self.status == "notes_fallback"

    @property
    def is_missing(self) -> bool:
        return self.status == "missing"


def _timing_path(
    repo: Path, iteration: str, artifact_root_ref: str = "tools/logs"
) -> Path:
    # artifact_root_ref comes from the config resolver (OrchPaths) in
    # production; the "tools/logs" default preserves the historical layout for
    # direct callers and tests.
    return repo / artifact_root_ref / iteration / "timing.jsonl"


def detect_evidence(log_dir: Path) -> TimingEvidence:
    """Detect whether an iteration has measured or fallback timing evidence."""
    timing_path = log_dir / "timing.jsonl"
    notes_path = log_dir / "notes.md"
    if timing_path.exists():
        return TimingEvidence(
            status="timing_jsonl",
            timing_path=timing_path,
            notes_path=notes_path,
        )
    if notes_path.exists():
        return TimingEvidence(
            status="notes_fallback",
            timing_path=timing_path,
            notes_path=notes_path,
        )
    return TimingEvidence(
        status="missing",
        timing_path=timing_path,
        notes_path=notes_path,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fmt(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}:{secs:02d}"


def record_event(
    repo: Path,
    iteration: str,
    kind: str,
    label: str,
    artifact_root_ref: str = "tools/logs",
) -> dict:
    """Append a ``{ts, kind, label}`` event and return the written record."""
    if kind not in ("start", "end"):
        raise ValueError(f"kind must be 'start' or 'end', got {kind!r}")
    if not label or not label.strip():
        raise ValueError("label must be a non-empty string")
    path = _timing_path(repo, iteration, artifact_root_ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": _now_iso(), "kind": kind, "label": label.strip()}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def _read_events(
    repo: Path, iteration: str, artifact_root_ref: str = "tools/logs"
) -> list[dict]:
    path = _timing_path(repo, iteration, artifact_root_ref)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


@dataclass
class _Span:
    label: str
    start_ts: str
    end_ts: str
    duration_s: int


def summarize(
    repo: Path, iteration: str, artifact_root_ref: str = "tools/logs"
) -> dict:
    """Return ``{spans, unpaired, total_s}`` for an iteration's timing log."""
    events = _read_events(repo, iteration, artifact_root_ref)
    open_starts: dict[str, list[dict]] = {}
    spans: list[_Span] = []
    unpaired: list[dict] = []
    for evt in events:
        label = evt.get("label", "")
        kind = evt.get("kind")
        if kind == "start":
            open_starts.setdefault(label, []).append(evt)
            continue
        if kind != "end":
            unpaired.append({**evt, "reason": "unknown kind"})
            continue
        queue = open_starts.get(label) or []
        if not queue:
            unpaired.append({**evt, "reason": "end without start"})
            continue
        start_evt = queue.pop(0)
        start_dt = datetime.fromisoformat(start_evt["ts"])
        end_dt = datetime.fromisoformat(evt["ts"])
        duration_s = max(0, int((end_dt - start_dt).total_seconds()))
        spans.append(
            _Span(
                label=label,
                start_ts=start_evt["ts"],
                end_ts=evt["ts"],
                duration_s=duration_s,
            )
        )
    for queue in open_starts.values():
        for evt in queue:
            unpaired.append({**evt, "reason": "start without end"})
    return {
        "spans": spans,
        "unpaired": unpaired,
        "total_s": sum(span.duration_s for span in spans),
    }


def render_report(summary: dict) -> str:
    """Render a markdown timing table with optional unpaired-event notes."""
    spans = summary["spans"]
    unpaired = summary["unpaired"]
    if not spans and not unpaired:
        return "_No timing events recorded._\n"
    lines: list[str] = []
    if spans:
        lines.extend(
            [
                "| Label | Duration | Started | Ended |",
                "|---|---|---|---|",
            ]
        )
        for span in spans:
            lines.append(
                f"| {span.label} | {_fmt(span.duration_s)} | "
                f"{span.start_ts} | {span.end_ts} |"
            )
        lines.append("")
        lines.append(f"**Total:** {_fmt(summary['total_s'])}")
    else:
        lines.append("_No completed timing spans recorded._")
    if unpaired:
        lines.append("")
        lines.append("**Unpaired events** (review for missed start/end):")
        for evt in unpaired:
            lines.append(
                f"- `{evt.get('label', '')}` ({evt.get('reason', '?')}) "
                f"at {evt.get('ts', '')}"
            )
    return "\n".join(lines) + "\n"
