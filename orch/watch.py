"""Read-only event-tail helpers for ``orch watch``."""
from __future__ import annotations

import json
from pathlib import Path


def recent_events(log_dir: Path, *, limit: int = 20) -> list[dict]:
    path = log_dir / "run_state.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    events = data.get("events", [])
    if not isinstance(events, list):
        return []
    safe_limit = max(0, int(limit))
    return events[-safe_limit:] if safe_limit else []


def render_recent_events(log_dir: Path, *, limit: int = 20) -> str:
    events = recent_events(log_dir, limit=limit)
    if not events:
        return "(no events)\n"
    lines: list[str] = []
    for event in events:
        meta = event.get("meta") if isinstance(event.get("meta"), dict) else {}
        event_name = meta.get("event")
        parts = [
            str(event.get("ts", "-")),
            str(event.get("kind", "-")),
        ]
        if event.get("task"):
            parts.append(str(event["task"]))
        if event.get("step"):
            parts.append(str(event["step"]))
        if event_name:
            parts.append(str(event_name))
        if event.get("status"):
            parts.append(str(event["status"]))
        lines.append(" | ".join(parts))
    return "\n".join(lines) + "\n"
