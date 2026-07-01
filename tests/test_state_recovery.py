"""State-file crash recovery characterization tests."""
from __future__ import annotations

import json
from pathlib import Path

from orch.state import STATUS_DONE, STATUS_IN_PROGRESS, StateStore


def _store(tmp_path: Path) -> StateStore:
    return StateStore(
        log_dir=tmp_path,
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
    )


def test_load_truncated_run_state_rebuilds_or_fails_cleanly(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_iteration_started()
    store.task_transition("I4-T1", STATUS_IN_PROGRESS)
    store.task_transition("I4-T1", STATUS_DONE)

    raw = store.path.read_bytes()
    tear_at = raw.index(b'"task_transition"') + len(b'"task_')
    store.path.write_bytes(raw[:tear_at])

    recovered = _store(tmp_path)
    try:
        recovered.load()
    except ValueError as exc:
        msg = str(exc).lower()
        assert "run_state.json" in msg
        assert "corrupt" in msg or "truncated" in msg
        assert recovered.events == []
    else:
        assert recovered.tasks["I4-T1"].status == STATUS_DONE
        assert recovered.events


def test_load_detects_or_handles_orphaned_run_state_tmp(tmp_path: Path):
    store = _store(tmp_path)
    store.mark_iteration_started()
    payload = json.loads(store.path.read_text())
    payload["events"].append(
        {
            "ts": "2026-06-17T00:00:00Z",
            "kind": "task_transition",
            "task": "I4-T1",
            "step": None,
            "status": None,
            "meta": {"status": STATUS_DONE},
        }
    )
    orphan = tmp_path / "run_state.orphan.tmp"
    orphan.write_text(json.dumps(payload), encoding="utf-8")

    recovered = _store(tmp_path)
    try:
        recovered.load()
    except ValueError as exc:
        msg = str(exc).lower()
        assert "run_state" in msg and ".tmp" in msg
    else:
        assert not list(tmp_path.glob("run_state.*.tmp"))
