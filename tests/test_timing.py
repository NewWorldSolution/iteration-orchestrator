"""Tests for orch.timing — phase-timing harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orch import timing as t


def test_record_event_appends_to_jsonl(tmp_path: Path):
    """Each call appends one well-formed JSON line."""
    repo = tmp_path
    t.record_event(repo, "pre-i8-orch", "start", "T1 impl")
    t.record_event(repo, "pre-i8-orch", "end", "T1 impl")
    log = repo / "tools" / "logs" / "pre-i8-orch" / "timing.jsonl"
    assert log.exists()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    e0 = json.loads(lines[0])
    assert e0["kind"] == "start"
    assert e0["label"] == "T1 impl"
    assert e0["ts"].endswith("+00:00")


def test_timing_honors_configured_artifact_root(tmp_path: Path):
    """A non-default artifact_root_ref relocates the timing log (config-routed)."""
    repo = tmp_path
    t.record_event(repo, "iter", "start", "A", artifact_root_ref="custom/logroot")
    t.record_event(repo, "iter", "end", "A", artifact_root_ref="custom/logroot")
    relocated = repo / "custom" / "logroot" / "iter" / "timing.jsonl"
    assert relocated.exists()
    # The historical default root is untouched when a custom root is given.
    assert not (repo / "tools" / "logs" / "iter" / "timing.jsonl").exists()
    summary = t.summarize(repo, "iter", artifact_root_ref="custom/logroot")
    assert len(summary["spans"]) == 1


def test_summarize_pairs_start_end_by_label(tmp_path: Path):
    """First unmatched start pairs with the next end of the same label."""
    repo = tmp_path
    t.record_event(repo, "iter", "start", "A")
    t.record_event(repo, "iter", "start", "B")
    t.record_event(repo, "iter", "end", "A")
    t.record_event(repo, "iter", "end", "B")
    summary = t.summarize(repo, "iter")
    assert len(summary["spans"]) == 2
    labels = {span.label for span in summary["spans"]}
    assert labels == {"A", "B"}
    assert summary["unpaired"] == []


def test_summarize_flags_unpaired_events(tmp_path: Path):
    """Missing start/end events are surfaced in the unpaired list."""
    repo = tmp_path
    t.record_event(repo, "iter", "start", "lonely-start")
    t.record_event(repo, "iter", "end", "lonely-end")
    summary = t.summarize(repo, "iter")
    assert summary["spans"] == []
    reasons = {evt["reason"] for evt in summary["unpaired"]}
    assert reasons == {"start without end", "end without start"}


def test_render_report_no_events(tmp_path: Path):
    """Empty log returns a friendly message, not an error."""
    summary = t.summarize(tmp_path, "iter")
    out = t.render_report(summary)
    assert "No timing events" in out


def test_record_event_rejects_bad_input(tmp_path: Path):
    with pytest.raises(ValueError):
        t.record_event(tmp_path, "iter", "middle", "x")
    with pytest.raises(ValueError):
        t.record_event(tmp_path, "iter", "start", "")
    with pytest.raises(ValueError):
        t.record_event(tmp_path, "iter", "start", "   ")


def test_detect_evidence_prefers_timing_jsonl(tmp_path: Path):
    log_dir = tmp_path / "tools" / "logs" / "iter"
    log_dir.mkdir(parents=True)
    (log_dir / "timing.jsonl").write_text("{}\n", encoding="utf-8")
    (log_dir / "notes.md").write_text("manual fallback\n", encoding="utf-8")

    evidence = t.detect_evidence(log_dir)

    assert evidence.status == "timing_jsonl"
    assert evidence.has_timing_log
    assert not evidence.has_notes_fallback
    assert not evidence.is_missing


def test_detect_evidence_uses_notes_fallback(tmp_path: Path):
    log_dir = tmp_path / "tools" / "logs" / "iter"
    log_dir.mkdir(parents=True)
    (log_dir / "notes.md").write_text("manual fallback\n", encoding="utf-8")

    evidence = t.detect_evidence(log_dir)

    assert evidence.status == "notes_fallback"
    assert evidence.has_notes_fallback
    assert not evidence.has_timing_log
    assert not evidence.is_missing


def test_detect_evidence_reports_missing(tmp_path: Path):
    evidence = t.detect_evidence(tmp_path / "tools" / "logs" / "iter")

    assert evidence.status == "missing"
    assert evidence.is_missing
    assert not evidence.has_timing_log
    assert not evidence.has_notes_fallback


def test_cli_timing_start_end_report_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """End-to-end CLI smoke: start + end + report writes and reads timing.jsonl."""
    from orch.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text("# minimal\n")

    assert main(["timing", "--iter", "pre-i8-orch", "start", "T1 impl"]) == 0
    assert main(["timing", "--iter", "pre-i8-orch", "end", "T1 impl"]) == 0
    assert main(["timing", "--iter", "pre-i8-orch", "report"]) == 0

    out = capsys.readouterr().out
    assert "T1 impl" in out

    log = tmp_path / "tools" / "logs" / "pre-i8-orch" / "timing.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
