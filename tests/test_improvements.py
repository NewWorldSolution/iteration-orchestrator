"""Tests for Six Sigma improvement records."""
from __future__ import annotations

import pytest

from orch.improvements import (
    ImprovementValidationError,
    append_record,
    encode_record,
    improvements_path,
    read_records,
    record_to_dict,
    validate_record,
)


def _record(**overrides):
    data = {
        "id": "imp-001",
        "source_iteration": "demo-i1",
        "source_event": "retro.completed",
        "title": "Reduce review churn",
        "problem": "Repeated review rounds found the same missing invariant.",
        "classification": "quality",
        "impact": "medium",
        "effort": "small",
        "status": "proposed",
        "control_mechanism": "",
    }
    data.update(overrides)
    return data


def test_valid_proposed_improvement_with_empty_control_passes():
    record = validate_record(_record())

    assert record.status == "proposed"
    assert record.control_mechanism == ""


def test_approved_improvement_without_control_fails():
    with pytest.raises(ImprovementValidationError, match="control_mechanism"):
        validate_record(_record(status="approved", control_mechanism=""))


def test_implemented_improvement_without_control_fails():
    with pytest.raises(ImprovementValidationError, match="control_mechanism"):
        validate_record(_record(status="implemented", control_mechanism=" "))


def test_approved_improvement_with_control_passes():
    record = validate_record(
        _record(
            status="approved",
            control_mechanism="golden event-log test",
        )
    )

    assert record.status == "approved"
    assert record.control_mechanism == "golden event-log test"


def test_invalid_classification_fails():
    with pytest.raises(ImprovementValidationError, match="invalid classification"):
        validate_record(_record(classification="example"))


def test_invalid_status_fails():
    with pytest.raises(ImprovementValidationError, match="invalid status"):
        validate_record(_record(status="ready"))


def test_jsonl_append_read_round_trip_is_deterministic(tmp_path):
    first = append_record(tmp_path, _record())
    second = append_record(
        tmp_path,
        _record(
            id="imp-002",
            status="approved",
            control_mechanism="fixture repo validation gate",
        ),
    )

    path = improvements_path(tmp_path)
    assert path.read_text(encoding="utf-8") == (
        encode_record(first) + "\n" + encode_record(second) + "\n"
    )

    records = read_records(path)
    assert [record_to_dict(record) for record in records] == [
        record_to_dict(first),
        record_to_dict(second),
    ]
