"""Six Sigma improvement records for orchestrator retrospectives.

Records are append-only JSONL artifacts under an iteration log directory:
``tools/logs/<iteration>/improvements.jsonl``.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

IMPROVEMENTS_FILENAME = "improvements.jsonl"

REQUIRED_FIELDS = (
    "id",
    "source_iteration",
    "source_event",
    "title",
    "problem",
    "classification",
    "impact",
    "effort",
    "status",
    "control_mechanism",
)

CLASSIFICATIONS = frozenset({"defect", "waste", "risk", "speed", "quality"})
STATUSES = frozenset(
    {"proposed", "reviewing", "approved", "implemented", "rejected", "deferred"}
)
CONTROL_REQUIRED_STATUSES = frozenset({"approved", "implemented"})


class ImprovementValidationError(ValueError):
    """Raised when an improvement record fails deterministic validation."""


@dataclass(frozen=True)
class ImprovementRecord:
    id: str
    source_iteration: str
    source_event: str
    title: str
    problem: str
    classification: str
    impact: str
    effort: str
    status: str
    control_mechanism: str


def improvements_path(log_dir: Path) -> Path:
    return log_dir / IMPROVEMENTS_FILENAME


def ensure_improvements_artifact(log_dir: Path) -> Path:
    path = improvements_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def _require_string(
    raw: Mapping[str, Any], field: str, *, allow_empty: bool = False
) -> str:
    if field not in raw:
        raise ImprovementValidationError(f"missing required field: {field}")
    value = raw[field]
    if not isinstance(value, str):
        raise ImprovementValidationError(f"field {field} must be a string")
    value = value.strip()
    if not allow_empty and not value:
        raise ImprovementValidationError(f"field {field} must be non-empty")
    return value


def validate_record(raw: Mapping[str, Any]) -> ImprovementRecord:
    """Validate and normalize one improvement record.

    The control mechanism gate fails closed: approved and implemented records
    are invalid unless ``control_mechanism`` contains non-whitespace text.
    """
    if not isinstance(raw, Mapping):
        raise ImprovementValidationError("improvement record must be an object")

    values: dict[str, str] = {}
    for field in REQUIRED_FIELDS:
        values[field] = _require_string(
            raw,
            field,
            allow_empty=(field == "control_mechanism"),
        )

    classification = values["classification"]
    if classification not in CLASSIFICATIONS:
        allowed = ", ".join(sorted(CLASSIFICATIONS))
        raise ImprovementValidationError(
            f"invalid classification {classification!r}; allowed: {allowed}"
        )

    status = values["status"]
    if status not in STATUSES:
        allowed = ", ".join(sorted(STATUSES))
        raise ImprovementValidationError(
            f"invalid status {status!r}; allowed: {allowed}"
        )

    if (
        status in CONTROL_REQUIRED_STATUSES
        and not values["control_mechanism"]
    ):
        raise ImprovementValidationError(
            "control_mechanism is required for "
            f"{status!r} improvement records"
        )

    return ImprovementRecord(**values)


def record_to_dict(record: ImprovementRecord | Mapping[str, Any]) -> dict[str, str]:
    validated = (
        record if isinstance(record, ImprovementRecord)
        else validate_record(record)
    )
    data = asdict(validated)
    return {field: data[field] for field in REQUIRED_FIELDS}


def encode_record(record: ImprovementRecord | Mapping[str, Any]) -> str:
    return json.dumps(record_to_dict(record), sort_keys=False)


def append_record(
    log_dir: Path,
    record: ImprovementRecord | Mapping[str, Any],
) -> ImprovementRecord:
    validated = (
        record if isinstance(record, ImprovementRecord)
        else validate_record(record)
    )
    path = ensure_improvements_artifact(log_dir)
    with path.open("a", encoding="utf-8") as f:
        f.write(encode_record(validated) + "\n")
    return validated


def read_records(path: Path) -> list[ImprovementRecord]:
    if not path.exists():
        return []
    out: list[ImprovementRecord] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except JSONDecodeError as exc:
            raise ImprovementValidationError(
                f"{path}:{line_no}: invalid JSON: {exc.msg}"
            ) from exc
        try:
            out.append(validate_record(raw))
        except ImprovementValidationError as exc:
            raise ImprovementValidationError(f"{path}:{line_no}: {exc}") from exc
    return out


def validate_file(path: Path) -> list[ImprovementRecord]:
    return read_records(path)


def status_counts(records: list[ImprovementRecord]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(STATUSES)}
    for record in records:
        counts[record.status] += 1
    return {status: count for status, count in counts.items() if count}
