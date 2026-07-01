"""Tests for orch.tasks_schema."""
from __future__ import annotations

from pathlib import Path

import pytest

from orch.config import (
    TASK_SCHEMA_POLICY_FLOOR,
    effective_task_schema_policy,
    load_config,
    patterns as config_patterns,
    task_schema_policy,
)
from orch.tasks_schema import (
    EMDASH,
    TasksMdError,
    is_planning_artifact_path,
    parse_tasks_md as _parse_tasks_md,
    planning_path_refusal_reason,
)


VALID_MD = f"""\
# Demo Iteration
## Task Board

**Status:** WAITING
**Iteration branch:** `demo/iteration-1`
**Depends on:** none
**Blocks:** none

---

## Execution Plan
- approach: task_by_task
- qa: standard
- note: implementer and reviewer chosen at runtime

---

## Dependency Map

Some prose a human wrote here.

---

## Tasks

| ID    | Title        | Owner | Status  | Depends on | Branch          |
|-------|--------------|-------|---------|------------|-----------------|
| I1-T1 | Do the thing | TBD   | WAITING | {EMDASH}   | `demo/i1/t1`    |
| I1-T2 | Next thing   | TBD   | WAITING | I1-T1      | `demo/i1/t2`    |

---

## Task Details

### I1-T1 {EMDASH} Do the thing

**Allowed files:**
```
src/demo.py
tests/test_demo.py     <- new file
```

**Done when:** it works.

### I1-T2 {EMDASH} Next thing

**Allowed files:**
```
src/demo_next.py
```

**Done when:** it also works.
"""

VALID_DIFF_CAP_OVERRIDE = (
    "max_diff_insertions_hard=1800; approved_by=operator; "
    "evidence=iterations/demo-i1/reviews/review-t1.md"
)

UNIVERSAL_TASK_SCHEMA_FLOOR = {
    key: list(values)
    for key, values in TASK_SCHEMA_POLICY_FLOOR.items()
}

EXAMPLE_TASK_SCHEMA_POLICY = {
    **UNIVERSAL_TASK_SCHEMA_FLOOR,
    "planning_refusal_prefixes": [
        *UNIVERSAL_TASK_SCHEMA_FLOOR["planning_refusal_prefixes"],
        "app/",
        "db/",
        "tests/",
        "static/",
        "seed/",
        "migrations/",
        "migration/",
    ],
}


def _write(tmp_path: Path, content: str, name: str = "tasks.md") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _example_patterns() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return config_patterns(load_config(repo_root / "examples" / "financial-saas" / "project.yaml"))


def parse_tasks_md(
    path: Path,
    *,
    patterns=None,
    task_schema_policy=None,
):
    return _parse_tasks_md(
        path,
        patterns=_example_patterns() if patterns is None else patterns,
        task_schema_policy=task_schema_policy,
    )


# ---------- happy path -----------------------------------------------------


def test_valid_parse(tmp_path: Path):
    board = parse_tasks_md(_write(tmp_path, VALID_MD))
    assert board.title == "Demo Iteration"
    assert board.iteration_branch == "demo/iteration-1"
    assert board.execution_plan.approach == "task_by_task"
    assert board.execution_plan.qa == "standard"
    assert len(board.tasks) == 2
    t1, t2 = board.tasks
    assert t1.id == "I1-T1"
    assert t1.depends_on == []
    assert t1.allowed_files == ["src/demo.py", "tests/test_demo.py"]
    assert t1.diff_cap_override is None
    assert t2.depends_on == ["I1-T1"]
    assert t2.branch == "demo/i1/t2"
    assert board.allowed_file_union == [
        "src/demo.py",
        "tests/test_demo.py",
        "src/demo_next.py",
    ]
    assert board.diff_cap_override is None


def test_generic_default_accepts_neutral_task_ids(tmp_path: Path):
    md = (
        VALID_MD
        .replace("I1-T1", "TASK-1-1")
        .replace("I1-T2", "TASK-1-2")
    )

    board = _parse_tasks_md(_write(tmp_path, md))

    assert [task.id for task in board.tasks] == ["TASK-1-1", "TASK-1-2"]
    assert board.by_id("TASK-1-2").depends_on == ["TASK-1-1"]


def test_generic_default_rejects_example_task_ids(tmp_path: Path):
    with pytest.raises(TasksMdError) as exc:
        _parse_tasks_md(_write(tmp_path, VALID_MD))

    assert any(error.rule == "task_id" for error in exc.value.errors)


def test_ready_tasks(tmp_path: Path):
    board = parse_tasks_md(_write(tmp_path, VALID_MD))
    ready = board.ready_tasks()
    assert [t.id for t in ready] == ["I1-T1"]
    ready2 = board.ready_tasks({"I1-T1": "DONE", "I1-T2": "WAITING"})
    assert [t.id for t in ready2] == ["I1-T2"]


def test_tasks_schema_uses_configured_task_id_pattern(tmp_path: Path):
    md = (
        VALID_MD
        .replace("I1-T1", "TASK-1-1")
        .replace("I1-T2", "TASK-1-2")
    )

    board = parse_tasks_md(
        _write(tmp_path, md),
        patterns={
            "task_id": r"^TASK-(?P<iteration>\d+)-(?P<task>\d+)$",
            "task_detail_heading": (
                rf"^###\s+(?P<id>TASK-\d+-\d+)\s+{EMDASH}\s+"
                r"(?P<title>.+?)\s*$"
            ),
        },
    )

    assert [task.id for task in board.tasks] == ["TASK-1-1", "TASK-1-2"]
    assert board.by_id("TASK-1-2").depends_on == ["TASK-1-1"]
    ready = board.ready_tasks({"TASK-1-1": "DONE", "TASK-1-2": "WAITING"})
    assert [task.id for task in ready] == ["TASK-1-2"]


def test_configured_task_id_pattern_without_ordering_groups_fails_closed(
    tmp_path: Path,
):
    md = VALID_MD.replace("I1-T1", "TASK-alpha").replace("I1-T2", "TASK-beta")

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(
            _write(tmp_path, md),
            patterns={
                "task_id": r"^TASK-[a-z]+$",
                "task_detail_heading": (
                    rf"^###\s+(?P<id>TASK-[a-z]+)\s+{EMDASH}\s+"
                    r"(?P<title>.+?)\s*$"
                ),
            },
        )

    assert any(error.rule == "pattern_invalid" for error in exc.value.errors)


def test_configured_task_id_pattern_with_non_numeric_ordering_fails_closed(
    tmp_path: Path,
):
    md = VALID_MD.replace("I1-T1", "TASK-alpha").replace("I1-T2", "TASK-beta")

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(
            _write(tmp_path, md),
            patterns={
                "task_id": r"^TASK-([a-z]+)$",
                "task_detail_heading": (
                    rf"^###\s+(?P<id>TASK-[a-z]+)\s+{EMDASH}\s+"
                    r"(?P<title>.+?)\s*$"
                ),
            },
        )

    assert any(error.rule == "pattern_invalid" for error in exc.value.errors)


def test_iteration_diff_cap_override_parsed(tmp_path: Path):
    md = VALID_MD.replace(
        "**Blocks:** none\n",
        f"**Blocks:** none\n"
        f"**Diff cap override:** `{VALID_DIFF_CAP_OVERRIDE}`\n",
    )
    board = parse_tasks_md(_write(tmp_path, md))
    override = board.diff_cap_override
    assert override is not None
    assert override.max_diff_insertions_hard == 1800
    assert override.approved_by == "operator"
    assert override.evidence == "iterations/demo-i1/reviews/review-t1.md"
    assert override.scope == "iteration"


def test_task_diff_cap_override_parsed(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        f"**Diff cap override:** `{VALID_DIFF_CAP_OVERRIDE}`\n\n",
    )
    board = parse_tasks_md(_write(tmp_path, md))
    override = board.by_id("I1-T1").diff_cap_override
    assert override is not None
    assert override.max_diff_insertions_hard == 1800
    assert override.scope == "I1-T1"
    assert board.by_id("I1-T2").diff_cap_override is None


def test_task_model_routing_parsed(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        "**Model routing:** "
        "`model_tier=standard; reasoning_effort=low; "
        "risk_category=architecture_core_logic`\n\n",
    )
    board = parse_tasks_md(_write(tmp_path, md))
    routing = board.by_id("I1-T1").model_routing

    assert routing is not None
    assert routing.model_tier == "standard"
    assert routing.reasoning_effort == "low"
    assert routing.risk_category == "architecture_core_logic"
    assert board.by_id("I1-T2").model_routing is None


def test_task_kind_parsed(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        "**Task kind:** `characterization-test`\n\n",
    )
    board = parse_tasks_md(_write(tmp_path, md))

    assert board.by_id("I1-T1").task_kind == "characterization-test"
    assert board.by_id("I1-T2").task_kind is None


def test_parallel_safe_missing_defaults_false(tmp_path: Path):
    board = parse_tasks_md(_write(tmp_path, VALID_MD))
    safety = board.by_id("I1-T1").parallel_safe

    assert safety.value is False
    assert safety.reason == ""
    assert safety.conflicts == ()
    assert safety.requires_serial_after == ()


def test_parallel_safe_valid_marker_parsed(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        "**Parallel safe:** "
        "`yes; reason=disjoint files; conflicts=I1-T2, app/routes; "
        "requires_serial_after=none`\n\n",
    )

    board = parse_tasks_md(_write(tmp_path, md))
    safety = board.by_id("I1-T1").parallel_safe

    assert safety.value is True
    assert safety.reason == "disjoint files"
    assert safety.conflicts == ("I1-T2", "app/routes")
    assert safety.requires_serial_after == ()


def test_parallel_safe_malformed_marker_rejected(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        "**Parallel safe:** maybe; reason=bad marker; conflicts=none\n\n",
    )

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))

    assert any(e.rule == "parallel_safe_invalid" for e in exc.value.errors)


def test_parallel_safe_requires_serial_after_unknown_task_rejected(
    tmp_path: Path,
):
    md = VALID_MD.replace(
        f"### I1-T2 {EMDASH} Next thing\n\n",
        f"### I1-T2 {EMDASH} Next thing\n\n"
        "**Parallel safe:** "
        "yes; reason=after unrelated setup; conflicts=none; "
        "requires_serial_after=I1-T9\n\n",
    )

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))

    assert any(e.rule == "parallel_safe_invalid" for e in exc.value.errors)


def test_invalid_task_model_routing_fails(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        "**Model routing:** "
        "`model_tier=tiny; reasoning_effort=low; risk_category=unknown`\n\n",
    )

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))

    assert any(e.rule == "model_routing_invalid" for e in exc.value.errors)


def test_invalid_task_kind_fails(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        "**Task kind:** `../escape`\n\n",
    )

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))

    assert any(e.rule == "task_kind_invalid" for e in exc.value.errors)


def test_planning_artifact_path_helper_accepts_docs_and_iterations():
    assert is_planning_artifact_path("docs/post-phase-organization/plan.md")
    assert is_planning_artifact_path("iterations/demo-i1/prompt.md")


@pytest.mark.parametrize(
    "path",
    [
        "app/main.py",
        "db/schema.sql",
        "src/orchestrator/runner.py",
        "tests/test_demo.py",
        "deploy/render.yaml",
        ".github/workflows/ci.yml",
        "seed/demo.sql",
        "migrations/001.sql",
        "src/demo.py",
        "../docs/escape.md",
    ],
)
def test_planning_artifact_path_helper_refuses_runtime_and_undeclared_roots(
    path: str,
):
    assert planning_path_refusal_reason(path) is not None


def test_no_pack_policy_floor_fires_for_protected_prefixes(tmp_path: Path):
    md = VALID_MD.replace("src/demo.py", ".git/config")

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))

    floor_error = next(
        error
        for error in exc.value.errors
        if error.rule == "allowed_forbidden_prefix"
    )
    assert "prefix '.git/'" in floor_error.message
    assert "extend scope explicitly" in floor_error.message

    floor_policy = effective_task_schema_policy({})
    for path in (
        ".git/config",
        ".github/workflows/ci.yml",
        "deploy/render.yaml",
    ):
        assert (
            planning_path_refusal_reason(path, task_schema_policy=floor_policy)
            is not None
        )
    assert (
        planning_path_refusal_reason(
            "docs/plan.md",
            task_schema_policy=floor_policy,
        )
        is None
    )


def test_example_policy_preserves_legacy_planning_outcomes(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    policy = task_schema_policy(
        load_config(repo_root / "examples" / "financial-saas" / "project.yaml")
    )

    assert policy == EXAMPLE_TASK_SCHEMA_POLICY
    assert (
        planning_path_refusal_reason(
            "docs/post-phase-organization/plan.md",
            task_schema_policy=policy,
        )
        is None
    )
    assert (
        planning_path_refusal_reason(
            "iterations/demo-i1/prompt.md",
            task_schema_policy=policy,
        )
        is None
    )
    for path, prefix in (
        ("app/main.py", "app/"),
        ("db/schema.sql", "db/"),
        ("tests/test_demo.py", "tests/"),
        ("deploy/render.yaml", "deploy/"),
        (".github/workflows/ci.yml", ".github/"),
        ("static/app.js", "static/"),
        ("seed/demo.sql", "seed/"),
        ("migrations/001.sql", "migrations/"),
        ("migration/legacy.sql", "migration/"),
    ):
        assert planning_path_refusal_reason(
            path,
            task_schema_policy=policy,
        ) == f"path is under forbidden planning prefix '{prefix}'"

    board = parse_tasks_md(
        _write(tmp_path, VALID_MD.replace("src/demo.py", "app/main.py")),
        task_schema_policy=policy,
    )
    assert board.by_id("I1-T1").allowed_files[0] == "app/main.py"


def test_floor_tripwire_refuses_allowed_file_with_extend_scope_message(
    tmp_path: Path,
):
    policy = effective_task_schema_policy(
        {"forbidden_allowed_prefixes": ["custom-engine/"]}
    )
    md = VALID_MD.replace("src/demo.py", "custom-engine/config.py")

    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md), task_schema_policy=policy)

    assert any(
        error.rule == "allowed_forbidden_prefix"
        and "prefix 'custom-engine/'" in error.message
        and "extend scope explicitly" in error.message
        for error in exc.value.errors
    )


def test_project_policy_extends_planning_without_narrowing_floor():
    policy = effective_task_schema_policy(
        {
            "planning_allowed_prefixes": ["plans/"],
            "planning_refusal_prefixes": ["src/"],
        }
    )

    assert (
        planning_path_refusal_reason(
            "docs/plan.md",
            task_schema_policy=policy,
        )
        is None
    )
    assert (
        planning_path_refusal_reason(
            "plans/new-plan.md",
            task_schema_policy=policy,
        )
        is None
    )
    assert planning_path_refusal_reason(
        "src/demo.py",
        task_schema_policy=policy,
    ) == "path is under forbidden planning prefix 'src/'"
    assert planning_path_refusal_reason(
        "lib/orchestrator/runner.py",
        task_schema_policy=policy,
    ) == "path must be under one of: docs/, iterations/, plans/"


@pytest.mark.parametrize(
    "raw",
    [
        "max_diff_insertions_hard=1800; evidence=note: approved in review",
        "max_diff_insertions_hard=1800; approved_by=operator",
    ],
)
def test_diff_cap_override_requires_approval_and_evidence(
    tmp_path: Path, raw: str
):
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        f"**Diff cap override:** `{raw}`\n\n",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(
        e.rule == "diff_cap_override_missing_field"
        for e in exc.value.errors
    )


@pytest.mark.parametrize("cap", ["abc", "0", "-1"])
def test_diff_cap_override_rejects_invalid_numeric_values(
    tmp_path: Path, cap: str
):
    raw = (
        f"max_diff_insertions_hard={cap}; approved_by=operator; "
        "evidence=note: approved in review"
    )
    md = VALID_MD.replace(
        f"### I1-T1 {EMDASH} Do the thing\n\n",
        f"### I1-T1 {EMDASH} Do the thing\n\n"
        f"**Diff cap override:** `{raw}`\n\n",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(
        e.rule == "diff_cap_override_invalid_numeric"
        for e in exc.value.errors
    )


# ---------- structural errors ---------------------------------------------


def test_file_missing(tmp_path: Path):
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(tmp_path / "nope.md")
    assert any(e.rule == "file_missing" for e in exc.value.errors)


def test_missing_h1(tmp_path: Path):
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, "Just text, no H1.\n"))
    assert any(e.rule == "h1_missing" for e in exc.value.errors)


def test_missing_exec_plan(tmp_path: Path):
    md = VALID_MD.replace("## Execution Plan", "## Something Else")
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    rules = {e.rule for e in exc.value.errors}
    assert "exec_plan_missing" in rules


def test_exec_plan_missing_field(tmp_path: Path):
    md = VALID_MD.replace("- qa: standard\n", "")
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "exec_plan_missing_field" for e in exc.value.errors)


def test_tasks_columns_wrong(tmp_path: Path):
    md = VALID_MD.replace(
        "| ID    | Title        | Owner | Status  | Depends on | Branch          |",
        "| ID    | Title        | Status  | Depends on | Branch          |",
    )
    # Also strip a cell from the separator + rows so the table is self-consistent
    md = md.replace(
        "|-------|--------------|-------|---------|------------|-----------------|",
        "|-------|--------------|---------|------------|-----------------|",
    )
    md = md.replace(
        "| I1-T1 | Do the thing | TBD   | WAITING |",
        "| I1-T1 | Do the thing | WAITING |",
    )
    md = md.replace(
        "| I1-T2 | Next thing   | TBD   | WAITING |",
        "| I1-T2 | Next thing   | WAITING |",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "tasks_columns" for e in exc.value.errors)


def test_bad_task_id(tmp_path: Path):
    md = VALID_MD.replace("I1-T1", "X1-T1")
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "task_id" for e in exc.value.errors)


def test_bad_status(tmp_path: Path):
    # Replace the task-row status, not the header Status.
    md = VALID_MD.replace(
        "| I1-T1 | Do the thing | TBD   | WAITING |",
        "| I1-T1 | Do the thing | TBD   | RUNNING |",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "task_status" for e in exc.value.errors)


def test_unknown_dependency(tmp_path: Path):
    md = VALID_MD.replace("I1-T1      |", "I9-T9      |", 1)
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "depends_unknown" for e in exc.value.errors)


def test_cycle_detected(tmp_path: Path):
    md = VALID_MD.replace(
        f"| I1-T1 | Do the thing | TBD   | WAITING | {EMDASH}   |",
        "| I1-T1 | Do the thing | TBD   | WAITING | I1-T2      |",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "dag_cycle" for e in exc.value.errors)


def test_allowed_files_glob_rejected(tmp_path: Path):
    md = VALID_MD.replace("src/demo.py", "src/*.py")
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "allowed_glob" for e in exc.value.errors)


def test_allowed_files_absolute_rejected(tmp_path: Path):
    md = VALID_MD.replace("src/demo.py", "/etc/passwd")
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    rules = {e.rule for e in exc.value.errors}
    assert "allowed_absolute" in rules


def test_allowed_files_forbidden_prefix(tmp_path: Path):
    md = VALID_MD.replace("src/demo.py", ".git/config")
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "allowed_forbidden_prefix" for e in exc.value.errors)


def test_allowed_files_tasks_md_rejected(tmp_path: Path):
    md = VALID_MD.replace(
        "src/demo.py",
        "iterations/phase-demo/demo-i1/tasks.md",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "allowed_tasks_md" for e in exc.value.errors)


def test_detail_section_missing(tmp_path: Path):
    md = VALID_MD.replace(
        f"### I1-T2 {EMDASH} Next thing\n\n"
        "**Allowed files:**\n```\nsrc/demo_next.py\n```\n\n"
        "**Done when:** it also works.\n",
        "",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "detail_missing" for e in exc.value.errors)


def test_duplicate_task_id(tmp_path: Path):
    md = VALID_MD.replace(
        "| I1-T2 | Next thing   | TBD   | WAITING | I1-T1      | `demo/i1/t2`    |",
        "| I1-T1 | Duplicate    | TBD   | WAITING | -          | `demo/i1/t1b`   |",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "task_id_dup" for e in exc.value.errors)


def test_errors_collected_with_line_numbers(tmp_path: Path):
    # Break two things: bad task-row status and unknown dep.
    md = VALID_MD.replace(
        "| I1-T1 | Do the thing | TBD   | WAITING |",
        "| I1-T1 | Do the thing | TBD   | RUNNING |",
    )
    md = md.replace("I1-T1      |", "I9-T9      |", 1)
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    rules = {e.rule for e in exc.value.errors}
    assert {"task_status", "depends_unknown"} <= rules
    # line numbers attached
    for e in exc.value.errors:
        assert e.line > 0


def test_parse_error_auto_fixable_populated(tmp_path: Path):
    """auto_fixable flag must be populated (unused in v1)."""
    # Wrong header field order triggers a flagged auto_fixable error
    md = VALID_MD.replace(
        "**Status:** WAITING\n**Iteration branch:** `demo/iteration-1`\n"
        "**Depends on:** none\n**Blocks:** none",
        "**Iteration branch:** `demo/iteration-1`\n**Status:** WAITING\n"
        "**Depends on:** none\n**Blocks:** none",
    )
    with pytest.raises(TasksMdError) as exc:
        parse_tasks_md(_write(tmp_path, md))
    assert any(e.rule == "kv_order" and e.auto_fixable for e in exc.value.errors)


def test_task_without_model_routing_still_parses_with_none(tmp_path: Path):
    board = parse_tasks_md(_write(tmp_path, VALID_MD))

    assert board.by_id("I1-T1").model_routing is None
    assert board.by_id("I1-T2").model_routing is None
