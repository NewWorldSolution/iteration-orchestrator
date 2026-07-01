"""Tests for deterministic Prompt Factory draft validation."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from orch.cli import main as cli_main
from orch.config import load_config, patterns as config_patterns
from orch.prompt_factory import (
    PromptFactoryApprovalError,
    PromptFactoryMaterializationError,
    PromptFactoryValidationError,
    check_prompt_factory_approval,
    build_review_prompts,
    decide_review_gate,
    load_operator_approval_json,
    materialize_prompt_factory_draft,
    parse_review_verdict,
    render_tasks_md,
    validate_draft,
    write_review_package,
)
from orch.tasks_schema import (
    ParseError,
    TasksMdError,
    parse_tasks_md as _parse_tasks_md,
)


PROJECT_PACK_YAML = """\
project:
  name: prompt-factory-test
  main_branch: main
  phase_branch_pattern: "phase-{phase}"
  iteration_branch_pattern: "{phase}/iteration-{n}"
  task_branch_pattern: "{phase}/i{n}/t{k}-{slug}"
stack:
  test: "echo test"
  lint: "echo lint"
risk:
  high_risk_globs: ["**/schema.sql"]
  sensitive_files: [".env"]
  forbidden_patterns: ["except: pass"]
agents:
  claude:
    type: claude
    cmd: "claude -p"
    family: anthropic
  codex:
    type: codex
    cmd: "codex"
    family: openai
costs:
  anthropic: {input: 3.0, output: 15.0}
  openai: {input: 2.5, output: 10.0}
"""


def _valid_draft() -> dict:
    return {
        "phase_or_iteration_id": "demo-i1",
        "iteration_branch": "demo/iteration-1",
        "final_pr": "TBD",
        "depends_on": "none",
        "blocks": "none",
        "execution_plan": {
            "approach": "task_by_task",
            "qa": "standard",
            "note": "operator chooses implementer and reviewer",
        },
        "tasks": [
            {
                "id": "TASK-1-1",
                "title": "Build first slice",
                "dependencies": [],
                "allowed_files": ["src/a.py", "tests/test_a.py"],
                "test_command": "pytest tests/test_a.py -q",
                "prompt_summary": "Implement the first deterministic slice.",
                "review_summary": "Check scope and parser compatibility.",
                "risk_category": "schema_data_structure",
                "model_tier": "strong",
                "reasoning_effort": "high",
            },
            {
                "id": "TASK-1-2",
                "title": "Build second slice",
                "dependencies": ["TASK-1-1"],
                "allowed_files": ["src/b.py"],
                "test_note": "Covered by the same parser smoke test.",
                "prompt_path_placeholder": "prompts/t2-second.md",
                "review_path_placeholder": "reviews/review-t2.md",
                "risk_category": "unknown",
            },
        ],
    }


def _assert_invalid(draft: dict, expected: str) -> None:
    with pytest.raises(PromptFactoryValidationError) as exc:
        validate_draft(draft)

    assert expected in str(exc.value)


def _generic_patterns() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    return config_patterns(
        load_config(repo_root / "examples" / "minimal" / "project.yaml")
    )


def parse_tasks_md(path: Path):
    return _parse_tasks_md(path, patterns=_generic_patterns())


def _write_project_pack(root: Path, extra: str = "") -> None:
    (root / ".orch").mkdir(exist_ok=True)
    project_yaml = root / ".orch" / "project.yaml"
    project_yaml.write_text(PROJECT_PACK_YAML + extra, encoding="utf-8")


def _pass_review_gate(draft_id: str = "demo-i1") -> dict:
    return {
        "draft_id": draft_id,
        "status": "PASS",
        "required_roles": ["prompt_expert", "technical_reviewer"],
        "roles": {
            "prompt_expert": {
                "state": "PRESENT",
                "verdict": "PASS",
                "error": None,
            },
            "technical_reviewer": {
                "state": "PRESENT",
                "verdict": "PASS",
                "error": None,
            },
        },
    }


def _approval(
    *,
    draft_id: str = "demo-i1",
    decision: str = "approved",
    review_gate_status: str = "PASS",
    approved_by: str = "operator",
    approved_at: str = "2026-05-25T12:00:00Z",
) -> dict:
    return {
        "draft_id": draft_id,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "decision": decision,
        "review_gate_status": review_gate_status,
        "notes": "approved after deterministic review gate",
    }


def _materialize(
    tmp_path: Path,
    *,
    draft: dict | None = None,
    gate: dict | None = None,
    approval: dict | None = None,
    target: Path | None = None,
    dry_run: bool = True,
    force: bool = False,
    project_extra: str = "",
):
    _write_project_pack(tmp_path, project_extra)
    return materialize_prompt_factory_draft(
        draft or _valid_draft(),
        gate or _pass_review_gate(),
        approval or _approval(),
        repo_root=tmp_path,
        target=target or Path("iterations/demo/demo-i1"),
        dry_run=dry_run,
        force=force,
    )


def test_valid_draft_passes_validation():
    draft = validate_draft(_valid_draft())

    assert draft.phase_or_iteration_id == "demo-i1"
    assert draft.tasks[0].id == "TASK-1-1"
    assert draft.tasks[1].dependencies == ("TASK-1-1",)


def test_valid_draft_renders_tasks_md_preview_string():
    rendered = render_tasks_md(_valid_draft())

    assert "# Prompt Factory Draft demo-i1" in rendered
    assert "**Final PR:** TBD" in rendered
    assert "| TASK-1-1 | Build first slice | TBD | WAITING" in rendered
    assert (
        "**Model routing:** "
        "`model_tier=strong; reasoning_effort=high; "
        "risk_category=schema_data_structure`"
    ) in rendered
    assert "prompts/t2-second.md" in rendered


def test_rendered_tasks_md_preview_passes_existing_tasks_schema(tmp_path: Path):
    rendered = render_tasks_md(_valid_draft())
    path = tmp_path / "tasks.md"
    path.write_text(rendered, encoding="utf-8")

    board = parse_tasks_md(path)

    assert board.iteration_branch == "demo/iteration-1"
    assert [task.id for task in board.tasks] == ["TASK-1-1", "TASK-1-2"]
    assert board.by_id("TASK-1-1").model_routing is not None
    assert board.by_id("TASK-1-1").model_routing.risk_category == (
        "schema_data_structure"
    )


def test_prompt_factory_round_trips_parallel_safe(tmp_path: Path):
    draft = _valid_draft()
    draft["tasks"][0]["parallel_safe"] = {
        "value": True,
        "reason": "disjoint files and no shared state",
        "conflicts": ["TASK-1-2", "app/routes"],
        "requires_serial_after": [],
    }

    validated = validate_draft(draft)
    rendered = render_tasks_md(validated)

    assert (
        "**Parallel safe:** yes; reason=disjoint files and no shared state; "
        "conflicts=TASK-1-2, app/routes; requires_serial_after=none"
    ) in rendered

    path = tmp_path / "tasks.md"
    path.write_text(rendered, encoding="utf-8")
    board = parse_tasks_md(path)
    safety = board.by_id("TASK-1-1").parallel_safe

    assert safety.value is True
    assert safety.reason == "disjoint files and no shared state"
    assert safety.conflicts == ("TASK-1-2", "app/routes")
    assert safety.requires_serial_after == ()


def test_missing_top_level_field_fails_with_clear_error():
    draft = _valid_draft()
    del draft["iteration_branch"]

    _assert_invalid(draft, "missing required top-level field: iteration_branch")


def test_missing_task_field_fails_with_clear_error():
    draft = _valid_draft()
    del draft["tasks"][0]["title"]

    _assert_invalid(draft, "tasks[0]: missing required task field: title")


def test_duplicate_task_id_fails():
    draft = _valid_draft()
    draft["tasks"][1]["id"] = "TASK-1-1"

    _assert_invalid(draft, "tasks[1].id duplicate task id 'TASK-1-1'")


def test_unknown_dependency_fails():
    draft = _valid_draft()
    draft["tasks"][1]["dependencies"] = ["TASK-1-9"]

    _assert_invalid(
        draft,
        "task TASK-1-2 dependency 'TASK-1-9' references unknown task",
    )


def test_invalid_risk_category_fails():
    draft = _valid_draft()
    draft["tasks"][0]["risk_category"] = "domain_specific"

    _assert_invalid(draft, "tasks[0].risk_category: invalid risk_category")


def test_invalid_model_tier_fails():
    draft = _valid_draft()
    draft["tasks"][0]["model_tier"] = "tiny"

    _assert_invalid(draft, "tasks[0].model_tier: invalid model_tier")


def test_invalid_reasoning_effort_fails():
    draft = _valid_draft()
    draft["tasks"][0]["reasoning_effort"] = "extreme"

    _assert_invalid(
        draft,
        "tasks[0].reasoning_effort: invalid reasoning_effort",
    )


def test_empty_allowed_files_fails():
    draft = _valid_draft()
    draft["tasks"][0]["allowed_files"] = []

    _assert_invalid(draft, "tasks[0].allowed_files must be non-empty")


def test_allowed_file_under_universal_forbidden_prefix_fails():
    draft = _valid_draft()
    draft["tasks"][0]["allowed_files"] = [".git/config"]

    _assert_invalid(
        draft,
        "forbidden prefix '.git/'",
    )


def test_invalid_task_id_shape_fails():
    draft = _valid_draft()
    draft["tasks"][0]["id"] = "T1"

    _assert_invalid(
        draft,
        "tasks[0].id 'T1' must match configured task_id pattern",
    )


def test_generic_no_pack_rejects_example_task_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    draft = _valid_draft()
    draft["tasks"][0]["id"] = "I1-T1"
    draft["tasks"][1]["id"] = "I1-T2"
    draft["tasks"][1]["dependencies"] = ["I1-T1"]
    monkeypatch.chdir(tmp_path)

    with pytest.raises(PromptFactoryValidationError) as exc:
        validate_draft(draft)

    message = str(exc.value)
    assert "configured task_id pattern '^TASK-(\\\\d+)-(\\\\d+)$'" in message
    assert "I<N>-T<K>" not in message


def test_draft_rejected_when_rendered_tasks_md_fails_tasks_schema():
    draft = deepcopy(_valid_draft())
    draft["execution_plan"]["note"] = "first line\nsecond line"

    _assert_invalid(
        draft,
        "rendered tasks.md failed tasks_schema validation",
    )


def test_validator_does_not_write_to_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _write_project_pack(tmp_path)
    monkeypatch.chdir(tmp_path)

    validate_draft(_valid_draft())

    assert not (tmp_path / "iterations").exists()


def test_valid_draft_can_produce_both_review_prompts():
    prompts = build_review_prompts(_valid_draft())

    assert set(prompts) == {"prompt_expert", "technical_reviewer"}
    assert "# Prompt Factory Review - Prompt Expert" in prompts["prompt_expert"]
    assert (
        "# Prompt Factory Review - Technical Reviewer"
        in prompts["technical_reviewer"]
    )


def test_review_prompts_include_preview_and_validation_context():
    prompts = build_review_prompts(_valid_draft())

    for text in prompts.values():
        assert "## Rendered tasks.md Preview" in text
        assert "# Prompt Factory Draft demo-i1" in text
        assert "PASS: prompt_factory.validate_draft accepted the draft" in text
        assert "tasks_schema.py" in text


def test_review_prompts_include_task_scope_dependencies_and_routing():
    prompts = build_review_prompts(_valid_draft())
    combined = prompts["prompt_expert"] + prompts["technical_reviewer"]

    assert "allowed_files: src/a.py, tests/test_a.py" in combined
    assert "dependencies: TASK-1-1" in combined
    assert "risk_category: schema_data_structure" in combined
    assert "resolved_routing: model_tier=max, reasoning_effort=high" in combined
    assert "dual_model_required=false" in combined


def test_prompt_expert_prompt_includes_prompt_quality_criteria():
    text = build_review_prompts(_valid_draft())["prompt_expert"]

    assert "Task slicing is small, ordered, and reviewable" in text
    assert "Implementation prompts are clear" in text
    assert "Review criteria are clear" in text
    assert "Acceptance tests are not missing" in text
    assert "Hidden assumptions are surfaced" in text
    assert "Scope is not overly broad" in text
    assert "Operator-decision leakage" in text


def test_technical_reviewer_prompt_includes_file_risk_test_schema_criteria():
    text = build_review_prompts(_valid_draft())["technical_reviewer"]

    assert "Allowed-files entries match" in text
    assert "Task dependencies are correct" in text
    assert "Risk categories match" in text
    assert "Model-routing metadata is correct" in text
    assert "Test commands or test notes are adequate" in text
    assert "compatible with tasks_schema.py" in text
    assert (
        "Project-specific invariants stay in project config or templates, "
        "not in reusable orchestrator core" in text
    )
    assert "Package/tooling paths are present only when the draft declares" in text


@pytest.mark.parametrize("verdict", ["PASS", "CHANGES_REQUIRED", "BLOCKED"])
def test_verdict_parser_accepts_supported_verdicts(verdict: str):
    assert parse_review_verdict(f"Verdict: {verdict}\n") == verdict


def test_malformed_verdict_is_rejected():
    with pytest.raises(ValueError, match="exactly one verdict line"):
        parse_review_verdict("Verdict: MAYBE\n")


def test_missing_review_produces_incomplete_gate_status():
    decision = decide_review_gate({"prompt_expert": "Verdict: PASS\n"})

    assert decision.status == "INCOMPLETE"
    assert [role.state for role in decision.roles] == ["PRESENT", "MISSING"]


def test_both_pass_produces_pass_gate_status():
    decision = decide_review_gate(
        {
            "prompt_expert": "Verdict: PASS\n",
            "technical_reviewer": "Verdict: PASS\n",
        }
    )

    assert decision.status == "PASS"


def test_one_changes_required_produces_changes_required_gate_status():
    decision = decide_review_gate(
        {
            "prompt_expert": "Verdict: PASS\n",
            "technical_reviewer": "Verdict: CHANGES_REQUIRED\n",
        }
    )

    assert decision.status == "CHANGES_REQUIRED"


def test_one_blocked_produces_blocked_gate_status():
    decision = decide_review_gate(
        {
            "prompt_expert": "Verdict: BLOCKED\n",
            "technical_reviewer": "Verdict: PASS\n",
        }
    )

    assert decision.status == "BLOCKED"


def test_malformed_review_produces_malformed_gate_status():
    decision = decide_review_gate(
        {
            "prompt_expert": "Verdict: PASS\n",
            "technical_reviewer": "No verdict here.\n",
        }
    )

    assert decision.status == "MALFORMED"
    assert decision.roles[1].state == "MALFORMED"


def test_review_package_does_not_write_to_iterations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _write_project_pack(tmp_path)
    monkeypatch.chdir(tmp_path)

    package = write_review_package(
        _valid_draft(),
        log_root=tmp_path / "tools" / "logs",
    )

    assert package.artifact_dir == (
        tmp_path / "tools" / "logs" / "prompt-factory" / "demo-i1"
    )
    assert package.decision.status == "INCOMPLETE"
    assert (
        package.artifact_dir / "prompt_expert_review_prompt.md"
    ).exists()
    assert (
        package.artifact_dir / "technical_reviewer_review_prompt.md"
    ).exists()
    assert (package.artifact_dir / "review_gate.json").exists()
    assert not (tmp_path / "iterations").exists()


def test_approved_artifact_with_pass_review_gate_passes_approval_validation():
    approval = check_prompt_factory_approval(
        _valid_draft(),
        _pass_review_gate(),
        _approval(),
    )

    assert approval.decision == "approved"
    assert approval.review_gate_status == "PASS"


def test_missing_approval_artifact_fails(tmp_path: Path):
    with pytest.raises(PromptFactoryApprovalError, match="file not found"):
        load_operator_approval_json(tmp_path / "missing-approval.json")


@pytest.mark.parametrize("decision", ["rejected", "deferred"])
def test_non_approved_decision_fails_materialization(
    tmp_path: Path,
    decision: str,
):
    with pytest.raises(PromptFactoryApprovalError, match="decision must be"):
        _materialize(tmp_path, approval=_approval(decision=decision))


def test_non_pass_review_gate_status_fails_materialization(tmp_path: Path):
    gate = _pass_review_gate()
    gate["status"] = "CHANGES_REQUIRED"

    with pytest.raises(
        PromptFactoryMaterializationError,
        match="review gate status must be PASS",
    ):
        _materialize(tmp_path, gate=gate)


def test_non_pass_approval_review_gate_status_fails_materialization(
    tmp_path: Path,
):
    with pytest.raises(
        PromptFactoryApprovalError,
        match="review_gate_status must be PASS",
    ):
        _materialize(
            tmp_path,
            approval=_approval(review_gate_status="INCOMPLETE"),
        )


def test_draft_id_mismatch_fails(tmp_path: Path):
    with pytest.raises(PromptFactoryApprovalError, match="draft_id mismatch"):
        _materialize(tmp_path, approval=_approval(draft_id="other-draft"))


def test_missing_approved_by_fails(tmp_path: Path):
    approval = _approval()
    del approval["approved_by"]

    with pytest.raises(PromptFactoryApprovalError, match="approved_by"):
        _materialize(tmp_path, approval=approval)


def test_missing_approved_at_fails(tmp_path: Path):
    approval = _approval()
    del approval["approved_at"]

    with pytest.raises(PromptFactoryApprovalError, match="approved_at"):
        _materialize(tmp_path, approval=approval)


def test_malformed_approval_artifact_fails(tmp_path: Path):
    path = tmp_path / "approval.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(PromptFactoryApprovalError, match="malformed"):
        load_operator_approval_json(path)


def test_materialization_re_runs_draft_validation(tmp_path: Path):
    draft = _valid_draft()
    del draft["tasks"][0]["title"]

    with pytest.raises(PromptFactoryValidationError, match="missing required"):
        _materialize(tmp_path, draft=draft)


def test_materialization_re_runs_tasks_schema_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import orch.prompt_factory as prompt_factory

    original_parse = prompt_factory.parse_tasks_md
    calls = {"count": 0}

    def parse_then_fail(path: Path, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise TasksMdError(
                [
                    ParseError(
                        line=1,
                        column=0,
                        rule="forced_materialization_check",
                        message="forced materialization parse failure",
                    )
                ],
                path,
            )
        return original_parse(path, **kwargs)

    monkeypatch.setattr(prompt_factory, "parse_tasks_md", parse_then_fail)

    with pytest.raises(
        PromptFactoryMaterializationError,
        match="tasks_schema validation failed during materialization",
    ):
        _materialize(tmp_path)

    assert calls["count"] == 2


def test_materialization_dry_run_returns_planned_files_without_writing(
    tmp_path: Path,
):
    result = _materialize(tmp_path, dry_run=True)

    assert result.dry_run is True
    assert result.written_files == ()
    assert sorted(path.relative_to(result.target_dir).as_posix()
                  for path in result.planned_files) == [
        "prompt.md",
        "prompts/task-1-1-build-first-slice.md",
        "prompts/task-1-2-build-second-slice.md",
        "reviews/review-task-1-1-build-first-slice.md",
        "reviews/review-task-1-2-build-second-slice.md",
        "tasks.md",
    ]
    assert not result.target_dir.exists()


def test_materialized_prompts_use_full_contract_sections(tmp_path: Path):
    result = _materialize(tmp_path, dry_run=True)
    task_text = result.planned_files[
        result.target_dir / "prompts" / "task-1-1-build-first-slice.md"
    ]
    review_text = result.planned_files[
        result.target_dir / "reviews" / "review-task-1-1-build-first-slice.md"
    ]

    assert "## Required Read Order" in task_text
    assert "Execution mode: orchestrator" in task_text
    assert "## Applicable Invariants" in task_text
    assert "<Invariant 1 - e.g. access boundary>" in task_text
    assert "<Invariant 1 - e.g. access boundary>" in review_text
    assert "## Acceptance Matrix" in task_text
    assert "## Final Action Contract" in task_text
    assert "## Verdict Output Contract" in review_text
    assert "Severity: should-fix" in review_text
    assert "`_prompt_rules.md` Rule 5 signatures" in review_text
    assert "## Gate 6 - Test Quality" in review_text
    assert "## Calibration" in review_text


def test_materialized_prompts_render_project_invariants(tmp_path: Path):
    result = _materialize(
        tmp_path,
        dry_run=True,
        project_extra="""
invariants:
  - name: Access boundary
    applies: applies
    evidence: pytest tests/test_access.py -q
    status: "<PASS/FAIL/N/A>"
""",
    )
    task_text = result.planned_files[
        result.target_dir / "prompts" / "task-1-1-build-first-slice.md"
    ]
    review_text = result.planned_files[
        result.target_dir / "reviews" / "review-task-1-1-build-first-slice.md"
    ]

    assert "| Access boundary | applies | pytest tests/test_access.py -q |" in task_text
    assert (
        "| Access boundary | applies | pytest tests/test_access.py -q | "
        "<PASS/FAIL/N/A> |"
    ) in review_text


def test_materialized_acceptance_matrix_escapes_multiline_table_cells(
    tmp_path: Path,
):
    draft = _valid_draft()
    draft["tasks"][0]["prompt_summary"] = "Implement first line\nwith | pipe"
    result = _materialize(tmp_path, draft=draft, dry_run=True)
    task_text = result.planned_files[
        result.target_dir / "prompts" / "task-1-1-build-first-slice.md"
    ]

    row = next(
        line for line in task_text.splitlines()
        if line.startswith("| Implement first line")
    )
    assert row == "| Implement first line with \\| pipe | pytest tests/test_a.py -q | yes |"


def test_materialization_rejects_target_outside_iterations(tmp_path: Path):
    with pytest.raises(
        PromptFactoryMaterializationError,
        match="target must be under iterations",
    ):
        _materialize(tmp_path, target=Path("not-iterations/demo-i1"))


def test_materialization_rejects_path_traversal_target(tmp_path: Path):
    with pytest.raises(PromptFactoryMaterializationError, match="must not contain"):
        _materialize(tmp_path, target=Path("iterations/../outside"))


def test_existing_target_files_are_not_overwritten_by_default(tmp_path: Path):
    target = tmp_path / "iterations" / "demo" / "demo-i1"
    target.mkdir(parents=True)
    (target / "prompt.md").write_text("existing", encoding="utf-8")

    with pytest.raises(PromptFactoryMaterializationError, match="overwrite"):
        _materialize(tmp_path, target=target, dry_run=False)


def test_successful_materialization_writes_iteration_package_in_temp_fixture(
    tmp_path: Path,
):
    target = (
        tmp_path
        / "iterations"
        / "prompt-factory-materialization-test"
        / "demo-i1"
    )

    result = _materialize(tmp_path, target=target, dry_run=False)

    assert result.dry_run is False
    assert (target / "prompt.md").exists()
    assert (target / "tasks.md").exists()
    assert (target / "prompts" / "task-1-1-build-first-slice.md").exists()
    assert (target / "reviews" / "review-task-1-1-build-first-slice.md").exists()
    assert len(result.written_files) == 6

    board = parse_tasks_md(target / "tasks.md")
    assert [task.id for task in board.tasks] == ["TASK-1-1", "TASK-1-2"]
    assert not (
        Path.cwd()
        / "iterations"
        / "prompt-factory-materialization-test"
        / "demo-i1"
    ).exists()


def test_materialization_is_idempotent_and_references_resolve(tmp_path: Path):
    """Round-trip: re-validating the draft and re-rendering tasks.md must
    produce byte-identical content to what was written, and every prompt /
    review file referenced by the produced tasks.md must exist on disk.

    This is the deterministic "trust anchor" guarantee — validate_draft +
    render_tasks_md + materialize must agree, and parse_tasks_md must round-
    trip the produced tasks.md without surprise.
    """
    draft = _valid_draft()
    target = (
        tmp_path
        / "iterations"
        / "prompt-factory-round-trip"
        / "demo-i1"
    )

    result = _materialize(tmp_path, draft=draft, target=target, dry_run=False)
    assert result.dry_run is False

    # Re-render the same draft and assert the on-disk tasks.md matches byte-
    # for-byte. Catches drift between validate_draft and render_tasks_md.
    rerendered = render_tasks_md(draft)
    on_disk = (target / "tasks.md").read_text(encoding="utf-8")
    assert on_disk == rerendered

    # Re-validate the same draft a second time — must produce the same task
    # ids and the same allowed-files content. Catches non-determinism in
    # validate_draft.
    revalidated = validate_draft(draft)
    assert [t.id for t in revalidated.tasks] == ["TASK-1-1", "TASK-1-2"]
    assert list(revalidated.tasks[0].allowed_files) == [
        "src/a.py",
        "tests/test_a.py",
    ]

    # Round-trip through tasks_schema.parse_tasks_md and assert allowed_files
    # survive. Catches drift between rendered preview and parser.
    board = parse_tasks_md(target / "tasks.md")
    parsed_t1 = board.by_id("TASK-1-1")
    assert parsed_t1 is not None
    assert list(parsed_t1.allowed_files) == ["src/a.py", "tests/test_a.py"]

    # Every prompt / review file referenced by the rendered package must
    # exist on disk. Catches dangling references between tasks.md and the
    # actual files materialize wrote.
    for written in result.written_files:
        assert written.exists(), f"declared written file missing: {written}"

    expected_files = {
        target / "prompt.md",
        target / "tasks.md",
        target / "prompts" / "task-1-1-build-first-slice.md",
        target / "reviews" / "review-task-1-1-build-first-slice.md",
        target / "prompts" / "task-1-2-build-second-slice.md",
        target / "reviews" / "review-task-1-2-build-second-slice.md",
    }
    assert set(result.written_files) == expected_files
    for path in expected_files:
        assert path.exists()


_CLI_ROUND_TRIP_PROJECT_YAML = """\
project:
  name: prompt-factory-cli-roundtrip
  main_branch: main
  phase_branch_pattern: "phase-{phase}"
  iteration_branch_pattern: "{phase}/iteration-{n}"
  task_branch_pattern: "{phase}/i{n}/t{k}-{slug}"
stack:
  test: "echo test"
  lint: "echo lint"
risk:
  high_risk_globs: ["**/schema.sql"]
  sensitive_files: [".env"]
  forbidden_patterns: ["except: pass"]
agents:
  claude:
    type: claude
    cmd: "claude -p"
    family: anthropic
  codex:
    type: codex
    cmd: "codex"
    family: openai
costs:
  anthropic: {input: 3.0, output: 15.0}
  openai: {input: 2.5, output: 10.0}
"""


def test_cli_materialize_then_orch_validate_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    """B-M3 — drive the CLI entry point end to end.

    Closes the gap Codex's cross-vendor review flagged
    (`tools/logs/orch-upgrade-final-audit/codex-review/report.md`): the
    helper-level round-trip alone did not prove
    ``main(["prompt-factory", "materialize", ...])`` and
    ``main(["validate", iter_id])`` agree on the produced artifacts.
    """
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(
        _CLI_ROUND_TRIP_PROJECT_YAML, encoding="utf-8",
    )

    draft = _valid_draft()
    gate = _pass_review_gate()
    approval = _approval()
    draft_path = tmp_path / "draft.json"
    gate_path = tmp_path / "review_gate.json"
    approval_path = tmp_path / "approval.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    approval_path.write_text(json.dumps(approval), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    assert cli_main(["prompt-factory", "validate", str(draft_path)]) == 0
    capsys.readouterr()

    assert cli_main(
        ["prompt-factory", "review-package", str(draft_path)]
    ) == 0
    capsys.readouterr()

    target = Path("iterations") / "prompt-factory-cli-roundtrip" / "demo-i1"
    assert cli_main(
        [
            "prompt-factory",
            "materialize",
            str(draft_path),
            str(gate_path),
            str(approval_path),
            "--target",
            str(target),
            "--dry-run",
        ]
    ) == 0
    out = capsys.readouterr().out
    assert "would write" in out
    assert not (tmp_path / target).exists()

    assert cli_main(
        [
            "prompt-factory",
            "materialize",
            str(draft_path),
            str(gate_path),
            str(approval_path),
            "--target",
            str(target),
        ]
    ) == 0
    capsys.readouterr()
    iter_dir = tmp_path / target
    assert (iter_dir / "tasks.md").exists()
    assert (iter_dir / "prompt.md").exists()
    assert (iter_dir / "prompts" / "task-1-1-build-first-slice.md").exists()
    assert (
        iter_dir / "reviews" / "review-task-1-1-build-first-slice.md"
    ).exists()
    assert (iter_dir / "prompts" / "task-1-2-build-second-slice.md").exists()
    assert (
        iter_dir / "reviews" / "review-task-1-2-build-second-slice.md"
    ).exists()

    assert cli_main(["validate", draft["phase_or_iteration_id"]]) == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert draft["iteration_branch"] in out


# ---------------------------------------------------------------------------
# Close-out batch: task-branch namespace (objective 4) and test-field
# runnability parity with scaffold_lint (objective 5).
# ---------------------------------------------------------------------------


def test_task_branches_do_not_nest_under_iteration_branch():
    """A git ref X cannot coexist with refs nested under X/; the pilot hit this
    collision (commit f5dd2eb). Task branches must live in a sibling namespace.
    """
    from orch.prompt_factory import _task_branch

    draft = validate_draft(_valid_draft())
    iter_branch = draft.iteration_branch  # "demo/iteration-1"
    for task in draft.tasks:
        branch = _task_branch(iter_branch, task)
        assert branch != iter_branch
        # NOT nested directly under the iteration branch ref.
        assert not branch.startswith(iter_branch + "/"), branch
        # Lives under the distinct "<iteration_branch>-tasks/" namespace.
        assert branch.startswith(iter_branch + "-tasks/"), branch


def test_rendered_branch_column_uses_task_namespace():
    rendered = render_tasks_md(_valid_draft())
    assert "`demo/iteration-1-tasks/task-1-1-build-first-slice`" in rendered
    # The old colliding scheme must be gone.
    assert "`demo/iteration-1/task-1-1" not in rendered


def test_validate_rejects_non_runnable_test_command():
    """A test_command that scaffold_lint would reject must fail PF validation
    (before materialization), not slip through to `orch validate`."""
    draft = _valid_draft()
    draft["tasks"][0]["test_command"] = "git diff --check -- docs/x.md"
    _assert_invalid(draft, "must start with one of")


def test_test_note_renders_as_manual_note_and_passes_scaffold_lint():
    from orch.scaffold_lint import check_test_fields_runnable

    rendered = render_tasks_md(_valid_draft())
    # test_note becomes a manual annotation, never a **Test:** command line.
    assert "**Test note:** Covered by the same parser smoke test." in rendered
    assert "**Test:** Covered by" not in rendered
    # Parity: the rendered tasks.md passes the exact check `orch validate` runs.
    assert check_test_fields_runnable(rendered) == []
    # And the draft still validates end-to-end.
    validate_draft(_valid_draft())
