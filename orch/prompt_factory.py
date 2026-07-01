"""Deterministic Prompt Factory draft schema and validator.

This module is intentionally local and deterministic. It does not call
agents, infer missing fields, approve generated work, or write runnable
iteration packages. A draft is accepted only if its rendered ``tasks.md``
preview also passes :mod:`orch.tasks_schema`.
"""
from __future__ import annotations

import json
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from orch.config import (
    CORE_DEFAULTS,
    ConfigError,
    default_project_yaml_path,
    invariants as config_invariants,
    load_config,
    patterns as config_patterns,
)
from orch.model_routing import (
    ModelRoutingError,
    ModelRoutingDeclaration,
    resolve_model_routing,
    validate_model_tier,
    validate_reasoning_effort,
    validate_risk_category,
)
from orch.tasks_schema import (
    EMDASH,
    TaskPatterns,
    TasksMdError,
    _compile_task_patterns,
    parse_tasks_md,
)
from orch.scaffold_lint import check_test_fields_runnable


TOP_LEVEL_FIELDS = (
    "phase_or_iteration_id",
    "iteration_branch",
    "final_pr",
    "depends_on",
    "blocks",
    "execution_plan",
    "tasks",
)
EXECUTION_PLAN_FIELDS = ("approach", "qa", "note")
TASK_REQUIRED_FIELDS = (
    "id",
    "title",
    "dependencies",
    "allowed_files",
    "risk_category",
)
TASK_TEXT_ALTERNATIVES = (
    ("test_command", "test_note"),
    ("prompt_summary", "prompt_path_placeholder"),
    ("review_summary", "review_path_placeholder"),
)
PROMPT_FACTORY_LOG_DIRNAME = "prompt-factory"
REVIEW_ROLES = ("prompt_expert", "technical_reviewer")
REVIEW_VERDICTS = ("PASS", "CHANGES_REQUIRED", "BLOCKED")
REVIEW_GATE_STATUSES = (
    "PASS",
    "CHANGES_REQUIRED",
    "BLOCKED",
    "INCOMPLETE",
    "MALFORMED",
)
ROLE_REVIEW_PROMPT_FILENAMES = {
    "prompt_expert": "prompt_expert_review_prompt.md",
    "technical_reviewer": "technical_reviewer_review_prompt.md",
}
ROLE_REVIEW_FILENAMES = {
    "prompt_expert": "prompt_expert_review.md",
    "technical_reviewer": "technical_reviewer_review.md",
}
REVIEW_GATE_FILENAME = "review_gate.json"
APPROVAL_DECISIONS = ("approved", "rejected", "deferred")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_VERDICT_RE = re.compile(
    r"^\s*(?:\*\*)?Verdict(?:\*\*)?:\s*"
    r"(?P<verdict>PASS|CHANGES_REQUIRED|BLOCKED)\s*$",
    re.MULTILINE,
)


class PromptFactoryValidationError(ValueError):
    """Raised when a Prompt Factory draft fails deterministic validation."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("\n".join(errors))


class PromptFactoryReviewError(ValueError):
    """Raised when Prompt Factory review artifacts are malformed."""


class PromptFactoryApprovalError(ValueError):
    """Raised when operator approval evidence is missing or invalid."""


class PromptFactoryMaterializationError(ValueError):
    """Raised when a Prompt Factory draft cannot be materialized safely."""


@dataclass(frozen=True)
class PromptFactoryExecutionPlan:
    approach: str
    qa: str
    note: str


@dataclass(frozen=True)
class PromptFactoryTask:
    id: str
    title: str
    dependencies: tuple[str, ...]
    allowed_files: tuple[str, ...]
    test_command: str | None
    test_note: str | None
    prompt_summary: str | None
    prompt_path_placeholder: str | None
    review_summary: str | None
    review_path_placeholder: str | None
    risk_category: str
    model_tier: str | None = None
    reasoning_effort: str | None = None
    parallel_safe: "PromptFactoryParallelSafety | None" = None


@dataclass(frozen=True)
class PromptFactoryDraft:
    phase_or_iteration_id: str
    iteration_branch: str
    final_pr: str
    depends_on: str
    blocks: str
    execution_plan: PromptFactoryExecutionPlan
    tasks: tuple[PromptFactoryTask, ...]


@dataclass(frozen=True)
class PromptFactoryParallelSafety:
    value: bool
    reason: str
    conflicts: tuple[str, ...]
    requires_serial_after: tuple[str, ...]


@dataclass(frozen=True)
class ProjectInvariant:
    name: str
    applies: str
    evidence: str
    status: str


@dataclass(frozen=True)
class PromptFactoryRoleDecision:
    role: str
    state: str
    verdict: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class PromptFactoryGateDecision:
    status: str
    roles: tuple[PromptFactoryRoleDecision, ...]


@dataclass(frozen=True)
class PromptFactoryReviewPackage:
    draft_id: str
    artifact_dir: Path
    prompt_paths: dict[str, Path]
    gate_path: Path
    decision: PromptFactoryGateDecision


@dataclass(frozen=True)
class PromptFactoryApproval:
    draft_id: str
    approved_by: str
    approved_at: str
    decision: str
    review_gate_status: str
    notes: str


@dataclass(frozen=True)
class PromptFactoryMaterializationResult:
    draft_id: str
    target_dir: Path
    planned_files: dict[Path, str]
    dry_run: bool
    written_files: tuple[Path, ...] = ()


def load_draft_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PromptFactoryValidationError(
            [f"{path}: draft JSON file not found"]
        ) from None
    except JSONDecodeError as exc:
        raise PromptFactoryValidationError(
            [f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"]
        ) from exc
    if not isinstance(raw, Mapping):
        raise PromptFactoryValidationError(["draft root must be a JSON object"])
    return raw


def load_review_gate_json(path: Path) -> Mapping[str, Any]:
    return _load_json_object(
        path,
        label="review gate",
        error_cls=PromptFactoryMaterializationError,
    )


def load_operator_approval_json(path: Path) -> Mapping[str, Any]:
    return _load_json_object(
        path,
        label="approval artifact",
        error_cls=PromptFactoryApprovalError,
    )


def _resolve_prompt_factory_patterns(
    *,
    patterns: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if patterns is not None:
        return dict(patterns)
    root = repo_root if repo_root is not None else Path.cwd()
    config_path = default_project_yaml_path(root)
    if not config_path.exists():
        return dict(CORE_DEFAULTS["patterns"])
    try:
        return dict(config_patterns(load_config(config_path)))
    except ConfigError as exc:
        raise PromptFactoryValidationError([f"project.yaml error: {exc}"]) from None


def _resolve_project_invariants(repo_root: Path) -> tuple[ProjectInvariant, ...]:
    config_path = default_project_yaml_path(repo_root)
    if not config_path.exists():
        return ()
    try:
        raw = config_invariants(load_config(config_path))
    except ConfigError as exc:
        raise PromptFactoryMaterializationError(
            f"project.yaml error: {exc}"
        ) from None
    out: list[ProjectInvariant] = []
    for item in raw:
        out.append(
            ProjectInvariant(
                name=item["name"],
                applies=item.get("applies", "<applies / preserve only / N/A>"),
                evidence=item.get("evidence", "<evidence or N/A reason>"),
                status=item.get("status", "<PASS/FAIL/N/A>"),
            )
        )
    return tuple(out)


def _compile_prompt_factory_task_patterns(
    patterns: Mapping[str, Any],
) -> TaskPatterns:
    try:
        return _compile_task_patterns(patterns)
    except ValueError as exc:
        raise PromptFactoryValidationError([str(exc)]) from None


def validate_draft(
    raw: Mapping[str, Any],
    *,
    patterns: Mapping[str, Any] | None = None,
) -> PromptFactoryDraft:
    resolved_patterns = _resolve_prompt_factory_patterns(patterns=patterns)
    task_patterns = _compile_prompt_factory_task_patterns(resolved_patterns)
    draft = _coerce_draft(raw, task_patterns=task_patterns)
    _validate_rendered_tasks_md(draft, patterns=resolved_patterns)
    return draft


def render_tasks_md(
    raw_or_draft: Mapping[str, Any] | PromptFactoryDraft,
    *,
    patterns: Mapping[str, Any] | None = None,
) -> str:
    draft = (
        raw_or_draft
        if isinstance(raw_or_draft, PromptFactoryDraft)
        else validate_draft(raw_or_draft, patterns=patterns)
    )
    return _render_tasks_md(draft)


def build_review_prompts(
    raw_or_draft: Mapping[str, Any] | PromptFactoryDraft,
    *,
    patterns: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build deterministic review prompts for the two required gate roles."""
    draft = (
        raw_or_draft
        if isinstance(raw_or_draft, PromptFactoryDraft)
        else validate_draft(raw_or_draft, patterns=patterns)
    )
    rendered = _render_tasks_md(draft)
    common = _render_review_common_context(draft, rendered)
    return {
        "prompt_expert": _render_prompt_expert_review_prompt(common),
        "technical_reviewer": _render_technical_reviewer_review_prompt(common),
    }


def parse_review_verdict(text: str) -> str:
    matches = [m.group("verdict") for m in _VERDICT_RE.finditer(text)]
    if len(matches) != 1:
        allowed = ", ".join(REVIEW_VERDICTS)
        raise PromptFactoryReviewError(
            "review artifact must contain exactly one verdict line in the "
            f"form 'Verdict: <value>'; allowed values: {allowed}"
        )
    return matches[0]


def decide_review_gate(
    review_texts: Mapping[str, str | None],
) -> PromptFactoryGateDecision:
    extra_roles = sorted(set(review_texts) - set(REVIEW_ROLES))
    if extra_roles:
        raise PromptFactoryReviewError(
            "unknown Prompt Factory review role(s): "
            + ", ".join(extra_roles)
        )

    role_decisions: list[PromptFactoryRoleDecision] = []
    for role in REVIEW_ROLES:
        text = review_texts.get(role)
        if not text or not text.strip():
            role_decisions.append(
                PromptFactoryRoleDecision(
                    role=role,
                    state="MISSING",
                    error="review artifact missing",
                )
            )
            continue
        try:
            verdict = parse_review_verdict(text)
        except PromptFactoryReviewError as exc:
            role_decisions.append(
                PromptFactoryRoleDecision(
                    role=role,
                    state="MALFORMED",
                    error=str(exc),
                )
            )
            continue
        role_decisions.append(
            PromptFactoryRoleDecision(
                role=role,
                state="PRESENT",
                verdict=verdict,
            )
        )

    if any(role.state == "MALFORMED" for role in role_decisions):
        status = "MALFORMED"
    elif any(role.state == "MISSING" for role in role_decisions):
        status = "INCOMPLETE"
    elif any(role.verdict == "BLOCKED" for role in role_decisions):
        status = "BLOCKED"
    elif any(role.verdict == "CHANGES_REQUIRED" for role in role_decisions):
        status = "CHANGES_REQUIRED"
    else:
        status = "PASS"
    return PromptFactoryGateDecision(
        status=status,
        roles=tuple(role_decisions),
    )


def prompt_factory_artifact_dir(
    log_root: Path,
    draft_id: str,
    *,
    log_dirname: str = PROMPT_FACTORY_LOG_DIRNAME,
) -> Path:
    return log_root / log_dirname / _safe_artifact_id(draft_id)


def write_review_package(
    raw_or_draft: Mapping[str, Any] | PromptFactoryDraft,
    *,
    log_root: Path,
    prompt_factory_log_dirname: str = PROMPT_FACTORY_LOG_DIRNAME,
) -> PromptFactoryReviewPackage:
    draft = (
        raw_or_draft
        if isinstance(raw_or_draft, PromptFactoryDraft)
        else validate_draft(raw_or_draft)
    )
    artifact_dir = prompt_factory_artifact_dir(
        log_root,
        draft.phase_or_iteration_id,
        log_dirname=prompt_factory_log_dirname,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_review_prompts(draft)
    prompt_paths: dict[str, Path] = {}
    for role in REVIEW_ROLES:
        path = artifact_dir / ROLE_REVIEW_PROMPT_FILENAMES[role]
        path.write_text(prompts[role], encoding="utf-8")
        prompt_paths[role] = path

    decision = decide_review_gate({})
    gate_path = _write_review_gate(
        artifact_dir,
        draft_id=draft.phase_or_iteration_id,
        decision=decision,
        prompt_paths=prompt_paths,
    )
    return PromptFactoryReviewPackage(
        draft_id=draft.phase_or_iteration_id,
        artifact_dir=artifact_dir,
        prompt_paths=prompt_paths,
        gate_path=gate_path,
        decision=decision,
    )


def write_review_gate_status(
    *,
    log_root: Path,
    draft_id: str,
    prompt_factory_log_dirname: str = PROMPT_FACTORY_LOG_DIRNAME,
) -> PromptFactoryGateDecision:
    artifact_dir = prompt_factory_artifact_dir(
        log_root,
        draft_id,
        log_dirname=prompt_factory_log_dirname,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    review_texts: dict[str, str | None] = {}
    for role in REVIEW_ROLES:
        path = artifact_dir / ROLE_REVIEW_FILENAMES[role]
        review_texts[role] = (
            path.read_text(encoding="utf-8") if path.exists() else None
        )
    decision = decide_review_gate(review_texts)
    prompt_paths = {
        role: artifact_dir / ROLE_REVIEW_PROMPT_FILENAMES[role]
        for role in REVIEW_ROLES
    }
    _write_review_gate(
        artifact_dir,
        draft_id=draft_id,
        decision=decision,
        prompt_paths=prompt_paths,
    )
    return decision


def validate_review_gate_artifact(
    raw: Mapping[str, Any],
    *,
    draft_id: str,
) -> str:
    if not isinstance(raw, Mapping):
        raise PromptFactoryMaterializationError(
            "review gate validation failed: review gate root must be an object"
        )
    gate_draft_id = _require_gate_string(raw, "draft_id")
    status = _require_gate_string(raw, "status")
    if gate_draft_id != draft_id:
        raise PromptFactoryMaterializationError(
            "review gate validation failed: draft_id mismatch "
            f"{gate_draft_id!r} != {draft_id!r}"
        )
    if status != "PASS":
        raise PromptFactoryMaterializationError(
            "review gate validation failed: review gate status must be PASS, "
            f"got {status!r}"
        )

    required_roles = raw.get("required_roles")
    if required_roles != list(REVIEW_ROLES):
        raise PromptFactoryMaterializationError(
            "review gate validation failed: required_roles must be exactly "
            f"{list(REVIEW_ROLES)!r}"
        )
    roles = raw.get("roles")
    if not isinstance(roles, Mapping):
        raise PromptFactoryMaterializationError(
            "review gate validation failed: roles must be an object"
        )
    for role in REVIEW_ROLES:
        role_payload = roles.get(role)
        if not isinstance(role_payload, Mapping):
            raise PromptFactoryMaterializationError(
                "review gate validation failed: missing role payload "
                f"for {role!r}"
            )
        verdict = role_payload.get("verdict")
        if verdict != "PASS":
            raise PromptFactoryMaterializationError(
                "review gate validation failed: role "
                f"{role!r} must have verdict PASS, got {verdict!r}"
            )
    return status


def validate_operator_approval(
    raw: Mapping[str, Any],
    *,
    draft_id: str,
    review_gate_status: str,
) -> PromptFactoryApproval:
    if not isinstance(raw, Mapping):
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: approval root must be an object"
        )
    missing = [
        field
        for field in (
            "draft_id",
            "approved_by",
            "approved_at",
            "decision",
            "review_gate_status",
            "notes",
        )
        if field not in raw
    ]
    if missing:
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: missing required field(s): "
            + ", ".join(missing)
        )

    approval_draft_id = _require_approval_string(raw, "draft_id")
    approved_by = _require_approval_string(raw, "approved_by")
    approved_at = _require_approval_string(raw, "approved_at")
    decision = _require_approval_string(raw, "decision")
    artifact_gate_status = _require_approval_string(
        raw,
        "review_gate_status",
    )
    notes = _require_approval_string(raw, "notes", allow_empty=True)

    if approval_draft_id != draft_id:
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: draft_id mismatch "
            f"{approval_draft_id!r} != {draft_id!r}"
        )
    if decision not in APPROVAL_DECISIONS:
        allowed = ", ".join(APPROVAL_DECISIONS)
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: invalid decision "
            f"{decision!r}; allowed: {allowed}"
        )
    if decision != "approved":
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: decision must be 'approved' "
            f"for materialization, got {decision!r}"
        )
    if review_gate_status != "PASS":
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: review gate status must be "
            f"PASS before materialization, got {review_gate_status!r}"
        )
    if artifact_gate_status != "PASS":
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: review_gate_status must be "
            f"PASS, got {artifact_gate_status!r}"
        )
    if artifact_gate_status != review_gate_status:
        raise PromptFactoryApprovalError(
            "approval artifact validation failed: review_gate_status mismatch "
            f"{artifact_gate_status!r} != {review_gate_status!r}"
        )
    return PromptFactoryApproval(
        draft_id=approval_draft_id,
        approved_by=approved_by,
        approved_at=approved_at,
        decision=decision,
        review_gate_status=artifact_gate_status,
        notes=notes,
    )


def check_prompt_factory_approval(
    raw_draft: Mapping[str, Any],
    review_gate: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> PromptFactoryApproval:
    draft = validate_draft(raw_draft)
    review_gate_status = validate_review_gate_artifact(
        review_gate,
        draft_id=draft.phase_or_iteration_id,
    )
    return validate_operator_approval(
        approval,
        draft_id=draft.phase_or_iteration_id,
        review_gate_status=review_gate_status,
    )


def materialize_prompt_factory_draft(
    raw_draft: Mapping[str, Any],
    review_gate: Mapping[str, Any],
    approval: Mapping[str, Any],
    *,
    repo_root: Path,
    target: Path,
    dry_run: bool,
    force: bool = False,
    iteration_root: str = "iterations",
    prompt_filename: str = "prompt.md",
    task_board_filename: str = "tasks.md",
    task_prompts_dirname: str = "prompts",
    task_reviews_dirname: str = "reviews",
) -> PromptFactoryMaterializationResult:
    resolved_patterns = _resolve_prompt_factory_patterns(repo_root=repo_root)
    draft = validate_draft(raw_draft, patterns=resolved_patterns)
    review_gate_status = validate_review_gate_artifact(
        review_gate,
        draft_id=draft.phase_or_iteration_id,
    )
    operator_approval = validate_operator_approval(
        approval,
        draft_id=draft.phase_or_iteration_id,
        review_gate_status=review_gate_status,
    )
    tasks_md = _render_tasks_md(draft)
    _validate_materialized_tasks_md(tasks_md, patterns=resolved_patterns)
    project_invariants = _resolve_project_invariants(repo_root)
    target_dir = _resolve_materialization_target(
        repo_root,
        target,
        iteration_root=iteration_root,
    )
    planned_files = _build_materialized_files(
        draft,
        approval=operator_approval,
        target_dir=target_dir,
        tasks_md=tasks_md,
        project_invariants=project_invariants,
        prompt_filename=prompt_filename,
        task_board_filename=task_board_filename,
        task_prompts_dirname=task_prompts_dirname,
        task_reviews_dirname=task_reviews_dirname,
    )
    if not force:
        existing = [path for path in planned_files if path.exists()]
        if existing:
            rendered = ", ".join(str(path) for path in sorted(existing))
            raise PromptFactoryMaterializationError(
                "overwrite safety failed: target file(s) already exist; "
                f"use --force to overwrite: {rendered}"
            )
    if dry_run:
        return PromptFactoryMaterializationResult(
            draft_id=draft.phase_or_iteration_id,
            target_dir=target_dir,
            planned_files=planned_files,
            dry_run=True,
        )

    written: list[Path] = []
    for path, content in planned_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return PromptFactoryMaterializationResult(
        draft_id=draft.phase_or_iteration_id,
        target_dir=target_dir,
        planned_files=planned_files,
        dry_run=False,
        written_files=tuple(written),
    )


def _coerce_draft(
    raw: Mapping[str, Any],
    *,
    task_patterns: TaskPatterns,
) -> PromptFactoryDraft:
    errors: list[str] = []
    if not isinstance(raw, Mapping):
        raise PromptFactoryValidationError(["draft root must be a JSON object"])

    for field in TOP_LEVEL_FIELDS:
        if field not in raw:
            errors.append(f"missing required top-level field: {field}")

    plan = _coerce_execution_plan(raw.get("execution_plan"), errors)
    tasks = _coerce_tasks(raw.get("tasks"), errors, task_patterns=task_patterns)

    if errors:
        raise PromptFactoryValidationError(errors)

    assert plan is not None
    return PromptFactoryDraft(
        phase_or_iteration_id=_require_string(raw, "phase_or_iteration_id"),
        iteration_branch=_require_string(raw, "iteration_branch"),
        final_pr=_require_string(raw, "final_pr"),
        depends_on=_require_string(raw, "depends_on"),
        blocks=_require_string(raw, "blocks"),
        execution_plan=plan,
        tasks=tuple(tasks),
    )


def _coerce_execution_plan(
    raw: Any, errors: list[str],
) -> PromptFactoryExecutionPlan | None:
    if not isinstance(raw, Mapping):
        if raw is not None:
            errors.append("execution_plan must be an object")
        return None
    for field in EXECUTION_PLAN_FIELDS:
        if field not in raw:
            errors.append(f"missing required execution_plan field: {field}")
    if any(field not in raw for field in EXECUTION_PLAN_FIELDS):
        return None
    try:
        return PromptFactoryExecutionPlan(
            approach=_require_string(raw, "approach", path="execution_plan"),
            qa=_require_string(raw, "qa", path="execution_plan"),
            note=_require_string(raw, "note", path="execution_plan"),
        )
    except PromptFactoryValidationError as exc:
        errors.extend(exc.errors)
        return None


def _coerce_tasks(
    raw: Any,
    errors: list[str],
    *,
    task_patterns: TaskPatterns,
) -> list[PromptFactoryTask]:
    if not isinstance(raw, list):
        if raw is not None:
            errors.append("tasks must be a list")
        return []
    if not raw:
        errors.append("tasks must contain at least one task")
        return []

    tasks: list[PromptFactoryTask] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw):
        path = f"tasks[{idx}]"
        if not isinstance(item, Mapping):
            errors.append(f"{path} must be an object")
            continue
        task = _coerce_task(
            item,
            path=path,
            errors=errors,
            task_patterns=task_patterns,
        )
        if task is None:
            continue
        if task.id in seen:
            errors.append(f"{path}.id duplicate task id {task.id!r}")
        seen.add(task.id)
        tasks.append(task)

    ids = {task.id for task in tasks}
    for task in tasks:
        for dep in task.dependencies:
            if dep not in ids:
                errors.append(
                    f"task {task.id} dependency {dep!r} references unknown task"
                )
        if task.parallel_safe is not None:
            for dep in task.parallel_safe.requires_serial_after:
                if dep not in ids:
                    errors.append(
                        f"task {task.id} parallel_safe.requires_serial_after "
                        f"{dep!r} references unknown task"
                    )
    return tasks


def _coerce_task(
    raw: Mapping[str, Any],
    *,
    path: str,
    errors: list[str],
    task_patterns: TaskPatterns,
) -> PromptFactoryTask | None:
    for field in TASK_REQUIRED_FIELDS:
        if field not in raw:
            errors.append(f"{path}: missing required task field: {field}")
    for left, right in TASK_TEXT_ALTERNATIVES:
        if not _has_text(raw, left) and not _has_text(raw, right):
            errors.append(
                f"{path}: one of {left} or {right} is required"
            )
    if any(field not in raw for field in TASK_REQUIRED_FIELDS):
        return None

    try:
        task_id = _require_string(raw, "id", path=path)
        title = _require_string(raw, "title", path=path)
        dependencies = _require_string_list(
            raw, "dependencies", path=path, allow_empty=True
        )
        allowed_files = _require_string_list(
            raw, "allowed_files", path=path, allow_empty=False
        )
        risk_category = _require_string(raw, "risk_category", path=path)
        test_command = _optional_string(raw, "test_command", path=path)
        test_note = _optional_string(raw, "test_note", path=path)
        prompt_summary = _optional_string(raw, "prompt_summary", path=path)
        prompt_path = _optional_string(
            raw, "prompt_path_placeholder", path=path
        )
        review_summary = _optional_string(raw, "review_summary", path=path)
        review_path = _optional_string(
            raw, "review_path_placeholder", path=path
        )
        model_tier = _optional_string(raw, "model_tier", path=path)
        reasoning_effort = _optional_string(
            raw, "reasoning_effort", path=path
        )
        parallel_safe = _coerce_parallel_safe(
            raw.get("parallel_safe"), path=path, errors=errors
        )
    except PromptFactoryValidationError as exc:
        errors.extend(exc.errors)
        return None

    if not task_patterns.task_id_re.match(task_id):
        errors.append(
            f"{path}.id {task_id!r} must match configured task_id pattern "
            f"{task_patterns.task_id_pattern!r}"
        )
    try:
        validate_risk_category(risk_category)
    except ModelRoutingError as exc:
        errors.append(f"{path}.risk_category: {exc}")
    if model_tier is not None:
        try:
            validate_model_tier(model_tier)
        except ModelRoutingError as exc:
            errors.append(f"{path}.model_tier: {exc}")
    if reasoning_effort is not None:
        try:
            validate_reasoning_effort(reasoning_effort)
        except ModelRoutingError as exc:
            errors.append(f"{path}.reasoning_effort: {exc}")
    return PromptFactoryTask(
        id=task_id,
        title=title,
        dependencies=tuple(dependencies),
        allowed_files=tuple(allowed_files),
        test_command=test_command,
        test_note=test_note,
        prompt_summary=prompt_summary,
        prompt_path_placeholder=prompt_path,
        review_summary=review_summary,
        review_path_placeholder=review_path,
        risk_category=risk_category,
        model_tier=model_tier,
        reasoning_effort=reasoning_effort,
        parallel_safe=parallel_safe,
    )


def _coerce_parallel_safe(
    raw: Any, *, path: str, errors: list[str],
) -> PromptFactoryParallelSafety | None:
    if raw is None:
        return None
    label = f"{path}.parallel_safe"
    if not isinstance(raw, Mapping):
        errors.append(f"{label} must be an object")
        return None
    missing = [
        field
        for field in ("value", "reason", "conflicts", "requires_serial_after")
        if field not in raw
    ]
    if missing:
        errors.append(f"{label} missing required field(s): {missing}")
        return None
    value = raw.get("value")
    if not isinstance(value, bool):
        errors.append(f"{label}.value must be a boolean")
        value = False
    try:
        reason = _require_string(raw, "reason", path=label)
        conflicts = _require_string_list(
            raw, "conflicts", path=label, allow_empty=True
        )
        requires_serial_after = _require_string_list(
            raw,
            "requires_serial_after",
            path=label,
            allow_empty=True,
        )
    except PromptFactoryValidationError as exc:
        errors.extend(exc.errors)
        return None
    if any(ch in reason for ch in "\n;"):
        errors.append(f"{label}.reason must not contain newlines or semicolons")
    for field, values in (
        ("conflicts", conflicts),
        ("requires_serial_after", requires_serial_after),
    ):
        for idx, item in enumerate(values):
            if any(ch in item for ch in "\n;,"):
                errors.append(
                    f"{label}.{field}[{idx}] must not contain newlines, "
                    "semicolons, or commas"
                )
    if any(error.startswith(label) for error in errors):
        return None
    return PromptFactoryParallelSafety(
        value=bool(value),
        reason=reason,
        conflicts=tuple(conflicts),
        requires_serial_after=tuple(requires_serial_after),
    )


def _require_string(
    raw: Mapping[str, Any], field: str, *, path: str = "",
) -> str:
    label = f"{path}.{field}" if path else field
    value = raw.get(field)
    if not isinstance(value, str):
        raise PromptFactoryValidationError([f"{label} must be a string"])
    value = value.strip()
    if not value:
        raise PromptFactoryValidationError([f"{label} must be non-empty"])
    return value


def _require_gate_string(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str):
        raise PromptFactoryMaterializationError(
            f"review gate validation failed: {field} must be a string"
        )
    value = value.strip()
    if not value:
        raise PromptFactoryMaterializationError(
            f"review gate validation failed: {field} must be non-empty"
        )
    return value


def _require_approval_string(
    raw: Mapping[str, Any],
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = raw.get(field)
    if not isinstance(value, str):
        raise PromptFactoryApprovalError(
            f"approval artifact validation failed: {field} must be a string"
        )
    value = value.strip()
    if not value and not allow_empty:
        raise PromptFactoryApprovalError(
            f"approval artifact validation failed: {field} must be non-empty"
        )
    return value


def _optional_string(
    raw: Mapping[str, Any], field: str, *, path: str,
) -> str | None:
    if field not in raw or raw[field] is None:
        return None
    return _require_string(raw, field, path=path)


def _require_string_list(
    raw: Mapping[str, Any],
    field: str,
    *,
    path: str,
    allow_empty: bool,
) -> list[str]:
    label = f"{path}.{field}"
    value = raw.get(field)
    if not isinstance(value, list):
        raise PromptFactoryValidationError([f"{label} must be a list"])
    if not value and not allow_empty:
        raise PromptFactoryValidationError([f"{label} must be non-empty"])
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise PromptFactoryValidationError(
                [f"{label}[{idx}] must be a string"]
            )
        item = item.strip()
        if not item:
            raise PromptFactoryValidationError(
                [f"{label}[{idx}] must be non-empty"]
            )
        out.append(item)
    return out


def _has_text(raw: Mapping[str, Any], field: str) -> bool:
    return isinstance(raw.get(field), str) and bool(raw[field].strip())


def _validate_rendered_tasks_md(
    draft: PromptFactoryDraft,
    *,
    patterns: Mapping[str, Any],
) -> None:
    preview = _render_tasks_md(draft)
    with tempfile.TemporaryDirectory(prefix="orch-prompt-factory-") as tmp:
        path = Path(tmp) / "tasks.md"
        path.write_text(preview, encoding="utf-8")
        try:
            parse_tasks_md(path, patterns=patterns)
        except TasksMdError as exc:
            raise PromptFactoryValidationError(
                ["rendered tasks.md failed tasks_schema validation:", str(exc)]
            ) from exc
    runnable_errors = check_test_fields_runnable(preview)
    if runnable_errors:
        raise PromptFactoryValidationError(
            [
                "rendered tasks.md has non-runnable **Test:** field(s) that "
                "would fail `orch validate` (scaffold_lint test_runnable). "
                "Provide a test_command starting with an allowed runner "
                "(pytest, python -m, ruff, npm, npx, bash -c) or use "
                "test_note for a manual check:",
                *runnable_errors,
            ]
        )


def _validate_materialized_tasks_md(
    rendered_tasks_md: str,
    *,
    patterns: Mapping[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="orch-prompt-factory-materialize-"
    ) as tmp:
        path = Path(tmp) / "tasks.md"
        path.write_text(rendered_tasks_md, encoding="utf-8")
        try:
            parse_tasks_md(path, patterns=patterns)
        except TasksMdError as exc:
            raise PromptFactoryMaterializationError(
                "tasks_schema validation failed during materialization: "
                f"{exc}"
            ) from exc
    runnable_errors = check_test_fields_runnable(rendered_tasks_md)
    if runnable_errors:
        raise PromptFactoryMaterializationError(
            "non-runnable **Test:** field(s) during materialization "
            f"(scaffold_lint test_runnable): {runnable_errors}"
        )


def _render_tasks_md(draft: PromptFactoryDraft) -> str:
    lines: list[str] = [
        f"# Prompt Factory Draft {draft.phase_or_iteration_id}",
        "## Task Board",
        "",
        "**Status:** WAITING",
        f"**Iteration branch:** `{draft.iteration_branch}`",
        f"**Final PR:** {draft.final_pr}",
        f"**Depends on:** {draft.depends_on}",
        f"**Blocks:** {draft.blocks}",
        "",
        "---",
        "",
        "## Execution Plan",
        f"- approach: {draft.execution_plan.approach}",
        f"- qa: {draft.execution_plan.qa}",
        f"- note: {draft.execution_plan.note}",
        "",
        "---",
        "",
        "## Tasks",
        "",
        "| ID | Title | Owner | Status | Depends on | Branch |",
        "|----|-------|-------|--------|------------|--------|",
    ]
    for task in draft.tasks:
        deps = ", ".join(task.dependencies) if task.dependencies else EMDASH
        branch = _task_branch(draft.iteration_branch, task)
        lines.append(
            f"| {task.id} | {task.title} | TBD | WAITING | {deps} | "
            f"`{branch}` |"
        )
    lines.extend(["", "---", "", "## Task Details", ""])
    for task in draft.tasks:
        lines.extend(_render_task_detail(task))
    return "\n".join(lines).rstrip() + "\n"


def _render_review_common_context(
    draft: PromptFactoryDraft,
    rendered_tasks_md: str,
) -> str:
    lines = [
        "# Prompt Factory Review Gate Context",
        "",
        "## Draft Summary",
        f"- phase_or_iteration_id: {draft.phase_or_iteration_id}",
        f"- iteration_branch: {draft.iteration_branch}",
        f"- final_pr: {draft.final_pr}",
        f"- depends_on: {draft.depends_on}",
        f"- blocks: {draft.blocks}",
        f"- execution_plan.approach: {draft.execution_plan.approach}",
        f"- execution_plan.qa: {draft.execution_plan.qa}",
        f"- execution_plan.note: {draft.execution_plan.note}",
        "",
        "## Validation Results",
        (
            "- PASS: prompt_factory.validate_draft accepted the draft and the "
            "rendered tasks.md preview passed tasks_schema.py."
        ),
        "",
        "## Task List",
    ]
    for task in draft.tasks:
        routing = resolve_model_routing(
            ModelRoutingDeclaration(
                model_tier=task.model_tier,
                reasoning_effort=task.reasoning_effort,
                risk_category=task.risk_category,
            )
        )
        deps = ", ".join(task.dependencies) if task.dependencies else "none"
        allowed = ", ".join(task.allowed_files)
        declared_tier = task.model_tier or "(default)"
        declared_effort = task.reasoning_effort or "(default)"
        lines.extend(
            [
                f"- {task.id}: {task.title}",
                f"  - dependencies: {deps}",
                f"  - allowed_files: {allowed}",
                f"  - risk_category: {task.risk_category}",
                (
                    "  - resolved_routing: "
                    f"model_tier={routing.model_tier}, "
                    f"reasoning_effort={routing.reasoning_effort}, "
                    f"declared_model_tier={declared_tier}, "
                    f"declared_reasoning_effort={declared_effort}, "
                    f"floor_model_tier={routing.floor_model_tier}, "
                    f"floor_reasoning_effort={routing.floor_reasoning_effort}, "
                    f"dual_model_required={str(routing.dual_model_required).lower()}"
                ),
            ]
        )
        if task.test_command:
            lines.append(f"  - test_command: {task.test_command}")
        elif task.test_note:
            lines.append(f"  - test_note: {task.test_note}")
        if task.prompt_summary:
            lines.append(f"  - prompt_summary: {task.prompt_summary}")
        elif task.prompt_path_placeholder:
            lines.append(
                f"  - prompt_path_placeholder: {task.prompt_path_placeholder}"
            )
        if task.review_summary:
            lines.append(f"  - review_summary: {task.review_summary}")
        elif task.review_path_placeholder:
            lines.append(
                f"  - review_path_placeholder: {task.review_path_placeholder}"
            )
        if task.parallel_safe is not None:
            safety = task.parallel_safe
            lines.append(
                "  - parallel_safe: "
                f"value={str(safety.value).lower()}, "
                f"reason={safety.reason}, "
                f"conflicts={list(safety.conflicts)}, "
                f"requires_serial_after={list(safety.requires_serial_after)}"
            )
    lines.extend(
        [
            "",
            "## Rendered tasks.md Preview",
            "```markdown",
            rendered_tasks_md.rstrip(),
            "```",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_prompt_expert_review_prompt(common_context: str) -> str:
    return "\n".join(
        [
            "# Prompt Factory Review - Prompt Expert",
            "",
            common_context.rstrip(),
            "",
            "## Review Criteria",
            "- Task slicing is small, ordered, and reviewable.",
            "- Implementation prompts are clear and complete enough to execute.",
            "- Review criteria are clear enough for an independent reviewer.",
            "- Acceptance tests are not missing or only implied.",
            "- Hidden assumptions are surfaced instead of embedded silently.",
            "- Scope is not overly broad for the declared task boundaries.",
            "- Operator-decision leakage is not present in generated task work.",
            "",
            "## Verdict Format",
            "Return exactly one verdict line:",
            "Verdict: PASS",
            "Verdict: CHANGES_REQUIRED",
            "Verdict: BLOCKED",
            "",
            (
                "Use PASS only when the draft is ready for operator approval "
                "from the Prompt Expert perspective."
            ),
        ]
    ).rstrip() + "\n"


def _render_technical_reviewer_review_prompt(common_context: str) -> str:
    return "\n".join(
        [
            "# Prompt Factory Review - Technical Reviewer",
            "",
            common_context.rstrip(),
            "",
            "## Review Criteria",
            "- Allowed-files entries match the proposed work and are not too broad.",
            "- Task dependencies are correct and form a safe execution order.",
            "- Risk categories match the blast radius of each task.",
            "- Model-routing metadata is correct and respects deterministic floors.",
            "- Test commands or test notes are adequate for the requested changes.",
            "- The rendered preview remains compatible with tasks_schema.py.",
            (
                "- Project-specific invariants stay in project config or "
                "templates, not in reusable orchestrator core."
            ),
            "- Package/tooling paths are present only when the draft declares "
            "tooling work.",
            "",
            "## Verdict Format",
            "Return exactly one verdict line:",
            "Verdict: PASS",
            "Verdict: CHANGES_REQUIRED",
            "Verdict: BLOCKED",
            "",
            (
                "Use PASS only when the draft is ready for operator approval "
                "from the Technical Reviewer perspective."
            ),
        ]
    ).rstrip() + "\n"


def _render_task_detail(task: PromptFactoryTask) -> list[str]:
    lines = [
        f"### {task.id} {EMDASH} {task.title}",
        "",
        "**Allowed files:**",
        "```",
    ]
    lines.extend(task.allowed_files)
    lines.extend(["```", ""])
    if task.test_command:
        lines.append(f"**Test:** `{task.test_command}`")
    elif task.test_note:
        # A test_note is a manual/non-automated check. Render it as a distinct
        # field, NOT as ``**Test:**`` — scaffold_lint's test_runnable check
        # requires every non-empty ``**Test:**`` line to be a runnable shell
        # command, so emitting ``**Test:** <prose>`` here would later fail
        # `orch validate`. ``**Test note:**`` is not matched by that check.
        lines.append(f"**Test note:** {task.test_note}")
    lines.append("")
    if task.parallel_safe is not None:
        safety = task.parallel_safe
        flag = "yes" if safety.value else "no"
        lines.append(
            f"**Parallel safe:** {flag}; reason={safety.reason}; "
            f"conflicts={_render_parallel_list(safety.conflicts)}; "
            "requires_serial_after="
            f"{_render_parallel_list(safety.requires_serial_after)}"
        )
        lines.append("")
    if task.prompt_summary:
        lines.append(f"**Prompt summary:** {task.prompt_summary}")
    elif task.prompt_path_placeholder:
        lines.append(
            f"**Prompt path placeholder:** {task.prompt_path_placeholder}"
        )
    lines.append("")
    if task.review_summary:
        lines.append(f"**Review summary:** {task.review_summary}")
    elif task.review_path_placeholder:
        lines.append(
            f"**Review path placeholder:** {task.review_path_placeholder}"
        )
    lines.append("")
    fields: list[str] = []
    if task.model_tier:
        fields.append(f"model_tier={task.model_tier}")
    if task.reasoning_effort:
        fields.append(f"reasoning_effort={task.reasoning_effort}")
    fields.append(f"risk_category={task.risk_category}")
    lines.extend(["**Model routing:** `" + "; ".join(fields) + "`", ""])
    return lines


def _render_parallel_list(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"


#: Suffix appended to the iteration branch to form the task-branch namespace.
#: Task branches MUST NOT nest directly under the iteration branch ref:
#: git stores ``refs/heads/<iteration_branch>`` as a file, so a sibling
#: ``refs/heads/<iteration_branch>/<task>`` cannot also exist (the file would
#: have to be a directory). Rendering task branches under a distinct
#: ``<iteration_branch>-tasks/`` namespace keeps them as a sibling ref name and
#: provably avoids that collision. See the goal-converter pilot, where the
#: original ``<iteration_branch>/<task>`` scheme had to be hand-corrected
#: (commit f5dd2eb, "avoid pf task branch ref conflict").
TASK_BRANCH_NAMESPACE_SUFFIX = "-tasks"


def _task_branch(iteration_branch: str, task: PromptFactoryTask) -> str:
    slug = _slug(task.title)
    suffix = task.id.lower()
    if slug:
        suffix = f"{suffix}-{slug}"
    namespace = f"{iteration_branch.rstrip('/')}{TASK_BRANCH_NAMESPACE_SUFFIX}"
    return f"{namespace}/{suffix}"


def _slug(value: str) -> str:
    lowered = value.lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", lowered)
    return normalized.strip("-")


def _load_json_object(
    path: Path,
    *,
    label: str,
    error_cls: type[ValueError],
) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise error_cls(f"{label} file not found: {path}") from None
    except JSONDecodeError as exc:
        raise error_cls(
            f"{path}:{exc.lineno}:{exc.colno}: malformed {label} JSON: "
            f"{exc.msg}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise error_cls(f"{label} root must be a JSON object")
    return raw


def _resolve_materialization_target(
    repo_root: Path,
    target: Path,
    *,
    iteration_root: str = "iterations",
) -> Path:
    if any(part == ".." for part in target.parts):
        raise PromptFactoryMaterializationError(
            "target path safety failed: target must not contain '..'"
        )
    root = repo_root.resolve()
    iterations_root = (root / iteration_root).resolve(strict=False)
    target_path = target if target.is_absolute() else root / target
    resolved = target_path.resolve(strict=False)
    if not resolved.is_relative_to(iterations_root):
        raise PromptFactoryMaterializationError(
            f"target path safety failed: target must be under {iteration_root}/"
        )
    if resolved == iterations_root:
        raise PromptFactoryMaterializationError(
            "target path safety failed: target must be an iteration package "
            "directory below iterations/"
        )
    return resolved


def _build_materialized_files(
    draft: PromptFactoryDraft,
    *,
    approval: PromptFactoryApproval,
    target_dir: Path,
    tasks_md: str,
    project_invariants: tuple[ProjectInvariant, ...],
    prompt_filename: str = "prompt.md",
    task_board_filename: str = "tasks.md",
    task_prompts_dirname: str = "prompts",
    task_reviews_dirname: str = "reviews",
) -> dict[Path, str]:
    files: dict[Path, str] = {
        target_dir / prompt_filename: _render_materialized_prompt_md(
            draft,
            approval,
        ),
        target_dir / task_board_filename: tasks_md,
    }
    for task in draft.tasks:
        task_filename = _task_doc_filename(task)
        files[target_dir / task_prompts_dirname / task_filename] = (
            _render_materialized_task_prompt(
                draft,
                task,
                project_invariants=project_invariants,
            )
        )
        files[target_dir / task_reviews_dirname / f"review-{task_filename}"] = (
            _render_materialized_task_review(
                draft,
                task,
                project_invariants=project_invariants,
            )
        )
    return files


def _render_materialized_prompt_md(
    draft: PromptFactoryDraft,
    approval: PromptFactoryApproval,
) -> str:
    lines = [
        f"# Prompt Factory Materialized Draft {draft.phase_or_iteration_id}",
        "",
        "## Purpose",
        (
            "This iteration package was materialized by deterministic "
            "Prompt Factory tooling after review-gate PASS and explicit "
            "operator approval."
        ),
        "",
        "## Draft",
        f"- phase_or_iteration_id: {draft.phase_or_iteration_id}",
        f"- iteration_branch: {draft.iteration_branch}",
        f"- final_pr: {draft.final_pr}",
        f"- depends_on: {draft.depends_on}",
        f"- blocks: {draft.blocks}",
        "",
        "## Execution Plan",
        f"- approach: {draft.execution_plan.approach}",
        f"- qa: {draft.execution_plan.qa}",
        f"- note: {draft.execution_plan.note}",
        "",
        "## Operator Approval",
        f"- approved_by: {approval.approved_by}",
        f"- approved_at: {approval.approved_at}",
        f"- decision: {approval.decision}",
        f"- review_gate_status: {approval.review_gate_status}",
        f"- notes: {approval.notes or '(none)'}",
        "",
        "## Tasks",
    ]
    for task in draft.tasks:
        deps = ", ".join(task.dependencies) if task.dependencies else "none"
        lines.append(f"- {task.id}: {task.title} (depends_on: {deps})")
    return "\n".join(lines).rstrip() + "\n"


def _render_materialized_task_prompt(
    draft: PromptFactoryDraft,
    task: PromptFactoryTask,
    *,
    project_invariants: tuple[ProjectInvariant, ...],
) -> str:
    prompt_body = (
        task.prompt_summary
        or "STOP: draft provided only a prompt_path_placeholder "
        f"({task.prompt_path_placeholder}); operator must replace this section "
        "before runtime use."
    )
    acceptance = (
        task.test_command
        or task.test_note
        or "STOP: no task-specific test command or test note was provided."
    )
    lines = [
        f"# {task.id} {EMDASH} {task.title}",
        "",
        "## Execution Metadata",
        "",
        f"- Iteration: `{draft.phase_or_iteration_id}`",
        f"- Task ID: `{task.id}`",
        f"- Risk category: `{task.risk_category}`",
        "- Execution mode: orchestrator (operator may rerun manually; manual "
        "mode then requires explicit commit/push)",
        "- Template source: Prompt Factory materialized full task contract",
        "",
        "## Required Read Order",
        "",
        "1. `CLAUDE.md`",
        "2. `templates/_prompt_rules.md`",
        "3. `<iteration>/prompt.md`",
        "4. `<iteration>/tasks.md`",
        "5. `<this task prompt>`",
        "",
        "## Goal",
        "",
        prompt_body,
        "",
        "## Non-Goals",
        "",
        "- Do not edit files outside the allowed set.",
        "- Do not weaken project invariants or tests to make the task pass.",
        "- Do not infer missing product/security requirements from the short draft.",
        "",
        "## Allowed Files",
        "",
    ]
    lines.extend(f"- {path}" for path in task.allowed_files)
    lines.extend(
        [
            "",
            "## Forbidden Files and Symbols",
            "",
            "- Anything not listed in Allowed Files.",
            "- `tasks.md` status edits unless the operator explicitly approves.",
            "- Package/tooling edits unless this is a declared tooling task.",
            "",
            "## Dependencies",
            "",
            ", ".join(task.dependencies) if task.dependencies else "none",
            "",
            "## Applicable Invariants",
            "",
            "| Invariant | Applies? | Required evidence |",
            "|---|---|---|",
            *_render_invariant_rows(project_invariants, review=False),
            "",
            "## Acceptance Matrix",
            "",
            "| Requirement | Evidence | Blocking? |",
            "|---|---|---|",
            f"| {_table_cell(prompt_body)} | {_table_cell(acceptance)} | yes |",
            "",
            "## Preserved Behavior or N/A",
            "",
            "- Preserved behavior fixture: `TBD`",
            "- N/A rationale if no fixture applies: `TBD`",
            "",
            "## Required Commands",
            "",
            "```bash",
            acceptance,
            "ruff check <changed python files or .>",
            "```",
            "",
            "## Deviation Trail",
            "",
            "If implementation improves on this prompt, record:",
            "",
            "```text",
            "Deviation: <what changed>",
            "Reason: <why this is better than the prompt>",
            "Evidence: <test/file/review proof>",
            "```",
            "",
            "## Final Action Contract",
            "",
            "Orchestrator mode: leave final status/commit/PR handling to `orch`.",
            "",
            "Manual mode final response must include changed files, evidence, tests, "
            "and any remaining risks.",
            "",
            "## Routing",
        ]
    )
    lines.extend(_routing_lines(task))
    return "\n".join(lines).rstrip() + "\n"


def _render_materialized_task_review(
    draft: PromptFactoryDraft,
    task: PromptFactoryTask,
    *,
    project_invariants: tuple[ProjectInvariant, ...],
) -> str:
    review_body = (
        task.review_summary
        or "STOP: draft provided only a review_path_placeholder "
        f"({task.review_path_placeholder}); operator must replace this section "
        "before runtime use."
    )
    acceptance = (
        task.test_command
        or task.test_note
        or "No task-specific test command was provided."
    )
    lines = [
        f"# Review - {task.id} - {task.title}",
        "",
        "## Review Metadata",
        "",
        f"- Iteration: `{draft.phase_or_iteration_id}`",
        f"- Task: `{task.id}`",
        "- Diff base: `<base sha/ref>`",
        "- Review mode: `<primary | secondary | manual>`",
        f"- Risk category: `{task.risk_category}`",
        "",
        "## Required Read Order",
        "",
        "1. `CLAUDE.md`",
        "2. `templates/_prompt_rules.md`",
        "3. `templates/_review_template.md`",
        "4. `<iteration>/prompt.md`",
        "5. `<iteration>/tasks.md`",
        "6. `<task prompt>`",
        "7. `<this review prompt>`",
        "",
        "## Verdict Output Contract",
        "",
        "The response must end with exactly one of these trailing blocks:",
        "",
        "```text",
        "Verdict: PASS",
        "```",
        "",
        "```text",
        "Verdict: CHANGES REQUIRED",
        "Severity: should-fix",
        "```",
        "",
        "```text",
        "Verdict: CHANGES REQUIRED",
        "Severity: block",
        "```",
        "",
        "```text",
        "Verdict: BLOCKED",
        "```",
        "",
        "No non-empty line may follow the verdict block.",
        "",
        "## Review Criteria",
        "",
        review_body,
        "",
        "## Scope",
    ]
    lines.extend(f"- {path}" for path in task.allowed_files)
    lines.extend(
        [
            "",
            "## Gate 1 - Scope and Structure",
            "",
            "- [ ] Changed files exactly match Allowed Files or documented generated artifacts.",
            "- [ ] No `tasks.md` edits unless approved.",
            "- [ ] No package/tooling edits unless this is a declared tooling sprint.",
            "- [ ] Lint command relevant to changed files is clean.",
            "",
            "## Gate 2 - Requirement Traceability",
            "",
            "| Requirement from task prompt | Evidence checked | Status |",
            "|---|---|---|",
            "| `<requirement>` | `<file:line/test/command>` | `<PASS/FAIL>` |",
            "",
            "## Gate 3 - Project Invariant Closure",
            "",
            "| Invariant | Applies? | Evidence | Status |",
            "|---|---|---|---|",
            *_render_invariant_rows(project_invariants, review=True),
            "",
            "## Gate 4 - Mechanical Diff Checks",
            "",
            "Run all ten `_prompt_rules.md` Rule 5 signatures unless the task "
            "prompt explicitly marks a signature not applicable. Record output "
            "or `clean`/`N/A` for every signature.",
            "",
            "## Gate 5 - Functional Evidence",
            "",
            f"- Required command or note: `{acceptance}`",
            "- Positive path:",
            "- Negative path:",
            "- Access-isolation path:",
            "- Retention/archive path:",
            "- Manual smoke:",
            "",
            "## Findings Format",
            "",
            "- `[CRITICAL | SHOULD_FIX | FUTURE] <summary>`",
            "  - File: `<file:line>`",
            "  - Evidence:",
            "  - Why it matters:",
            "  - Required fix:",
            "",
            "## Gate 6 - Test Quality",
            "",
            "- [ ] Test names match bodies.",
            "- [ ] Tests fail for the intended reason if behavior regresses.",
            "- [ ] Existing tests were not weakened.",
            "- [ ] No broad `try/except/pass`.",
            "- [ ] No fuzzy status-code assertions.",
            "",
            "## Calibration",
            "",
            "- `PASS`: all blocking requirements and applicable invariants pass.",
            "- `CHANGES REQUIRED` + `Severity: should-fix`: concrete "
            "non-blocking issue that may be deferred to QA within budget.",
            "- `CHANGES REQUIRED` + `Severity: block`: concrete bug, missing "
            "requirement, security issue, broken test, or missing evidence.",
            "- `BLOCKED`: the prompt/spec is contradictory, diff base is wrong, "
            "evidence is missing, or an operator decision is required.",
            "",
            "## Routing",
        ]
    )
    lines.extend(_routing_lines(task))
    return "\n".join(lines).rstrip() + "\n"


def _render_invariant_rows(
    project_invariants: tuple[ProjectInvariant, ...],
    *,
    review: bool,
) -> list[str]:
    if not project_invariants:
        if review:
            return [
                "| <Invariant 1 - e.g. access boundary> | "
                "<applies / preserve only / N/A> | "
                "<evidence or N/A reason> | <PASS/FAIL/N/A> |"
            ]
        return [
            "| <Invariant 1 - e.g. access boundary> | "
            "<applies / preserve only / N/A> | "
            "<evidence or N/A reason> |"
        ]

    rows: list[str] = []
    for item in project_invariants:
        if review:
            rows.append(
                f"| {_table_cell(item.name)} | {_table_cell(item.applies)} | "
                f"{_table_cell(item.evidence)} | {_table_cell(item.status)} |"
            )
        else:
            rows.append(
                f"| {_table_cell(item.name)} | {_table_cell(item.applies)} | "
                f"{_table_cell(item.evidence)} |"
            )
    return rows


def _table_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().replace("|", r"\|")


def _routing_lines(task: PromptFactoryTask) -> list[str]:
    routing = resolve_model_routing(
        ModelRoutingDeclaration(
            model_tier=task.model_tier,
            reasoning_effort=task.reasoning_effort,
            risk_category=task.risk_category,
        )
    )
    return [
        f"- risk_category: {routing.risk_category}",
        f"- model_tier: {routing.model_tier}",
        f"- reasoning_effort: {routing.reasoning_effort}",
        f"- floor_model_tier: {routing.floor_model_tier}",
        f"- floor_reasoning_effort: {routing.floor_reasoning_effort}",
        f"- dual_model_required: {str(routing.dual_model_required).lower()}",
    ]


def _task_doc_filename(task: PromptFactoryTask) -> str:
    slug = _slug(task.title)
    stem = task.id.lower()
    if slug:
        stem = f"{stem}-{slug}"
    return f"{stem}.md"


def _safe_artifact_id(value: str) -> str:
    if not _ARTIFACT_ID_RE.match(value):
        raise PromptFactoryValidationError(
            [
                "draft artifact id must match "
                "[A-Za-z0-9][A-Za-z0-9._-]*"
            ]
        )
    return value


def _write_review_gate(
    artifact_dir: Path,
    *,
    draft_id: str,
    decision: PromptFactoryGateDecision,
    prompt_paths: Mapping[str, Path],
) -> Path:
    gate_path = artifact_dir / REVIEW_GATE_FILENAME
    payload = _review_gate_payload(
        artifact_dir,
        draft_id=draft_id,
        decision=decision,
        prompt_paths=prompt_paths,
    )
    gate_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return gate_path


def _review_gate_payload(
    artifact_dir: Path,
    *,
    draft_id: str,
    decision: PromptFactoryGateDecision,
    prompt_paths: Mapping[str, Path],
) -> dict[str, Any]:
    role_payloads: dict[str, dict[str, str | None]] = {}
    decisions = {role.role: role for role in decision.roles}
    for role in REVIEW_ROLES:
        role_decision = decisions[role]
        role_payloads[role] = {
            "state": role_decision.state,
            "verdict": role_decision.verdict,
            "error": role_decision.error,
            "prompt_path": str(prompt_paths[role]),
            "review_path": str(artifact_dir / ROLE_REVIEW_FILENAMES[role]),
        }
    return {
        "draft_id": draft_id,
        "status": decision.status,
        "required_roles": list(REVIEW_ROLES),
        "roles": role_payloads,
    }
