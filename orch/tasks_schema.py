"""Strict parser for iteration tasks.md files.

No tolerant parsing — every
format violation raises TasksMdError with file:line messages. Parser
errors carry an ``auto_fixable`` flag that is populated but unused in v1
(reserved for ``orch validate --fix`` in a later version).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Pattern

from orch.config import (
    CORE_DEFAULTS,
    ConfigError,
    default_project_yaml_path,
    effective_task_schema_policy,
    load_config,
    patterns as config_patterns,
    task_schema_policy,
)
from orch.model_routing import (
    ModelRoutingDeclaration,
    ModelRoutingError,
    parse_model_routing_field,
)

EMDASH = "\u2014"

DEFAULT_TASK_ID_PATTERN = str(CORE_DEFAULTS["patterns"]["task_id"])
DEFAULT_DETAIL_SECTION_PATTERN = str(
    CORE_DEFAULTS["patterns"]["task_detail_heading"]
)

ID_RE = re.compile(DEFAULT_TASK_ID_PATTERN)
KV_LINE_RE = re.compile(r"^\*\*(?P<key>[A-Za-z][A-Za-z ]+):\*\*\s*(?P<val>.*?)\s*$")
H1_RE = re.compile(r"^#\s+(?P<title>.+?)\s*$")
EXEC_PLAN_RE = re.compile(r"^##\s+Execution Plan\s*$")
TASKS_HEADER_RE = re.compile(r"^##\s+Tasks\s*$")
DETAIL_SECTION_RE = re.compile(DEFAULT_DETAIL_SECTION_PATTERN)
PLANNING_REFUSAL_FILENAMES = {
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
}
_BACKTICK_RE = re.compile(r"`([^`]+)`")
# Strip inline arrow comments of any shape (ASCII or Unicode) plus any
# trailing text. Matches optional whitespace, an arrow marker, then the
# rest of the line. Keep this list in sync with ``_ARROW_MARKERS`` below.
_INLINE_COMMENT_RE = re.compile(r"\s*(?:\u2190|\u2192|<--|<-|->).*$")
_ARROW_MARKERS = ("\u2190", "\u2192", "<--", "<-", "->")
DIFF_CAP_OVERRIDE_KEY = "Diff cap override"
TASK_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

DEFAULT_STATUS_VALUES = {
    "WAITING", "IN_PROGRESS", "DONE", "BLOCKED", "NEEDS_HUMAN_MERGE",
}

TaskSchemaPolicy = dict[str, tuple[str, ...]]


def _normalize_task_schema_policy(
    policy: Mapping[str, Any] | None,
) -> TaskSchemaPolicy:
    section: dict[str, Any] = {}
    if policy:
        section = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in policy.items()
        }
    effective = effective_task_schema_policy(section)
    return {
        key: tuple(values)
        for key, values in effective.items()
    }


def _nearest_project_yaml_path(source_path: Path) -> Path | None:
    start = source_path if source_path.is_dir() else source_path.parent
    for parent in (start, *start.parents):
        candidate = default_project_yaml_path(parent)
        if candidate.exists():
            return candidate
    return None


def _default_task_schema_policy(
    source_path: Path | None = None,
) -> TaskSchemaPolicy:
    config_path = (
        _nearest_project_yaml_path(source_path)
        if source_path is not None
        else _nearest_project_yaml_path(Path.cwd())
    )
    if config_path is not None:
        try:
            return _normalize_task_schema_policy(
                task_schema_policy(load_config(config_path))
            )
        except ConfigError:
            return _normalize_task_schema_policy(None)
    return _normalize_task_schema_policy(None)


def _default_task_patterns(source_path: Path | None = None) -> dict[str, Any]:
    config_path = (
        _nearest_project_yaml_path(source_path)
        if source_path is not None
        else _nearest_project_yaml_path(Path.cwd())
    )
    if config_path is not None:
        return dict(config_patterns(load_config(config_path)))
    return dict(CORE_DEFAULTS["patterns"])


@dataclass
class ParseError:
    line: int
    column: int
    rule: str
    message: str
    auto_fixable: bool = False


class TasksMdError(Exception):
    """Raised when tasks.md fails strict validation.

    ``errors`` holds every collected ParseError. The ``__str__`` is a
    newline-joined ``path:line: [rule] message`` list — operator-facing.
    """

    def __init__(self, errors: list[ParseError], path: Path):
        self.errors = list(errors)
        self.path = path
        rendered = "\n".join(
            f"{path}:{e.line}: [{e.rule}] {e.message}" for e in errors
        )
        super().__init__(rendered or f"{path}: unknown parse error")


@dataclass(frozen=True)
class TaskPatterns:
    task_id_re: Pattern[str]
    detail_section_re: Pattern[str]
    task_id_pattern: str
    task_detail_heading_pattern: str


@dataclass
class ExecutionPlan:
    approach: str
    qa: str
    note: str


@dataclass(frozen=True)
class DiffCapOverride:
    max_diff_insertions_hard: int
    approved_by: str
    evidence: str
    scope: str
    line: int


@dataclass(frozen=True)
class ParallelSafetyDeclaration:
    value: bool = False
    reason: str = ""
    conflicts: tuple[str, ...] = ()
    requires_serial_after: tuple[str, ...] = ()
    line: int = 0


@dataclass
class Task:
    id: str
    title: str
    owner: str
    status: str
    depends_on: list[str]
    branch: str
    allowed_files: list[str] = field(default_factory=list)
    test_cmd: str | None = None  # optional per-task test override
    diff_cap_override: DiffCapOverride | None = None
    model_routing: ModelRoutingDeclaration | None = None
    task_kind: str | None = None
    parallel_safe: ParallelSafetyDeclaration = field(
        default_factory=ParallelSafetyDeclaration
    )
    sort_index: tuple[int, int] | None = None
    section_line: int = 0  # 1-based line of the "### <ID> — ..." header

    @property
    def index(self) -> tuple[int, int]:
        """Return (iteration-number, task-number) — used for ordering."""
        if self.sort_index is not None:
            return self.sort_index
        return _task_index_from_id(self.id)


@dataclass
class TaskBoard:
    path: Path
    title: str
    iteration_branch: str
    status: str
    depends_on_header: str
    blocks_header: str
    execution_plan: ExecutionPlan
    tasks: list[Task]
    diff_cap_override: DiffCapOverride | None = None

    def by_id(self, tid: str) -> Task:
        for t in self.tasks:
            if t.id == tid:
                return t
        raise KeyError(tid)

    @property
    def allowed_file_union(self) -> list[str]:
        """Return every task allowed file once, preserving first-seen order."""
        seen: set[str] = set()
        out: list[str] = []
        for task in self.tasks:
            for path in task.allowed_files:
                if path in seen:
                    continue
                seen.add(path)
                out.append(path)
        return out

    def ready_tasks(
        self, current_states: dict[str, str] | None = None
    ) -> list[Task]:
        """Return tasks eligible to run: WAITING with all deps DONE.

        If ``current_states`` is provided, it overrides the in-file
        statuses — the orchestrator always passes live state from
        ``run_state.json``.
        """
        states = {t.id: t.status for t in self.tasks}
        if current_states:
            states.update(current_states)
        ready = []
        for t in self.tasks:
            if states.get(t.id) != "WAITING":
                continue
            if all(states.get(dep) == "DONE" for dep in t.depends_on):
                ready.append(t)
        ready.sort(key=lambda t: t.index)
        return ready


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(
        self,
        path: Path,
        lines: list[str],
        patterns: TaskPatterns,
        task_schema_policy: TaskSchemaPolicy,
    ) -> None:
        self.path = path
        self.lines = lines
        self.patterns = patterns
        self.task_schema_policy = task_schema_policy
        self.errors: list[ParseError] = []
        self.i = 0  # 0-based cursor
        self.diff_cap_override: DiffCapOverride | None = None

    # --- positioning helpers ------------------------------------------------

    @property
    def lineno(self) -> int:
        return self.i + 1

    def cur(self) -> str | None:
        return self.lines[self.i] if self.i < len(self.lines) else None

    def advance(self) -> None:
        self.i += 1

    def skip_blank_and_hr(self) -> None:
        while self.i < len(self.lines):
            s = self.lines[self.i].strip()
            if s == "" or s == "---":
                self.advance()
            else:
                return

    def err(
        self,
        rule: str,
        message: str,
        *,
        line: int | None = None,
        column: int = 0,
        auto_fixable: bool = False,
    ) -> None:
        self.errors.append(
            ParseError(
                line=line if line is not None else self.lineno,
                column=column,
                rule=rule,
                message=message,
                auto_fixable=auto_fixable,
            )
        )

    # --- top-level ----------------------------------------------------------

    def parse(self) -> TaskBoard | None:
        self.skip_blank_and_hr()
        title = self._expect_h1()
        self.skip_blank_and_hr()

        # Optional "## Task Board" subtitle
        if self.cur() is not None and self.cur().startswith("## Task Board"):
            self.advance()
            self.skip_blank_and_hr()

        kv = self._parse_kv_block(
            required=["Status", "Iteration branch", "Depends on", "Blocks"]
        )
        self.skip_blank_and_hr()

        plan = self._parse_execution_plan()
        self.skip_blank_and_hr()

        # Optional "## Dependency Map" and similar narrative — skip until ## Tasks
        self._advance_until(lambda s: TASKS_HEADER_RE.match(s) is not None,
                            rule="tasks_section_missing",
                            message="expected '## Tasks' section")

        tasks = self._parse_tasks_table()
        # Collect per-task detail sections (allowed files etc.)
        details = self._parse_all_detail_sections()

        # Merge details into tasks
        seen_ids = {t.id for t in tasks}
        for tid in seen_ids:
            if tid not in details:
                # find table row line for better error positioning
                row = next((t for t in tasks if t.id == tid), None)
                self.err(
                    "detail_missing",
                    f"missing '### {tid} {EMDASH} ...' detail section "
                    f"with Allowed files block",
                    line=row.section_line if row else self.lineno,
                )
        for (
            tid,
            (
                sec_line,
                allowed,
                test_cmd,
                diff_override,
                model_routing,
                task_kind,
                parallel_safe,
            ),
        ) in details.items():
            if tid not in seen_ids:
                self.err(
                    "detail_unknown",
                    f"detail section for '{tid}' has no matching row in "
                    f"'## Tasks' table",
                    line=sec_line,
                )
                continue
            task = next(t for t in tasks if t.id == tid)
            task.allowed_files = allowed
            task.test_cmd = test_cmd
            task.diff_cap_override = diff_override
            task.model_routing = model_routing
            task.task_kind = task_kind
            task.parallel_safe = parallel_safe
            task.section_line = sec_line

        return TaskBoard(
            path=self.path,
            title=title or "",
            iteration_branch=_extract_backtick_value(
                kv.get("Iteration branch", "")
            ),
            status=kv.get("Status", ""),
            depends_on_header=kv.get("Depends on", ""),
            blocks_header=kv.get("Blocks", ""),
            execution_plan=plan,
            tasks=tasks,
            diff_cap_override=self.diff_cap_override,
        )

    # --- block parsers ------------------------------------------------------

    def _expect_h1(self) -> str | None:
        line = self.cur()
        if line is None:
            self.err("h1_missing", "file must start with '# <title>' heading")
            return None
        m = H1_RE.match(line)
        if not m:
            self.err(
                "h1_missing",
                "first non-blank line must be '# <title>' (H1)",
            )
            return None
        self.advance()
        return m.group("title")

    def _parse_kv_block(self, *, required: list[str]) -> dict[str, str]:
        """Parse consecutive ``**Key:** value`` lines until a blank line or HR.

        Enforces that every required key is present and in the given order.
        Unknown keys produce a warning error with auto_fixable=True.
        """
        kv: dict[str, str] = {}
        order: list[str] = []
        start_line = self.lineno
        while self.i < len(self.lines):
            line = self.lines[self.i]
            stripped = line.strip()
            if stripped == "" or stripped == "---":
                break
            m = KV_LINE_RE.match(line)
            if not m:
                self.err(
                    "kv_malformed",
                    "expected '**Key:** value' line in header block",
                )
                self.advance()
                continue
            key = m.group("key").strip()
            val = m.group("val").strip()
            if key in kv:
                self.err("kv_duplicate", f"duplicate '**{key}:**' field")
            if key == DIFF_CAP_OVERRIDE_KEY:
                if self.diff_cap_override is not None:
                    self.err(
                        "diff_cap_override_duplicate",
                        "duplicate '**Diff cap override:**' in header block",
                    )
                else:
                    self.diff_cap_override = self._parse_diff_cap_override(
                        val,
                        line=self.lineno,
                        scope="iteration",
                    )
            kv[key] = val
            order.append(key)
            self.advance()
        for k in required:
            if k not in kv:
                self.err(
                    "kv_missing",
                    f"missing '**{k}:**' in header block starting at line "
                    f"{start_line}",
                    line=start_line,
                )
        # Order check
        filtered = [k for k in order if k in required]
        if filtered != required[: len(filtered)]:
            self.err(
                "kv_order",
                f"header fields must appear in order {required}, got "
                f"{filtered}",
                line=start_line,
                auto_fixable=True,
            )
        return kv

    def _parse_execution_plan(self) -> ExecutionPlan:
        header_line = self.cur()
        if header_line is None or not EXEC_PLAN_RE.match(header_line):
            self.err(
                "exec_plan_missing",
                "expected '## Execution Plan' section",
            )
            return ExecutionPlan(approach="", qa="", note="")
        self.advance()
        self.skip_blank_and_hr()

        # Expect three bullet lines: - approach: ..., - qa: ..., - note: ...
        fields = {"approach": None, "qa": None, "note": None}
        order: list[str] = []
        bullet_re = re.compile(
            r"^-\s+(?P<key>approach|qa|note)\s*:\s*(?P<val>.*?)\s*$"
        )
        while self.i < len(self.lines):
            line = self.lines[self.i]
            stripped = line.strip()
            if stripped == "" or stripped == "---":
                break
            if stripped.startswith("## "):
                break
            m = bullet_re.match(line)
            if not m:
                self.err(
                    "exec_plan_line",
                    "expected '- approach: ...', '- qa: ...', or '- note: ...'",
                )
                self.advance()
                continue
            key = m.group("key")
            val = m.group("val").strip()
            if fields[key] is not None:
                self.err("exec_plan_dup", f"duplicate '{key}' in Execution Plan")
            fields[key] = val
            order.append(key)
            self.advance()
        for f in ("approach", "qa", "note"):
            if fields[f] is None:
                self.err(
                    "exec_plan_missing_field",
                    f"Execution Plan missing required field '{f}'",
                )
                fields[f] = ""
        return ExecutionPlan(
            approach=fields["approach"] or "",
            qa=fields["qa"] or "",
            note=fields["note"] or "",
        )

    def _advance_until(
        self,
        predicate,
        *,
        rule: str,
        message: str,
    ) -> None:
        while self.i < len(self.lines):
            if predicate(self.lines[self.i]):
                return
            self.advance()
        self.err(rule, message, line=len(self.lines) or 1)

    # --- tasks table --------------------------------------------------------

    _EXPECTED_COLUMNS = [
        "ID", "Title", "Owner", "Status", "Depends on", "Branch",
    ]

    def _parse_tasks_table(self) -> list[Task]:
        header = self.cur()
        if header is None or not TASKS_HEADER_RE.match(header):
            self.err("tasks_header", "expected '## Tasks' section header")
            return []
        self.advance()
        self.skip_blank_and_hr()
        # Header row
        hdr_line = self.cur()
        if hdr_line is None or "|" not in hdr_line:
            self.err("tasks_table_missing", "expected tasks table after '## Tasks'")
            return []
        cols = [c.strip() for c in hdr_line.strip().strip("|").split("|")]
        if cols != self._EXPECTED_COLUMNS:
            self.err(
                "tasks_columns",
                f"tasks table columns must be {self._EXPECTED_COLUMNS}, got {cols}",
            )
        self.advance()
        # Separator row
        sep = self.cur()
        if sep is None or not re.match(r"^\s*\|[\s\-|:]+\|\s*$", sep):
            self.err("tasks_separator", "expected '|---|...|' separator row")
            return []
        self.advance()
        # Data rows
        tasks: list[Task] = []
        seen_ids: set[str] = set()
        while self.i < len(self.lines):
            line = self.lines[self.i]
            stripped = line.strip()
            if stripped == "" or stripped == "---":
                break
            if stripped.startswith("#"):
                break
            if "|" not in stripped:
                break
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) != len(self._EXPECTED_COLUMNS):
                self.err(
                    "task_row_cells",
                    f"task row must have {len(self._EXPECTED_COLUMNS)} cells, "
                    f"got {len(cells)}",
                )
                self.advance()
                continue
            tid, title, owner, status, deps_str, branch = cells
            id_match = self.patterns.task_id_re.match(tid)
            if not id_match:
                self.err(
                    "task_id",
                    f"task id '{tid}' must match configured task id pattern",
                )
            if tid in seen_ids:
                self.err("task_id_dup", f"duplicate task id '{tid}'")
            seen_ids.add(tid)
            if status not in DEFAULT_STATUS_VALUES:
                self.err(
                    "task_status",
                    f"status '{status}' for {tid} not in "
                    f"{sorted(DEFAULT_STATUS_VALUES)}",
                )
            deps = self._parse_depends(deps_str, tid)
            # Strip surrounding backticks from branch cell
            branch_clean = branch.strip("`").strip()
            sort_index: tuple[int, int] | None = None
            if id_match is not None:
                try:
                    sort_index = _task_index_from_match(id_match)
                except ValueError as exc:
                    self.err("pattern_invalid", str(exc))
            tasks.append(
                Task(
                    id=tid, title=title, owner=owner, status=status,
                    depends_on=deps, branch=branch_clean,
                    sort_index=sort_index,
                )
            )
            self.advance()
        return tasks

    def _parse_depends(self, raw: str, tid: str) -> list[str]:
        s = raw.strip().strip("`")
        if s in ("", EMDASH, "-", "none", "None"):
            return []
        # Comma-separated IDs
        parts = [p.strip() for p in s.split(",")]
        out: list[str] = []
        for p in parts:
            if not self.patterns.task_id_re.match(p):
                self.err(
                    "task_depends",
                    f"'Depends on' for {tid} has invalid id '{p}' "
                    f"(expected configured task id pattern or '{EMDASH}')",
                )
                continue
            out.append(p)
        return out

    # --- per-task detail sections ------------------------------------------

    def _parse_all_detail_sections(
        self,
    ) -> dict[
        str,
        tuple[
            int,
            list[str],
            str | None,
            DiffCapOverride | None,
            ModelRoutingDeclaration | None,
            str | None,
            ParallelSafetyDeclaration,
        ],
    ]:
        """Scan remainder for ``### <ID> — ...`` sections and parse allowed
        files blocks + optional test command.

        Returns mapping: id -> (section_line, allowed_files, test_cmd,
        diff_cap_override, model_routing).

        Unknown content between task sections is tolerated (Agent Rules,
        Status Legend, etc. appear in real files).
        """
        out: dict[
            str,
            tuple[
                int,
                list[str],
                str | None,
                DiffCapOverride | None,
                ModelRoutingDeclaration | None,
                str | None,
                ParallelSafetyDeclaration,
            ],
        ] = {}
        while self.i < len(self.lines):
            line = self.lines[self.i]
            m = self.patterns.detail_section_re.match(line)
            if not m:
                self.advance()
                continue
            tid = m.group("id")
            sec_line = self.lineno
            self.advance()
            (
                allowed,
                test_cmd,
                diff_override,
                model_routing,
                task_kind,
                parallel_safe,
            ) = self._parse_task_detail_block(tid, sec_line)
            if tid in out:
                self.err(
                    "detail_dup",
                    f"duplicate detail section for {tid}",
                    line=sec_line,
                )
                continue
            out[tid] = (
                sec_line,
                allowed,
                test_cmd,
                diff_override,
                model_routing,
                task_kind,
                parallel_safe,
            )
        return out

    def _parse_task_detail_block(
        self, tid: str, sec_line: int
    ) -> tuple[
        list[str],
        str | None,
        DiffCapOverride | None,
        ModelRoutingDeclaration | None,
        str | None,
        ParallelSafetyDeclaration,
    ]:
        """Parse a task detail section for allowed files and optional test cmd.

        Returns ``(allowed_files, test_cmd, diff_cap_override, model_routing,
        task_kind, parallel_safe)``.
        """
        allowed: list[str] = []
        test_cmd: str | None = None
        diff_cap_override: DiffCapOverride | None = None
        model_routing: ModelRoutingDeclaration | None = None
        task_kind: str | None = None
        parallel_safe = ParallelSafetyDeclaration()
        marker_seen = False
        while self.i < len(self.lines):
            line = self.lines[self.i]
            if self.patterns.detail_section_re.match(line):
                break  # next task section
            stripped = line.strip()
            if stripped.startswith("**Allowed files:**"):
                marker_seen = True
                self.advance()
                # skip blanks
                while (self.i < len(self.lines)
                       and self.lines[self.i].strip() == ""):
                    self.advance()
                # expect fence
                fence = self.cur()
                if fence is None or not fence.strip().startswith("```"):
                    self.err(
                        "allowed_fence",
                        f"{tid}: expected fenced code block after "
                        f"'**Allowed files:**'",
                    )
                    break
                self.advance()
                while self.i < len(self.lines):
                    f_line = self.lines[self.i]
                    if f_line.strip().startswith("```"):
                        self.advance()
                        break
                    entry = self._clean_allowed_entry(f_line)
                    if entry:
                        self._validate_allowed_path(entry, tid)
                        allowed.append(entry)
                    self.advance()
                continue
            if stripped.startswith("**Test:**"):
                test_cmd = stripped.removeprefix("**Test:**").strip()
                if test_cmd.startswith("`") and test_cmd.endswith("`"):
                    test_cmd = test_cmd[1:-1].strip()
                test_cmd = test_cmd or None
                self.advance()
                continue
            if stripped.startswith("**Diff cap override:**"):
                raw = stripped.removeprefix("**Diff cap override:**").strip()
                if diff_cap_override is not None:
                    self.err(
                        "diff_cap_override_duplicate",
                        f"{tid}: duplicate '**Diff cap override:**'",
                    )
                else:
                    diff_cap_override = self._parse_diff_cap_override(
                        raw,
                        line=self.lineno,
                        scope=tid,
                    )
                self.advance()
                continue
            if stripped.startswith("**Model routing:**"):
                raw = _extract_backtick_value(
                    stripped.removeprefix("**Model routing:**").strip()
                )
                if model_routing is not None:
                    self.err(
                        "model_routing_duplicate",
                        f"{tid}: duplicate '**Model routing:**'",
                    )
                else:
                    try:
                        model_routing = parse_model_routing_field(
                            raw,
                            source_line=self.lineno,
                        )
                    except ModelRoutingError as exc:
                        self.err("model_routing_invalid", f"{tid}: {exc}")
                self.advance()
                continue
            if stripped.startswith("**Task kind:**"):
                raw = _extract_backtick_value(
                    stripped.removeprefix("**Task kind:**").strip()
                )
                if task_kind is not None:
                    self.err(
                        "task_kind_duplicate",
                        f"{tid}: duplicate '**Task kind:**'",
                    )
                elif not TASK_KIND_RE.match(raw):
                    self.err(
                        "task_kind_invalid",
                        f"{tid}: Task kind must be a non-empty "
                        "alphanumeric/dot/dash/underscore value",
                    )
                else:
                    task_kind = raw
                self.advance()
                continue
            if stripped.startswith("**Parallel safe:**"):
                raw = _extract_backtick_value(
                    stripped.removeprefix("**Parallel safe:**").strip()
                )
                if parallel_safe.line:
                    self.err(
                        "parallel_safe_duplicate",
                        f"{tid}: duplicate '**Parallel safe:**'",
                    )
                else:
                    parallel_safe = self._parse_parallel_safe(
                        raw,
                        tid=tid,
                        line=self.lineno,
                    )
                self.advance()
                continue
            self.advance()
        if not marker_seen:
            self.err(
                "allowed_missing",
                f"{tid}: missing '**Allowed files:**' block",
                line=sec_line,
            )
        return (
            allowed,
            test_cmd,
            diff_cap_override,
            model_routing,
            task_kind,
            parallel_safe,
        )

    def _parse_parallel_safe(
        self, raw: str, *, tid: str, line: int
    ) -> ParallelSafetyDeclaration:
        body = raw.strip()
        fallback = ParallelSafetyDeclaration(line=line)
        if not body:
            self.err(
                "parallel_safe_invalid",
                f"{tid}: Parallel safe marker must start with yes or no",
                line=line,
            )
            return fallback

        parts = [part.strip() for part in body.split(";")]
        flag = parts[0].lower()
        if flag not in {"yes", "no"}:
            self.err(
                "parallel_safe_invalid",
                f"{tid}: Parallel safe value must be yes or no",
                line=line,
            )
            return fallback

        fields: dict[str, str] = {}
        ok = True
        for part in parts[1:]:
            if not part:
                continue
            if "=" not in part:
                self.err(
                    "parallel_safe_invalid",
                    f"{tid}: Parallel safe field '{part}' must use key=value",
                    line=line,
                )
                ok = False
                continue
            key, value = [piece.strip() for piece in part.split("=", 1)]
            if key in fields:
                self.err(
                    "parallel_safe_invalid",
                    f"{tid}: duplicate Parallel safe field '{key}'",
                    line=line,
                )
                ok = False
            fields[key] = value

        required = {"reason", "conflicts"}
        missing = sorted(key for key in required if not fields.get(key))
        if missing:
            self.err(
                "parallel_safe_invalid",
                f"{tid}: Parallel safe missing required field(s): {missing}",
                line=line,
            )
            ok = False
        allowed = {"reason", "conflicts", "requires_serial_after"}
        unknown = sorted(set(fields) - allowed)
        if unknown:
            self.err(
                "parallel_safe_invalid",
                f"{tid}: Parallel safe has unknown field(s): {unknown}",
                line=line,
            )
            ok = False

        reason = fields.get("reason", "").strip()
        if not reason:
            self.err(
                "parallel_safe_invalid",
                f"{tid}: Parallel safe reason must not be empty",
                line=line,
            )
            ok = False

        conflicts = tuple(_parse_parallel_list(fields.get("conflicts", "")))
        requires_serial_after = tuple(
            _parse_parallel_list(fields.get("requires_serial_after", ""))
        )
        for dep in requires_serial_after:
            if not self.patterns.task_id_re.match(dep):
                self.err(
                    "parallel_safe_invalid",
                    f"{tid}: requires_serial_after entry '{dep}' must match "
                    "configured task id pattern",
                    line=line,
                )
                ok = False
        if not ok:
            return fallback
        return ParallelSafetyDeclaration(
            value=flag == "yes",
            reason=reason,
            conflicts=conflicts,
            requires_serial_after=requires_serial_after,
            line=line,
        )

    def _parse_diff_cap_override(
        self, raw: str, *, line: int, scope: str
    ) -> DiffCapOverride | None:
        body = _extract_backtick_value(raw).strip()
        if not body:
            self.err(
                "diff_cap_override_empty",
                "Diff cap override must include "
                "`max_diff_insertions_hard=<int>; approved_by=<identity>; "
                "evidence=<relative path or note:...>`",
                line=line,
            )
            return None

        fields: dict[str, str] = {}
        ok = True
        for part in body.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                self.err(
                    "diff_cap_override_malformed",
                    f"Diff cap override field '{part}' must use key=value",
                    line=line,
                )
                ok = False
                continue
            key, value = [piece.strip() for piece in part.split("=", 1)]
            if key in fields:
                self.err(
                    "diff_cap_override_duplicate_field",
                    f"duplicate Diff cap override field '{key}'",
                    line=line,
                )
                ok = False
            fields[key] = value

        required = {"max_diff_insertions_hard", "approved_by", "evidence"}
        missing = sorted(k for k in required if not fields.get(k))
        if missing:
            self.err(
                "diff_cap_override_missing_field",
                f"Diff cap override missing required field(s): {missing}",
                line=line,
            )
            ok = False
        unknown = sorted(set(fields) - required)
        if unknown:
            self.err(
                "diff_cap_override_unknown_field",
                f"Diff cap override has unknown field(s): {unknown}",
                line=line,
            )
            ok = False

        cap_raw = fields.get("max_diff_insertions_hard", "")
        if cap_raw and (not cap_raw.isdecimal() or int(cap_raw) <= 0):
            self.err(
                "diff_cap_override_invalid_numeric",
                "max_diff_insertions_hard must be a positive integer",
                line=line,
            )
            ok = False

        evidence = fields.get("evidence", "")
        if evidence:
            ok = self._validate_diff_cap_evidence(evidence, line=line) and ok

        if not ok:
            return None
        return DiffCapOverride(
            max_diff_insertions_hard=int(cap_raw),
            approved_by=fields["approved_by"],
            evidence=evidence,
            scope=scope,
            line=line,
        )

    def _validate_diff_cap_evidence(self, evidence: str, *, line: int) -> bool:
        if evidence.startswith("note:"):
            if evidence.removeprefix("note:").strip():
                return True
            self.err(
                "diff_cap_override_invalid_evidence",
                "Diff cap override evidence note must not be empty",
                line=line,
            )
            return False

        path = Path(evidence)
        if path.is_absolute() or any(part == ".." for part in path.parts):
            self.err(
                "diff_cap_override_invalid_evidence",
                "Diff cap override evidence path must be relative and must "
                "not contain '..'",
                line=line,
            )
            return False
        if any(ch in evidence for ch in "*?[]"):
            self.err(
                "diff_cap_override_invalid_evidence",
                "Diff cap override evidence path must not contain glob chars",
                line=line,
            )
            return False
        return True

    @staticmethod
    def _clean_allowed_entry(raw: str) -> str:
        # Strip backticks, inline arrow comments (ASCII "<-"/"->", Unicode
        # "\u2190"/"\u2192"), and whitespace. Comment text after the marker
        # is discarded; final value must be a clean relative file path.
        s = raw.strip()
        if not s:
            return ""
        s = _INLINE_COMMENT_RE.sub("", s)
        s = s.strip().strip("`").strip()
        return s

    def _validate_allowed_path(self, path: str, tid: str) -> None:
        for marker in _ARROW_MARKERS:
            if marker in path:
                self.err(
                    "allowed_comment_not_stripped",
                    f"{tid}: allowed file '{path}' contains arrow marker "
                    f"'{marker}'; remove inline comments from the allowed "
                    f"files block",
                )
        if path.startswith("/"):
            self.err(
                "allowed_absolute",
                f"{tid}: allowed file '{path}' must be relative, not absolute",
            )
        if any(ch in path for ch in "*?[]"):
            self.err(
                "allowed_glob",
                f"{tid}: allowed file '{path}' must not contain glob chars",
            )
        for prefix in self.task_schema_policy["forbidden_allowed_prefixes"]:
            if path.startswith(prefix):
                self.err(
                    "allowed_forbidden_prefix",
                    f"{tid}: allowed file '{path}' is under a forbidden "
                    f"prefix '{prefix}'; extend scope explicitly with the "
                    "logged scope-extension carve-out before touching "
                    "protected roots",
                )
        if Path(path).name == "tasks.md":
            self.err(
                "allowed_tasks_md",
                f"{tid}: allowed files must not include 'tasks.md' "
                f"(orchestrator writes status updates)",
            )


def _extract_backtick_value(raw: str) -> str:
    """Extract content from backtick-wrapped value, ignoring trailing comments.

    ``\\`demo/iteration-1\\` <- some note`` → ``demo/iteration-1``
    ``demo/iteration-1``                   → ``demo/iteration-1``
    """
    m = _BACKTICK_RE.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip().strip("`").strip()


def _parse_parallel_list(raw: str) -> list[str]:
    value = raw.strip()
    if value in {"", "-", EMDASH, "none", "None"}:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_relative_task_path(path: str) -> str:
    """Return a normalized repo-relative task path for policy checks."""
    cleaned = path.strip().replace("\\", "/").strip("/")
    parts = [part for part in cleaned.split("/") if part not in {"", "."}]
    return "/".join(parts)


def planning_path_refusal_reason(
    path: str,
    *,
    task_schema_policy: Mapping[str, Any] | None = None,
) -> str | None:
    """Return why ``path`` cannot be written by planning-team mode."""
    policy = (
        _default_task_schema_policy()
        if task_schema_policy is None
        else _normalize_task_schema_policy(task_schema_policy)
    )
    normalized = normalize_relative_task_path(path)
    if not normalized:
        return "empty path"
    if Path(path).is_absolute() or any(part == ".." for part in Path(path).parts):
        return "path must be repo-relative and must not contain '..'"
    if normalized in PLANNING_REFUSAL_FILENAMES:
        return "deployment surface is forbidden"
    for prefix in policy["planning_refusal_prefixes"]:
        if normalized == prefix.rstrip("/") or normalized.startswith(prefix):
            return f"path is under forbidden planning prefix '{prefix}'"
    allowed_prefixes = policy["planning_allowed_prefixes"]
    if not normalized.startswith(allowed_prefixes):
        allowed = ", ".join(allowed_prefixes)
        return f"path must be under one of: {allowed}"
    return None


def is_planning_artifact_path(
    path: str,
    *,
    task_schema_policy: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when ``path`` is allowed for planning-team writes."""
    return planning_path_refusal_reason(
        path,
        task_schema_policy=task_schema_policy,
    ) is None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _compile_task_patterns(patterns: Mapping[str, Any] | None) -> TaskPatterns:
    configured = patterns or {}
    task_id_pattern = configured.get("task_id", DEFAULT_TASK_ID_PATTERN)
    detail_pattern = configured.get(
        "task_detail_heading", DEFAULT_DETAIL_SECTION_PATTERN
    )
    if not isinstance(task_id_pattern, str) or not task_id_pattern:
        raise ValueError("patterns.task_id must be a non-empty regex string")
    if not isinstance(detail_pattern, str) or not detail_pattern:
        raise ValueError(
            "patterns.task_detail_heading must be a non-empty regex string"
        )
    try:
        task_id_re = re.compile(task_id_pattern)
        detail_section_re = re.compile(detail_pattern)
    except re.error as exc:
        raise ValueError(f"invalid task pattern regex: {exc}") from None

    if not task_id_re.groups:
        raise ValueError(
            "patterns.task_id must expose numeric iteration/task ordering groups"
        )
    if "id" not in detail_section_re.groupindex:
        raise ValueError("patterns.task_detail_heading must define group 'id'")
    return TaskPatterns(
        task_id_re=task_id_re,
        detail_section_re=detail_section_re,
        task_id_pattern=task_id_pattern,
        task_detail_heading_pattern=detail_pattern,
    )


def _task_index_from_match(match: re.Match[str]) -> tuple[int, int]:
    group_names = match.re.groupindex
    if {"iteration", "task"} <= set(group_names):
        return int(match.group("iteration")), int(match.group("task"))

    numeric_groups: list[int] = []
    for group in match.groups():
        if group is not None and group.isdecimal():
            numeric_groups.append(int(group))
    if len(numeric_groups) >= 2:
        return numeric_groups[0], numeric_groups[1]
    if len(numeric_groups) == 1:
        return 0, numeric_groups[0]
    raise ValueError(
        "patterns.task_id must expose numeric iteration/task ordering groups"
    )


def _task_index_from_id(task_id: str) -> tuple[int, int]:
    numeric_parts = [
        int(part) for part in re.findall(r"\d+", task_id)
    ]
    if len(numeric_parts) >= 2:
        return numeric_parts[-2], numeric_parts[-1]
    if len(numeric_parts) == 1:
        return 0, numeric_parts[0]
    raise ValueError(
        "task id must contain numeric iteration/task ordering components"
    )


def parse_tasks_md(
    path: Path,
    *,
    patterns: Mapping[str, Any] | None = None,
    task_schema_policy: Mapping[str, Any] | None = None,
) -> TaskBoard:
    """Parse and validate an iteration's tasks.md.

    Raises TasksMdError with a collected list of ParseError objects if any
    validation rule fails.
    """
    if not path.exists():
        raise TasksMdError(
            [ParseError(line=0, column=0, rule="file_missing",
                        message=f"tasks.md not found: {path}")],
            path,
        )
    text = path.read_text()
    lines = text.splitlines()
    try:
        task_patterns = _compile_task_patterns(
            _default_task_patterns(path) if patterns is None else patterns
        )
    except (ConfigError, ValueError) as exc:
        raise TasksMdError(
            [ParseError(
                line=0,
                column=0,
                rule="pattern_invalid",
                message=str(exc),
            )],
            path,
        ) from None

    policy = (
        _default_task_schema_policy(path)
        if task_schema_policy is None
        else _normalize_task_schema_policy(task_schema_policy)
    )
    parser = _Parser(path, lines, task_patterns, policy)
    board = parser.parse()
    # DAG check runs even if earlier errors were collected, so the caller
    # sees a complete diagnostic list rather than one error at a time.
    if board is not None and board.tasks:
        _check_dag(board, parser)
    if parser.errors:
        raise TasksMdError(parser.errors, path)
    assert board is not None
    return board


def _check_dag(board: TaskBoard, parser: _Parser) -> None:
    ids = {t.id for t in board.tasks}
    for t in board.tasks:
        for dep in t.depends_on:
            if dep not in ids:
                parser.err(
                    "depends_unknown",
                    f"{t.id} depends on unknown task '{dep}'",
                    line=t.section_line or 0,
                )
        for dep in t.parallel_safe.requires_serial_after:
            if dep not in ids:
                parser.err(
                    "parallel_safe_invalid",
                    f"{t.id} requires_serial_after unknown task '{dep}'",
                    line=t.parallel_safe.line or t.section_line or 0,
                )
            if dep == t.id:
                parser.err(
                    "parallel_safe_invalid",
                    f"{t.id} requires_serial_after must not reference itself",
                    line=t.parallel_safe.line or t.section_line or 0,
                )
    # Cycle detection (Kahn's algorithm)
    in_degree = {t.id: 0 for t in board.tasks}
    adj: dict[str, list[str]] = {t.id: [] for t in board.tasks}
    for t in board.tasks:
        for dep in t.depends_on:
            if dep in in_degree:
                adj[dep].append(t.id)
                in_degree[t.id] += 1
    queue = [tid for tid, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        n = queue.pop()
        visited += 1
        for m in adj[n]:
            in_degree[m] -= 1
            if in_degree[m] == 0:
                queue.append(m)
    if visited != len(board.tasks):
        cyclic = [tid for tid, d in in_degree.items() if d > 0]
        parser.err(
            "dag_cycle",
            f"cycle detected in task dependencies: {sorted(cyclic)}",
            line=0,
        )
