"""Command-line entry point for the orchestrator.

Subcommands:
    validate   Validate project.yaml and an iteration's tasks.md
    status     Print a short run-state summary
    report     Render the readiness report for an iteration
    run        Walk the DAG and execute the per-task state machine
    resume     Like run, but requires existing run_state.json
    revert     Revert the merge commit recorded for a task
    cleanup    Delete local task branches for DONE tasks
    qa         Run 5 parallel QA reviewers on a completed iteration
    retro      Run 3 parallel retrospective perspectives
    iteration  Chain run -> qa -> retro sequentially
    prompt-factory
               Validate, render, review, approve, or materialize deterministic
               Prompt Factory drafts
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from orch.agents import build_adapter
from orch.config import (
    ConfigError,
    default_project_yaml_path,
    load_config,
    timeouts,
)
from orch.cost import CostLogger
from orch.doctor import (
    render_json as render_doctor_json,
    render_report as render_doctor_report,
    run_doctor,
)
from orch.git_ops import (
    BranchFreshnessCondition,
    classify_branch_freshness,
    cleanup_orch_workdir,
    ensure_orch_workdir,
    orch_workdir,
    preferred_remote_ref,
    render_branch_freshness_recovery,
    WorktreePreflightError,
)
from orch.hooks import build_hook_dispatcher
from orch.improvements import (
    IMPROVEMENTS_FILENAME,
    ImprovementValidationError,
    encode_record,
    improvements_path,
    read_records,
)
from orch.locks import RunLockError, RunStateLock
from orch.paths import OrchPaths, resolve_orch_paths
from orch.prompt_factory import (
    PromptFactoryApprovalError,
    PromptFactoryMaterializationError,
    PromptFactoryReviewError,
    PromptFactoryValidationError,
    REVIEW_GATE_FILENAME,
    check_prompt_factory_approval,
    load_draft_json,
    load_operator_approval_json,
    load_review_gate_json,
    materialize_prompt_factory_draft,
    render_tasks_md,
    validate_draft,
    write_review_gate_status,
    write_review_package,
)
from orch.providers import CommandProvider, SubprocessCommandProvider
from orch.qa import (
    QaDiffBaseError,
    QaEmptyDiffError,
    run_qa,
    run_qa_team_mode,
)
from orch.retro import run_retro, run_retro_team_mode
from orch.lifecycle import (
    PhaseResolutionError,
    RecoverApplyError,
    build_recovery_plan,
    cleanup_task_branches,
    cleanup_recover_workdir,
    head_sha_guard,
    remove_recoverable_lock,
    reset_in_progress_tasks_for_recovery,
    resolve_phase_branch,
    revert_task_merge,
    salvage_dirty_recover_workdir,
)
from orch.report import build_report
from orch.runner import (
    IterationRunner,
    RunnerDeps,
    RunnerError,
    RunOptions,
)
from orch.state import (
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_NEEDS_HUMAN_MERGE,
    STATUS_STOPPED_PREFIX,
    StateStore,
)
from orch.task_flow import depends_transitively_on
from orch.tasks_schema import TasksMdError, parse_tasks_md
from orch.watch import render_recent_events

_DEFAULT_COMMAND_PROVIDER = SubprocessCommandProvider()


class IterationLookupError(ValueError):
    pass


def _repo_root() -> Path:
    return Path.cwd()


def _load_config_and_paths(repo: Path):
    cfg = load_config(default_project_yaml_path(repo))
    return cfg, resolve_orch_paths(repo, cfg)


def _iter_dir(iteration: str, repo: Path, orch_paths: OrchPaths) -> Path | None:
    root = orch_paths.iteration_root
    if not root.exists():
        return None
    matches: list[Path] = []
    direct = root / iteration
    if direct.is_dir():
        matches.append(direct)
    for p in root.rglob(iteration):
        if p.is_dir():
            matches.append(p)
    unique = sorted(set(matches))
    if len(unique) > 1:
        rel = ", ".join(str(p.relative_to(root)) for p in unique)
        raise IterationLookupError(
            f"iteration '{iteration}' is ambiguous under {root}: {rel}"
        )
    return unique[0] if unique else None


def _log_dir(
    repo: Path,
    iteration: str,
    orch_paths: OrchPaths | None = None,
) -> Path:
    if orch_paths is None:
        _, orch_paths = _load_config_and_paths(repo)
    return orch_paths.iteration_log_dir(iteration)


def _acquire_iteration_lock(
    *,
    repo: Path,
    iteration: str,
    iter_branch: str,
    command: str,
    orch_paths: OrchPaths | None = None,
) -> RunStateLock | None:
    if orch_paths is None:
        try:
            _, orch_paths = _load_config_and_paths(repo)
        except ConfigError as exc:
            print(f"project.yaml error: {exc}", file=sys.stderr)
            return None
    lock = RunStateLock(
        log_dir=_log_dir(repo, iteration, orch_paths),
        iteration=iteration,
        iter_branch=iter_branch,
        command=command,
        repo_root=repo,
        orch_workdir=orch_workdir(
            repo,
            iteration,
            worktree_root=orch_paths.worktree_root,
        ),
    )
    try:
        return lock.acquire()
    except RunLockError as exc:
        print(str(exc), file=sys.stderr)
        return None


def _release_runner_lock(runner: IterationRunner) -> None:
    lock = runner.deps.run_lock
    if lock is not None:
        lock.release()


def _resolve_iter(iteration: str, repo: Path, orch_paths: OrchPaths) -> Path | None:
    try:
        d = _iter_dir(iteration, repo, orch_paths)
    except IterationLookupError as exc:
        print(str(exc), file=sys.stderr)
        return None
    if d is None:
        print(
            f"iteration '{iteration}' not found under {orch_paths.iteration_root}",
            file=sys.stderr,
        )
    return d


def _load(repo: Path, iteration: str):
    """Load config + board + state store. Returns (cfg, board, store, log_dir)
    or prints to stderr and returns None on failure."""
    try:
        cfg, orch_paths = _load_config_and_paths(repo)
    except ConfigError as exc:
        print(f"project.yaml error: {exc}", file=sys.stderr)
        return None
    iter_dir = _resolve_iter(iteration, repo, orch_paths)
    if iter_dir is None:
        return None
    try:
        board = parse_tasks_md(
            orch_paths.task_board_path(iter_dir),
            patterns=cfg.data.get("patterns", {}),
        )
    except TasksMdError as exc:
        print(str(exc), file=sys.stderr)
        return None
    log_dir = orch_paths.iteration_log_dir(iteration)
    hook_dispatcher = build_hook_dispatcher(
        cfg.data.get("hooks", {}), repo_root=repo
    )
    store = StateStore(
        log_dir=log_dir,
        iteration=iteration,
        iter_branch=board.iteration_branch,
        hook_dispatcher=hook_dispatcher,
    )
    if store.exists():
        store.load()
    return cfg, board, store, log_dir


def _check_retro_gate(
    board, repo: Path, orch_paths: OrchPaths
) -> tuple[bool, str]:
    """Return (ok, error_message) for the carried-forward retro gate."""
    depends_on = (board.depends_on_header or "").strip()
    if not depends_on or depends_on.lower() in {"none", "—", "-"}:
        return True, ""

    # Iterate ALL matches and pick the LAST one whose retro exists on disk.
    # This avoids picking a broad historical mention before the actually
    # relevant dependency token.
    iter_matches = list(
        re.finditer(
            r"(?i)\b([a-z0-9][a-z0-9_.-]*-i\d+|i\d+)\b",
            depends_on,
        )
    )
    if not iter_matches:
        return True, ""

    prev_iter: str | None = None
    retro_path = None
    for m in iter_matches:
        candidate = m.group(1).lower()
        candidate_retro = (
            orch_paths.iteration_log_dir(candidate) / "retrospective.md"
        )
        if candidate_retro.exists():
            prev_iter = candidate
            retro_path = candidate_retro
    if prev_iter is None or retro_path is None:
        return True, ""

    prompt_path = orch_paths.prompt_path(board.path.parent)
    if not prompt_path.exists():
        return (
            False,
            f"prompt.md not found at {prompt_path}; cannot verify retro gate",
        )

    prompt_text = prompt_path.read_text()
    # Accept trailing parenthetical / suffix text after the heading
    # (e.g. "(Rule 7 / orch-validate gate)"). The \b word-boundary
    # preserves rejection of typo'd headings like
    # "carried-forward action items_garbage".
    heading_re = re.compile(
        r"^##\s+carried.forward action items\b",
        re.IGNORECASE | re.MULTILINE,
    )
    heading_match = heading_re.search(prompt_text)
    if heading_match is None:
        return (
            False,
            "validate failed: prior iteration "
            f"'{prev_iter}' has a retrospective ({retro_path}) but this "
            "iteration's prompt.md lacks a '## Carried-forward action items' "
            "section.\n"
            "Add the section with either:\n"
            f"  - '(none — all closed in iteration {prev_iter})'\n"
            "  - A bulleted list of carried items\n"
            f"See: {retro_path}",
        )

    section_body = prompt_text[heading_match.end():]
    next_heading = re.search(r"^##\s+", section_body, re.MULTILINE)
    if next_heading is not None:
        section_body = section_body[: next_heading.start()]

    has_none = bool(re.search(r"\(none", section_body, re.IGNORECASE))
    has_bullet = bool(re.search(r"^\s*-\s+\S", section_body, re.MULTILINE))
    if not has_none and not has_bullet:
        return (
            False,
            "validate failed: '## Carried-forward action items' section in "
            "prompt.md is empty — add '(none — all closed)' or enumerate "
            f"items.\nSee: {retro_path}",
        )

    return True, ""


# ---------------------------------------------------------------------------
# validate / status / report
# ---------------------------------------------------------------------------


def _resolve_phase_branch(cfg, iteration: str) -> str | None:
    """Resolve the phase branch name for an iteration via project.yaml."""
    return resolve_phase_branch(cfg, iteration)


def _check_iteration_branch_freshness(
    iter_branch: str, phase_branch: str, repo: Path
) -> tuple[bool, str]:
    """Pre-flight detection of a stale local iteration branch.

    Returns (ok, error_message). If the iteration branch does not exist
    locally, validate proceeds silently (branch will be created during
    `run`). If it exists, it must contain the phase HEAD. Equal heads
    and iteration work on top of phase are fresh. Branches missing phase
    commits or diverged from phase fail with distinct errors.
    """
    phase_ref = preferred_remote_ref(repo, phase_branch)
    freshness = classify_branch_freshness(
        repo, branch=iter_branch, base_ref=phase_ref
    )
    if freshness.condition == BranchFreshnessCondition.MISSING_BRANCH:
        return True, ""  # branch absent — orch run will create it
    if freshness.condition == BranchFreshnessCondition.MISSING_BASE:
        # No phase branch locally — can't compare; let validate continue.
        return True, ""
    if freshness.contains_base:
        return True, ""
    return (
        False,
        "validate failed: "
        + render_branch_freshness_recovery(freshness, gate="validate"),
    )


def _warn_if_agent_worktree(repo: Path) -> None:
    """Warn when validate runs from a configured agent worktree.

    Generated artifacts are often gitignored, so retrospective reports may not
    be present inside an agent worktree. Emit a warning so the operator knows
    the retro-gate signal may be incomplete from this location.
    """
    parts = repo.resolve().parts
    if ".claude" in parts:
        idx = parts.index(".claude")
        if idx + 1 < len(parts) and parts[idx + 1] == "worktrees":
            print(
                "WARNING: running orch validate from an agent worktree — the "
                "retro-gate may give a false-positive OK because tools/logs/ "
                "retros aren't checked out here. Run from the main worktree "
                "before declaring scaffold green.",
                file=sys.stderr,
            )


def _run_scaffold_lint(
    repo: Path,
    iter_dir: Path,
    *,
    command_provider: CommandProvider | None = None,
) -> int:
    """Invoke the packaged scaffold linter as a subprocess.

    Returns the lint exit code. On non-zero, the caller should print the
    captured output to stderr and short-circuit validate before the
    retro-gate runs (so a scaffold mistake doesn't get masked by a
    retro-gate pass).
    """
    result = (command_provider or _DEFAULT_COMMAND_PROVIDER).run(
        [sys.executable, "-m", "orch.scaffold_lint", str(iter_dir)],
        cwd=repo,
    )
    if result.returncode != 0:
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        print(
            "validate failed: scaffold_lint did not pass; fix above issues "
            "first.",
            file=sys.stderr,
        )
    return result.returncode


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate tasks/prompt scaffolding from the main worktree only."""
    repo = _repo_root()
    _warn_if_agent_worktree(repo)

    try:
        _, orch_paths = _load_config_and_paths(repo)
    except ConfigError as exc:
        print(f"project.yaml error: {exc}", file=sys.stderr)
        return 1

    iter_dir = _resolve_iter(args.iteration, repo, orch_paths)
    if iter_dir is None:
        return 1
    rc = _run_scaffold_lint(repo, iter_dir)
    if rc != 0:
        return 1

    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, _, _ = loaded
    orch_paths = resolve_orch_paths(repo, cfg)

    try:
        phase_branch = _resolve_phase_branch(cfg, args.iteration)
    except PhaseResolutionError as exc:
        print(f"validate failed: {exc}", file=sys.stderr)
        return 1
    if phase_branch and board.iteration_branch:
        ok, err = _check_iteration_branch_freshness(
            board.iteration_branch, phase_branch, repo
        )
        if not ok:
            print(err, file=sys.stderr)
            return 1

    ok, err = _check_retro_gate(board, repo, orch_paths)
    if not ok:
        print(err, file=sys.stderr)
        return 1
    print(
        f"OK: {board.path} ({len(board.tasks)} tasks, "
        f"iter branch '{board.iteration_branch}')"
    )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    repo = _repo_root()
    report = run_doctor(
        repo,
        command_provider=getattr(args, "command_provider", None),
    )
    if args.json:
        print(render_doctor_json(report), end="")
    else:
        print(render_doctor_report(report), end="")
    return report.exit_code


def cmd_status(args: argparse.Namespace) -> int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    _, board, store, _ = loaded
    _print_status(store, board_task_count=len(board.tasks))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    repo = _repo_root()
    try:
        _, orch_paths = _load_config_and_paths(repo)
    except ConfigError as exc:
        print(f"project.yaml error: {exc}", file=sys.stderr)
        return 1
    if _resolve_iter(args.iteration, repo, orch_paths) is None:
        return 1
    print(
        render_recent_events(
            orch_paths.iteration_log_dir(args.iteration),
            limit=args.limit,
        ),
        end="",
    )
    return 0


def _print_status(store, *, board_task_count: int) -> None:
    snap = store.snapshot
    tasks = store.tasks
    buckets = {
        "DONE": 0, "IN_PROGRESS": 0, "STOPPED": 0, "NEEDS_HUMAN_MERGE": 0,
        "BLOCKED_UPSTREAM": 0, "OTHER": 0,
    }
    for t in tasks.values():
        if t.status == STATUS_DONE:
            buckets["DONE"] += 1
        elif t.status.startswith(STATUS_STOPPED_PREFIX):
            buckets["STOPPED"] += 1
        elif t.status == STATUS_NEEDS_HUMAN_MERGE:
            buckets["NEEDS_HUMAN_MERGE"] += 1
        elif t.status == STATUS_BLOCKED_UPSTREAM:
            buckets["BLOCKED_UPSTREAM"] += 1
        elif t.status == "IN_PROGRESS":
            buckets["IN_PROGRESS"] += 1
        else:
            buckets["OTHER"] += 1

    print(f"iteration:   {snap.iteration}")
    print(f"branch:      {snap.iter_branch}")
    print(f"started:     {snap.started_at or '-'}")
    print(f"finished:    {snap.finished_at or '-'}")
    print(f"tasks on board: {board_task_count}")
    print(f"tracked:     {len(tasks)}")
    for k, v in buckets.items():
        if v or k in ("DONE", "IN_PROGRESS"):
            print(f"  {k:<18} {v}")


def cmd_report(args: argparse.Namespace) -> int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, _, store, log_dir = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    text = build_report(
        state=store,
        cost_jsonl=log_dir / "cost.jsonl",
        high_risk_globs=list(cfg.data.get("risk", {}).get("high_risk_globs") or []),
        artifact_root_ref=orch_paths.artifact_root_ref,
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"wrote {out_path}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_improvements(args: argparse.Namespace) -> int:
    repo = _repo_root()
    try:
        _, orch_paths = _load_config_and_paths(repo)
    except ConfigError as exc:
        print(f"project.yaml error: {exc}", file=sys.stderr)
        return 1
    artifact = improvements_path(orch_paths.iteration_log_dir(args.iteration))
    artifact_ref = orch_paths.artifact_ref(args.iteration, IMPROVEMENTS_FILENAME)
    try:
        records = read_records(artifact)
    except ImprovementValidationError as exc:
        print(f"improvements validation failed: {exc}", file=sys.stderr)
        return 1

    if args.improvements_verb == "list":
        for record in records:
            print(encode_record(record))
        return 0
    if args.improvements_verb == "validate":
        print(f"OK: {len(records)} improvement record(s) at {artifact_ref}")
        return 0

    print(
        f"error: unknown improvements verb {args.improvements_verb!r}",
        file=sys.stderr,
    )
    return 2


def cmd_prompt_factory(args: argparse.Namespace) -> int:
    repo = _repo_root()
    try:
        if args.prompt_factory_verb == "validate":
            raw = load_draft_json(Path(args.draft_json))
            draft = validate_draft(raw)
            print(
                f"OK: prompt factory draft has {len(draft.tasks)} task(s); "
                "rendered tasks.md preview passes tasks_schema.py"
            )
            return 0
        if args.prompt_factory_verb == "render":
            raw = load_draft_json(Path(args.draft_json))
            sys.stdout.write(render_tasks_md(raw))
            return 0
        if args.prompt_factory_verb == "review-package":
            _, orch_paths = _load_config_and_paths(repo)
            raw = load_draft_json(Path(args.draft_json))
            package = write_review_package(
                raw,
                log_root=orch_paths.artifact_root,
                prompt_factory_log_dirname=orch_paths.prompt_factory_log_dirname,
            )
            print(f"wrote {package.artifact_dir}")
            print(f"gate status: {package.decision.status}")
            for role, prompt_path in package.prompt_paths.items():
                print(f"{role}: {prompt_path}")
            print(f"review gate: {package.gate_path}")
            return 0
        if args.prompt_factory_verb == "review-status":
            _, orch_paths = _load_config_and_paths(repo)
            decision = write_review_gate_status(
                log_root=orch_paths.artifact_root,
                draft_id=args.draft_id,
                prompt_factory_log_dirname=orch_paths.prompt_factory_log_dirname,
            )
            artifact_dir = orch_paths.prompt_factory_artifact_dir(args.draft_id)
            print(f"{args.draft_id}: {decision.status}")
            for role in decision.roles:
                status = role.verdict or role.state
                print(f"{role.role}: {status}")
            print(f"review gate: {artifact_dir / REVIEW_GATE_FILENAME}")
            return 0
        if args.prompt_factory_verb == "approve-check":
            approval = check_prompt_factory_approval(
                load_draft_json(Path(args.draft_json)),
                load_review_gate_json(Path(args.review_gate_json)),
                load_operator_approval_json(Path(args.approval_json)),
            )
            print(
                "OK: Prompt Factory approval gate passed for "
                f"{approval.draft_id} "
                f"(approved_by={approval.approved_by})"
            )
            return 0
        if args.prompt_factory_verb == "materialize":
            _, orch_paths = _load_config_and_paths(repo)
            result = materialize_prompt_factory_draft(
                load_draft_json(Path(args.draft_json)),
                load_review_gate_json(Path(args.review_gate_json)),
                load_operator_approval_json(Path(args.approval_json)),
                repo_root=repo,
                target=Path(args.target),
                dry_run=args.dry_run,
                force=args.force,
                iteration_root=orch_paths.iteration_root.relative_to(
                    repo
                ).as_posix(),
                prompt_filename=orch_paths.iteration_prompt_filename,
                task_board_filename=orch_paths.task_board_filename,
                task_prompts_dirname=orch_paths.task_prompts_dirname,
                task_reviews_dirname=orch_paths.task_reviews_dirname,
            )
            action = "would write" if result.dry_run else "wrote"
            print(
                f"{action} {len(result.planned_files)} file(s) "
                f"under {result.target_dir}"
            )
            for path in result.planned_files:
                print(path)
            return 0
    except (
        PromptFactoryApprovalError,
        PromptFactoryMaterializationError,
        PromptFactoryReviewError,
        PromptFactoryValidationError,
    ) as exc:
        print(f"prompt-factory validation failed: {exc}", file=sys.stderr)
        return 1
    except ConfigError as exc:
        print(f"project.yaml error: {exc}", file=sys.stderr)
        return 1

    print(
        f"error: unknown prompt-factory verb {args.prompt_factory_verb!r}",
        file=sys.stderr,
    )
    return 2


# ---------------------------------------------------------------------------
# run / resume
# ---------------------------------------------------------------------------


def _build_runner(
    args: argparse.Namespace, *, require_state: bool
) -> IterationRunner | int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, log_dir = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch=board.iteration_branch,
        command="resume" if require_state else "run",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1

    try:
        # Reload under lock so resumes/retries see the latest state after any
        # previous command released the same iteration lock.
        if store.exists():
            store.load()
        if require_state and not store.exists():
            print(
                f"resume: no run_state.json for {args.iteration}; "
                f"use 'orch run' first",
                file=sys.stderr,
            )
            return 1

        try:
            orch_cwd = ensure_orch_workdir(
                repo,
                args.iteration,
                board.iteration_branch,
                worktree_root=orch_paths.worktree_root,
            )
        except WorktreePreflightError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        # Build adapters for every configured agent — unused ones sit idle.
        adapters = {}
        for name, spec in cfg.data["agents"].items():
            adapters[name] = build_adapter(name, spec)

        options = RunOptions(
            implementer=getattr(args, "implementer", "") or "",
            reviewer=getattr(args, "reviewer", "") or "",
            secondary_reviewer=getattr(args, "secondary_reviewer", "") or "",
            independence=getattr(args, "independence", "") or "",
            stop_on_first_failure=bool(
                getattr(args, "stop_on_first_failure", False)
            ),
            accept_external_sha=bool(getattr(args, "accept_external", False)),
            override_agents=bool(getattr(args, "override_agents", False)),
            skip_impl_tasks=list(getattr(args, "skip_impl", []) or []),
            allow_noop_acceptance_reason=(
                getattr(args, "allow_noop_acceptance", None) or ""
            ),
            dry_run=bool(getattr(args, "dry_run", False)),
            poll_ci=not bool(getattr(args, "dry_run", False)),
            team_mode=getattr(args, "team_mode", "") or "",
        )

        if store.exists():
            guard = head_sha_guard(
                store, cwd=orch_cwd, iter_branch=board.iteration_branch,
                accept_external=options.accept_external_sha,
            )
            if not guard.ok:
                print(guard.reason, file=sys.stderr)
                return 1

        cost = CostLogger(
            path=log_dir / "cost.jsonl",
            cost_table=cfg.data["costs"],
            iteration=args.iteration,
        )
        deps = RunnerDeps(repo_root=repo, cwd=orch_cwd, run_lock=lock)
        runner = IterationRunner(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters=adapters, options=options, deps=deps,
        )
        lock = None
        return runner
    finally:
        if lock is not None:
            lock.release()


def cmd_run(args: argparse.Namespace) -> int:
    runner = _build_runner(args, require_state=False)
    if isinstance(runner, int):
        return runner
    try:
        return runner.run()
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        _release_runner_lock(runner)


def cmd_resume(args: argparse.Namespace) -> int:
    runner = _build_runner(args, require_state=True)
    if isinstance(runner, int):
        return runner
    try:
        return runner.run()
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        _release_runner_lock(runner)


# ---------------------------------------------------------------------------
# revert / cleanup
# ---------------------------------------------------------------------------


def cmd_retry(args: argparse.Namespace) -> int:
    """Reset a single task to WAITING and re-run the iteration."""
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, _ = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch=board.iteration_branch,
        command="retry",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1
    try:
        if store.exists():
            store.load()
        return _cmd_retry_locked(args, board, store)
    finally:
        lock.release()


def _cmd_retry_locked(
    args: argparse.Namespace, board, store: StateStore,
) -> int:
    if not store.exists():
        print(
            f"retry: no run_state.json for {args.iteration}; "
            f"use 'orch run' first",
            file=sys.stderr,
        )
        return 1
    task_id = args.task
    # Validate that the task exists on the board
    try:
        board.by_id(task_id)
    except KeyError:
        print(f"retry: unknown task '{task_id}'", file=sys.stderr)
        return 1
    store.reset_task(task_id)
    # Also reset any downstream tasks that were BLOCKED_UPSTREAM
    for t in board.tasks:
        ts = store.tasks.get(t.id)
        if (
            ts
            and ts.status == STATUS_BLOCKED_UPSTREAM
            and depends_transitively_on(board, t.id, task_id)
        ):
            store.reset_task(t.id)
    print(f"reset {task_id} to WAITING — run 'orch resume {args.iteration}' to continue")
    return 0


def cmd_revert(args: argparse.Namespace) -> int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, _ = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch=board.iteration_branch,
        command="revert",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1
    try:
        if store.exists():
            store.load()
        if not store.exists():
            print(
                f"revert: no run_state.json for {args.iteration}",
                file=sys.stderr,
            )
            return 1
        res = revert_task_merge(store, board, args.task, cwd=repo)
        if not res.ok:
            print(f"revert failed: {res.message}", file=sys.stderr)
            return 1
        print(res.message)
        return 0
    finally:
        lock.release()


def cmd_cleanup(args: argparse.Namespace) -> int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, _ = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch=board.iteration_branch,
        command="cleanup",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1
    try:
        if store.exists():
            store.load()
        res = cleanup_task_branches(store, board, cwd=repo, force=args.force)
        for b in res.deleted:
            print(f"deleted {b}")
        for b, reason in res.skipped:
            print(f"skipped {b}: {reason}", file=sys.stderr)
        return 0
    finally:
        lock.release()


def cmd_cleanup_workdir(args: argparse.Namespace) -> int:
    repo = _repo_root()
    try:
        _, orch_paths = _load_config_and_paths(repo)
    except ConfigError as exc:
        print(f"project.yaml error: {exc}", file=sys.stderr)
        return 1
    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch="",
        command="cleanup-workdir",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1
    try:
        cleanup_orch_workdir(
            repo,
            args.iteration,
            worktree_root=orch_paths.worktree_root,
        )
        print(f"removed orch workdir for {args.iteration}")
        return 0
    finally:
        lock.release()


def _render_recover_plan(plan) -> str:
    lines = [f"recover plan for {plan.iteration}"]
    lock = plan.lock
    if lock.exists:
        if lock.info is None:
            lines.append(
                f"- lock: {lock.path} unreadable ({lock.read_error})"
            )
        else:
            safety = "removable" if lock.removable_without_force else "blocked"
            if lock.force_required:
                safety = "requires --force-lock"
            if not lock.belongs_to_iteration:
                safety = "blocked: different iteration"
            lines.append(
                f"- lock: {lock.path} pid_status={lock.pid_status} "
                f"iteration={lock.info.iteration} safety={safety}"
            )
    else:
        lines.append(f"- lock: none at {lock.path}")

    workdir = plan.workdir
    if not workdir.exists:
        lines.append(f"- workdir: none at {workdir.path}")
    else:
        if workdir.dirty is None:
            cleanliness = f"unknown ({workdir.dirty_error})"
        else:
            cleanliness = "dirty" if workdir.dirty else "clean"
        lines.append(f"- workdir: {workdir.path} exists, {cleanliness}")
        if workdir.dirty:
            lines.append(
                f"- salvage: would park dirty workdir at "
                f"salvage/{plan.iteration}/recover before cleanup"
            )

    if plan.in_progress_tasks:
        tasks = ", ".join(plan.in_progress_tasks)
        lines.append(f"- task resets: {tasks} -> WAITING")
    else:
        lines.append("- task resets: none")
    lines.append(
        f"- next: run `python -m orch resume {plan.iteration}` "
        "after applying recovery"
    )
    return "\n".join(lines)


def cmd_recover(args: argparse.Namespace) -> int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, log_dir = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    plan = build_recovery_plan(
        store,
        repo_root=repo,
        iteration=args.iteration,
        log_dir=log_dir,
        worktree_root=orch_paths.worktree_root,
    )

    if not args.apply:
        print("DRY RUN - no files, locks, worktrees, or state will change")
        print(_render_recover_plan(plan))
        return 0

    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch=board.iteration_branch,
        command="recover",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1

    try:
        try:
            lock_removed_meta = remove_recoverable_lock(
                plan, force_lock=args.force_lock,
            )
        except RecoverApplyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        if store.exists():
            store.load()
        if lock_removed_meta is not None:
            store.append_event(kind="note", meta=lock_removed_meta)

        try:
            salvage_dirty_recover_workdir(
                store,
                workdir=plan.workdir.path,
                iteration=args.iteration,
            )
            cleaned = cleanup_recover_workdir(
                store,
                repo_root=repo,
                iteration=args.iteration,
                worktree_root=orch_paths.worktree_root,
            )
        except RecoverApplyError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        reset = reset_in_progress_tasks_for_recovery(store)
        if lock_removed_meta is not None:
            print(f"removed recoverable lock at {lock_removed_meta['path']}")
        if cleaned:
            print(f"cleaned orch workdir for {args.iteration}")
        for task_id in reset:
            print(f"reset {task_id} to WAITING")
        print(f"next: python -m orch resume {args.iteration}")
        return 0
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# qa / retro / iteration
# ---------------------------------------------------------------------------


def _build_adapters(cfg):
    adapters = {}
    for name, spec in cfg.data["agents"].items():
        adapters[name] = build_adapter(name, spec)
    return adapters


def _recorded_implementer_names(store) -> set[str]:
    """Implementer agent names recorded in run_state (per-task, else run-level)."""
    names = {
        ts.implementer
        for ts in store.tasks.values()
        if getattr(ts, "implementer", None)
    }
    if not names:
        for event in store.events:
            meta = event.get("meta") or {}
            if meta.get("event") == "agents_resolved" and meta.get("implementer"):
                names.add(meta["implementer"])
    return names


def _warn_if_same_family_as_implementer(store, adapters, agent_name, *, role):
    """B-14 R8: warn (non-blocking) when a QA/retro agent shares the model
    family of the recorded implementer — independent review is weakened.

    Only the run-start independence gate hard-blocks the implement↔review pair;
    qa/retro are run separately, so an accidental same-family reviewer would
    otherwise pass silently.
    """
    reviewer_family = getattr(adapters.get(agent_name), "family", None)
    if not reviewer_family:
        return
    for impl_name in sorted(_recorded_implementer_names(store)):
        impl_family = getattr(adapters.get(impl_name), "family", None)
        if impl_family and impl_family == reviewer_family:
            print(
                f"WARNING: {role} '{agent_name}' is the same model family "
                f"('{reviewer_family}') as the recorded implementer "
                f"'{impl_name}'. Independent review is weakened; prefer an "
                "agent from a different model family (B-14 R8).",
                file=sys.stderr,
            )
            return


def _incomplete_required_role_names(report) -> list[str]:
    return [
        r.role for r in getattr(report, "roles", [])
        if not getattr(r, "ok", False)
    ]


def _print_incomplete_review_error(stage: str, roles: list[str]) -> None:
    role_text = ", ".join(roles)
    print(
        f"{stage} incomplete: required role(s) failed or timed out: "
        f"{role_text}. Re-run with --allow-partial to accept incomplete "
        "review coverage explicitly.",
        file=sys.stderr,
    )


def _run_qa_loaded(
    args: argparse.Namespace,
    repo: Path,
    cfg,
    board,
    store: StateStore,
    log_dir: Path,
    *,
    diff_error_prefix: str | None = None,
) -> int:
    adapters = _build_adapters(cfg)
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=cfg.data["costs"],
        iteration=args.iteration,
    )
    reviewer = args.reviewer or next(iter(cfg.data["agents"]))
    if reviewer not in adapters:
        print(f"unknown agent '{reviewer}'", file=sys.stderr)
        return 1
    _warn_if_same_family_as_implementer(
        store, adapters, reviewer, role="QA reviewer"
    )
    roles = args.roles.split(",") if getattr(args, "roles", None) else None
    qa_runner = (
        run_qa_team_mode
        if bool(getattr(args, "team_mode", False))
        else run_qa
    )
    try:
        report = qa_runner(
            cfg=cfg, board=board, state=store, cost=cost,
            adapters=adapters, iteration=args.iteration, cwd=repo,
            reviewer_agent=reviewer, roles=roles,
            synthesize=bool(getattr(args, "synthesize", False)),
            timeout=int(timeouts(cfg)["qa"]),
            allow_empty_diff_reason=getattr(args, "allow_empty_diff", None),
        )
    except QaEmptyDiffError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except QaDiffBaseError as exc:
        if diff_error_prefix:
            print(f"{diff_error_prefix}: {exc}", file=sys.stderr)
            return 0
        print(str(exc), file=sys.stderr)
        return 1
    incomplete = _incomplete_required_role_names(report)
    if incomplete and not bool(getattr(args, "allow_partial", False)):
        _print_incomplete_review_error("QA", incomplete)
        return 1
    return 0


def _run_retro_loaded(
    args: argparse.Namespace,
    repo: Path,
    cfg,
    board,
    store: StateStore,
    log_dir: Path,
) -> int:
    adapters = _build_adapters(cfg)
    cost = CostLogger(
        path=log_dir / "cost.jsonl",
        cost_table=cfg.data["costs"],
        iteration=args.iteration,
    )
    agent = (
        getattr(args, "agent", None)
        or getattr(args, "retro_agent", None)
        or next(iter(cfg.data["agents"]))
    )
    if agent not in adapters:
        print(f"unknown agent '{agent}'", file=sys.stderr)
        return 1
    _warn_if_same_family_as_implementer(
        store, adapters, agent, role="retrospective agent"
    )
    roles = args.roles.split(",") if getattr(args, "roles", None) else None
    retro_runner = (
        run_retro_team_mode
        if bool(getattr(args, "team_mode", False))
        else run_retro
    )
    report = retro_runner(
        cfg=cfg, board=board, state=store, cost=cost,
        adapters=adapters, iteration=args.iteration, cwd=repo,
        agent_name=agent, roles=roles, timeout=int(timeouts(cfg)["retro"]),
    )
    incomplete = _incomplete_required_role_names(report)
    if incomplete and not bool(getattr(args, "allow_partial", False)):
        _print_incomplete_review_error("Retro", incomplete)
        return 1
    return 0


def cmd_qa(args: argparse.Namespace) -> int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, log_dir = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch=board.iteration_branch,
        command="qa",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1
    try:
        if store.exists():
            store.load()
        return _run_qa_loaded(args, repo, cfg, board, store, log_dir)
    finally:
        lock.release()


def cmd_retro(args: argparse.Namespace) -> int:
    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, log_dir = loaded
    orch_paths = resolve_orch_paths(repo, cfg)
    lock = _acquire_iteration_lock(
        repo=repo,
        iteration=args.iteration,
        iter_branch=board.iteration_branch,
        command="retro",
        orch_paths=orch_paths,
    )
    if lock is None:
        return 1
    try:
        if store.exists():
            store.load()
        return _run_retro_loaded(args, repo, cfg, board, store, log_dir)
    finally:
        lock.release()


def cmd_iteration(args: argparse.Namespace) -> int:
    """Chain: run -> qa -> retro sequentially."""
    rc = cmd_run(args)
    if rc != 0:
        print(
            f"run failed (exit {rc}); skipping qa/retro",
            file=sys.stderr,
        )
        return rc

    repo = _repo_root()
    loaded = _load(repo, args.iteration)
    if loaded is None:
        return 1
    cfg, board, store, log_dir = loaded
    orch_paths = resolve_orch_paths(repo, cfg)

    # Step 2: qa
    if not args.skip_qa:
        lock = _acquire_iteration_lock(
            repo=repo,
            iteration=args.iteration,
            iter_branch=board.iteration_branch,
            command="qa",
            orch_paths=orch_paths,
        )
        if lock is None:
            return 1
        try:
            if store.exists():
                store.load()
            qa_args = argparse.Namespace(
                iteration=args.iteration,
                reviewer=args.reviewer,
                roles=None,
                synthesize=False,
                allow_partial=getattr(args, "allow_partial", False),
                allow_empty_diff=getattr(args, "allow_empty_diff", None),
            )
            rc = _run_qa_loaded(
                qa_args, repo, cfg, board, store, log_dir,
                diff_error_prefix="QA skipped",
            )
            if rc != 0:
                return rc
        finally:
            lock.release()

    # Step 3: retro
    if not args.skip_retro:
        lock = _acquire_iteration_lock(
            repo=repo,
            iteration=args.iteration,
            iter_branch=board.iteration_branch,
            command="retro",
            orch_paths=orch_paths,
        )
        if lock is None:
            return 1
        try:
            if store.exists():
                store.load()
            retro_args = argparse.Namespace(
                iteration=args.iteration,
                agent=getattr(args, "retro_agent", None),
                roles=None,
                allow_partial=getattr(args, "allow_partial", False),
            )
            rc = _run_retro_loaded(
                retro_args, repo, cfg, board, store, log_dir,
            )
            if rc != 0:
                return rc
        finally:
            lock.release()

    return 0


def cmd_timing(args: argparse.Namespace) -> int:
    """Mark or report manual phase-timing events for an iteration."""
    import os

    from orch import timing as timing_mod

    repo = _repo_root()
    iteration = args.iter or os.environ.get("ORCH_ITER")
    if not iteration:
        print(
            "error: --iter required (or set ORCH_ITER env var). "
            "Example: python -m orch timing start 'T2 impl' "
            "--iter pre-i8-orch",
            file=sys.stderr,
        )
        return 2
    try:
        _, orch_paths = _load_config_and_paths(repo)
        artifact_root_ref = orch_paths.artifact_root_ref
    except Exception:
        # Timing is a lightweight manual harness; if the project config is
        # absent or incomplete, fall back to the historical default artifact
        # root rather than failing the command.
        artifact_root_ref = "tools/logs"
    if args.timing_verb == "start":
        rec = timing_mod.record_event(
            repo, iteration, "start", args.label, artifact_root_ref
        )
        print(f"start  {rec['label']:<40} {rec['ts']}")
        return 0
    if args.timing_verb == "end":
        rec = timing_mod.record_event(
            repo, iteration, "end", args.label, artifact_root_ref
        )
        print(f"end    {rec['label']:<40} {rec['ts']}")
        return 0
    if args.timing_verb == "report":
        summary = timing_mod.summarize(repo, iteration, artifact_root_ref)
        print(timing_mod.render_report(summary), end="")
        return 0
    print(f"error: unknown timing verb {args.timing_verb!r}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="orch",
        description=(
            "Autonomous iteration-completion orchestrator. "
            "See ARCHITECTURE.md."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="Validate project.yaml and tasks.md")
    v.add_argument("iteration")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("status", help="Print a short run-state summary")
    s.add_argument("iteration")
    s.set_defaults(func=cmd_status)

    w = sub.add_parser("watch", help="Print recent run-state events")
    w.add_argument("iteration")
    w.add_argument("--limit", type=int, default=20)
    w.set_defaults(func=cmd_watch)

    r = sub.add_parser("report", help="Render the readiness report")
    r.add_argument("iteration")
    r.add_argument("--out", default=None,
                   help="Write report to this path instead of stdout")
    r.set_defaults(func=cmd_report)

    doctor = sub.add_parser(
        "doctor",
        help="Check project pack, agent CLIs, git/gh, and Python deps",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="Emit a structured JSON report",
    )
    doctor.set_defaults(func=cmd_doctor)

    improvements_cmd = sub.add_parser(
        "improvements",
        help="List or validate Six Sigma improvement records",
    )
    improvements_subs = improvements_cmd.add_subparsers(
        dest="improvements_verb",
        required=True,
    )
    improvements_list = improvements_subs.add_parser(
        "list",
        help="Print improvement records for an iteration as JSONL",
    )
    improvements_list.add_argument("iteration")
    improvements_list.set_defaults(func=cmd_improvements)
    improvements_validate = improvements_subs.add_parser(
        "validate",
        help="Validate improvement records for an iteration",
    )
    improvements_validate.add_argument("iteration")
    improvements_validate.set_defaults(func=cmd_improvements)

    prompt_factory_cmd = sub.add_parser(
        "prompt-factory",
        help=(
            "Validate, render, review, approve, or materialize deterministic "
            "Prompt Factory drafts"
        ),
    )
    prompt_factory_subs = prompt_factory_cmd.add_subparsers(
        dest="prompt_factory_verb",
        required=True,
    )
    prompt_factory_validate = prompt_factory_subs.add_parser(
        "validate",
        help="Validate a Prompt Factory draft JSON file",
    )
    prompt_factory_validate.add_argument("draft_json")
    prompt_factory_validate.set_defaults(func=cmd_prompt_factory)
    prompt_factory_render = prompt_factory_subs.add_parser(
        "render",
        help="Print a tasks.md preview for a Prompt Factory draft JSON file",
    )
    prompt_factory_render.add_argument("draft_json")
    prompt_factory_render.set_defaults(func=cmd_prompt_factory)
    prompt_factory_review_package = prompt_factory_subs.add_parser(
        "review-package",
        help="Write deterministic review-gate prompt artifacts",
    )
    prompt_factory_review_package.add_argument("draft_json")
    prompt_factory_review_package.set_defaults(func=cmd_prompt_factory)
    prompt_factory_review_status = prompt_factory_subs.add_parser(
        "review-status",
        help="Read review artifacts and update the Prompt Factory gate status",
    )
    prompt_factory_review_status.add_argument("draft_id")
    prompt_factory_review_status.set_defaults(func=cmd_prompt_factory)
    prompt_factory_approve_check = prompt_factory_subs.add_parser(
        "approve-check",
        help="Validate draft, review gate, and operator approval artifacts",
    )
    prompt_factory_approve_check.add_argument("draft_json")
    prompt_factory_approve_check.add_argument("review_gate_json")
    prompt_factory_approve_check.add_argument("approval_json")
    prompt_factory_approve_check.set_defaults(func=cmd_prompt_factory)
    prompt_factory_materialize = prompt_factory_subs.add_parser(
        "materialize",
        help="Materialize an approved draft into an iteration package",
    )
    prompt_factory_materialize.add_argument("draft_json")
    prompt_factory_materialize.add_argument("review_gate_json")
    prompt_factory_materialize.add_argument("approval_json")
    prompt_factory_materialize.add_argument("--target", required=True)
    prompt_factory_materialize.add_argument("--dry-run", action="store_true")
    prompt_factory_materialize.add_argument("--force", action="store_true")
    prompt_factory_materialize.set_defaults(func=cmd_prompt_factory)

    def _add_run_flags(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("iteration")
        parser.add_argument("--implementer", default=None,
                            help="Agent name (from project.yaml agents:)")
        parser.add_argument("--reviewer", default=None)
        parser.add_argument(
            "--secondary-reviewer",
            default=None,
            help="Independent reviewer for dual-model agreement tasks",
        )
        parser.add_argument("--independence",
                            choices=["session", "model", "model_family"],
                            default=None)
        parser.add_argument("--stop-on-first-failure", action="store_true")
        parser.add_argument("--dry-run", action="store_true",
                            help="Skip PR/merge/CI — used for local smoke tests")
        parser.add_argument(
            "--override-agents",
            action="store_true",
            help=(
                "Allow a different implementer/reviewer than the pair "
                "recorded in run_state"
            ),
        )
        parser.add_argument(
            "--allow-noop-acceptance",
            metavar="REASON",
            default=None,
            help=(
                "Allow no-CI local merge when the acceptance test command is "
                "empty or a known no-op; records REASON in run_state"
            ),
        )
        parser.add_argument(
            "--team-mode",
            choices=["planning"],
            default=None,
            help="Run docs/iterations planning tasks through planning-team mode",
        )

    run_cmd = sub.add_parser(
        "run",
        help="Run the full pipeline for an iteration",
    )
    _add_run_flags(run_cmd)
    run_cmd.set_defaults(func=cmd_run)

    resume_cmd = sub.add_parser(
        "resume",
        help="Resume a previously-started iteration from its run_state.json",
    )
    _add_run_flags(resume_cmd)
    resume_cmd.add_argument("--accept-external", action="store_true",
                            help="Accept the current iter-branch HEAD even "
                                 "if it differs from the recorded SHA")
    resume_cmd.add_argument("--skip-impl", action="append", default=[],
                            metavar="TASK_ID",
                            help="Skip the implement step for this task "
                                 "(repeatable)")
    resume_cmd.set_defaults(func=cmd_resume)

    retry_cmd = sub.add_parser(
        "retry",
        help="Reset a stopped/failed task to WAITING and unblock downstream",
    )
    retry_cmd.add_argument("iteration")
    retry_cmd.add_argument("task", help="Task ID to retry (e.g. I4-T2)")
    retry_cmd.set_defaults(func=cmd_retry)

    rv = sub.add_parser(
        "revert",
        help="Revert the merge commit recorded for a task",
    )
    rv.add_argument("iteration")
    rv.add_argument("task")
    rv.set_defaults(func=cmd_revert)

    cl = sub.add_parser(
        "cleanup",
        help="Delete local task branches for DONE tasks",
    )
    cl.add_argument("iteration")
    cl.add_argument("--force", action="store_true",
                    help="Also delete non-DONE task branches (uses -D)")
    cl.set_defaults(func=cmd_cleanup)

    clw = sub.add_parser(
        "cleanup-workdir",
        help="Remove the dedicated orch sub-worktree for an iteration",
    )
    clw.add_argument("iteration")
    clw.set_defaults(func=cmd_cleanup_workdir)

    recover_cmd = sub.add_parser(
        "recover",
        help="Plan or apply cautious recovery for an interrupted iteration",
    )
    recover_cmd.add_argument("iteration")
    recover_cmd.add_argument(
        "--apply",
        action="store_true",
        help="Apply safe recovery actions; omitted mode is a dry-run plan",
    )
    recover_cmd.add_argument(
        "--force-lock",
        action="store_true",
        help="Allow removal of active/unknown locks after manual confirmation",
    )
    recover_cmd.set_defaults(func=cmd_recover)

    qa_cmd = sub.add_parser(
        "qa",
        help="Run 5 parallel QA reviewers on a completed iteration",
    )
    qa_cmd.add_argument("iteration")
    qa_cmd.add_argument("--reviewer", default=None,
                        help="Agent name for QA reviews (from project.yaml)")
    qa_cmd.add_argument("--synthesize", action="store_true",
                        help="Add a synthesis pass that merges all reviews")
    qa_cmd.add_argument("--roles", default=None,
                        help="Comma-separated subset of roles "
                             "(security,architecture,test,product,process)")
    qa_cmd.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow QA to exit 0 when a required role fails or times out",
    )
    qa_cmd.add_argument(
        "--allow-empty-diff",
        metavar="REASON",
        default=None,
        help="Allow QA to review an empty or placeholder diff with a reason",
    )
    qa_cmd.add_argument(
        "--team-mode",
        action="store_true",
        help="Run QA reviewers through read-only Agent Team mode",
    )
    qa_cmd.set_defaults(func=cmd_qa)

    retro_cmd = sub.add_parser(
        "retro",
        help="Run 3 parallel retrospective perspectives",
    )
    retro_cmd.add_argument("iteration")
    retro_cmd.add_argument("--agent", default=None,
                           help="Agent name for retro (from project.yaml)")
    retro_cmd.add_argument("--roles", default=None,
                           help="Comma-separated subset of roles "
                                "(developer,product_owner,scrum_master)")
    retro_cmd.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow retro to exit 0 when a required role fails or times out",
    )
    retro_cmd.add_argument(
        "--team-mode",
        action="store_true",
        help="Run retro perspectives through read-only Agent Team mode",
    )
    retro_cmd.set_defaults(func=cmd_retro)

    iter_cmd = sub.add_parser(
        "iteration",
        help="Chain run -> qa -> retro sequentially",
    )
    _add_run_flags(iter_cmd)
    iter_cmd.add_argument("--skip-qa", action="store_true",
                          help="Skip the QA step")
    iter_cmd.add_argument("--skip-retro", action="store_true",
                          help="Skip the retrospective step")
    iter_cmd.add_argument(
        "--retro-agent",
        default=None,
        help="Agent name for the chained retrospective phase",
    )
    iter_cmd.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow QA/retro to exit 0 with incomplete role coverage",
    )
    iter_cmd.add_argument(
        "--allow-empty-diff",
        metavar="REASON",
        default=None,
        help="Allow chained QA to review an empty or placeholder diff",
    )
    iter_cmd.set_defaults(func=cmd_iteration)

    ptiming = sub.add_parser(
        "timing",
        help="Mark or report manual phase-timing events.",
    )
    ptiming.add_argument(
        "--iter",
        help="Iteration id (defaults to $ORCH_ITER).",
    )
    timing_subs = ptiming.add_subparsers(dest="timing_verb", required=True)
    timing_start = timing_subs.add_parser("start", help="Mark a phase start")
    timing_start.add_argument("label")
    timing_end = timing_subs.add_parser("end", help="Mark a phase end")
    timing_end.add_argument("label")
    timing_subs.add_parser("report", help="Print a markdown timing report")
    ptiming.set_defaults(func=cmd_timing)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
