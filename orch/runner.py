"""Main orchestration loop — ties every core primitive into the
per-task state machine.

The loop is a pure object. Subprocess boundaries (agent adapters,
``git``, ``gh``, acceptance commands) are injected, so tests can drive
the whole machine with fakes while production wires in the real CLIs.

Exit model:
    * ``run_iteration`` returns 0 on full success, 1 on any blocker.
    * Every task walks through :meth:`_execute_task` and ends in exactly
      one of: DONE, NEEDS_HUMAN_MERGE, STOPPED:<reason>, BLOCKED_UPSTREAM.
    * ``--stop-on-first-failure`` converts the first non-merge STOP into
      an immediate loop exit.
"""
from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from orch.agents import (
    AgentAdapter,
    AgentInvocationOptions,
    compose_prompt,
)
from orch.agents.base import (
    cost_record_usage_kwargs,
    dispatched_model_from_options,
    ensure_usage_json_args,
)
from orch.checks import (
    acceptance_test_command_is_noop,
    check_diff_size,
    check_forbidden_patterns,
    check_scope,
    check_sensitive_files,
    check_tasks_md_status_only,
    check_tasks_md_touched,
    effective_acceptance_test_command,
    load_nav_discoverability_evidence,
    load_scope_exception_evidence,
    run_acceptance,
)
from orch.config import LoadedConfig
from orch.cost import CostLogger
from orch.final_gates import (
    FinalNavDiscoverabilityGateDecision,
    FinalScopeGateDecision,
    evaluate_final_outward_scope_gate,
    evaluate_final_nav_discoverability_gate,
    final_nav_discoverability_gate_input,
    final_scope_base_ref_yields_diff,
    unresolved_final_nav_discoverability_base_ref_decision,
    unresolved_final_scope_base_ref_decision,
)
from orch.git_ops import (
    classify_branch_freshness,
    checkout,
    cleanup_orch_workdir,
    commit,
    create_or_reset_branch,
    current_sha,
    diff_files,
    diff_stats,
    diff_text,
    fetch,
    git,
    GitError,
    merge_no_ff,
    preferred_remote_ref,
    push_branch,
    pull_ff_only,
    render_branch_freshness_recovery,
    revert_paths,
    salvage_worktree,
    stage_all,
    upstream_ref,
    working_tree_clean,
)
from orch.hooks import HookVeto
from orch.locks import RunStateLock
from orch.lifecycle import (
    PhaseResolutionError,
    resolve_phase_branch,
)
from orch.merge import (
    build_task_pr_request,
    comment_pr,
    evaluate_auto_merge,
    merge_pr,
    open_pr,
    parse_merge_sha,
    query_pr_state,
    render_guard_comment,
    wait_for_ci,
)
from orch.model_routing import (
    ROUTING_WARNINGS_FILENAME,
    ResolvedModelRouting,
    append_unknown_risk_warning,
    resolve_agent_routing_options,
    resolve_model_routing,
    routing_to_dict,
)
from orch.parallel import validate_conflict_tokens
from orch.parallel_runner import ParallelExecutionMixin
from orch.paths import resolve_orch_paths
from orch.planning_team import (
    PLANNING_REVIEW_INDEPENDENCE_POLICY,
    PlanningTeamCandidate,
    PlanningTeamError,
    PlanningTeamRefusal,
    has_overlapping_allowed_files,
    run_planning_team,
    validate_planning_candidate,
)
from orch.preflight import Tier, estimate, timeouts_for_tier
from orch.review import (
    Verdict,
    check_independence,
    parse_verdict,
)
from orch import review_flow as _review_flow
from orch.review_flow import (
    build_task_review_prompt_text,
    extract_review_findings as _extract_review_findings,
    load_review_prompt_contract,
    review_prompt_candidates,
)
from orch.run_loop import (
    mark_downstream_blocked,
    pick_next_ready_task,
)
from orch.recovery import (
    append_recovery_note,
    impl_failure_recovery_note,
    stop_reason_recovery_note,
)
from orch import triage as _triage
from orch.state import (
    EVT_HOOK,
    EVT_PAIR_SWAP,
    STATUS_BLOCKED_UPSTREAM,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_HUMAN_MERGE,
    STATUS_STOPPED_PREFIX,
    StateStore,
)
from orch.stops import (
    STOP_BRANCH_FRESHNESS,
    STOP_CHECKS,
    STOP_CONFIG,
    STOP_DUAL_REVIEW_FAIL,
    STOP_DUAL_REVIEW_MALFORMED,
    STOP_DUAL_REVIEW_REQUIRED,
    STOP_HOOK_VETO,
    STOP_IMPL_FAILED,
    STOP_IMPL_TIMEOUT,
    STOP_INDEPENDENCE,
    STOP_INTERNAL,
    STOP_PREFLIGHT,
    STOP_REVIEW_FAIL,
    STOP_REVIEW_MALFORMED,
    STOP_SCOPE,
    STOP_STRUCTURAL,
    _TaskStopped,
)
from orch.task_execution import (
    RATE_LIMIT_HINT as _RATE_LIMIT_HINT,
    classify_impl_failure as _classify_impl_failure,
    diff_introduces_conflict_marker_pair as _diff_introduces_conflict_marker_pair,
    rate_limit_signature as _rate_limit_signature,
    stderr_tail as _stderr_tail,
)
from orch.tasks_schema import DiffCapOverride, Task, TaskBoard
from orch import finalization as _finalization


RUNTIME_FALLBACK_REVIEW_CONTRACT = _review_flow.RUNTIME_FALLBACK_REVIEW_CONTRACT


@dataclass
class RunOptions:
    implementer: str = ""
    reviewer: str = ""
    secondary_reviewer: str = ""
    independence: str = ""                 # empty → config default
    stop_on_first_failure: bool = False
    accept_external_sha: bool = False      # resume flag
    override_agents: bool = False
    skip_impl_tasks: list[str] = field(default_factory=list)
    allow_noop_acceptance_reason: str = ""
    dry_run: bool = False                  # skip PR/merge/gh calls
    poll_ci: bool = True                   # set False when dry_run
    team_mode: str = ""                    # empty | planning


@dataclass
class RunnerDeps:
    """Injection surface. Tests stub these; production wires the real tools."""
    repo_root: Path
    cwd: Path
    # gh layer — tests stub these to avoid network / real gh calls
    open_pr: Callable = open_pr
    merge_pr: Callable = merge_pr
    comment_pr: Callable = comment_pr
    wait_for_ci: Callable = wait_for_ci
    push_branch: Callable = push_branch
    # NEEDS_HUMAN_MERGE hardening + external-merge detection
    query_pr_state: Callable = query_pr_state
    # Held by CLI commands that mutate run_state/worktrees.
    run_lock: RunStateLock | None = None


class RunnerError(Exception):
    pass


class IterationRunner(ParallelExecutionMixin):
    """Walk the DAG and run per-task state machine.

    Parameters
    ----------
    cfg, board, state, cost
        Pre-loaded config, parsed task board, state store, cost logger.
    adapters
        Mapping of agent name → concrete :class:`AgentAdapter`. The runner
        picks implementer / reviewer from this table.
    options
        Per-run flags (implementer/reviewer names, independence override,
        dry-run, stop-on-first-failure).
    deps
        Injection surface for subprocess boundaries.
    """

    def __init__(
        self,
        *,
        cfg: LoadedConfig,
        board: TaskBoard,
        state: StateStore,
        cost: CostLogger,
        adapters: dict[str, AgentAdapter],
        options: RunOptions,
        deps: RunnerDeps,
    ) -> None:
        self.cfg = cfg
        self.board = board
        self.state = state
        self.cost = cost
        self.adapters = adapters
        self.options = options
        self.deps = deps
        self.iter_branch = board.iteration_branch
        self.main_branch = cfg.data["project"]["main_branch"]
        self.iteration = state.iteration
        self.repo_root = deps.repo_root.resolve()
        self.orch_cwd = deps.cwd.resolve()
        self.orch_paths = resolve_orch_paths(self.repo_root, cfg)
        self.tasks_md_path_abs = board.path.resolve()
        self.tasks_md_rel = str(self.tasks_md_path_abs.relative_to(self.repo_root))
        self.tasks_md_worktree_path = self.orch_cwd / self.tasks_md_rel
        # Per-task review-finding fingerprints, ordered oldest →
        # newest. Used by the triage classifier (repeat-failure
        # detection). In-memory only; persisted indirectly via the
        # `triage` event log if operators need to reconstruct.
        self._finding_fingerprints: dict[str, list[str]] = {}
        self._model_routing: dict[str, ResolvedModelRouting] = {}
        self._unknown_risk_warned: set[str] = set()
        self._local_merge_reconciled_this_run = False
        self._planning_team_serial_fallback = False

    # ------------------------------------------------------------------
    # Progress output (P0-2)
    # ------------------------------------------------------------------

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[orch] {msg}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------

    def run(self) -> int:
        # P2-5: abort early if the working tree is dirty
        if not working_tree_clean(self.deps.cwd):
            self._log("ERROR: working tree is dirty — commit or stash before running")
            return 1

        self._log(f"Starting iteration {self.iteration} on {self.iter_branch}")
        self.state.mark_iteration_started()
        self.state.set_iter_branch_sha(current_sha(self.deps.cwd, self.iter_branch))

        # Auto-detect external PR merges so
        # `orch resume` doesn't need a manual run_state.json edit after
        # the operator merges by hand.
        self._reconcile_local_merges()
        self._check_external_merges()

        implementer_name, reviewer_name, agent_source = self._resolve_agents()
        ind = self._check_global_independence(implementer_name, reviewer_name)
        if not ind.ok:
            self.state.append_event(
                kind="note",
                meta={"event": "stop_global", "reason": STOP_INDEPENDENCE,
                      "msg": ind.reason},
            )
            return 1
        self.state.append_event(
            kind="note",
            meta={
                "event": "agents_resolved",
                "implementer": implementer_name,
                "reviewer": reviewer_name,
                "source": agent_source,
            },
        )
        if self.options.team_mode:
            if self.options.team_mode != "planning":
                raise RunnerError(f"unsupported team mode: {self.options.team_mode}")
            if not self._prepare_planning_team_mode(
                implementer_name, reviewer_name
            ):
                return 1
            if self._run_serial_loop(implementer_name, reviewer_name):
                return 1
            max_concurrency = 1
        else:
            max_concurrency = self._parallel_max_concurrency()

        if max_concurrency > 1:
            if not self._iteration_lock_held():
                msg = (
                    "parallel execution requires the parent iteration lock "
                    "to be held"
                )
                self.state.append_event(
                    kind="note",
                    meta={"event": "stop_global", "reason": STOP_INTERNAL,
                          "msg": msg},
                )
                self._log(f"ERROR: {msg}")
                return 1
            try:
                validate_conflict_tokens(self.board)
            except ValueError as exc:
                self.state.append_event(
                    kind="note",
                    meta={"event": "stop_global", "reason": STOP_CONFIG,
                          "msg": str(exc)},
                )
                self._log(f"ERROR: {exc}")
                return 1
            if self._run_parallel_loop(
                implementer_name, reviewer_name, max_concurrency
            ):
                return 1
        elif self._run_serial_loop(implementer_name, reviewer_name):
            return 1

        # Loop done — summarize.
        skipped = [
            t for t in self.state.tasks.values()
            if t.status.startswith(STATUS_STOPPED_PREFIX)
            or t.status == STATUS_IN_PROGRESS
        ]
        if skipped:
            for t in skipped:
                hint = (
                    "likely interrupted mid-task; "
                    if t.status == STATUS_IN_PROGRESS
                    else ""
                )
                self._log(
                    f"NOTE: {t.id} is {t.status} — resume does not "
                    f"re-run it; {hint}to re-run: python -m "
                    f"orch retry {self.iteration} {t.id}"
                )
            self.state.append_event(
                kind="note",
                meta={
                    "event": "resume_skipped_tasks",
                    "tasks": [
                        {"id": t.id, "status": t.status}
                        for t in skipped
                    ],
                    "msg": (
                        "these tasks are not re-run by run/resume; use "
                        "'orch retry <iter> <task>' to reset one to WAITING"
                    ),
                },
            )
        any_blocker = any(
            t.status.startswith(STATUS_STOPPED_PREFIX)
            or t.status == STATUS_NEEDS_HUMAN_MERGE
            or t.status == STATUS_BLOCKED_UPSTREAM
            or t.status == STATUS_IN_PROGRESS
            for t in self.state.tasks.values()
        )
        if not any_blocker:
            final_scope_msg = self._final_scope_gate()
            if final_scope_msg is not None:
                self.state.mark_iteration_finished(verdict="HALTED")
                self.state.append_event(
                    kind="note",
                    meta={
                        "event": "orch_workdir_preserved",
                        "msg": f"orch workdir preserved at {self.orch_cwd}",
                    },
                )
                return 1
            nav_gate_msg = self._final_nav_discoverability_gate()
            if nav_gate_msg is not None:
                self.state.mark_iteration_finished(verdict="HALTED")
                self.state.append_event(
                    kind="note",
                    meta={
                        "event": "orch_workdir_preserved",
                        "msg": f"orch workdir preserved at {self.orch_cwd}",
                    },
                )
                return 1
        verdict = "PARTIAL" if any_blocker else "READY"
        self.state.mark_iteration_finished(verdict=verdict)
        if any_blocker:
            self.state.append_event(
                kind="note",
                meta={
                    "event": "orch_workdir_preserved",
                    "msg": f"orch workdir preserved at {self.orch_cwd}",
                },
            )
        else:
            cleanup_orch_workdir(
                self.repo_root,
                self.iteration,
                worktree_root=self.orch_paths.worktree_root,
            )
        return 1 if any_blocker else 0


    def _prepare_planning_team_mode(
        self,
        implementer_name: str,
        reviewer_name: str,
    ) -> bool:
        if not self._iteration_lock_held():
            msg = "planning-team mode requires the parent iteration lock"
            self.state.append_event(
                kind="note",
                meta={
                    "event": "stop_global",
                    "reason": STOP_INTERNAL,
                    "msg": msg,
                },
            )
            self._log(f"ERROR: {msg}")
            return False

        impl_family = self.adapters[implementer_name].family
        reviewer_family = self.adapters[reviewer_name].family
        if impl_family == reviewer_family:
            msg = (
                "planning-team review independence policy is cross-vendor; "
                f"implementer '{implementer_name}' and reviewer "
                f"'{reviewer_name}' both use family '{impl_family}'"
            )
            self.state.append_event(
                kind="note",
                meta={
                    "event": "stop_global",
                    "reason": STOP_INDEPENDENCE,
                    "policy": PLANNING_REVIEW_INDEPENDENCE_POLICY,
                    "msg": msg,
                },
            )
            self._log(f"ERROR: {msg}")
            return False

        try:
            for task in self.board.tasks:
                validate_planning_candidate(
                    PlanningTeamCandidate.from_task(
                        task,
                        prompt="",
                        artifact_root=self.state.log_dir / "planning_team",
                    )
                )
        except PlanningTeamRefusal as exc:
            msg = str(exc)
            self.state.append_event(
                kind="note",
                meta={
                    "event": "stop_global",
                    "reason": STOP_CONFIG,
                    "msg": msg,
                },
            )
            self._log(f"ERROR: {msg}")
            return False

        self._planning_team_serial_fallback = has_overlapping_allowed_files(
            self.board.tasks
        )
        self.state.append_event(
            kind="note",
            meta={
                "event": "planning_team_mode_enabled",
                "policy": PLANNING_REVIEW_INDEPENDENCE_POLICY,
                "serialized": self._planning_team_serial_fallback,
            },
        )
        if self._planning_team_serial_fallback:
            self.state.append_event(
                kind="note",
                meta={
                    "event": "planning_team_serialized",
                    "reason": "overlapping allowed files",
                    "msg": (
                        "planning-team candidates overlap; using existing "
                        "non-team serial runner path"
                    ),
                },
            )
        return True


    def _iteration_lock_held(self) -> bool:
        lock = self.deps.run_lock
        return bool(lock is not None and getattr(lock, "acquired", False))

    def _parallel_error(self, message: str) -> Exception:
        return RunnerError(message)

    def _parallel_child_runner(self, **kwargs):
        return IterationRunner(**kwargs)

    def _pull_iter_branch_ff(
        self, *, context: str, task: str | None = None
    ) -> bool:
        """Fast-forward the local iter branch, surfacing (not swallowing) failure.

        A non-zero ``git pull --ff-only`` means the local branch could not be
        advanced to its remote (diverged, no upstream, or network). It is left
        non-fatal — local-only / no-CI flows may legitimately have no remote —
        but B-14 R5: it is recorded and warned so a stale local branch is never
        silently relied upon.
        """
        result = pull_ff_only(self.deps.cwd)
        if not result.ok:
            self.state.append_event(
                kind="note",
                task=task,
                meta={
                    "event": "pull_ff_only_failed",
                    "context": context,
                    "branch": self.iter_branch,
                    "exit_code": result.exit_code,
                    "stderr": (result.stderr or "").strip()[-500:],
                },
            )
            self._log(
                f"WARNING: git pull --ff-only on {self.iter_branch} failed "
                f"({context}); proceeding on the local branch — verify it is "
                "not stale. See pull_ff_only_failed in run_state.json."
            )
        return result.ok

    def _run_serial_loop(self, implementer_name: str, reviewer_name: str) -> bool:
        while True:
            task = self._pick_next_ready()
            if task is None:
                return False
            if self._run_serial_task(task, implementer_name, reviewer_name):
                return True

    def _run_serial_task(
        self, task: Task, implementer_name: str, reviewer_name: str
    ) -> bool:
        self._log(f">>> Task {task.id}: {task.title}")
        try:
            self._execute_task(task, implementer_name, reviewer_name)
            self._log(f"<<< Task {task.id}: DONE")
        except _TaskStopped as stop:
            # v2.2 — one-shot model-pair swap on REVIEW_FAIL only.
            swap_stop = self._maybe_swap_retry(
                task, stop, implementer_name, reviewer_name,
            )
            if swap_stop is None:
                self._log(f"<<< Task {task.id}: DONE (after pair swap)")
                return False
            return self._after_task_stop(task, swap_stop.reason)
        except Exception as exc:
            self._record_internal_stop(task, exc)
            return self._after_task_stop(task, STOP_INTERNAL)
        return False

    # ------------------------------------------------------------------
    # Final accumulated scope gate
    # ------------------------------------------------------------------

    def _final_scope_gate(self) -> str | None:
        """Block READY if the final iteration diff leaks outside task scope."""
        base_ref = self._expected_iteration_base_ref()
        if base_ref is None:
            decision = unresolved_final_scope_base_ref_decision(
                iter_branch=self.iter_branch,
                tasks_md_path=self.tasks_md_rel,
                stop_reason=STOP_SCOPE,
            )
            return self._record_final_scope_decision(decision)

        # Fail closed when the resolved base ref would make the
        # base...iter_branch diff vacuous (missing ref, or the base already
        # contains the iteration branch after a merge). Otherwise an empty
        # changed-file set passes the scope gate silently.
        freshness = classify_branch_freshness(
            self.deps.cwd, branch=self.iter_branch, base_ref=base_ref
        )
        if not final_scope_base_ref_yields_diff(freshness):
            decision = unresolved_final_scope_base_ref_decision(
                iter_branch=self.iter_branch,
                tasks_md_path=self.tasks_md_rel,
                stop_reason=STOP_SCOPE,
                detail=(
                    f"base ref '{base_ref}' is {freshness.condition.value}; "
                    "the iteration branch is not strictly ahead of it, so the "
                    "scope diff would be vacuous"
                ),
            )
            return self._record_final_scope_decision(decision)

        checkout(self.deps.cwd, self.iter_branch)
        changed = diff_files(self.deps.cwd, base_ref, self.iter_branch)
        tasks_md_status_only = False
        if self.tasks_md_rel in changed:
            tasks_md_status_only = self._tasks_md_final_status_update_only(
                base_ref, self.iter_branch
            )

        allowed = self.board.allowed_file_union
        evidence_path = self.state.log_dir / "scope_exceptions.md"
        evidence = load_scope_exception_evidence(evidence_path)
        decision = evaluate_final_outward_scope_gate(
            base_ref=base_ref,
            changed_files=changed,
            allowed_files=allowed,
            tasks_md_path=self.tasks_md_rel,
            tasks_md_status_only=tasks_md_status_only,
            evidence=evidence,
            stop_reason=STOP_SCOPE,
            generated_artifact_prefixes=(
                self.orch_paths.generated_artifact_exclusion_prefixes
            ),
        )
        return self._record_final_scope_decision(decision)

    def _record_final_scope_decision(
        self, decision: FinalScopeGateDecision,
    ) -> str | None:
        if decision.failure_meta is not None:
            self.state.append_event(
                kind="note",
                meta=decision.failure_meta,
            )
            return decision.message
        if decision.exception_meta is not None:
            self.state.append_event(kind="note", meta=decision.exception_meta)
        if decision.passed_meta is not None:
            self.state.append_event(kind="note", meta=decision.passed_meta)
        return None

    def _tasks_md_final_status_update_only(self, base_ref: str, head: str) -> bool:
        base_text = self._read_ref_file(base_ref, self.tasks_md_rel)
        head_text = self._read_ref_file(head, self.tasks_md_rel)
        if base_text is None or head_text is None:
            return False
        return check_tasks_md_status_only(
            base_text, head_text, [task.id for task in self.board.tasks]
        )

    def _read_ref_file(self, ref: str, rel_path: str) -> str | None:
        res = git(["show", f"{ref}:{rel_path}"], cwd=self.deps.cwd)
        return res.stdout if res.ok else None

    # ------------------------------------------------------------------
    # Final nav-discoverability gate (inward-gap rule)
    # ------------------------------------------------------------------

    def _final_nav_discoverability_gate(self) -> str | None:
        """Block READY when route-visible surfaces lack a nav update or
        an explicit operator-approved no-nav decision.

        Mirrors :meth:`_final_scope_gate` for the inward counterpart:
        outward leak watches files that escape the allowed-file union;
        inward gap watches user-facing surfaces that escape the nav. The
        gate is path-pattern-driven so it stays deterministic and does
        not parse application code or templates.
        """
        base_ref = self._expected_iteration_base_ref()
        if base_ref is None:
            decision = unresolved_final_nav_discoverability_base_ref_decision(
                iter_branch=self.iter_branch,
            )
            return self._record_final_nav_decision(decision)

        # Fail closed on a vacuous base...iter diff (missing ref, or base
        # already contains the iteration branch), mirroring the scope gate.
        freshness = classify_branch_freshness(
            self.deps.cwd, branch=self.iter_branch, base_ref=base_ref
        )
        if not final_scope_base_ref_yields_diff(freshness):
            decision = unresolved_final_nav_discoverability_base_ref_decision(
                iter_branch=self.iter_branch,
                detail=(
                    f"base ref '{base_ref}' is {freshness.condition.value}; "
                    "the iteration branch is not strictly ahead of it, so the "
                    "nav diff would be vacuous"
                ),
            )
            return self._record_final_nav_decision(decision)

        checkout(self.deps.cwd, self.iter_branch)
        changed = diff_files(self.deps.cwd, base_ref, self.iter_branch)
        visibility_cfg = self.cfg.data.get("ui_route_visibility", {})
        gate_input = final_nav_discoverability_gate_input(
            changed,
            generated_artifact_prefixes=(
                self.orch_paths.generated_artifact_exclusion_prefixes
            ),
            route_globs=visibility_cfg.get("route_globs"),
            nav_anchor_paths=visibility_cfg.get("nav_anchor_paths"),
        )
        evidence = None
        if gate_input.requires_evidence:
            evidence_path = self.state.log_dir / "nav_discoverability.md"
            evidence = load_nav_discoverability_evidence(evidence_path)

        decision = evaluate_final_nav_discoverability_gate(
            base_ref=base_ref,
            gate_input=gate_input,
            evidence=evidence,
        )
        return self._record_final_nav_decision(decision)

    def _record_final_nav_decision(
        self,
        decision: FinalNavDiscoverabilityGateDecision,
    ) -> str | None:
        if decision.failure_meta is not None:
            self.state.append_event(kind="note", meta=decision.failure_meta)
            return decision.message
        if decision.exception_meta is not None:
            self.state.append_event(kind="note", meta=decision.exception_meta)
        if decision.passed_meta is not None:
            self.state.append_event(kind="note", meta=decision.passed_meta)
        return None

    # ------------------------------------------------------------------
    # v2.2 — bounded model-pair swap on REVIEW_FAIL
    # ------------------------------------------------------------------

    def _maybe_swap_retry(
        self,
        task: Task,
        stop: "_TaskStopped",
        primary_impl: str,
        primary_rev: str,
    ) -> "_TaskStopped | None":
        """Retry ``task`` once with swapped roles if review convergence failed.

        Returns ``None`` if the swap succeeded (task ended DONE or
        NEEDS_HUMAN_MERGE). Otherwise returns the terminal ``_TaskStopped``
        (either the original, when the swap is not eligible, or the one
        raised by the swapped run). The caller treats the returned stop as
        the task's final outcome.
        """
        # Gate 1 — only swap on REVIEW_FAIL. Other stop reasons are either
        # structural (SCOPE/STRUCTURAL/IMPL_*), environmental (CHECKS,
        # PREFLIGHT_SIZE), or already terminal in a way swapping can't fix
        # (REVIEW_MALFORMED, INDEPENDENCE, NEEDS_HUMAN_MERGE).
        if stop.reason != STOP_REVIEW_FAIL:
            return stop
        # Gate 2 — one swap per task.
        ts = self.state.tasks.get(task.id)
        if ts is None or ts.swap_attempted:
            return stop
        # Gate 3 — swap needs a distinct-family adapter pair available.
        swap_impl, swap_rev = primary_rev, primary_impl
        if swap_impl not in self.adapters or swap_rev not in self.adapters:
            return stop
        i_fam = self.adapters[swap_impl].family
        r_fam = self.adapters[swap_rev].family
        level = self.options.independence or self.cfg.data.get(
            "independence", {}
        ).get("level", "model_family")
        ind = check_independence(
            i_fam, r_fam, level,
            implementer_name=swap_impl, reviewer_name=swap_rev,
        )
        if not ind.ok:
            return stop

        self._log(
            f"  v2.2: swapping model pair for {task.id} "
            f"({primary_impl}/{primary_rev} -> {swap_impl}/{swap_rev}); "
            f"reason={stop.reason}"
        )
        self.state.append_event(
            kind=EVT_PAIR_SWAP,
            task=task.id,
            meta={
                "primary_implementer": primary_impl,
                "primary_reviewer": primary_rev,
                "swap_implementer": swap_impl,
                "swap_reviewer": swap_rev,
                "reason": stop.reason,
            },
        )
        # Record swap fields + mark swap_attempted before running so state
        # is auditable even if the swap itself crashes.
        self.state.task_meta(
            task.id,
            swap_attempted=True,
            swap_implementer=swap_impl,
            swap_reviewer=swap_rev,
            swap_reason=stop.reason,
        )
        # Reset counters so the swapped run starts clean. reset_task clears
        # impl_attempts / fix_rounds / review_fix_rounds / stop_reason /
        # stop_msg but preserves the swap_* fields set above.
        self.state.reset_task(task.id)
        # Reset the task branch to iter-branch HEAD — drop the primary-pair
        # commits so the swapped implementer starts from a clean base.
        checkout(self.deps.cwd, self.iter_branch)
        create_or_reset_branch(self.deps.cwd, task.branch, self.iter_branch)

        try:
            self._execute_task(task, swap_impl, swap_rev)
        except _TaskStopped as swap_stop:
            self.state.task_meta(task.id, swap_outcome=swap_stop.reason)
            return swap_stop
        self.state.task_meta(task.id, swap_outcome="PASS")
        return None

    # ------------------------------------------------------------------
    # Global set-up helpers
    # ------------------------------------------------------------------

    def _recorded_agent_pair(self) -> tuple[str, str] | None:
        for event in reversed(self.state.events):
            if (
                event.get("kind") == "note"
                and (event.get("meta") or {}).get("event") == "agents_resolved"
            ):
                meta = event.get("meta") or {}
                impl = meta.get("implementer")
                reviewer = meta.get("reviewer")
                if impl and reviewer:
                    return str(impl), str(reviewer)

        for task in self.board.tasks:
            state = self.state.tasks.get(task.id)
            if state and state.implementer and state.reviewer:
                return state.implementer, state.reviewer
        return None

    def _resolve_agents(self) -> tuple[str, str, str]:
        recorded = self._recorded_agent_pair()
        flags_given = bool(self.options.implementer or self.options.reviewer)
        if not flags_given:
            if recorded is not None:
                impl, reviewer = recorded
                source = "run_state"
            else:
                impl = _first_name(self.cfg.data["agents"])
                reviewer = _pick_reviewer(self.cfg.data["agents"], impl)
                source = "config_default"
        else:
            impl = (
                self.options.implementer
                or (recorded[0] if recorded is not None else "")
                or _first_name(self.cfg.data["agents"])
            )
            reviewer = (
                self.options.reviewer
                or (recorded[1] if recorded is not None else "")
                or _pick_reviewer(self.cfg.data["agents"], impl)
            )
            source = "flags"
            if recorded is not None and (impl, reviewer) != recorded:
                if not self.options.override_agents:
                    rec_impl, rec_reviewer = recorded
                    raise RunnerError(
                        "resume: agents differ from the recorded pair "
                        f"(recorded: implementer={rec_impl}, "
                        f"reviewer={rec_reviewer}; requested: "
                        f"implementer={impl}, reviewer={reviewer}). "
                        "Pass --override-agents to switch agents deliberately."
                    )
                source = "flags_override"

        for name in (impl, reviewer):
            if name not in self.adapters:
                raise RunnerError(
                    f"adapter '{name}' not configured; available: "
                    f"{sorted(self.adapters)}"
                )
        return impl, reviewer, source

    def _check_global_independence(self, impl: str, reviewer: str):
        level = self.options.independence or \
            self.cfg.data.get("independence", {}).get("level", "model_family")
        i_fam = self.adapters[impl].family
        r_fam = self.adapters[reviewer].family
        return check_independence(
            i_fam, r_fam, level,
            implementer_name=impl, reviewer_name=reviewer,
        )

    # ------------------------------------------------------------------
    # DAG walker
    # ------------------------------------------------------------------

    def _pick_next_ready(self) -> Task | None:
        return pick_next_ready_task(self.board, self.state)

    def _mark_downstream_blocked(self, stopped_id: str) -> None:
        mark_downstream_blocked(self.board, self.state, stopped_id)

    def _after_task_stop(self, task: Task, reason: str) -> bool:
        self._log(f"<<< Task {task.id}: STOPPED ({reason})")
        if self.options.stop_on_first_failure and reason != "NEEDS_HUMAN_MERGE":
            self.state.mark_iteration_finished(verdict="HALTED")
            return True
        self._mark_downstream_blocked(task.id)
        if self.options.stop_on_first_failure:
            self.state.mark_iteration_finished(verdict="HALTED")
            return True
        return False

    def _record_internal_stop(self, task: Task, exc: Exception) -> None:
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        msg = (
            f"unexpected internal error in task {task.id}: "
            f"{exc_type}: {exc_msg}"
        )
        msg = append_recovery_note(
            msg,
            stop_reason_recovery_note(STOP_INTERNAL, iteration=self.iteration),
        )
        self.state.task_transition(
            task.id,
            STATUS_STOPPED_PREFIX + STOP_INTERNAL,
            reason=STOP_INTERNAL,
            msg=msg,
        )
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "internal_error",
                "exception_type": exc_type,
                "msg": exc_msg,
            },
        )

    # ------------------------------------------------------------------
    # Model routing
    # ------------------------------------------------------------------

    def _resolve_model_routing(self, task: Task) -> ResolvedModelRouting:
        routing = self._model_routing.get(task.id)
        if routing is None:
            routing = resolve_model_routing(task.model_routing)
            self._model_routing[task.id] = routing
        return routing

    def _model_routing_meta(self, task: Task) -> dict:
        return routing_to_dict(self._resolve_model_routing(task))

    def _model_routing_cost_extra(self, task: Task) -> dict:
        return {"model_routing": self._model_routing_meta(task)}

    def _cost_usage_kwargs(
        self,
        result,
        *,
        agent_name: str,
        adapter: AgentAdapter,
        routing_options: AgentInvocationOptions | None = None,
    ) -> dict:
        return cost_record_usage_kwargs(
            result,
            family=adapter.family,
            provider=agent_name,
            model=dispatched_model_from_options(routing_options),
        )

    def _cost_extra(self, base: dict, result) -> dict:
        out = dict(base)
        result_extra = getattr(result, "extra", None)
        if result_extra:
            out["agent_result_extra"] = dict(result_extra)
        raw = getattr(result, "raw_terminal_json", None)
        if raw is not None:
            out["raw_terminal_json"] = raw
        return out

    def _agent_invocation_options(
        self, agent_name: str, task: Task
    ) -> AgentInvocationOptions:
        return resolve_agent_routing_options(
            self.cfg.data.get("model_routing"),
            agent_name=agent_name,
            routing=self._resolve_model_routing(task),
        )

    def _record_model_routing(self, task: Task) -> ResolvedModelRouting:
        routing = self._resolve_model_routing(task)
        routing_meta = self._model_routing_meta(task)
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={"event": "model_routing_resolved", **routing_meta},
        )
        if routing.unknown_risk and task.id not in self._unknown_risk_warned:
            append_unknown_risk_warning(
                self.state.log_dir,
                iteration=self.iteration,
                task_id=task.id,
                routing=routing,
            )
            self._unknown_risk_warned.add(task.id)
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "model_routing_unknown_risk",
                    "artifact": self.orch_paths.artifact_ref(
                        self.iteration,
                        ROUTING_WARNINGS_FILENAME,
                    ),
                    "msg": (
                        "risk_category is unknown; model routing used "
                        "strong/medium defaults"
                    ),
                },
            )
        return routing

    def _timeouts_for_task(self, task: Task, tier: Tier) -> dict:
        resolved = dict(timeouts_for_tier(tier, self.cfg.data["timeouts"]))
        if task.task_kind is None:
            return resolved
        profiles = self.cfg.data["timeouts"].get("task_kind_profiles") or {}
        profile = profiles.get(task.task_kind)
        if profile is None:
            self._stop(
                task,
                STOP_CONFIG,
                f"unknown task_kind timeout profile {task.task_kind!r}",
            )
        resolved.update(profile)
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "task_kind_timeout_profile_applied",
                "task_kind": task.task_kind,
                "overrides": dict(profile),
                "timeouts": dict(resolved),
            },
        )
        return resolved

    # ------------------------------------------------------------------
    # Per-task state machine
    # ------------------------------------------------------------------


    def _execute_task(
        self, task: Task, implementer: str, reviewer: str
    ) -> None:
        if (
            self.options.team_mode == "planning"
            and not self._planning_team_serial_fallback
        ):
            self._execute_planning_task(task, implementer, reviewer)
            return
        self._execute_task_standard(task, implementer, reviewer)

    def _execute_task_standard(
        self, task: Task, implementer: str, reviewer: str
    ) -> None:
        model_routing = self._record_model_routing(task)
        self._emit_blocking_hook(
            task,
            "task.before_start",
            implementer=implementer,
            reviewer=reviewer,
            task_branch=task.branch,
            allowed_files=task.allowed_files,
            model_routing=routing_to_dict(model_routing),
        )
        self.state.task_transition(task.id, STATUS_IN_PROGRESS)
        self.state.task_meta(
            task.id, branch=task.branch, implementer=implementer,
            reviewer=reviewer,
        )

        prompt_text = self._load_prompt(task)

        # Step 0 — preflight
        pf = estimate(
            allowed_files=task.allowed_files,
            prompt_text=prompt_text,
            preflight_cfg=self.cfg.data["preflight"],
        )
        if pf.refused:
            self._stop(task, STOP_PREFLIGHT, "; ".join(pf.refuse_reasons))
        timeouts = self._timeouts_for_task(task, pf.tier)

        # Step 1 — prep branch
        self._emit_blocking_hook(
            task,
            "task.before_branch_prepare",
            iter_branch=self.iter_branch,
            task_branch=task.branch,
            allowed_files=task.allowed_files,
        )
        fetch(self.deps.cwd)
        checkout(self.deps.cwd, self.iter_branch)
        self._pull_iter_branch_ff(context="task branch prep", task=task.id)
        expected_base_ref = self._expected_iteration_base_ref()
        if expected_base_ref is not None:
            self._enforce_branch_contains_base(
                task,
                branch=self.iter_branch,
                base_ref=expected_base_ref,
                gate="task start",
            )
        base_sha = current_sha(self.deps.cwd, self.iter_branch)
        if task.id in self.options.skip_impl_tasks:
            self._enforce_branch_contains_base(
                task,
                branch=task.branch,
                base_ref=self.iter_branch,
                gate="skip-impl",
            )
            checkout(self.deps.cwd, task.branch)
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "skip_impl_branch_preserved",
                    "branch": task.branch,
                    "sha": current_sha(self.deps.cwd),
                },
            )
        else:
            create_or_reset_branch(self.deps.cwd, task.branch, self.iter_branch)

        # Step 2 — implement
        if task.id not in self.options.skip_impl_tasks:
            self._implement(task, implementer, prompt_text, timeouts, pf.tier)

        # Step 3 — deterministic checks + acceptance with fix loop
        self._checks_with_fix_loop(task, implementer, timeouts, base_sha)

        # Step 4 — AI review
        verdict = self._review_with_fix_loop(
            task, implementer, reviewer, timeouts, base_sha
        )
        self._dual_review_gate(task, reviewer, verdict, timeouts, base_sha)

        # Step 5/6 — PR + guarded merge (dry-run short-circuits)
        if self.options.dry_run:
            # P0-1: merge task branch into iter_branch locally so downstream
            # tasks see upstream code even in dry-run mode.
            self._pre_merge_freshness_gates(task)
            self._log(f"  dry-run: merging {task.branch} into {self.iter_branch}")
            self._merge_task_branch_locally(
                task,
                message=f"Merge {task.branch} into {self.iter_branch} (dry-run)",
            )
            self.state.task_transition(task.id, STATUS_DONE)
            self._write_tasks_md_done(task)
            return

        self._pr_and_merge(task, verdict, base_sha, timeouts)
        if self.state.tasks[task.id].status == STATUS_DONE:
            self._write_tasks_md_done(task)

    def _execute_planning_task(
        self, task: Task, implementer: str, reviewer: str
    ) -> None:
        model_routing = self._record_model_routing(task)
        self._emit_blocking_hook(
            task,
            "task.before_start",
            implementer=implementer,
            reviewer=reviewer,
            task_branch=task.branch,
            allowed_files=task.allowed_files,
            model_routing=routing_to_dict(model_routing),
            team_mode="planning",
        )
        self.state.task_transition(task.id, STATUS_IN_PROGRESS)
        self.state.task_meta(
            task.id, branch=task.branch, implementer=implementer,
            reviewer=reviewer,
        )

        prompt_text = self._load_prompt(task)
        pf = estimate(
            allowed_files=task.allowed_files,
            prompt_text=prompt_text,
            preflight_cfg=self.cfg.data["preflight"],
        )
        if pf.refused:
            self._stop(task, STOP_PREFLIGHT, "; ".join(pf.refuse_reasons))
        timeouts = self._timeouts_for_task(task, pf.tier)

        self._emit_blocking_hook(
            task,
            "task.before_branch_prepare",
            iter_branch=self.iter_branch,
            task_branch=task.branch,
            allowed_files=task.allowed_files,
            team_mode="planning",
        )
        fetch(self.deps.cwd)
        checkout(self.deps.cwd, self.iter_branch)
        self._pull_iter_branch_ff(context="planning-team task branch prep",
                                  task=task.id)
        expected_base_ref = self._expected_iteration_base_ref()
        if expected_base_ref is not None:
            self._enforce_branch_contains_base(
                task,
                branch=self.iter_branch,
                base_ref=expected_base_ref,
                gate="planning-team task start",
            )
        base_sha = current_sha(self.deps.cwd, self.iter_branch)
        create_or_reset_branch(self.deps.cwd, task.branch, self.iter_branch)

        result = self._planning_team_implement(
            task,
            implementer,
            prompt_text,
            timeout=timeouts["impl"],
        )
        adapter = self.adapters[implementer]
        extra = self._model_routing_cost_extra(task)
        extra.update(
            {
                "team_mode": "planning",
                "cost_estimate": not result.tokens_exact,
                "planning_review_independence_policy": (
                    PLANNING_REVIEW_INDEPENDENCE_POLICY
                ),
            }
        )
        self.cost.record(
            task=task.id, step="IMPL", agent=implementer,
            **cost_record_usage_kwargs(
                result,
                family=adapter.family,
                provider=implementer,
            ),
            duration_s=result.duration_s,
            exit_code=result.exit_code if result.exit_code is not None else -1,
            partial=not result.ok,
            extra=self._cost_extra(extra, result),
        )
        self.state.record_impl_attempt_end(
            task.id,
            agent=implementer,
            exit_code=result.exit_code if result.exit_code is not None else -1,
            duration_s=result.duration_s,
            classification=None if result.ok else result.status,
            stderr_tail=result.error or "",
        )
        if not result.ok:
            self._stop(
                task,
                STOP_IMPL_FAILED,
                result.error or result.text or "planning-team agent failed",
            )

        git(["add", "--", *task.allowed_files], cwd=self.deps.cwd, check=True)
        commit_res = commit(
            self.deps.cwd, f"{task.id}: planning-team checkpoint"
        )
        if not commit_res.ok:
            self._stop(
                task,
                STOP_IMPL_FAILED,
                "planning-team produced no committable declared output",
            )

        self._checks_with_fix_loop(task, implementer, timeouts, base_sha)
        verdict = self._review_with_fix_loop(
            task, implementer, reviewer, timeouts, base_sha
        )
        self._dual_review_gate(task, reviewer, verdict, timeouts, base_sha)

        if self.options.dry_run:
            self._pre_merge_freshness_gates(task)
            self._log(
                f"  dry-run: merging {task.branch} into {self.iter_branch}"
            )
            self._merge_task_branch_locally(
                task,
                message=f"Merge {task.branch} into {self.iter_branch} (dry-run)",
            )
            self.state.task_transition(task.id, STATUS_DONE)
            self._write_tasks_md_done(task)
            return

        self._pr_and_merge(task, verdict, base_sha, timeouts)
        if self.state.tasks[task.id].status == STATUS_DONE:
            self._write_tasks_md_done(task)

    def _planning_team_implement(
        self,
        task: Task,
        implementer: str,
        prompt_text: str,
        *,
        timeout: int,
    ):
        candidate = PlanningTeamCandidate.from_task(
            task,
            prompt=prompt_text,
            artifact_root=self.state.log_dir / "planning_team",
        )
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "planning_team_spawn",
                "allowed_files": list(task.allowed_files),
                "agent": implementer,
            },
        )
        try:
            results = run_planning_team(
                team_name=f"planning-{self.iteration}-{task.id}",
                candidates=[candidate],
                cwd=self.deps.cwd,
                command=_agent_command(self.cfg, implementer),
                timeout=timeout,
            )
        except PlanningTeamRefusal as exc:
            self._stop(task, STOP_SCOPE, str(exc))
        except PlanningTeamError as exc:
            self._stop(task, STOP_IMPL_FAILED, str(exc))
        result = results[0]
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "planning_team_complete",
                "status": result.status,
                "changed_files": list(result.changed_files),
                "artifact_dir": str(result.artifact_dir),
            },
        )
        return result

    # ---- step 2 -------------------------------------------------------

    def _salvage_leftovers(self, task: Task) -> str | None:
        """Best-effort salvage of uncommitted impl leftovers."""
        branch = f"salvage/{self.iteration}/{task.id}"
        sha = salvage_worktree(
            self.deps.cwd,
            branch,
            f"orch salvage: {task.id} uncommitted work at terminal stop",
        )
        if sha is None:
            return None
        self._log(f"  salvaged uncommitted work to {branch} ({sha[:7]})")
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={"event": "impl_salvage", "branch": branch, "sha": sha},
        )
        return branch

    def _implement(
        self,
        task: Task,
        implementer: str,
        prompt_text: str,
        timeouts: dict,
        tier: Tier,
    ) -> None:
        full_prompt = compose_prompt(task.allowed_files, prompt_text)
        limits_cfg = self.cfg.data["limits"]
        adapter = self.adapters[implementer]
        self._log(f"  implement: agent={implementer}, max_attempts={limits_cfg['impl_attempts']}")
        max_attempts = int(limits_cfg["impl_attempts"])
        partials = 0
        for attempt in range(1, max_attempts + 1):
            self._emit_blocking_hook(
                task,
                "task.before_implement",
                implementer=implementer,
                attempt=attempt,
                max_attempts=int(limits_cfg["impl_attempts"]),
                tier=tier.value,
                timeout=timeouts["impl"],
                allowed_files=task.allowed_files,
                model_routing=self._model_routing_meta(task),
            )
            routing_options = self._agent_invocation_options(
                implementer, task
            )
            result = adapter.invoke(
                full_prompt,
                timeout=timeouts["impl"],
                workdir=self.deps.cwd,
                routing_options=routing_options,
            )
            self.cost.record(
                task=task.id, step="IMPL", agent=implementer,
                **self._cost_usage_kwargs(
                    result,
                    agent_name=implementer,
                    adapter=adapter,
                    routing_options=routing_options,
                ),
                duration_s=result.duration_s,
                exit_code=result.exit_code,
                partial=result.partial,
                extra=self._cost_extra(
                    self._model_routing_cost_extra(task),
                    result,
                ),
            )
            classification: str | None = None
            tail: str | None = None
            if result.exit_code != 0:
                classification = _classify_impl_failure(
                    result.stderr, result.exit_code
                )
                # Always record a tail string on failure (empty string when
                # stderr is empty) so meta consistently carries both T40
                # diagnostics for every IMPL_FAILED event.
                tail = _stderr_tail(result.stderr)
            self.state.record_impl_attempt_end(
                task.id, agent=implementer, exit_code=result.exit_code,
                duration_s=result.duration_s,
                classification=classification,
                stderr_tail=tail,
            )
            if result.partial:
                # Timeout counts as a retry, not a fix round (per C4).
                partials += 1
                continue
            if result.exit_code != 0:
                if result.output_tokens == 0 and result.duration_s < 60:
                    self.state.append_event(
                        kind="note",
                        task=task.id,
                        meta={
                            "msg": (
                                "impl_fast_exit: suspected rate-limit or "
                                "config error"
                            ),
                            "exit_code": result.exit_code,
                            "stderr_tail": (
                                result.stderr[-500:] if result.stderr else ""
                            ),
                        },
                    )
                stop_msg = (
                    f"adapter exited {result.exit_code} (attempt {attempt})"
                )
                if _rate_limit_signature(result.stderr):
                    stop_msg = f"{stop_msg}\n\n{_RATE_LIMIT_HINT}"
                stop_msg = append_recovery_note(
                    stop_msg,
                    impl_failure_recovery_note(
                        classification, iteration=self.iteration
                    ),
                )
                salvage_branch = self._salvage_leftovers(task)
                if salvage_branch is not None:
                    stop_msg = (
                        f"{stop_msg}\n\nUncommitted work preserved for "
                        f"inspection on branch '{salvage_branch}' — it is "
                        "NOT reviewed or resumed automatically."
                    )
                self._stop(task, STOP_IMPL_FAILED, stop_msg)
            # If the implementer produced a commit, diff_stats will show it.
            stats = diff_stats(self.deps.cwd, self.iter_branch)
            if stats.files > 0 or stats.insertions > 0:
                return
            # No commit — stage + commit what's in the worktree. Adapters
            # that commit themselves make this a no-op; otherwise it gives
            # the orchestrator a clean commit to inspect.
            stage_all(self.deps.cwd)
            commit_res = commit(
                self.deps.cwd, f"{task.id}: implementer checkpoint"
            )
            if commit_res.ok:
                return
        # Ran out of attempts.
        if not _any_commit(self.deps.cwd, self.iter_branch):
            if partials == max_attempts:
                msg = (
                    f"implementation timed out on all {max_attempts} "
                    f"attempts (impl timeout {timeouts['impl']}s, "
                    f"tier {tier.value}); no commit produced"
                )
                reason = STOP_IMPL_TIMEOUT
            else:
                msg = f"no commit produced after {max_attempts} attempts"
                if partials:
                    msg = (
                        f"{msg} ({partials} of {max_attempts} attempts "
                        "timed out)"
                    )
                msg = append_recovery_note(
                    msg,
                    impl_failure_recovery_note(
                        "unknown", iteration=self.iteration
                    ),
                )
                reason = STOP_IMPL_FAILED
            salvage_branch = self._salvage_leftovers(task)
            if salvage_branch is not None:
                msg = (
                    f"{msg}\n\nUncommitted work preserved for inspection "
                    f"on branch '{salvage_branch}' — it is NOT reviewed "
                    "or resumed automatically."
                )
            self._stop(task, reason, msg)

    # ---- step 3 -------------------------------------------------------

    def _check_scope_gate(
        self, task: Task, base_sha: str, *, auto_revert: bool,
    ) -> None:
        """Run the scope check at any pipeline point."""
        if not auto_revert:
            changed = [
                p for p in diff_files(self.deps.cwd, base_sha)
                if not self.orch_paths.is_generated_artifact_path(p)
            ]
            outside = check_scope(changed, task.allowed_files)
            tasks_md_touched = check_tasks_md_touched(
                changed, self.tasks_md_rel
            )
            if outside or tasks_md_touched:
                self._stop(
                    task, STOP_SCOPE,
                    f"scope violation in fix/pre-PR path "
                    f"(outside={outside}, tasks_md={tasks_md_touched})",
                )
            return

        for attempt in range(
            int(self.cfg.data["limits"]["scope_auto_revert"]) + 1
        ):
            changed = [
                p for p in diff_files(self.deps.cwd, base_sha)
                if not self.orch_paths.is_generated_artifact_path(p)
            ]
            outside = check_scope(changed, task.allowed_files)
            tasks_md_touched = check_tasks_md_touched(
                changed, self.tasks_md_rel
            )
            if not outside and not tasks_md_touched:
                return
            if attempt >= int(self.cfg.data["limits"]["scope_auto_revert"]):
                self._stop(
                    task, STOP_SCOPE,
                    f"scope violation (outside={outside}, "
                    f"tasks_md={tasks_md_touched})",
                )
            # Auto-revert the outside files (and tasks.md if touched).
            victims = list(outside)
            if tasks_md_touched:
                victims.append(self.tasks_md_rel)
            revert_paths(self.deps.cwd, victims, base_sha)
            stage_all(self.deps.cwd)
            commit(self.deps.cwd, f"{task.id}: auto-revert out-of-scope")
            self.state.append_event(
                kind="scope_revert", task=task.id,
                meta={"files": victims},
            )
            # A revert that wipes the entire diff almost always means the
            # allowed_files list is malformed (e.g. arrow-comment residue)
            # rather than a real scope violation. Bail out instead of
            # feeding an empty diff into review.
            post_stats = diff_stats(self.deps.cwd, base_sha)
            if post_stats.insertions == 0:
                self._stop(
                    task, STOP_SCOPE,
                    f"scope revert produced empty diff (reverted "
                    f"{len(victims)} file(s): {victims}); check "
                    f"allowed_files in tasks.md",
                )

    def _checks_with_fix_loop(
        self,
        task: Task,
        implementer: str,
        timeouts: dict,
        base_sha: str,
    ) -> None:
        self._structural_checks(task, base_sha)

        limits_cfg = self.cfg.data["limits"]
        max_fix = int(limits_cfg["fix_rounds_acceptance"])
        for round_num in range(max_fix + 1):
            report = run_acceptance(
                self.cfg.data["stack"],
                cwd=self.deps.cwd,
                timeout=timeouts["acceptance"],
                test_cmd_override=task.test_cmd,
            )
            if report.ok:
                return
            if round_num >= max_fix:
                self._stop(
                    task, STOP_CHECKS,
                    f"acceptance failing after {max_fix} fix rounds",
                )
            # Fixer invocation (narrow input per C4).
            self._fix(
                task, implementer, report.combined_output(), base_sha=base_sha,
                cause="acceptance",
                timeout=timeouts["fix"],
            )

    def _structural_checks(self, task: Task, base_sha: str) -> None:
        self._check_scope_gate(task, base_sha, auto_revert=True)

        # Structural content checks.
        text = diff_text(self.deps.cwd, base_sha)
        changed = [
            p for p in diff_files(self.deps.cwd, base_sha)
            if not self.orch_paths.is_generated_artifact_path(p)
        ]
        stats = diff_stats(self.deps.cwd, base_sha)
        hard_limit, diff_cap_override = self._effective_diff_hard_limit(task)
        if diff_cap_override is not None:
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "diff_cap_override_applied",
                    "scope": diff_cap_override.scope,
                    "max_diff_insertions_hard": (
                        diff_cap_override.max_diff_insertions_hard
                    ),
                    "approved_by": diff_cap_override.approved_by,
                    "evidence": diff_cap_override.evidence,
                    "source_line": diff_cap_override.line,
                    "default_max_diff_insertions_hard": int(
                        self.cfg.data["limits"]["max_diff_insertions_hard"]
                    ),
                },
            )
        if check_diff_size(stats.insertions, hard_limit):
            msg = f"diff insertions {stats.insertions} over hard cap {hard_limit}"
            if diff_cap_override is not None:
                msg += (
                    " (diff cap override "
                    f"scope={diff_cap_override.scope}, "
                    f"approved_by={diff_cap_override.approved_by}, "
                    f"evidence={diff_cap_override.evidence})"
                )
            self._stop(task, STOP_STRUCTURAL,
                       msg)
        if _diff_introduces_conflict_marker_pair(text):
            self._stop(task, STOP_STRUCTURAL, "conflict markers in diff")
        patterns = self.cfg.data["risk"].get("forbidden_patterns") or []
        pat_hits = check_forbidden_patterns(text, patterns)
        if pat_hits:
            self._stop(task, STOP_STRUCTURAL,
                       f"forbidden patterns: {pat_hits}")
        sens = self.cfg.data["risk"].get("sensitive_files") or []
        sens_hits = check_sensitive_files(changed, sens)
        if sens_hits:
            self._stop(task, STOP_STRUCTURAL,
                       f"sensitive files touched: {sens_hits}")
        self.state.task_meta(task.id, diff_insertions=stats.insertions)

    def _effective_diff_hard_limit(
        self, task: Task
    ) -> tuple[int, DiffCapOverride | None]:
        override = task.diff_cap_override or self.board.diff_cap_override
        if override is None:
            return int(self.cfg.data["limits"]["max_diff_insertions_hard"]), None
        return override.max_diff_insertions_hard, override

    # _fix threads base_sha so post-fix scope checks diff against task base.
    def _fix(
        self,
        task: Task,
        implementer: str,
        failure_payload: str,
        *,
        base_sha: str,
        cause: str,
        timeout: int,
    ) -> None:
        prefix_prompt = compose_prompt(task.allowed_files, "")
        prompt = (
            f"{prefix_prompt}"
            f"The following check failed. Fix only this failure — do not "
            f"introduce unrelated changes.\n\n{failure_payload}\n"
        )
        adapter = self.adapters[implementer]
        self._emit_blocking_hook(
            task,
            "task.before_fix",
            implementer=implementer,
            cause=cause,
            timeout=timeout,
            allowed_files=task.allowed_files,
            failure_excerpt=failure_payload[:2000],
            model_routing=self._model_routing_meta(task),
        )
        routing_options = self._agent_invocation_options(implementer, task)
        result = adapter.invoke(
            prompt,
            timeout=timeout,
            workdir=self.deps.cwd,
            routing_options=routing_options,
        )
        self.cost.record(
            task=task.id, step="FIX", agent=implementer,
            **self._cost_usage_kwargs(
                result,
                agent_name=implementer,
                adapter=adapter,
                routing_options=routing_options,
            ),
            duration_s=result.duration_s,
            exit_code=result.exit_code,
            partial=result.partial,
            cause=cause,
            extra=self._cost_extra(
                self._model_routing_cost_extra(task),
                result,
            ),
        )
        self.state.record_fix_round_end(
            task.id, cause=cause, agent=implementer,
            exit_code=result.exit_code, duration_s=result.duration_s,
        )
        # Stage+commit fixer's changes if it didn't commit itself.
        stage_all(self.deps.cwd)
        commit(self.deps.cwd, f"{task.id}: fix ({cause})")
        # Scope gate — fix rounds must not silently expand scope.
        self._check_scope_gate(task, base_sha, auto_revert=False)

    # ---- step 4 -------------------------------------------------------

    def _iteration_path_in_workdir(self) -> Path:
        try:
            rel = self.board.path.parent.relative_to(self.deps.repo_root)
        except ValueError:
            return self.board.path.parent
        return self.deps.cwd / rel

    def _review_prompt_candidates(self, task: Task) -> list[Path]:
        reviews_dir = self.orch_paths.task_reviews_dir(
            self._iteration_path_in_workdir()
        )
        return review_prompt_candidates(reviews_dir, task)

    def _load_review_prompt_contract(self, task: Task) -> tuple[str | None, Path | None]:
        reviews_dir = self.orch_paths.task_reviews_dir(
            self._iteration_path_in_workdir()
        )
        candidates = self._review_prompt_candidates(task)
        return load_review_prompt_contract(reviews_dir, candidates)

    def _build_task_review_prompt(
        self,
        task: Task,
        *,
        fresh_diff: str,
        base_sha: str,
        reviewer_role: str,
        round_num: int,
        max_rounds: int,
        primary_reviewer: str | None = None,
    ) -> str:
        contract, contract_path = self._load_review_prompt_contract(task)
        prompt, stop_msg = build_task_review_prompt_text(
            task=task,
            contract=contract,
            contract_path=contract_path,
            candidates=self._review_prompt_candidates(task),
            cwd=self.deps.cwd,
            fresh_diff=fresh_diff,
            base_sha=base_sha,
            reviewer_role=reviewer_role,
            round_num=round_num,
            max_rounds=max_rounds,
            primary_reviewer=primary_reviewer,
        )
        if stop_msg is not None:
            self._stop(task, STOP_STRUCTURAL, stop_msg)
        return prompt

    def _review_with_fix_loop(
        self,
        task: Task,
        implementer: str,
        reviewer: str,
        timeouts: dict,
        base_sha: str,
    ) -> Verdict:
        max_rounds = int(self.cfg.data["limits"]["review_rounds"])
        regex = self.cfg.data["review"]["verdict_regex"]
        adapter = self.adapters[reviewer]

        for round_num in range(1, max_rounds + 1):
            self._log(f"  review: round {round_num}/{max_rounds}, agent={reviewer}")
            fresh_diff = diff_text(self.deps.cwd, base_sha)
            prompt = self._build_task_review_prompt(
                task,
                fresh_diff=fresh_diff,
                base_sha=base_sha,
                reviewer_role="primary",
                round_num=round_num,
                max_rounds=max_rounds,
            )
            self._emit_blocking_hook(
                task,
                "task.before_review",
                reviewer=reviewer,
                round=round_num,
                max_rounds=max_rounds,
                timeout=timeouts["review"],
                diff_insertions=diff_stats(self.deps.cwd, base_sha).insertions,
                allowed_files=task.allowed_files,
                model_routing=self._model_routing_meta(task),
            )
            routing_options = self._agent_invocation_options(
                reviewer, task
            )
            result = adapter.invoke(
                prompt,
                timeout=timeouts["review"],
                workdir=self.deps.cwd,
                routing_options=routing_options,
            )
            # Persist full review stdout regardless of verdict so operators
            # can diagnose PASS/CHANGES REQUIRED/BLOCKED/MALFORMED outcomes
            # without re-running the review.
            reviews_dir = self.state.log_dir / "reviews"
            reviews_dir.mkdir(parents=True, exist_ok=True)
            artifact = reviews_dir / self._review_artifact_filename(
                task.id, round_num
            )
            artifact.write_text(result.stdout)
            self.cost.record(
                task=task.id, step="REVIEW", agent=reviewer,
                **self._cost_usage_kwargs(
                    result,
                    agent_name=reviewer,
                    adapter=adapter,
                    routing_options=routing_options,
                ),
                duration_s=result.duration_s,
                exit_code=result.exit_code,
                partial=result.partial,
                extra=self._cost_extra(
                    self._model_routing_cost_extra(task),
                    result,
                ),
            )
            # Guard: partial (timed-out) review must not drive verdict.
            if result.partial:
                self.state.record_review_result(
                    task.id, reviewer=reviewer, verdict="MALFORMED",
                    round_num=round_num,
                )
                self._stop(
                    task, STOP_REVIEW_MALFORMED,
                    append_recovery_note(
                        "reviewer timed out (partial output) — verdict "
                        "not parsed",
                        stop_reason_recovery_note(
                            STOP_REVIEW_MALFORMED,
                            iteration=self.iteration,
                            review_artifact=self._review_artifact_path(
                                task.id, round_num
                            ),
                        ),
                    ),
                )
            parsed = parse_verdict(result.stdout, regex)
            self._log(f"  review verdict: {parsed.verdict.value if not parsed.malformed else 'MALFORMED'}")
            if parsed.malformed:
                self.state.record_review_result(
                    task.id, reviewer=reviewer, verdict="MALFORMED",
                    round_num=round_num,
                )
                self._stop(
                    task, STOP_REVIEW_MALFORMED,
                    append_recovery_note(
                        parsed.message,
                        stop_reason_recovery_note(
                            STOP_REVIEW_MALFORMED,
                            iteration=self.iteration,
                            review_artifact=self._review_artifact_path(
                                task.id, round_num
                            ),
                        ),
                    ),
                )
            self.state.record_review_result(
                task.id, reviewer=reviewer, verdict=parsed.verdict.value,
                round_num=round_num,
            )
            findings_text = _extract_review_findings(result.stdout)
            # Record per-round confidence (None until reviewers emit a
            # structured value) and fingerprint the finding for
            # repeat-failure detection.
            self.state.record_confidence(
                task.id, round_num=round_num, value=None,
            )
            self._record_finding_fingerprint(
                task.id,
                round_num=round_num,
                fingerprint=_triage.fingerprint_findings(findings_text),
            )

            triage_input = self._build_triage_input(task, parsed)
            decision = _triage.classify(triage_input)
            self._log(
                f"  triage: action={decision.action} "
                f"reason={decision.reason!r}"
            )
            self.state.append_triage_decision(
                task.id,
                action=decision.action,
                reason=decision.reason,
                round_num=round_num,
                verdict=parsed.verdict.value,
                severity=parsed.severity,
                increments_defer_budget=decision.increments_defer_budget,
                confidence_history=list(triage_input.confidence_history),
            )

            if decision.action == _triage.ACTION_PROCEED:
                return parsed.verdict
            if decision.action == _triage.ACTION_DEFER_TO_QA:
                # Note the deferred finding for QA pickup; treat as accept.
                self.state.append_event(
                    kind="note",
                    task=task.id,
                    meta={
                        "event": "deferred_finding",
                        "round": round_num,
                        "verdict": parsed.verdict.value,
                        "severity": parsed.severity,
                        "reason": decision.reason,
                        "findings_excerpt": findings_text[:2000],
                    },
                )
                return parsed.verdict
            if decision.action == _triage.ACTION_STOP_HUMAN:
                msg = (
                    f"triage STOP_HUMAN at round {round_num}: "
                    f"{decision.reason} (verdict={parsed.verdict.value})"
                )
                self._stop(
                    task, STOP_REVIEW_FAIL,
                    append_recovery_note(
                        msg,
                        stop_reason_recovery_note(
                            STOP_REVIEW_FAIL,
                            iteration=self.iteration,
                            review_artifact=self._review_artifact_path(
                                task.id, round_num
                            ),
                        ),
                    ),
                )
            # ACTION_FIX_NOW falls through. Apply the existing
            # round-budget guard so review_rounds is still respected.
            if round_num >= max_rounds:
                msg = (
                    f"verdict={parsed.verdict.value} at round {round_num} "
                    "(triage FIX_NOW after exhausting review_rounds budget)"
                )
                self._stop(
                    task, STOP_REVIEW_FAIL,
                    append_recovery_note(
                        msg,
                        stop_reason_recovery_note(
                            STOP_REVIEW_FAIL,
                            iteration=self.iteration,
                            review_artifact=self._review_artifact_path(
                                task.id, round_num
                            ),
                        ),
                    ),
                )
            self._fix(
                task, implementer, findings_text,
                base_sha=base_sha, cause="review",
                timeout=timeouts["fix"],
            )
        # Fallthrough (shouldn't reach here).
        self._stop(
            task, STOP_REVIEW_FAIL,
            append_recovery_note(
                "review loop exhausted",
                stop_reason_recovery_note(
                    STOP_REVIEW_FAIL, iteration=self.iteration
                ),
            ),
        )

    def _secondary_reviewer_name(self) -> str:
        configured = self.options.secondary_reviewer or self.cfg.data.get(
            "review", {}
        ).get("secondary_reviewer", "")
        return str(configured or "").strip()

    def _dual_review_gate(
        self,
        task: Task,
        primary_reviewer: str,
        primary_verdict: Verdict,
        timeouts: dict,
        base_sha: str,
    ) -> None:
        routing = self._resolve_model_routing(task)
        if not routing.dual_model_required:
            return

        if primary_verdict != Verdict.PASS:
            msg = (
                "dual-model agreement requires primary reviewer PASS, but "
                f"primary reviewer '{primary_reviewer}' returned "
                f"Verdict: {primary_verdict.value}"
            )
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "dual_review_failed",
                    "primary_reviewer": primary_reviewer,
                    "risk_category": routing.risk_category,
                    "verdict": primary_verdict.value,
                    "msg": msg,
                },
            )
            self._stop(task, STOP_DUAL_REVIEW_FAIL, msg)

        secondary = self._secondary_reviewer_name()
        if not secondary:
            msg = (
                "dual-model agreement required for risk_category="
                f"{routing.risk_category}, but no secondary reviewer is "
                "configured"
            )
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "dual_review_required",
                    "primary_reviewer": primary_reviewer,
                    "risk_category": routing.risk_category,
                    "msg": msg,
                },
            )
            self._stop(task, STOP_DUAL_REVIEW_REQUIRED, msg)

        if secondary not in self.adapters:
            msg = (
                f"dual-model secondary reviewer '{secondary}' is not "
                f"configured; available reviewers: {sorted(self.adapters)}"
            )
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "dual_review_required",
                    "primary_reviewer": primary_reviewer,
                    "secondary_reviewer": secondary,
                    "risk_category": routing.risk_category,
                    "msg": msg,
                },
            )
            self._stop(task, STOP_DUAL_REVIEW_REQUIRED, msg)

        primary_family = self.adapters[primary_reviewer].family
        secondary_adapter = self.adapters[secondary]
        if secondary == primary_reviewer:
            msg = (
                "dual-model secondary reviewer must be different from "
                f"primary reviewer '{primary_reviewer}'"
            )
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "dual_review_required",
                    "primary_reviewer": primary_reviewer,
                    "secondary_reviewer": secondary,
                    "risk_category": routing.risk_category,
                    "msg": msg,
                },
            )
            self._stop(task, STOP_DUAL_REVIEW_REQUIRED, msg)
        if secondary_adapter.family == primary_family:
            msg = (
                "dual-model secondary reviewer must use a different model "
                f"family than primary reviewer '{primary_reviewer}' "
                f"(family={primary_family})"
            )
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "dual_review_required",
                    "primary_reviewer": primary_reviewer,
                    "secondary_reviewer": secondary,
                    "primary_family": primary_family,
                    "secondary_family": secondary_adapter.family,
                    "risk_category": routing.risk_category,
                    "msg": msg,
                },
            )
            self._stop(task, STOP_DUAL_REVIEW_REQUIRED, msg)

        self._run_secondary_review(
            task,
            primary_reviewer=primary_reviewer,
            secondary_reviewer=secondary,
            timeout=timeouts["review"],
            base_sha=base_sha,
        )

    def _run_secondary_review(
        self,
        task: Task,
        *,
        primary_reviewer: str,
        secondary_reviewer: str,
        timeout: int,
        base_sha: str,
    ) -> None:
        adapter = self.adapters[secondary_reviewer]
        regex = self.cfg.data["review"]["verdict_regex"]
        fresh_diff = diff_text(self.deps.cwd, base_sha)
        prompt = self._build_task_review_prompt(
            task,
            fresh_diff=fresh_diff,
            base_sha=base_sha,
            reviewer_role="secondary",
            round_num=1,
            max_rounds=1,
            primary_reviewer=primary_reviewer,
        )
        self._emit_blocking_hook(
            task,
            "task.before_review",
            reviewer=secondary_reviewer,
            primary_reviewer=primary_reviewer,
            dual_review=True,
            round=1,
            max_rounds=1,
            timeout=timeout,
            diff_insertions=diff_stats(self.deps.cwd, base_sha).insertions,
            allowed_files=task.allowed_files,
            model_routing=self._model_routing_meta(task),
        )
        routing_options = self._agent_invocation_options(
            secondary_reviewer, task
        )
        result = adapter.invoke(
            prompt,
            timeout=timeout,
            workdir=self.deps.cwd,
            routing_options=routing_options,
        )
        reviews_dir = self.state.log_dir / "reviews"
        reviews_dir.mkdir(parents=True, exist_ok=True)
        artifact = reviews_dir / f"dual_review_{task.id}_{secondary_reviewer}.md"
        artifact.write_text(result.stdout)
        artifact_ref = self._dual_review_artifact_path(
            task.id, secondary_reviewer
        )
        extra = self._model_routing_cost_extra(task)
        extra.update(
            {
                "review_role": "secondary",
                "primary_reviewer": primary_reviewer,
            }
        )
        self.cost.record(
            task=task.id, step="REVIEW", agent=secondary_reviewer,
            **self._cost_usage_kwargs(
                result,
                agent_name=secondary_reviewer,
                adapter=adapter,
                routing_options=routing_options,
            ),
            duration_s=result.duration_s,
            exit_code=result.exit_code,
            partial=result.partial,
            extra=self._cost_extra(extra, result),
        )
        if result.partial:
            self._dual_review_stop(
                task,
                STOP_DUAL_REVIEW_MALFORMED,
                event="dual_review_malformed",
                primary_reviewer=primary_reviewer,
                secondary_reviewer=secondary_reviewer,
                artifact=artifact_ref,
                msg=(
                    "secondary reviewer timed out (partial output); "
                    "verdict not parsed"
                ),
            )
        if result.exit_code != 0:
            self._dual_review_stop(
                task,
                STOP_DUAL_REVIEW_FAIL,
                event="dual_review_failed",
                primary_reviewer=primary_reviewer,
                secondary_reviewer=secondary_reviewer,
                artifact=artifact_ref,
                msg=(
                    f"secondary reviewer '{secondary_reviewer}' exited "
                    f"{result.exit_code}"
                ),
            )

        parsed = parse_verdict(result.stdout, regex)
        if parsed.malformed:
            self._dual_review_stop(
                task,
                STOP_DUAL_REVIEW_MALFORMED,
                event="dual_review_malformed",
                primary_reviewer=primary_reviewer,
                secondary_reviewer=secondary_reviewer,
                artifact=artifact_ref,
                msg=parsed.message,
            )
        if parsed.verdict != Verdict.PASS:
            self._dual_review_stop(
                task,
                STOP_DUAL_REVIEW_FAIL,
                event="dual_review_failed",
                primary_reviewer=primary_reviewer,
                secondary_reviewer=secondary_reviewer,
                artifact=artifact_ref,
                msg=(
                    "secondary reviewer returned "
                    f"Verdict: {parsed.verdict.value}"
                ),
                verdict=parsed.verdict.value,
            )

        routing = self._resolve_model_routing(task)
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "dual_review_passed",
                "primary_reviewer": primary_reviewer,
                "secondary_reviewer": secondary_reviewer,
                "primary_family": self.adapters[primary_reviewer].family,
                "secondary_family": self.adapters[secondary_reviewer].family,
                "risk_category": routing.risk_category,
                "artifact": artifact_ref,
                "verdict": parsed.verdict.value,
            },
        )

    def _dual_review_stop(
        self,
        task: Task,
        stop_reason: str,
        *,
        event: str,
        primary_reviewer: str,
        secondary_reviewer: str,
        artifact: str,
        msg: str,
        verdict: str | None = None,
    ) -> None:
        routing = self._resolve_model_routing(task)
        meta = {
            "event": event,
            "primary_reviewer": primary_reviewer,
            "secondary_reviewer": secondary_reviewer,
            "risk_category": routing.risk_category,
            "artifact": artifact,
            "msg": msg,
        }
        if verdict is not None:
            meta["verdict"] = verdict
        self.state.append_event(kind="note", task=task.id, meta=meta)
        self._stop(
            task,
            stop_reason,
            append_recovery_note(
                msg,
                stop_reason_recovery_note(
                    stop_reason,
                    iteration=self.iteration,
                    review_artifact=artifact,
                ),
            ),
        )

    def _review_artifact_path(self, task_id: str, round_num: int) -> str:
        return self.orch_paths.artifact_ref(
            self.iteration,
            f"reviews/{self._review_artifact_filename(task_id, round_num)}",
        )

    def _dual_review_artifact_path(
        self, task_id: str, secondary_reviewer: str
    ) -> str:
        return self.orch_paths.artifact_ref(
            self.iteration,
            f"reviews/dual_review_{task_id}_{secondary_reviewer}.md",
        )

    def _review_artifact_filename(self, task_id: str, round_num: int) -> str:
        assert task_id, "review artifacts must be keyed by task id"
        assert round_num >= 1, "review artifact round numbers are 1-based"
        return f"review_{task_id}_r{round_num}.md"

    # ---- triage helpers -----------------------------------------------

    def _record_finding_fingerprint(
        self, task_id: str, *, round_num: int, fingerprint: str,
    ) -> None:
        """Append a finding fingerprint for the task's review history.

        Stored in-memory on the runner; the triage classifier reads the
        last 3 entries to detect repeat-failure. Persistence is
        indirect via the `triage` event log — the fingerprint shape
        itself doesn't need round-trip semantics.
        """
        self._finding_fingerprints.setdefault(task_id, []).append(fingerprint)

    def _build_triage_input(
        self, task: Task, parsed,
    ) -> "_triage.TriageInput":
        """Compose the classifier inputs from current task state +
        latest parsed verdict."""
        ts = self.state.tasks.get(task.id)
        defer_used = ts.defer_budget_used if ts else 0
        # structural failures already raised would have stopped the task
        # via _stop, so this flag is normally False inside the review
        # loop. We keep it wired to support defensive callers.
        structural_failure = bool(
            ts and ts.stop_reason in (
                STOP_SCOPE, STOP_STRUCTURAL, STOP_PREFLIGHT,
            )
        )
        confidence_history = tuple(ts.confidence_history) if ts else ()
        recent = tuple(self._finding_fingerprints.get(task.id, []))
        limits_cfg = self.cfg.data.get("limits", {})
        return _triage.TriageInput(
            verdict=parsed.verdict.value,
            severity=parsed.severity,
            structural_failure=structural_failure,
            defer_budget_used=defer_used,
            defer_budget_max=int(
                limits_cfg.get("defer_budget", _triage.DEFAULT_DEFER_BUDGET_MAX)
            ),
            recent_findings=recent,
            confidence_history=confidence_history,
            confidence_drop_threshold=float(
                limits_cfg.get(
                    "confidence_drop", _triage.DEFAULT_CONFIDENCE_DROP_THRESH
                )
            ),
        )

    # ---- steps 5/6/7 --------------------------------------------------


    def _pre_merge_freshness_gates(self, task: Task) -> None:
        fetch(self.deps.cwd)
        expected_base_ref = self._expected_iteration_base_ref()
        if expected_base_ref is not None:
            self._enforce_branch_contains_base(
                task,
                branch=self.iter_branch,
                base_ref=expected_base_ref,
                gate="merge",
            )
        self._enforce_branch_contains_base(
            task,
            branch=task.branch,
            base_ref=self.iter_branch,
            gate="merge",
        )

    def _apply_task_branch_merge_locally(self, task: Task, *, message: str) -> str:
        checkout(self.deps.cwd, self.iter_branch)
        merge_no_ff(self.deps.cwd, task.branch, message=message)
        merge_sha = current_sha(self.deps.cwd, self.iter_branch)
        self.state.set_iter_branch_sha(merge_sha)
        return merge_sha

    def _merge_task_branch_locally(self, task: Task, *, message: str) -> str:
        return self._apply_task_branch_merge_locally(task, message=message)

    def _record_local_merge_intent(
        self, task: Task, *, message: str, mode: str,
    ) -> dict:
        return self.state.record_merge_intent(
            task.id,
            target_branch=self.iter_branch,
            target_sha_before=current_sha(self.deps.cwd, self.iter_branch),
            task_branch=task.branch,
            task_sha=current_sha(self.deps.cwd, task.branch),
            message=message,
            mode=mode,
        )["meta"]

    def _noop_acceptance_gate_allows_no_ci_merge(self, task: Task) -> bool:
        stack = self.cfg.data["stack"]
        if not acceptance_test_command_is_noop(
            stack,
            test_cmd_override=task.test_cmd,
        ):
            return True

        test_cmd = effective_acceptance_test_command(
            stack,
            test_cmd_override=task.test_cmd,
        )
        reason = (self.options.allow_noop_acceptance_reason or "").strip()
        if reason:
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "noop_acceptance_allowed",
                    "test_cmd": test_cmd or "",
                    "reason": reason,
                },
            )
            return True

        msg = (
            "no-CI local merge blocked because the effective acceptance test "
            f"command is a no-op ({test_cmd!r}); pass "
            "--allow-noop-acceptance <reason> to acknowledge the manual "
            "acceptance evidence for this run."
        )
        self.state.task_transition(
            task.id,
            STATUS_NEEDS_HUMAN_MERGE,
            reason="NOOP_ACCEPTANCE",
            msg=msg,
        )
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "noop_acceptance_blocked",
                "test_cmd": test_cmd or "",
                "msg": msg,
            },
        )
        return False

    def _pr_and_merge(
        self, task: Task, verdict: Verdict, base_sha: str, timeouts: dict,
    ) -> None:
        self._pre_merge_freshness_gates(task)
        # Pre-PR scope gate — near-PR diffs must not silently expand scope.
        self._check_scope_gate(task, base_sha, auto_revert=False)
        pre_pr_changed = diff_files(self.deps.cwd, base_sha)
        pre_pr_stats = diff_stats(self.deps.cwd, base_sha)
        self._emit_blocking_hook(
            task,
            "task.before_pr",
            verdict=verdict.value,
            task_branch=task.branch,
            iter_branch=self.iter_branch,
            changed_files=pre_pr_changed,
            diff_insertions=pre_pr_stats.insertions,
            allowed_files=task.allowed_files,
            model_routing=self._model_routing_meta(task),
        )
        pr_request = build_task_pr_request(
            task_id=task.id,
            task_title=task.title,
            iteration=self.iteration,
            iter_branch=self.iter_branch,
            task_branch=task.branch,
            allowed_files=task.allowed_files,
            test_cmd=task.test_cmd,
        )
        ok, url_or_err = self.deps.open_pr(
            cwd=self.deps.cwd,
            title=pr_request.title,
            body=pr_request.body,
            base=pr_request.base,
            head=pr_request.head,
        )
        if not ok:
            self.state.task_transition(task.id, STATUS_NEEDS_HUMAN_MERGE)
            self.state.append_event(
                kind="note", task=task.id,
                meta={"event": "pr_failed", "msg": url_or_err},
            )
            return
        pr_url = url_or_err
        self.state.record_pr(task.id, pr_url)

        auto_merge_cfg = self.cfg.data["auto_merge"]
        no_ci = bool(auto_merge_cfg.get("no_ci", False))

        # CI poll
        ci_passed = False if no_ci else True
        if self.options.poll_ci and not no_ci:
            status = self.deps.wait_for_ci(
                task.branch, cwd=self.deps.cwd,
                ci_wait_seconds=timeouts["ci"],
            )
            ci_passed = status.passed

        # Recompute facts for the guard.
        stats = diff_stats(self.deps.cwd, base_sha)
        changed = diff_files(self.deps.cwd, base_sha)
        sens = check_sensitive_files(
            changed, self.cfg.data["risk"].get("sensitive_files") or []
        )
        forb = check_forbidden_patterns(
            diff_text(self.deps.cwd, base_sha),
            self.cfg.data["risk"].get("forbidden_patterns") or [],
        )
        ts = self.state.tasks[task.id]

        decision = evaluate_auto_merge(
            verdict=verdict.value,
            changed_files=changed,
            sensitive_hits=sens,
            forbidden_hits=forb,
            diff_insertions=stats.insertions,
            fix_rounds=ts.fix_rounds + ts.review_fix_rounds,
            high_risk_globs=self.cfg.data["risk"].get("high_risk_globs") or [],
            auto_merge_cfg=auto_merge_cfg,
            ci_passed=ci_passed,
        )
        if not decision.should_auto_merge:
            self.deps.comment_pr(
                cwd=self.deps.cwd, pr_url=pr_url,
                body=render_guard_comment(
                    decision, no_ci=no_ci, iter_branch=self.iter_branch,
                ),
            )
            self.state.task_transition(task.id, STATUS_NEEDS_HUMAN_MERGE)
            self._emit_needs_human_merge_diagnostics(
                task, pr_url,
                ci_passed=ci_passed,
                decision_reasons=decision.reasons,
            )
            return

        if not no_ci:
            self.deps.comment_pr(
                cwd=self.deps.cwd, pr_url=pr_url,
                body=render_guard_comment(decision),
            )
        if no_ci and not self._noop_acceptance_gate_allows_no_ci_merge(task):
            return

        self._emit_blocking_hook(
            task,
            "task.before_merge",
            pr_url=pr_url,
            ci_passed=ci_passed,
            changed_files=changed,
            diff_insertions=stats.insertions,
            guard_reasons=decision.reasons,
            no_ci=no_ci,
            model_routing=self._model_routing_meta(task),
        )
        if no_ci:
            message = (
                f"Merge {task.branch} into {self.iter_branch} "
                "(no-CI local)"
            )
            try:
                intent_meta = self._record_local_merge_intent(
                    task, message=message, mode="no_ci_local",
                )
                merge = (
                    self._apply_task_branch_merge_locally
                    if self._local_merge_reconciled_this_run
                    else self._merge_task_branch_locally
                )
                merge_sha = merge(
                    task, message=message,
                )
            except GitError as exc:
                self.state.task_transition(task.id, STATUS_NEEDS_HUMAN_MERGE)
                self.state.append_event(
                    kind="note", task=task.id,
                    meta={"event": "local_merge_failed", "msg": str(exc)},
                )
                return
            self.state.record_merge_complete(
                task.id,
                target_branch=self.iter_branch,
                merge_sha=merge_sha,
                target_sha_before=str(intent_meta["target_sha_before"]),
                task_branch=task.branch,
                task_sha=str(intent_meta["task_sha"]),
                mode="no_ci_local",
            )
            self.state.record_merge(
                task.id, auto_merged=True, merge_sha=merge_sha,
            )
            self.state.task_transition(task.id, STATUS_DONE)
            self.deps.comment_pr(
                cwd=self.deps.cwd, pr_url=pr_url,
                body=render_guard_comment(
                    decision, no_ci=True, iter_branch=self.iter_branch,
                ),
            )
            return

        ok, msg = self.deps.merge_pr(cwd=self.deps.cwd, pr_url=pr_url)
        if not ok:
            self.state.task_transition(task.id, STATUS_NEEDS_HUMAN_MERGE)
            self.state.append_event(
                kind="note", task=task.id,
                meta={"event": "merge_failed", "msg": msg},
            )
            self._emit_needs_human_merge_diagnostics(
                task, pr_url, ci_passed=ci_passed,
            )
            return
        merge_sha = parse_merge_sha(msg)
        self.state.record_merge(
            task.id, auto_merged=True, merge_sha=merge_sha,
        )
        self.state.task_transition(task.id, STATUS_DONE)
        # refresh iter-branch SHA
        checkout(self.deps.cwd, self.iter_branch)
        self._pull_iter_branch_ff(
            context="post auto-merge SHA refresh", task=task.id
        )
        self.state.set_iter_branch_sha(
            current_sha(self.deps.cwd, self.iter_branch)
        )

    # NEEDS_HUMAN_MERGE hardening helpers
    # ----------------------------------------------------------------

    def _emit_needs_human_merge_diagnostics(
        self,
        task: Task,
        pr_url: str,
        *,
        ci_passed: bool,
        decision_reasons: list[str] | None = None,
    ) -> None:
        _finalization.emit_needs_human_merge_diagnostics(
            self,
            task,
            pr_url,
            ci_passed=ci_passed,
            decision_reasons=decision_reasons,
        )

    def _check_external_merges(self) -> None:
        _finalization.check_external_merges(self)

    def _reconcile_local_merges(self) -> None:
        _finalization.reconcile_local_merges(self)

    def _write_tasks_md_done(self, task: Task) -> None:
        _finalization.write_tasks_md_done(self, task)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expected_iteration_base_ref(self) -> str | None:
        phase_branch = self._resolve_phase_branch()
        if phase_branch is not None:
            return preferred_remote_ref(self.deps.cwd, phase_branch)
        return upstream_ref(self.deps.cwd, self.iter_branch)

    def _resolve_phase_branch(self) -> str | None:
        try:
            return resolve_phase_branch(self.cfg, self.iteration)
        except PhaseResolutionError as exc:
            raise RunnerError(str(exc)) from exc

    def _enforce_branch_contains_base(
        self,
        task: Task,
        *,
        branch: str,
        base_ref: str,
        gate: str,
    ) -> None:
        freshness = classify_branch_freshness(
            self.deps.cwd, branch=branch, base_ref=base_ref
        )
        if freshness.contains_base:
            return
        msg = render_branch_freshness_recovery(freshness, gate=gate)
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "branch_freshness_gate",
                "gate": gate,
                "branch": branch,
                "base_ref": base_ref,
                "condition": freshness.condition.value,
                "ahead_count": freshness.ahead_count,
                "behind_count": freshness.behind_count,
                "msg": msg,
            },
        )
        self._stop(task, STOP_BRANCH_FRESHNESS, msg)

    def _load_prompt(self, task: Task) -> str:
        # Prompt path convention: iterations/<...>/prompts/<slug>.md next to tasks.md
        base = task.branch.split("/")[-1] if "/" in task.branch else task.id
        prompts_dir = self.orch_paths.task_prompts_dir(self.board.path.parent)
        for candidate in [
            prompts_dir / f"{base}.md",
            prompts_dir / f"{task.id.lower()}.md",
        ]:
            if candidate.exists():
                return candidate.read_text()
        # Fall back to the task title — enough to feed a fake adapter in tests.
        return f"Task {task.id}: {task.title}\n"

    def _emit_blocking_hook(
        self, task: Task, event_name: str, **payload,
    ) -> None:
        if not self.state.hooks_enabled():
            return
        try:
            self.state.append_event(
                kind=EVT_HOOK,
                task=task.id,
                step="HOOK",
                meta={"event": event_name, **payload},
                hook_blocking=True,
            )
        except HookVeto as exc:
            msg = (
                f"hook veto at {event_name} from {exc.handler}: "
                f"{exc.message or exc.reason}"
            )
            self._stop(task, STOP_HOOK_VETO, msg)

    def _stop(self, task: Task, reason: str, msg: str) -> None:
        msg = append_recovery_note(
            msg,
            stop_reason_recovery_note(reason, iteration=self.iteration),
        )
        self.state.task_transition(
            task.id, STATUS_STOPPED_PREFIX + reason, reason=reason, msg=msg,
        )
        raise _TaskStopped(reason=reason)




# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _first_name(agents: dict) -> str:
    for name in agents:
        return name
    raise RunnerError("no agents configured")


def _pick_reviewer(agents: dict, implementer: str) -> str:
    i_fam = agents[implementer].get("family")
    # Prefer a different family; else any non-implementer.
    for name, spec in agents.items():
        if name == implementer:
            continue
        if spec.get("family") != i_fam:
            return name
    for name in agents:
        if name != implementer:
            return name
    raise RunnerError(
        "cannot pick reviewer distinct from implementer "
        f"'{implementer}' — only one agent configured"
    )


def _agent_command(cfg: LoadedConfig, agent_name: str) -> tuple[str, ...]:
    spec = cfg.data.get("agents", {}).get(agent_name, {})
    cmd = str(spec.get("cmd") or agent_name).strip()
    if not cmd:
        raise RunnerError(f"agent {agent_name!r} has no command")
    provider = str(spec.get("type") or spec.get("family") or agent_name)
    return tuple(ensure_usage_json_args(shlex.split(cmd), provider))


def _any_commit(cwd: Path, base: str) -> bool:
    try:
        return diff_stats(cwd, base).files > 0
    except Exception:
        return False
