"""Parallel execution mixin for the iteration runner."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from orch.checks import (
    check_diff_size,
    check_forbidden_patterns,
    check_scope,
    check_sensitive_files,
    check_tasks_md_touched,
)
from orch.cost import CostLogger
from orch.git_ops import (
    checkout,
    current_sha,
    diff_files,
    diff_stats,
    diff_text,
    ensure_task_workdir,
    fetch,
    GitError,
    merge_no_ff,
)
from orch.merge import (
    build_task_pr_request,
    evaluate_auto_merge,
    parse_merge_sha,
    render_guard_comment,
)
from orch.model_routing import routing_to_dict
from orch.parallel import plan_parallel_waves
from orch.preflight import estimate
from orch.review import Verdict
from orch.state import (
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_HUMAN_MERGE,
    StateStore,
)
from orch.stops import (
    STOP_INTERNAL,
    STOP_PREFLIGHT,
    STOP_SCOPE,
    STOP_STRUCTURAL,
    _TaskStopped,
)
from orch.task_execution import (
    diff_introduces_conflict_marker_pair as _diff_introduces_conflict_marker_pair,
)
from orch.tasks_schema import Task


@dataclass(frozen=True)
class _ParallelTaskResult:
    task: Task
    workdir: Path
    base_sha: str
    task_sha: str
    verdict: Verdict | None
    stop_reason: str | None
    events: tuple[dict, ...]
    cost_path: Path


class ParallelExecutionMixin:
    def _parallel_error(self, message: str) -> Exception:
        return RuntimeError(message)

    def _parallel_child_runner(self, **kwargs):
        raise RuntimeError("parallel child runner factory is not configured")

    def _parallel_max_concurrency(self) -> int:
        value = self.cfg.data.get("parallel", {}).get("max_concurrency", 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise self._parallel_error(
                "parallel.max_concurrency must be a positive integer"
            )
        return value

    def _run_parallel_loop(
        self,
        implementer_name: str,
        reviewer_name: str,
        max_concurrency: int,
    ) -> bool:
        while True:
            wave = self._next_parallel_wave(max_concurrency)
            if len(wave) < 2:
                task = self._pick_next_ready()
                if task is None:
                    return False
                if self._run_serial_task(task, implementer_name, reviewer_name):
                    return True
                continue

            wave_stopped = self._run_parallel_wave(
                wave, implementer_name, reviewer_name, max_concurrency
            )
            if wave_stopped:
                # T6: a wave failure stops later wave scheduling. Final
                # summary below will produce PARTIAL/READY truthfully.
                return False

    def _next_parallel_wave(self, max_concurrency: int) -> tuple[Task, ...]:
        live = {tid: state.status for tid, state in self.state.tasks.items()}
        waves = plan_parallel_waves(
            self.board,
            live_statuses=live,
            max_concurrency=max_concurrency,
        )
        if not waves:
            return ()
        skipped = set(self.options.skip_impl_tasks)
        return tuple(task for task in waves[0] if task.id not in skipped)

    def _emit_parallel_branch_prepare_hooks(
        self, wave: tuple[Task, ...]
    ) -> tuple[tuple[Task, ...], bool]:
        """Emit task.before_branch_prepare for each wave task (serial parity).

        Returns ``(active_wave, abort)``. A hook veto stops that task; if
        ``_after_task_stop`` signals a run-level halt, ``abort`` is True and the
        caller stops the wave. Vetoed tasks are dropped from ``active_wave``.
        With hooks disabled ``_emit_blocking_hook`` is a no-op, so this returns
        the wave unchanged.
        """
        active: list[Task] = []
        for task in wave:
            try:
                self._emit_blocking_hook(
                    task,
                    "task.before_branch_prepare",
                    iter_branch=self.iter_branch,
                    task_branch=task.branch,
                    allowed_files=task.allowed_files,
                )
            except _TaskStopped as stop:
                if self._after_task_stop(task, stop.reason):
                    return (), True
                continue
            active.append(task)
        return tuple(active), False

    def _run_parallel_wave(
        self,
        wave: tuple[Task, ...],
        implementer_name: str,
        reviewer_name: str,
        max_concurrency: int,
    ) -> bool:
        # Hook parity (QA-A3): emit the blocking task.before_branch_prepare hook
        # per task — like the serial path — before any branch/worktree prep. A
        # veto stops that task and drops it from the wave; survivors continue.
        # With hooks disabled this is a no-op, so the default path is unchanged.
        active_wave, abort = self._emit_parallel_branch_prepare_hooks(wave)
        if abort or not active_wave:
            return True
        wave = active_wave
        self._assert_parallel_review_artifacts_are_isolated(wave)
        self._prepare_parallel_wave(wave)
        wave_base_sha = current_sha(self.deps.cwd, self.iter_branch)
        workdirs = {
            task.id: ensure_task_workdir(
                self.repo_root,
                self.iteration,
                task.id,
                task.branch,
                self.iter_branch,
                worktree_root=self.orch_paths.worktree_root,
            )
            for task in wave
        }
        self.state.append_event(
            kind="note",
            meta={
                "event": "parallel_wave_started",
                "tasks": [task.id for task in wave],
                "max_concurrency": max_concurrency,
                "base_sha": wave_base_sha,
                "lock_held": self._iteration_lock_held(),
            },
        )

        results_by_task: dict[str, _ParallelTaskResult] = {}
        with ThreadPoolExecutor(
            max_workers=min(max_concurrency, len(wave))
        ) as pool:
            futures = {
                pool.submit(
                    self._run_parallel_child_task,
                    task,
                    implementer_name,
                    reviewer_name,
                    workdirs[task.id],
                    wave_base_sha,
                ): task
                for task in wave
            }
            for future in as_completed(futures):
                task = futures[future]
                results_by_task[task.id] = future.result()

        wave_stopped = False
        for task in sorted(wave, key=lambda item: item.index):
            result = results_by_task[task.id]
            self._replay_parallel_task_result(result)
            if result.stop_reason is not None:
                wave_stopped = True
                if self._after_task_stop(task, result.stop_reason):
                    return True
                continue
            try:
                finalized = self._finalize_parallel_task(result)
            except _TaskStopped as stop:
                wave_stopped = True
                if self._after_task_stop(task, stop.reason):
                    return True
                continue
            if finalized:
                self._log(f"<<< Task {task.id}: DONE (parallel wave)")
            else:
                wave_stopped = True

        self.state.append_event(
            kind="note",
            meta={
                "event": "parallel_wave_finished",
                "tasks": [task.id for task in wave],
                "stopped": wave_stopped,
            },
        )
        return wave_stopped

    def _prepare_parallel_wave(self, wave: tuple[Task, ...]) -> None:
        self._log(
            ">>> Parallel wave: "
            + ", ".join(f"{task.id}: {task.title}" for task in wave)
        )
        fetch(self.deps.cwd)
        checkout(self.deps.cwd, self.iter_branch)
        self._pull_iter_branch_ff(context="parallel wave prep")
        expected_base_ref = self._expected_iteration_base_ref()
        if expected_base_ref is not None:
            for task in wave:
                self._enforce_branch_contains_base(
                    task,
                    branch=self.iter_branch,
                    base_ref=expected_base_ref,
                    gate="parallel wave start",
                )

    def _assert_parallel_review_artifacts_are_isolated(
        self, wave: tuple[Task, ...]
    ) -> None:
        # Parallel children share one reviews directory. Primary review
        # artifacts must include task id + round so same-round child reviews
        # cannot overwrite each other.
        paths = [self._review_artifact_path(task.id, 1) for task in wave]
        if len(paths) != len(set(paths)):
            raise self._parallel_error(
                "parallel review artifact paths must be unique per task"
            )

    def _run_parallel_child_task(
        self,
        task: Task,
        implementer_name: str,
        reviewer_name: str,
        workdir: Path,
        base_sha: str,
    ) -> _ParallelTaskResult:
        cost_path = self.state.log_dir / "parallel" / f"{task.id}.cost.jsonl"
        try:
            cost_path.unlink()
        except FileNotFoundError:
            pass
        child_state = _BufferedStateStore(
            log_dir=self.state.log_dir,
            iteration=self.iteration,
            iter_branch=self.iter_branch,
            hook_dispatcher=self.state.hook_dispatcher,
        )
        child_cost = CostLogger(
            path=cost_path,
            cost_table=self.cfg.data["costs"],
            iteration=self.iteration,
        )
        child_deps = replace(
            self.deps,
            cwd=workdir,
        )
        child = self._parallel_child_runner(
            cfg=self.cfg,
            board=self.board,
            state=child_state,
            cost=child_cost,
            adapters=self.adapters,
            options=self.options,
            deps=child_deps,
        )
        try:
            verdict = child._execute_task_until_review(
                task, implementer_name, reviewer_name, base_sha
            )
            stop_reason = None
        except _TaskStopped as stop:
            verdict = None
            stop_reason = stop.reason
        except Exception as exc:
            verdict = None
            child._record_internal_stop(task, exc)
            stop_reason = STOP_INTERNAL
        return _ParallelTaskResult(
            task=task,
            workdir=workdir,
            base_sha=base_sha,
            task_sha=current_sha(workdir),
            verdict=verdict,
            stop_reason=stop_reason,
            events=tuple(child_state.events),
            cost_path=cost_path,
        )

    def _replay_parallel_task_result(self, result: _ParallelTaskResult) -> None:
        for event in result.events:
            self.state.append_event(
                kind=event.get("kind"),
                task=event.get("task"),
                step=event.get("step"),
                status=event.get("status"),
                meta=event.get("meta") or {},
                ts=event.get("ts"),
                dispatch_hooks=False,
            )
        if result.cost_path.exists():
            lines = result.cost_path.read_text().splitlines()
            if lines:
                self.cost.path.parent.mkdir(parents=True, exist_ok=True)
                with self.cost.path.open("a") as f:
                    for line in lines:
                        if line.strip():
                            f.write(line.rstrip() + "\n")

    def _execute_task_until_review(
        self,
        task: Task,
        implementer: str,
        reviewer: str,
        base_sha: str,
    ) -> Verdict:
        """Run a prepared task branch through review without merging it.

        Parallel children use this after the parent has created a dedicated
        task worktree rooted at the wave's iteration tip. Shared iteration
        branch mutation remains parent-owned.
        """
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
        pf = estimate(
            allowed_files=task.allowed_files,
            prompt_text=prompt_text,
            preflight_cfg=self.cfg.data["preflight"],
        )
        if pf.refused:
            self._stop(task, STOP_PREFLIGHT, "; ".join(pf.refuse_reasons))
        timeouts = self._timeouts_for_task(task, pf.tier)

        if task.id not in self.options.skip_impl_tasks:
            self._implement(task, implementer, prompt_text, timeouts, pf.tier)

        self._checks_with_fix_loop(task, implementer, timeouts, base_sha)
        verdict = self._review_with_fix_loop(
            task, implementer, reviewer, timeouts, base_sha
        )
        self._dual_review_gate(task, reviewer, verdict, timeouts, base_sha)
        return verdict

    def _finalize_parallel_task(self, result: _ParallelTaskResult) -> bool:
        task = result.task
        assert result.verdict is not None

        current_iter_sha = current_sha(self.deps.cwd, self.iter_branch)
        if current_iter_sha != result.base_sha:
            if not self._refresh_parallel_task_branch(result):
                return False

        self._parallel_structural_checks(task, result.base_sha, result.task_sha)
        self._parallel_pre_merge_freshness_gates(task)

        if self.options.dry_run:
            self._log(
                f"  dry-run: merging {task.branch} into {self.iter_branch} "
                "(parallel wave)"
            )
            self._merge_task_branch_locally(
                task,
                message=(
                    f"Merge {task.branch} into {self.iter_branch} "
                    "(parallel dry-run)"
                ),
            )
            self.state.task_transition(task.id, STATUS_DONE)
            self._write_tasks_md_done(task)
            return True

        return self._parallel_pr_and_merge(
            task,
            result.verdict,
            result.base_sha,
            result.task_sha,
        )

    def _refresh_parallel_task_branch(
        self, result: _ParallelTaskResult
    ) -> bool:
        task = result.task
        try:
            checkout(result.workdir, task.branch)
            merge_no_ff(
                result.workdir,
                self.iter_branch,
                message=(
                    f"Merge advanced {self.iter_branch} into {task.branch} "
                    "(parallel wave refresh)"
                ),
            )
        except GitError as exc:
            self.state.task_transition(task.id, STATUS_NEEDS_HUMAN_MERGE)
            self.state.append_event(
                kind="note",
                task=task.id,
                meta={
                    "event": "parallel_branch_refresh_failed",
                    "branch": task.branch,
                    "iter_branch": self.iter_branch,
                    "msg": str(exc),
                },
            )
            return False
        self.state.append_event(
            kind="note",
            task=task.id,
            meta={
                "event": "parallel_branch_refreshed",
                "branch": task.branch,
                "iter_branch": self.iter_branch,
                "sha": current_sha(result.workdir),
            },
        )
        return True

    def _parallel_pre_merge_freshness_gates(self, task: Task) -> None:
        fetch(self.deps.cwd)
        expected_base_ref = self._expected_iteration_base_ref()
        if expected_base_ref is not None:
            self._enforce_branch_contains_base(
                task,
                branch=self.iter_branch,
                base_ref=expected_base_ref,
                gate="parallel merge",
            )
        self._enforce_branch_contains_base(
            task,
            branch=task.branch,
            base_ref=self.iter_branch,
            gate="parallel merge",
        )

    def _parallel_structural_checks(
        self, task: Task, base_sha: str, head_sha: str
    ) -> None:
        changed = [
            p for p in diff_files(self.deps.cwd, base_sha, head_sha)
            if not self.orch_paths.is_generated_artifact_path(p)
        ]
        outside = check_scope(changed, task.allowed_files)
        tasks_md_touched = check_tasks_md_touched(changed, self.tasks_md_rel)
        if outside or tasks_md_touched:
            self._stop(
                task,
                STOP_SCOPE,
                f"scope violation in parallel pre-merge path "
                f"(outside={outside}, tasks_md={tasks_md_touched})",
            )

        text = diff_text(self.deps.cwd, base_sha, head_sha)
        stats = diff_stats(self.deps.cwd, base_sha, head_sha)
        hard_limit, diff_cap_override = self._effective_diff_hard_limit(task)
        if check_diff_size(stats.insertions, hard_limit):
            msg = f"diff insertions {stats.insertions} over hard cap {hard_limit}"
            if diff_cap_override is not None:
                msg += (
                    " (diff cap override "
                    f"scope={diff_cap_override.scope}, "
                    f"approved_by={diff_cap_override.approved_by}, "
                    f"evidence={diff_cap_override.evidence})"
                )
            self._stop(task, STOP_STRUCTURAL, msg)
        if _diff_introduces_conflict_marker_pair(text):
            self._stop(task, STOP_STRUCTURAL, "conflict markers in diff")
        patterns = self.cfg.data["risk"].get("forbidden_patterns") or []
        pat_hits = check_forbidden_patterns(text, patterns)
        if pat_hits:
            self._stop(task, STOP_STRUCTURAL, f"forbidden patterns: {pat_hits}")
        sens = self.cfg.data["risk"].get("sensitive_files") or []
        sens_hits = check_sensitive_files(changed, sens)
        if sens_hits:
            self._stop(task, STOP_STRUCTURAL, f"sensitive files touched: {sens_hits}")

    def _parallel_pr_and_merge(
        self,
        task: Task,
        verdict: Verdict,
        base_sha: str,
        head_sha: str,
    ) -> bool:
        changed = diff_files(self.deps.cwd, base_sha, head_sha)
        stats = diff_stats(self.deps.cwd, base_sha, head_sha)
        self._emit_blocking_hook(
            task,
            "task.before_pr",
            verdict=verdict.value,
            task_branch=task.branch,
            iter_branch=self.iter_branch,
            changed_files=changed,
            diff_insertions=stats.insertions,
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
            return False
        pr_url = url_or_err
        self.state.record_pr(task.id, pr_url)

        auto_merge_cfg = self.cfg.data["auto_merge"]
        no_ci = bool(auto_merge_cfg.get("no_ci", False))
        ci_passed = False if no_ci else True
        if self.options.poll_ci and not no_ci:
            status = self.deps.wait_for_ci(
                task.branch, cwd=self.deps.cwd,
                ci_wait_seconds=self.cfg.data["timeouts"]["ci"],
            )
            ci_passed = status.passed

        sens = check_sensitive_files(
            changed, self.cfg.data["risk"].get("sensitive_files") or []
        )
        forb = check_forbidden_patterns(
            diff_text(self.deps.cwd, base_sha, head_sha),
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
            return False

        if no_ci and not self._noop_acceptance_gate_allows_no_ci_merge(task):
            return False

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
            try:
                merge_sha = self._merge_task_branch_locally(
                    task,
                    message=(
                        f"Merge {task.branch} into {self.iter_branch} "
                        "(parallel no-CI local)"
                    ),
                )
            except GitError as exc:
                self.state.task_transition(task.id, STATUS_NEEDS_HUMAN_MERGE)
                self.state.append_event(
                    kind="note", task=task.id,
                    meta={"event": "local_merge_failed", "msg": str(exc)},
                )
                return False
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
            self._write_tasks_md_done(task)
            return True

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
            return False
        merge_sha = parse_merge_sha(msg)
        self.state.record_merge(
            task.id, auto_merged=True, merge_sha=merge_sha,
        )
        self.state.task_transition(task.id, STATUS_DONE)
        checkout(self.deps.cwd, self.iter_branch)
        self._pull_iter_branch_ff(
            context="post auto-merge SHA refresh", task=task.id
        )
        self.state.set_iter_branch_sha(
            current_sha(self.deps.cwd, self.iter_branch)
        )
        self._write_tasks_md_done(task)
        return True


class _BufferedStateStore(StateStore):
    """StateStore variant used by parallel child tasks.

    Child tasks need normal snapshot semantics while they run, but only the
    parent may mutate the shared run_state.json. ``save`` therefore rebuilds
    the in-memory snapshot without touching disk; the parent later replays the
    buffered events in deterministic task order.
    """

    def save(self) -> None:
        self.snapshot = self._rebuild()
