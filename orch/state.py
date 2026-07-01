"""Iteration run state: append-only events as source of truth.

Events are the single source of truth. The ``tasks`` snapshot is recomputed
from events on every save and persisted alongside them as a debugging aid
— never read as authoritative. On ``load`` the snapshot is discarded and
rebuilt from events.

State file layout (``tools/logs/<iter>/run_state.json``):

    {
      "iteration": "demo-i1",
      "iter_branch": "demo/iteration-1",
      "started_at": "2026-04-15T09:42:11Z",
      "iter_branch_sha": "abc123...",
      "tasks": { "<id>": {...snapshot...} },
      "events": [ {ts, kind, task, step, status, meta}, ... ]
    }
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from orch.hooks import HookContext, HookDispatcher

# Event kinds — enumerated so typos surface as test failures.
EVT_ITERATION   = "iteration"         # meta: started_at | finished_at | verdict
EVT_SHA         = "sha"               # meta: sha
EVT_TASK        = "task_transition"   # meta: status[, reason, msg]
EVT_TASK_META   = "task_meta"         # meta: arbitrary task fields to merge
EVT_IMPL        = "impl_attempt"      # meta: phase (start|end), agent, exit_code
EVT_FIX         = "fix_round"         # meta: phase, cause (acceptance|review), agent
EVT_REVIEW      = "review"            # meta: phase, reviewer, verdict, round
EVT_SCOPE       = "scope_revert"      # meta: files
EVT_PR          = "pr_opened"         # meta: pr_url
EVT_MERGE_INTENT = "merge_intent"     # meta: target/task refs before merge
EVT_MERGE_COMPLETE = "merge_complete" # meta: merge_sha after git mutation
EVT_MERGE       = "merge"             # meta: auto_merged, merge_sha
EVT_PAIR_SWAP   = "pair_swap"         # meta: primary_implementer, primary_reviewer,
                                       # swap_implementer, swap_reviewer, reason
EVT_NOTE        = "note"              # meta: arbitrary logging
EVT_TRIAGE      = "triage"            # meta: TriageDecision dict + inputs snapshot
EVT_CONFIDENCE  = "confidence"        # meta: round, value (float|None)
EVT_HOOK        = "hook"              # meta: event + hook boundary payload

# Task status values — includes runtime-only states.
STATUS_WAITING           = "WAITING"
STATUS_IN_PROGRESS       = "IN_PROGRESS"
STATUS_DONE              = "DONE"
STATUS_NEEDS_HUMAN_MERGE = "NEEDS_HUMAN_MERGE"
STATUS_BLOCKED_UPSTREAM  = "BLOCKED_UPSTREAM"
STATUS_STOPPED_PREFIX    = "STOPPED:"   # concatenated with reason, e.g. "STOPPED:CHECKS"


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TaskState:
    id: str
    status: str = STATUS_WAITING
    impl_attempts: int = 0
    fix_rounds: int = 0           # acceptance-driven
    review_fix_rounds: int = 0    # review-driven
    branch: str = ""
    implementer: str | None = None
    reviewer: str | None = None
    pr_url: str | None = None
    verdict: str | None = None
    diff_insertions: int = 0
    auto_merged: bool = False
    merge_sha: str | None = None
    stop_reason: str | None = None
    stop_msg: str | None = None
    # v2.2 — one-shot model-pair swap on REVIEW_FAIL
    swap_attempted: bool = False
    swap_implementer: str | None = None
    swap_reviewer: str | None = None
    swap_reason: str | None = None
    swap_outcome: str | None = None  # "PASS" | "REVIEW_FAIL" | other stop reason
    # Failure-triage state used by the review-failure decision tree.
    triage_decisions: list[dict] = field(default_factory=list)
    defer_budget_used: int = 0
    confidence_history: list[float | None] = field(default_factory=list)
    triage_outcome: str | None = None


@dataclass
class IterationSnapshot:
    iteration: str
    iter_branch: str
    started_at: str | None = None
    finished_at: str | None = None
    iter_branch_sha: str | None = None
    tasks: dict[str, TaskState] = field(default_factory=dict)


class StateStore:
    """Append-only event log with a rebuilt snapshot.

    Operators read ``run_state.json`` directly; higher layers call
    ``append_*`` helpers. ``load()`` rebuilds the snapshot from events.
    """

    def __init__(
        self,
        log_dir: Path,
        iteration: str,
        iter_branch: str,
        hook_dispatcher: HookDispatcher | None = None,
    ) -> None:
        self.log_dir = log_dir
        self.path = log_dir / "run_state.json"
        self.iteration = iteration
        self.iter_branch = iter_branch
        self.events: list[dict] = []
        self.hook_dispatcher = hook_dispatcher
        self.snapshot = IterationSnapshot(
            iteration=iteration, iter_branch=iter_branch
        )

    def hooks_enabled(self) -> bool:
        if self.hook_dispatcher is None:
            return False
        handlers = getattr(self.hook_dispatcher, "handlers", None)
        if handlers is not None:
            return bool(handlers)
        return True

    # --- persistence -------------------------------------------------------

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> None:
        orphaned_tmp = sorted(self.log_dir.glob("run_state.*.tmp"))
        if orphaned_tmp and not self.path.exists():
            names = ", ".join(str(path) for path in orphaned_tmp)
            raise ValueError(
                f"run_state.json is missing but orphaned run_state .tmp "
                f"file(s) exist at {names}; inspect before resuming"
            )
        if not self.path.exists():
            self.events = []
            self.snapshot = IterationSnapshot(
                iteration=self.iteration, iter_branch=self.iter_branch
            )
            return
        try:
            raw = json.loads(self.path.read_text())
        except json.JSONDecodeError as exc:
            self.events = []
            self.snapshot = IterationSnapshot(
                iteration=self.iteration, iter_branch=self.iter_branch
            )
            raise ValueError(
                f"corrupt or truncated run_state.json at {self.path}: {exc}"
            ) from exc
        # Iteration / branch must match — refuse to load mismatched state
        if raw.get("iteration") != self.iteration:
            raise ValueError(
                f"state iteration '{raw.get('iteration')}' does not match "
                f"'{self.iteration}' at {self.path}"
            )
        self.iter_branch = raw.get("iter_branch", self.iter_branch)
        self.events = list(raw.get("events", []))
        self.snapshot = self._rebuild()
        for tmp in orphaned_tmp:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    def save(self) -> None:
        self.snapshot = self._rebuild()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "iteration": self.snapshot.iteration,
            "iter_branch": self.snapshot.iter_branch,
            "started_at": self.snapshot.started_at,
            "finished_at": self.snapshot.finished_at,
            "iter_branch_sha": self.snapshot.iter_branch_sha,
            "tasks": {
                tid: asdict(ts) for tid, ts in self.snapshot.tasks.items()
            },
            "events": self.events,
        }
        # Atomic write: tmp in same dir + os.replace
        fd, tmp = tempfile.mkstemp(
            prefix="run_state.", suffix=".tmp", dir=str(self.log_dir)
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=False)
                f.write("\n")
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # --- event append ------------------------------------------------------

    def append_event(
        self,
        *,
        kind: str,
        task: str | None = None,
        step: str | None = None,
        status: str | None = None,
        meta: dict[str, Any] | None = None,
        ts: str | None = None,
        dispatch_hooks: bool = True,
        hook_blocking: bool = False,
    ) -> dict:
        ev = {
            "ts": ts or utcnow_iso(),
            "kind": kind,
            "task": task,
            "step": step,
            "status": status,
            "meta": copy.deepcopy(meta or {}),
        }
        self.events.append(ev)
        self.save()
        if dispatch_hooks and self.hooks_enabled():
            context = HookContext(
                event=copy.deepcopy(ev),
                snapshot={
                    "iteration": self.snapshot.iteration,
                    "iter_branch": self.snapshot.iter_branch,
                    "started_at": self.snapshot.started_at,
                    "finished_at": self.snapshot.finished_at,
                    "iter_branch_sha": self.snapshot.iter_branch_sha,
                    "tasks": {
                        tid: asdict(ts)
                        for tid, ts in self.snapshot.tasks.items()
                    },
                },
                iteration=self.iteration,
                iter_branch=self.iter_branch,
                task_id=task,
                blocking=hook_blocking,
                payload=copy.deepcopy(ev["meta"]),
            )
            self.hook_dispatcher.dispatch(
                context,
                emit_internal=lambda internal_meta: self.append_event(
                    kind=EVT_NOTE,
                    task=task,
                    meta=internal_meta,
                    dispatch_hooks=False,
                ),
            )
        return ev

    # --- convenience helpers ----------------------------------------------

    def mark_iteration_started(self) -> None:
        if self.snapshot.started_at is None:
            self.append_event(
                kind=EVT_ITERATION,
                meta={"started_at": utcnow_iso()},
            )

    def mark_iteration_finished(self, *, verdict: str) -> None:
        self.append_event(
            kind=EVT_ITERATION,
            meta={"finished_at": utcnow_iso(), "verdict": verdict},
        )

    def set_iter_branch_sha(self, sha: str) -> None:
        self.append_event(kind=EVT_SHA, meta={"sha": sha})

    def task_transition(
        self,
        task: str,
        status: str,
        *,
        reason: str | None = None,
        msg: str | None = None,
    ) -> None:
        meta: dict[str, Any] = {"status": status}
        if reason is not None:
            meta["reason"] = reason
        if msg is not None:
            meta["msg"] = msg
        self.append_event(kind=EVT_TASK, task=task, meta=meta)

    def task_meta(self, task: str, **fields: Any) -> None:
        self.append_event(kind=EVT_TASK_META, task=task, meta=fields)

    def record_impl_attempt_end(
        self,
        task: str,
        *,
        agent: str,
        exit_code: int,
        duration_s: float,
        classification: str | None = None,
        stderr_tail: str | None = None,
    ) -> None:
        # When the impl agent fails, the caller passes a heuristic
        # classification (rate_limit / disk / oom / permission / unknown)
        # plus the last ~50 lines of stderr so operators can triage
        # without rerunning blindly. On success neither field is set.
        meta: dict[str, Any] = {
            "phase": "end",
            "agent": agent,
            "exit_code": exit_code,
            "duration_s": duration_s,
        }
        if classification is not None:
            meta["classification"] = classification
        if stderr_tail is not None:
            # Persist even an empty tail so IMPL_FAILED events always carry
            # the stderr_tail field next to classification.
            meta["stderr_tail"] = stderr_tail
        self.append_event(
            kind=EVT_IMPL,
            task=task,
            step="IMPL",
            meta=meta,
        )

    def record_fix_round_end(
        self,
        task: str,
        *,
        cause: str,
        agent: str,
        exit_code: int,
        duration_s: float,
    ) -> None:
        assert cause in ("acceptance", "review"), cause
        self.append_event(
            kind=EVT_FIX,
            task=task,
            step="FIX",
            meta={
                "phase": "end",
                "cause": cause,
                "agent": agent,
                "exit_code": exit_code,
                "duration_s": duration_s,
            },
        )

    def record_review_result(
        self,
        task: str,
        *,
        reviewer: str,
        verdict: str,
        round_num: int,
    ) -> None:
        self.append_event(
            kind=EVT_REVIEW,
            task=task,
            step="REVIEW",
            meta={
                "phase": "end",
                "reviewer": reviewer,
                "verdict": verdict,
                "round": round_num,
            },
        )

    def record_pr(self, task: str, pr_url: str) -> None:
        self.append_event(
            kind=EVT_PR,
            task=task,
            step="PR",
            meta={"pr_url": pr_url},
        )

    def record_merge(
        self,
        task: str,
        *,
        auto_merged: bool,
        merge_sha: str | None = None,
    ) -> None:
        self.append_event(
            kind=EVT_MERGE,
            task=task,
            step="MERGE",
            meta={"auto_merged": auto_merged, "merge_sha": merge_sha},
        )

    def record_merge_intent(
        self,
        task: str,
        *,
        target_branch: str,
        target_sha_before: str,
        task_branch: str,
        task_sha: str,
        message: str,
        mode: str,
    ) -> dict:
        return self.append_event(
            kind=EVT_MERGE_INTENT,
            task=task,
            step="MERGE",
            meta={
                "target_branch": target_branch,
                "target_sha_before": target_sha_before,
                "task_branch": task_branch,
                "task_sha": task_sha,
                "message": message,
                "mode": mode,
            },
        )

    def record_merge_complete(
        self,
        task: str,
        *,
        target_branch: str,
        merge_sha: str,
        target_sha_before: str,
        task_branch: str,
        task_sha: str,
        mode: str,
        reconciled: bool = False,
    ) -> dict:
        return self.append_event(
            kind=EVT_MERGE_COMPLETE,
            task=task,
            step="MERGE",
            meta={
                "target_branch": target_branch,
                "merge_sha": merge_sha,
                "target_sha_before": target_sha_before,
                "task_branch": task_branch,
                "task_sha": task_sha,
                "mode": mode,
                "reconciled": bool(reconciled),
            },
        )

    def append_triage_decision(
        self,
        task: str,
        *,
        action: str,
        reason: str,
        round_num: int,
        verdict: str,
        severity: str | None,
        increments_defer_budget: bool,
        confidence_history: list[float | None] | None = None,
    ) -> None:
        """Persist a triage decision."""
        self.append_event(
            kind=EVT_TRIAGE,
            task=task,
            step="TRIAGE",
            meta={
                "action": action,
                "reason": reason,
                "round": round_num,
                "verdict": verdict,
                "severity": severity,
                "increments_defer_budget": bool(increments_defer_budget),
                "confidence_history": list(confidence_history or []),
            },
        )

    def record_confidence(
        self, task: str, *, round_num: int, value: float | None,
    ) -> None:
        self.append_event(
            kind=EVT_CONFIDENCE,
            task=task,
            step="REVIEW",
            meta={"round": round_num, "value": value},
        )

    # --- snapshot reconstruction ------------------------------------------

    def _rebuild(self) -> IterationSnapshot:
        snap = IterationSnapshot(
            iteration=self.iteration, iter_branch=self.iter_branch
        )
        for ev in self.events:
            _apply_event(snap, ev)
        return snap

    # --- task reset (P1-4: selective retry) --------------------------------

    def reset_task(self, task_id: str) -> None:
        """Append a reset event that returns a task to WAITING.

        This allows ``orch retry`` to re-run a single task without
        discarding the rest of the iteration state.
        """
        self.append_event(
            kind=EVT_TASK,
            task=task_id,
            meta={"status": STATUS_WAITING, "reset": True},
        )

    # --- read-only convenience --------------------------------------------

    @property
    def tasks(self) -> dict[str, TaskState]:
        return self.snapshot.tasks

    def get_task(self, tid: str) -> TaskState:
        if tid not in self.snapshot.tasks:
            self.snapshot.tasks[tid] = TaskState(id=tid)
        return self.snapshot.tasks[tid]


def _get_task(snap: IterationSnapshot, tid: str) -> TaskState:
    if tid not in snap.tasks:
        snap.tasks[tid] = TaskState(id=tid)
    return snap.tasks[tid]


def _apply_event(snap: IterationSnapshot, ev: dict) -> None:
    kind = ev.get("kind")
    tid = ev.get("task")
    meta = ev.get("meta") or {}

    if kind == EVT_ITERATION:
        if "started_at" in meta and snap.started_at is None:
            snap.started_at = meta["started_at"]
        if "finished_at" in meta:
            snap.finished_at = meta["finished_at"]
        return

    if kind == EVT_SHA:
        snap.iter_branch_sha = meta.get("sha")
        return

    if tid is None:
        return  # non-task events beyond iteration/sha are audit-only

    t = _get_task(snap, tid)

    if kind == EVT_TASK:
        t.status = meta.get("status", t.status)
        if meta.get("reset"):
            # P1-4: reset clears counters so the task can be re-run cleanly
            t.impl_attempts = 0
            t.fix_rounds = 0
            t.review_fix_rounds = 0
            # B-14 R4: preserve pr_url across reset. The outcome fields below
            # are cleared for a clean re-run, but the PR reference is retained
            # so external-merge reconciliation can still detect that the prior
            # PR merged (the wipe lost that link). record_pr overwrites pr_url
            # when the re-run opens a new PR, so a retained value cannot
            # mis-gate a later attempt.
            t.verdict = None
            t.auto_merged = False
            t.merge_sha = None
            t.stop_reason = None
            t.stop_msg = None
            # Clear triage history on retry
            t.triage_decisions = []
            t.defer_budget_used = 0
            t.confidence_history = []
            t.triage_outcome = None
        if "reason" in meta:
            t.stop_reason = meta["reason"]
        if "msg" in meta:
            t.stop_msg = meta["msg"]
        return

    if kind == EVT_TASK_META:
        for k, v in meta.items():
            if hasattr(t, k):
                setattr(t, k, v)
        return

    if kind == EVT_IMPL:
        if meta.get("phase") == "end":
            t.impl_attempts += 1
        return

    if kind == EVT_FIX:
        if meta.get("phase") == "end":
            cause = meta.get("cause", "acceptance")
            if cause == "acceptance":
                t.fix_rounds += 1
            else:
                t.review_fix_rounds += 1
        return

    if kind == EVT_REVIEW:
        if meta.get("phase") == "end":
            t.verdict = meta.get("verdict", t.verdict)
            t.reviewer = meta.get("reviewer", t.reviewer)
        return

    if kind == EVT_PR:
        t.pr_url = meta.get("pr_url", t.pr_url)
        return

    if kind == EVT_MERGE:
        t.auto_merged = bool(meta.get("auto_merged"))
        t.merge_sha = meta.get("merge_sha", t.merge_sha)
        return

    if kind == EVT_TRIAGE:
        # Append-only triage decision history.
        t.triage_decisions.append(dict(meta))
        action = meta.get("action")
        if action:
            t.triage_outcome = action
        if meta.get("increments_defer_budget"):
            t.defer_budget_used += 1
        return

    if kind == EVT_CONFIDENCE:
        t.confidence_history.append(meta.get("value"))
        return

    # EVT_SCOPE, EVT_NOTE: audit-only, do not mutate snapshot
