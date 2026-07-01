"""Finalization helpers for merge diagnostics and task status updates."""
from __future__ import annotations

from orch.git_ops import checkout, commit, current_sha, git, stage_all
from orch.merge import (
    build_needs_human_merge_meta,
    classify_pr_status,
    is_external_merge_complete,
)
from orch.state import (
    EVT_MERGE_COMPLETE,
    EVT_MERGE_INTENT,
    STATUS_DONE,
    STATUS_IN_PROGRESS,
    STATUS_NEEDS_HUMAN_MERGE,
)
from orch.tasks_schema import Task
from orch.tasks_writer import update_task_status


def emit_needs_human_merge_diagnostics(
    runner,
    task: Task,
    pr_url: str,
    *,
    ci_passed: bool,
    decision_reasons: list[str] | None = None,
) -> None:
    """Record NEEDS_HUMAN_MERGE diagnostics on the runner state."""
    try:
        snap = runner.deps.query_pr_state(pr_url, cwd=runner.deps.cwd)
    except Exception as exc:  # network / gh failure -> keep going
        classification, msg_detail = (
            "lookup-failed",
            f"gh pr view failed: {exc}",
        )
    else:
        classification, msg_detail = classify_pr_status(snap.rollup)
    meta = build_needs_human_merge_meta(
        classification=classification,
        msg_detail=msg_detail,
        ci_passed=ci_passed,
        pr_url=pr_url,
        iteration=runner.iteration,
        task_id=task.id,
        decision_reasons=decision_reasons,
        artifact_root_ref=runner.orch_paths.artifact_root_ref,
    )
    runner.state.append_event(kind="note", task=task.id, meta=meta)


def check_external_merges(runner) -> None:
    """Scan tasks with open PRs that may have merged externally."""
    for task_id, ts in list(runner.state.tasks.items()):
        if ts.status not in {STATUS_NEEDS_HUMAN_MERGE, STATUS_IN_PROGRESS}:
            continue
        pr_url = ts.pr_url
        if not pr_url:
            continue
        try:
            snap = runner.deps.query_pr_state(pr_url, cwd=runner.deps.cwd)
        except Exception as exc:
            runner.state.append_event(
                kind="note",
                task=task_id,
                meta={
                    "event": "external_merge_lookup_failed",
                    "msg": str(exc),
                },
            )
            continue
        if is_external_merge_complete(snap):
            runner._log(
                f"  detected external merge of {task_id} "
                f"(PR merged at {snap.merge_sha[:7]}); "
                "transitioning to DONE"
            )
            runner.state.record_merge(
                task_id, auto_merged=False, merge_sha=snap.merge_sha
            )
            runner.state.task_transition(
                task_id,
                STATUS_DONE,
                reason="external_merge_detected",
                msg=f"PR merged externally at {snap.merge_sha}",
            )


def _is_ancestor(runner, ancestor: str, descendant: str) -> bool:
    if not ancestor or not descendant:
        return False
    return git(
        ["merge-base", "--is-ancestor", ancestor, descendant],
        cwd=runner.deps.cwd,
    ).ok


def _latest_merge_journal(runner, task_id: str) -> tuple[dict | None, dict | None]:
    intent: dict | None = None
    complete: dict | None = None
    for event in runner.state.events:
        if event.get("task") != task_id:
            continue
        if event.get("kind") == EVT_MERGE_INTENT:
            intent = event
            complete = None
        elif event.get("kind") == EVT_MERGE_COMPLETE and intent is not None:
            complete = event
    return intent, complete


def _safe_landed_local_merge(
    runner,
    task: Task,
    task_state,
    intent: dict,
    complete: dict | None,
) -> tuple[bool, str | None, str]:
    meta = intent.get("meta") or {}
    if meta.get("target_branch") != runner.iter_branch:
        return False, None, "target_branch_mismatch"
    if meta.get("task_branch") != task.branch:
        return False, None, "task_branch_mismatch"

    target_sha_before = str(meta.get("target_sha_before") or "")
    task_sha = str(meta.get("task_sha") or "")
    if not target_sha_before or not task_sha:
        return False, None, "missing_expected_sha"
    if task_sha == target_sha_before:
        return False, None, "task_branch_not_ahead"
    if not _is_ancestor(runner, target_sha_before, task_sha):
        return False, None, "task_sha_not_based_on_target"
    if _is_ancestor(runner, task_sha, target_sha_before):
        return False, None, "task_sha_already_in_target"

    current_iter_sha = current_sha(runner.deps.cwd, runner.iter_branch)
    if current_iter_sha == target_sha_before:
        return False, None, "target_branch_not_advanced"
    if not _is_ancestor(runner, target_sha_before, current_iter_sha):
        return False, None, "current_iter_not_descendant_of_target"
    if not _is_ancestor(runner, task_sha, current_iter_sha):
        return False, None, "task_sha_not_in_current_iter"

    complete_meta = (complete or {}).get("meta") or {}
    merge_sha = (
        task_state.merge_sha
        or complete_meta.get("merge_sha")
        or current_iter_sha
    )
    if not _is_ancestor(runner, str(merge_sha), current_iter_sha):
        return False, None, "recorded_merge_sha_not_in_current_iter"
    return True, str(merge_sha), "ok"


def reconcile_local_merges(runner) -> None:
    """Repair local no-CI merge crashes without re-running git merge.

    A task is reconciled only when the append-only merge intent plus git
    ancestry prove that the exact task SHA named before the mutation is now in
    the iteration branch. The repair path records state and tasks.md status
    only; it never invokes a merge command.
    """
    for task in runner.board.tasks:
        task_state = runner.state.tasks.get(task.id)
        if task_state is None:
            continue
        intent, complete = _latest_merge_journal(runner, task.id)
        if intent is None:
            continue
        if task_state.status not in {
            STATUS_DONE,
            STATUS_IN_PROGRESS,
            STATUS_NEEDS_HUMAN_MERGE,
        }:
            continue

        safe, merge_sha, reason = _safe_landed_local_merge(
            runner, task, task_state, intent, complete,
        )
        if not safe:
            if task_state.status != STATUS_DONE:
                runner.state.append_event(
                    kind="note",
                    task=task.id,
                    meta={
                        "event": "local_merge_reconcile_skipped",
                        "reason": reason,
                    },
                )
            continue

        if complete is None:
            meta = intent.get("meta") or {}
            runner.state.record_merge_complete(
                task.id,
                target_branch=runner.iter_branch,
                merge_sha=str(merge_sha),
                target_sha_before=str(meta.get("target_sha_before") or ""),
                task_branch=task.branch,
                task_sha=str(meta.get("task_sha") or ""),
                mode=str(meta.get("mode") or "local"),
                reconciled=True,
            )
        if not task_state.auto_merged or task_state.merge_sha != merge_sha:
            runner.state.record_merge(
                task.id, auto_merged=True, merge_sha=merge_sha,
            )
        runner._local_merge_reconciled_this_run = True
        if task_state.status != STATUS_DONE:
            runner.state.task_transition(
                task.id,
                STATUS_DONE,
                reason="local_merge_reconciled",
                msg=f"local merge already present at {merge_sha}",
            )
            runner._log(
                f"  reconciled local merge for {task.id} at "
                f"{str(merge_sha)[:7]}"
            )
        write_tasks_md_done(runner, task)


def write_tasks_md_done(runner, task: Task) -> None:
    try:
        # Write on the iter branch so the status update survives a merge.
        checkout(runner.deps.cwd, runner.iter_branch)
        changed = update_task_status(runner.tasks_md_worktree_path, task.id, "DONE")
        if changed:
            stage_all(runner.deps.cwd)
            commit(runner.deps.cwd, f"chore: mark {task.id} as DONE")
            push = runner.deps.push_branch(runner.deps.cwd, runner.iter_branch)
            if push.ok:
                runner.state.append_event(
                    kind="note",
                    task=task.id,
                    meta={
                        "event": "tasks_md_done_pushed",
                        "branch": runner.iter_branch,
                    },
                )
            else:
                runner.state.append_event(
                    kind="note",
                    task=task.id,
                    meta={
                        "event": "tasks_md_done_push_failed",
                        "branch": runner.iter_branch,
                        "msg": push.stderr.strip() or push.stdout.strip(),
                    },
                )
    except (FileNotFoundError, ValueError) as exc:
        runner.state.append_event(
            kind="note",
            task=task.id,
            meta={"event": "tasks_md_update_failed", "msg": str(exc)},
        )
