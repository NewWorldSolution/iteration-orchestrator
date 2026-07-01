"""Thread-race characterization for parallel wave child event replay."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from orch.agents.base import AgentResult
from orch.git_ops import commit, stage_all
from orch.state import STATUS_DONE
from tests.test_runner import (
    FakeAdapter,
    _enable_parallel_config,
    _enable_parallel_tasks,
    _make_runner,
    repo as _runner_repo,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _runner_repo.__wrapped__(tmp_path)


def _task_for_prompt(prompt: str) -> tuple[str, str, str]:
    if "src/a.py" in prompt or "I1-T1" in prompt:
        return "I1-T1", "src/a.py", "def a():\n    return 1\n"
    if "src/b.py" in prompt or "I1-T2" in prompt:
        return "I1-T2", "src/b.py", "def b():\n    return 2\n"
    if "src/c.py" in prompt or "I1-T3" in prompt:
        return "I1-T3", "src/c.py", "def c():\n    return 3\n"
    raise AssertionError(f"unexpected prompt: {prompt}")


def _event_marker(event: dict) -> str | None:
    meta = event.get("meta") or {}
    return (
        meta.get("event")
        or meta.get("status")
        or meta.get("verdict")
        or meta.get("phase")
        or ("sha" if "sha" in meta else None)
    )


def _task_event_indices(events: list[dict], task_id: str) -> list[int]:
    return [
        index for index, event in enumerate(events)
        if event.get("task") == task_id
    ]


def test_parallel_wave_replays_concurrent_child_events_in_task_index_order(
    repo: Path,
):
    _enable_parallel_config(repo, max_concurrency=3)
    _enable_parallel_tasks(repo, include_third=True)

    task_ids = ("I1-T1", "I1-T2", "I1-T3")
    impl_ready = threading.Barrier(len(task_ids) + 1)
    review_ready = threading.Barrier(len(task_ids) + 1)
    release_impl = {task_id: threading.Event() for task_id in task_ids}
    release_review = {task_id: threading.Event() for task_id in task_ids}
    impl_done = {task_id: threading.Event() for task_id in task_ids}
    review_done = {task_id: threading.Event() for task_id in task_ids}
    impl_done_order: list[str] = []
    review_done_order: list[str] = []
    order_lock = threading.Lock()

    def barrier_staggered_edit(adapter, prompt, workdir):
        task_id, relpath, content = _task_for_prompt(prompt)
        impl_ready.wait()
        release_impl[task_id].wait()
        path = Path(workdir) / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        stage_all(Path(workdir))
        commit(Path(workdir), f"{adapter.name}: edit {relpath}")
        with order_lock:
            impl_done_order.append(task_id)
        impl_done[task_id].set()
        return AgentResult(
            exit_code=0,
            stdout="",
            stderr="",
            duration_s=0.1,
            input_tokens=50,
            output_tokens=20,
            tokens_exact=False,
        )

    def barrier_staggered_review(adapter, prompt, workdir):
        task_id, _, _ = _task_for_prompt(prompt)
        review_ready.wait()
        release_review[task_id].wait()
        with order_lock:
            review_done_order.append(task_id)
        review_done[task_id].set()
        return AgentResult(
            exit_code=0,
            stdout="Looks good.\nVerdict: PASS\n",
            stderr="",
            duration_s=0.1,
            input_tokens=30,
            output_tokens=10,
            tokens_exact=False,
        )

    impl = FakeAdapter(
        name="claude",
        family="anthropic",
        script=[barrier_staggered_edit] * len(task_ids),
    )
    rev = FakeAdapter(
        name="codex",
        family="openai",
        script=[barrier_staggered_review] * len(task_ids),
    )
    runner, store, _ = _make_runner(
        repo,
        {"claude": impl, "codex": rev},
        dry_run=True,
        deps_kwargs={"run_lock": SimpleNamespace(acquired=True)},
        implementer="claude",
        reviewer="codex",
    )
    result = SimpleNamespace(rc=None, error=None)

    def run_parallel_wave():
        try:
            result.rc = runner.run()
        except BaseException as exc:  # pragma: no cover - surfaced below
            result.error = exc
            for event in [*release_impl.values(), *release_review.values()]:
                event.set()

    run_thread = threading.Thread(target=run_parallel_wave)
    run_thread.start()

    impl_ready.wait()
    for task_id in reversed(task_ids):
        release_impl[task_id].set()
        impl_done[task_id].wait()

    review_ready.wait()
    for task_id in reversed(task_ids):
        release_review[task_id].set()
        review_done[task_id].wait()

    run_thread.join()

    assert result.error is None
    assert result.rc == 0
    assert impl_done_order == ["I1-T3", "I1-T2", "I1-T1"]
    assert review_done_order == ["I1-T3", "I1-T2", "I1-T1"]

    for task_id in task_ids:
        assert store.tasks[task_id].status == STATUS_DONE
        markers = [
            _event_marker(event)
            for event in store.events
            if event.get("task") == task_id
        ]
        assert "model_routing_resolved" in markers
        assert markers.count("IN_PROGRESS") == 1
        assert markers.count("end") == 1
        assert markers.count("DONE") == 1
        assert [
            event.get("meta", {}).get("verdict")
            for event in store.events
            if event.get("task") == task_id and event.get("kind") == "review"
        ] == ["PASS"]
        assert [
            event.get("meta", {}).get("action")
            for event in store.events
            if event.get("task") == task_id and event.get("kind") == "triage"
        ] == ["PROCEED"]

    task_event_order = [
        event.get("task")
        for event in store.events
        if event.get("task") in task_ids
    ]
    assert task_event_order == sorted(
        task_event_order, key=lambda task_id: task_ids.index(task_id)
    )
    assert max(_task_event_indices(store.events, "I1-T1")) < min(
        _task_event_indices(store.events, "I1-T2")
    )
    assert max(_task_event_indices(store.events, "I1-T2")) < min(
        _task_event_indices(store.events, "I1-T3")
    )
