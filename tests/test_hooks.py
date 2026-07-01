from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from orch.hooks import (
    CommandHookHandler,
    HookContext,
    HookDispatcher,
    HookResult,
    HookVeto,
)


class _StaticHandler:
    def __init__(
        self,
        *,
        name: str = "static",
        events: tuple[str, ...] = ("task.before_pr",),
        required: bool = False,
        result: HookResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.required = required
        self.result = result or HookResult.ok()
        self.error = error

    def handles(self, event_name: str) -> bool:
        return event_name in self.events

    def handle(self, context: HookContext) -> HookResult:
        if self.error is not None:
            raise self.error
        return self.result


def _context(*, blocking: bool = True, event_name: str = "task.before_pr"):
    event = {
        "ts": "2026-05-20T00:00:00Z",
        "kind": "hook",
        "task": "I1-T1",
        "step": "HOOK",
        "status": None,
        "meta": {"event": event_name, "payload_key": "payload-value"},
    }
    return HookContext(
        event=event,
        snapshot={"iteration": "demo-i1", "tasks": {}},
        iteration="demo-i1",
        iter_branch="demo/iteration-1",
        task_id="I1-T1",
        blocking=blocking,
        payload=event["meta"],
    )


def test_dispatcher_raises_veto_for_blocking_event():
    incidents: list[dict] = []
    dispatcher = HookDispatcher(
        [
            _StaticHandler(
                result=HookResult.veto(
                    reason="policy",
                    message="blocked by policy",
                    metadata={"rule": "demo"},
                )
            )
        ]
    )

    with pytest.raises(HookVeto) as exc:
        dispatcher.dispatch(_context(blocking=True), incidents.append)

    assert exc.value.reason == "policy"
    assert incidents == [
        {
            "event": "hook_veto",
            "hook_event": "task.before_pr",
            "handler": "static",
            "required": False,
            "blocking": True,
            "reason": "policy",
            "msg": "blocked by policy",
            "metadata": {"rule": "demo"},
        }
    ]


def test_nonblocking_veto_is_logged_and_ignored():
    incidents: list[dict] = []
    dispatcher = HookDispatcher(
        [
            _StaticHandler(
                result=HookResult.veto(
                    reason="advisory",
                    message="nonblocking veto",
                )
            )
        ]
    )

    dispatcher.dispatch(_context(blocking=False), incidents.append)

    assert incidents[0]["event"] == "hook_ignored_veto"
    assert incidents[0]["reason"] == "advisory"


def test_required_blocking_handler_failure_fails_closed():
    incidents: list[dict] = []
    dispatcher = HookDispatcher(
        [
            _StaticHandler(
                required=True,
                error=RuntimeError("broken hook"),
            )
        ]
    )

    with pytest.raises(HookVeto) as exc:
        dispatcher.dispatch(_context(blocking=True), incidents.append)

    assert exc.value.reason == "handler_failed"
    assert [incident["event"] for incident in incidents] == [
        "hook_failure",
        "hook_veto",
    ]
    assert incidents[0]["required"] is True


def test_optional_handler_failure_fails_open():
    incidents: list[dict] = []
    dispatcher = HookDispatcher(
        [_StaticHandler(required=False, error=RuntimeError("broken hook"))]
    )

    dispatcher.dispatch(_context(blocking=True), incidents.append)

    assert len(incidents) == 1
    assert incidents[0]["event"] == "hook_failure"
    assert incidents[0]["error"] == "broken hook"


def test_command_hook_handler_receives_context_and_returns_veto(tmp_path: Path):
    script = tmp_path / "hook.py"
    script.write_text(
        "import json, sys\n"
        "ctx = json.load(sys.stdin)\n"
        "assert ctx['event']['meta']['event'] == 'task.before_pr'\n"
        "print(json.dumps({"
        "'status': 'veto', "
        "'reason': 'script_policy', "
        "'message': ctx['task_id'], "
        "'metadata': {'iteration': ctx['iteration']}"
        "}))\n"
    )
    handler = CommandHookHandler(
        name="script",
        events=["task.before_pr"],
        cmd=f"{sys.executable} {script}",
        required=True,
        timeout=5,
        cwd=tmp_path,
    )

    result = handler.handle(_context(blocking=True))

    assert result.status == "veto"
    assert result.reason == "script_policy"
    assert result.message == "I1-T1"
    assert result.metadata == {"iteration": "demo-i1"}


def test_command_hook_timeout_required_blocking_fails_closed(tmp_path: Path):
    script = tmp_path / "slow_hook.py"
    script.write_text("import time\n" "time.sleep(30)\n")
    handler = CommandHookHandler(
        name="slow",
        events=["task.before_pr"],
        cmd=f"{sys.executable} {script}",
        required=True,
        timeout=1,
        cwd=tmp_path,
    )
    dispatcher = HookDispatcher([handler])
    incidents: list[dict] = []

    started = time.monotonic()
    with pytest.raises(HookVeto) as exc:
        dispatcher.dispatch(_context(blocking=True), incidents.append)
    elapsed = time.monotonic() - started

    assert elapsed < 5
    assert exc.value.reason == "handler_failed"
    assert [incident["event"] for incident in incidents] == [
        "hook_failure",
        "hook_veto",
    ]
    assert incidents[0]["error_type"] == "TimeoutExpired"
    assert incidents[0]["timeout"] == 1


def test_command_hook_invalid_json_optional_failure_fails_open(tmp_path: Path):
    script = tmp_path / "bad_json_hook.py"
    script.write_text("print('not-json')\n")
    handler = CommandHookHandler(
        name="bad-json",
        events=["task.before_pr"],
        cmd=f"{sys.executable} {script}",
        required=False,
        timeout=5,
        cwd=tmp_path,
    )
    dispatcher = HookDispatcher([handler])
    incidents: list[dict] = []

    dispatcher.dispatch(_context(blocking=True), incidents.append)

    assert len(incidents) == 1
    assert incidents[0]["event"] == "hook_failure"
    assert incidents[0]["error_type"] == "HookHandlerError"
    assert "invalid JSON" in incidents[0]["error"]
