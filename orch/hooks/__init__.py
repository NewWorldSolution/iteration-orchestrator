"""Hook dispatch for orchestrator lifecycle events.

Hooks subscribe to persisted state events. The state store remains the single
funnel: events are saved first, then handlers observe the saved event through
this dispatcher. Production handlers are command-backed so failures and hangs
can be isolated from the runner process.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

BLOCKING_HOOK_EVENTS = {
    "task.before_start",
    "task.before_branch_prepare",
    "task.before_implement",
    "task.before_fix",
    "task.before_review",
    "task.before_pr",
    "task.before_merge",
}

INTERNAL_HOOK_EVENTS = {
    "hook_failure",
    "hook_veto",
    "hook_ignored_veto",
}


@dataclass(frozen=True)
class HookContext:
    event: dict[str, Any]
    snapshot: dict[str, Any]
    iteration: str
    iter_branch: str
    task_id: str | None
    blocking: bool
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def event_name(self) -> str:
        meta = self.event.get("meta") or {}
        name = meta.get("event")
        return str(name) if name else str(self.event.get("kind") or "")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "snapshot": self.snapshot,
            "iteration": self.iteration,
            "iter_branch": self.iter_branch,
            "task_id": self.task_id,
            "blocking": self.blocking,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class HookResult:
    status: str = "ok"
    reason: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, **metadata: Any) -> "HookResult":
        return cls(status="ok", metadata=metadata)

    @classmethod
    def veto(
        cls, *, reason: str, message: str, metadata: dict[str, Any] | None = None
    ) -> "HookResult":
        return cls(
            status="veto",
            reason=reason,
            message=message,
            metadata=dict(metadata or {}),
        )


class HookVeto(RuntimeError):
    def __init__(
        self,
        *,
        handler: str,
        event_name: str,
        reason: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.handler = handler
        self.event_name = event_name
        self.reason = reason
        self.message = message
        self.metadata = dict(metadata or {})


class HookHandlerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        exit_code: int | None = None,
        stderr_tail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.stderr_tail = stderr_tail


class HookHandler(Protocol):
    name: str
    required: bool

    def handles(self, event_name: str) -> bool: ...

    def handle(self, context: HookContext) -> HookResult: ...


class CommandHookHandler:
    """Command-backed hook handler.

    The context is written as JSON to stdin. The command must emit a JSON object
    compatible with HookResult on stdout.
    """

    def __init__(
        self,
        *,
        name: str,
        events: Sequence[str],
        cmd: str,
        required: bool = False,
        timeout: int = 10,
        cwd: Path,
    ) -> None:
        if not name:
            raise ValueError("hook handler requires a name")
        if not events:
            raise ValueError(f"hook handler {name!r} requires at least one event")
        if not cmd:
            raise ValueError(f"hook handler {name!r} requires a cmd")
        self.name = name
        self.events = tuple(str(event) for event in events)
        self.cmd = cmd
        self.required = bool(required)
        self.timeout = int(timeout)
        self.cwd = cwd

    def handles(self, event_name: str) -> bool:
        return event_name in self.events

    def handle(self, context: HookContext) -> HookResult:
        proc = subprocess.run(
            shlex.split(self.cmd),
            cwd=str(self.cwd),
            timeout=self.timeout,
            input=json.dumps(context.to_json_dict()),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            stderr_tail = "\n".join((proc.stderr or "").splitlines()[-20:])
            raise HookHandlerError(
                f"hook exited {proc.returncode}: {(proc.stderr or '').strip()}",
                exit_code=proc.returncode,
                stderr_tail=stderr_tail,
            )
        try:
            raw = json.loads(proc.stdout or "")
        except json.JSONDecodeError as exc:
            raise HookHandlerError(f"hook emitted invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise HookHandlerError("hook JSON output must be an object")
        status = str(raw.get("status") or "ok")
        if status not in {"ok", "veto"}:
            raise HookHandlerError(f"unknown hook status {status!r}")
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise HookHandlerError("hook metadata must be an object")
        return HookResult(
            status=status,
            reason=str(raw.get("reason") or ""),
            message=str(raw.get("message") or ""),
            metadata=metadata,
        )


class HookDispatcher:
    def __init__(self, handlers: Sequence[HookHandler] | None = None) -> None:
        self.handlers = list(handlers or [])

    def dispatch(
        self,
        context: HookContext,
        emit_internal: Callable[[dict[str, Any]], None],
    ) -> None:
        event_name = context.event_name
        if event_name in INTERNAL_HOOK_EVENTS:
            return
        for handler in self.handlers:
            if not handler.handles(event_name):
                continue
            try:
                result = handler.handle(context)
            except subprocess.TimeoutExpired as exc:
                result = self._handle_failure(handler, context, emit_internal, exc)
            except Exception as exc:
                result = self._handle_failure(handler, context, emit_internal, exc)
            if result.status != "veto":
                continue
            self._handle_veto(handler, context, result, emit_internal)

    def _handle_failure(
        self,
        handler: HookHandler,
        context: HookContext,
        emit_internal: Callable[[dict[str, Any]], None],
        exc: Exception,
    ) -> HookResult:
        meta = {
            "event": "hook_failure",
            "hook_event": context.event_name,
            "handler": handler.name,
            "required": handler.required,
            "blocking": context.blocking,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        if isinstance(exc, HookHandlerError):
            if exc.exit_code is not None:
                meta["exit_code"] = exc.exit_code
            if exc.stderr_tail is not None:
                meta["stderr_tail"] = exc.stderr_tail
        if isinstance(exc, subprocess.TimeoutExpired):
            meta["timeout"] = exc.timeout
            stderr = exc.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            if stderr:
                meta["stderr_tail"] = "\n".join(str(stderr).splitlines()[-20:])
        emit_internal(meta)
        if handler.required and context.blocking:
            return HookResult.veto(
                reason="handler_failed",
                message=(
                    f"required hook {handler.name!r} failed for "
                    f"{context.event_name}: {exc}"
                ),
                metadata=meta,
            )
        return HookResult.ok()

    def _handle_veto(
        self,
        handler: HookHandler,
        context: HookContext,
        result: HookResult,
        emit_internal: Callable[[dict[str, Any]], None],
    ) -> None:
        meta = {
            "event": "hook_veto" if context.blocking else "hook_ignored_veto",
            "hook_event": context.event_name,
            "handler": handler.name,
            "required": handler.required,
            "blocking": context.blocking,
            "reason": result.reason,
            "msg": result.message,
            "metadata": result.metadata,
        }
        emit_internal(meta)
        if context.blocking:
            raise HookVeto(
                handler=handler.name,
                event_name=context.event_name,
                reason=result.reason,
                message=result.message,
                metadata=result.metadata,
            )


def build_hook_dispatcher(
    hooks_cfg: dict[str, Any] | None,
    *,
    repo_root: Path,
) -> HookDispatcher:
    handlers: list[HookHandler] = []
    for spec in (hooks_cfg or {}).get("handlers", []) or []:
        kind = spec.get("type", "command")
        if kind != "command":
            raise ValueError(f"unknown hook handler type {kind!r}")
        handlers.append(
            CommandHookHandler(
                name=str(spec["name"]),
                events=list(spec["events"]),
                cmd=str(spec["cmd"]),
                required=bool(spec.get("required", False)),
                timeout=int(spec.get("timeout", 10)),
                cwd=repo_root,
            )
        )
    return HookDispatcher(handlers)
