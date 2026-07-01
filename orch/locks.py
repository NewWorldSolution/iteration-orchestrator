"""Per-iteration lock files for orchestrator run-state/worktree exclusivity."""
from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from orch.recovery import lock_recovery_note
from orch.state import utcnow_iso

RUN_LOCK_FILENAME = "run.lock"
RECOVER_LOCK_FILENAME = "recover.lock"
LOCK_GUARD_FILENAME = "lock.guard"
PidStatus = Literal["active", "stale", "unknown"]


class RunLockError(RuntimeError):
    """Raised when an iteration lock cannot be acquired."""


@dataclass(frozen=True)
class RunLockInfo:
    iteration: str
    iter_branch: str
    command: str
    pid: int
    hostname: str
    repo_root: str
    orch_workdir: str
    created_at: str
    token: str

    @classmethod
    def from_dict(cls, raw: dict) -> "RunLockInfo":
        return cls(
            iteration=str(raw.get("iteration") or ""),
            iter_branch=str(raw.get("iter_branch") or ""),
            command=str(raw.get("command") or ""),
            pid=int(raw.get("pid") or 0),
            hostname=str(raw.get("hostname") or ""),
            repo_root=str(raw.get("repo_root") or ""),
            orch_workdir=str(raw.get("orch_workdir") or ""),
            created_at=str(raw.get("created_at") or ""),
            token=str(raw.get("token") or ""),
        )

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "iter_branch": self.iter_branch,
            "command": self.command,
            "pid": self.pid,
            "hostname": self.hostname,
            "repo_root": self.repo_root,
            "orch_workdir": self.orch_workdir,
            "created_at": self.created_at,
            "token": self.token,
        }


def default_pid_status(pid: int, hostname: str) -> PidStatus:
    """Classify whether ``pid`` still appears alive on this host.

    Cross-host locks are treated as ``unknown`` because PID namespaces are
    host-local. Permission-denied checks are active enough for safety.
    """
    if pid <= 0:
        return "unknown"
    if hostname and hostname != socket.gethostname():
        return "unknown"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "stale"
    except PermissionError:
        return "active"
    except OSError:
        return "unknown"
    return "active"


class RunStateLock:
    """Atomic lock around one iteration's run state and orch worktree.

    The lock is intentionally cautious: an existing lock always blocks.
    When the recorded PID is gone, the diagnostic says "stale" and tells the
    operator which exact lock file to inspect/remove manually.
    """

    def __init__(
        self,
        *,
        log_dir: Path,
        iteration: str,
        iter_branch: str,
        command: str,
        repo_root: Path,
        orch_workdir: Path,
        pid_status: Callable[[int, str], PidStatus] = default_pid_status,
    ) -> None:
        self.log_dir = log_dir
        self.iteration = iteration
        self.iter_branch = iter_branch
        self.command = command
        self.repo_root = repo_root
        self.orch_workdir = orch_workdir
        self._pid_status = pid_status
        self._token = uuid.uuid4().hex
        self._acquired = False
        self.guard_path = log_dir / LOCK_GUARD_FILENAME
        self.recovery_path = log_dir / RECOVER_LOCK_FILENAME
        self.path = (
            self.recovery_path
            if command == "recover"
            else log_dir / RUN_LOCK_FILENAME
        )

    @property
    def acquired(self) -> bool:
        return self._acquired

    def acquire(self) -> "RunStateLock":
        self.log_dir.mkdir(parents=True, exist_ok=True)
        info = RunLockInfo(
            iteration=self.iteration,
            iter_branch=self.iter_branch,
            command=self.command,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            repo_root=str(self.repo_root),
            orch_workdir=str(self.orch_workdir),
            created_at=utcnow_iso(),
            token=self._token,
        )
        payload = json.dumps(info.to_dict(), indent=2, sort_keys=True) + "\n"
        try:
            guard_fd = self._acquire_guard(payload)
        except FileExistsError as exc:
            raise RunLockError(
                f"orch lock acquisition is already in progress at "
                f"{self.guard_path}. Wait for the other orchestrator command "
                "to finish acquiring its lock, or inspect that file if the "
                "process crashed."
            ) from exc
        try:
            if self.command != "recover" and self.recovery_path.exists():
                raise RunLockError(
                    f"orch recovery lock exists at {self.recovery_path}; "
                    "wait for `orch recover --apply` to finish before "
                    "starting another orchestrator command."
                )
            try:
                fd = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError as exc:
                raise RunLockError(
                    self._render_existing_lock_message()
                ) from exc
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(payload)
            except Exception:
                try:
                    self.path.unlink()
                except OSError:
                    pass
                raise
        finally:
            self._release_guard(guard_fd)
        self._acquired = True
        return self

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            raw = json.loads(self.path.read_text())
            current = RunLockInfo.from_dict(raw)
        except (OSError, ValueError, TypeError):
            self._acquired = False
            return
        if current.token == self._token:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._acquired = False

    def _acquire_guard(self, payload: str) -> int:
        try:
            fd = os.open(
                self.guard_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            if self._recover_stale_guard():
                fd = os.open(
                    self.guard_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            else:
                raise
        try:
            os.write(fd, payload.encode())
        except Exception:
            os.close(fd)
            try:
                self.guard_path.unlink()
            except OSError:
                pass
            raise
        return fd

    def _release_guard(self, fd: int) -> None:
        try:
            os.close(fd)
        finally:
            try:
                self.guard_path.unlink()
            except FileNotFoundError:
                pass

    def _recover_stale_guard(self) -> bool:
        info, error = self._read_lock_info(self.guard_path)
        if info is None:
            raise RunLockError(
                f"orch lock acquisition guard exists at {self.guard_path}, "
                f"but it could not be parsed ({error}). Next action: inspect "
                "that file and remove it only after confirming no "
                "orchestrator command is acquiring a lock."
            )
        status = self._pid_status(info.pid, info.hostname)
        if status != "stale":
            state = "active" if status == "active" else "unknown process state"
            raise RunLockError(
                f"orch lock acquisition guard with {state} exists at "
                f"{self.guard_path}; command={info.command!r}; pid={info.pid}; "
                f"host={info.hostname!r}; created_at={info.created_at!r}. "
                "Wait for the other orchestrator command to finish acquiring "
                "its lock, or inspect that file if the process crashed."
            )
        try:
            self.guard_path.unlink()
        except FileNotFoundError:
            pass
        return True

    def _read_lock_info(
        self, path: Path,
    ) -> tuple[RunLockInfo | None, str | None]:
        try:
            raw = json.loads(path.read_text())
            return RunLockInfo.from_dict(raw), None
        except FileNotFoundError:
            return None, "lock disappeared before it could be read"
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return None, f"malformed lock file: {exc}"

    def _read_existing(self) -> tuple[RunLockInfo | None, str | None]:
        return self._read_lock_info(self.path)

    def _render_existing_lock_message(self) -> str:
        info, error = self._read_existing()
        if info is None:
            return (
                f"orch run lock exists at {self.path}, but it could not be "
                f"parsed ({error}). Next action: inspect that file and remove "
                "it only after confirming no orchestrator run is active. "
                f"{lock_recovery_note(str(self.path))}"
            )

        status = self._pid_status(info.pid, info.hostname)
        if status == "stale":
            headline = "stale orch run lock"
            next_action = (
                f"No running process with pid {info.pid} was found on host "
                f"{info.hostname!r}. Remove {self.path} only after confirming "
                "no orchestrator run is active, then rerun the command."
            )
        elif status == "active":
            headline = "active orch run lock"
            next_action = (
                "Wait for the running orchestrator command to finish, or "
                "investigate that process before removing the lock."
            )
        else:
            headline = "orch run lock with unknown process state"
            next_action = (
                "The recorded process cannot be verified from this host. "
                "Confirm the other run is inactive before removing the lock."
            )

        return (
            f"{headline} at {self.path}; iteration={info.iteration!r}; "
            f"branch={info.iter_branch!r}; command={info.command!r}; "
            f"pid={info.pid}; host={info.hostname!r}; "
            f"created_at={info.created_at!r}; repo_root={info.repo_root!r}; "
            f"orch_workdir={info.orch_workdir!r}. Next action: {next_action} "
            f"{lock_recovery_note(str(self.path))}"
        )


__all__ = [
    "LOCK_GUARD_FILENAME",
    "RUN_LOCK_FILENAME",
    "RECOVER_LOCK_FILENAME",
    "RunLockError",
    "RunLockInfo",
    "RunStateLock",
    "default_pid_status",
]
