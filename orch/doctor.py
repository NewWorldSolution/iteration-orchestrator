"""Read-only environment doctor for configured orchestrator projects.

``orch doctor`` checks the local project pack, agent CLI reachability, reviewer
independence, git/GitHub CLI availability, and Python runtime dependencies
before a run starts.

Agent authentication is CLI-specific: a ``--version`` or ``--help`` probe only
proves installation, while a tiny real prompt is the cheapest portable signal
that also catches "not logged in" failures for Claude/Codex-style CLIs. The
probe stays injectable through :class:`orch.providers.CommandProvider`, so tests
never call real CLIs and operators can audit the exact command shape.
"""
from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from orch.agents import build_adapter
from orch.config import (
    ConfigError,
    LoadedConfig,
    default_project_yaml_path,
    load_config,
)
from orch.providers import CommandProvider, SubprocessCommandProvider
from orch.review import check_independence
from orch.task_execution import classify_impl_failure, stderr_tail

STATUS_OK = "OK"
STATUS_FAIL = "FAIL"
STATUS_WARN = "WARN"
STATUS_SKIP = "SKIP"
STATUS_NOT_FOUND = "NOT FOUND"
STATUS_NOT_AUTHENTICATED = "NOT AUTHENTICATED"
STATUS_ERROR = "ERROR"

AGENT_PROBE_PROMPT = "orch doctor: reply OK"
DEFAULT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    label: str
    status: str
    message: str = ""
    required: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in {STATUS_OK, STATUS_SKIP} or not self.required


@dataclass(frozen=True)
class DoctorReport:
    project_name: str
    repo_root: str
    checks: list[DoctorCheck]

    @property
    def failed_required(self) -> list[DoctorCheck]:
        return [check for check in self.checks if check.required and not check.ok]

    @property
    def warnings(self) -> list[DoctorCheck]:
        return [check for check in self.checks if check.status == STATUS_WARN]

    @property
    def ok(self) -> bool:
        return not self.failed_required

    @property
    def exit_code(self) -> int:
        return 0 if self.ok else 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "project_name": self.project_name,
            "repo_root": self.repo_root,
            "summary": {
                "failed_required": len(self.failed_required),
                "warnings": len(self.warnings),
            },
            "checks": [asdict(check) for check in self.checks],
        }


def run_doctor(
    repo_root: Path,
    *,
    command_provider: CommandProvider | None = None,
    dependency_importer: Callable[[str], Any] | None = None,
) -> DoctorReport:
    """Run every doctor check and return a structured report."""
    repo_root = repo_root.resolve()
    provider = command_provider or SubprocessCommandProvider()
    checks: list[DoctorCheck] = []

    cfg, project_check = check_project_pack(repo_root)
    checks.append(project_check)
    project_name = "iteration-orchestrator"
    if cfg is not None:
        project_name = str(
            cfg.data.get("project", {}).get("name") or project_name
        )
        checks.extend(check_agents(cfg, repo_root, provider))
        checks.append(check_reviewer_independence(cfg))
        checks.extend(check_git_and_gh(cfg, repo_root, provider))
    else:
        checks.append(
            DoctorCheck(
                name="agents",
                label="configured agents",
                status=STATUS_SKIP,
                message="project pack did not load",
                required=False,
            )
        )
        checks.append(
            DoctorCheck(
                name="independence",
                label="implementer/reviewer families",
                status=STATUS_SKIP,
                message="project pack did not load",
                required=False,
            )
        )
        checks.append(check_git(repo_root, provider))

    checks.append(check_python_deps(importer=dependency_importer))
    return DoctorReport(
        project_name=project_name,
        repo_root=str(repo_root),
        checks=checks,
    )


def check_project_pack(repo_root: Path) -> tuple[LoadedConfig | None, DoctorCheck]:
    path = default_project_yaml_path(repo_root)
    try:
        cfg = load_config(path)
    except ConfigError as exc:
        return None, DoctorCheck(
            name="project_pack",
            label=f"project pack ({path.relative_to(repo_root).as_posix()})",
            status=STATUS_FAIL,
            message=str(exc),
        )
    return cfg, DoctorCheck(
        name="project_pack",
        label=f"project pack ({path.relative_to(repo_root).as_posix()})",
        status=STATUS_OK,
        message="loaded",
        details={"path": str(path)},
    )


def check_agents(
    cfg: LoadedConfig,
    repo_root: Path,
    provider: CommandProvider,
) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []
    for name, raw_spec in cfg.data.get("agents", {}).items():
        spec = dict(raw_spec or {})
        try:
            adapter = build_adapter(name, spec)
        except ValueError as exc:
            checks.append(
                DoctorCheck(
                    name=f"agent:{name}",
                    label=f"agent '{name}'",
                    status=STATUS_FAIL,
                    message=str(exc),
                )
            )
            continue
        probe = agent_probe_argv(spec)
        checks.append(
            probe_agent(
                name,
                adapter_family=str(getattr(adapter, "family", spec.get("family", ""))),
                cmd=str(spec.get("cmd", "")),
                probe_argv=probe,
                repo_root=repo_root,
                provider=provider,
            )
        )
    return checks


def agent_probe_argv(spec: Mapping[str, Any]) -> list[str]:
    cmd = str(spec.get("cmd") or "").strip()
    if not cmd:
        return []
    kind = str(spec.get("type") or "shell").strip().lower()
    rendered = cmd.replace("{prompt}", os.devnull)
    argv = shlex.split(rendered)
    if kind in {"claude", "codex"}:
        return [*argv, AGENT_PROBE_PROMPT]
    return argv


def probe_agent(
    name: str,
    *,
    adapter_family: str,
    cmd: str,
    probe_argv: list[str],
    repo_root: Path,
    provider: CommandProvider,
) -> DoctorCheck:
    label = f"agent '{name}' ({cmd})"
    if not probe_argv:
        return DoctorCheck(
            name=f"agent:{name}",
            label=label,
            status=STATUS_FAIL,
            message="empty command",
            details={"family": adapter_family},
        )
    try:
        result = provider.run(
            probe_argv,
            cwd=repo_root,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return DoctorCheck(
            name=f"agent:{name}",
            label=label,
            status=STATUS_NOT_FOUND,
            message=f"command not found: {probe_argv[0]}",
            details={"family": adapter_family, "argv": probe_argv},
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            name=f"agent:{name}",
            label=label,
            status=STATUS_ERROR,
            message=f"probe timed out after {DEFAULT_TIMEOUT_SECONDS}s",
            details={"family": adapter_family, "argv": probe_argv},
        )
    except OSError as exc:
        return DoctorCheck(
            name=f"agent:{name}",
            label=label,
            status=STATUS_ERROR,
            message=str(exc),
            details={"family": adapter_family, "argv": probe_argv},
        )

    if result.returncode == 0:
        return DoctorCheck(
            name=f"agent:{name}",
            label=label,
            status=STATUS_OK,
            message="reachable",
            details={"family": adapter_family, "argv": probe_argv},
        )

    combined = "\n".join(
        part for part in (result.stderr, result.stdout) if part
    )
    bucket = classify_impl_failure(combined, int(result.returncode))
    if bucket == "auth":
        return DoctorCheck(
            name=f"agent:{name}",
            label=label,
            status=STATUS_NOT_AUTHENTICATED,
            message=stderr_tail(combined) or "authentication failed",
            details={
                "family": adapter_family,
                "argv": probe_argv,
                "exit_code": result.returncode,
            },
        )
    return DoctorCheck(
        name=f"agent:{name}",
        label=label,
        status=STATUS_ERROR,
        message=stderr_tail(combined) or f"exit code {result.returncode}",
        details={
            "family": adapter_family,
            "argv": probe_argv,
            "exit_code": result.returncode,
            "classification": bucket,
        },
    )


def check_reviewer_independence(cfg: LoadedConfig) -> DoctorCheck:
    agents = cfg.data.get("agents", {})
    impl, reviewer = resolve_default_agent_pair(agents)
    if impl is None or reviewer is None:
        return DoctorCheck(
            name="independence",
            label="implementer/reviewer families",
            status=STATUS_WARN,
            message="at least two configured agents are needed for independent review",
            required=False,
        )
    impl_family = str(agents[impl].get("family", ""))
    reviewer_family = str(agents[reviewer].get("family", ""))
    level = str(cfg.data.get("independence", {}).get("level", "model_family"))
    result = check_independence(
        impl_family,
        reviewer_family,
        level,
        implementer_name=impl,
        reviewer_name=reviewer,
    )
    if result.ok:
        return DoctorCheck(
            name="independence",
            label="implementer/reviewer families",
            status=STATUS_OK,
            message=f"{impl_family} vs {reviewer_family}",
            required=False,
            details={"implementer": impl, "reviewer": reviewer, "level": level},
        )
    return DoctorCheck(
        name="independence",
        label="implementer/reviewer families",
        status=STATUS_WARN,
        message=result.reason,
        required=False,
        details={"implementer": impl, "reviewer": reviewer, "level": level},
    )


def resolve_default_agent_pair(
    agents: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    names = list(agents)
    if not names:
        return None, None
    impl = names[0]
    impl_family = agents[impl].get("family")
    for name in names[1:]:
        if agents[name].get("family") != impl_family:
            return impl, name
    if len(names) > 1:
        return impl, names[1]
    return impl, impl


def check_git_and_gh(
    cfg: LoadedConfig,
    repo_root: Path,
    provider: CommandProvider,
) -> list[DoctorCheck]:
    checks = [check_git(repo_root, provider)]
    if gh_required(cfg):
        checks.append(check_gh_auth(repo_root, provider))
    else:
        checks.append(
            DoctorCheck(
                name="gh_auth",
                label="gh auth",
                status=STATUS_SKIP,
                message="auto_merge.no_ci=true",
                required=False,
            )
        )
    return checks


def gh_required(cfg: LoadedConfig) -> bool:
    return not bool(cfg.data.get("auto_merge", {}).get("no_ci", False))


def check_git(repo_root: Path, provider: CommandProvider) -> DoctorCheck:
    return _check_command(
        name="git",
        label="git",
        argv=["git", "--version"],
        repo_root=repo_root,
        provider=provider,
        timeout=10,
    )


def check_gh_auth(repo_root: Path, provider: CommandProvider) -> DoctorCheck:
    return _check_command(
        name="gh_auth",
        label="gh auth",
        argv=["gh", "auth", "status"],
        repo_root=repo_root,
        provider=provider,
        timeout=30,
        auth_sensitive=True,
    )


def _check_command(
    *,
    name: str,
    label: str,
    argv: list[str],
    repo_root: Path,
    provider: CommandProvider,
    timeout: int,
    auth_sensitive: bool = False,
) -> DoctorCheck:
    try:
        result = provider.run(argv, cwd=repo_root, timeout=timeout)
    except FileNotFoundError:
        return DoctorCheck(
            name=name,
            label=label,
            status=STATUS_NOT_FOUND,
            message=f"command not found: {argv[0]}",
        )
    except subprocess.TimeoutExpired:
        return DoctorCheck(
            name=name,
            label=label,
            status=STATUS_ERROR,
            message=f"probe timed out after {timeout}s",
        )
    except OSError as exc:
        return DoctorCheck(
            name=name,
            label=label,
            status=STATUS_ERROR,
            message=str(exc),
        )
    if result.returncode == 0:
        return DoctorCheck(
            name=name,
            label=label,
            status=STATUS_OK,
            message=(result.stdout or "").splitlines()[0] if result.stdout else "",
        )
    combined = "\n".join(
        part for part in (result.stderr, result.stdout) if part
    )
    if auth_sensitive and classify_impl_failure(combined, result.returncode) == "auth":
        status = STATUS_NOT_AUTHENTICATED
    else:
        status = STATUS_ERROR
    return DoctorCheck(
        name=name,
        label=label,
        status=status,
        message=stderr_tail(combined) or f"exit code {result.returncode}",
    )


def check_python_deps(
    *,
    importer: Callable[[str], Any] | None = None,
) -> DoctorCheck:
    importer = importer or importlib.import_module
    required = {"jsonschema": "jsonschema", "PyYAML": "yaml"}
    missing: list[str] = []
    for label, module in required.items():
        try:
            importer(module)
        except ImportError:
            missing.append(label)
    if missing:
        return DoctorCheck(
            name="python_deps",
            label="python deps",
            status=STATUS_FAIL,
            message="missing: " + ", ".join(missing),
        )
    return DoctorCheck(
        name="python_deps",
        label="python deps",
        status=STATUS_OK,
        message=", ".join(required),
    )


def render_report(report: DoctorReport) -> str:
    width = max(len(check.label) for check in report.checks)
    lines = [f"orch doctor - {report.project_name}"]
    for check in report.checks:
        dots = "." * max(2, 34 - len(check.label))
        line = f"  {check.label:<{width}} {dots} {check.status}"
        if check.message:
            line += f" - {check.message}"
        lines.append(line)
    if report.ok:
        if report.warnings:
            count = len(report.warnings)
            noun = "warning" if count == 1 else "warnings"
            lines.append(f"PASS: all required checks passed ({count} {noun}).")
        else:
            lines.append("PASS: all required checks passed.")
    else:
        count = len(report.failed_required)
        noun = "check" if count == 1 else "checks"
        verb = "needs" if count == 1 else "need"
        lines.append(f"FAIL: {count} required {noun} {verb} attention.")
    return "\n".join(lines) + "\n"


def render_json(report: DoctorReport) -> str:
    return json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n"
