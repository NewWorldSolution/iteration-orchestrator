"""Tests for the read-only orch doctor command."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orch import cli as cli_mod
from orch import doctor as doctor_mod


PROJECT_YAML = """\
project:
  name: demo-project
  main_branch: main
  phase_branch_pattern: "phase-{phase}"
  iteration_branch_pattern: "{phase}/iteration-{n}"
  task_branch_pattern: "{phase}/i{n}/t{k}-{slug}"
stack:
  test: "pytest -q"
  lint: "ruff check ."
risk:
  high_risk_globs: []
  sensitive_files: []
  forbidden_patterns: []
agents:
  claude:
    type: claude
    cmd: "claude -p"
    family: anthropic
  codex:
    type: codex
    cmd: "codex exec"
    family: openai
costs:
  anthropic: {input: 3.0, output: 15.0}
  openai: {input: 2.5, output: 10.0}
auto_merge:
  no_ci: false
"""


def _write_project(tmp_path: Path, text: str = PROJECT_YAML) -> Path:
    (tmp_path / ".orch").mkdir()
    (tmp_path / ".orch" / "project.yaml").write_text(text, encoding="utf-8")
    return tmp_path


class FakeCommandProvider:
    def __init__(self, responses: dict[tuple[str, ...], object]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def run(self, argv, *, cwd: Path, timeout: int | None = None):
        key = tuple(argv)
        self.calls.append({"argv": list(argv), "cwd": cwd, "timeout": timeout})
        response = self.responses.get(key)
        if isinstance(response, BaseException):
            raise response
        if response is None:
            raise AssertionError(f"unexpected command: {list(argv)!r}")
        return response


def _proc(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _passing_provider() -> FakeCommandProvider:
    return FakeCommandProvider(
        {
            ("claude", "-p", doctor_mod.AGENT_PROBE_PROMPT): _proc(stdout="ok\n"),
            ("codex", "exec", doctor_mod.AGENT_PROBE_PROMPT): _proc(stdout="ok\n"),
            ("git", "--version"): _proc(stdout="git version 2.0\n"),
            ("gh", "auth", "status"): _proc(stdout="Logged in\n"),
        }
    )


def test_doctor_all_pass_reports_ok(tmp_path: Path):
    repo = _write_project(tmp_path)
    provider = _passing_provider()

    report = doctor_mod.run_doctor(repo, command_provider=provider)
    text = doctor_mod.render_report(report)

    assert report.ok
    assert report.exit_code == 0
    assert "agent 'claude' (claude -p)" in text
    assert "gh auth" in text
    assert "PASS: all required checks passed." in text
    assert ["gh", "auth", "status"] in [call["argv"] for call in provider.calls]


def test_agent_probe_classifies_not_found(tmp_path: Path):
    provider = FakeCommandProvider({("missing-cli",): FileNotFoundError()})

    check = doctor_mod.probe_agent(
        "missing",
        adapter_family="local",
        cmd="missing-cli",
        probe_argv=["missing-cli"],
        repo_root=tmp_path,
        provider=provider,
    )

    assert check.status == doctor_mod.STATUS_NOT_FOUND
    assert "missing-cli" in check.message


def test_agent_probe_classifies_not_authenticated(tmp_path: Path):
    provider = FakeCommandProvider(
        {("codex", "exec", doctor_mod.AGENT_PROBE_PROMPT): _proc(1, stderr="not logged in")}
    )

    check = doctor_mod.probe_agent(
        "codex",
        adapter_family="openai",
        cmd="codex exec",
        probe_argv=["codex", "exec", doctor_mod.AGENT_PROBE_PROMPT],
        repo_root=tmp_path,
        provider=provider,
    )

    assert check.status == doctor_mod.STATUS_NOT_AUTHENTICATED
    assert "not logged in" in check.message


def test_project_pack_invalid_fails_required(tmp_path: Path):
    repo = _write_project(tmp_path, "project:\n  name: broken\n")
    provider = FakeCommandProvider(
        {("git", "--version"): _proc(stdout="git version 2.0\n")}
    )

    report = doctor_mod.run_doctor(repo, command_provider=provider)

    assert not report.ok
    assert report.exit_code == 1
    project = next(check for check in report.checks if check.name == "project_pack")
    assert project.status == doctor_mod.STATUS_FAIL
    assert "missing required key" in project.message


def test_python_deps_missing_fails_required():
    def importer(name: str):
        if name == "yaml":
            raise ImportError(name)
        return object()

    check = doctor_mod.check_python_deps(importer=importer)

    assert check.status == doctor_mod.STATUS_FAIL
    assert "PyYAML" in check.message


def test_same_family_independence_warns_without_failing(tmp_path: Path):
    same_family = PROJECT_YAML.replace(
        "    family: openai",
        "    family: anthropic",
    ).replace(
        "  openai: {input: 2.5, output: 10.0}\n",
        "",
    ).replace("no_ci: false", "no_ci: true")
    repo = _write_project(tmp_path, same_family)
    provider = FakeCommandProvider(
        {
            ("claude", "-p", doctor_mod.AGENT_PROBE_PROMPT): _proc(stdout="ok\n"),
            ("codex", "exec", doctor_mod.AGENT_PROBE_PROMPT): _proc(stdout="ok\n"),
            ("git", "--version"): _proc(stdout="git version 2.0\n"),
        }
    )

    report = doctor_mod.run_doctor(repo, command_provider=provider)
    independence = next(check for check in report.checks if check.name == "independence")

    assert report.ok
    assert independence.status == doctor_mod.STATUS_WARN
    assert "model_family" in independence.message


def test_json_shape_and_cmd_doctor_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    repo = _write_project(tmp_path)
    provider = FakeCommandProvider(
        {
            ("claude", "-p", doctor_mod.AGENT_PROBE_PROMPT): _proc(stdout="ok\n"),
            ("codex", "exec", doctor_mod.AGENT_PROBE_PROMPT): _proc(
                1,
                stderr="Authentication failed: login required",
            ),
            ("git", "--version"): _proc(stdout="git version 2.0\n"),
            ("gh", "auth", "status"): _proc(stdout="Logged in\n"),
        }
    )
    monkeypatch.chdir(repo)

    rc = cli_mod.cmd_doctor(
        SimpleNamespace(json=True, command_provider=provider)
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["ok"] is False
    assert payload["summary"]["failed_required"] == 1
    codex = next(
        check for check in payload["checks"] if check["name"] == "agent:codex"
    )
    assert codex["status"] == doctor_mod.STATUS_NOT_AUTHENTICATED


def _report_with_checks(checks: list[doctor_mod.DoctorCheck]) -> doctor_mod.DoctorReport:
    return doctor_mod.DoctorReport(
        project_name="summary-test",
        repo_root="/tmp/summary-test",
        checks=checks,
    )


def _summary_line(report: doctor_mod.DoctorReport) -> str:
    return doctor_mod.render_report(report).splitlines()[-1]


def test_doctor_summary_pluralizes_required_failures():
    one_fail = _report_with_checks([
        doctor_mod.DoctorCheck(
            name="pack",
            label="project pack",
            status=doctor_mod.STATUS_FAIL,
        )
    ])
    two_fail = _report_with_checks([
        doctor_mod.DoctorCheck(
            name="pack",
            label="project pack",
            status=doctor_mod.STATUS_FAIL,
        ),
        doctor_mod.DoctorCheck(
            name="deps",
            label="python deps",
            status=doctor_mod.STATUS_FAIL,
        ),
    ])

    assert _summary_line(one_fail) == "FAIL: 1 required check needs attention."
    assert _summary_line(two_fail) == "FAIL: 2 required checks need attention."


def test_doctor_summary_pluralizes_warnings():
    no_warning = _report_with_checks([
        doctor_mod.DoctorCheck(
            name="pack",
            label="project pack",
            status=doctor_mod.STATUS_OK,
        )
    ])
    one_warning = _report_with_checks([
        doctor_mod.DoctorCheck(
            name="pack",
            label="project pack",
            status=doctor_mod.STATUS_OK,
        ),
        doctor_mod.DoctorCheck(
            name="independence",
            label="implementer/reviewer families",
            status=doctor_mod.STATUS_WARN,
            required=False,
        ),
    ])
    two_warnings = _report_with_checks([
        doctor_mod.DoctorCheck(
            name="pack",
            label="project pack",
            status=doctor_mod.STATUS_OK,
        ),
        doctor_mod.DoctorCheck(
            name="independence",
            label="implementer/reviewer families",
            status=doctor_mod.STATUS_WARN,
            required=False,
        ),
        doctor_mod.DoctorCheck(
            name="gh",
            label="gh auth",
            status=doctor_mod.STATUS_WARN,
            required=False,
        ),
    ])

    assert _summary_line(no_warning) == "PASS: all required checks passed."
    assert _summary_line(one_warning) == (
        "PASS: all required checks passed (1 warning)."
    )
    assert _summary_line(two_warnings) == (
        "PASS: all required checks passed (2 warnings)."
    )
