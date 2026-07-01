"""Tests for orch.checks."""
from __future__ import annotations

import sys
from pathlib import Path

from orch.checks import (
    DEFAULT_NAV_ANCHOR_PATHS,
    DEFAULT_ROUTE_GLOBS,
    acceptance_test_command_is_noop,
    check_diff_size,
    check_forbidden_patterns,
    check_scope,
    check_sensitive_files,
    check_tasks_md_status_only,
    check_tasks_md_touched,
    detect_nav_anchor_updates,
    detect_route_visible_surfaces,
    is_nav_anchor_path,
    is_route_visible_path,
    load_nav_discoverability_evidence,
    parse_nav_discoverability_evidence,
    parse_scope_exception_evidence,
    run_acceptance,
)

EXAMPLE_ROUTE_GLOBS = ["app/routes/*.py", "app/templates/**/*.html"]
EXAMPLE_NAV_ANCHOR_PATHS = [
    "app/templates/base.html",
    "app/templates/_nav.html",
    "app/templates/partials/_nav.html",
    "app/templates/nav.html",
]


def test_check_scope_detects_outside_files():
    changed = ["app/a.py", "app/b.py", "docs/readme.md"]
    allowed = ["app/a.py", "app/b.py"]
    assert check_scope(changed, allowed) == ["docs/readme.md"]


def test_check_scope_all_allowed():
    assert check_scope(["a.py"], ["a.py", "b.py"]) == []


def test_check_tasks_md_touched():
    assert check_tasks_md_touched(
        ["iterations/demo/i1/tasks.md", "app/x.py"],
        "iterations/demo/i1/tasks.md",
    )
    assert not check_tasks_md_touched(["app/x.py"], "iterations/demo/i1/tasks.md")


def test_tasks_md_status_only_allows_only_status_cell():
    base = (
        "| ID | Title | Owner | Status  | Depends on | Branch |\n"
        "| I1-T1 | First | TBD | WAITING | \u2014 | d/i1/t1 |\n"
    )
    head = (
        "| ID | Title | Owner | Status  | Depends on | Branch |\n"
        "| I1-T1 | First | TBD | DONE    | \u2014 | d/i1/t1 |\n"
    )
    assert check_tasks_md_status_only(base, head, ["I1-T1"])


def test_tasks_md_status_only_rejects_non_status_change():
    base = "| I1-T1 | First | TBD | WAITING | \u2014 | d/i1/t1 |\n"
    head = "| I1-T1 | Renamed | TBD | DONE    | \u2014 | d/i1/t1 |\n"
    assert not check_tasks_md_status_only(base, head, ["I1-T1"])


def test_parse_scope_exception_evidence_requires_exact_approval():
    evidence = parse_scope_exception_evidence(
        "\n".join([
            "approved_by: operator",
            "reason: approved fold-in from review",
            "paths:",
            "- docs/approved.md",
        ]),
        source="scope_exceptions.md",
    )
    assert evidence.ok
    assert evidence.approved_by == "operator"
    assert evidence.reason == "approved fold-in from review"
    assert evidence.approved_paths == ["docs/approved.md"]


def test_parse_scope_exception_evidence_rejects_vague_or_blanket_paths():
    evidence = parse_scope_exception_evidence(
        "\n".join([
            "approved_by: operator",
            "reason: broad approval",
            "paths:",
            "- docs/**",
        ]),
        source="scope_exceptions.md",
    )
    assert not evidence.ok
    assert evidence.approved_paths == []
    assert any("glob chars" in err for err in evidence.errors)


def test_forbidden_patterns_scan_added_lines_only():
    diff = "\n".join([
        "diff --git a/x b/x",
        "+++ b/x",
        "+    except: pass",
        "-    DROP DATABASE x",    # removed — ignored
    ])
    hits = check_forbidden_patterns(diff, ["except: pass", "DROP DATABASE"])
    assert hits == ["except: pass"]


def test_sensitive_files_matches_basename_and_path():
    changed = ["app/.env", "secrets.yaml", "docs/x.md"]
    hits = check_sensitive_files(changed, [".env", "secrets.yaml"])
    assert set(hits) == {"app/.env", "secrets.yaml"}


def test_diff_size_check():
    assert check_diff_size(1600, 1500)
    assert not check_diff_size(1500, 1500)
    assert not check_diff_size(10, 1500)


# ---------------------------------------------------------------------------
# run_acceptance — uses real subprocess with python -c commands
# ---------------------------------------------------------------------------


def test_noop_acceptance_command_detection():
    for cmd in (None, "", "   ", "true", " true ", ":", "/bin/true"):
        assert acceptance_test_command_is_noop({"test": cmd})


def test_real_acceptance_command_is_not_noop():
    assert not acceptance_test_command_is_noop(
        {"test": f"{sys.executable} -c \"print('ok')\""}
    )
    assert not acceptance_test_command_is_noop({"test": "pytest -q"})


def test_noop_acceptance_detection_uses_effective_override():
    stack = {"test": "pytest -q"}

    assert acceptance_test_command_is_noop(
        stack,
        test_cmd_override="true",
    )
    assert not acceptance_test_command_is_noop(
        {"test": "true"},
        test_cmd_override=f"{sys.executable} -c \"print('scoped')\"",
    )


def test_run_acceptance_all_pass(tmp_path: Path):
    stack = {
        "test": f"{sys.executable} -c \"print('ok')\"",
        "lint": f"{sys.executable} -c \"import sys; sys.exit(0)\"",
        "typecheck": None,
        "build": None,
        "test_env": {"ENVIRONMENT": "test"},
    }
    report = run_acceptance(stack, cwd=tmp_path, timeout=30)
    assert report.ok
    assert [r.name for r in report.results] == ["test", "lint"]


def test_run_acceptance_failure_captured(tmp_path: Path):
    stack = {
        "test": f"{sys.executable} -c \"import sys; print('boom'); sys.exit(2)\"",
        "lint": f"{sys.executable} -c \"print('clean')\"",
    }
    report = run_acceptance(stack, cwd=tmp_path, timeout=30)
    assert not report.ok
    assert len(report.failures) == 1
    assert report.failures[0].name == "test"
    assert report.failures[0].exit_code == 2
    assert "boom" in report.combined_output()


def test_run_acceptance_timeout(tmp_path: Path):
    stack = {
        "test": f"{sys.executable} -c \"import time; time.sleep(10)\"",
        "lint": None,
    }
    report = run_acceptance(stack, cwd=tmp_path, timeout=1)
    assert not report.ok
    assert report.results[0].timed_out
    assert report.results[0].duration_s < 8


def test_run_acceptance_injects_test_env(tmp_path: Path):
    stack = {
        "test": (
            f"{sys.executable} -c \"import os,sys; "
            f"sys.exit(0 if os.environ.get('ENVIRONMENT')=='test' else 3)\""
        ),
        "lint": None,
        "test_env": {"ENVIRONMENT": "test"},
    }
    report = run_acceptance(stack, cwd=tmp_path, timeout=15)
    assert report.ok


# ---------------------------------------------------------------------------
# Nav-discoverability inward gap
# ---------------------------------------------------------------------------


def test_default_route_visibility_is_inert_without_project_override():
    assert DEFAULT_ROUTE_GLOBS == ()
    assert DEFAULT_NAV_ANCHOR_PATHS == ()
    assert not is_route_visible_path("app/routes/transactions.py")
    assert not is_route_visible_path("app/templates/base.html")
    assert not is_nav_anchor_path("app/templates/base.html")
    assert detect_route_visible_surfaces([
        "app/routes/transactions.py",
        "app/templates/transactions/list.html",
    ]) == []
    assert detect_nav_anchor_updates(["app/templates/base.html"]) == []


def test_example_route_visibility_matches_top_level_route_files():
    assert is_route_visible_path(
        "app/routes/transactions.py",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert is_route_visible_path(
        "app/routes/admin_audit.py",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )


def test_example_route_visibility_skips_init_and_nested_package_files():
    assert not is_route_visible_path(
        "app/routes/__init__.py",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert not is_route_visible_path(
        "app/routes/admin/dashboard.py",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert not is_route_visible_path(
        "nested/app/routes/foo.py",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )


def test_example_route_visibility_matches_templates():
    assert is_route_visible_path(
        "app/templates/transactions/list.html",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert is_route_visible_path(
        "app/templates/base.html",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert not is_route_visible_path(
        "nested/app/templates/base.html",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )


def test_example_route_visibility_rejects_other_files():
    assert not is_route_visible_path(
        "app/services/balance.py",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert not is_route_visible_path(
        "docs/transactions.md",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert not is_route_visible_path(
        "static/main.css",
        route_globs=EXAMPLE_ROUTE_GLOBS,
    )
    assert not is_route_visible_path("")


def test_example_nav_anchor_path_matches_base_template():
    assert is_nav_anchor_path(
        "app/templates/base.html",
        nav_anchor_paths=EXAMPLE_NAV_ANCHOR_PATHS,
    )
    assert not is_nav_anchor_path(
        "app/templates/transactions/list.html",
        nav_anchor_paths=EXAMPLE_NAV_ANCHOR_PATHS,
    )
    assert not is_nav_anchor_path(
        "app/routes/transactions.py",
        nav_anchor_paths=EXAMPLE_NAV_ANCHOR_PATHS,
    )


def test_detect_route_visible_surfaces_collects_and_dedupes():
    changed = [
        "app/routes/transactions.py",
        "app/services/balance.py",
        "app/templates/transactions/list.html",
        "app/templates/transactions/list.html",  # dup
        "tests/test_transactions.py",
    ]
    assert detect_route_visible_surfaces(
        changed,
        route_globs=EXAMPLE_ROUTE_GLOBS,
    ) == [
        "app/routes/transactions.py",
        "app/templates/transactions/list.html",
    ]


def test_detect_nav_anchor_updates_collects_base_template():
    changed = [
        "app/templates/base.html",
        "app/routes/transactions.py",
    ]
    assert detect_nav_anchor_updates(
        changed,
        nav_anchor_paths=EXAMPLE_NAV_ANCHOR_PATHS,
    ) == ["app/templates/base.html"]


def test_route_visibility_uses_configured_globs_and_nav_paths():
    changed = [
        "src/pages/billing.py",
        "src/pages/__init__.py",
        "src/templates/billing.html",
        "app/routes/transactions.py",
        "src/layout/nav.html",
    ]

    assert detect_route_visible_surfaces(
        changed,
        route_globs=["src/pages/*.py", "src/templates/**/*.html"],
    ) == ["src/pages/billing.py", "src/templates/billing.html"]
    assert detect_nav_anchor_updates(
        changed,
        nav_anchor_paths=["src/layout/nav.html"],
    ) == ["src/layout/nav.html"]
    assert not is_route_visible_path(
        "app/routes/transactions.py",
        route_globs=["src/pages/*.py"],
    )
    assert is_nav_anchor_path(
        "src/layout/nav.html",
        nav_anchor_paths=["src/layout/nav.html"],
    )


def test_parse_nav_discoverability_evidence_valid():
    evidence = parse_nav_discoverability_evidence(
        "\n".join([
            "approved_by: operator",
            "reason: admin sub-page reached from admin dashboard, not nav",
            "paths:",
            "- app/routes/admin_audit.py",
        ]),
        source="nav_discoverability.md",
    )
    assert evidence.ok
    assert evidence.approved_by == "operator"
    assert (
        evidence.reason
        == "admin sub-page reached from admin dashboard, not nav"
    )
    assert evidence.approved_paths == ["app/routes/admin_audit.py"]


def test_parse_nav_discoverability_evidence_rejects_missing_approver():
    evidence = parse_nav_discoverability_evidence(
        "\n".join([
            "reason: missing approval identity",
            "paths:",
            "- app/routes/admin_audit.py",
        ]),
        source="nav_discoverability.md",
    )
    assert not evidence.ok
    assert evidence.approved_paths == []
    assert any("missing approved_by" in err for err in evidence.errors)


def test_parse_nav_discoverability_evidence_rejects_missing_reason():
    evidence = parse_nav_discoverability_evidence(
        "\n".join([
            "approved_by: operator",
            "paths:",
            "- app/routes/admin_audit.py",
        ]),
        source="nav_discoverability.md",
    )
    assert not evidence.ok
    assert any("missing reason" in err for err in evidence.errors)


def test_parse_nav_discoverability_evidence_rejects_blanket_path():
    evidence = parse_nav_discoverability_evidence(
        "\n".join([
            "approved_by: operator",
            "reason: blanket exemption attempt",
            "paths:",
            "- app/routes/**",
        ]),
        source="nav_discoverability.md",
    )
    assert not evidence.ok
    assert any("glob chars" in err for err in evidence.errors)


def test_parse_nav_discoverability_evidence_rejects_absolute_path():
    evidence = parse_nav_discoverability_evidence(
        "\n".join([
            "approved_by: operator",
            "reason: absolute path attempt",
            "paths:",
            "- /etc/passwd",
        ]),
        source="nav_discoverability.md",
    )
    assert not evidence.ok
    assert any("relative" in err for err in evidence.errors)


def test_parse_nav_discoverability_evidence_requires_at_least_one_path():
    evidence = parse_nav_discoverability_evidence(
        "\n".join([
            "approved_by: operator",
            "reason: forgot to list paths",
            "paths:",
        ]),
        source="nav_discoverability.md",
    )
    assert not evidence.ok
    assert any("missing paths entries" in err for err in evidence.errors)


def test_load_nav_discoverability_evidence_missing_file_is_valid_empty(
    tmp_path: Path,
):
    evidence = load_nav_discoverability_evidence(
        tmp_path / "nav_discoverability.md"
    )
    # Missing file is an empty, valid sentinel — the gate then falls back
    # to in-diff nav-anchor detection, not an evidence-required block.
    assert evidence.approved_paths == []
    assert evidence.approved_by is None
    assert evidence.reason is None
    # No errors (missing file is not malformed evidence).
    assert evidence.errors == ()
