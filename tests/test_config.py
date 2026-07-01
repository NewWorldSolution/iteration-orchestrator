"""Tests for orch.config."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from orch.config import (
    CORE_DEFAULTS,
    TASK_SCHEMA_POLICY_FLOOR,
    ConfigError,
    LoadedConfig,
    _deep_merge,
    agents,
    auto_merge,
    costs,
    effective_task_schema_policy,
    hooks,
    independence,
    invariants,
    limits,
    load_config,
    load_project_yaml,
    model_routing,
    parallel,
    paths as path_settings,
    patterns,
    review,
    risk,
    scaffold,
    stack,
    task_schema_policy,
    tasks_md,
    templates,
    timeouts,
    ui_route_visibility,
)

VALID_PROJECT_YAML = """\
project:
  name: demo
  main_branch: main
  phase_branch_pattern: "phase-{phase}"
  iteration_branch_pattern: "{phase}/iteration-{n}"
  task_branch_pattern: "{phase}/i{n}/t{k}-{slug}"

stack:
  test: "pytest -q"
  lint: "ruff check ."

risk:
  high_risk_globs: ["**/schema.sql"]
  sensitive_files: [".env"]
  forbidden_patterns: ["except: pass"]

agents:
  claude: {cmd: "claude -p", family: anthropic}
  codex:  {cmd: "codex",     family: openai}

costs:
  anthropic: {input: 3.0, output: 15.0}
  openai:    {input: 2.5, output: 10.0}
"""

GENERIC_SCAFFOLD_DEFAULTS = {
    "post_phase_docs_root": "docs/post-phase",
    "post_phase_iteration_root": "iterations/post-phase",
    "tooling_iteration_root": "iterations/tools",
    "post_phase_integration_branch": "post-phase-integration",
}

EXAMPLE_SCAFFOLD = {
    "post_phase_docs_root": "docs/post-phase-organization",
    "post_phase_iteration_root": "iterations/post-phase",
    "tooling_iteration_root": "iterations/tools",
    "post_phase_integration_branch": "post-phase-integration",
}

EXAMPLE_UI_ROUTE_VISIBILITY = {
    "route_globs": ["app/routes/*.py", "app/templates/**/*.html"],
    "nav_anchor_paths": [
        "app/templates/base.html",
        "app/templates/_nav.html",
    ],
}

GENERIC_PATTERNS = {
    "task_id": r"^TASK-(\d+)-(\d+)$",
    "task_detail_heading": (
        r"^###\s+(?P<id>TASK-\d+-\d+)\s+—\s+(?P<title>.+?)\s*$"
    ),
    "iteration_id": (
        r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$"
    ),
    "phase_branch": (
        r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*$"
    ),
}

EXAMPLE_PATTERNS = {
    **GENERIC_PATTERNS,
    "task_id": r"^I(\d+)-T(\d+)$",
    "task_detail_heading": (
        r"^###\s+(?P<id>I\d+-T\d+)\s+—\s+(?P<title>.+?)\s*$"
    ),
    "phase_branch": r"^phase-[A-Za-z0-9][A-Za-z0-9_-]*$",
}

UNIVERSAL_TASK_SCHEMA_FLOOR = {
    key: list(values)
    for key, values in TASK_SCHEMA_POLICY_FLOOR.items()
}

EXAMPLE_TASK_SCHEMA_POLICY = {
    **UNIVERSAL_TASK_SCHEMA_FLOOR,
    "planning_refusal_prefixes": [
        *UNIVERSAL_TASK_SCHEMA_FLOOR["planning_refusal_prefixes"],
        "app/",
        "db/",
        "tests/",
        "static/",
        "seed/",
        "migrations/",
        "migration/",
    ],
}

ACCESSOR_CASES = [
    ("agents", agents),
    ("costs", costs),
    ("limits", limits),
    ("auto_merge", auto_merge),
    ("timeouts", timeouts),
    ("independence", independence),
    ("risk", risk),
    ("stack", stack),
    ("review", review),
    ("tasks_md", tasks_md),
    ("tasks_schema", task_schema_policy),
    ("paths", path_settings),
    ("templates", templates),
    ("invariants", invariants),
    ("scaffold", scaffold),
    ("ui_route_visibility", ui_route_visibility),
    ("parallel", parallel),
    ("model_routing", model_routing),
    ("patterns", patterns),
    ("hooks", hooks),
]


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "project.yaml"
    p.write_text(content)
    return p


def _config_freeze_fixture(name: str) -> dict:
    path = Path(__file__).parent / "fixtures" / "config_freeze" / name
    return json.loads(path.read_text())


def test_load_valid_project_yaml(tmp_path: Path):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    # Core defaults present
    assert cfg["limits"]["impl_attempts"] == 3
    assert cfg["limits"]["fix_rounds_acceptance"] == 3
    assert cfg["limits"]["review_rounds"] == 2
    # Timeout defaults
    assert cfg["timeouts"]["impl_high"] == 2700
    assert cfg["timeouts"]["impl_medium"] == 1800
    assert cfg["timeouts"]["impl_low"] == 900
    assert cfg["timeouts"]["fix_high"] == 1200
    # Auto-merge defaults
    assert cfg["auto_merge"]["max_fix_rounds_default"] == 1
    assert cfg["auto_merge"]["max_fix_rounds_high_risk"] == 0
    assert cfg["auto_merge"]["no_ci"] is False
    assert cfg["hooks"]["handlers"] == []
    assert cfg["model_routing"]["agent_overrides"] == {}
    assert cfg["invariants"] == []
    assert cfg["timeouts"]["task_kind_profiles"] == {}
    # Project data merged in
    assert cfg["project"]["name"] == "demo"
    assert cfg["agents"]["claude"]["family"] == "anthropic"


def test_project_overrides_core(tmp_path: Path):
    override = VALID_PROJECT_YAML + (
        "\nauto_merge:\n  max_diff_insertions: 750\n"
    )
    cfg = load_config(_write(tmp_path, override))
    assert cfg["auto_merge"]["max_diff_insertions"] == 750
    # Other defaults survive
    assert cfg["auto_merge"]["ci_wait_seconds"] == 300


def test_config_auto_merge_no_ci_default_and_override(tmp_path: Path):
    default_cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    assert CORE_DEFAULTS["auto_merge"]["no_ci"] is False
    assert auto_merge(default_cfg)["no_ci"] is False

    override = VALID_PROJECT_YAML + "\nauto_merge:\n  no_ci: true\n"
    override_cfg = load_config(_write(tmp_path, override))
    assert auto_merge(override_cfg)["no_ci"] is True


def test_invariants_default_and_override(tmp_path: Path):
    default_cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    assert invariants(default_cfg) == []

    override = VALID_PROJECT_YAML + """
invariants:
  - name: Access boundary
    applies: applies
    evidence: pytest tests/test_access.py -q
    status: "<PASS/FAIL/N/A>"
"""
    override_cfg = load_config(_write(tmp_path, override))

    assert invariants(override_cfg) == [
        {
            "name": "Access boundary",
            "applies": "applies",
            "evidence": "pytest tests/test_access.py -q",
            "status": "<PASS/FAIL/N/A>",
        }
    ]


@pytest.mark.parametrize(
    ("snippet", "match"),
    [
        ("invariants: {}\n", "section 'invariants' must be a list"),
        ("invariants:\n  - not-a-mapping\n", "invariants\\[0\\] must be a mapping"),
        ("invariants:\n  - applies: applies\n", "invariants\\[0\\]\\.name"),
        (
            "invariants:\n  - name: Bad|Cell\n",
            "must not contain newlines or '\\|'",
        ),
    ],
)
def test_invariants_validation_fails_closed(
    tmp_path: Path,
    snippet: str,
    match: str,
):
    with pytest.raises(ConfigError, match=match):
        load_project_yaml(_write(tmp_path, VALID_PROJECT_YAML + "\n" + snippet))


def test_parallel_max_concurrency_default_and_override(tmp_path: Path):
    default_cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))

    assert CORE_DEFAULTS["parallel"]["max_concurrency"] == 1
    assert parallel(default_cfg)["max_concurrency"] == 1

    override = VALID_PROJECT_YAML + "\nparallel:\n  max_concurrency: 3\n"
    override_cfg = load_config(_write(tmp_path, override))

    assert parallel(override_cfg)["max_concurrency"] == 3


@pytest.mark.parametrize("value", ["0", "-1", "true"])
def test_parallel_max_concurrency_must_be_positive_integer(
    tmp_path: Path, value: str
):
    config = VALID_PROJECT_YAML + f"\nparallel:\n  max_concurrency: {value}\n"

    with pytest.raises(ConfigError, match="parallel.max_concurrency"):
        load_project_yaml(_write(tmp_path, config))


def test_model_routing_agent_overrides_default_and_override(tmp_path: Path):
    override = VALID_PROJECT_YAML + """
model_routing:
  agent_overrides:
    codex:
      max:
        high:
          args: ["--model", "fixture-codex-model"]
          env:
            ORCH_REASONING_EFFORT: "high"
"""

    cfg = load_config(_write(tmp_path, override))

    assert model_routing(cfg)["agent_overrides"]["codex"]["max"]["high"] == {
        "args": ["--model", "fixture-codex-model"],
        "env": {"ORCH_REASONING_EFFORT": "high"},
    }


@pytest.mark.parametrize(
    ("snippet", "match"),
    [
        (
            """
model_routing:
  agent_overrides:
    unknown:
      max:
        high: {}
""",
            "unknown agent",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      tiny:
        high: {}
""",
            "invalid tier",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      max:
        extreme: {}
""",
            "invalid effort",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      max:
        high:
          args: "--model fixture"
""",
            "args must be a list",
        ),
    ],
)
def test_invalid_model_routing_config_fails_closed(
    tmp_path: Path, snippet: str, match: str
):
    with pytest.raises(ConfigError, match=match):
        load_project_yaml(_write(tmp_path, VALID_PROJECT_YAML + snippet))


def test_task_kind_timeout_profiles_default_and_override(tmp_path: Path):
    override = VALID_PROJECT_YAML + """
timeouts:
  task_kind_profiles:
    characterization:
      impl: 2400
      fix: 1200
"""

    cfg = load_config(_write(tmp_path, override))

    assert timeouts(cfg)["task_kind_profiles"]["characterization"] == {
        "impl": 2400,
        "fix": 1200,
    }


@pytest.mark.parametrize("value", ["0", "-1", "true"])
def test_task_kind_timeout_profile_values_must_be_positive_integers(
    tmp_path: Path, value: str
):
    config = VALID_PROJECT_YAML + f"""
timeouts:
  task_kind_profiles:
    characterization:
      impl: {value}
"""

    with pytest.raises(ConfigError, match="task_kind_profiles"):
        load_project_yaml(_write(tmp_path, config))


def test_config_timeouts_include_qa_and_retro_defaults(tmp_path: Path):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))

    assert CORE_DEFAULTS["timeouts"]["qa"] == 900
    assert CORE_DEFAULTS["timeouts"]["retro"] == 900
    assert timeouts(cfg)["qa"] == 900
    assert timeouts(cfg)["retro"] == 900

    override = VALID_PROJECT_YAML + "\ntimeouts:\n  qa: 321\n  retro: 654\n"
    override_cfg = load_config(_write(tmp_path, override))
    assert timeouts(override_cfg)["qa"] == 321
    assert timeouts(override_cfg)["retro"] == 654


def test_config_freeze_defaults_include_generic_blueprint_sections(
    tmp_path: Path,
):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    golden = _config_freeze_fixture("default_sections.json")

    assert path_settings(cfg) == golden["paths"]
    assert templates(cfg) == golden["templates"]
    assert invariants(cfg) == []
    assert task_schema_policy(cfg) == UNIVERSAL_TASK_SCHEMA_FLOOR
    assert scaffold(cfg) == GENERIC_SCAFFOLD_DEFAULTS
    assert ui_route_visibility(cfg) == {
        "route_globs": [],
        "nav_anchor_paths": [],
    }
    assert patterns(cfg)["task_id"] == golden["patterns"]["task_id"]
    assert patterns(cfg) == GENERIC_PATTERNS
    assert sorted(patterns(cfg)) == sorted(golden["patterns"]["pattern_keys"])


def test_core_defaults_route_nav_and_scaffold_are_generic():
    relocated_defaults = {
        "scaffold": CORE_DEFAULTS["scaffold"],
        "ui_route_visibility": CORE_DEFAULTS["ui_route_visibility"],
    }

    assert CORE_DEFAULTS["scaffold"] == GENERIC_SCAFFOLD_DEFAULTS
    assert CORE_DEFAULTS["ui_route_visibility"] == {
        "route_globs": [],
        "nav_anchor_paths": [],
    }
    serialized = json.dumps(relocated_defaults, sort_keys=True)
    assert "app/" not in serialized
    assert "3a" not in serialized
    assert "templates/base" not in serialized


def test_core_defaults_patterns_are_generic():
    assert CORE_DEFAULTS["patterns"] == GENERIC_PATTERNS
    serialized = json.dumps(CORE_DEFAULTS["patterns"], sort_keys=True)
    assert "I\\d+" not in serialized
    assert "I(" not in serialized
    assert "phase-" not in serialized


@pytest.mark.parametrize(
    ("snippet", "match"),
    [
        (
            """
patterns:
  task_id: "["
""",
            r"patterns\.task_id has invalid regex",
        ),
        (
            """
patterns:
  task_id: ""
""",
            r"patterns\.task_id must be a non-empty regex string",
        ),
        (
            """
patterns: []
""",
            "section 'patterns' must be a mapping",
        ),
    ],
)
def test_patterns_validation_fails_closed(
    tmp_path: Path,
    snippet: str,
    match: str,
):
    with pytest.raises(ConfigError, match=match):
        load_project_yaml(_write(tmp_path, VALID_PROJECT_YAML + snippet))


def test_tasks_schema_defaults_are_floor_not_empty_pack():
    assert CORE_DEFAULTS["tasks_schema"] == {
        "forbidden_allowed_prefixes": [],
        "planning_allowed_prefixes": [],
        "planning_refusal_prefixes": [],
    }
    assert effective_task_schema_policy(CORE_DEFAULTS["tasks_schema"]) == (
        UNIVERSAL_TASK_SCHEMA_FLOOR
    )
    assert all(UNIVERSAL_TASK_SCHEMA_FLOOR.values())


def test_project_tasks_schema_extends_without_narrowing_floor(tmp_path: Path):
    config = VALID_PROJECT_YAML + """
tasks_schema:
  forbidden_allowed_prefixes: []
  planning_allowed_prefixes:
    - "plans/"
  planning_refusal_prefixes:
    - "src/"
"""

    cfg = load_config(_write(tmp_path, config))
    policy = task_schema_policy(cfg)

    assert policy["forbidden_allowed_prefixes"] == (
        UNIVERSAL_TASK_SCHEMA_FLOOR["forbidden_allowed_prefixes"]
    )
    assert policy["planning_allowed_prefixes"] == [
        *UNIVERSAL_TASK_SCHEMA_FLOOR["planning_allowed_prefixes"],
        "plans/",
    ]
    assert policy["planning_refusal_prefixes"] == [
        *UNIVERSAL_TASK_SCHEMA_FLOOR["planning_refusal_prefixes"],
        "src/",
    ]


@pytest.mark.parametrize(
    ("snippet", "match"),
    [
        (
            """
tasks_schema:
  planning_refusal_prefixes: "app/"
""",
            "planning_refusal_prefixes must be a list",
        ),
        (
            """
tasks_schema:
  planning_refusal_prefixes:
    - "/app/"
""",
            "repo-relative",
        ),
        (
            """
tasks_schema:
  planning_refusal_prefixes:
    - "../app/"
""",
            "repo-relative",
        ),
        (
            """
tasks_schema:
  planning_refusal_prefixes:
    - "app/*"
""",
            "glob chars",
        ),
        (
            """
tasks_schema:
  planning_refusal_prefixes:
    - "app"
""",
            "must end with '/'",
        ),
    ],
)
def test_tasks_schema_policy_validation_fails_closed(
    tmp_path: Path,
    snippet: str,
    match: str,
):
    with pytest.raises(ConfigError, match=match):
        load_project_yaml(_write(tmp_path, VALID_PROJECT_YAML + snippet))


def test_example_project_pack_restores_relocated_policy():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "examples" / "financial-saas" / "project.yaml")

    assert scaffold(cfg) == EXAMPLE_SCAFFOLD
    assert ui_route_visibility(cfg) == EXAMPLE_UI_ROUTE_VISIBILITY
    assert task_schema_policy(cfg) == EXAMPLE_TASK_SCHEMA_POLICY
    assert patterns(cfg) == EXAMPLE_PATTERNS


def test_global_hard_diff_cap_default_is_stable(tmp_path: Path):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    assert CORE_DEFAULTS["limits"]["max_diff_insertions_hard"] == 1500
    assert cfg["limits"]["max_diff_insertions_hard"] == 1500


def test_project_hard_diff_cap_override_does_not_mutate_core(tmp_path: Path):
    override = VALID_PROJECT_YAML + (
        "\nlimits:\n  max_diff_insertions_hard: 1800\n"
    )
    cfg = load_config(_write(tmp_path, override))
    assert cfg["limits"]["max_diff_insertions_hard"] == 1800
    assert CORE_DEFAULTS["limits"]["max_diff_insertions_hard"] == 1500


def test_missing_file(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_project_yaml(tmp_path / "nope.yaml")


def test_top_level_not_mapping(tmp_path: Path):
    p = tmp_path / "project.yaml"
    p.write_text("- just a list\n")
    with pytest.raises(ConfigError, match="top-level must be a mapping"):
        load_project_yaml(p)


@pytest.mark.parametrize(
    "section,stripped_key",
    [
        ("project", "name"),
        ("stack", "test"),
        ("risk", "high_risk_globs"),
    ],
)
def test_missing_required_key(tmp_path, section, stripped_key):
    import yaml as _yaml

    data = _yaml.safe_load(VALID_PROJECT_YAML)
    del data[section][stripped_key]
    p = tmp_path / "project.yaml"
    p.write_text(_yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=f"{section}.{stripped_key}"):
        load_project_yaml(p)


def test_missing_agents_section(tmp_path):
    import yaml as _yaml

    data = _yaml.safe_load(VALID_PROJECT_YAML)
    del data["agents"]
    p = tmp_path / "project.yaml"
    p.write_text(_yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="agents"):
        load_project_yaml(p)


def test_agent_missing_family(tmp_path):
    broken = VALID_PROJECT_YAML.replace("family: anthropic", "")
    with pytest.raises(ConfigError, match="family"):
        load_project_yaml(_write(tmp_path, broken))


def test_agent_family_must_have_matching_cost_mapping(tmp_path: Path):
    broken = VALID_PROJECT_YAML.replace(
        "codex:  {cmd: \"codex\",     family: openai}",
        "codex:  {cmd: \"codex\",     family: opneai}",
    )

    with pytest.raises(ConfigError, match="no matching costs entry"):
        load_project_yaml(_write(tmp_path, broken))


def test_cost_entry_incomplete(tmp_path):
    broken = VALID_PROJECT_YAML.replace(
        "anthropic: {input: 3.0, output: 15.0}",
        "anthropic: {input: 3.0}",
    )
    with pytest.raises(ConfigError, match="input.*output"):
        load_project_yaml(_write(tmp_path, broken))


def test_invalid_yaml(tmp_path: Path):
    p = _write(tmp_path, "project:\n  name: x\n  : bad\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_project_yaml(p)


def test_unknown_top_level_config_section_warns(tmp_path: Path):
    config = VALID_PROJECT_YAML + "\nunknown_section:\n  enabled: true\n"

    with pytest.warns(
        UserWarning,
        match="unknown top-level config section 'unknown_section'",
    ):
        load_project_yaml(_write(tmp_path, config))


def test_unknown_known_section_key_warns(tmp_path: Path):
    config = VALID_PROJECT_YAML + "\nauto_merge:\n  no_cii: true\n"

    with pytest.warns(UserWarning, match=r"auto_merge\.no_cii"):
        load_project_yaml(_write(tmp_path, config))


def test_agents_and_cost_families_still_allow_project_defined_names(
    tmp_path: Path,
    recwarn: pytest.WarningsRecorder,
):
    config = VALID_PROJECT_YAML + """\

agents:
  local_specialist:
    cmd: "local-agent"
    family: local_family
costs:
  local_family:
    input: 1.25
    output: 2.50
"""

    cfg = load_config(_write(tmp_path, config))

    assert agents(cfg)["local_specialist"]["cmd"] == "local-agent"
    assert costs(cfg)["local_family"]["output"] == 2.50
    assert list(recwarn) == []


def test_dead_tasks_md_project_keys_removed_or_rejected(
    recwarn: pytest.WarningsRecorder,
):
    repo_root = Path(__file__).resolve().parents[1]
    project_path = repo_root / "examples" / "financial-saas" / "project.yaml"

    project = load_project_yaml(project_path)
    cfg = load_config(project_path)

    assert project.get("tasks_md", {}) == {}
    assert tasks_md(cfg)["status_values"] == CORE_DEFAULTS["tasks_md"][
        "status_values"
    ]
    assert auto_merge(cfg)["no_ci"] is False
    assert list(recwarn) == []


def test_deep_merge_preserves_unrelated_keys():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    overlay = {"a": {"y": 99, "z": 3}}
    merged = _deep_merge(base, overlay)
    assert merged == {"a": {"x": 1, "y": 99, "z": 3}, "b": 1}


def test_core_defaults_independence_level():
    assert CORE_DEFAULTS["independence"]["level"] == "model_family"


@pytest.mark.parametrize("section,accessor", ACCESSOR_CASES)
def test_config_accessors_round_trip_sections(tmp_path: Path, section, accessor):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    assert accessor(cfg) is cfg.data[section]


def test_config_accessors_do_not_mutate_loaded_config(tmp_path: Path):
    cfg = load_config(_write(tmp_path, VALID_PROJECT_YAML))
    original = copy.deepcopy(cfg.data)
    original_path = cfg.path

    for _, accessor in ACCESSOR_CASES:
        accessor(cfg)

    assert cfg.data == original
    assert cfg.path == original_path


def test_project_override_values_are_visible_through_accessors(tmp_path: Path):
    override = VALID_PROJECT_YAML + """\

stack:
  test: "pytest tools/tests/orch -q"
  lint: "ruff check tools tests"
risk:
  high_risk_globs: ["**/schema.sql", "**/migrations/*.sql"]
  sensitive_files: [".env", "secrets.toml"]
  forbidden_patterns: ["except: pass", "eval("]
agents:
  claude: {cmd: "claude --model sonnet", family: anthropic}
  codex:  {cmd: "codex --oss", family: openai}
costs:
  anthropic: {input: 4.0, output: 20.0}
  openai:    {input: 3.0, output: 12.0}
limits:
  impl_attempts: 5
auto_merge:
  max_diff_insertions: 750
timeouts:
  review: 123
independence:
  level: model
review:
  verdict_regex: "^custom$"
tasks_md:
  status_values: ["WAITING", "DONE", "CUSTOM"]
parallel:
  max_concurrency: 4
hooks:
  handlers:
    - name: prompt-policy
      events: ["task.before_implement"]
      cmd: "python hook.py"
      required: true
      timeout: 5
"""
    cfg = load_config(_write(tmp_path, override))

    assert stack(cfg)["lint"] == "ruff check tools tests"
    assert risk(cfg)["sensitive_files"] == [".env", "secrets.toml"]
    assert agents(cfg)["claude"]["cmd"] == "claude --model sonnet"
    assert costs(cfg)["openai"]["output"] == 12.0
    assert limits(cfg)["impl_attempts"] == 5
    assert auto_merge(cfg)["max_diff_insertions"] == 750
    assert timeouts(cfg)["review"] == 123
    assert independence(cfg)["level"] == "model"
    assert review(cfg)["verdict_regex"] == "^custom$"
    assert tasks_md(cfg)["status_values"] == ["WAITING", "DONE", "CUSTOM"]
    assert parallel(cfg)["max_concurrency"] == 4
    assert hooks(cfg)["handlers"][0]["name"] == "prompt-policy"


def test_required_hook_must_subscribe_to_blocking_event(tmp_path: Path):
    bad = VALID_PROJECT_YAML + """\

hooks:
  handlers:
    - name: bad
      events: ["task_transition"]
      cmd: "python hook.py"
      required: true
"""

    with pytest.raises(ConfigError, match="required=true.*non-blocking"):
        load_project_yaml(_write(tmp_path, bad))


@pytest.mark.parametrize("section,accessor", ACCESSOR_CASES)
def test_config_accessors_raise_raw_key_error_for_missing_section(
    section,
    accessor,
):
    cfg = LoadedConfig(path=Path("project.yaml"), data={})

    with pytest.raises(KeyError) as raw:
        cfg.data[section]
    with pytest.raises(KeyError) as via_accessor:
        accessor(cfg)

    assert via_accessor.value.args == raw.value.args


def test_model_routing_agent_overrides_accept_config_sugar(tmp_path: Path):
    config = VALID_PROJECT_YAML + """
model_routing:
  agent_overrides:
    codex:
      model_flag: "--model"
      tier_models:
        standard: "codex-standard"
        strong: "codex-strong"
        max: "codex-max"
      effort_flags:
        low:
          args: ["--reasoning-effort", "low"]
        medium:
          args: ["--reasoning-effort", "medium"]
        high:
          args: ["--reasoning-effort", "high"]
        max:
          args: ["--reasoning-effort", "max"]
          env:
            ORCH_REASONING_EFFORT: "max"
      max:
        max:
          args: ["--model", "one-off-model"]
          env:
            RAW_OVERRIDE: "1"
"""

    cfg = load_config(_write(tmp_path, config))
    codex = model_routing(cfg)["agent_overrides"]["codex"]

    assert codex["model_flag"] == "--model"
    assert codex["tier_models"]["max"] == "codex-max"
    assert codex["effort_flags"]["max"] == {
        "args": ["--reasoning-effort", "max"],
        "env": {"ORCH_REASONING_EFFORT": "max"},
    }
    assert codex["max"]["max"] == {
        "args": ["--model", "one-off-model"],
        "env": {"RAW_OVERRIDE": "1"},
    }


@pytest.mark.parametrize(
    ("snippet", "match"),
    [
        (
            """
model_routing:
  agent_overrides:
    codex:
      model_flag: ""
      tier_models:
        standard: "codex-standard"
        strong: "codex-strong"
        max: "codex-max"
""",
            "model_flag must be a non-empty string",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      model_flag: "--model"
""",
            "model_flag requires tier_models",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      tier_models:
        standard: "codex-standard"
        strong: "codex-strong"
        max: "codex-max"
""",
            "tier_models requires model_flag",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      model_flag: "--model"
      tier_models:
        standard: "codex-standard"
        max: "codex-max"
""",
            "tier_models must define every tier",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      model_flag: "--model"
      tier_models:
        standard: "codex-standard"
        strong: "codex-strong"
        max: "codex-max"
        tiny: "codex-tiny"
""",
            "invalid tier",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      effort_flags: ["--reasoning-effort", "max"]
""",
            "effort_flags must be a mapping",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      effort_flags:
        low: {}
        medium: {}
        high: {}
""",
            "effort_flags must define every effort",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      effort_flags:
        low: {}
        medium: {}
        high: {}
        max: {}
        extreme: {}
""",
            "invalid effort",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      effort_flags:
        low: {}
        medium: {}
        high: {}
        max: ["--reasoning-effort", "max"]
""",
            "effort_flags.max must be a mapping",
        ),
        (
            """
model_routing:
  agent_overrides:
    codex:
      effort_flags:
        low: {}
        medium: {}
        high: {}
        max:
          args: "--reasoning-effort max"
""",
            "args must be a list",
        ),
    ],
)
def test_invalid_model_routing_sugar_config_fails_closed(
    tmp_path: Path, snippet: str, match: str
):
    with pytest.raises(ConfigError, match=match):
        load_project_yaml(_write(tmp_path, VALID_PROJECT_YAML + snippet))


def test_project_yaml_model_routing_populates_provider_sugar():
    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(repo_root / "examples" / "financial-saas" / "project.yaml")

    configured_agents = agents(cfg)
    overrides = model_routing(cfg)["agent_overrides"]

    assert configured_agents["claude"]["family"] == "anthropic"
    assert configured_agents["codex"]["family"] == "openai"
    assert overrides["claude"] == {
        "model_flag": "--model",
        "tier_models": {
            "standard": "claude-haiku-example",
            "strong": "claude-sonnet-example",
            "max": "claude-opus-example",
        },
    }
    assert overrides["codex"] == {
        "model_flag": "-m",
        "tier_models": {
            "standard": "gpt-example",
            "strong": "gpt-example",
            "max": "gpt-example",
        },
        "effort_flags": {
            "low": {"args": ["-c", "model_reasoning_effort=low"]},
            "medium": {"args": ["-c", "model_reasoning_effort=medium"]},
            "high": {"args": ["-c", "model_reasoning_effort=high"]},
            "max": {"args": ["-c", "model_reasoning_effort=xhigh"]},
        },
    }
