# Architecture

`orch` is a **deterministic state machine** that orchestrates LLM coding agents.
The guiding rule: **no LLM in the control path.** Every decision that must be
reproducible — gates, scope checks, merges, state transitions, cost accounting —
is plain Python. LLMs are confined to swappable worker adapters at the edges.

## The control loop

```
validate ──▶ run ────────────────────────────────────────────────▶ qa ──▶ retro
             │                                                       (5      (3
             ▼   for each ready task in the DAG:                     reviewers) perspectives)
      ┌──────────────────────────────────────────────────────┐
      │ implement (agent A)                                   │
      │   └─▶ deterministic checks (scope / diff-size /       │
      │        forbidden patterns / sensitive files)          │
      │   └─▶ acceptance command (project-defined)            │
      │   └─▶ review (agent B, different model family)        │
      │        └─▶ triage: PROCEED / FIX / DEFER / STOP       │
      │        └─▶ dual review (agent C) for high-risk tasks  │
      │   └─▶ final gates (scope leak / nav / branch fresh)   │
      │   └─▶ guarded merge (CI + policy) or STOP:<reason>    │
      └──────────────────────────────────────────────────────┘
```

Any step can end the task with a **typed stop reason** (`SCOPE`, `STRUCTURAL`,
`CHECKS`, `REVIEW_FAIL`, `DUAL_REVIEW_FAIL`, `BRANCH_FRESHNESS`, `PREFLIGHT_SIZE`,
…), each with an operator recovery note and a resume command. The system never
guesses past ambiguity.

## State model

State is an **append-only event log** (`state.py`); the run-state snapshot is a
*derived* view, never the source of truth. This gives clean crash recovery:
`resume` reconciles a half-finished task (e.g. merged-but-not-recorded) to a
consistent state; `retry` resets one task + its downstream to `WAITING`;
`revert` undoes a merged task. Persistence is atomic (`mkstemp` + `os.replace`).

## Deterministic gates (`checks.py`, `final_gates.py`, `preflight.py`)

Pure functions over (changed files, diff text, config) — explicitly **no model
calls**:

- **Scope**: changed files must match the task's allowed-files; out-of-scope edits
  fail closed (with a bounded auto-revert helper).
- **Diff size**: a preflight tier estimate + a hard cap (`PREFLIGHT_SIZE`).
- **Forbidden patterns**: conflict markers, debug leftovers, hardcoded secrets.
- **Final scope / nav / branch-freshness**: post-merge integrity, computed against
  a resolved base ref; MISSING_BASE / FRESH / BEHIND all fail closed.

## Review and cross-model independence (`review.py`, `model_routing.py`)

- The verdict parser is strict: the final non-empty line must be a valid verdict,
  else the review is malformed and the task stops.
- **Independence is enforced before invocation**: implementer and reviewer families
  must differ; risk categories `architecture_core_logic` / `merge_critical_gate`
  additionally require a third-family secondary reviewer that must agree.
- **Triage** (`triage.py`) maps `(verdict × severity × defer-budget ×
  confidence-history)` to `PROCEED / FIX_NOW / DEFER_TO_QA / STOP_HUMAN`, with
  repeat-failure and confidence-drop detection firing before budget logic.

## Model routing (`model_routing.py`)

Each task declares a `model_tier`, `reasoning_effort`, and `risk_category`. Risk
**floors** are hard minimums — sensitive categories can never route below their
floor regardless of savings. Resolved routing maps to concrete provider CLI args.

## Project pack (config as data)

Everything project-specific lives in a `project.yaml` pack, not in the engine:
branches, allowed-file patterns, risk globs, forbidden patterns, agents, costs,
timeouts, model routing, and declared invariants. The engine ships with generic,
inert defaults; a project supplies its own conventions. See
[`examples/minimal/project.yaml`](examples/minimal/project.yaml) for a zero-domain
starting point and [`examples/financial-saas/`](examples/financial-saas/) for a
full worked example.

## Agent adapters (`agents/`)

A small `AgentAdapter` protocol wraps each worker CLI (`claude`, `codex`, or a
generic `shell` adapter) as a subprocess with timeout handling
(SIGTERM → grace → SIGKILL) and usage capture. Adapters are injected, which is why
the whole loop is unit-testable without real model calls.

## Module map (condensed)

| Group | Modules |
|---|---|
| Coordinator | `runner.py`, `cli.py`, `run_loop.py`, `task_flow.py`, `lifecycle.py` |
| State / recovery | `state.py`, `locks.py`, `recovery.py`, `finalization.py` |
| Gates / checks | `checks.py`, `final_gates.py`, `preflight.py`, `review.py`, `triage.py` |
| Merge / git | `merge.py`, `git_ops.py` |
| Quality passes | `qa.py`, `retro.py`, `report.py` |
| Routing / cost | `model_routing.py`, `cost.py` |
| Workers | `agents/` (`base`, `claude`, `codex`, `shell`), `providers.py` |
| Config / schema | `config.py`, `tasks_schema.py`, `paths.py` |
| Experimental | `prompt_factory.py`, `parallel_runner.py`, `parallel.py`, `teams.py`, `team_mode.py`, `planning_team.py` |

## Extending it

- **New worker**: implement the `AgentAdapter` protocol and register it in
  `agents/`; select it via `--implementer` / `--reviewer` or the project pack.
- **New gate**: add a pure function in `checks.py` / `final_gates.py` and wire it
  into the runner's check phase — keep it deterministic and fail-closed.
- **New project**: write a `project.yaml` pack; do not edit the engine.
