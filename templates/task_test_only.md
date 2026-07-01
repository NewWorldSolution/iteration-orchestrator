# <TASK_ID> - <Test-Only Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: `<orchestrator | manual>`
- Risk category: `<risk_category>`
- Task type: test-only

## Required Read Order

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `<source files whose current behavior is being pinned>`
4. `<existing tests/fixtures to reuse>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Goal

After this task, tests pin `<current behavior or regression class>` without
changing runtime code.

## Current Behavior to Pin

- Behavior:
- Source file/line:
- Current expected status/output/state:
- Whether this is desired behavior or characterization only:

## Non-Goals

- Do not change `app/`, `db/`, `static/`, templates, or runtime config.
- Do not fix defects found while writing tests unless explicitly authorized.
- Do not weaken or delete existing tests.
- Do not use broad fixtures when existing ones are available.

## Allowed Files

```text
tests/<new_or_existing_test_file>.py
```

## Forbidden Files

```text
app/
db/
static/
templates/
orch/
```

## Test Contract

Each test must:

- have a name matching the behavior,
- include a one-line docstring or comment with source `file:line`,
- assert current behavior accurately,
- use existing fixtures where possible,
- include `# OBSERVED:` if current behavior differs from prompt prediction.

## Acceptance Matrix

| Behavior pinned | Source evidence | Test name | Blocking? |
|---|---|---|---|
| `<behavior>` | `<file:line>` | `<test_name>` | yes |

## Required Cases

- Happy path:
- Negative path:
- Access-boundary path if relevant:
- Role/access path if relevant:
- Archive path if relevant:
- Redirect `Location` if redirect status is asserted:

## Required Commands

```bash
pytest <new test file> -q
pytest tests/ -x
ruff check <new test file>
```

## Preserved Behavior or N/A

State one of:

- Preserved-behavior fixture: `<path and command>`
- Preserved-behavior checks: `<existing tests that still pass>`
- N/A because: `<why this characterization cannot affect existing behavior>`

For test-only tasks, preserving behavior means runtime files are unchanged and
the new tests pin current behavior without weakening existing tests.

## Stop Conditions

Stop instead of guessing when:

- allowed files are insufficient,
- behavior depends on an operator-owned product decision,
- a migration/deployment/secrets decision is missing,
- a required invariant conflicts with the requested change,
- the implementation would need broad mechanical sweep work,
- tests reveal an unrelated existing defect,
- or the task requires touching forbidden files.

## Final Action

For orchestrator mode:

- Leave a clean, test-passing diff.
- Do not edit `tasks.md`.
- Do not modify runtime files.

For manual mode:

```bash
git add <explicit test files>
git status --short
git commit -m "<TASK_ID>: <test summary>" -m "<behavior pinned, tests run, observed deviations>"
git push -u origin <branch>
```

Include any `# OBSERVED:` deviations in the commit body/report.
