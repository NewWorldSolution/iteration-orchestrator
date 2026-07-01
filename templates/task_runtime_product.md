# <TASK_ID> - <Runtime Product Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: `<orchestrator | manual>`
- Risk category: `<risk_category>`
- Model routing: `<model_tier/reasoning_effort or default>`
- Implementer/reviewer expectation: `<agent family or pairing>`

## Required Read Order

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `iterations/_review_template.md`
4. `<iteration>/prompt.md`
5. `<iteration>/tasks.md`
6. `<feature docs or code references>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Goal

After this task, `<observable product/runtime behavior>` works for
`<user/role/workflow>`.

## Why This Exists

`<Business, product, security, or operational reason.>`

## Current State

- Existing behavior:
- Existing tests or docs that define baseline:
- Known gap:

## Target State

- New behavior:
- Preserved behavior:
- User/operator-visible result:

## Non-Goals

- Do not `<adjacent feature>`.
- Do not `<refactor not required>`.
- Do not `<future phase work>`.

## Allowed Files

```text
<exact file>
<exact file>
```

## Forbidden Files and Symbols

- Forbidden files:
  - `<path>`
- Frozen symbols:
  - `<file>::<function/class/template block>`
- Forbidden behavior:
  - No unrelated route/template/service refactors.
  - No `tasks.md` edits.
  - No `orch/` edits.

## Applicable Project Invariants

| Invariant | Classification | Required evidence or N/A reason |
|---|---|---|
| <Invariant 1 - e.g. access boundary> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 2 - e.g. data retention rule> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 3 - e.g. deterministic logic boundary> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |

## Edit Contract

### Authorized edits

- `<file>::<symbol>` - `<allowed change>`

### Frozen behavior

- `<file>::<symbol>` - `<must remain behavior-identical>`

## Sweep-Scope Clause

> **Sweep-scope clause.** This task authorizes editing only the
> enumerated symbols: `<explicit list of functions / queries /
> templates>`. Other members of any set that could be grepped ("every
> SELECT on <table>", "every TemplateResponse for create.html",
> "every handler that calls X") are OUT of scope regardless of whether
> they appear in the same file. If you notice a related symbol that
> seems to want the same edit, STOP and flag it — do not sweep.
> Gratuitous subquery wraps (`SELECT COUNT(*) FROM (SELECT ...)`),
> wholesale function rewrites, and "while I'm here" refactors are
> automatic REVIEW_FAIL.

## Access-Scope Filter Audit

> **Access-scope filter audit.** Every SELECT/UPDATE/DELETE in this task's
> diff — including resolution helpers, audit scripts, and migration
> queries — MUST include `WHERE <scope_column> = ?`. A query without a
> access scope filter is a CRITICAL security finding at QA regardless of
> whether it is read-only or admin-only. Grep signatures 7 and 8 above
> are mandatory pre-QA.

Do not introduce access-scope filter exceptions in this template. A rare token or
globally unique lookup exception requires an explicit operator-approved
exception in the task prompt, with rationale, non-enumeration evidence, and
access-boundary tests. Without that explicit exception, missing access scope filters are
blocking defects.

## Implementation Notes

- `<specific rule>`
- `<specific rule>`
- `<copy-paste snippets only when safe and executable>`

## Acceptance Matrix

| Requirement | Evidence | Command/test/manual check | Blocking? |
|---|---|---|---|
| `<requirement>` | `<test/manual/file:line>` | `<command>` | yes |

## Functional Test Requirements

- Positive path:
- Negative path:
- Access-boundary path:
- Archived row path:
- Boolean absent/unchecked/checked matrix if applicable:
- Redirect `Location` assertion if redirects are touched:

## Manual Smoke

If UI changed:

- URL:
- Role/account:
- Actions:
- Expected state:
- Localized characters:

If no UI changed: `N/A - no UI touched`.

## Required Commands

```bash
<focused test command>
<broader regression command if shared behavior changed>
ruff check <changed python files or .>
```

## Preserved Behavior or N/A

State one of:

- Preserved-behavior fixture: `<path and command>`
- Preserved-behavior checks: `<existing tests/fixtures>`
- N/A because: `<why this task cannot affect existing behavior>`

If existing validation, calculation, reporting, security, data integrity, or
workflow behavior can change, `N/A` is not acceptable.

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
- Do not create unrelated commits.

For manual mode:

```bash
git add <explicit files>
git status --short
git commit -m "<TASK_ID>: <summary>" -m "<why, tests, deviations>"
git push -u origin <branch>
```

If implementation intentionally improves on or deviates from this prompt,
include a `Deviation:` block in the commit body/report with prompt text,
implemented behavior, why it is safer, and evidence.
