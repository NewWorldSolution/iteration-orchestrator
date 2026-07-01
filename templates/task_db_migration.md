# <TASK_ID> - <Database Migration Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: `<orchestrator | manual>`
- Risk category: `schema_data_structure`
- Reviewer expectation: cross-family review required

## Required Read Order

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `iterations/_review_template.md`
4. `<iteration>/prompt.md`
5. `db/schema.sql`
6. `db/init_db.py`
7. `<existing migration reference>`
8. `<tests touching the affected schema>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Goal

After this task, fresh installs and upgraded existing databases both satisfy
`<schema/data invariant>`.

## Why This Exists

`<Data integrity, access scope isolation, product, or deployment reason.>`

## Current Schema and Data State

- Current table/column/index/constraint:
- Existing data shape:
- Existing migration behavior:
- Known production/sandbox/test engines:

## Target Schema and Data State

- Fresh install:
- SQLite test upgrade:
- PostgreSQL upgrade:
- Idempotent rerun:
- Rollback/backup expectation:

## Non-Goals

- Do not change unrelated tables.
- Do not add DB constraints that project rules forbid.
- Do not store resolved values that can be recalculated deterministically.
- Do not change application behavior outside the migration contract unless
  explicitly authorized.

## Allowed Files

```text
db/schema.sql
db/init_db.py
<migration tests>
<VERIFY_AT_GOLIVE or docs file if required>
```

## Forbidden Files and Symbols

- No route/template/frontend changes unless explicitly listed.
- No unrelated service refactors.
- No `tasks.md` edits.
- No `orch/` edits.

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

## Applicable Project Invariants

| Invariant | Classification | Required evidence or N/A reason |
|---|---|---|
| <Invariant 1 - e.g. schema compatibility> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 2 - e.g. data retention rule> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 3 - e.g. environment parity> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |

## Migration Contract

### Fresh install path

- `db/schema.sql` must define:
  - `<table/column/index/constraint>`

### Upgrade path

- Migration function:
  - name:
  - call location:
  - ordering invariant:

### Idempotency

Running initialization twice must:

- not duplicate rows,
- not fail on existing columns/indexes,
- not loosen constraints,
- not lose data.

### SQLite branch

- Expected SQL/rebuild behavior:
- Foreign-key handling:
- Index recreation:

### PostgreSQL branch

- Expected SQL:
- Online/offline risk:
- Go-live manual verification if needed:

## Data Safety

- Backup required? `<yes/no>`
- `pg_dump` command if yes:
- Orphan-row behavior:
- Backfill source of truth:
- Rows that cannot be mapped:

## Acceptance Matrix

| Requirement | Evidence | Command/test/manual check | Blocking? |
|---|---|---|---|
| Fresh install has target schema | test | `<command>` | yes |
| Existing DB migrates safely | test | `<command>` | yes |
| Migration idempotent | test | `<command>` | yes |
| PostgreSQL SQL path covered | mock/live/manual | `<command or VERIFY_AT_GOLIVE>` | yes |

## Required Tests

- Fresh install test.
- Pre-migration fixture upgrade test.
- Idempotency test.
- Constraint/index verification.
- Data backfill verification.
- Access-scope verification if a scoped table is touched.

## Required Commands

```bash
pytest <migration tests> -v
pytest tests/ -x
ruff check db tests
```

## Preserved Behavior or N/A

State one of:

- Preserved-behavior fixture: `<path and command>`
- Preserved-behavior checks: `<existing migration/schema tests>`
- N/A because: `<why this migration cannot affect existing behavior>`

For schema/data migrations, `N/A` is rarely acceptable. Preserve existing data,
declared invariants, referential integrity, and fresh-install behavior unless the task
explicitly changes them.

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
- Include fresh-install, upgrade, idempotency, and PostgreSQL evidence in the
  task report if the adapter/report supports it.

For manual mode:

```bash
git add <explicit files>
git status --short
git commit -m "<TASK_ID>: <migration summary>" -m "<fresh-install, upgrade, idempotency, PostgreSQL evidence, deviations>"
git push -u origin <branch>
```

If implementation intentionally improves on or deviates from this prompt,
include a `Deviation:` block in the commit body/report with prompt text,
implemented behavior, why it is safer, and evidence.
