# <TASK_ID> - <Docs-Only Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: `<orchestrator | manual>`
- Risk category: `unknown`
- Task type: docs-only

## Required Read Order

1. `CLAUDE.md`
2. `<source docs that define current truth>`
3. `<target docs to edit>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Goal

After this task, `<doc or operator-facing artifact>` accurately reflects
`<current source of truth>`.

## Source of Truth

- Canonical source:
- Secondary evidence:
- Stale/incorrect claim to remove:

## Non-Goals

- Do not edit runtime code.
- Do not edit tests.
- Do not edit orchestrator code.
- Do not invent roadmap, status, shipped claims, dates, PRs, or decisions.
- Do not create branches, PRs, or run deployment commands unless explicitly
  asked.

## Allowed Files

```text
<docs path>
<docs path>
```

## Forbidden Files

```text
app/
db/
tests/
orch/
static/
```

## Edit Contract

- Update:
- Remove:
- Preserve:

## Verification

| Requirement | Evidence | Command/manual check | Blocking? |
|---|---|---|---|
| Doc matches source | source comparison | `<manual/file refs>` | yes |
| Markdown clean | command | `git diff --check -- <files>` | yes |
| Links reasonable | manual or command | `<check>` | no/yes |

## Required Commands

```bash
git diff --check -- <allowed docs files>
```

If no runnable test applies, write a short verification note instead of
inventing a fake test.

## Stop Conditions

Stop when:

- source docs contradict each other,
- the change would require touching forbidden files,
- or a claim cannot be verified from the named source of truth.

## Final Action

For orchestrator mode:

- Leave a clean diff limited to the allowed docs files.
- Do not edit `tasks.md`.
- The orchestrator owns commits and task state.

For manual mode:

```bash
git add <docs files>
git status --short
git commit -m "<TASK_ID>: <docs summary>" -m "<source of truth and checks>"
git push -u origin <branch>
```
