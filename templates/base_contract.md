# Base Prompt Contract

Use this block as the shared contract for task, review, QA, and retro prompts.
Specialized task templates must embed these sections inline, not merely
reference this file. Specialized templates may add stricter rules, but should
not remove these protections without an explicit operator decision.

## Required Read Order

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `iterations/_review_template.md`
4. `<iteration>/prompt.md`
5. `<iteration>/tasks.md`
6. `<task-specific source files or docs>`

If required sources conflict, stop and report:

- conflicting files,
- exact claims,
- the safer interpretation,
- and the operator decision needed.

## Scope Contract

Every prompt must state:

- exact allowed files,
- forbidden files,
- allowed symbols/functions/routes/templates,
- frozen symbols/functions/routes/templates,
- whether `tasks.md` may be touched,
- whether `orch/` may be touched,
- whether the task is orchestrator-run or manual.

No prompt may hide uncertainty by allowing broad directories unless the work
genuinely requires every file under that directory.

## Applicable Project Invariants

Classify every invariant as `applies`, `preserve only`, or `not applicable`.
Do not leave a row blank.

| Invariant | Classification | Required evidence or N/A reason |
|---|---|---|
| <Invariant 1 - e.g. access boundary> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 2 - e.g. data retention rule> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 3 - e.g. deterministic logic boundary> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |

## Requirement-to-Evidence Matrix

Every blocking requirement must have evidence.

| Requirement | Evidence type | Command/file/manual check | Blocking? |
|---|---|---|---|
| `<requirement>` | `<test/grep/manual/file>` | `<evidence>` | `<yes/no>` |

## Preserved Behavior or N/A

Every task prompt must state one of:

- Preserved-behavior fixture: `<path and command>`
- Preserved-behavior checks: `<existing tests/fixtures>`
- N/A because: `<why this task cannot affect existing behavior>`

`N/A` is not acceptable for tasks that can affect declared project
invariants, data integrity, security, deployment, or user-facing workflows.

## Stop Conditions

Stop instead of guessing when:

- allowed files are insufficient,
- behavior depends on an operator-owned product decision,
- a migration/deployment/secrets decision is missing,
- a required invariant conflicts with the requested change,
- the implementation would need broad mechanical sweep work,
- tests reveal an unrelated existing defect,
- or the task requires touching forbidden files.

## Deviation Rule

If the final work intentionally differs from the prompt, including a strictly
safer improvement, the commit body or report must include:

```text
Deviation:
- Prompt said:
- Implemented instead:
- Why this is safer and still in scope:
- Evidence:
```

Silent "good drift" is still drift.

## Final Action Contract

Every task prompt must state one of these execution modes.

### Orchestrator final action

- Leave a clean, test-passing diff.
- Do not edit `tasks.md`.
- Do not bypass orchestrator-owned state transitions.

### Manual final action

```bash
git add <explicit files>
git status --short
git commit -m "<TASK_ID>: <summary>" -m "<why, tests, deviations>"
git push -u origin <branch>
```
