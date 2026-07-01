# <TASK_ID> - <Audit/Planning Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: `<manual | orchestrator>`
- Risk category: `unknown`
- Task type: read-only audit or planning-only

## Required Read Order

1. `CLAUDE.md`
2. `<canonical source docs>`
3. `<target area files>`
4. `<prior reports or retros>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Audit Goal

Produce `<report/planning artifact>` that answers `<specific question>`.

## Scope

In scope:

- `<area>`

Out of scope:

- No runtime edits.
- No tests edits.
- No migrations.
- No deployment changes.
- No branch/PR/merge actions unless explicitly asked.

## Allowed Files

If report-only:

```text
<new report path>
```

If strictly read-only, write:

```text
No file edits allowed.
```

## Evidence Requirements

| Question | Evidence source | Required output |
|---|---|---|
| `<question>` | `<file/command/report>` | `<finding or table>` |

## Report Format

```markdown
# <Report Title>

## Executive Summary

## Findings

- [CRITICAL | SHOULD_FIX | FUTURE] <finding>
  - Evidence:
  - Impact:
  - Recommendation:

## Decisions Needed

## Proposed Next Prompts

## Open Questions
```

## Allowed Commands

Read-only commands only:

```bash
git status --short
git diff --stat
rg "<pattern>"
sed -n '<range>p' <file>
pytest --collect-only <optional if needed>
```

Do not run destructive cleanup, migrations, deployment, branch deletion, or
formatters.

## Verification

If a report file is written:

```bash
git diff --check -- <report file>
```

## Stop Conditions

Stop when:

- source docs contradict each other,
- the change would require touching forbidden files,
- or a claim cannot be verified from the named source of truth.

## Final Action

For report-only manual mode, commit only the report if the operator asked for
a committed artifact. Otherwise leave it as an uncommitted draft.
