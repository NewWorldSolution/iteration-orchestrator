# <TASK_ID> - <Tooling/Orchestrator Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: manual tooling sprint unless explicitly orchestrator-safe
- Risk category: `coordinator_state_hook_runtime`
- Reviewer expectation: cross-family review recommended

## Required Read Order

1. `CLAUDE.md`
2. `ARCHITECTURE.md`
3. `iterations/_prompt_rules.md`
4. `iterations/_review_template.md`
5. `<relevant orch modules>`
6. `<focused tests tests>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Tooling-Sprint Warning

If this task allows `orch/` edits, do not run normal
`python -m orch validate <iter>` unless the iteration is explicitly
designed for orchestrator validation. The normal tasks schema forbids
`orch/` allowed files for product iterations.

## Goal

After this task, the orchestrator/tooling does `<observable operator or
state-machine behavior>`.

## Current Failure or Gap

- Current behavior:
- Stop reason or user-visible problem:
- Existing tests:
- Design source:

## Target Behavior

- New deterministic behavior:
- State/log/report artifact changes:
- Backward compatibility:
- Failure mode:

## Non-Goals

- Do not change product runtime code.
- Do not alter project business invariants.
- Do not add autonomous merge/main behavior.
- Do not bypass operator approval gates.
- Do not weaken fail-closed behavior.

## Allowed Files

```text
orch/<module>.py
tests/<test_module>.py
<docs or iteration files if applicable>
```

## Forbidden Files

```text
app/
db/
static/
seed/
production deployment config unless explicitly listed
```

## Compatibility Contract

- Existing `run_state.json` files still load.
- Existing CLI command behavior remains unless explicitly changed.
- Existing review verdict parsing remains backward compatible.
- Errors fail closed with actionable recovery notes.

## Orch-Domain Design Checklist (scaffold author MUST answer each; added 2026-06-11)

Every orch-tooling scaffold so far missed exactly one item from this gap
class (B-2: unpushed-branch PR lifecycle; sprint2: salvage-before-clean).
Answer each question IN the prompt — "N/A because <reason>" counts; silence
does not:

- [ ] **Work retention:** does any step delete, reset, or force-remove a
      branch/worktree/file that could hold uncommitted or unreviewed work?
      If yes: salvage first (reuse `git_ops.salvage_worktree` →
      `salvage/<iter>/<context>` + note event), never destroy silently.
- [ ] **Branch/push position:** for every branch the change creates, merges,
      or advances — does it get pushed, when, and what does GitHub show in
      the meantime (open PRs, stale diffs)? State the position explicitly.
- [ ] **Lock lifecycle:** does the change run while holding the iteration
      lock, after removing one, or without one? Name the window where a
      concurrent `run`/`resume` could race it, and close or accept it
      explicitly.
- [ ] **Stop-reason truth:** any new failure path must surface as a typed
      `STOPPED:<reason>` (or clean CLI exit 1 + stderr), never a raw
      traceback or a silently mislabeled reason.
- [ ] **Salvage/inspection refs are inert:** nothing auto-feeds a salvage or
      recovery ref into review, acceptance, resume, retry, or merge.

## Acceptance Matrix

| Requirement | Evidence | Command/test/manual check | Blocking? |
|---|---|---|---|
| `<tooling behavior>` | `<test>` | `<command>` | yes |

## Required Tests

- Unit test for new helper/state transition.
- CLI test if command behavior changes.
- Backward compatibility test if state/config shape changes.
- Failure-path test.
- Report/log artifact test if output changes.

## Required Commands

```bash
ruff check orch tests
pytest tests -q
```

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

Manual mode is expected unless otherwise stated:

```bash
git add <explicit files>
git status --short
git commit -m "<TASK_ID>: <tooling summary>" -m "<tests and compatibility notes>"
git push -u origin <branch>
```
