# Review - <TASK_ID> - <Title>

## Review Metadata

- Task branch: `<branch>`
- Diff base: `<base sha/ref>`
- Review mode: `<primary | secondary | manual>`
- Risk category: `<risk_category>`
- Task prompt: `<path>`
- Review prompt: `<this path>`

## Required Read Order

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `iterations/_review_template.md`
4. `<iteration>/prompt.md`
5. `<iteration>/tasks.md`
6. `<task prompt>`
7. `<this review prompt>`

## Verdict Output Contract

The response must end with exactly one of these trailing blocks:

```text
Verdict: PASS
```

```text
Verdict: CHANGES REQUIRED
Severity: should-fix
```

```text
Verdict: CHANGES REQUIRED
Severity: block
```

```text
Verdict: BLOCKED
```

No non-empty line may follow the verdict block.

## Setup

```bash
git status --short
git diff <diff-base>..HEAD --name-only
git diff <diff-base>..HEAD --stat
```

Diff-base resolution:

- Orchestrator-run task: diff base is injected in Runtime Review Metadata; use it as-is.
- Task PR off an iteration branch: `git diff origin/<iteration-branch>..HEAD`.
- Manual/tooling task with a recorded pre-task SHA: `git diff $TASK_BASE..HEAD`.
- Never compare a task branch against the phase branch - inherited earlier-task
  work would appear as scope violations.

## Gate 1 - Scope and Structure

- [ ] Changed files exactly match the task Allowed Files or documented
      generated artifacts.
- [ ] No `tasks.md` edits unless this is an approved recovery exception.
- [ ] No `orch/` edits unless this is a declared tooling sprint.
- [ ] No sensitive files or forbidden patterns.
- [ ] Lint command relevant to changed files is clean.

## Gate 2 - Requirement Traceability

Fill before assigning a verdict.

| Requirement from task prompt | Evidence checked | Status |
|---|---|---|
| `<requirement>` | `<file:line/test/command>` | `<PASS/FAIL>` |

Any blocking requirement with `FAIL` requires `CHANGES REQUIRED` or `BLOCKED`.

## Gate 3 - Project Invariant Closure

| Invariant | Applies? | Evidence | Status |
|---|---|---|---|
| <Invariant 1 - e.g. access boundary> | `<yes/no>` | `<evidence>` | `<PASS/FAIL/N/A>` |
| <Invariant 2 - e.g. data retention rule> | `<yes/no>` | `<evidence>` | `<PASS/FAIL/N/A>` |
| <Invariant 3 - e.g. deterministic logic boundary> | `<yes/no>` | `<evidence>` | `<PASS/FAIL/N/A>` |

## Gate 4 - Mechanical Diff Checks

Run every signature below unless the task prompt explicitly marks it not
applicable. For each skipped signature, write a one-line `N/A` rationale in
the review evidence. Add any task-specific signatures below these common
checks.

```bash
DIFF="git diff <diff-base>..HEAD"

# 1. Deleted imports in additions-only test files
$DIFF -- tests/ '*.py' | grep -E '^-\s*(import |from )' && echo "FAIL 1 deleted imports"

# 2. Deleted test bodies
$DIFF -- tests/ | grep -E '^-\s*(async def test_|def test_)' && echo "FAIL 2 deleted test body"

# 3. Deleted validator errors / raised exceptions
$DIFF -- app/services/ | grep -E '^-.*errors?\.append\(|^-.*raise ' && echo "FAIL 3 deleted validator error"

# 4. Gratuitous subquery wraps
$DIFF | grep -E 'FROM \(SELECT|SELECT COUNT\(\*\) FROM \(' && echo "FAIL 4 subquery wrap"

# 5. Form(default=...) with non-None non-empty default on bool-ish fields
$DIFF | grep -E 'Form\(default="[^"]+"' | grep -v 'default=None' && echo "FAIL 5 bool default trap"

# 6. bool(form.get(...)) without an adjacent __<field>_submitted marker
$DIFF | grep -E '^\+.*bool\(.*form\.get\(|^\+.*bool\(.*Form\(default=None' && echo "FAIL 6 inspect bool marker"

# 7. Removed WHERE <scope_column> clauses
$DIFF -- '*.py' '*.sql' | grep -E '^-.*WHERE.*<scope_column>' && echo "FAIL 7 removed access scope filter"

# 8. New access scope-resolution helpers
$DIFF | grep -E '^\+.*def _?(resolve|resolve)_\w+_<scope_column>' && echo "FAIL 8 access scope helper"

# 9. New early-return inside validator error accumulation
$DIFF -- app/services/validation.py 2>/dev/null | grep -E '^\+\s*(return errors|return \[)' && echo "FAIL 9 validator early return"

# 10. Tests accepting 302 as PASS alongside 4xx
$DIFF -- tests/ | grep -E '^\+.*in \{302,.*4\d\d\}|^\+.*\{.*302.*403|^\+.*\{.*302.*404' && echo "FAIL 10 fuzzy redirect pass"
```

Every hit must be one of:

- a finding,
- a documented exception,
- or a false positive with explanation.

Review evidence must include output or `clean`/`N/A` for each signature.

## Gate 5 - Functional Evidence

Check behavior, not just pattern presence.

- Positive path:
- Negative path:
- Access-boundary path:
- Archive path:
- Boolean absent/unchecked/checked matrix:
- Redirect `Location` assertion:
- Manual smoke:

## Gate 6 - Test Quality

- [ ] Test names match bodies.
- [ ] Tests fail for the intended reason if behavior regresses.
- [ ] Existing tests were not weakened.
- [ ] No broad `try/except/pass`.
- [ ] No fuzzy status-code assertions.

## Required Commands

```bash
<focused command from task prompt>
<broader regression command if needed>
ruff check <changed python files or .>
```

## Findings Format

- `[CRITICAL | SHOULD_FIX | FUTURE] <summary>`
  - File: `<file:line>`
  - Evidence:
  - Why it matters:
  - Required fix:

## Calibration

- `PASS`: all blocking requirements and applicable invariants pass.
- `CHANGES REQUIRED` + `Severity: should-fix`: concrete non-blocking issue
  that may be deferred to QA within budget.
- `CHANGES REQUIRED` + `Severity: block`: concrete bug, missing requirement,
  security issue, broken test, or missing evidence.
- `BLOCKED`: the prompt/spec is contradictory, diff base is wrong, evidence is
  missing, or an operator decision is required.
