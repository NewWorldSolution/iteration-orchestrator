# Iteration QA - <ITERATION_ID> - <Perspective>

## QA Metadata

- Perspective: `<Security | Architecture | Test | Product | Process>`
- Diff base: `<phase branch or sha>`
- Iteration branch: `<branch>`
- Reviewer: `<agent/family>`
- QA mode: `<standard | tooling | docs-only>`

## Required Inputs

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `iterations/_review_template.md`
4. `<iteration>/prompt.md`
5. `<iteration>/tasks.md`
6. `<iteration>/prompts/*.md`
7. `<iteration>/reviews/*.md`
8. `tools/logs/<iter>/run_state.json`
9. `tools/logs/<iter>/cost.jsonl`
10. `tools/logs/<iter>/reviews/*.md`
11. grep exception artifacts if present

## Output Contract

Use this section order:

1. `## Summary`
2. `## Blocking Decision`
3. `## Invariant Closure`
4. `## Findings`
5. `## Evidence`
6. final line: `QA Verdict: OK | CONCERNS | BLOCK | INCOMPLETE`

## Gate 1 - Diff Base and Inverse Diff

```bash
git diff <diff-base>..<iteration-branch> --name-only
git diff <iteration-branch>..origin/<phase-branch> --name-only
```

Classify every inverse-diff file as:

- expected post-branch phase work,
- missing sync/stale base,
- not applicable,
- or blocking unexplained diff.

## Gate 2 - Allowed-File Union

Forward diff must be inside the union of task allowed files plus documented
orchestrator-owned status artifacts.

## Gate 3 - Invariant Closure

Fill one row per invariant from `<iteration>/prompt.md` and per applicable
project invariant.

| Invariant | Evidence | Status |
|---|---|---|
| `<invariant>` | `<test/query/file:line>` | `<PASS/FAIL>` |

Any `FAIL` on a blocking invariant means `QA Verdict: BLOCK`.

## Gate 4 - Cross-Task Grep Gates

Run all ten `_prompt_rules.md` Rule 5 signatures plus iteration-specific
signatures. Every hit must map to one of:

- documented exception,
- real finding,
- false positive with evidence.

## Gate 5 - Preserved Behavior

```bash
pytest <preserved behavior fixture> -v
```

If the fixture is required but missing, report a Process finding and use
`QA Verdict: INCOMPLETE` or `BLOCK` depending on risk.

## Gate 6 - Per-Task Review Audit

| Task | Review artifact present? | Verdicts | Deferred findings | Suspicious gaps |
|---|---|---|---|---|
| `<task>` | `<yes/no>` | `<verdicts>` | `<notes>` | `<notes>` |

## Gate 7 - Regression Evidence

Record exact commands and result tails:

- focused tests:
- full tests:
- lint:
- manual smoke:
- CI evidence if available:

## Perspective-Specific Deep Read

### Security

- Access scope isolation.
- Authz/authn.
- Token/secret handling.
- PII/logging.
- Injection/XSS/command risk.
- CSRF/CORS/session.
- Deployment secret exposure.

### Architecture

- Layering.
- Rule-of-one.
- Migration safety.
- Schema compatibility.
- Module boundaries.
- Performance implications.

### Test

- Acceptance criteria coverage.
- Negative paths.
- Test oracle quality.
- Deterministic isolation.
- CI parity.
- No weakened tests.

### Product

- User workflow.
- Role-specific behavior.
- Copy/localization.
- PL locale and diacritics.
- Missing UX states.
- Business intent.

### Process

- Task slicing.
- Scope creep.
- Branch freshness.
- Cost/wall time.
- Pair swaps.
- Manual intervention.
- Prompt quality.

## Findings Format

Use stable finding IDs so QA findings can be referenced by retro and the next
iteration prompt:

- Security: `QA-S1`, `QA-S2`, ...
- Architecture: `QA-A1`, `QA-A2`, ...
- Test: `QA-T1`, `QA-T2`, ...
- Product: `QA-P1`, `QA-P2`, ...
- Process: `QA-PR1`, `QA-PR2`, ...

Format each issue as:

- `[<ID>] [CRITICAL | SHOULD_FIX | FUTURE] <summary>`
  - Task(s):
  - File: `<file:line>`
  - Evidence:
  - Required fix or follow-up:

## Blocking Decision Guide

- `OK`: zero CRITICAL, no invariant failure, evidence complete.
- `CONCERNS`: non-blocking SHOULD_FIX/FUTURE findings only.
- `BLOCK`: invariant failure, access scope/security/data integrity issue, or real
  blocking regression.
- `INCOMPLETE`: required evidence missing, diff base invalid, role timed out,
  or QA could not inspect required artifacts.
