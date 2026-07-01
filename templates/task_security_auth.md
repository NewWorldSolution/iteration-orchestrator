# <TASK_ID> - <Security/Auth Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: `<orchestrator | manual>`
- Risk category: `security_compliance`
- Reviewer expectation: cross-family review required

## Required Read Order

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `iterations/_review_template.md`
4. `<iteration>/prompt.md`
5. `<auth/security source files>`
6. `<existing security tests>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Goal

After this task, `<security property>` is enforced without changing unrelated
user-visible behavior.

## Threat / Risk

- Threat:
- Current exploit or weakness:
- Data/user impact:
- Required defensive property:

## Non-Goals

- Do not change login/session behavior outside this scope.
- Do not reveal whether an account/resource exists unless explicitly intended.
- Do not log secrets, tokens, emails, regulated identifiers, or PII beyond approved
  operational identifiers.
- Do not change deployment secrets or environment configuration unless listed.

## Allowed Files

```text
<exact auth/security files>
<exact tests>
```

## Forbidden Files and Symbols

- No unrelated middleware refactors.
- No broad route rewrites.
- No new dependencies unless explicitly approved.
- No `tasks.md` edits.

## Applicable Security Invariants

| Invariant | Classification | Required evidence |
|---|---|---|
| <Invariant 1 - e.g. access boundary> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 2 - e.g. response consistency> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 3 - e.g. log minimization> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |

## Edit Contract

### Authorized edits

- `<file>::<symbol>` - `<allowed change>`

### Frozen behavior

- `<file>::<symbol>` - `<must stay identical>`

## Implementation Rules

- Use parameterized SQL only.
> **Access-scope filter audit.** Every SELECT/UPDATE/DELETE in this task's
> diff — including resolution helpers, audit scripts, and migration
> queries — MUST include `WHERE <scope_column> = ?`. A query without a
> access scope filter is a CRITICAL security finding at QA regardless of
> whether it is read-only or admin-only. Grep signatures 7 and 8 above
> are mandatory pre-QA.
- Do not introduce access-scope filter exceptions in this template. A rare token or
  globally unique lookup exception requires an explicit operator-approved
  exception in the task prompt, with rationale, non-enumeration evidence, and
  access-boundary tests. Without that explicit exception, missing access scope filters
  are blocking defects.
- Client-side checks are progressive enhancement only; server-side checks are
  definitive.
- Error responses must not introduce enumeration signals.
- Logs must use approved fields only.

## Acceptance Matrix

| Requirement | Evidence | Command/test/manual check | Blocking? |
|---|---|---|---|
| `<security property>` | `<test>` | `<command>` | yes |

## Required Security Tests

- Authorized happy path.
- Unauthorized same-scope role path.
- Access-boundary path.
- Missing/expired/invalid token or session path if relevant.
- Backend enforcement bypass attempt.
- Log/PII assertion if logging changed.
- Enumeration-safe response equality if auth recovery/invite/reset changed.

## Mechanical Checks

```bash
git diff <base>..HEAD | grep -E 'secret|token|password|email' || true
git diff <base>..HEAD | grep -E 'SELECT|UPDATE|DELETE' || true
```

Manual inspect every hit.

## Required Commands

```bash
pytest <focused security tests> -v
pytest tests/ -x
ruff check <changed python files>
```

## Preserved Behavior or N/A

State one of:

- Preserved-behavior fixture: `<path and command>`
- Preserved-behavior checks: `<existing tests/fixtures>`
- N/A because: `<why this task cannot affect existing behavior>`

If auth, access scope, token, anti-enumeration, or logging behavior can change,
`N/A` is not acceptable.

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
- Record any operator-approved access-scope filter exception in the task report.

For manual mode, the commit body must include security evidence and any
operator-approved access-scope filter/token exception.

If implementation intentionally improves on or deviates from this prompt,
include a `Deviation:` block in the commit body/report with prompt text,
implemented behavior, why it is safer, and evidence.
