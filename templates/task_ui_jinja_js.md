# <TASK_ID> - <UI/Template/JS Task Title>

## Execution Metadata

- Iteration: `<iteration-id>`
- Task branch: `<branch>`
- Depends on: `<task ids or none>`
- Execution mode: `<orchestrator | manual>`
- Risk category: `<risk_category>`

## Required Read Order

1. `CLAUDE.md`
2. `iterations/_prompt_rules.md`
3. `iterations/_review_template.md`
4. `<iteration>/prompt.md`
5. `<affected route/template/static files>`
6. `<localization dictionaries if user-facing text changes>`

If any required file is missing or contradicts this prompt, STOP and report the contradiction.

## Goal

After this task, `<role/user>` can complete `<workflow>` in the server-rendered
UI.

## UX Current State

- Current screen/route:
- Current pain or defect:
- Existing copy/localization keys:

## UX Target State

- New rendered state:
- Form behavior:
- Error/empty/loading state:
- Configured-locale copy:
- Accessibility/basic keyboard behavior:

## Non-Goals

- Do not add a JS framework or SPA behavior.
- Do not create new CSS/JS files unless explicitly allowed.
- Do not move business validation to JavaScript.
- Do not change backend behavior outside the stated UI workflow.

## Allowed Files

```text
app/templates/<...>.html
app/routes/<...>.py
static/<...>.js
app/locales/<locale_a>.py
app/locales/<locale_b>.py
tests/<...>.py
```

## Forbidden Files and Symbols

- No unrelated templates.
- No unrelated routes.
- No business logic changes outside approved service calls.
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

| Invariant | Classification | Required evidence |
|---|---|---|
| <Invariant 1 - e.g. backend authority> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 2 - e.g. access boundary> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |
| <Invariant 3 - e.g. localization coverage> | `<applies/preserve only/not applicable>` | `<evidence or N/A reason>` |

## UI Contract

- Route(s):
- Template(s):
- Form fields:
- Hidden-marker pattern: every checkbox/toggle renders a sibling `<input type="hidden" name="__<field>_submitted" value="1">` (Rule 1).
- Server-side validation:
- Client-side enhancement:
- Error state:
- Empty state:
- Redirect behavior:

## Acceptance Matrix

| Requirement | Evidence | Command/test/manual check | Blocking? |
|---|---|---|---|
| `<workflow works>` | `<TestClient/manual smoke>` | `<command>` | yes |

## Required Tests

- GET renders expected UI.
- POST happy path if form changed.
- POST invalid path with inline error.
- Access-boundary access if any ID is in URL/form.
- Redirect `Location` pinned if redirect is expected.
- localization keys present in configured locales when copy changed.
- Boolean 3-scenario matrix for every checkbox/toggle touched (Rule 10):
  absent from POST body -> preserved/default; marker present + unchecked ->
  False; marker present + checked -> True. TestClient round-trip is the
  minimum evidence; grep/AST checks do not satisfy this.

## Manual Smoke

Run local app if the UI cannot be verified by TestClient alone.

- URL:
- Role:
- Browser viewport:
- Actions:
- Expected visible text:
- Localized characters:
- No overlapping/truncated text:

## Required Commands

```bash
pytest <focused route/ui tests> -v
pytest tests/ -x
ruff check <changed python files>
```

## Preserved Behavior or N/A

State one of:

- Preserved-behavior fixture: `<path and command>`
- Preserved-behavior checks: `<existing route/template/localization tests>`
- N/A because: `<why this UI task cannot affect existing behavior>`

If route behavior, validation display, redirects, localization, access scope filtering, or
form submission semantics can change, `N/A` is not acceptable.

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
- Include manual smoke evidence in the task report if UI changed.

For manual mode:

```bash
git add <explicit files>
git status --short
git commit -m "<TASK_ID>: <UI summary>" -m "<tests, manual smoke, deviations>"
git push -u origin <branch>
```

If implementation intentionally improves on or deviates from this prompt,
include a `Deviation:` block in the commit body/report with prompt text,
implemented behavior, why it is safer, and evidence.
