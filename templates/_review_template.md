> **AUDIT SCOPE:** This QA reviews the iteration's product code AND the
> surrounding pipeline (tests, CI, deploy, dev-env, prompts/scaffolding).
> Issues found in pipeline surface are equally critical to issues in
> product code. An iteration-QA scoped only to orch internals is
> incomplete — pipeline regressions ship just as easily as product bugs.
>
> **Tooling-iteration variant:** iterations under `iterations/tools/*`
> skip the **Product** perspective and emphasize **Architecture** +
> **Process**. Substitute "operator workflow" for "end-user flow".

# Review template skeleton (per-task + iteration-level)

**Use:** copy into `iterations/<phase>/<iter>/reviews/review-t<k>.md` (per
task) or `review-iteration.md` (QA), then specialize.

**Companion doc:** `iterations/_prompt_rules.md` (must read first — this
template implements its Rules 2b, 4, 5, 8, 10).

**Origin:** distilled from review templates at
`iterations/<phase>/<iter>/reviews/review-t*.md`, after adversarial
analysis revealed most checks were grep-for-pattern-exists and missed
the CRITICAL defects in an early iteration. The template shifts the default from
[GREP] to [FUNCTIONAL] everywhere it matters.

---

## PER-TASK review template — copy → specialize

```markdown
# Review — I<N>-T<k> — <title>

**Task branch:** `<branch>`
**Diff base:** `<task-base>`
**Depends on:** I<N>-T<k-1> DONE (if any)

## Setup (mandatory — run once before the gates below)

For iterations that DON'T branch sub-tasks off the iteration branch
(i.e. tooling sprints where each task lives on its own branch off the
iteration branch), record the diff base before each task starts:

```bash
# Before handing the task prompt to its agent:
git checkout <iteration-branch>
git rev-parse HEAD          # <-- copy this SHA; that's TASK_BASE for this task
```

Then before reviewing:

```bash
export TASK_BASE=<SHA you recorded above>
git diff $TASK_BASE..HEAD --name-only   # sanity: should print only the task's intended files
```

For orch-driven iterations where each task gets its own branch
(`<phase>/iN/tM-slug`), the diff base is the task PR's actual base branch,
which is normally the iteration branch (`<phase>/iteration-N`). `TASK_BASE`
export is not needed; substitute that branch name in the gates below.

**Important:** per-task review scope is always computed against the task
PR base (or the pre-recorded task-start SHA for non-PR/task-base flows),
NOT against the phase branch merge-base. Comparing a task branch to
`phase-*` after earlier task PRs have already merged into the iteration
branch will make inherited T1/T2/T3 work look like part of T4 and
produces false scope failures.

Use this decision rule:

```bash
# Task PR off an iteration branch (common product flow):
git diff origin/<iteration-branch>..HEAD --name-only

# Task branch with a recorded pre-task SHA (tooling/manual flow):
git diff $TASK_BASE..HEAD --name-only

# Full iteration QA only (NOT per-task review):
git diff <phase-branch>..<iteration-branch> --name-only
```

## Read first
1. `iterations/_prompt_rules.md` (current: v2)
2. `iterations/_review_template.md` (this file — companion to rules)
3. `iterations/<phase>/<iter>/prompts/t<k>-<slug>.md` (what codex was told)
4. `iterations/<phase>/<iter>/prompt.md` (iteration context)

## Verdict format (mandatory — orch parses this)
Your response must end with exactly one of these trailing blocks:

```text
Verdict: PASS
```

```text
Verdict: CHANGES REQUIRED
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

No non-empty line may follow the verdict block. Extra `Verdict:` lines
earlier in the response are allowed only as quoted evidence; the parser at
`orch/review.py` evaluates the trailing verdict block strictly.

### Optional severity tag

When (and only when) `Verdict: CHANGES REQUIRED`, you MAY append one
`Severity:` line after the verdict to disambiguate severity for the
failure-triage classifier (`orch/triage.py`):

```
Verdict: CHANGES REQUIRED
Severity: should-fix
```

Allowed values: `should-fix` | `block` (case-insensitive).

* `Severity: should-fix` — concrete but non-blocking (style nits with
  evidence, redundant import, missing docstring on public API). The
  triage classifier may DEFER_TO_QA (note the finding, ship the task,
  surface in iteration QA report) up to the configured defer budget
  (default 2 deferrals per task).
* `Severity: block` — must be fixed before the task can ship. The
  triage classifier routes to FIX_NOW.
* **Absent severity** — treated as `block` (fail-safe; matches the
  original fix-loop behaviour). Reviewers that don't emit `Severity:` keep
  working unchanged.

Unknown severity tokens (e.g. `Severity: nit`) degrade silently to
"absent" — the verdict still parses but the classifier treats the round
as `block`.

---

## 1. Structural checks [STRUCTURAL]

- [ ] `git diff <task-base>..HEAD --name-only` matches the task's Allowed
      files list EXACTLY. No extras, no escapees.
- [ ] `ruff check <changed .py files>` clean (no warnings suppressed).
- [ ] No `orch/` edits (iteration metadata only — operator owns).
- [ ] No edits to `tasks.md` (status column is orch-owned).

## 2. Scope-sweep gate [STRUCTURAL — catches mechanical sweeps]

Run against the task diff (NOT the full iteration — that's the QA gate):

```bash
DIFF="git diff <task-base>..HEAD"

# Gratuitous subquery wrap signature
$DIFF | grep -E 'FROM \(SELECT|SELECT COUNT\(\*\) FROM \(' && echo "FAIL: subquery wrap"

# New access scope-resolution helpers signature
$DIFF | grep -E '^\+.*def _?(resolve|resolve)_\w+_<scope_column>' && echo "FAIL: new access scope-resolver"

# Removed WHERE <scope_column> clauses
$DIFF | grep -E '^-.*WHERE.*<scope_column>' && echo "FAIL: removed access scope filter"

# Form(default="...") with non-None default on bool-ish fields
$DIFF | grep -E 'Form\(default="[^"]+"' | grep -v 'default=None' && echo "FAIL: Form default trap"

# Early return in validator error accumulation — T2 short-circuit
$DIFF -- app/services/validation.py 2>/dev/null | grep -E '^\+\s*(return errors|return \[)' && echo "FAIL: early return"

# Deleted imports in additions-only test files
$DIFF -- tests/ | grep -E '^-\s*(import |from )' && echo "FAIL: deleted imports"
```

Any `FAIL:` = `Verdict: CHANGES REQUIRED` — do not proceed to §3.

## 3. Preserved-behavior gate [FUNCTIONAL — catches silent drops]

```bash
pytest tests/_preserved_behavior_i<N>.py -v
```

This file was snapshot on <phase-branch> HEAD before the iteration
started (per `_prompt_rules.md` Rule 2b). Any failure =
`Verdict: CHANGES REQUIRED` with the specific fixture name.

If this file doesn't exist, the iteration was scaffolded without
Rule 2b — flag to operator, do NOT proceed.

## 4. Functional acceptance checks [FUNCTIONAL]

Specialize per task. The template below is the shape every check must
take. Ban pure `grep -n "pattern"` checks — if a pattern's presence
matters, show the pattern executes correctly via TestClient + DB read.

### 4.1 Defect-class coverage (5 classes per _prompt_rules.md Rule 4b)

For this specific task, tick which classes apply and fill in the check:

- [ ] **(a) Default-value semantics with omitted fields.**
  ```python
  # Submit endpoint with <target field> ABSENT from body
  resp = client.post("<endpoint>", data={"<other_required>": "..."})
  assert resp.status_code in (<expected_status_set>)
  row = db.execute("SELECT <field> FROM ... WHERE id=?", [<id>]).fetchone()
  assert row["<field>"] == <expected_on_absent>, \
      "<describe the absence semantics>"
  ```

- [ ] **(b) Hidden-marker round-trip.**
  ```python
  # Marker absent → preserve existing value
  db.execute("UPDATE ... SET <field>=<original> WHERE id=?", [<id>])
  client.post("<endpoint>", data={"<other>": "..."})  # no marker, no field
  assert db.execute(...).fetchone()["<field>"] == <original>, "preserved"

  # Marker present, field absent → unchecked → False/empty
  client.post("<endpoint>", data={"__<field>_submitted": "1", "<other>": "..."})
  assert db.execute(...).fetchone()["<field>"] == <unchecked_value>

  # Marker present, field set → checked → True/value
  client.post("<endpoint>", data={"__<field>_submitted": "1", "<field>": "1"})
  assert db.execute(...).fetchone()["<field>"] == <checked_value>
  ```

- [ ] **(c) Access-boundary access on ID-bearing endpoints.**
  ```python
  # Seed two access scopes; call as scope A with scope B's resource id
  seed_scope(1, resource=<row_A>)
  seed_scope(2, resource=<row_B>)
  _login_as_scope(client, 1)
  resp = client.get("<endpoint>", params={"id": row_B.id})
  assert resp.status_code == 404
  # For CLIs / audits:
  result = subprocess.run([..., "--scope-id", "1", "--resource-id", str(row_B.id)])
  assert result.returncode == 1
  assert "<scope_b_unique_token>" not in (result.stdout + result.stderr)
  ```

- [ ] **(d) Archived row visibility.**
  ```python
  db.execute("UPDATE ... SET is_active = FALSE WHERE id=?", [<id>])
  # List: must exclude
  assert <id> not in [r["id"] for r in list_resource(db, <scope_column>)]
  # Detail: must 404
  assert client.get(f"<detail_url>/{<id>}").status_code == 404
  ```

- [ ] **(e) Redirect-location assertion (prevents CSRF-false-pass).**
  ```python
  # Don't write `assert resp.status_code in {302, 403, 404}` — 302 with
  # a Location pointing anywhere (including /login) would pass.
  resp = client.post("<endpoint>", data=<invalid_for_access scope>,
                     follow_redirects=False)
  assert resp.status_code == 404, resp.status_code  # or specific
  # If 302 is genuinely correct, pin the Location:
  # assert resp.headers["Location"] == "<expected_location>"
  ```

### 4.2 Two-input minimum

Every functional check uses ≥2 distinct inputs. A single input that
happens to match the expected output on a default-populated row proves
nothing. C2 would have slipped with one input if the test only used an
already-False row.

## 5. Test discipline [STRUCTURAL]

- [ ] All test changes in files listed as `additions only` are indeed
      additions — verified by §2 deleted-imports / deleted-tests greps.
- [ ] Every new test function name describes what it actually tests
      (e.g. `test_admin_create_record_unchecked_persists_false` must
      POST to `/admin/<entity>/new`, not `/edit`). Review by reading
      the test body, not just the name.
- [ ] No `try: ... except: pass` as test oracle; use `pytest.raises`.
- [ ] No `assert status_code in {302, 4xx}` — pin the exact expected
      status or assert redirect-location.

## 6. Behavior diff vs phase branch [FUNCTIONAL — catches silent refactors]

```bash
# Full test suite on task branch — must pass
pytest tests/ -x

# If this task modified a validator / calculator / query helper:
# run the preserved-behavior fixture (Rule 2b)
pytest tests/_preserved_behavior_i<N>.py -v
```

## 7. Manual smoke (only if form/UI touched)

**Only applies to tasks that modify templates/ or static/js/.** If this
task is backend-only, write "N/A — backend-only task" and skip.

If form/UI:
- [ ] Scaffolder-specified manual flow ran against local uvicorn on
      seeded sandbox data
- [ ] Report lists exact URLs visited, buttons clicked, and observed
      DB state after each action
- [ ] Localized characters visible (no `?` or mojibake)

## Report format (what you write in the review response)

```
Verdict: <PASS | CHANGES REQUIRED | BLOCKED>

## Findings (if CHANGES REQUIRED / BLOCKED)
- [<severity>] <one-sentence problem>
  File: <file:line>
  Required fix: <concrete rewrite or behavior to restore>

## Evidence (always)
- §1 structural: <summary + relevant commands run>
- §2 sweep gate: <output of each grep>
- §3 preserved behavior: <pytest output tail>
- §4 functional checks: <list checks run + results>
- §5 test discipline: <confirmation>
- §6 behavior diff: <pytest tail>
- §7 manual smoke: <evidence or N/A>
```
```

---

## ITERATION-LEVEL review template (review-iteration.md)

QA runs after all tasks DONE. 5 reviewers in parallel. Each gets a
different perspective (Security, Architecture, Test, Product, Process).
This skeleton is shared; specialize perspective at the top.

For phase-closing iterations, and for any iteration that gates a
production/phase merge, run iteration QA from both model families
(codex + claude). Multiple iterations proved that same-frame reviewers
can miss absence-class defects: in separate iterations the second family
caught a missing migration and an out-of-sync hotfix only because it
reviewed from a different frame of reference.

```markdown
# Iteration Review — I<N> — <perspective>

**Your perspective:** <Security | Architecture | Test | Product | Process>

**Diff base:** `<phase-branch>`
**Scope:** all merged task commits for I<N>

## Read first
1. `iterations/_prompt_rules.md`
2. `iterations/_review_template.md`
3. `iterations/<phase>/<iter>/prompt.md`
4. `iterations/<phase>/<iter>/tasks.md` (all tasks' Goals + Allowed files)

## 1. Cross-task structural diff gate

Run both diff directions before perspective-specific review:

```bash
# Forward diff — what this iteration added or changed.
git diff <phase-base>..<iter-branch> --name-only

# Inverse diff — required for closer iterations and any iteration
# whose phase branch may have advanced. Catches missing cherry-picks /
# stale-base / out-of-band hotfix-not-synced. Validated in practice: one
# family's QA caught an unsynced hotfix via this inverse diff while the
# other missed it by relying only on the forward diff.
git diff <iter-branch>..origin/<phase-branch> --name-only
```

The forward diff must stay inside the iteration's allowed-file union.
The inverse diff must be classified explicitly: either expected
post-branch phase work, or a merge-blocking stale-base / missing-sync
finding. A phase-closing iteration with an unexplained inverse diff is
not merge-ready.

## Invariant closure (mandatory per-task — catches drops)

For each Task Goal invariant listed in `prompt.md`:

| # | Invariant | Status | Evidence |
|---|-----------|--------|----------|
| 1 | <invariant> | PASS / FAIL | <command or file:line> |
| ... |

A FAIL here = blocking regardless of your perspective's separate findings.

## Cross-task grep gate (mandatory — catches silent refactor across task boundaries)

Run the full-iteration signature list from `_prompt_rules.md` Rule 5
(10 signatures). Any hit that isn't documented in
`tools/logs/<iter>/grep_exceptions.md` = blocking.

## Preserved-behavior regression (mandatory)

```bash
pytest tests/_preserved_behavior_i<N>.py -v
```

Any failure = blocking. If the fixture file doesn't exist, the iteration
was scaffolded without Rule 2b — report as PROCESS FAILURE.

## Perspective-specific deep read

### Security
- Access-boundary access on every endpoint added/modified (not just grep —
  seed two access scopes, curl as A with B's ids)
- New SQL queries: every SELECT/UPDATE/DELETE has `WHERE <scope_column> = ?`
- New form fields: default-value semantics match operator intent
- Secrets / PII in logs or error messages

**Production-pipeline scope:**
- CI workflow security: `permissions:` minimal, `secrets.*` not echoed,
  no `pull_request_target` on untrusted forks
- Deploy pipeline auth: SSH keys / tokens not embedded in repo or logs
- Runtime env hardening: secrets injection (env vars, .env not
  committed), file-permission expectations on prod
- Test fixtures: any seeded credentials are non-production / clearly
  fake

### Architecture
- Layering: no inversion (services importing from higher-level modules)
- Rule-of-one: no duplicated business logic across validator / route /
  audit / template
- Raw SQL vs service call consistency
- Migration path: new schema changes have a tracked migration OR a
  VERIFY_AT_GOLIVE comment + backlog entry

**Production-pipeline scope:**
- Test infrastructure: `conftest.py` fixtures support access-scope scoping,
  no shared mutable state across tests
- CI workflow structure: jobs ordered correctly (lint → test → deploy
  gate), required-checks list matches branch protection
- Deploy gate paths: dry-run staging step exists for any production
  schema/code change; `docs/deployment.md` reflects current reality

### Test
- Every task's acceptance checklist actually exercises the behavior
  (re-run §4 of per-task template)
- New tests use correct scenarios (name matches body)
- Additions-only discipline held across all task diffs

**Production-pipeline scope:**
- Meta-test-suite: orch's own tests (`tests/test_orch_*`), `scaffold_lint`,
  `scaffold_review` cover the iteration-discipline rules they claim to
- New rules added in this iteration have at least one passing test that
  would fail if the rule were removed

### Product
- Feature works end-to-end for the real end user (operator / admin) —
  trace one full happy-path + one failure path through the
  merged diff
- UI locks + backend validator agree (belt-and-suspenders)
- localization: every supported locale covered, diacritics correct

**Production-pipeline scope:**
- Operator-facing surface: `run.md`, `CLAUDE.md`, dashboard outputs,
  readiness/QA/retro reports — text changes match the new reality
- Failure messages name the next operator action (not just the symptom)

### Process
- Task decomposition stayed single-purpose (no bundled concerns)
- Each task's diff size fits its contract
- Pair-swap history: if fired, was the root cause prompt / pairing /
  reviewer-calibration? (Rule 7 intent)
- Cost within the configured per-iteration budget (watch for pair-swap overruns)
- Reporting unit: prefer wall-time as the headline metric; treat dollar
  figures as secondary.

**Production-pipeline scope:**
- CI feedback loop: failed CI runs surface in the iteration's logs
  (not silently green-merged); broken-windows policy enforced
- Deploy dry-run gate: staging Postgres dry-run executed for any
  production-touching change (per `docs/deployment.md`)
- Retrospective discipline: discipline items from `_carry_forward_to_i*`
  were applied this iteration (not just listed)

## Findings

For each issue found:
- **[CRITICAL | SHOULD_FIX | FUTURE]** <one-sentence summary>
  - File: `<file:line>`
  - Why it matters: <1-2 sentences>
  - Required fix / recommendation: <concrete>

## Verdict

One of: `OK` / `CONCERNS` / `BLOCK`.
  - `OK`: zero CRITICAL, ≤2 SHOULD_FIX, all invariants PASS
  - `CONCERNS`: 0-1 CRITICAL or 3+ SHOULD_FIX, iteration mergeable after
    fix-up patch
  - `BLOCK`: 2+ CRITICAL, or any invariant FAIL, or any grep-gate hit
    that's a real regression. Iteration does NOT merge without a patch.
```

---

## How this template prevents the absence-class miss

| Defect class | Which section catches it |
|---|---|
| Boolean default (`Form(default="1")` unchecked-creates-True) | §2 (grep signature 4 — `Form(default=...)` trap) + §4.1(a) default-absent test |
| Edit silently flips a field without a marker | §4.1(b) hidden-marker round-trip — 3 inputs |
| Gratuitous subquery wrap (`FROM (SELECT ...)`) | §2 (grep signature 1 — `FROM \(SELECT`) |
| Access-scope leak in a new resolver helper | §2 (grep signature 2 — new access scope-resolver) + §4.1(c) access-boundary cURL |
| Silently dropped conditional-field validation rule | §3 preserved-behavior fixture |
| Validator early-return short-circuit | §2 (grep signature 5 — early-return in validator) |
| Import deletions in an additions-only test file | §2 (grep signature 6 — deleted imports) |
| Misnamed test hiding a real defect | §5 — "read test body not name" |
| `{302, 403, 404}` in an access-scope isolation test | §5 + §4.1(e) redirect-location |

Nine defect classes, nine separate catches. Zero would need per-task
reviewer cleverness — all are mechanical.
