# Iteration prompt-writing rules

**v2.** Rewritten after adversarial-agent reviews of v1 against a real
iteration's prompts and defects. v1 was prose guidance the scaffolder
applied by memory; v2 ships mandatory boilerplate, executable fixtures,
and an expanded grep-gate list. Goal: drive cross-cutting QA defects down
to a low single-digit count per iteration.

**Total: 15 rules.** The original v1 had 7 rules (1-7 below); v2 added
three more (8: access-scope isolation rule per SELECT/UPDATE/DELETE, 9: prompt
code examples are executable rules, 10: boolean-field mandatory tests).
Later revisions added Rule 11: explicit "Final step — commit and push"
section in every prompt, after several tasks shipped with
implementer-completed-but-uncommitted edits; Rule 12: orchestrator
state-machine recovery scope exception; Rule 13: branch freshness before
task fire; Rule 14: config/pytest-marker allowlisting when prompts
introduce new test taxonomy; and Rule 15: verification right-sizing per
task type.
**If you skim past Rule 7 looking for "the end of the rules",
you've missed eight.** A scanner that reads "1-7" without checking the
file's end will silently bypass the access-scope filter audit, the early-return
ban, the boolean-field 3-scenario test matrix, and the commit-instruction
that prevents 5/14 of recent codex tasks from ending in uncommitted state.

**Who reads this:** codex (implementer) and claude (reviewer) — via the
iteration `prompt.md` and per-task `prompts/t<k>-*.md`, which MUST link
to this doc with `Read first: _prompt_rules.md`. The
operator's scaffolder (usually Claude-in-session) is responsible for the
self-check list at the bottom before `orch validate`.

---

## 1. Resolve contradictions BEFORE writing the prompt

A prompt that says both "default True on create" AND "unchecked checkbox
means False" is literally unsatisfiable. In conversation this surfaces
as "wait, those don't both work." In a prompt, codex picks one path
silently and both failure modes ship together.

**Before handing off:** read your own prompt twice. For every `if X then
Y` rule, ask "what if NOT X?" Write the negative branch explicitly.

**Hidden-marker pattern** — any checkbox, toggle, or nullable-boolean
form field MUST render a sibling `<input type="hidden" name="__<field>_submitted" value="1">`.
Handler parses: if marker absent → skip update (edit) or use default
(create); if marker present → field value is definitive including
"absent means False." No exceptions.

---

## 2. Preserved-behavior section — executable, not prose

Every prompt that touches existing code MUST include both of:

### 2a. Preserved-behavior list (prose)

> **Preserved behavior.** The following rules/outputs/error messages
> must remain byte-identical after this task. Any diff line
> `-.*errors?\.append\(` or `-.*raise ` inside the listed files = STOP.
>   - `<rule 1 with file:line reference from <phase-branch> HEAD>`
>   - `<rule 2>`
>   - ...

### 2b. Behavior-regression fixture (executable)

Before the iteration starts, scaffolder runs on `<phase-branch>` HEAD:

```python
# tests/_preserved_behavior_i<N>.py — snapshot on <phase-branch> HEAD
FIXTURES = [
    ("manual_mode_with_extra_option",
     {"mode": "manual", "dependent_pct": "50", ...},
     ["Dependent % must be empty when mode is manual."]),
    ("unknown_parent_plus_bad_quantity",
     {"parent_id": 99999, "quantity": "-5", ...},
     ["Parent must reference an active record.", "Quantity must be > 0."]),
]
@pytest.mark.parametrize("name, inp, expected", FIXTURES)
def test_behavior_preserved(name, inp, expected, db):
    assert sorted(validate_record(inp, db, 1)) == sorted(expected)
```

**Discovery method (how to populate FIXTURES):** pick 5-10 input shapes
from `tests/test_<module>.py` on `<phase-branch>`, run the current function,
capture its output, paste as expected. Covers the rules you'd otherwise
forget were there. This single step would have caught a dropped
conditional-field rule ("field X forbidden when mode is manual") plus a
validator short-circuit in an early iteration.

---

## 3. No mechanical sweeps — **mandatory prompt clause**

Every schema/query/render/handler task MUST include VERBATIM this block
in the prompt body (not "see rules"):

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

**Antipattern flags to grep your own prompt for** before handoff:

- "every SELECT" / "every render" / "every handler" / "all callers"
- "Grep for X and extend every match"
- "only if blocked by" / "only if needed" / "or similar"
- A specific named symbol (e.g. `count_active_records`) included in a
  sweep list without the author checking whether the sweep applies

If any of these appear in your prompt, rewrite as an explicit enumerated
list. In one iteration, a symbol dropped into a sweep list without a
scope check authored a gratuitous-subquery defect directly.

---

## 4. Functional acceptance checks — template, not abstraction

v1 said "add a functional check." Authors wrote one. It didn't bite.
v2 ships the template.

### 4a. Template

Every form/route/validator task MUST include in its review template:

```python
# In review-t<k>.md acceptance section, copy-paste this shape:

# Seed two distinct DB states
seed_row_A = seed(<field_under_test>=<value_A>)
seed_row_B = seed(<field_under_test>=<value_B>)

# Exercise the NEGATIVE path (field absent from body)
resp = client.post(endpoint, data={"other_field": "value"})  # field omitted
row = db.execute("SELECT <field_under_test> FROM ... WHERE id=?",
                 [seed_row_A.id]).fetchone()
assert row[0] == <expected_value_when_field_absent>, \
    "<describe what the absence semantics are>"

# Exercise the POSITIVE path (field explicitly set)
resp = client.post(endpoint, data={"<field_under_test>": "<new_value>",
                                   "__<field>_submitted": "1"})
assert db.execute(...).fetchone()[0] == <new_value>

# Access-boundary assertion if endpoint takes any ID
resp = client.post(endpoint, data={"id": seed_row_B.id})  # different access scope
assert resp.status_code == 404
```

### 4b. Defect-class checklist — every form/route task must cover

- [ ] **(a) Default-value semantics with omitted fields** — what does
      "field not in body" persist? (default-value / boolean class)
- [ ] **(b) Hidden-marker round-trip** — marker absent = preserve,
      marker present = honor checkbox
- [ ] **(c) Access-boundary access on ID-bearing endpoints** — seed access scope
      B, call as scope A, assert no leak (access-scope-leak class)
- [ ] **(d) Archived row visibility** — `is_active=FALSE` rows must
      not appear in list / must return 404 on detail
- [ ] **(e) Redirect-location assertion** — status code alone is not
      enough; assert `Location` header for 302 (catches CSRF-false-pass)

### 4c. Two-input minimum

Every functional check exercises ≥2 distinct inputs. A single input
that happens to match the expected output on a default-populated row
proves nothing. (A single-input check would have slipped even with a
functional check if the test only used an already-False row.)

---

## 5. Cross-task grep gate — mandatory pre-QA scan

After the last task merges, before QA runs, scaffolder runs this over
the whole iteration diff vs. the phase branch. Any hit = iteration
STOPS for operator review. Ten signatures (v1 had 4):

```bash
BASE=<phase-branch>; DIFF="git diff $BASE..HEAD"

# 1. Deleted imports (additions-only rule)
$DIFF -- tests/ '*.py' | grep -E '^-\s*(import |from )' && echo "FAIL 1"

# 2. Deleted test bodies
$DIFF -- tests/ | grep -E '^-\s*(async def test_|def test_)' && echo "FAIL 2"

# 3. Deleted validator errors / raised exceptions
$DIFF -- app/services/ | grep -E '^-.*errors?\.append\(|^-.*raise ' && echo "FAIL 3"

# 4. Gratuitous subquery wraps
$DIFF | grep -E 'FROM \(SELECT|SELECT COUNT\(\*\) FROM \(' && echo "FAIL 4"

# 5. Form(default=...) with non-None non-empty default on bool-ish fields
$DIFF | grep -E 'Form\(default="[^"]+"' | grep -v 'default=None' && echo "FAIL 5"

# 6. bool(form.get(...)) without an adjacent __<field>_submitted marker
$DIFF | grep -E '^\+.*bool\(.*form\.get\(|^\+.*bool\(.*Form\(default=None' \
  | while read line; do
      file=$(echo "$line" | cut -d: -f1)
      grep -q "__\w\+_submitted" "$file" || echo "FAIL 6 at $line"
    done

# 7. Removed WHERE <scope_column> clauses
$DIFF -- '*.py' '*.sql' | grep -E '^-.*WHERE.*<scope_column>' && echo "FAIL 7"

# 8. New access scope-resolution helpers (always suspicious — inspect manually)
$DIFF | grep -E '^\+.*def _?(resolve|resolve)_\w+_<scope_column>' && echo "FAIL 8 — manual review"

# 9. New early-return inside validator error accumulation
$DIFF -- app/services/validation.py | grep -E '^\+\s*(return errors|return \[)' && echo "FAIL 9"

# 10. Tests accepting 302 as PASS alongside 4xx (CSRF-false-pass class)
$DIFF -- tests/ | grep -E '^\+.*in \{302,.*4\d\d\}|^\+.*{.*302.*403|^\+.*{.*302.*404' && echo "FAIL 10"
```

If any grep prints a hit, scaffolder (not codex) decides: (a) legitimate
change → document in `tools/logs/<iter>/grep_exceptions.md` with 1-line
justification; (b) regression → operator intervention before QA runs.

---

## 6. Additions-only test discipline

Existing test files listed in a task's `Allowed files` may only be
APPENDED to. Removing a test, modifying existing assertions, or
deleting an import = automatic REVIEW_FAIL. Caught by signatures 1 + 2
above. If an existing test reveals a real defect in application code,
STOP — do not fix in the test task; report to the operator.

---

## 7. Capture round-2 rejection text on pair-swap

*(Orchestrator change; operator-only. Keeps existing v1 text.)*
When the primary pair fails two rounds and triggers a swap, the
orchestrator currently records only "REVIEW_FAIL, reason=round-2-
changes-required". Append full rejection text to `run_state.json` as a
`review_rejection` event so retros can classify reviewer-too-strict vs
implementer-wrong-shape vs prompt-ambiguous. Without this, pair-swap
remains opaque. **Track as backlog for `orch/` — out of prompt-
scaffolder scope; flag to operator when it bites.**

---

## 8. Access-scope isolation rule — every SELECT, every helper (NEW v2)

Any prompt that references a tenant-scoped table (`records`,
`transactions`, `users`, `categories`, `subscriptions`, or any
access-scoped table) MUST include verbatim:

> **Access-scope filter audit.** Every SELECT/UPDATE/DELETE in this task's
> diff — including resolution helpers, audit scripts, and migration
> queries — MUST include `WHERE <scope_column> = ?`. A query without a
> access scope filter is a CRITICAL security finding at QA regardless of
> whether it is read-only or admin-only. Grep signatures 7 and 8 above
> are mandatory pre-QA.

Reasoning: in one iteration, a resolution helper
(`_resolve_<entity>_<scope_column>`) leaked access-boundary metadata from
a read-only audit tool. Read-only doesn't make a leak
acceptable. Every layer of defense-in-depth matters — if the caller's
<scope_column> isn't threaded through every query, a future caller-
convenience helper leaks without warning.

---

## 9. Prompt code examples are executable rules (NEW v2)

Every code snippet in a prompt is treated by the implementer as
definitive. If the snippet shows `return errors` after the first
error, the implementer will short-circuit (a task once authored this
bug in its own example). Before handoff, scaffolder greps each prompt's code
blocks for:

- `return errors` / `return None` / `return []` / `raise ` inside
  error-accumulation examples — must match the existing file's pattern
  or explicitly document the change in `Preserved behavior`
- `SELECT .* FROM <table>` without `WHERE <scope_column> = ?` in the same
  snippet — shown example is an access-scope leak
- `Form(default="1")` / `Form(default="")` — shown example authors the
  checkbox-semantics bug
- Hardcoded magic strings (`"automatic"`, `"active"`) that differ from
  the codebase's canonical constants

The prompt example MUST be copy-pasteable into the codebase without
violating any other rule in this doc.

---

## 10. Boolean-field mandatory tests (NEW v2)

For every checkbox, toggle, or nullable boolean touched by a task, the
acceptance checklist MUST include these three functional scenarios (not
two — two related boolean-field defects together proved that):

- [ ] **Absent from POST body** → assert preserved (edit) or default
      (create). Uses zero form fields.
- [ ] **Explicitly unchecked with marker present** → assert False.
- [ ] **Explicitly checked** → assert True.

Grep-based checks do not satisfy this rule. AST-based "field exists"
checks do not satisfy this rule. The TestClient-round-trip is the
minimum evidence.

---

## 11. Final step — commit and push (NEW v2.1, 2026-04-30)

Codex implementations frequently leave edits in the worktree without
running `git add && git commit`. This pattern recurred across several
early iterations — the operator had to manually commit on the
implementer's behalf multiple times. The pattern is
prompt-shape, not codex-bug: prompts that title the closing section
"Commit body convention" or "Acceptance" describe the commit message as
*documentation*, not as the *next action*. Codex treats the suggested
message as a static artifact rather than an instruction.

**Mandatory closing section in every `prompts/t<k>-*.md`:**

```markdown
## Final step (commit + push) — RUN, do not just document

Once the regression test passes and the diff matches the allowed-files
list, run:

\`\`\`bash
git add <each-allowed-file-explicitly>
git status --short          # confirm clean except staged
git commit -m "<title from below>" -m "<body from below>"
git push -u origin <branch>
\`\`\`

**Commit title:** `<iteration>-T<k>: <slug>`

**Commit body:**

\`\`\`
<short why; reference the synthesis item or memory artifact this closes;
mention regression test name>
\`\`\`

If the commit fails (pre-commit hook, signing, etc.), fix the underlying
issue and create a NEW commit. Do not amend. Do not skip the push.
```

The headings `## Final step (commit + push) — RUN, do not just document`
and the imperative "Once the regression test passes... run:" matter. The
old wording (`## Commit body convention` followed by a fenced block)
parses as documentation in codex's planning. The new wording parses as
the next executable step.

**Why this is a rule and not a template note:** prompt templates rot
silently. A rule with a Self-check entry forces the scaffolder to verify
the closing section exists at the right shape on every iteration.

---

## 12. Orchestrator state-machine recovery is an authorized scope exception (NEW v2.2, 2026-05-03)

When orch's automation cannot close out a task and the operator + safety
sandbox have explicitly authorized a manual recovery (e.g. flipping
`NEEDS_HUMAN_MERGE` → `DONE` in `run_state.json` after a manually-merged
PR), modifying `iterations/<phase>/<iter>/tasks.md`'s `Status:` field IS
in scope for that recovery, even if the iteration's own T4-style
"`tasks.md` is orchestrator-managed" clause says otherwise. Document
each manual flip as its own commit with the form:

```
chore: mark <T> as DONE after manual <PR> merge
```

This rule was extracted from an iteration retrospective: several tasks
all stalled in `NEEDS_HUMAN_MERGE` because the auto-merge wait timed
out. Recovery required hand-editing `run_state.json` AND committing the
status flip in `tasks.md` so the next `orch resume` saw a consistent
view. The Process reviewer flagged this as a scope-hygiene smell. The
correct framing is: this is bookkeeping to repair orch's state machine,
not implementer behaviour, and it should be visible in the audit trail.

The long-term fix is to auto-detect external PR merges during `orch
resume` (planned). Once that
ships, manual flips should disappear except for genuinely irreversible
states; this rule remains as the documented exception, not the
preferred path.

---

## 13. Iteration branch freshness is a pre-task invariant (NEW v2.3, 2026-05-06)

Before any task prompt is handed to an implementer, the scaffolder must
prove that the iteration branch tracks the current phase branch head.
Run this in the repo that will execute the task:

```bash
git fetch origin
git rev-parse origin/<phase-branch>
git merge-base origin/<phase-branch> <iteration-branch>
```

The two SHAs must match. If they do not, STOP: reset/recreate the
iteration branch from `origin/<phase-branch>` before T1 starts, or
explicitly document the stale-base risk in the iteration runbook and
review template.

Reasoning: in one iteration a stale branch missed a prior PR's CSP
cleanup until codex ran the inverse diff during cross-family QA. A
perfect task prompt against a stale branch can still ship a regression.

---

## 14. New pytest markers / config taxonomy require allowlisted config (NEW v2.3, 2026-05-06)

Any prompt that introduces a new pytest marker, pytest option, lint
profile, CI selector, or other project-level test taxonomy MUST also
allowlist the config file that registers it (`pyproject.toml`,
`pytest.ini`, or the repo's existing equivalent). If the task itself
must stay test-only, the next closer/polish task must explicitly include
the config file in its allowed-files list and acceptance check.

Example: a prompt that adds `@pytest.mark.performance` must include:

```toml
[tool.pytest.ini_options]
markers = [
    "performance: postgres-only performance smoke tests",
]
```

Unregistered markers are not harmless. They emit warnings, make
`pytest -m "not performance"` unreliable, and force a QA-fix loop.

---

## 15. Verification right-sizing per task type (NEW v2.4, 2026-06-10)

Per-task prompts MUST right-size verification to the task type.
Full-suite demands inside per-task prompts are forbidden: the operator
runs the full suite once at iteration close and again at phase-merge
gates. Any prompt that instructs a pytest suite run MUST pin
`pytest -n auto`; do not write plain `pytest` for a suite command.

| Task type | Per-task verification | Not allowed per task |
|---|---|---|
| Test-only / disjoint-file task | The task's own new or changed test file, e.g. `pytest -n auto tests/<file> -q`, plus `ruff` on that file. | Full suite. Reverse-dependency sweeps. |
| Code-changing task | Own/touched test files plus the characterization smoke suite, e.g. `pytest -n auto tests/test_tier2_*_characterization.py -q`, plus `ruff` on changed Python files. | Full suite. Untimed broad "just in case" runs. |
| Iteration close / phase merge | Full suite once, always with `pytest -n auto`. | Requiring every task prompt to prove full-suite green. |

The purpose is right-sized confidence, not weaker verification. Wall-time
saving is a measured-later bonus; do not claim a percentage until the
change has been timed. The suite was already fast under `pytest -n
auto`, so prompts must not assume an unmeasured speedup.

Interaction note: once no-CI local merge ships, task branches fork
from the updated iteration branch. Each task's smoke run therefore sees
prior tasks' merged code, and that smoke run is the cross-task safety net.
Do not add manual per-task full-suite gates to compensate.

---

## Self-check before handing prompt to orchestrator

Before `python -m orch validate <iter>`, the scaffolding author
runs through this list. **Every box must be checkable honestly — no
"mostly," no "close enough."**

### Structure
- [ ] Iteration branch freshness proved: `git merge-base
      origin/<phase-branch> <iteration-branch>` equals
      `origin/<phase-branch>` HEAD before T1 starts (Rule 13)
- [ ] Every `if X` has an explicit `if not X` branch (Rule 1)
- [ ] Preserved-behavior section lists ≥3 existing rules with
      file:line references (Rule 2a)
- [ ] Behavior-regression fixture file committed on `<phase-branch>` HEAD
      BEFORE iteration starts (Rule 2b)
- [ ] Sweep-scope clause present verbatim in every schema/query/render
      task (Rule 3)
- [ ] Allowed-files list is minimal — **no "only if"** escape hatches
      (Rule 3 antipatterns)

### Acceptance checks
- [ ] Functional-check template applied per form/route/validator task
      (Rule 4a)
- [ ] All 5 defect classes (a-e) covered by at least one check (Rule 4b)
- [ ] Every functional check uses ≥2 distinct inputs (Rule 4c)
- [ ] Cross-task grep-gate script committed at
      `orch/grep_gates/<iter>.sh` or equivalent (Rule 5)
- [ ] Any new pytest marker / project-level test selector has its
      registration config allowlisted and included in acceptance
      checks (Rule 14)
- [ ] Per-task verification scope matches Rule 15's table; `-n auto`
      is pinned wherever a suite run is instructed (Rule 15)

### Security & invariants
- [ ] Access-scope filter audit clause present in any task touching
      access-scope scoped tables (Rule 8)
- [ ] Prompt code examples grep-checked for access scope leaks, bool-default
      traps, early-returns (Rule 9)
- [ ] Every boolean form field has the 3-scenario test matrix (Rule 10)
- [ ] Hidden-marker pattern used for every checkbox/toggle (Rule 1)

### Prompt hygiene
- [ ] No "Grep for X and extend every match" language (Rule 3)
- [ ] No specific symbol named in a sweep list without author
      confirming the sweep applies (Rule 3)
- [ ] `Read first: _prompt_rules.md` in both `prompt.md`
      and every `prompts/t<k>-*.md`
- [ ] `Allowed files` lists are minimal, explicit, no conditionals
- [ ] `NOT allowed` lists name specific files other tasks own

### Closing section (Rule 11)
- [ ] Each `prompts/t<k>-*.md` ends with `## Final step (commit + push)
      — RUN, do not just document` containing an explicit
      `git add && git commit && git push` block, not a
      `## Commit body convention` documentation block

---

## Tracking & retrospective

Target: **≤2 cross-cutting defects at QA per iteration**. The v1 baseline
was an iteration that shipped a high double-digit count. If an
iteration ships with 3+, the
bottleneck is one of:

- **Scaffolder skipped the self-check** → process failure, not a rules
  failure
- **A defect class v2 doesn't cover** → update rules post-retro, don't
  blame the scaffolder
- **Pair-swap obscured a real prompt issue** → Rule 7 backlog item
  (orchestrator change) becomes urgent

The retro MUST classify each cross-cutting defect into one of
these three buckets — not "per-task review weak again" (v1's conclusion
five times over). Root-cause analysis lands on the specific rule that
failed, not the abstract category.
