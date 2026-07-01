# Prompt Template Family

These templates are the adopted canonical scaffolding for iteration prompts
(adopted 2026-06-10; mandated in `CLAUDE.md` → "Preparing an Iteration").
They do not replace `iterations/_prompt_rules.md` or
`iterations/_review_template.md` — they embed and complement them, and those
remain the rule source. The orchestrator runtime prompts in `orch/`
consume the same contracts (see Adoption Checklist below).

Always start a new iteration's prompt/review/QA/retro files from the matching
template here. Do not copy historical prompts — older files reflect superseded
prompt-rule versions.

## Template Selection

Start with `base_contract.md`, then choose one task template:

| Work type | Task template |
|---|---|
| Runtime product/application change | `task_runtime_product.md` |
| Database schema or migration work | `task_db_migration.md` |
| Auth, access scope isolation, secrets, anti-enumeration | `task_security_auth.md` |
| Server-rendered template or lightweight JS UI work | `task_ui_jinja_js.md` |
| Docs-only change | `task_docs_only.md` |
| Tests-only characterization or regression net | `task_test_only.md` |
| Orchestrator/tooling work | `task_tooling_orch.md` |
| Read-only audit or planning work | `task_audit_planning_only.md` |

Then create matching gates:

| Gate | Template |
|---|---|
| Per-task review | `review_task.md` |
| Iteration QA | `qa_iteration.md` |
| Retrospective | `retro_iteration.md` |

## Composition Rule

Task templates are self-contained. Every generated task prompt must contain
all base-contract protections inline; never generate a prompt from a task
variant assuming base-contract sections are implied.

`iterations/templates/base_contract.md` is the normative reference. When a
variant and the base contract disagree, the base contract wins, and the
variant must be updated for parity.

A generated prompt missing any of these sections is invalid and must not be
handed to an implementer:

- Stop Conditions
- read-order conflict-STOP line
- Deviation rule
- Preserved Behavior or N/A
- invariant classification table
- Final Action block

## Authoring Rules

- Keep placeholders visible until intentionally filled.
- Delete sections only when they are genuinely not applicable and state why.
- Do not weaken project invariants for convenience.
- Do not copy old "first line" verdict wording. Use the trailing verdict
  blocks from `review_task.md`.
- For runtime work, every requirement must map to evidence in an acceptance
  matrix.
- For Prompt Factory materialization, docs-only tasks may use compact prompts;
  runtime/product tasks should use the full relevant template.

## Adoption Checklist

Status: **adopted** (2026-06-10). Runtime binding — the original adoption
blocker — is complete; the runtime builders consume the same contracts:

- Done: `orch/runner.py` loads authored review prompts when an iteration
  scaffolds reviews, fails closed when that scaffold is incomplete, and falls
  back to an embedded verdict contract for legacy reviews-less iterations.
- Done: `orch/qa.py` and `orch/retro.py` runtime prompts use the
  same contracts.
- Done: `orch/prompt_factory.py` materializes the full task/review
  contracts (Gate 6, Calibration, execution-mode metadata, escaped cells).

Remaining hardening (tracked, not blocking use):

- Scaffold lint to mechanically check the required sections. Until it lands,
  section completeness is enforced by per-task review + iteration QA, not by a
  gate.
- Richer Prompt Factory JSON schema before materializing runtime/product
  tasks — treat the first such use as a pilot.
