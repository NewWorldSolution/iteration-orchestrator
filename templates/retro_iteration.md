# Retrospective - <ITERATION_ID> - <Perspective>

## Retro Metadata

- Perspective: `<Developer | Product Owner | Scrum Master>`
- Iteration branch: `<branch>`
- QA report: `tools/logs/<iter>/qa_report.md`
- Run state: `tools/logs/<iter>/run_state.json`
- Timing evidence: `<present/missing/operator notes>`

## Required Inputs

1. `CLAUDE.md`
2. `<iteration>/prompt.md`
3. `<iteration>/tasks.md`
4. `<iteration>/prompts/*.md`
5. `<iteration>/reviews/*.md`
6. `tools/logs/<iter>/run_state.json`
7. `tools/logs/<iter>/cost.jsonl`
8. `tools/logs/<iter>/qa_report.md`
9. previous retrospectives relevant to the pattern

## Output Contract

Use this section order:

1. `## Outcome`
2. `## What Worked`
3. `## What Failed or Was Expensive`
4. `## Root-Cause Classification`
5. `## Prompt Quality Assessment`
6. `## Improvement Records`
7. `## Carry-Forward Recommendations`
8. final line: `Retro Verdict: COMPLETE | COMPLETE_WITH_FOLLOWUPS | INCOMPLETE`

Use stable record IDs so retro conclusions can reference QA findings and feed
the next planning prompt:

- Developer: `RETRO-D1`, `RETRO-D2`, ...
- Product Owner: `RETRO-PO1`, `RETRO-PO2`, ...
- Scrum Master: `RETRO-SM1`, `RETRO-SM2`, ...

## Outcome

- Built:
- Not built:
- QA decision:
- Merge/readiness recommendation:
- Wall time:
- Estimated dollar cost:

## What Worked

- `<specific evidence-backed point>`

## What Failed or Was Expensive

- `<specific evidence-backed point>`

## Root-Cause Taxonomy

Classify every material issue with one primary cause:

- `prompt_ambiguity`
- `prompt_missing_invariant`
- `review_template_gap`
- `review_calibration_too_strict`
- `review_calibration_too_weak`
- `qa_template_gap`
- `deterministic_gate_missing`
- `model_fit_or_pairing`
- `operator_decision_gap`
- `environment_or_ci`
- `implementation_error`
- `scope_or_branch_hygiene`

## Root-Cause Classification

| ID | Issue | Evidence | Primary cause | Secondary cause | Preventive control |
|---|---|---|---|---|---|
| `<RETRO-*>` | `<issue>` | `<event/QA finding/file>` | `<taxonomy>` | `<optional>` | `<control>` |

## Prompt Quality Assessment

For each task:

| ID | Task | Signal | Root cause | Template change needed? | Exact rewrite |
|---|---|---|---|---|---|
| `<RETRO-*>` | `<task>` | `<event/review/QA finding>` | `<taxonomy>` | `<yes/no>` | `<pasteable text>` |

For each review prompt:

| ID | Review | Calibration | Missed? | Too strict? | Change |
|---|---|---|---|---|---|
| `<RETRO-*>` | `<review>` | `<good/weak/strict>` | `<yes/no>` | `<yes/no>` | `<change>` |

For QA:

| ID | QA finding | Should have been caught earlier? | Missing gate |
|---|---|---|---|
| `<RETRO-*>` | `<QA-ID>` | `<yes/no>` | `<gate>` |

## Improvement Records

Each proposed improvement must be backlog-ready.

| ID | Action | Owner | Priority | Acceptance check | Control mechanism |
|---|---|---|---|---|---|
| `<id>` | `<action>` | `<owner>` | `<priority>` | `<check>` | `<control>` |

Do not mark an improvement approved or implemented unless operator approval
and `control_mechanism` exist.

## Carry-Forward Recommendations

- Prompt-rule updates:
- Review-template updates:
- QA-template updates:
- Retro-template updates:
- Orchestrator deterministic gates:
- Model routing or pairing changes:
- Product follow-ups:

## Final Retro Verdict

Use:

- `COMPLETE` when no required follow-up remains for the iteration itself.
- `COMPLETE_WITH_FOLLOWUPS` when work is shippable but follow-up records must
  carry forward.
- `INCOMPLETE` when QA/evidence is missing or the retro could not classify
  material issues.
