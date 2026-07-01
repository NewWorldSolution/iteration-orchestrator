# iteration-orchestrator

> A deterministic control plane that drives LLM coding agents through a governed
> iteration — **implement → deterministic gates → multi-reviewer QA → retrospective** —
> with fail-closed checks and cross-model review. The control logic is plain,
> testable Python; the LLMs are swappable workers at the edges.

<!-- Optional: add a CI badge once GitHub Actions is set up. -->

## What it is

`orch` takes a **pre-planned iteration** (a strict `tasks.md` + per-task prompts)
and runs it end to end: it invokes an implementer agent, enforces deterministic
gates on the diff, runs a reviewer agent (of a *different* model family), performs
a guarded merge, then runs a multi-reviewer QA pass and a structured retrospective —
stopping with a single typed reason and a resume command whenever a human decision
is genuinely needed.

It was built as internal tooling and dogfooded on a real production SaaS, then
generalized. **Treat it as a reference implementation of governed agentic SDLC —
not a turnkey product.**

## Why it's interesting

- **No LLM in the control logic.** Gates, scope checks, merges, state, and cost
  accounting are deterministic Python. LLMs are confined to worker adapters. This
  is the core design bet: *verification, not generation, is the bottleneck.*
- **Cross-model review as a hard gate.** The implementer and reviewer must be from
  different model families; high-risk tasks additionally require a third-family
  reviewer. A single-vendor agent structurally cannot make this guarantee.
- **Fail-closed everything.** Ambiguous review output, a missing base ref, a stale
  branch, an out-of-scope edit → the run stops with a typed reason, never a guess.
- **Event-sourced state + crash recovery.** State is an append-only event log; the
  snapshot is a derived optimization. Runs resume, retry, and revert cleanly.

## Status: working vs experimental

Honest labeling — some surface is real-but-experimental (off by default):

| Area | Status |
|---|---|
| Iteration runner: implement → checks → review → guarded merge | ✅ Working |
| Deterministic gates (scope, diff-size, forbidden patterns, final scope/nav) | ✅ Working |
| Multi-reviewer QA (5 roles) + retrospective (3 perspectives) | ✅ Working |
| Cross-family / dual-model review independence | ✅ Working |
| Event-sourced state + crash recovery (`resume` / `retry` / `revert`) | ✅ Working |
| Deterministic model routing with risk floors | ✅ Working |
| Cost logging | ✅ Working |
| Prompt Factory (spec → reviewed task package) | 🧪 Experimental — one pilot |
| Parallel execution (worktree-per-task) | 🧪 Experimental — off by default |
| Agent teams (read-only QA/retro, docs-only planning) | 🧪 Experimental — off by default |

## Install

```bash
pip install -e ".[dev]"     # from a clone
python -m orch --help
```

Requires Python 3.11+, and the CLI(s) you want as workers — e.g. the **Claude Code**
and/or **Codex** CLIs — installed and logged in. `orch` shells out to them; it does
not store or require an API key of its own.

## Connecting your agent CLIs

`orch` is **bring-your-own-CLI**: it shells out to the agent CLIs you already have
installed and logged in. It never sees, stores, or asks for your credentials — your
subscription session lives in each vendor's CLI, not in `orch`. There is no login
screen and no auto-connect step.

**One-time setup per machine:**

1. **Install the CLI(s)** you want as workers — e.g. Claude Code (`claude`) and/or
   Codex (`codex`).
2. **Log in to each CLI** — this is where your subscription connects, inside the
   vendor's CLI (not in `orch`): `claude` logs in against your Claude subscription,
   `codex` against your ChatGPT / OpenAI account. The session persists in the CLI's
   own config (`~/.claude`, `~/.codex`, …).
3. **Point your project pack at those commands** in `.orch/project.yaml`:
   ```yaml
   agents:
     claude: { type: claude, cmd: "claude -p",  family: anthropic }
     codex:  { type: codex,  cmd: "codex exec", family: openai }
   ```
4. **Run** — `orch` spawns those CLIs as subprocesses, reusing your logged-in session.

If a CLI isn't logged in, the run stops with an auth error rather than prompting you —
run `orch doctor` to check your configured CLIs are installed and logged in before
a run.

> **Heads-up:** subscription CLIs have **usage / rate limits** (a full iteration —
> implement → multi-reviewer QA → retro — makes many calls), and some vendors treat
> automated use differently from interactive use in their terms. Check your plan's
> limits and terms. `orch` is agnostic and also works with an API-key-authenticated CLI.

## Quickstart

1. Add a **project pack** to your target repo describing its conventions. Start
   from [`examples/minimal/project.yaml`](examples/minimal/project.yaml) (zero-domain),
   or see [`examples/financial-saas/`](examples/financial-saas/) for a full worked
   example — the orchestrator configured for a multi-tenant financial SaaS. A pack
   declares branches, allowed-file patterns, risk globs, model routing, project
   invariants, and safe merge defaults (`no_ci: false`, a real `test:` command).
2. Author an **iteration** from the templates in [`templates/`](templates/): a
   strict `tasks.md`, one prompt per task, one review per task.
3. Run it:
   ```bash
   python -m orch validate <iteration-id>                         # config + scaffold check
   python -m orch iteration <iteration-id> \                      # run → qa → retro
       --implementer codex --reviewer claude
   ```
   The implementer and reviewer must be different model families. On any real
   fork the run stops with a typed reason and a resume command.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the control loop, state model, gates,
and how to plug in your own agent adapter.

## Honest limitations

- **It needs a project pack.** There is no zero-config mode; you describe your repo.
- **The autonomous "operator" is intentionally NOT included.** A human still drives
  stops. Unattended failure-handling is a deliberate non-goal of this release — the
  tool is honest about where human judgment belongs.
- **You bring your own agent CLI + subscription.** Costs and rate limits are the
  vendor's; the built-in cost meter is approximate.
- `runner.py` and `cli.py` are large modules (a known, tracked refactor).
