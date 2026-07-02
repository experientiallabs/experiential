# .agents — the agents' workspace

The unclean side of the work: one-off scripts, experiment runners, plans, scratchpads, drafts.
Committed so it transfers across worktrees and chats — but that is the only guarantee.

Contract (AGENTS.md rule 5):
- **Exempt from the gate.** ruff/ty/pytest do not run here; nothing here is reviewed to
  production standards.
- **Nothing may depend on it.** `wmh/`, `examples/`, `docs/`, and `web/` must never import from
  or link to `.agents/` as if it were permanent.
- **Pruned periodically.** Anything here may be deleted at any time. If work matures, promote
  the product out — report → `docs/`, reusable code → `wmh/`, dataset tooling →
  `examples/<task>/` — and let the scraps die here.

Layout is free-form; `scripts/` for runnable one-offs is the only suggested convention.
