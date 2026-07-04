# Sim-real validation: does world-model evolution transfer to reality?

The auto-harness (`wmh.agent`) evolves harness variants by scoring them **against the world model**.
That signal is only worth optimizing if a variant that wins in simulation also wins for real. This is
the feature's load-bearing assumption, and the DGM cautionary tale (an evolving agent that fabricated
success once the objective could be gamed) says to assume it will be violated until measured. This doc
is the plan — and the tooling — for measuring it.

## The hypothesis

> **Harness variants ranked by closed-loop success against the world model are ranked the same way by
> real E2B execution.** Formally: high rank correlation between per-variant sim success and real
> success, and high per-(variant, task) outcome agreement.

Falsifiable, with a headline number. If it holds, sim-driven evolution is trustworthy. If it fails,
*that is the finding* — it quantifies how much the world model must improve (more/better traces)
before evolution can be believed, and which cells it gets wrong.

## The metrics (`wmh.agent.agreement`)

Score each variant in both worlds (k passes/task each; k=3 per the repo convention), then:

- **Outcome agreement** — binarize each (variant, task) cell to pass/fail at `--threshold` (default
  0.5 of k passes), and report the fraction of cells where sim and real agree, plus the 2×2 confusion.
  The cell to watch is **sim-pass & real-FAIL**: the mirage evolution chases.
- **Rank correlation** — Spearman over per-variant aggregate success (sim vs real), average-rank, no
  scipy. This is the number that predicts transfer: evolution *ranks* variants and keeps the top one,
  so rank agreement matters more than absolute-score agreement.
- **Mean absolute gap** — mean |sim_success − real_success| across variants; a calibration read.

`compute_agreement` is pure and unit-tested (`agreement_test.py`); `sim_real_agreement` runs the
expensive both-worlds scoring.

## Guardrails baked in

- **Oracle-gated tasks** (`wmh agent gate`): a task no strong agent can solve — or whose gold never
  fires on a genuine success — is a broken eval that poisons every score. The gate runs the baseline
  agent for real in E2B and flags any task below a real-success threshold, mirroring Terminal-Bench's
  reference-solution admission test. Only admitted tasks enter the study.
- **The real leg is an independent check the harness can't game.** Success is judged against held gold
  by a judge that runs outside the agent's sandbox; the DGM failure mode (self-reported success) is
  exactly what the real-E2B leg catches.
- **k=3, deterministic** everywhere (repo conventions).

## Runbook (the creds/budget-gated real run)

Needs `uv sync --extra e2b`, `$E2B_API_KEY`, and a frontier provider (Bedrock/Anthropic creds).

```bash
# 0. Gate the suite: drop/fix any task the strong baseline can't solve for real.
wmh agent gate examples/agent_tasks.jsonl --k 3

# 1. Collect real traces with the (gated) baseline harness.
wmh agent collect examples/agent_tasks.jsonl --out .wmh/traces/shell.otel.jsonl

# 2. Build a world model from the agent's own behavior.
wmh build --name shell --file .wmh/traces/shell.otel.jsonl

# 3. Evolve harness variants against the world model; keep the whole archive.
wmh agent evolve examples/agent_tasks.jsonl --name shell --generations 8 \
    --archive .wmh/archive.json --out .wmh/best.json

# 4. THE VALIDATION: score every archived variant in the world model AND real E2B, report agreement.
wmh agent verify examples/agent_tasks.jsonl --name shell --archive .wmh/archive.json --k 3
```

## Reading the verdict

| Signal | Interpretation | Action |
|---|---|---|
| High rank corr (≳0.7), high outcome agreement | Sim-driven evolution transfers. | Trust the loop; scale the search (MAP-Elites, merge). |
| High outcome agreement, low rank corr | Sim gets pass/fail right but mis-orders close variants. | Evolution still helps but ties are noisy; raise k, add tasks. |
| Many **sim-pass & real-FAIL** cells | Sim over-credits — evolution is chasing mirages. | The world model needs more/broader traces; re-collect before trusting evolve. |
| Best-by-sim also best-by-real | The one thing evolution must get right, it did. | Ship the evolved harness. |

The single most important thing to report is whether **the evolve winner (best by sim) also beats the
seed in reality**. That is the feature working or not, in one comparison.

## Status

Tooling landed and unit-tested offline (`agreement.py`, `real_loop.py`, `wmh agent verify`, `wmh agent
gate`, 15-task gated suite). The real run (steps 0–4) is pending creds/budget sign-off — it spends
frontier-model and live-sandbox time — and should be registered in the shared experiments tracker
before kicking off.
