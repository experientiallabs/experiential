---
source: https://app.notion.com/38e0f8b3f59181da8220fa5bf10f612b
area: Evaluation
status: Current
migrated: 2026-07-02
---

# Iterating on BASE_ENV_PROMPT with replay fidelity

`BASE_ENV_PROMPT` (`wmh/engine/prompts.py`) is layer (a): the env-agnostic prompt GEPA evolves from.
We want it both **general** (works across domains) and a **strong GEPA starting surface** (high
zero-/few-shot reconstruction fidelity before any evolution). We tune it by measuring, not guessing.

## The measurement loop

`wmh eval` (engine: `wmh/engine/eval.py` → `wmh/engine/replay.py`) replays each trace file's held-out
split through a prompt (teacher-forced, open-loop) and scores predicted vs. real observations. The
default `--judge rubric` is a reference-grounded 5-dimension scorer (format / factuality /
consistency / realism / quality, mean → 0..1) modeled on Qwen-AgentWorld, with a
deterministic-vs-volatile content split (computed outputs must match; PIDs/timestamps judged on
plausibility); `--judge match` is the simpler functional-equivalence `LLMJudge`. It reports per-file
and overall fidelity as mean±std across steps. `--sample-turns sampled` scores 5 turns/trace (Qwen's
protocol) instead of all. Run:

```bash
AWS_REGION=us-east-1 uv run wmh eval \
  examples/tau-bench/traces.otel.jsonl examples/terminal-tasks/traces.otel.jsonl \
  --out report.json
```

`wmh eval` takes one or more trace files — or use the committed per-example suites: `wmh eval run
tau-bench` (see the *Eval suites* doc). The committed corpora are
`examples/{tau-bench,terminal-tasks,swe-bench}/traces.otel.jsonl` (1033 / 280 / 87 traces); new
benchmarks are added via the self-contained `examples/<task>/` capture pattern (see *Benchmarks →
traces: the real trace source*).

Inspect the lowest-scoring steps' critiques (and per-dimension scores) to find systematic failure
modes, change the prompt, re-run on the same split+seed, and keep changes that move fidelity without
overfitting to one benchmark. This is open-loop; closed-loop task-success is a future direction
(see *Closed-loop evaluation*).

## What the first baseline taught us

Initial overall fidelity was ~0.43, but inspecting results surfaced two distinct issues:

1. **A data artifact, not a model failure.** Half the held-out steps had *empty* ground truth —
   they were the agent's final submit/answer turn, which has no environment reply. Scoring against a
   non-existent observation dragged the number down. Fix: trace ingestion only emits a step for an
   agent turn that HAS a following environment reply (unpaired trailing turns are dropped). On steps
   with a real observation, fidelity was already ~0.62.
2. **Real, fixable model failure modes** (drove the prompt rewrite):
   - *Fabricating concrete data* the environment alone knows (e.g. inventing SQL result rows instead
     of the DB's actual contents).
   - *Inventing stdout* when the real command prints nothing (assignments, writes, redirects).
   - *Guessing success vs. error wrong* (heredoc syntax errors predicted as success, and vice-versa).

## The resulting base prompt

The rewritten `BASE_ENV_PROMPT` targets those three failure modes while staying domain-agnostic, and
adds explicit **stay-in-character** guidance for edge cases: a terminal env must answer a stray "hi"
with `command not found` (not a chat reply); an API env must answer an unknown call with that API's
own error shape. It instructs the model to ground values in STATE/HISTORY (never fabricate), to
output only the bytes that actually reach the agent (empty when nothing prints), and to decide
success/error from what the action would really do.

## Iteration: "a lookup of a record that exists returns populated data"

Measured base-only (no GEPA) on both committed corpora (tau2-bench + terminal-tasks, 132 held-out
steps, `--train-split 0.7 --seed 0`, rubric judge, Bedrock Opus 4.8):

| base prompt | tau2-bench | terminal-tasks | overall |
| --- | --- | --- | --- |
| before this edit | 0.760 | 0.592 | 0.699 |
| **after** | **0.805** | 0.582 | **0.724** |

The dominant base failure on tau2 was catastrophic and uniform: for `get_reservation_details` on a
reservation that **exists**, the empty `state_before` led the model to default to *"reservation not
found"* — flipping the outcome to error (format/factuality/consistency all 0). The edit adds general
guidance: a lookup/read of something the task implies EXISTS returns the full populated, schema-
correct record, not a "not found" error. That lifted tau2 +0.045 and overall +0.025, with terminal
flat within the ≈±0.02 cross-run judge noise.

A more defensive variant that also warned "execution can still fail, don't assume success" was
tested and **rejected** — it diluted the tau2 gain (0.805 → 0.778) without improving terminal. This
is taste-driven hand-tuning: keep an edit only when it measurably helps across both, and the better
base is the *starting point* GEPA specializes from, not a substitute for GEPA.

## Notes

- This is deliberately NOT GEPA-on-the-base: we keep the base general and hand-tuned, then let GEPA
  specialize per-project from this stronger starting point.
- The fidelity numbers in this doc were measured on the June 2026 corpora, before the corpus growth
  to 1033 tau / 280 terminal / 87 swe traces; the methodology is unchanged but rerun the commands
  above for current numbers. Note the trace scaling law result (see *Trace scaling law*): fidelity
  saturates with a handful of training traces, so corpus growth barely moves these numbers — prompt
  quality is the leverage, which is exactly what this doc's loop tunes.
