# Verbalized confidence: the world model knows what it can't know

*WS-A6, 2026-07-06. The lever: an opt-in `confidence` field in the world-model output contract —
a 0.0–1.0 (one decimal) self-assessment of the emitted observation, decoded after `output`/
`is_error` so it conditions on the answer actually given. This report measures whether that
number is real (calibration, across three serving models), what it means (a calibrated
P(good step)), what it buys (selective fidelity), and what it saves (confidence-gated verify
and confidence-gated model escalation). PR #120, stacked on #55; decisions D75–D77.*

## Protocol

The fixed measurement protocol (D12): splits by `trace_id` hash (test 0.2 / valid 0.15), test
caps tau 40 / terminal 40 / swe 20 traces, Qwen-style 5-turn sampling, seeds 0+1 pooled, serve
`us.anthropic.claude-opus-4-7`, judge pinned `us.anthropic.claude-opus-4-8` (RubricJudge),
swe-bench on the healthy corpus (`--drop-degenerate`). n_train: tau 200 / terminal 160 / swe 24.
"Good step" = judge fidelity ≥ 0.8. The judge never sees the stated confidence, and GEPA cannot
optimize with the field on (both regression-tested — a gameable confidence would invalidate all
of this).

The measurement contract is entirely in `wmh.research` and this section: any runner that feeds
`TraceScalingAblation` the composable modes below under the protocol above yields comparable
cells. One such runner existed at `.agents/scripts/run_trace_scaling.py` when this was measured
(`.agents/` is disposable; git history at this report's commit preserves it):

```
AWS_REGION=us-east-1 uv run python .agents/scripts/run_trace_scaling.py tau-bench \
  --counts 200 --modes base+conf,reason+conf,reason+confwhy,reason+gateverify@0.6 \
  --seeds 0,1 --test-cap 40 --results-dir <dir> --out <report.json>
```

## Finding 1 — the confidence is real, and it is UNDERconfident

Every mode states a confidence on ~100% of steps (no contract erosion), and the lever is
fidelity-neutral: adding `+conf` moves fidelity by ≤0.008 on terminal/swe and **+0.013 on tau**
(the field acts as a micro-deliberation on tool-call APIs). Against the pinned judge:

| suite × mode | fidelity | mean conf | AUROC(good) | Spearman ρ | ECE | conf − fid |
|---|---|---|---|---|---|---|
| tau `base+conf` | .914 | .74 | **.950** | .72 | .155 | **−.18** |
| tau `reason+conf` | .916 | .77 | .973 | .72 | .126 | −.14 |
| terminal `base+conf` | .867 | .73 | .877 | .73 | .090 | −.14 |
| terminal `reason+conf` | .883 | .73 | .840 | .69 | .040 | −.15 |
| swe `base+conf` | .795 | .64 | .845 | .70 | .091 | −.15 |
| swe `reason+conf` | .801 | .53 | .849 | .71 | .117 | **−.27** |
| swe `reason+workspace+conf` | .843 | .64 | **.881** | .78 | .102 | −.21 |

Three results against the literature's priors:

- **Discrimination is strong everywhere** (AUROC .84–.98). The reliability curves
  (below) are monotone: a step stated at 0.3 really is ~3× more likely to be wrong than one at
  0.9.
- **Against the continuous judge score the model looks UNDERconfident** — mean stated
  confidence sits 0.11–0.27 below mean fidelity, the opposite of the classic
  LLM-overconfidence prior. The mechanism section below resolves what the number actually
  means.
- **The model knows what it can't know.** Its lowest confidences land precisely on the
  irreducible-uncertainty populations: live-web curl bodies on terminal, unknowable held-out
  records on tau, unseen repo content on swe. And ground-truth injection raises confidence
  exactly there: swe reason .53 → workspace .64; terminal reason .73 → fetch .78 — with AUROC
  *improving* (workspace+conf .881, the best on swe), i.e. the confidence tracks the evidence
  it was given.

**Justification helps.** The `+confwhy` variant (one-line justification decoded before the
number) improves ECE on every suite (terminal reason .040 → .023, tau base .155 → .116,
swe reason .117 → .082) at ~2k extra output tokens per 100 steps — worth it wherever the
confidence will be consumed downstream.

(Reliability diagrams are monotone on every suite/mode — the marker-mass curves sit above the
diagonal throughout; vector sources shipped with the raw results.)

### What the number actually means: a calibrated P(good step)

Testing the stated confidence against different targets (pooled `+conf` cells, seeds 0+1)
settles its implicit semantics:

| target | tau: conf − rate / ECE | terminal | swe |
|---|---|---|---|
| judge ≥ 0.8 ("good") | **−.06 / .12** | **−.005 / .02** | **−.02 / .05** |
| exact match (score = 1.0) | +.12 / .12 | +.27 / .27 | +.41 / .41 |
| factuality ≥ 0.8 | +.02 / .10 | +.06 / .07 | +.11 / .11 |

The model's number is **already a calibrated probability that the step is good** (aggregate
gap ≤ .06 on every suite) — the apparent underconfidence against the mean judge score is an
artifact of comparing a probability to a partial-credit average. Against *exact match* it is
overconfident by a corpus-dependent margin (+.12 to +.41). Two consequences:

- Consumers wanting "should I trust this step?" can use the raw number as-is.
- Consumers wanting P(exactly right) need a per-model remap — and a ~500-step monotone
  (isotonic) map fitted on ONE seed transfers to the other: ECE(exact) terminal .27 → **.04**,
  swe .40 → **.07** (tau .17 → .14; its score distribution is too discrete for exact-match
  remapping to fully close). Fitted on the valid split at build time, this is a cheap artifact
  addition (proposed via DECISIONS.md — `card.json` is registry-owned surface).

Per-action-class reliability is tight almost everywhere (terminal web/net gap +.01, write/fs
−.06; swe's dominant compound-command class ±.00) with **one real miscalibrated pocket:
program execution (`python`/`pip`/`npm`/`make`) on terminal at +.21 overconfident** — output
semantics the model cannot simulate and doesn't fully know it can't. That pocket is the
natural target for execution-grounding channels (the channel map's next legs).

## Finding 2 — selective fidelity: abstention buys 5–10 fidelity points

The risk–coverage sweep (answer only when confidence ≥ τ) is the scaling chart the lever
exists for. Fidelity of the covered steps rises monotonically as coverage falls, on every suite
and every mode:

- **tau**: .916 at full coverage → **.988 at 70% coverage** → 1.00 at 40%. A tau world model
  that abstains on its lowest-confidence 30% of steps is a ~99%-faithful environment.
- **terminal**: .883 → .96 at ~55% coverage.
- **swe** (hardest): .80 → .875 at ~70%, .94 at ~35% — the curve is shallower because judge
  partial credit is more continuous, but never flat.

For the product this is the sanctioned use: a serving world model can now *flag* the steps a
consumer (RL trainer, eval gate, demo) should distrust or re-roll, at zero extra completions.

## Finding 3 — confidence-gated verify beats always-verify (Pareto)

The verify lever (second self-check completion) was the expensive lift on swe (#55: .818 at
~2× cost). Gating it on the draft's stated confidence (`verify_below=τ`: verify only when
confidence < τ; missing = low) dominates blanket verification on all three suites:

| suite | never | gated | always |
|---|---|---|---|
| tau (τ=0.6) | .916 ±.001, $5.5 | **.921 ±.005, $7.0** (23% verified) | .920 ±.000, $11.3 |
| terminal (τ=0.5) | .883 ±.002, $3.2 | **.890 ±.007, $4.1** (25% verified) | .885 ±.002, $6.5 |
| swe (τ=0.5) | .801 ±.006, $6.4 | **.820 ±.002, $8.7** (35% verified) | .812 ±.000, $13.1 |
| swe (τ=0.7) | — | **.826 ±.003, $10.3** (64% verified) | — |

($ = serve-side per cell, judge excluded; verified% = fraction of steps that took the second
completion.)

![gated frontier](confidence_gated_frontier.png)

The headline is swe: **gated@0.7 reaches .826 — above always-verify's .812 — at 79% of its
cost**, and gated@0.5 still beats it at 66%. Gating doesn't merely recover the cost of the
skipped verifications; it *removes the harm* blanket verify does to confident drafts (the
"do not invent differences" failure: a revision pass on an already-right answer sometimes
un-fixes it). Verify is only valuable where the model already said it was unsure — and it
knows. On terminal, τ=0.5 sits on the frontier while τ=0.7 dips below never-verify
(seed spread ±.007 — treat the terminal gated lift as ≤ noise; the cost saving is real either
way).

Proposed follow-up (coordinated via DECISIONS.md, autoconfig is shared surface): add
`reason+gateverify@τ` as an autoconfig candidate between `reason` and `reason+verify` in the
price ladder — it needs the confidence flag on, so it slots in as a composite candidate.

## Finding 4 — calibration is intrinsic, not learned from traces

Joining the trace-scaling law (fidelity saturates by ~10 traces): does more RAG data at least
make the *confidence* better-calibrated? No — `base+conf` swept over n_train:

| suite | n_train | AUROC | calibration MSE |
|---|---|---|---|
| tau | 10 / 50 / 200 | .974 / .959 / .950 | .052 / .058 / .055 |
| terminal | 10 / 50 / 160 | .912 / .891 / .877 | .053 / .052 / .055 |
| swe | 6 / 12 / 24 | .844 / .866 / .845 | .064 / .052 / .051 |

AUROC is flat-to-slightly-declining with more data (more demos make more steps answerable, so
the *easy/hard mix* shifts; the self-knowledge itself doesn't move). Calibration ships with the
serving model — you cannot buy it with more traces. But it does not ship with *every* model:

## Finding 5 — calibration is intrinsic to model STRENGTH, not universal

The same `base+conf` cells on three serving models (judge pinned 4.8 throughout):

| serve model | tau: fid / AUROC / conf−P(good)ᵃ | terminal | swe |
|---|---|---|---|
| Haiku 4.5 | .915 / .80 / −.01 | .838 / .82 / **+.06** | .612 / **.61** / **+.22** |
| Opus 4.7 | .914 / .95 / −.06 | .867 / .84 / −.00 | .795 / .85 / −.02 |
| Opus 4.8 | **.926** / **.97** / cal. | .862 / .89 / cal. | .808 / .78 / cal. |

ᵃ aggregate gap vs P(judge ≥ 0.8); "cal." = within a few points, same signature as 4.7.

Strong models are calibrated-to-underconfident with high discrimination everywhere. The small
model holds calibration while the task is within reach (tau — where Haiku *matches Opus
fidelity at ~8× lower cost*) and drifts into classic overconfidence as difficulty rises,
collapsing on swe (+.22 overconfident, AUROC .61, ECE .50). Self-knowledge degrades before —
and faster than — fidelity does. Any consumer of a cheap serving model must check its
reliability curve first; one `+conf` replay cell produces it.

## Finding 6 — confidence-gated model escalation: a serving-cost lever

The inverse of gated verify: draft every step on Haiku, re-predict from scratch on Opus 4.7
only when the draft's stated confidence < τ (`escalate_below` in replay). Measured ladders
($/cell serve-side, seeds 0+1):

| terminal-tasks | fid | $ | | swe-bench | fid | $ |
|---|---|---|---|---|---|---|
| Haiku only | .838 | 0.44 | | Haiku only | .612 | 1.11 |
| **gate@0.9 (29% esc.)** | **.860** | **1.36** | | gate@0.9 (40% esc.) | .749 | 3.36 |
| Opus direct | .867 | 2.70 | | Opus direct | .795 | 6.36 |
| escalate-all | .877 | 3.13 | | escalate-all | .794 | 6.93 |

Where the cheap model's calibration holds (terminal), the gate recovers **~76% of the
Haiku→Opus fidelity gap at half of Opus-direct's cost**, escalating only 29% of steps. Where
its calibration is broken (swe), the gate still lifts +.14 over Haiku but leaves a .046
residual below Opus — the confidently-wrong steps it cannot see — versus terminal's .007. The
residual gap is the price of miscalibration, measured. On tau no ladder is needed: Haiku alone
matches Opus (.915 vs .914) at ~8× lower cost. Deployment rule of thumb: *check the cheap
model's AUROC; ≥ .8 → gate pays; ≤ .6 → pay for the strong model.*

## Caveats

- Two seeds; the population-sizing rule (a channel touching <5% of steps can't beat seed noise
  at n=2) bounded every claim here — gated populations were sized from phase-1 confidence
  distributions before any cell ran (23–64% of steps, comfortably above the floor).
- "Good = judge ≥ 0.8" is a choice; AUROC at ≥ 0.5 and the threshold-free Spearman ρ agree
  directionally everywhere (both in the result JSONs).
- Findings 1–4 are Opus-4.7-serve; Finding 5 spans Haiku 4.5 / Opus 4.7 / Opus 4.8 but no
  non-Anthropic family (OPENAI key dead org-wide at time of writing) — the strength→calibration
  claim is one-family evidence.
- The escalation ladders use `base+conf` (no reasoning contract) and two τ points; the frontier
  shape is robust in sign, not finely resolved in τ.
- Judge provenance: every number here was scored by the ORIGINAL RubricJudge (pre-#83), Opus
  4.8 pinned. The #83 judge overhaul (rubric v2 / `JUDGE_VERSION`, factuality-weighted headline)
  scores the same predictions ~0.12 lower by design — never compare these absolute fidelities
  against rubric-v2 results, or across judges generally.

Raw per-step results (the (confidence, judge-score) joins), calibration summaries, usage/cost
records, SVG sources, and the analysis/figure scripts lived under
`.agents/docs/research/agentic_results/confidence/` and `.agents/scripts/` when this shipped.
`.agents/` is disposable scratch — nothing in this report depends on it surviving; the numbers,
protocol, and figures above are self-contained, and git history at this report's commit
preserves the raw evidence for re-audit.
